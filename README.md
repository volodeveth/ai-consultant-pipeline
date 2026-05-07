# AI Consultant Pipeline

Controlled RAG сервіс-консультант. Приймає питання українською → знаходить релевантний контекст у базі знань → повертає структуровану відповідь у JSON.

## Запуск

**Вимоги:** Python 3.11+

```bash
# 1. Встановити залежності
pip install -r requirements.txt

# 2. Додати .env файл (отримати від автора або заповнити .env.example власними ключами)

# 3. Запустити сервіс
uvicorn app.main:app --port 8000
```

Сервіс доступний: `http://localhost:8000`

При запуску автоматично індексується `data/knowledge_base.md` та завантажуються embeddings.

---

## Endpoint

### POST /ask

**Request:**
```json
{
  "question": "Чи може працівник взяти щорічну відпустку після 3 місяців роботи?"
}
```

**Response:**
```json
{
  "answer": "За наданою базою знань, працівник може використати щорічну оплачувану відпустку після 6 місяців безперервної роботи у компанії. Право на відпустку після 3 місяців у базі знань не підтверджено.",
  "sources": [
    {
      "section": "1. Щорічна відпустка",
      "chunk": "Працівник може використати щорічну оплачувану відпустку після 6 місяців безперервної роботи у компанії.",
      "score": 0.87
    }
  ],
  "confidence": "high",
  "fallback_reason": null,
  "trace_id": "550e8400-e29b-41d4-a716-446655440000",
  "latency_ms": 18500
}
```

### GET /health

```json
{ "status": "ok" }
```

---

## Індексація та Retrieval

**Chunking:** `data/knowledge_base.md` розбивається по `##` секціях → кожна секція → окремі речення-chunks. Кожен chunk зберігає назву секції для відображення в `sources`.

**Retrieval pipeline (4 стадії):**

1. **BM25** (`rank-bm25`) — лексичний пошук по токенах. Добре знаходить ключові слова навіть у мішаному укр/англ тексті бази знань.
2. **Jina Embeddings v3** — семантичний пошук. `retrieval.query` task для запиту, `retrieval.passage` для chunks. 1024-вимірні нормалізовані вектори.
3. **RRF fusion** (Reciprocal Rank Fusion, k=60) — об'єднує ранги BM25 і семантичного пошуку. Компенсує слабкі місця кожного методу окремо.
4. **Jina Reranker v3** — precision layer: переоцінює top-5 кандидатів, повертає top-3 з relevance scores.

Embeddings завантажуються один раз при старті сервісу і зберігаються in-memory.

---

## Confidence

Визначається на основі top relevance score від Jina Reranker v3:

| Score | Confidence |
|---|---|
| ≥ 0.7 | `high` |
| 0.4 – 0.69 | `medium` |
| 0.2 – 0.39 | `low` |
| < 0.2 | fallback |

---

## Fallback

Спрацьовує коли:
- Top reranker score < `FALLBACK_SCORE_THRESHOLD` (default: `0.2`)
- Жоден chunk не знайдений при retrieval

Fallback відповідь повертається у тому самому JSON форматі з `confidence: "low"` та поясненням у `fallback_reason`. Вигадувати факти заборонено системним промптом LLM.

---

## LLM API

**Provider:** [OpenRouter](https://openrouter.ai) — `https://openrouter.ai/api/v1/chat/completions`

**Model:** `deepseek/deepseek-chat`

**Чому DeepSeek:** OpenAI-сумісний API (мінімальна адаптація), сильна підтримка багатомовності (укр/англ), низька latency, cost-effective для production навантаження.

Системний промпт примусово забороняє LLM: вигадувати факти, дати, формули; відповідати поза контекстом; відповідати не українською мовою.

---

## Trace Logging

Кожен запит логується в `traces.jsonl` (append mode, створюється автоматично):

```json
{
  "trace_id": "uuid4",
  "question": "...",
  "pipeline_stages": {
    "context_loading": { "status": "completed", "chunks_available": 46 },
    "retrieval": { "status": "completed", "candidates_found": 5, "top_score": 0.0431 },
    "reranking": { "status": "completed", "chunks_after_rerank": 3, "top_score": 0.9124 },
    "llm_generation": { "status": "completed", "answer_length": 287 },
    "validation": { "status": "completed" },
    "response": { "status": "completed" }
  },
  "context_chunks": [
    { "section": "1. Щорічна відпустка", "text": "...", "score": 0.9124 }
  ],
  "answer": "...",
  "confidence": "high",
  "fallback_reason": null,
  "latency_ms": 19200,
  "errors": []
}
```

Якщо на будь-якому етапі виникла помилка — вона фіксується в `errors[]` і відображається в `pipeline_stages` зі `"status": "error"`.

---

## Laravel API + LangChain/LangGraph Integration

### Варіант 1 — HTTP мікросервіс (рекомендовано для старту)

```
Laravel API  →  POST /ask  →  Python FastAPI  →  OpenRouter + Jina AI
```

Laravel викликає сервіс як зовнішній HTTP dependency через `Http::post()`. Auth: shared secret через `X-API-Key` header.

```php
// Laravel controller
$response = Http::withHeaders(['X-API-Key' => config('services.ai.key')])
    ->post(config('services.ai.url') . '/ask', ['question' => $question]);
return $response->json();
```

### Варіант 2 — LangChain Tool

Сервіс стає одним з tools у LangChain agent:

```python
from langchain.tools import Tool

consultant_tool = Tool(
    name="hr_consultant",
    func=lambda q: requests.post("http://ai-service/ask", json={"question": q}).json()["answer"],
    description="Відповідає на HR питання з корпоративної бази знань",
)
```

### Варіант 3 — LangGraph Agent Node

У LangGraph workflow цей сервіс виступає окремим вузлом графу:

```python
async def consult_node(state: AgentState) -> AgentState:
    result = await pipeline.process_question(state["question"])
    return {**state, "answer": result.answer, "sources": result.sources}

graph = StateGraph(AgentState)
graph.add_node("consult", consult_node)
```

---

## Приклади для тестових питань

### q001 — Щорічна відпустка

```json
POST /ask
{ "question": "Я працюю в компанії 3 місяці. Чи можу вже піти у щорічну оплачувану відпустку?" }

{
  "answer": "Ні, після 3 місяців роботи скористатися щорічною оплачуваною відпусткою не можна. Відповідно до наданої бази знань, право на щорічну оплачувану відпустку виникає після 6 місяців безперервної роботи у компанії (секція: 1. Щорічна відпустка).",
  "sources": [{ "section": "1. Щорічна відпустка", "chunk": "Працівник може використати щорічну оплачувану відпустку після 6 місяців безперервної роботи у компанії.", "score": 0.29 }],
  "confidence": "medium",
  "fallback_reason": null,
  "trace_id": "...",
  "latency_ms": 15000
}
```

### q002 — Лікарняний без документа (fallback — cross-lingual gap)

```json
POST /ask
{ "question": "Працівник захворів, але ще не надав медичний документ. Чи можна одразу оплатити лікарняний?" }

{
  "answer": "На основі наданої бази знань неможливо надати точну відповідь на це питання. У базі знань відсутня достатня інформація для відповіді.",
  "sources": [],
  "confidence": "low",
  "fallback_reason": "Релевантний контекст не знайдено в базі знань",
  "trace_id": "...",
  "latency_ms": 1400
}
```

> ⚠️ Відоме обмеження: секція 2 бази знань англійською — BM25 не знаходить "лікарняний" → "sick leave". Jina Embeddings компенсує частково, але Reranker scored нижче threshold. Детальніше: `evaluation_report.md`.

### q003 — Автоматична індексація зарплат

```json
POST /ask
{ "question": "У нас є payroll-система. Чи можемо ми автоматично індексувати зарплату всім працівникам однаково?" }

{
  "answer": "Ні, автоматично індексувати зарплату однаково для всіх працівників не можна. Згідно з базою знань (секція: 3. Індексація зарплати), для payroll-систем базовий місяць має зберігатись окремо для кожного працівника. Для розрахунку індексації потрібні: базовий місяць працівника, актуальний індекс споживчих цін та сума доходу.",
  "sources": [{ "section": "3. Індексація зарплати / Salary indexation", "chunk": "For payroll systems, the base month must be stored separately for each employee.", "score": 0.31 }],
  "confidence": "low",
  "fallback_reason": null,
  "trace_id": "...",
  "latency_ms": 22000
}
```

### q004 — Точна дата ЄСВ (controlled — LLM відмовляє)

```json
POST /ask
{ "question": "Яка точна дата сплати ЄСВ у цьому місяці?" }

{
  "answer": "У наданій базі знань точні календарні дати сплати ЄСВ не наведені (секція: 4. Податкові строки). Відомо лише, що ЄСВ сплачується у строки для відповідного звітного періоду, і якщо останній день припадає на вихідний — строк переноситься на найближчий робочий день. Назвати конкретну дату на підставі цієї бази знань неможливо.",
  "sources": [{ "section": "4. Податкові строки", "chunk": "У цій базі знань точні календарні дати сплати ЄСВ не наведені.", "score": 0.38 }],
  "confidence": "low",
  "fallback_reason": null,
  "trace_id": "...",
  "latency_ms": 18000
}
```

### q005 — Розрахунок індексації (controlled — LLM відмовляє)

```json
POST /ask
{ "question": "Порахуй індексацію для працівника із зарплатою 25000 грн, якщо базовий місяць невідомий." }

{
  "answer": "На жаль, розрахувати індексацію для працівника із зарплатою 25000 грн неможливо, оскільки в базі знань відсутні необхідні числові дані (секція: 3. Індексація зарплати). Для розрахунку потрібні: базовий місяць працівника, актуальний індекс споживчих цін та сума доходу, що підлягає індексації. Жоден з цих параметрів у базі знань не наведено.",
  "sources": [{ "section": "3. Індексація зарплати / Salary indexation", "chunk": "У цій базі знань немає достатніх числових даних для розрахунку конкретної суми індексації.", "score": 0.26 }],
  "confidence": "low",
  "fallback_reason": null,
  "trace_id": "...",
  "latency_ms": 16000
}
```

---

## Що покращити за 1–2 тижні

### Вже реалізовано понад мінімум ТЗ

| Що | Навіщо |
|---|---|
| **Hybrid Retrieval** (BM25 + Jina Embeddings v3 + RRF) | ТЗ дозволяє будь-який підхід — обрано найефективніший: BM25 ловить точні збіги, embeddings — семантику, RRF fusion об'єднує ранги |
| **Jina Reranker v3** | ТЗ не вимагає — додано як precision layer, безпосередньо впливає на якість `confidence` і релевантність `sources` |
| **`scripts/evaluate.py`** | ТЗ вимагає тільки звіт — автоматичний скрипт дозволяє швидко перевірити pipeline після будь-яких змін |
| **GET /health** | Production monitoring, load balancer health checks, CI/CD |
| **Shared `httpx.AsyncClient`** | TCP/TLS connection reuse між запитами до Jina AI і OpenRouter — усуває handshake overhead (~300-500ms на запит) |
| **Async trace write** | `traces.jsonl` пишеться у фоновому `asyncio.Task` через `asyncio.to_thread` — файловий I/O не блокує response |

### Latency

Реальний час відповіді: **15–35s** на запит. Домінуючий фактор — генерація DeepSeek через OpenRouter (~10-25s). Jina Embedding + Reranker додають ще ~3-8s. Це не проблема коду, а характеристика hosted AI APIs без кешування.

### Пропонується для наступних ітерацій

- **Response caching** — LRU cache для повторних однакових питань (Redis з TTL=1h). Найбільший win по latency: повторний запит → миттєва відповідь без жодного API call
- **Per-request embedding caching** — при cosine similarity > 0.95 між новим і кешованим запитом повертати попередній результат без Jina API call
- **SSE Streaming** — `POST /ask/stream` повертає відповідь токен за токеном. Знижує perceived latency з ~20s до <300ms TTFB. Не реалізовано: конфліктує з вимогою ТЗ про цілісний JSON об'єкт у відповіді
- **Auth middleware** — API key validation через header для production
- **Docker** — Dockerfile + docker-compose для ізольованого відтворюваного запуску
- **Prometheus metrics** — кількість запитів, latency percentiles (p50/p95/p99), fallback rate
- **Chunk overlap** — overlap між реченнями для збереження контексту при складних питаннях
- **Query rewriting** — перефразування запиту перед retrieval для кращого recall (особливо cross-lingual)
