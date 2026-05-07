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
  "latency_ms": 920
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
  "latency_ms": 1240,
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

## Що покращити за 1–2 тижні

- **Async trace writing** — винести запис у background task щоб не блокувати response
- **Response caching** — LRU cache для повторних питань (in-memory або Redis з TTL=1h)
- **Streaming endpoint** — `POST /ask/stream` з SSE для зниження perceived latency
- **Auth middleware** — API key validation для production
- **Docker** — Dockerfile + docker-compose для ізольованого запуску
- **Prometheus metrics** — кількість запитів, latency percentiles, fallback rate
- **Chunk overlap** — додати overlap між реченнями для кращого контексту при складних питаннях

---

## Мої пропозиції (поза вимогами ТЗ)

### Реалізовано понад ТЗ

**1. Hybrid Retrieval (BM25 + Jina Embeddings + RRF)**
ТЗ дозволяє BM25 або embeddings. Реалізовано обидва підходи з RRF fusion. Це суттєво покращує recall на синонімах і мішаному укр/англ тексті бази знань — BM25 ловить точні збіги, embeddings — семантичну схожість.

**2. Jina Reranker v3**
ТЗ не вимагає reranking. Додано як precision layer — переоцінює top-5 кандидатів від retrieval і повертає top-3 з реальними relevance scores. Безпосередньо впливає на якість `confidence` метрики і релевантність `sources` у відповіді.

**3. Автоматичний `scripts/evaluate.py`**
ТЗ вимагає тільки `evaluation_report.md`. Додано скрипт який автоматично прожене всі 5 тест-питань і виведе структуровані результати — це дозволяє швидко перевірити роботу сервісу після будь-яких змін.

**4. GET /health endpoint**
Для production monitoring, load balancer health checks і CI/CD pipeline verification.

### Пропонується для production

**5. SSE Streaming**
`POST /ask/stream` — повертає відповідь токен за токеном через Server-Sent Events. Знижує perceived latency з ~1.5s до <300ms TTFB. Не реалізовано в поточній версії: конфліктує з вимогою ТЗ про structured JSON response в одному об'єкті.

**6. Per-request embedding caching**
Кешувати embedding запиту — при cosine similarity > 0.95 між новим і кешованим запитом повертати попередній результат без API calls. Зменшує latency і витрати на Jina API.

**7. Response caching**
LRU cache з TTL для повторних однакових питань — нема сенсу двічі звертатись до LLM з тим самим запитом.
