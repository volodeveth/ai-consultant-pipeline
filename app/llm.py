import os

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """Ти AI-консультант з кадрових питань. Відповідай ВИКЛЮЧНО на основі наданих фрагментів бази знань.

Правила:
1. Відповідай ЗАВЖДИ українською мовою, навіть якщо контекст наданий англійською
2. Використовуй ТІЛЬКИ інформацію з наданих фрагментів контексту
3. НЕ вигадуй факти, точні дати, суми, формули або винятки, яких немає в контексті
4. Якщо контексту недостатньо — чітко поясни, що в базі знань немає потрібної інформації
5. Якщо питання потребує розрахунків, а дані відсутні — поясни, чому розрахунок неможливий
6. Посилайся на конкретні секції бази знань у відповіді"""


async def generate_answer(
    question: str,
    context_chunks: list[tuple],
) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")

    context = "\n\n---\n\n".join(
        f"[Секція: {chunk.section}]\n{chunk.text}"
        for chunk, _ in context_chunks
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Контекст з бази знань:\n{context}\n\n---\n\nПитання: {question}",
        },
    ]

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 1000,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    return data["choices"][0]["message"]["content"].strip()
