#!/usr/bin/env python3
"""RAG-бот с техниками промптинга Few-shot + Chain-of-Thought (Задание 4).

Цепочка RAG (что происходит «под капотом»):
  1. приём текстового запроса пользователя;
  2. эмбеддинг запроса тем же энкодером, что и при построении индекса (bge-m3);
  3. поиск ближайших чанков в FAISS (Task3/index/);
  4. сборка промпта: System (CoT) + Few-shot примеры + найденный контекст;
  5. вызов LLM (Claude через Anthropic SDK);
  6. возврат ответа с указанием источников.

Техники промптинга:
  * Few-shot — 2 примера Q→A из той же предметной области (см. FEWSHOT);
  * Chain-of-Thought — System-промпт требует пошагового рассуждения в ответе.

Честность: модель отвечает ТОЛЬКО из найденного контекста. Если ответа в базе нет
(в т.ч. для терминов, отсутствующих в анонимизированной базе, — «Геральт», «Франция»),
бот пишет «Я не знаю.». Это и проверяет работу RAG, а не память модели.

Интерфейс — простой REPL (консоль). Запуск: python Task4/rag_bot.py
Требуется переменная окружения ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import anthropic
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

REPO = Path(__file__).resolve().parent.parent
INDEX_DIR = REPO / "Task3" / "index"
EMBED_MODEL = "BAAI/bge-m3"                      # тот же энкодер, что в Task3
# Самая дешёвая модель Claude ($1/$5 за 1M токенов) — выбрана под Free Tier.
# Переопределяется переменной окружения CLAUDE_MODEL.
LLM_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")
TOP_K = 4
MAX_TOKENS = 1000

# ── System-промпт: Chain-of-Thought + строгая опора на контекст ───────────────
SYSTEM_PROMPT = """Ты — корпоративный ассистент базы знаний QuantumForge. \
Ты сначала размышляешь по шагам, а затем даёшь итоговый ответ.

Правила:
1. Отвечай СТРОГО на основе блока КОНТЕКСТ. Не используй внешние знания и ничего не домысливай.
2. Если в КОНТЕКСТЕ нет ответа на вопрос — итоговым ответом напиши ровно «Я не знаю.» \
и кратко поясни, какой информации не хватает.
3. Всегда показывай свои шаги рассуждения.

Формат ответа строго такой:
Рассуждение:
1. <шаг>
2. <шаг>
Ответ: <краткий итоговый ответ или «Я не знаю.»>
Источник: <имена файлов из контекста или «—»>"""

# ── Few-shot: 2 примера из той же предметной области (взяты из базы знаний) ────
# Пример 1 — ответ найден; Пример 2 — честный отказ «Я не знаю».
FEWSHOT = [
    {
        "role": "user",
        "content": (
            "КОНТЕКСТ:\n"
            "[Источник: ворожей.md] Ворожей владеют так называемыми Знаками — "
            "разновидностью боевой простейшей магии, которая появляется у охотников "
            "на чудовищ после Испытаний.\n\n"
            "ВОПРОС: Что такое Знаки и кто ими владеет?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Рассуждение:\n"
            "1. В контексте сказано, что Знаки — это разновидность боевой простейшей магии.\n"
            "2. Там же указано, что ими владеют ворожей, охотники на чудовищ, после Испытаний.\n"
            "Ответ: Знаки — это простая боевая магия, которой владеют ворожей после Испытаний.\n"
            "Источник: ворожей.md"
        ),
    },
    {
        "role": "user",
        "content": (
            "КОНТЕКСТ:\n"
            "[Источник: булькало.md] Булькало — водный гуманоидный трупоед, "
            "кровожадное и неразумное чудовище.\n\n"
            "ВОПРОС: Какая столица у Франции?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Рассуждение:\n"
            "1. Вопрос про столицу Франции.\n"
            "2. В контексте говорится только про чудовище булькало; о Франции данных нет.\n"
            "Ответ: Я не знаю. В базе знаний нет информации о Франции и её столице.\n"
            "Источник: —"
        ),
    },
]


def load_store() -> tuple[FAISS, HuggingFaceEmbeddings]:
    if not INDEX_DIR.exists():
        sys.exit("Индекс не найден. Сначала постройте его: python Task3/build_index.py")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    store = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
    return store, embeddings


def format_context(retrieved) -> str:
    """Собирает блок КОНТЕКСТ из найденных чанков с метками источников."""
    parts = []
    for doc, score in retrieved:
        m = doc.metadata
        text = " ".join(doc.page_content.split())
        parts.append(f"[Источник: {m['source']}, чанк #{m['chunk_id']}] {text}")
    return "\n\n".join(parts)


def build_messages(query: str, retrieved) -> list[dict]:
    """Few-shot примеры + реальный запрос с найденным контекстом (последним ходом)."""
    context = format_context(retrieved)
    user_turn = {
        "role": "user",
        "content": f"КОНТЕКСТ:\n{context}\n\nВОПРОС: {query}",
    }
    return FEWSHOT + [user_turn]


def ask(store: FAISS, client: anthropic.Anthropic, query: str) -> tuple[str, list]:
    retrieved = store.similarity_search_with_score(query, k=TOP_K)
    messages = build_messages(query, retrieved)
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    answer = next((b.text for b in response.content if b.type == "text"), "").strip()
    return answer, retrieved


def repl() -> int:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Не задан ANTHROPIC_API_KEY. Экспортируйте ключ:\n"
              "  export ANTHROPIC_API_KEY=sk-ant-...", file=sys.stderr)
        return 1

    print(f"Загрузка индекса и модели эмбеддингов {EMBED_MODEL}…")
    store, _ = load_store()
    client = anthropic.Anthropic()
    print(f"Готово. LLM: {LLM_MODEL}. Введите вопрос (пустая строка или 'exit' — выход).\n")

    while True:
        try:
            query = input("Вопрос> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in {"exit", "quit", "выход"}:
            break
        try:
            answer, retrieved = ask(store, client, query)
        except anthropic.APIError as exc:
            print(f"[Ошибка Claude API: {exc}]\n", file=sys.stderr)
            continue
        print()
        print(answer)
        top = retrieved[0]
        print(f"\n(найдено чанков: {len(retrieved)}; ближайший: "
              f"{top[0].metadata['source']}, score={top[1]:.3f})\n")
    return 0


if __name__ == "__main__":
    sys.exit(repl())
