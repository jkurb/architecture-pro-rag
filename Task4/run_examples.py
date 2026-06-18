#!/usr/bin/env python3
"""Неинтерактивный прогон демонстрационных диалогов RAG-бота (Задание 4).

Прогоняет набор запросов: часть — на которые ответ есть в базе, часть —
на которые бот честно отвечает «Я не знаю» (вопрос вне базы или о термине,
которого в анонимизированной базе нет, — «Геральт» переименован в «Гаврилу»).

Запуск: ANTHROPIC_API_KEY=... python Task4/run_examples.py
Вывод удобно сохранить в Task4/dialogs.md.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anthropic  # noqa: E402
import rag_bot  # noqa: E402

# 5 запросов с ответом из базы + 2 на честный отказ «Я не знаю».
QUERIES = [
    "Кто такой Гаврила из Зажопинска и чем он занимается?",
    "Что такое Шальная Облава?",
    "Какими Знаками владеет ворожей и откуда они берутся?",
    "Что такое Голубая Кровища?",
    "Расскажи про чудовище булькало.",
    # вне базы / переименованные термины → ожидается «Я не знаю»
    "Кто такой Геральт из Ривии?",
    "Какая столица у Франции?",
]


def main() -> int:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Не задан ANTHROPIC_API_KEY.", file=sys.stderr)
        return 1
    store, _ = rag_bot.load_store()
    client = anthropic.Anthropic()
    print(f"# Демонстрационные диалоги RAG-бота\n\nLLM: {rag_bot.LLM_MODEL} · "
          f"эмбеддинги: {rag_bot.EMBED_MODEL} · top-k={rag_bot.TOP_K}\n")
    # Пауза между запросами под лимиты тарифа. Tier 1 = 50 RPM, поэтому хватает
    # пары секунд; для Free Tier (5 RPM) задайте PACE_SECONDS=18.
    pace = float(os.getenv("PACE_SECONDS", "2"))
    for i, q in enumerate(QUERIES, 1):
        if i > 1:
            time.sleep(pace)
        answer, retrieved = rag_bot.ask(store, client, q)
        top = retrieved[0]
        print("=" * 90)
        print(f"## Диалог {i}\n")
        print(f"**Вопрос:** {q}\n")
        print(answer + "\n")
        print(f"_(найдено чанков: {len(retrieved)}; ближайший: "
              f"{top[0].metadata['source']}, score={top[1]:.3f})_\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
