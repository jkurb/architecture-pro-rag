#!/usr/bin/env python3
"""Локальный Telegram-бот поверх RAG-ядра (Задания 4 + 5).

Бот работает ЛОКАЛЬНО через long-polling: процесс крутится на машине пользователя,
публичный URL и вебхуки не нужны. Индекс FAISS и эмбеддинги — локальные; наружу идут
лишь два вызова: Telegram (приём/отправка сообщений) и Anthropic (LLM).

Опционально можно направить бота на self-hosted сервер tdlib/telegram-bot-api,
задав TELEGRAM_API_BASE_URL=http://localhost:8081/bot (см. README). По умолчанию
используется облачный api.telegram.org — для текстового бота этого достаточно.

Переменные окружения:
  TELEGRAM_BOT_TOKEN     — токен от @BotFather (обязательно);
  ANTHROPIC_API_KEY      — ключ Anthropic (обязательно);
  TELEGRAM_API_BASE_URL  — базовый URL локального Bot API сервера (опционально);
  CLAUDE_MODEL           — модель Claude (по умолчанию claude-haiku-4-5).

Запуск: python Task5/telegram_bot.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_core import LLM_MODEL, RagEngine  # noqa: E402

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("rag-bot")

TG_MAX = 4096  # лимит длины сообщения Telegram

WELCOME = (
    "Привет! Я ассистент базы знаний QuantumForge.\n\n"
    "Задайте вопрос — я найду ответ в корпоративной базе и покажу шаги рассуждения "
    "и источник. Если ответа в базе нет, честно отвечу «Я не знаю».\n\n"
    "Команды: /help — справка."
)
HELP = (
    "Просто напишите вопрос текстом. Примеры:\n"
    "• Кто такой Гаврила из Зажопинска?\n"
    "• Что такое Шальная Облава?\n"
    "• Какими Знаками владеет ворожей?\n\n"
    "Бот отвечает строго по базе знаний (RAG), применяет Few-shot + Chain-of-Thought "
    "и защищён от промпт-инъекций в документах."
)


async def cmd_start(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME)


async def cmd_help(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP)


async def on_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = (update.message.text or "").strip()
    if not query:
        return
    engine: RagEngine = ctx.application.bot_data["engine"]
    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        # RAG-вызов синхронный (FAISS + Anthropic SDK) — выносим в поток,
        # чтобы не блокировать event loop бота.
        answer, retrieved, report = await asyncio.to_thread(engine.ask, query)
    except Exception as exc:  # noqa: BLE001
        log.exception("Ошибка при обработке запроса")
        await update.message.reply_text(f"Не удалось обработать запрос: {exc}")
        return

    if report.get("dropped"):
        srcs = ", ".join(d["source"] for d in report["dropped"])
        log.warning("Фильтр отбросил вредоносные чанки: %s", srcs)

    await update.message.reply_text(answer[:TG_MAX] or "Пустой ответ.")


def main() -> int:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Не задан TELEGRAM_BOT_TOKEN. Получите токен у @BotFather и задайте:\n"
              "  export TELEGRAM_BOT_TOKEN=123456:ABC-...", file=sys.stderr)
        return 1
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Не задан ANTHROPIC_API_KEY.", file=sys.stderr)
        return 1

    log.info("Загрузка RAG-ядра (индекс + эмбеддинги, модель %s)…", LLM_MODEL)
    engine = RagEngine()
    log.info("RAG-ядро готово.")

    builder = Application.builder().token(token)
    base_url = os.getenv("TELEGRAM_API_BASE_URL")
    if base_url:                          # self-hosted tdlib/telegram-bot-api
        builder = builder.base_url(base_url)
        log.info("Используется локальный Bot API сервер: %s", base_url)

    app = builder.build()
    app.bot_data["engine"] = engine
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    log.info("Бот запущен (long-polling). Откройте чат с ботом в Telegram. Ctrl+C — стоп.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
