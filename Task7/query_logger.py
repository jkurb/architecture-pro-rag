#!/usr/bin/env python3
"""Логирование запросов к RAG-боту (Задание 7).

Каждый запрос сохраняется строкой JSON (JSONL) с полями для последующей аналитики:
текст запроса, timestamp, найдены ли чанки, длина ответа, флаг успешного ответа,
найденные источники. Модуль переиспользуем: его вызывает evaluate.py, но так же
можно обернуть «живой» бот (rag_core.RagEngine.ask) для логирования продакшн-трафика.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

# Признаки честного отказа в ответе бота (формат «Ответ: Я не знаю.»).
REFUSAL_RE = re.compile(r"я не знаю|не знаю|нет информации|отсутствует информац",
                        re.IGNORECASE)


def extract_answer_line(text: str) -> str:
    """Достаёт строку «Ответ: …» из CoT-ответа; иначе — весь текст."""
    m = re.search(r"Ответ:\s*(.+?)(?:\nИсточник:|\Z)", text, re.IGNORECASE | re.DOTALL)
    return (m.group(1) if m else text).strip()


def is_refusal(text: str) -> bool:
    return bool(REFUSAL_RE.search(extract_answer_line(text)))


class QueryLogger:
    """Пишет по строке JSON на запрос в JSONL-файл."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # перезаписываем при старте прогона
        self.path.write_text("", encoding="utf-8")

    def log(self, query: str, answer: str, retrieved, *, extra: dict | None = None) -> dict:
        """retrieved — список (Document, score) из similarity_search_with_score."""
        sources = [d.metadata.get("source", "?") for d, _ in retrieved]
        refused = is_refusal(answer)
        record = {
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "query": query,
            "chunks_found": len(retrieved),
            "sources": sources,
            "top_score": round(float(retrieved[0][1]), 3) if retrieved else None,
            "answer_length": len(answer),
            "refused": refused,
            # эвристический флаг успеха «по умолчанию»: дан непустой неотказной ответ
            "answered": (not refused) and len(answer) > 0,
        }
        if extra:
            record.update(extra)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
