#!/usr/bin/env python3
"""Пример поиска по векторному индексу (Задание 3).

Загружает FAISS-индекс Task3/index/, прогоняет несколько русских запросов и
печатает топ-k релевантных чанков с метаданными (заголовок, источник, id чанка,
позиция). Запросы используют вымышленные термины базы — это и проверяет, что
поиск опирается на индекс, а не на память модели.

score — расстояние L2 между нормированными векторами: меньше = ближе.
"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

REPO = Path(__file__).resolve().parent.parent
INDEX_DIR = REPO / "Task3" / "index"
MODEL_NAME = "BAAI/bge-m3"
TOP_K = 3

QUERIES = [
    "Кто такой Гаврила из Зажопинска и чем он занимается?",
    "Что такое Голубая Кровища?",
    "Расскажи про Шальную Облаву и её призрачных всадников.",
    "Какие знаки использует ворожей в бою?",
]


def main() -> int:
    if not INDEX_DIR.exists():
        print("Индекс не найден — сначала запустите build_index.py", file=sys.stderr)
        return 1

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    store = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)

    for query in QUERIES:
        print("=" * 88)
        print("ЗАПРОС:", query)
        for rank, (doc, score) in enumerate(store.similarity_search_with_score(query, k=TOP_K), 1):
            m = doc.metadata
            snippet = " ".join(doc.page_content.split())[:260]
            print(f"\n  [{rank}] score={score:.3f} | «{m['title']}» "
                  f"({m['source']}, чанк #{m['chunk_id']}, поз. {m.get('start_index', '?')})")
            print(f"      {snippet}…")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
