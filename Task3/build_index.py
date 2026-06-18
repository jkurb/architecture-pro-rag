#!/usr/bin/env python3
"""Построение векторного индекса базы знаний (Задание 3).

Пайплайн: Task2/knowledge_base/*.md → чанкинг (RecursiveCharacterTextSplitter)
→ эмбеддинги (BAAI/bge-m3, локально, 1024-dim) → индекс FAISS.

Эмбеддинг-модель выбрана в Задании 1: локальная многоязычная Sentence-Transformers
`BAAI/bge-m3` (база знаний на русском, поэтому модель обязана быть многоязычной).
Каждый чанк сохраняет метаданные для цитирования: источник, заголовок, id чанка,
позицию (start_index) в исходном документе.

Результат: Task3/index/ (index.faiss + index.pkl) и Task3/build_stats.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

REPO = Path(__file__).resolve().parent.parent
KB_DIR = REPO / "Task2" / "knowledge_base"
INDEX_DIR = REPO / "Task3" / "index"
STATS_PATH = REPO / "Task3" / "build_stats.json"

# Модель из Задания 1: 1024-dim, многоязычная, без query/passage-префиксов.
MODEL_NAME = "BAAI/bge-m3"
CHUNK_SIZE = 1000          # символов (~150–200 русских слов, в коридоре 100–300)
CHUNK_OVERLAP = 150


def load_documents() -> list[Document]:
    """Один Document на файл; заголовок берётся из первой строки '# ...'."""
    docs: list[Document] = []
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = path.stem
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        docs.append(Document(page_content=text, metadata={"source": path.name, "title": title}))
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,                       # позиция чанка в документе
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    # сквозной id чанка в пределах каждого документа
    counters: dict[str, int] = {}
    for ch in chunks:
        src = ch.metadata["source"]
        ch.metadata["chunk_id"] = counters.get(src, 0)
        counters[src] = ch.metadata["chunk_id"] + 1
    return chunks


def main() -> int:
    if not KB_DIR.exists() or not any(KB_DIR.glob("*.md")):
        print("База знаний не найдена в Task2/knowledge_base/", file=sys.stderr)
        return 1

    docs = load_documents()
    chunks = split_documents(docs)
    print(f"Документов: {len(docs)}, чанков: {len(chunks)} "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    print(f"Загрузка эмбеддинг-модели {MODEL_NAME} (CPU)…")
    t0 = time.perf_counter()
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )
    dim = len(embeddings.embed_query("проба"))
    load_s = time.perf_counter() - t0
    print(f"Модель загружена за {load_s:.1f} c, размерность эмбеддингов: {dim}")

    print("Генерация эмбеддингов и построение индекса FAISS…")
    t1 = time.perf_counter()
    store = FAISS.from_documents(chunks, embeddings)
    build_s = time.perf_counter() - t1

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX_DIR))
    print(f"Индекс построен за {build_s:.1f} c и сохранён в {INDEX_DIR.relative_to(REPO)}/")

    stats = {
        "model": MODEL_NAME,
        "embedding_dim": dim,
        "num_documents": len(docs),
        "num_chunks": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "model_load_seconds": round(load_s, 1),
        "index_build_seconds": round(build_s, 1),
    }
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Статистика:", json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
