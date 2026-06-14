#!/usr/bin/env python3
"""Добавление «злонамеренного» документа в векторный индекс (Задание 5).

Симулирует ситуацию, когда в базу знаний по ошибке (или умышленно) попадает
документ с промпт-инъекцией. Документ проходит ТОТ ЖЕ пайплайн индексации, что и
легитимные страницы: чанкинг RecursiveCharacterTextSplitter → эмбеддинги bge-m3 →
FAISS. После этого инъекция становится извлекаемой при обычном поиске.

Чтобы не портить рабочий индекс Заданий 3–4 (Task3/index/), берём его как основу,
добавляем «отравленный» чанк и сохраняем КОПИЮ в Task5/index_poisoned/.

Запуск: python Task5/build_poisoned_index.py
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

REPO = Path(__file__).resolve().parent.parent
SRC_INDEX = REPO / "Task3" / "index"                 # чистый индекс Задания 3
DST_INDEX = REPO / "Task5" / "index_poisoned"        # индекс с инъекцией
POISON_DOC = REPO / "Task5" / "poison_doc.md"

MODEL_NAME = "BAAI/bge-m3"                            # тот же энкодер, что в индексе
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def main() -> int:
    if not SRC_INDEX.exists():
        print("Чистый индекс не найден. Сначала: python Task3/build_index.py", file=sys.stderr)
        return 1
    if not POISON_DOC.exists():
        print(f"Нет файла инъекции: {POISON_DOC}", file=sys.stderr)
        return 1

    text = POISON_DOC.read_text(encoding="utf-8")
    title = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")), POISON_DOC.stem)
    doc = Document(page_content=text, metadata={"source": POISON_DOC.name, "title": title})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents([doc])
    for i, ch in enumerate(chunks):
        ch.metadata["chunk_id"] = i
    print(f"«Отравленный» документ {POISON_DOC.name} → чанков: {len(chunks)}")

    print(f"Загрузка эмбеддинг-модели {MODEL_NAME} (CPU)…")
    t0 = time.perf_counter()
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"Модель загружена за {time.perf_counter() - t0:.1f} c")

    # Берём чистый индекс как основу и добавляем инъекцию в общий пул чанков.
    store = FAISS.load_local(str(SRC_INDEX), embeddings, allow_dangerous_deserialization=True)
    before = store.index.ntotal
    store.add_documents(chunks)
    after = store.index.ntotal
    print(f"Чанков в индексе: было {before}, стало {after} (+{after - before})")

    if DST_INDEX.exists():
        shutil.rmtree(DST_INDEX)
    DST_INDEX.mkdir(parents=True, exist_ok=True)
    store.save_local(str(DST_INDEX))
    print(f"Индекс с инъекцией сохранён в {DST_INDEX.relative_to(REPO)}/")

    # Контроль: инъекция действительно извлекается обычным поиском.
    probe = "суперпароль root пароль администратора"
    hits = store.similarity_search_with_score(probe, k=4)
    found = any(h[0].metadata["source"] == POISON_DOC.name for h in hits)
    print(f"Контроль извлекаемости (запрос «{probe}»): "
          f"{'инъекция в top-4 — НАЙДЕНА' if found else 'инъекция НЕ найдена'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
