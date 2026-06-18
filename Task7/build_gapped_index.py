#!/usr/bin/env python3
"""Создание индекса с искусственными пробелами (Задание 7).

Чтобы протестировать поведение бота при отсутствии информации, удаляем из копии
рабочего индекса несколько ключевых сущностей. Прод-индекс Task3/index/ не трогаем —
эксперимент изолирован в Task7/index_gapped/.

Удаляются чанки документов (по полю metadata.source), перечисленных в
golden_questions.json → removed_entities. Используется FAISS.delete по источнику —
переэмбеддинг не требуется.

Запуск: python Task7/build_gapped_index.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Task5"))
from rag_core import CLEAN_INDEX, EMBED_MODEL  # noqa: E402

from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DST_INDEX = REPO / "Task7" / "index_gapped"
GOLDEN = REPO / "Task7" / "golden_questions.json"


def ids_for_source(store: FAISS, name: str) -> list[str]:
    return [i for i, d in store.docstore._dict.items() if d.metadata.get("source") == name]


def main() -> int:
    if not CLEAN_INDEX.exists():
        print("Чистый индекс не найден. Сначала: python Task3/build_index.py", file=sys.stderr)
        return 1
    removed = json.loads(GOLDEN.read_text(encoding="utf-8"))["removed_entities"]

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    store = FAISS.load_local(str(CLEAN_INDEX), embeddings, allow_dangerous_deserialization=True)
    before = store.index.ntotal

    total_removed = 0
    for name in removed:
        ids = ids_for_source(store, name)
        if ids:
            store.delete(ids)
            total_removed += len(ids)
            print(f"Удалён документ {name}: -{len(ids)} чанков")
        else:
            print(f"Внимание: {name} не найден в индексе", file=sys.stderr)

    if DST_INDEX.exists():
        shutil.rmtree(DST_INDEX)
    DST_INDEX.mkdir(parents=True, exist_ok=True)
    store.save_local(str(DST_INDEX))
    print(f"Индекс с пробелами: {before} → {store.index.ntotal} чанков "
          f"(-{total_removed}), удалено сущностей: {len(removed)}")
    print(f"Сохранён в {DST_INDEX.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
