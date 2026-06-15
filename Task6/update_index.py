#!/usr/bin/env python3
"""Инкрементальное обновление векторного индекса (Задание 6).

Автоматизирует поддержание индекса FAISS в актуальном состоянии:
  1. сканирует источник (папку базы знаний) и по SHA-256 находит новые,
     изменённые и удалённые документы относительно манифеста прошлого запуска;
  2. разбивает изменённые документы на чанки (как в Заданиях 2–3);
  3. генерирует эмбеддинги (BAAI/bge-m3, тот же энкодер, что в индексе);
  4. обновляет векторную БД: удаляет чанки изменённых/удалённых файлов,
     добавляет чанки новых/изменённых файлов;
  5. логирует процесс (человекочитаемый лог + JSONL + stdout).

Источник: локальная папка Task2/knowledge_base/ — «живая» база знаний компании,
которая прирастает новыми документами. «Подхват» нового файла = появление *.md
в этой папке: при следующем запуске его хэш отсутствует в манифесте → файл
индексируется. Источник легко заменить на S3/Git/RSS, изменив scan_source().

Обновляемый индекс — Task3/index/ (его читает Telegram-бот), поэтому после запуска
бот сразу отвечает по обновлённой базе.

Запуск вручную: python Task6/update_index.py
Периодический запуск: cron внутри Docker-контейнера (см. Task6/README.md).
Код возврата: 0 — успех (в т.ч. «нет изменений»), 1 — была ошибка.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

REPO = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO / "Task2" / "knowledge_base"     # источник «живых» документов
INDEX_DIR = REPO / "Task3" / "index"               # обновляемый индекс (его читает бот)
MANIFEST = REPO / "Task6" / "index_manifest.json"  # состояние прошлого запуска
LOG_DIR = REPO / "Task6" / "logs"
LOG_JSONL = LOG_DIR / "update_log.jsonl"           # машинный лог (по записи на запуск)
LOG_TXT = LOG_DIR / "update.log"                   # человекочитаемый лог
LOCK = REPO / "Task6" / ".update.lock"             # защита от параллельных запусков

MODEL_NAME = "BAAI/bge-m3"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
LOCK_STALE_SEC = 3600                               # старше часа — считаем зависшим

log = logging.getLogger("update-index")


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    fh = logging.FileHandler(LOG_TXT, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.handlers = [fh, sh]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan_source() -> dict[str, str]:
    """Текущее состояние источника: {имя файла: sha256}."""
    return {p.name: file_hash(p) for p in sorted(SOURCE_DIR.glob("*.md"))}


def load_manifest() -> dict[str, str] | None:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return None


def save_manifest(manifest: dict[str, str]) -> None:
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")


def bootstrap_manifest(store: FAISS) -> dict[str, str]:
    """Манифест из источников, уже присутствующих в индексе (первый запуск).

    Хэшируем текущие файлы на диске, чтобы первый запуск не дублировал уже
    проиндексированные документы и не считал их «новыми».
    """
    indexed = {d.metadata.get("source") for d in store.docstore._dict.values()}
    manifest = {}
    for name in indexed:
        p = SOURCE_DIR / name
        if p.exists():
            manifest[name] = file_hash(p)
    return manifest


def chunk_file(path: Path) -> list[Document]:
    """Документ → чанки с метаданными (как в Задании 3)."""
    text = path.read_text(encoding="utf-8")
    title = path.stem
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break
    doc = Document(page_content=text, metadata={"source": path.name, "title": title})
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents([doc])
    for i, ch in enumerate(chunks):
        ch.metadata["chunk_id"] = i
    return chunks


def ids_for_source(store: FAISS, name: str) -> list[str]:
    return [i for i, d in store.docstore._dict.items() if d.metadata.get("source") == name]


def acquire_lock() -> bool:
    if LOCK.exists():
        age = time.time() - LOCK.stat().st_mtime
        if age < LOCK_STALE_SEC:
            return False
        log.warning("Найден устаревший lock (%.0f c) — перехватываю.", age)
    LOCK.write_text(str(time.time()), encoding="utf-8")
    return True


def write_jsonl(record: dict) -> None:
    with LOG_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    setup_logging()
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).astimezone()
    errors: list[str] = []
    added_files: list[str] = []
    changed_files: list[str] = []
    removed_files: list[str] = []
    added_chunks = removed_chunks = 0
    index_size = 0

    if not acquire_lock():
        log.warning("Предыдущий запуск ещё идёт (lock свежий) — выхожу.")
        return 0

    try:
        if not INDEX_DIR.exists():
            raise FileNotFoundError(
                f"Индекс не найден: {INDEX_DIR}. Сначала: python Task3/build_index.py")
        if not SOURCE_DIR.exists():
            raise FileNotFoundError(f"Источник не найден: {SOURCE_DIR}")

        log.info("Старт обновления. Источник: %s, индекс: %s",
                 SOURCE_DIR.relative_to(REPO), INDEX_DIR.relative_to(REPO))
        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        store = FAISS.load_local(str(INDEX_DIR), embeddings,
                                 allow_dangerous_deserialization=True)

        manifest = load_manifest()
        if manifest is None:
            manifest = bootstrap_manifest(store)
            log.info("Манифест отсутствует — инициализирован из индекса (%d файлов).",
                     len(manifest))

        disk = scan_source()
        new = sorted(set(disk) - set(manifest))
        changed = sorted(f for f in set(disk) & set(manifest) if disk[f] != manifest[f])
        removed = sorted(set(manifest) - set(disk))
        log.info("Найдено: новых=%d, изменённых=%d, удалённых=%d "
                 "(всего в источнике %d файлов)",
                 len(new), len(changed), len(removed), len(disk))

        # 1) удаляем чанки изменённых и удалённых файлов
        for name in changed + removed:
            try:
                ids = ids_for_source(store, name)
                if ids:
                    store.delete(ids)
                    removed_chunks += len(ids)
                    log.info("Удалены чанки (%d) устаревшего файла %s", len(ids), name)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"delete {name}: {exc}")
                log.exception("Ошибка удаления чанков %s", name)

        # 2) добавляем чанки новых и изменённых файлов
        for name in new + changed:
            try:
                chunks = chunk_file(SOURCE_DIR / name)
                store.add_documents(chunks)
                added_chunks += len(chunks)
                (added_files if name in new else changed_files).append(name)
                log.info("Проиндексирован %s: +%d чанков", name, len(chunks))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"add {name}: {exc}")
                log.exception("Ошибка индексации %s", name)

        removed_files = removed

        # 3) сохраняем индекс и манифест, только если что-то изменилось
        if new or changed or removed:
            store.save_local(str(INDEX_DIR))
            save_manifest(disk)
            log.info("Индекс и манифест сохранены.")
        else:
            # первый запуск без манифеста — зафиксируем baseline
            if not MANIFEST.exists():
                save_manifest(disk)
            log.info("Изменений нет — индекс не трогаем.")

        index_size = store.index.ntotal

    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
        log.exception("Критическая ошибка обновления")
    finally:
        LOCK.unlink(missing_ok=True)

    finished_at = datetime.now(timezone.utc).astimezone()
    duration = round(time.perf_counter() - started, 1)
    record = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "duration_seconds": duration,
        "files_added": added_files,
        "files_changed": changed_files,
        "files_removed": removed_files,
        "new_chunks": added_chunks,
        "removed_chunks": removed_chunks,
        "index_size": index_size,
        "errors": errors,
    }
    write_jsonl(record)
    # строка в формате из ТЗ
    log.info("index updated at %s, %d files added, %d errors",
             finished_at.date().isoformat(), len(added_files), len(errors))
    log.info("Итог: +%d/-%d файлов, +%d/-%d чанков, размер индекса=%d, ошибок=%d, %.1f c",
             len(added_files), len(removed_files), added_chunks, removed_chunks,
             index_size, len(errors), duration)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
