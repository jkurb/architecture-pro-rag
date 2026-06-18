#!/usr/bin/env python3
"""Анонимизация сырых текстов: raw_pages/*.txt → knowledge_base/*.md.

Заменяет термины The Witcher на вымышленные по terms_map.json так, чтобы
тексты остались связными, но больше не распознавались как The Witcher.

Логика замены (учёт русской морфологии):
  * Ключ словаря — ОСНОВА термина; падежное окончание захватывается группой
    (\\w*) и отбрасывается, а замена ставится в именительном падеже
    («стратегия номинатива»). Это покрывает любые склонения одним правилом:
    Геральт/Геральта/Геральтом → Гаврила.
  * Longest-match-first: ключи сортируются по убыванию длины, поэтому
    «Трисс Меригольд» срабатывает раньше, чем «Трисс».
  * Регистронезависимый матч с сохранением регистра оригинала (match_case).
  * Один проход pattern.sub() — без каскадных перезамен.
  * Границы \\b с обеих сторон отсекают подстроки внутри слов.

Trade-off: изредка возможна падежная несогласованность («встретил Гаврила»
вместо «Гаврилу») — текст остаётся связным и полностью неузнаваемым.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "raw_pages"
KB_DIR = ROOT / "knowledge_base"
MAP_PATH = ROOT / "terms_map.json"


def load_terms_map() -> dict[str, str]:
    """Читает словарь, отбрасывая служебные ключи на '_' (комментарии)."""
    data = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def match_case(source: str, replacement: str) -> str:
    """Переносит регистр оригинала на замену (UPPER / Title / lower)."""
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def build_replacer(terms_map: dict[str, str]):
    """Собирает один компилированный regex и функцию-replacer."""
    keys = sorted(terms_map, key=len, reverse=True)          # longest-match-first
    lower_map = {k.lower(): v for k, v in terms_map.items()}
    alternation = "|".join(re.escape(k) for k in keys)
    # \b(основа)(\w*) — захват падежного окончания; I|U обязательны для кириллицы
    pattern = re.compile(rf"\b({alternation})(\w*)", re.IGNORECASE | re.UNICODE)

    def replacer(match: re.Match) -> str:
        base = lower_map[match.group(1).lower()]
        return match_case(match.group(1), base)              # окончание group(2) отброшено

    return pattern, replacer


def slugify(title: str) -> str:
    return re.sub(r"[^\w]+", "-", title.lower(), flags=re.UNICODE).strip("-")


def main() -> int:
    if not RAW_DIR.exists() or not any(RAW_DIR.glob("*.txt")):
        print("raw_pages/ пуст — сначала запустите scripts/fetch_pages.py", file=sys.stderr)
        return 1

    terms_map = load_terms_map()
    pattern, replacer = build_replacer(terms_map)
    KB_DIR.mkdir(exist_ok=True)

    used_names: dict[str, int] = {}
    count = 0
    for src in sorted(RAW_DIR.glob("*.txt")):
        clean_text = pattern.sub(replacer, src.read_text(encoding="utf-8"))
        # имя файла тоже анонимизируем (slug мог содержать оригинальный термин)
        new_stem = slugify(pattern.sub(replacer, src.stem.replace("-", " ")))
        if new_stem in used_names:                            # защита от коллизий имён
            used_names[new_stem] += 1
            new_stem = f"{new_stem}-{used_names[new_stem]}"
        else:
            used_names[new_stem] = 0
        out = KB_DIR / f"{new_stem}.md"
        out.write_text(clean_text, encoding="utf-8")
        print(f"{src.name} -> {out.name}")
        count += 1

    print(f"\nГотово: {count} документов в knowledge_base/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
