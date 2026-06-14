#!/usr/bin/env python3
"""Скачивание страниц русской вики The Witcher и очистка их до чистого текста.

Источник: https://witcher.fandom.com/ru  (MediaWiki API).
Метод: action=parse&prop=text — отрендеренный HTML статьи, который затем
чистится BeautifulSoup до связной прозы. Расширение TextExtracts
(prop=extracts) на Fandom отсутствует, поэтому используется именно parse.

Результат: raw_pages/<slug>.txt — промежуточные СЫРЫЕ тексты (узнаваемый
The Witcher). В репозиторий они не коммитятся (см. raw_pages/.gitignore);
их анонимизирует scripts/anonymize.py.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

API = "https://witcher.fandom.com/ru/api.php"
RAW_DIR = Path(__file__).resolve().parent.parent / "raw_pages"
HEADERS = {"User-Agent": "QuantumForge-RAG-KB-Builder/1.0 (educational project)"}
REQUEST_PAUSE_SEC = 1.0      # вежливый rate-limit к Fandom
MIN_TEXT_LEN = 500           # отсев заглушек/disambig-страниц

# Список сущностей базы знаний (персонажи, чудовища, магия, локации, расы).
# Неточные заголовки подстрахованы redirects=1 на стороне API.
PAGES = [
    # Персонажи
    "Геральт из Ривии", "Йеннифэр из Венгерберга", "Трисс Меригольд", "Цири",
    "Весемир", "Лютик", "Эмгыр вар Эмрейс", "Вильгефорц", "Эмиель Регис",
    "Ламберт", "Золтан Хивай",
    # Чудовища
    "Утопец", "Леший", "Грифон", "Стрыга", "Кикимора", "Накер",
    "Высший вампир", "Допплер", "Виверна",
    # Магия и профессия
    "Ведьмак", "Ведьмачьи Знаки", "Игни", "Аард", "Квен", "Мутаген", "Испытание Травами",
    # Локации и державы
    "Каэр Морхен", "Нильфгаард", "Новиград", "Велен", "Скеллиге", "Цинтра",
    "Темерия", "Редания", "Туссент", "Венгерберг",
    # Организации и расы
    "Скоя’таэли", "Дикая Охота", "Ложа чародеек", "Эльфы",
]

# Селекторы не-прозаического мусора, удаляемого перед извлечением текста.
DROP_SELECTORS = (
    "table", "figure", "style", "script", "audio", "img",
    "sup.reference", "ol.references", "div.navbox", "div.toc",
    "span.mw-editsection", "aside", "dl", ".hatnote", ".dablink",
    ".navigation-not-searchable", ".canontabs", ".tabbertab", ".mw-empty-elt",
)


def fetch_html(title: str) -> tuple[str, str]:
    """Возвращает (каноничный_заголовок, html) статьи. redirects=1 разрешает алиасы."""
    params = {
        "action": "parse", "page": title, "prop": "text|displaytitle",
        "redirects": 1, "format": "json", "formatversion": 2,
    }
    resp = requests.get(API, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(data["error"].get("info", "unknown API error"))
    parse = data["parse"]
    return parse["title"], parse["text"]


# Хвостовые служебные разделы: ссылки, цитаты, авторы, названия книг/серий —
# основной источник латинского «мусора». На первом таком заголовке сбор текста
# прекращается (лор всегда идёт раньше этих разделов).
STOP_SECTIONS = (
    "примечани", "источник", "литератур", "ссылк", "галере", "появлени",
    "интересные факт", "за кулисам", "каноничн", "видео", "навигаци",
    "см. также", "смотрите также", "сноск", "библиограф", "в других",
    "на других", "цитат", "разное",
)


def clean_html(raw_html: str) -> str:
    """Превращает HTML статьи в связный plain-text (заголовки + абзацы + списки)."""
    soup = BeautifulSoup(raw_html, "lxml")
    content = soup.select_one("div.mw-parser-output") or soup
    for selector in DROP_SELECTORS:
        for element in content.select(selector):
            element.decompose()

    parts: list[str] = []
    for element in content.find_all(["h2", "h3", "p", "li"]):
        text = element.get_text(" ", strip=True)
        text = re.sub(r"\[\d+\]", "", text)          # хвосты сносок [12]
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 3:
            continue
        if element.name in ("h2", "h3"):
            head = text.lower().strip(" .:")
            if any(head.startswith(s) for s in STOP_SECTIONS):
                break                                # дальше — служебные разделы
            parts.append(f"\n## {text}\n")
        else:
            parts.append(text)

    text = "\n\n".join(parts)
    return tidy(text)


def tidy(text: str) -> str:
    """Финальная чистка: снять знаки ударения и невидимые символы, убрать латиницу
    (этимология, Старшая речь, цитаты, названия книг/игр) — база знаний должна быть
    на чистом русском, а латинские вкрапления узнаваемы как The Witcher. Затем
    подчистить осиротевшую пунктуацию."""
    # снять ТОЛЬКО знаки ударения (U+0301 акут, U+0300 гравис: «Ге́ральт»
    # ломает \\b-сопоставление). NFD нельзя — он разложил бы й→и и ё→е.
    text = re.sub("[\u0300\u0301\u0340\u0341]", "", text)
    text = re.sub("[\u00ad\u200b\u200c\u200d\ufeff]", "", text)  # переносы, zero-width
    text = re.sub(r"[A-Za-z][A-Za-z'’\-]*", "", text)        # латинские токены
    text = re.sub(r"«\s*»|\(\s*[—,;:.]*\s*\)|\[\s*\]|\{\s*\}", "", text)  # пустые скобки/кавычки
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)             # пробел перед пунктуацией
    text = re.sub(r"([(«]) +", r"\1", text)                  # пробел после открывающей скобки
    text = re.sub(r"[ \t]{2,}", " ", text)                   # двойные пробелы
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def slugify(title: str) -> str:
    """Имя файла из заголовка: кириллица сохраняется, пунктуация → '-'."""
    return re.sub(r"[^\w]+", "-", title.lower(), flags=re.UNICODE).strip("-")


def main() -> int:
    RAW_DIR.mkdir(exist_ok=True)
    failed: list[str] = []
    for i, title in enumerate(PAGES, 1):
        try:
            canonical, html = fetch_html(title)
            text = clean_html(html)
            if len(text) < MIN_TEXT_LEN:
                raise RuntimeError(f"слишком короткий текст ({len(text)} симв.)")
            out_path = RAW_DIR / f"{slugify(canonical)}.txt"
            out_path.write_text(f"# {canonical}\n\n{text}\n", encoding="utf-8")
            print(f"[{i:>2}/{len(PAGES)}] OK  {title} -> {out_path.name} ({len(text)} симв.)")
        except Exception as exc:  # noqa: BLE001 — намеренно не падаем на первой ошибке
            print(f"[{i:>2}/{len(PAGES)}] ERR {title}: {exc}", file=sys.stderr)
            failed.append(title)
        time.sleep(REQUEST_PAUSE_SEC)

    saved = len(PAGES) - len(failed)
    print(f"\nИтог: сохранено {saved}, ошибок {len(failed)}.")
    if failed:
        print(f"Не удалось: {failed}", file=sys.stderr)
    return 1 if saved < 30 else 0


if __name__ == "__main__":
    sys.exit(main())
