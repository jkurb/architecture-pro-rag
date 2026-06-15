#!/usr/bin/env python3
"""Автоматическая оценка качества RAG-бота на «золотом наборе» (Задание 7).

Прогоняет golden_questions.json через RAG-бот поверх индекса с искусственными
пробелами (Task7/index_gapped/), логирует каждый запрос в logs.jsonl и оценивает
корректность:

  * known-вопрос (should_answer=true): успех = бот ответил И покрыл ≥50% ожидаемых
    ключевых слов; иначе coverage_miss (отказался) или weak_answer (ответил неполно);
  * gap-вопрос (should_answer=false): успех = бот честно отказался («Я не знаю»);
    иначе hallucination (выдал ответ на удалённую/отсутствующую тему).

Результат: logs.jsonl (по строке на запрос) + report.md (агрегаты и выводы).
Запуск: ANTHROPIC_API_KEY=... python Task7/evaluate.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Task5"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_core import DEFENDED, EMBED_MODEL, LLM_MODEL, RagEngine  # noqa: E402
from query_logger import QueryLogger, extract_answer_line, is_refusal  # noqa: E402

GOLDEN = REPO / "Task7" / "golden_questions.json"
GAPPED_INDEX = REPO / "Task7" / "index_gapped"
LOGS = REPO / "Task7" / "logs.jsonl"
REPORT = REPO / "Task7" / "report.md"
PACE = float(os.getenv("PACE_SECONDS", "2"))
COVERAGE_PASS = 0.5     # доля ожидаемых ключевых слов для зачёта known-вопроса


def score(q: dict, answer: str) -> dict:
    """Оценка одного ответа: статус, успех, покрытие ключевых слов."""
    refused = is_refusal(answer)
    line = extract_answer_line(answer).lower()
    kws = q.get("expected_keywords", [])
    matched = [k for k in kws if k.lower() in line or k.lower() in answer.lower()]
    coverage = round(len(matched) / len(kws), 2) if kws else None

    if q["should_answer"]:
        if refused:
            status, ok = "coverage_miss", False
        elif coverage is not None and coverage >= COVERAGE_PASS:
            status, ok = "answered_ok", True
        else:
            status, ok = "weak_answer", False
    else:
        status, ok = ("correct_refusal", True) if refused else ("hallucination", False)

    return {"refused": refused, "matched_keywords": matched,
            "keyword_coverage": coverage, "status": status, "success": ok}


def build_report(rows: list[dict], removed: list[str]) -> str:
    known = [r for r in rows if r["should_answer"]]
    gaps = [r for r in rows if not r["should_answer"]]
    ok_known = [r for r in known if r["status"] == "answered_ok"]
    weak = [r for r in known if r["status"] == "weak_answer"]
    miss = [r for r in known if r["status"] == "coverage_miss"]
    good_refusal = [r for r in gaps if r["status"] == "correct_refusal"]
    halluc = [r for r in gaps if r["status"] == "hallucination"]
    passed = [r for r in rows if r["success"]]
    cov = [r["keyword_coverage"] for r in known if r["keyword_coverage"] is not None]
    mean_cov = round(sum(cov) / len(cov), 2) if cov else 0.0

    L = []
    L.append("# Отчёт об оценке покрытия и качества базы знаний (Задание 7)\n")
    L.append(f"Модель: `{LLM_MODEL}` · эмбеддинги: `{EMBED_MODEL}` · "
             f"индекс: `Task7/index_gapped/` (прод-индекс минус удалённые сущности)\n")
    L.append(f"Искусственно удалены: {', '.join('`'+e+'`' for e in removed)}\n")

    L.append("## Сводные метрики\n")
    L.append("| Метрика | Значение |")
    L.append("|---|---|")
    L.append(f"| Всего вопросов | {len(rows)} |")
    L.append(f"| Общая точность (ожидаемое поведение) | "
             f"{len(passed)}/{len(rows)} = {round(100*len(passed)/len(rows))}% |")
    L.append(f"| Known: полный ответ | {len(ok_known)}/{len(known)} |")
    L.append(f"| Known: неполный ответ (weak) | {len(weak)}/{len(known)} |")
    L.append(f"| Known: пропуск (бот отказался) | {len(miss)}/{len(known)} |")
    L.append(f"| Known: среднее покрытие ключевых слов | {mean_cov} |")
    L.append(f"| Gaps: корректный отказ | {len(good_refusal)}/{len(gaps)} |")
    L.append(f"| Gaps: галлюцинации (ответил, хотя не должен) | {len(halluc)}/{len(gaps)} |")
    L.append("")

    L.append("## Результаты по вопросам\n")
    L.append("| # | Тема | Ожид. | Статус | Покрытие | Ближайший источник |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        exp = "ответ" if r["should_answer"] else "отказ"
        cov = "—" if r["keyword_coverage"] is None else f"{r['keyword_coverage']:.2f}"
        src = r["sources"][0] if r["sources"] else "—"
        mark = "✅" if r["success"] else "❌"
        L.append(f"| {r['id']} | {r['topic']} | {exp} | {mark} {r['status']} | {cov} | {src} |")
    L.append("")

    L.append("## Плохо покрытые темы\n")
    bad = miss + weak + halluc
    if bad:
        for r in bad:
            why = {"coverage_miss": "бот отказался, хотя тема в базе",
                   "weak_answer": "ответ неполный (мало ключевых фактов)",
                   "hallucination": "бот ответил на удалённую/отсутствующую тему"}[r["status"]]
            L.append(f"- **{r['topic']}** (вопрос #{r['id']}): {why}.")
    else:
        L.append("- Критичных провалов не выявлено на known-темах.")
    L.append("")

    L.append("## Нерелевантные источники (на удалённых темах)\n")
    L.append("После удаления дедицированных документов бот всё ещё извлекает по теме "
             "тангенциальные чанки из других файлов — потенциальный источник «уверенных, "
             "но неверных» ответов:\n")
    for r in gaps:
        L.append(f"- #{r['id']} «{r['topic']}» → извлечено: "
                 f"{', '.join(r['sources']) or '—'} (top score {r['top_score']}).")
    L.append("")

    n_gaps = len(miss) + len(weak) + len(halluc) + len(removed)
    L.append("## Выводы\n")
    L.append(f"- **Выявлено пробелов:** {len(removed)} удалённых сущности "
             f"(булькало, шальная-облава, орлокот) + "
             f"{len(miss)+len(weak)} проблемных known-ответов.")
    L.append(f"- **Корректность отказов:** {len(good_refusal)}/{len(gaps)} — бот честно "
             f"говорит «не знаю» на пробелах, галлюцинаций: {len(halluc)}.")
    L.append(f"- **Покрытие known-тем:** {len(ok_known)}/{len(known)} полных ответов, "
             f"среднее покрытие ключевых слов {mean_cov}.")
    L.append("")
    L.append("## Рекомендации по улучшению базы\n")
    L.append("1. Вернуть/дополнить дедицированные документы по выявленным пробелам "
             "(булькало, шальная-облава, орлокот) — это даст прямые ответы вместо отказов.")
    L.append("2. Для weak-ответов — обогатить документы недостающими фактами "
             "(ключевые слова из золотого набора, которых не было в ответе).")
    L.append("3. Регулярно прогонять `evaluate.py` после обновления индекса (Задание 6) "
             "и отслеживать долю отказов/галлюцинаций как метрику качества.")
    L.append("4. Логи `logs.jsonl` использовать для анализа реальных запросов: темы с "
             "частыми отказами или низким top_score — кандидаты на пополнение базы.")
    return "\n".join(L)


def main() -> int:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Не задан ANTHROPIC_API_KEY.", file=sys.stderr)
        return 1
    if not GAPPED_INDEX.exists():
        print("Нет индекса с пробелами. Сначала: python Task7/build_gapped_index.py",
              file=sys.stderr)
        return 1

    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    removed = data["removed_entities"]
    questions = data["questions"]

    engine = RagEngine(index_dir=GAPPED_INDEX, defense=DEFENDED)
    logger = QueryLogger(LOGS)

    rows = []
    print(f"Оценка {len(questions)} вопросов на индексе с пробелами…\n")
    for i, q in enumerate(questions):
        if i:
            time.sleep(PACE)
        answer, retrieved, _ = engine.ask(q["question"])
        sc = score(q, answer)
        rec = logger.log(q["question"], answer, retrieved, extra={
            "topic": q["topic"],
            "should_answer": q["should_answer"],
            "expected_keywords": q.get("expected_keywords", []),
            **sc,
        })
        row = {"id": q["id"], "topic": q["topic"], "should_answer": q["should_answer"],
               "sources": rec["sources"], "top_score": rec["top_score"], **sc}
        rows.append(row)
        mark = "OK " if sc["success"] else "FAIL"
        print(f"[{mark}] #{q['id']:>2} {sc['status']:<16} {q['topic']}")

    REPORT.write_text(build_report(rows, removed), encoding="utf-8")
    passed = sum(r["success"] for r in rows)
    print(f"\nИтог: {passed}/{len(rows)} ожидаемое поведение. "
          f"Лог: {LOGS.relative_to(REPO)}, отчёт: {REPORT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
