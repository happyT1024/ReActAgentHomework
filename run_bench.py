# -*- coding: utf-8 -*-
"""Замер (шаг 3 задания): все обязательные конфигурации на всём наборе.

Запуск: python3 run_bench.py [--dataset data/fresh_2026.jsonl] [--limit N] [--max-steps 8] [--budget 5.0]
Результат: traces/<config>/<model>/<id>.json, results_raw.csv, results.md, img/money_chart.png.
Бюджет по заданию: 59 вопросов в пяти конфигурациях обходятся в один-два доллара;
если счётчик ушёл за пять — стоп.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

import agent_lib as al

# Обязательные строки таблицы из задания
CONFIGS = [
    ("сильная модель, без инструментов", al.MODELS["strong"], []),
    ("дешёвая модель, без инструментов", al.MODELS["cheap"], []),
    ("дешёвая модель, только поиск", al.MODELS["cheap"], ["web_search"]),
    ("дешёвая модель, поиск и чтение страницы", al.MODELS["cheap"], ["web_search", "page_find"]),
    ("средняя модель, лучший набор инструментов", al.MODELS["mid"], ["web_search", "page_find"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/fresh_2026.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="взять только первые N вопросов (0 = все)")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--budget", type=float, default=5.0, help="стоп, если суммарные расходы превысят")
    args = ap.parse_args()

    al.require_key()
    tasks = al.load_tasks(args.dataset)
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"набор: {args.dataset}, вопросов: {len(tasks)}", flush=True)

    frames = []
    for config, model, tools in CONFIGS:
        print(f"\n=== {config} | {model} | инструменты: {tools or '—'} ===", flush=True)
        started = al.ledger.total
        frames.append(al.run_tasks(tasks, model, tools, config, max_steps=args.max_steps, budget=args.budget))
        print(f"конфигурация стоила ${al.ledger.total - started:.3f}, всего ${al.ledger.total:.3f}", flush=True)

    results = pd.concat(frames, ignore_index=True)
    table = al.report(results)

    results.to_csv("results_raw.csv", index=False)
    Path("results.md").write_text(table.to_string(index=False), encoding="utf-8")
    al.IMG.mkdir(exist_ok=True)
    al.money_chart(table, "img/money_chart.png")

    print("\n=== ИТОГОВАЯ ТАБЛИЦА ===")
    print(table.to_string(index=False))
    print(f"\nвсего потрачено: ${al.ledger.total:.3f}")
    print("файлы: results_raw.csv, results.md, img/money_chart.png, traces/")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"\nСТОП: {e}", file=sys.stderr)
        sys.exit(1)
