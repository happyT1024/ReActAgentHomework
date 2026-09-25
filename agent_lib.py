# -*- coding: utf-8 -*-
"""Ядро ReAct-агента для домашнего задания 1.

Основа — семинарский ноутбук homework 1/ai-agent-01.ipynb с заполненными дырками.
Отличия от семинара:
- web_search: исправлены опечатки в параметрах API (explaintext, exintro),
  вступление обрезается до 1500 символов вместо полного текста (экономия контекста и денег);
- добавлен page_find(title, keywords) — чтение тела длинных обзорных статей
  («2026 in Japan»), главный инструмент задания;
- ключ OPENROUTER_API_KEY читается из .env в корне репозитория или в homework 1/`,
  проверка ключа — в момент вызова модели, а не при импорте (чтобы тесты
  инструментов шли без ключа).
"""
import os, re, sys, json, time, ast, operator, subprocess
from pathlib import Path
from dataclasses import dataclass, field
import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field

REPO = Path(__file__).resolve().parent


def _load_env():
    """Читаем .env из корня репозитория и из homework 1 1/ (не перезаписывая уже заданное)."""
    for env in [REPO / ".env", REPO / "homework 1 1" / ".env"]:
        if not env.exists():
            continue
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


_load_env()

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = {"cheap": "openai/gpt-4o-mini", "mid": "anthropic/claude-haiku-4.5", "strong": "anthropic/claude-sonnet-4.6"}
COLORS = {"violet": "#5436A3", "amber": "#F09000", "teal": "#00838F", "red": "#C43C3C", "grey": "#787882"}
TRACES, IMG = Path("traces"), Path("img")


def require_key():
    assert os.getenv("OPENROUTER_API_KEY"), (
        "нет ключа: скопируйте .env.example в .env в корне репозитория и впишите OPENROUTER_API_KEY")


def _headers():
    require_key()
    return {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "HTTP-Referer": "https://postypashki.ru", "X-Title": "agents-course-homework01"}


# ---------- спринт 1: один вызов, цена и время ----------

@dataclass
class Ledger:
    calls: list = field(default_factory=list)

    def add(self, tag, model, usage, seconds=0.0):
        p, c = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        cost = usage.get("cost") or 0.0
        self.calls.append({"tag": tag, "model": model.split("/")[-1], "prompt": p, "completion": c,
                           "cost": cost, "seconds": round(seconds, 2)})
        return cost

    @property
    def total(self):
        return sum(c["cost"] for c in self.calls)

    def table(self):
        df = pd.DataFrame(self.calls)
        return df.groupby(["tag", "model"]).agg(calls=("cost", "size"), prompt=("prompt", "sum"),
                                                completion=("completion", "sum"), cost=("cost", "sum"),
                                                seconds=("seconds", "sum")).round(5)


ledger = Ledger()


def post_with_retry(body, attempts=4):
    problem = "нет ответа"
    for attempt in range(attempts):
        r = requests.post(CHAT_URL, json=body, headers=_headers(), timeout=120)
        if r.status_code == 200:
            return r.json()
        problem = f"HTTP {r.status_code} : {r.text[:5000]}"
        if r.status_code in (429, 500, 502, 503) and attempt < attempts - 1:
            time.sleep(1.5 * (attempt + 1))
            continue
    raise RuntimeError(problem)


def chat(messages, model, tools=None, tag="chat", temperature=None):
    body = {"model": model, "messages": messages, "usage": {"include": True}}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if temperature is not None:
        body["temperature"] = temperature
    started = time.perf_counter()
    data = post_with_retry(body)
    ledger.add(tag, model, data.get("usage") or {}, time.perf_counter() - started)
    return data["choices"][0]["message"]


# ---------- спринт 3: проверка ответа ----------

def normalize(text):
    text = re.sub(r"[^\w\s]", " ", str(text).lower().replace(",", ""))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def is_correct(task, answer):
    return normalize(task["answer"]) in normalize(answer)


def summary(config, model, n, correct, cost, steps, seconds=0.0):
    return {"config": config, "model": model.split("/")[-1], "n": n, "accuracy": round(correct / n, 2),
            "cost_per_task": round(cost / n, 5), "cost_per_correct": round(cost / correct, 5) if correct else float("inf"),
            "avg_steps": round(steps, 2), "avg_seconds": round(seconds, 1)}


# ---------- спринт 4: цикл ReAct ----------

TOOLS = {}


def register(fn, args_model, description):
    TOOLS[fn.__name__] = {"fn": fn, "args": args_model, "schema": {"type": "function", "function": {
        "name": fn.__name__, "description": description, "parameters": args_model.model_json_schema()}}}


OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Pow: operator.pow, ast.USub: operator.neg, ast.Mod: operator.mod}


def evaluate(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](evaluate(node.left), evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](evaluate(node.operand))
    raise ValueError("допустимы только числа и арифметика")


class CalcArgs(BaseModel):
    expr: str = Field(description="арифметическое выражение, например 17*23+5")


def calculator(expr: str) -> str:
    try:
        val = evaluate(ast.parse(expr.replace(",", "."), mode="eval").body)
        return str(int(val)) if float(val).is_integer() else f"{val:.4f}".rstrip("0")
    except Exception as e:
        return f"ошибка вычисления: {e}"


register(calculator, CalcArgs, "Считает арифметическое выражение: числа, скобки, + - * / ** %")


class ExecArgs(BaseModel):
    code: str = Field(description="код на Python; результат надо напечатать через print")


def python_exec(code: str) -> str:
    try:
        r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired:
        return "Код превысил 5 сек"
    out = (r.stdout + r.stderr).strip()
    return out[:5000] if out else "Код исполнился но результат пустой"


register(python_exec, ExecArgs, "Выполняет код на Python в отдельном процессе и возвращает то, что он напечатал")


# ---------- спринт 5: поиск и чтение страницы ----------

WIKI = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "agents-course-homework01/1.0 (https://postypashki.ru; educational project)"}


def wiki(params, attempts=3):
    for attempt in range(attempts):
        try:
            r = requests.get(WIKI, params={**params, "format": "json"}, headers=WIKI_HEADERS, timeout=15)
        except requests.RequestException:
            r = None
        if r is not None and r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
            return r.json()["query"]
        time.sleep(1.0 + attempt)
    raise RuntimeError("Википедия недоступна")


class SearchArgs(BaseModel):
    query: str = Field(description="короткий поисковый запрос: имя, название, термин из вопроса")


def web_search(query: str) -> str:
    """Поиск по англоязычной Википедии: вступление лучшей статьи + ещё два кандидата."""
    hits = wiki({"action": "query", "list": "search", "srsearch": query, "srlimit": 3})["search"]
    if not hits:
        return "nothing found"
    pages = wiki({"action": "query", "prop": "extracts", "explaintext": 1, "exintro": 1,
                  "titles": hits[0]["title"]})["pages"]
    text = " ".join(next(iter(pages.values())).get("extract", "").split())[:1500]
    others = ", ".join(h["title"] for h in hits[1:])
    return f"[{hits[0]['title']}] {text}" + (f" | other articles: {others}" if others else "")


register(web_search, SearchArgs,
         "Searches English Wikipedia and returns the intro of the best article plus titles of two more")


class PageFindArgs(BaseModel):
    title: str = Field(description="точное название статьи Википедии, например '2026 in Japan'")
    keywords: str = Field(description="два-пять ключевых слов из вопроса, которые искать в статье")


def page_find(title: str, keywords: str) -> str:
    """Чтение тела статьи: строки полного текста, где встречается хотя бы половина ключевых слов."""
    pages = wiki({"action": "query", "prop": "extracts", "explaintext": 1, "titles": title, "redirects": 1})["pages"]
    text = next(iter(pages.values())).get("extract", "")
    if not text:
        return "no such page"
    words = [w for w in re.findall(r"\w+", keywords.lower()) if len(w) > 2]
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    hits = [l for l in lines if sum(w in l.lower() for w in words) >= max(1, (len(words) + 1) // 2)]
    return "\n".join(h[:400] for h in hits[:6]) or "keywords not found on the page"


register(page_find, PageFindArgs,
         "Looks inside the full text of a Wikipedia article (for example '2026 in Japan') and returns "
         "the lines containing the keywords; use it when the intro found by web_search is not enough")


# ---------- спринт 4-6: цикл, трейсы, отчёт ----------

def system_prompt(tool_names):
    p = "You answer questions about events of 2025-2026. "
    if "web_search" in tool_names and "page_find" in tool_names:
        p += ("Never answer from memory: use web_search to find the right article, then page_find to look inside "
              "long articles such as '2026 in <country or field>' for the specific fact. ")
    elif "web_search" in tool_names:
        p += "Never answer from memory: use web_search to find the right article and read its intro. "
    if "calculator" in tool_names or "python_exec" in tool_names:
        p += "For any arithmetic or date computation use the tools instead of mental math. "
    p += "Do not repeat the same tool call. When done, write the last line as FINAL: <short answer>."
    return p


@dataclass
class Run:
    question: str
    answer: str
    steps: int
    messages: list
    cost: float = 0.0
    seconds: float = 0.0


def looped(seen, calls):
    keys = [(c["function"]["name"], c["function"]["arguments"]) for c in calls]
    repeated = any(k in seen for k in keys)
    seen.update(keys)
    return repeated


def finish(model, messages, step):
    del messages[-1]
    messages.append({"role": "user", "content": "Инструменты больше недоступны. Ответь по тому, что уже известно, последней строкой FINAL: <ответ>."})
    msg = chat(messages, model, tag="agent")
    messages.append(msg)
    return msg.get("content") or "", step + 1


def run_tool(call):
    name = call["function"]["name"]
    try:
        spec = TOOLS[name]
        args = spec["args"].model_validate_json(call["function"]["arguments"])
        result = spec["fn"](**args.model_dump())
    except Exception as e:
        result = f"НЕ удалось вызвать инструмент {name}: {e}"
    return {"role": "tool", "tool_call_id": call["id"], "content": str(result)[:2000]}


def agent_loop(messages, model, tool_names, max_steps):
    seen = set()
    for step in range(1, max_steps + 1):
        msg = chat(messages, model, tools=[TOOLS[n]["schema"] for n in tool_names] or None, tag="agent")
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            return msg.get("content") or "", step
        if looped(seen, calls) or step == max_steps:
            return finish(model, messages, step)
        messages += [run_tool(c) for c in calls]


def agent(question, model, tool_names, max_steps=8):
    messages = [{"role": "system", "content": system_prompt(tool_names)}, {"role": "user", "content": question}]
    before, started = ledger.total, time.perf_counter()
    answer, steps = agent_loop(messages, model, tool_names, max_steps)
    return Run(question, answer, steps, messages, ledger.total - before, time.perf_counter() - started)


def show_trace(run):
    for m in run.messages[1:]:
        calls = "; ".join(f"{c['function']['name']}{c['function']['arguments']}" for c in m.get("tool_calls") or [])
        text = " ".join((m.get("content") or "").split())[:180]
        print(f"{m['role']:9s}| {text} {calls}")
    print(f"шагов: {run.steps}, цена: {run.cost * 100:.3f} ¢, время: {run.seconds:.1f} c")


def final_answer(text):
    m = re.search(r"FINAL:\s*(.+)", text or "")
    return m.group(1).strip() if m else (text or "").strip()


def load_tasks(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def run_tasks(tasks, model, tool_names, config, max_steps=8, budget=None):
    folder = TRACES / config / model.split("/")[-1]
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in tasks:
        if budget is not None and ledger.total > budget:
            raise RuntimeError(f"бюджет ${budget:.2f} исчерпан, прогон остановлен на {task['id']}")
        try:
            run = agent(task["question"], model, tool_names, max_steps=max_steps)
        except Exception as e:
            run = Run(task["question"], f"ошибка: {e}", 0, [])
        ok = is_correct(task, final_answer(run.answer))
        rows.append({"config": config, "model": model.split("/")[-1], "id": task["id"], "source": task["source"],
                     "correct": ok, "steps": run.steps, "tool_calls": sum(m["role"] == "tool" for m in run.messages),
                     "cost": run.cost, "seconds": round(run.seconds, 1), "answer": final_answer(run.answer)[:60],
                     "gold": task["answer"]})
        (folder / f"{task['id']}.json").write_text(
            json.dumps({"task": task, "answer": run.answer, "correct": ok, "cost": run.cost,
                        "messages": run.messages}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  [{'+' if ok else '-'}] {task['id']} шагов {run.steps} ${run.cost:.4f}", flush=True)
    return pd.DataFrame(rows)


def report(results):
    rows = [summary(config, model, len(df), int(df["correct"].sum()), df["cost"].sum(), df["steps"].mean(), df["seconds"].mean())
            for (config, model), df in results.groupby(["config", "model"])]
    return pd.DataFrame(rows).sort_values("cost_per_task").reset_index(drop=True)


def money_chart(table, path="img/money_chart.png"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, r in table.iterrows():
        color = COLORS["amber"] if "sonnet" in r["model"] else COLORS["violet"]
        ax.scatter(r["cost_per_task"] * 100, r["accuracy"] * 100, s=110, color=color)
        ax.annotate(f"{r['config']}\n{r['model']}", (r["cost_per_task"] * 100, r["accuracy"] * 100),
                    fontsize=8, xytext=(6, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("цена задачи, центы (логарифмическая шкала)")
    ax.set_ylabel("доля верных ответов, %")
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def steps_chart(results):
    g = results.groupby("config").agg(steps=("steps", "mean"), tool_calls=("tool_calls", "mean"), seconds=("seconds", "mean"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    g[["steps", "tool_calls"]].plot.bar(ax=axes[0], color=[COLORS["violet"], COLORS["teal"]], rot=12)
    axes[0].set_ylabel("в среднем на задачу")
    g["seconds"].plot.bar(ax=axes[1], color=COLORS["amber"], rot=12)
    axes[1].set_ylabel("секунд на задачу")
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
        ax.set_xlabel("")
    fig.tight_layout()
    return fig
