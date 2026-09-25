# -*- coding: utf-8 -*-
"""Тесты инструментов агента: по три случая на каждый — нормальный вход, пустой результат, ошибка.

Запускаются без модели и без ключа (network нужен только Википедии).
Запуск: python3 test_tools.py  или  python3 -m pytest test_tools.py -q
"""
import sys
from unittest import mock

import requests

import agent_lib as al


# ---------- calculator ----------

def test_calculator_normal():
    assert al.calculator("17*23+5") == "396"
    assert al.calculator("2**10") == "1024"


def test_calculator_empty():
    assert al.calculator("").startswith("ошибка вычисления")


def test_calculator_error():
    assert "ошибка вычисления" in al.calculator("__import__('os')")


# ---------- python_exec ----------

def test_python_exec_normal():
    assert al.python_exec("print(sum(range(10)))") == "45"


def test_python_exec_empty():
    assert "пустой" in al.python_exec("pass")


def test_python_exec_error():
    assert "ZeroDivisionError" in al.python_exec("1/0")


# ---------- web_search (реальная Википедия, ключ не нужен) ----------

def test_web_search_normal():
    assert al.web_search("Scott Derrickson").startswith("[Scott Derrickson]")


def test_web_search_empty():
    assert al.web_search("qzxwv no such term 47821 blah") == "nothing found"


def test_web_search_error():
    with mock.patch.object(al.requests, "get", side_effect=requests.RequestException("network down")):
        try:
            al.web_search("test")
            raised = False
        except RuntimeError:
            raised = True
        assert raised, "ожидали RuntimeError при недоступности Википедии"


# ---------- page_find (реальная Википедия, ключ не нужен) ----------

def test_page_find_normal():
    out = al.page_find("2026 in India", "January India")
    assert out not in ("no such page", "keywords not found on the page")
    assert len(out) > 50


def test_page_find_empty():
    assert al.page_find("2026 in India", "qzxwv nothere") == "keywords not found on the page"


def test_page_find_no_page():
    assert al.page_find("Xyzzy Page Does Not Exist 98123", "anything") == "no such page"


# ---------- контракт цикла: ошибка инструмента возвращается текстом, а не роняет программу ----------

def test_run_tool_unknown_tool():
    call = {"id": "t1", "function": {"name": "no_such_tool", "arguments": "{}"}}
    msg = al.run_tool(call)
    assert msg["role"] == "tool" and "НЕ удалось" in msg["content"]


def test_run_tool_broken_arguments():
    call = {"id": "t2", "function": {"name": "web_search", "arguments": "не json"}}
    msg = al.run_tool(call)
    assert msg["role"] == "tool" and "НЕ удалось" in msg["content"]


def test_register_schema():
    for name in ["calculator", "python_exec", "web_search", "page_find"]:
        schema = al.TOOLS[name]["schema"]["function"]
        assert schema["name"] == name and schema["description"] and schema["parameters"]["properties"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"OK    {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} прошло")
    sys.exit(1 if failed else 0)
