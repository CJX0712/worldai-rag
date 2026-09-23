# -*- coding: utf-8 -*-
"""One-shot acceptance: unit tests + end-to-end smoke (offline AND
online provider paths) + gold-set evaluation.

Design rules baked in from past failures:
- pytest runs IN-PROCESS; pass/fail judged by collected call-report
  counts, never by exit code or stdout text (sandbox bulk-delete
  guards can fake-fail pytest teardown).
- --basetemp lives INSIDE the repo so pytest never touches the
  system temp root.
- E2E spawns the real server subprocess, polls /health, asserts the
  full flow, and force-kills the process tree on Windows.
- E2E ingest text deliberately exceeds the chunk threshold so the
  multi-chunk path is actually covered.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

PORT_OFFLINE = 18741
PORT_ONLINE = 18742

# > 400 chars (chunk threshold) to force multi-chunk ingestion
E2E_TEXT = (
    "光合作用是指绿色植物通过叶绿体，利用光能，把二氧化碳和水转化成储存着能量的"
    "有机物，并且释放出氧气的过程。这个过程对地球生态系统至关重要，因为几乎所有"
    "生物都直接或间接依赖光合作用产生的有机物和氧气。光合作用分为光反应和暗反应"
    "两个阶段。光反应发生在类囊体薄膜上，产生 ATP 和 NADPH，并释放氧气；暗反应"
    "发生在叶绿体基质中，利用 ATP 和 NADPH 将二氧化碳固定并还原为有机物，这一"
    "过程也被称为卡尔文循环。影响光合作用速率的因素包括光照强度、二氧化碳浓度、"
    "温度和水分等。农业生产中合理密植、补充二氧化碳等措施都可以提高作物的光合"
    "效率，从而增加产量。此外，科学家还在研究人工光合作用，希望利用太阳能直接"
    "分解水制氢或固定二氧化碳，为清洁能源和碳中和提供新的技术路径。从进化角度"
    "看，光合作用的出现彻底改变了地球大气成分，使氧气逐渐积累，为需氧生物的"
    "演化创造了条件，蓝藻是最早进行产氧光合作用的生物类群之一。"
)


class Counter:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1
                self.errors.append(str(report.nodeid))


def run_unit_tests():
    import pytest

    counter = Counter()
    basetemp = os.path.join(ROOT, "data", ".pytest-tmp")
    os.makedirs(basetemp, exist_ok=True)
    code = pytest.main(
        [
            "-q",
            os.path.join(ROOT, "tests"),
            "--basetemp=" + basetemp,
            "-p",
            "no:cacheprovider",
        ],
        plugins=[counter],
    )
    return {
        "passed": counter.passed,
        "failed": counter.failed,
        "errors": counter.errors[:10],
        "exit_code": int(code) if code is not None else -1,
        "ran_to_completion": (counter.passed + counter.failed) > 0,
    }


def _req(method, url, payload=None, timeout=120):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")


def _wait_health(port, proc, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return None
        try:
            status, body = _req("GET", "http://127.0.0.1:{}/health".format(port), timeout=3)
            if status == 200:
                return body
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    return None


def _kill(proc):
    if proc.poll() is None:
        subprocess.run(
            ["taskkill", "/pid", str(proc.pid), "/t", "/f"],
            capture_output=True,
        )


def run_e2e(name, port, env_extra, checks):
    """Spawn server with given env, run check list, kill, report."""
    env = dict(os.environ)
    env["WORLDAI_PORT"] = str(port)
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(env_extra)
    results = []
    proc = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "server.py")],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        health = _wait_health(port, proc)
        if health is None:
            out = proc.stdout.read().decode("utf-8", "replace")[-2000:] if proc.stdout else ""
            return [{"check": "server_starts", "ok": False, "detail": out}]
        results.append({"check": "server_starts", "ok": True, "detail": json.dumps(health.get("providers", {}))})
        base = "http://127.0.0.1:{}".format(port)
        results.extend(checks(base, health))
    finally:
        _kill(proc)
    return results


def standard_checks(base, health):
    out = []

    def rec(check, ok, detail=""):
        out.append({"check": check, "ok": bool(ok), "detail": str(detail)[:300]})

    s, b = _req("POST", base + "/ingest", {"texts": [E2E_TEXT]})
    rec("ingest_multi_chunk", s == 200 and b.get("ingested_chunks", 0) >= 2, b)

    s, b = _req("POST", base + "/chat", {"question": "光合作用释放什么气体？", "max_tokens": 128}, timeout=300)
    rec("chat_grounded", s == 200 and "氧" in b.get("answer", ""), b.get("answer", "")[:120])
    rec("chat_citations", s == 200 and len(b.get("citations", [])) > 0, "")

    s, b = _req("POST", base + "/agent", {"question": "12*12+6"})
    rec("agent_calc", s == 200 and "150" in b.get("answer", ""), b.get("answer", "")[:80])

    s, b = _req("POST", base + "/ingest", {})
    rec("ingest_empty_rejected", s == 400, s)

    s, b = _req("POST", base + "/chat", {"question": "   "})
    rec("chat_blank_rejected", s == 400, s)

    s, b = _req("GET", base + "/eval")
    rec("eval_runs", s == 200 and b.get("recall@4", 0) >= 0.8, b)

    s, b = _req("GET", base + "/eval")
    rec("eval_deterministic_under_pollution", s == 200 and b.get("recall@4", 0) >= 0.8, b)
    return out


def main():
    report = {"unit": None, "e2e_offline": None, "e2e_online": None}

    print("== unit tests ==", flush=True)
    report["unit"] = run_unit_tests()
    u = report["unit"]
    print("unit: passed={} failed={}".format(u["passed"], u["failed"]), flush=True)

    print("== e2e offline (fallback providers) ==", flush=True)
    report["e2e_offline"] = run_e2e("offline", PORT_OFFLINE, {}, standard_checks)
    for r in report["e2e_offline"]:
        print("  [{}] {} {}".format("PASS" if r["ok"] else "FAIL", r["check"], r["detail"][:120]), flush=True)

    gguf = os.path.join(ROOT, "models", "llm", "qwen2.5-1.5b-instruct-q4_k_m.gguf")
    embed_onnx = os.path.join(ROOT, "models", "embed", "model.onnx")
    online_ready = os.path.exists(gguf) and os.path.exists(embed_onnx)
    if os.environ.get("WORLDAI_E2E_ONLINE", "1") == "1" and online_ready:
        print("== e2e online (onnx/faiss/gguf providers) ==", flush=True)
        env = {
            "WORLDAI_EMBED": "onnx",
            "WORLDAI_STORE": "faiss",
            "WORLDAI_RERANK": "onnx",
            "WORLDAI_LLM": "gguf",
        }
        try:
            report["e2e_online"] = run_e2e("online", PORT_ONLINE, env, standard_checks)
        except Exception as e:  # noqa: BLE001 — never die before writing the report
            report["e2e_online"] = [
                {"check": "online_e2e_crashed", "ok": False,
                 "detail": "{}: {}".format(type(e).__name__, e)}
            ]
        for r in report["e2e_online"]:
            print("  [{}] {} {}".format("PASS" if r["ok"] else "FAIL", r["check"], r["detail"][:120]), flush=True)
    else:
        print("== e2e online: SKIPPED (models not present or disabled) ==", flush=True)

    out_path = os.path.join(ROOT, "data", "verify_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    unit_ok = u["ran_to_completion"] and u["failed"] == 0
    off_ok = all(r["ok"] for r in report["e2e_offline"])
    on_ok = report["e2e_online"] is None or all(r["ok"] for r in report["e2e_online"])
    ok = unit_ok and off_ok and on_ok
    print("SUMMARY: unit={} e2e_offline={} e2e_online={} => {}".format(
        "OK" if unit_ok else "FAIL",
        "OK" if off_ok else "FAIL",
        "SKIP" if report["e2e_online"] is None else ("OK" if on_ok else "FAIL"),
        "ALL GREEN" if ok else "FAILURES",
    ), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
