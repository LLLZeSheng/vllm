# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched streaming acceptance/latency benchmark, D counters only for PD."""

import argparse
import concurrent.futures
import copy
import hashlib
import json
import random
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

import regex as re

SPEC = re.compile(
    r"^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total(?:\{[^}]*\})?\s+([\d.eE+-]+)$"
)


def get_metrics(url):
    with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=30) as f:
        text = f.read().decode()
    values = {
        k: 0.0
        for k in ("drafts", "draft_tokens", "accepted_tokens", "success", "outstanding")
    }
    for line in text.splitlines():
        m = SPEC.match(line)
        if m:
            values[m[1]] += float(m[2])
        elif line.startswith("vllm:request_success_total{"):
            values["success"] += float(line.rsplit(" ", 1)[1])
        elif line.startswith(
            ("vllm:num_requests_running{", "vllm:num_requests_waiting{")
        ):
            values["outstanding"] += float(line.rsplit(" ", 1)[1])
    return text, values


def select(path, count, seed):
    rng = random.Random(seed)
    chosen = []
    with path.open() as f:
        for index, line in enumerate(f):
            if index < count:
                chosen.append((index, line))
            else:
                j = rng.randrange(index + 1)
                if j < count:
                    chosen[j] = (index, line)
    assert len(chosen) == count
    return [(i, json.loads(line)) for i, line in sorted(chosen)]


def send(url, model, index, payload, cap, run_id, request_timeout=1800):
    allowed = {
        "messages",
        "temperature",
        "top_p",
        "tools",
        "tool_choice",
        "stop",
        "seed",
        "parallel_tool_calls",
        "chat_template_kwargs",
    }
    body = {k: v for k, v in payload.items() if k in allowed}
    # vLLM accepts omitted function.parameters, but the deployed Rust router
    # requires an object. Normalize identically in TP and PD, never drop rows.
    normalized_functions = []
    if body.get("tools"):
        body["tools"] = copy.deepcopy(body["tools"])
        for tool_index, tool in enumerate(body["tools"]):
            function = tool.get("function")
            if function is not None and function.get("parameters") is None:
                function["parameters"] = {}
                normalized_functions.append(tool_index)
    if body.get("tools") == []:
        body.pop("tools")
    body.update(model=model, stream=True, stream_options={"include_usage": True})
    body.setdefault("seed", 20260918 + int(payload.get("__source_index", index)))
    body["max_tokens"] = min(
        cap,
        int(payload.get("max_tokens") or payload.get("max_completion_tokens") or cap),
    )
    encoded = json.dumps(body, ensure_ascii=False).encode()
    row = {
        "index": index,
        "source_index": payload.get("__source_index", index),
        "request_sha256": hashlib.sha256(encoded).hexdigest(),
        "function_parameters_defaulted_at_tool_indices": normalized_functions,
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=encoded,
        headers={
            "Content-Type": "application/json",
            "x-request-id": f"{run_id}-{index}",
        },
    )
    start = time.monotonic()
    first = last = None
    text_parts = []
    events = 0
    usage = {}
    finish = None
    done = False
    try:
        with urllib.request.urlopen(request, timeout=request_timeout) as response:
            row["prefill_instance"] = response.headers.get("x-prefill-instance")
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                if not data:
                    continue
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(str(event["error"])[:500])
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    nonempty = any(
                        delta.get(k)
                        for k in (
                            "content",
                            "reasoning",
                            "reasoning_content",
                            "tool_calls",
                            "function_call",
                        )
                    )
                    if nonempty:
                        now = time.monotonic()
                        if first is None:
                            first = now
                        last = now
                        events += 1
                        text_parts.append(
                            json.dumps(delta, sort_keys=True, ensure_ascii=False)
                        )
                    if choice.get("finish_reason") is not None:
                        finish = choice["finish_reason"]
        end = time.monotonic()
        if not done or "completion_tokens" not in usage:
            raise RuntimeError("Incomplete SSE stream or missing final token usage")
        n = usage["completion_tokens"]
        if n <= 0 or finish not in ("stop", "length", "tool_calls", "function_call"):
            raise RuntimeError(
                f"Failed generation: completion_tokens={n}, finish_reason={finish!r}"
            )
        row["output_deltas"] = [json.loads(x) for x in text_parts]
        row["output_content"] = "".join(
            d.get("content") or "" for d in row["output_deltas"]
        )
        row["output_reasoning"] = "".join(
            d.get("reasoning_content") or d.get("reasoning") or ""
            for d in row["output_deltas"]
        )
        row.update(
            status=200,
            elapsed_s=end - start,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=n,
            ttft_s=None if first is None else first - start,
            tpot_s=None if first is None or n <= 1 else (end - first) / (n - 1),
            last_output_s=None if last is None else last - start,
            output_events=events,
            finish_reason=finish,
            output_sha256=hashlib.sha256("".join(text_parts).encode()).hexdigest(),
        )
    except Exception as e:
        row.update(
            status=getattr(e, "code", "error"),
            error=repr(e)[:600],
            elapsed_s=time.monotonic() - start,
        )
        if isinstance(e, urllib.error.HTTPError):
            row["error_body"] = e.read().decode(errors="replace")[:700]
    return row


def distribution(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None

    def quantile(q):
        x = (len(vals) - 1) * q
        i = int(x)
        f = x - i
        return vals[i] * (1 - f) + vals[min(i + 1, len(vals) - 1)] * f

    return {
        "count": len(vals),
        "mean": statistics.mean(vals),
        "p50": quantile(0.5),
        "p95": quantile(0.95),
    }


def producer_snapshots(manifest):
    snapshots = {}
    for service in manifest.get("services", []):
        if service["role"] != "prefill":
            continue
        url = f"http://{service['host']}:{service['port']}"
        raw, counts = get_metrics(url)
        counts["nixl_bytes_sent"] = sum(
            float(line.rsplit(" ", 1)[1])
            for line in raw.splitlines()
            if line.startswith("vllm:nixl_bytes_transferred_sum{")
        )
        for key, metric in [
            ("nixl_failed", "vllm:nixl_num_failed_transfers_total{"),
            ("kv_expired", "vllm:nixl_num_kv_expired_reqs_total{"),
        ]:
            counts[key] = sum(
                float(line.rsplit(" ", 1)[1])
                for line in raw.splitlines()
                if line.startswith(metric)
            )
        snapshots[url] = counts
    return snapshots


def mooncake_producer_snapshots(manifest, output, phase):
    """The fork logs transfer bytes but has no Mooncake Prometheus exporter.

    Record real producer request counters rather than pretending NIXL's
    transfer metrics also exist for Mooncake.
    """
    snapshots = {}
    for role, service in manifest.get("roles", {}).items():
        if not role.startswith("p"):
            continue
        command = service["command"]
        port = command[command.index("--port") + 1]
        url = "http://" + service["host"] + ":" + port
        raw, counts = get_metrics(url)
        output.with_suffix(f".{role}.metrics-{phase}.txt").write_text(raw)
        snapshots[role] = counts
    return snapshots


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--metrics-url", help="Direct decode metrics endpoint for PD")
    p.add_argument("--model", default="glm53-dspark-benchmark")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=800)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--num-spec-tokens", type=int, required=True)
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--wait-ready-seconds", type=int, default=0)
    p.add_argument("--request-timeout", type=int, default=1800)
    p.add_argument("--fail-fast-errors", type=int, default=1)
    a = p.parse_args()
    if a.manifest is None:
        a.manifest = a.output.parent / "manifest.json"
    assert not a.output.exists(), "Refuse overwriting completed benchmark"
    a.output.parent.mkdir(parents=True, exist_ok=True)
    selected = select(a.dataset, a.count, a.seed)
    deadline = time.monotonic() + a.wait_ready_seconds
    while a.wait_ready_seconds:
        try:
            with urllib.request.urlopen(
                a.base_url.rstrip("/") + "/health", timeout=5
            ) as response:
                if response.status == 200:
                    break
        except Exception:
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError("Service readiness deadline expired")
        time.sleep(10)
    metrics_url = a.metrics_url or a.base_url
    before_text, before = get_metrics(metrics_url)
    assert before["outstanding"] == 0, "Benchmark endpoint is not drained"
    time.sleep(12)
    before_text, before = get_metrics(metrics_url)
    a.output.with_suffix(".metrics-before.txt").write_text(before_text)
    a.output.with_suffix(".sample-indices.json").write_text(
        json.dumps([i for i, _ in selected])
    )
    manifest = json.loads(a.manifest.read_text())
    producer_before = producer_snapshots(manifest)
    mooncake_before = mooncake_producer_snapshots(manifest, a.output, "before")
    start = time.monotonic()
    run_id = f"pd031-{time.time_ns()}"
    results = []
    abort_reason = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        futures = [
            pool.submit(
                send,
                a.base_url,
                a.model,
                i,
                row,
                a.max_tokens,
                run_id,
                a.request_timeout,
            )
            for i, row in selected
        ]
        for f in concurrent.futures.as_completed(futures):
            if f.cancelled():
                continue
            row = f.result()
            results.append(row)
            with a.output.with_suffix(".progress.jsonl").open("a") as out:
                out.write(json.dumps(row) + "\n")
            errors = sum(x["status"] != 200 for x in results)
            if (
                a.fail_fast_errors
                and errors >= a.fail_fast_errors
                and abort_reason is None
            ):
                abort_reason = (
                    f"Stopping queued requests after {errors} error(s); "
                    "preserve this attempt as diagnostic only"
                )
                print(abort_reason, flush=True)
                for pending in futures:
                    pending.cancel()
            if len(results) % 25 == 0:
                print(
                    f"{a.output.name}: {len(results)}/{len(selected)} "
                    f"errors={sum(x['status'] != 200 for x in results)}",
                    flush=True,
                )
    elapsed = time.monotonic() - start
    # A producer lease can outlive the final HTTP response. Retain all peers
    # through a full lease window so delayed expiry cannot be hidden by the
    # controller immediately restarting the next configuration.
    settle_seconds = 12
    if a.count > 8 and producer_before:
        leases = []
        for service in manifest.get("services", []):
            if service["role"] != "prefill":
                continue
            command = service["command"]
            kv = json.loads(command[command.index("--kv-transfer-config") + 1])
            leases.append(kv["kv_connector_extra_config"]["kv_lease_duration"])
        settle_seconds = max(12, max(leases) + 15)
        print(
            f"{a.output.name}: requests finished; "
            f"auditing producer leases for {settle_seconds}s",
            flush=True,
        )
    time.sleep(settle_seconds)
    after_text, after = get_metrics(metrics_url)
    producer_after = producer_snapshots(manifest)
    mooncake_after = mooncake_producer_snapshots(manifest, a.output, "after")
    a.output.with_suffix(".metrics-after.txt").write_text(after_text)
    delta = {k: after[k] - before[k] for k in before if k != "outstanding"}
    good = [r for r in results if r["status"] == 200]
    clean = (
        delta["success"] == len(good)
        and after["outstanding"] == 0
        and delta["drafts"] > 0
    )
    # The scheduler may legitimately trim the last draft at a request's
    # output/context limit, even with fixed-length verification.
    clean = clean and 0 < delta["draft_tokens"] <= a.num_spec_tokens * delta["drafts"]
    clean = clean and 0 <= delta["accepted_tokens"] <= delta["draft_tokens"]
    transfers = {
        url: {
            key: producer_after[url][key] - before_values[key]
            for key in ("success", "nixl_bytes_sent", "nixl_failed", "kv_expired")
        }
        for url, before_values in producer_before.items()
    }
    if transfers:
        clean = (
            clean
            and len(transfers) == 3
            and all(row["nixl_bytes_sent"] > 0 for row in transfers.values())
        )
        clean = clean and all(
            row["nixl_failed"] == 0 and row["kv_expired"] == 0
            for row in transfers.values()
        )
        clean = clean and sum(row["success"] for row in transfers.values()) == a.count
        clean = clean and all(
            row["outstanding"] == 0 for row in producer_after.values()
        )
    mooncake_counts = {
        role: {
            "successful_prefills": mooncake_after[role]["success"] - initial["success"],
            "outstanding_after": mooncake_after[role]["outstanding"],
        }
        for role, initial in mooncake_before.items()
    }
    if mooncake_counts:
        clean = (
            clean
            and sum(r["successful_prefills"] for r in mooncake_counts.values())
            == a.count
        )
        clean = clean and all(
            r["successful_prefills"] >= 0 and r["outstanding_after"] == 0
            for r in mooncake_counts.values()
        )
    tokens = sum(r["completion_tokens"] for r in good)
    tpot_rows = [r for r in good if r["tpot_s"] is not None]
    denominator = sum(r["completion_tokens"] - 1 for r in tpot_rows)
    report = {
        "format": "pd-speculator-comparison-v1",
        "manifest": manifest,
        "dataset_sha256": hashlib.file_digest(
            a.dataset.open("rb"), "sha256"
        ).hexdigest(),
        "count": a.count,
        "concurrency": a.concurrency,
        "max_tokens_cap": a.max_tokens,
        "seed": a.seed,
        "elapsed_s": elapsed,
        "success_count": len(good),
        "error_count": len(results) - len(good),
        "clean_counters": clean,
        "spec_counters": delta,
        "acceptance_length_including_base": 1
        + delta["accepted_tokens"] / delta["drafts"]
        if delta["drafts"]
        else None,
        "accepted_draft_tokens_per_round": delta["accepted_tokens"] / delta["drafts"]
        if delta["drafts"]
        else None,
        "verified_draft_tokens_per_round": delta["draft_tokens"] / delta["drafts"]
        if delta["drafts"]
        else None,
        "adaptive": a.adaptive,
        "ttft_s": distribution(r.get("ttft_s") for r in good),
        "tpot_s": distribution(r.get("tpot_s") for r in good),
        "token_weighted_tpot_s": sum(
            r["tpot_s"] * (r["completion_tokens"] - 1) for r in tpot_rows
        )
        / denominator
        if denominator
        else None,
        "output_tokens": tokens,
        "output_tokens_per_wall_second": tokens / elapsed,
        "prompt_tokens": distribution(r.get("prompt_tokens") for r in good),
        "results": sorted(results, key=lambda r: r["index"]),
    }
    report["producer_transfer_counters"] = transfers
    report["aborted_early_reason"] = abort_reason
    report["mooncake_producer_request_counters"] = mooncake_counts
    if mooncake_counts:
        report["mooncake_transfer_metrics_note"] = (
            "Transfer bytes/failures are in engine logs; "
            "this fork has no Mooncake Prometheus exporter."
        )
    report["metrics_url"] = metrics_url
    report["tool_schema_normalization"] = (
        "Missing/null function.parameters set to {}; "
        "identical TP/PD protocol; no rows removed."
    )
    position_re = re.compile(
        r"^vllm:spec_decode_num_accepted_tokens_per_pos_total\{([^}]*)\}\s+([\d.eE+-]+)$"
    )

    def position_counts(raw):
        counts = {}
        for line in raw.splitlines():
            match = position_re.match(line)
            if match:
                label = re.search(r'position="(\d+)"', match[1])
                if label:
                    i = int(label[1])
                    counts[i] = counts.get(i, 0) + float(match[2])
        return counts

    initial_positions = position_counts(before_text)
    final_positions = position_counts(after_text)
    report["per_position"] = [
        {
            "position": i + 1,
            "accepted_tokens": final_positions.get(i, 0) - initial_positions.get(i, 0),
            "accepted_per_round": (
                final_positions.get(i, 0) - initial_positions.get(i, 0)
            )
            / delta["drafts"]
            if delta["drafts"]
            else None,
        }
        for i in range(a.num_spec_tokens)
    ]
    report["post_request_settle_seconds"] = settle_seconds
    # v0.31 scheduler counters describe proposed drafts before the GPU-side
    # adaptive budget is trimmed. Do not label that count as actually verified.
    report["proposed_draft_tokens_per_round"] = report[
        "verified_draft_tokens_per_round"
    ]
    if a.adaptive:
        report["verified_draft_tokens_per_round"] = None
        report["adaptive_counter_note"] = (
            "Acceptance includes scheduled draft rounds even when GPU "
            "verification trims their budget to zero; draft_tokens counts "
            "pre-trim proposals, not actual verified tokens."
        )
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("results", "manifest")},
            indent=2,
        ),
        flush=True,
    )
    if not clean or len(good) != a.count:
        raise SystemExit(
            "Incomplete or contaminated benchmark; retained as diagnostic only"
        )


if __name__ == "__main__":
    main()


# Independent audit: real trained projections and real paged long contexts.
# This runs only after the small native-backend check, before full evaluation.
if __name__ == "__main__":
    import sys

    if "--output" in sys.argv:
        _output = Path(sys.argv[sys.argv.index("--output") + 1])
        if _output.name in ("0831-16-grouped.json", "0831-8-native-readiness.json"):
            _root = _output.parent.parent
            _workers = [
                json.loads(f.read_text())
                for f in (_root / "results").glob("profile-d-*.json")
            ]
            _pids = sorted(set(v["pid"] for v in _workers))
            assert len(_pids) == 4, _pids
            _checks = []
            for _pid in _pids:
                _file = _root / "audit" / f"attention-parity-{_pid}.jsonl"
                if _file.exists():
                    _checks.extend(
                        json.loads(line) for line in _file.read_text().splitlines()
                    )
            _audit = dict(
                worker_pids=_pids,
                check_count=len(_checks),
                checks=_checks,
                max_relative_l2=max((v["relative_l2"] for v in _checks), default=None),
                source="actual step1200 MLA projections and long paged contexts",
            )
            (_root / "audit/real-weights-attention-parity.json").write_text(
                json.dumps(_audit, indent=2)
            )
            assert _checks and all(
                v["finite"] and v["relative_l2"] < 0.01 for v in _checks
            ), _audit
