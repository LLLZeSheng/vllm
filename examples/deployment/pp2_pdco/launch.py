# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build or launch one PP2 PDCO producer or one DP4 decode service."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("role", choices=("p", "d"))
    p.add_argument("--target", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--gpus", required=True, help="Two P GPUs or four D GPUs")
    p.add_argument("--served-model-name", default="dspark-alaya")
    p.add_argument("--port", type=int)
    p.add_argument("--bootstrap-port", type=int, default=8999)
    p.add_argument("--rdma-devices", required=True)
    p.add_argument("--spec-tokens", type=int, choices=(5, 7, 8), default=8)
    p.add_argument("--draft-kind", choices=("gqa", "mla"), default="gqa")
    p.add_argument("--max-model-len", type=int, default=524288)
    p.add_argument("--batch-tokens", type=int)
    p.add_argument("--max-num-seqs", type=int)
    p.add_argument("--memory-utilization", type=float)
    p.add_argument("--full-prefill-draft", action="store_true")
    p.add_argument("--memory-audit", action="store_true")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--dry-run", action="store_true")
    return p


def build(args):
    producer = args.role == "p"
    devices = [int(x) for x in args.gpus.split(",")]
    if len(devices) != (2 if producer else 4) or len(set(devices)) != len(devices):
        raise ValueError("PP2 requires two distinct P GPUs; DP4 requires four D GPUs")
    if min(devices) < 0 or not args.rdma_devices.strip():
        raise ValueError("GPU indices and existing RDMA devices must be supplied")
    utilization = args.memory_utilization
    if utilization is None:
        utilization = 0.92 if producer else 0.94
    if not 0 < utilization <= 1:
        raise ValueError("Memory utilization must be in (0, 1]")
    batch = args.batch_tokens or (16384 if producer else 128)
    seqs = args.max_num_seqs or (8 if producer else 32)
    if min(batch, seqs, args.max_model_len) < 1:
        raise ValueError("Context and batch limits must be positive")
    pdco = producer and not args.full_prefill_draft
    spec = {
        "method": "dspark",
        "model": args.draft,
        "num_speculative_tokens": args.spec_tokens,
        "draft_tensor_parallel_size": 1,
        "draft_sample_method": "probabilistic",
        "rejection_sample_method": "block" if args.draft_kind == "gqa" else "standard",
        "enable_adaptive_verification": False,
        "attention_backend": "FLASH_ATTN",
        "kv_cache_dtype": "fp8" if args.draft_kind == "gqa" else "bfloat16",
    }
    if args.draft_kind == "mla":
        spec["prefill_context_kv_only"] = pdco
    connector = {
        "kv_connector": "MooncakeConnector",
        "kv_role": "kv_producer" if producer else "kv_consumer",
        "kv_connector_extra_config": {
            "mooncake_protocol": "rdma",
            "device_name": args.rdma_devices,
        },
    }
    overrides = {
        "CUDA_VISIBLE_DEVICES": args.gpus,
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_KV_CACHE_LAYOUT": "BLHNC",
        "VLLM_MOONCAKE_BOOTSTRAP_PORT": str(args.bootstrap_port),
        "VLLM_DSPARK_PREFILL_CONTEXT_KV_ONLY": "1" if pdco else "0",
    }
    if producer:
        overrides["VLLM_PP_LAYER_PARTITION"] = "38,40"
        remove = ["VLLM_SWA_PD_GROUP_SPLIT"]
    else:
        overrides["VLLM_SWA_PD_GROUP_SPLIT"] = "1"
        remove = ["VLLM_PP_LAYER_PARTITION"]
    # Do not inherit the experimental PP4 in-flight override in this profile.
    remove.append("VLLM_SWA_PREFILL_INFLIGHT_BATCHES")
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.target,
        "--host",
        "0.0.0.0",
        "--port",
        str(args.port or (8111 if producer else 8112)),
        "--served-model-name",
        args.served_model_name,
        "--tensor-parallel-size",
        "1",
        "--distributed-executor-backend",
        "mp",
        "--block-size",
        "64",
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(utilization),
        "--max-num-batched-tokens",
        str(batch),
        "--max-num-seqs",
        str(seqs),
        "--no-enable-prefix-caching",
        "--kv-cache-dtype",
        "fp8",
        "--speculative-config",
        json.dumps(spec),
        "--kv-transfer-config",
        json.dumps(connector),
        "--kernel-config",
        '{"enable_flashinfer_autotune":false}',
        "--reasoning-parser",
        "glm47",
        "--tool-call-parser",
        "glm47",
        "--enable-auto-tool-choice",
        "--trust-remote-code",
        "--safetensors-load-strategy",
        "lazy",
    ]
    if args.draft_kind == "mla":
        cmd += [
            "--enforce-eager",
            "--attention-config",
            json.dumps(
                {
                    "backend": "FLASHMLA_SPARSE",
                    "mla_prefill_backend": "FLASH_ATTN",
                }
            ),
        ]
    else:
        cmd += ["--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}']
    if args.memory_audit:
        cmd += [
            "--worker-extension-cls",
            "examples.deployment.pp2_pdco.memory_audit.MemoryAudit",
        ]
    if producer:
        cmd += ["--pipeline-parallel-size", "2"]
    else:
        cmd += [
            "--data-parallel-size",
            "4",
            "--data-parallel-size-local",
            "4",
            "--data-parallel-backend",
            "mp",
            "--enable-expert-parallel",
        ]
    return dict(
        command=cmd,
        environment=overrides,
        unset_environment=remove,
        physical_gpus=devices,
        pdco=pdco,
        speculative_config=spec,
    )


def main():
    args = parser().parse_args()
    plan = build(args)
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    for root in (args.target, args.draft):
        if not Path(root, "config.json").is_file():
            raise FileNotFoundError(f"Missing model config: {root}")
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    used = {int(a): int(b) for a, b in (row.split(",") for row in query.splitlines())}
    if any(used[i] >= 100 for i in plan["physical_gpus"]):
        raise RuntimeError("Assigned GPUs are occupied; this launcher never stops them")
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(plan, indent=2) + "\n")
    env = dict(os.environ)
    for key in plan["unset_environment"]:
        env.pop(key, None)
    env.update(plan["environment"])
    repo = Path(__file__).resolve().parents[3]
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    os.execvpe(sys.executable, plan["command"], env)


if __name__ == "__main__":
    main()
