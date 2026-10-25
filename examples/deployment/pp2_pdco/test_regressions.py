# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for PP/PD cache grouping and selected-weight loading."""

import ast
import importlib.util
import logging
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import regex as re
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]


def function(path, name, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if getattr(n, "name", "") == class_name).body
    node = next(n for n in nodes if getattr(n, "name", "") == name)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.Module(body=[future, node], type_ignores=[])
    return compile(ast.fix_missing_locations(module), path, "exec")


def load_function(path, name, namespace, class_name=None):
    exec(function(path, name, class_name), namespace)
    return namespace[name]


class RegressionTests(unittest.TestCase):
    def test_context_loader_copies_k_and_v_to_distinct_fused_slices(self):
        self.check_loader()

    def test_context_loader_rejects_incomplete_duplicate_or_bad_weights(self):
        for failure in ("missing", "duplicate", "shape", "dtype"):
            with self.subTest(failure=failure):
                self.check_loader(failure)

    def check_loader(self, failure=None):
        path = "vllm/model_executor/models/dspark_context_kv.py"
        method = load_function(
            path,
            "load_context_weights",
            {
                "re": re,
                "torch": torch,
                "logger": logging.getLogger(__name__),
            },
            "DSparkContextKVModel",
        )
        model = nn.Module()
        model.config = SimpleNamespace(
            num_key_value_heads=1, head_dim=2, num_hidden_layers=1
        )
        model.is_mla = False
        model.fc = nn.Linear(3, 3, bias=False, dtype=torch.bfloat16)
        model.hidden_norm = nn.Module()
        model.hidden_norm.weight = nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.k_norm = nn.Module()
        layer.self_attn.k_norm.weight = nn.Parameter(
            torch.zeros(2, dtype=torch.bfloat16)
        )
        model.layers = nn.ModuleList([layer])
        model._fused_kv_weight = nn.Parameter(torch.zeros(4, 3, dtype=torch.bfloat16))
        model._build_fused_kv_buffers = lambda: None
        weights = [
            ("fc.weight", torch.full((3, 3), 1, dtype=torch.bfloat16)),
            ("hidden_norm.weight", torch.full((3,), 2, dtype=torch.bfloat16)),
            (
                "layers.0.self_attn.k_norm.weight",
                torch.full((2,), 3, dtype=torch.bfloat16),
            ),
            (
                "layers.0.self_attn.k_proj.weight",
                torch.full((2, 3), 4, dtype=torch.bfloat16),
            ),
            (
                "layers.0.self_attn.v_proj.weight",
                torch.full((2, 3), 5, dtype=torch.bfloat16),
            ),
            ("lm_head.weight", torch.empty(0)),
        ]
        if failure == "missing":
            weights.pop(0)
        elif failure == "duplicate":
            weights.append(weights[0])
        elif failure == "shape":
            weights[0] = (weights[0][0], torch.zeros(1, dtype=torch.bfloat16))
        elif failure == "dtype":
            weights[0] = (weights[0][0], weights[0][1].float())
        if failure:
            with self.assertRaises(ValueError):
                method(model, weights)
        else:
            method(model, weights)
            self.assertTrue(torch.all(model._fused_kv_weight[:2] == 4))
            self.assertTrue(torch.all(model._fused_kv_weight[2:] == 5))
            self.assertTrue(torch.all(model.fc.weight == 1))
            self.assertTrue(torch.all(model.hidden_norm.weight == 2))

    def test_pdco_rejects_decode_or_combined_kv_roles(self):
        path = "vllm/model_executor/models/dspark_context_kv.py"
        check = load_function(path, "context_kv_only", {"os": os, "torch": torch})
        spec = SimpleNamespace(
            method="dspark",
            prefill_context_kv_only=True,
            draft_model_config=SimpleNamespace(
                architectures=["Qwen3DSparkModel"], dtype=torch.bfloat16
            ),
            draft_parallel_config=SimpleNamespace(tensor_parallel_size=1),
            target_parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
            ),
        )
        with patch.dict(os.environ, {"VLLM_DSPARK_PREFILL_CONTEXT_KV_ONLY": "1"}):
            for role in (None, "kv_consumer", "kv_both"):
                with self.subTest(role=role), self.assertRaises(ValueError):
                    check(
                        SimpleNamespace(
                            speculative_config=spec,
                            kv_transfer_config=SimpleNamespace(kv_role=role),
                            use_v2_model_runner=True,
                        )
                    )
            self.assertTrue(
                check(
                    SimpleNamespace(
                        speculative_config=spec,
                        kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
                        use_v2_model_runner=True,
                    )
                )
            )

    def test_pp2_and_pp4_keep_identical_swa_layer_groups_on_decode(self):
        class SlidingWindowSpec:
            page_size_bytes = 262144

        class UniformTypeKVCacheSpecs:
            @classmethod
            def from_specs(cls, specs):
                return SimpleNamespace(kv_cache_specs=specs)

        class KVCacheGroupSpec:
            def __init__(self, names, spec, is_eagle_group=False):
                self.layer_names = names
                self.kv_cache_spec = spec
                self.is_eagle_group = is_eagle_group

        namespace = dict(
            os=os,
            SlidingWindowSpec=SlidingWindowSpec,
            UniformTypeKVCacheSpecs=UniformTypeKVCacheSpecs,
            KVCacheGroupSpec=KVCacheGroupSpec,
            cdiv=lambda a, b: (a + b - 1) // b,
            logger=logging.getLogger(__name__),
            _get_per_layer_spec=lambda g, n: g.kv_cache_spec.kv_cache_specs[n],
        )
        fn = load_function(
            "vllm/v1/core/kv_cache_utils.py",
            "_cap_pp_sliding_window_group_width",
            namespace,
        )
        for pp, layers, split in [(4, 4, 2), (4, 5, 3), (2, 4, 1), (2, 5, 1)]:
            with self.subTest(pp=pp, layers=layers), patch.dict(os.environ):
                names = [f"draft.{i}" for i in range(layers)]
                swa = {n: SlidingWindowSpec() for n in names}
                size = (
                    623616
                    if pp == 4 and layers == 5
                    else 885760
                    if pp == 4
                    else 1357824
                )
                target = SimpleNamespace(page_size_bytes=size)
                full = KVCacheGroupSpec(
                    ["target"], SimpleNamespace(kv_cache_specs={"target": target})
                )
                sliding = KVCacheGroupSpec(names, SimpleNamespace(kv_cache_specs=swa))
                specs = [{"target": target} for _ in range(pp)]
                specs[-1].update(swa)
                os.environ.pop("VLLM_SWA_PD_GROUP_SPLIT", None)
                p = fn([full, sliding], specs)
                os.environ["VLLM_SWA_PD_GROUP_SPLIT"] = str(split)
                d = fn([full, sliding], [dict(target=target, **swa)])
                self.assertEqual([g.layer_names for g in p], [g.layer_names for g in d])

    def test_launcher_keeps_pdco_on_producer_and_complete_draft_on_decode(self):
        module_spec = importlib.util.spec_from_file_location(
            "pp2_launch", Path(__file__).with_name("launch.py")
        )
        launch = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(launch)
        for kind in ("gqa", "mla"):
            for role, gpus in (("p", "0,1"), ("d", "4,5,6,7")):
                with self.subTest(kind=kind, role=role):
                    args = launch.parser().parse_args(
                        [
                            role,
                            "--target",
                            "/target",
                            "--draft",
                            "/draft",
                            "--gpus",
                            gpus,
                            "--rdma-devices",
                            "mlx5_0",
                            "--draft-kind",
                            kind,
                            "--dry-run",
                        ]
                    )
                    plan = launch.build(args)
                    self.assertEqual(plan["pdco"], role == "p")
                    self.assertEqual(
                        plan["speculative_config"]["num_speculative_tokens"], 8
                    )
                    self.assertFalse(
                        plan["speculative_config"]["enable_adaptive_verification"]
                    )
                    self.assertIn("--no-enable-prefix-caching", plan["command"])
                    self.assertEqual(
                        plan["environment"]["VLLM_DSPARK_PREFILL_CONTEXT_KV_ONLY"],
                        "1" if role == "p" else "0",
                    )
                    if role == "p":
                        self.assertEqual(
                            plan["environment"]["VLLM_PP_LAYER_PARTITION"], "38,40"
                        )
                    else:
                        self.assertIn(
                            "VLLM_PP_LAYER_PARTITION", plan["unset_environment"]
                        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
