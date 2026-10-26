# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise production DSA PP relay methods with deterministic toy layers.

The CUDA test captures the relay on one device; it does not launch an engine
or establish MG46 output quality. Run this file directly to avoid loading the
GPU integration fixtures in tests/conftest.py.
"""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from itertools import islice
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
from test_regressions import load_function


class IntermediateTensors:
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, key):
        return self.tensors[key]


def load_methods():
    """Use the actual model methods without importing unrelated GPU kernels."""
    namespace = {
        "torch": torch,
        "islice": islice,
        "IntermediateTensors": IntermediateTensors,
        "fused_allreduce_rms_norm": lambda h, r, norm: (h + r, None),
        "_materialize_aux_hidden_state": lambda h, r, sp: h if r is None else h + r,
    }
    for name in ("forward", "_make_pp_intermediate_tensors"):
        load_function(
            "vllm/models/deepseek_v32/nvidia/model.py",
            name,
            namespace,
            "DeepseekV32Model",
        )
    return namespace


class ToyLayer:
    def __init__(self, layer_id, buffer):
        self.layer_id = layer_id
        self.buffer = buffer

    def __call__(self, positions, hidden_states, residual, attn_in):
        rows = positions.shape[0]
        # MG46's index_topk_freq=4, index_skip_topk_offset=3.
        if max(self.layer_id - 2, 0) % 4 == 0:
            indices = positions[:, None] + self.layer_id
            self.buffer[:rows].copy_(indices.expand(-1, self.buffer.shape[1]))
        contribution = self.buffer[:rows, :1].to(hidden_states.dtype) / 128
        return hidden_states + contribution, torch.zeros_like(hidden_states)


class Relay:
    def __init__(self, partition, device="cpu", auxiliary=False):
        self.namespace = load_methods()
        self.partition = partition
        self.models = []
        self.groups = []
        start = 0
        for rank, count in enumerate(partition):
            group = SimpleNamespace(
                world_size=len(partition),
                is_first_rank=rank == 0,
                is_last_rank=rank == len(partition) - 1,
            )
            buffer = torch.full((16, 8), -999, dtype=torch.int32, device=device)
            layers = [ToyLayer(i, buffer) for i in range(sum(partition))]
            model = SimpleNamespace(
                config=SimpleNamespace(hidden_size=4, index_topk=8),
                topk_indices_buffer=buffer,
                start_layer=start,
                end_layer=start + count,
                layers=layers,
                use_sequence_parallel=False,
                replicated_embed=False,
                aux_hidden_state_layers=(2, 39, 40, 77) if auxiliary else (),
                norm=None,
                embed_input_ids=lambda ids: ids[:, None].float().expand(-1, 4),
            )
            aux_keys = [x for x in model.aux_hidden_state_layers if x <= start]
            aux_base = len(aux_keys)
            model.collect_remote_aux_hidden_states = lambda it, keys=aux_keys: [
                it[f"aux_{i}"] for i in range(len(keys))
            ]
            model.pack_local_aux_hidden_states = lambda values, base=aux_base: {
                f"aux_{base + i}": value for i, value in enumerate(values)
            }
            model.forward = MethodType(self.namespace["forward"], model)
            model.factory = MethodType(
                self.namespace["_make_pp_intermediate_tensors"], model
            )
            self.models.append(model)
            self.groups.append(group)
            start += count

    def run(self, ids, positions, transfer_topk=True):
        intermediate = None
        for model, group in zip(self.models, self.groups):
            self.namespace["get_pp_group"] = lambda group=group: group
            received_aux = {}
            if intermediate is not None:
                received = model.factory(ids.shape[0], torch.float32, ids.device)
                for key, value in intermediate.tensors.items():
                    if key == "topk_indices" and not transfer_topk:
                        received.tensors[key].fill_(-999)
                    elif key in received.tensors:
                        received.tensors[key].copy_(value)
                    else:
                        received.tensors[key] = value.clone()
                intermediate = received
                received_aux = {
                    key: value
                    for key, value in received.tensors.items()
                    if key.startswith("aux_")
                }
            intermediate = model.forward(ids, positions, intermediate)
            if isinstance(intermediate, IntermediateTensors):
                # PPHandler forwards upstream features past intermediate stages.
                intermediate.tensors.update(received_aux)
        return intermediate


class PPTopKTests(unittest.TestCase):
    def test_partitions_match_unsplit_reference_across_changing_batches(self):
        for partition in [(38, 40), (40, 38), (39, 1, 38)]:
            for auxiliary in [False, True]:
                with self.subTest(partition=partition, auxiliary=auxiliary):
                    relay = Relay(partition, auxiliary=auxiliary)
                    reference = Relay((78,), auxiliary=auxiliary)
                    for rows, offset in [(7, 3000), (1, 9000), (4, 5000)]:
                        ids = torch.arange(rows) + offset
                        positions = torch.arange(rows) + offset
                        for model in relay.models:
                            model.topk_indices_buffer.fill_(-999)
                        actual = relay.run(ids, positions)
                        expected = reference.run(ids, positions)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_unaligned_boundary_requires_transferred_indices(self):
        ids = torch.arange(4)
        positions = ids + 4096
        broken = Relay((40, 38)).run(ids, positions, transfer_topk=False)
        expected = Relay((78,)).run(ids, positions)
        self.assertFalse(torch.equal(broken, expected))

    def test_factory_preserves_index_dtype_and_active_batch_rows(self):
        relay = Relay((40, 38))
        self_model = relay.models[1]
        relay.namespace["get_pp_group"] = lambda: relay.groups[1]
        tensors = self_model.factory(3, torch.bfloat16, torch.device("cpu"))
        self.assertEqual(tensors["topk_indices"].shape, (3, 8))
        self.assertEqual(tensors["topk_indices"].dtype, torch.int32)
        self.assertEqual(tensors["hidden_states"].dtype, torch.bfloat16)
        relay.namespace["get_pp_group"] = lambda: SimpleNamespace(world_size=1)
        tensors = self_model.factory(3, torch.float32, torch.device("cpu"))
        self.assertNotIn("topk_indices", tensors.tensors)

    def test_launcher_supports_40_38_without_changing_other_settings(self):
        spec = importlib.util.spec_from_file_location(
            "pp2_launch_topk", Path(__file__).with_name("launch.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        common = [
            "p",
            "--target",
            "/target",
            "--draft",
            "/draft",
            "--gpus",
            "0,1",
            "--rdma-devices",
            "mlx5_0",
        ]
        before = module.build(module.parser().parse_args(common))
        after = module.build(
            module.parser().parse_args(common + ["--pp-partition", "40,38"])
        )
        self.assertEqual(after["environment"].pop("VLLM_PP_LAYER_PARTITION"), "40,38")
        self.assertEqual(before["environment"].pop("VLLM_PP_LAYER_PARTITION"), "38,40")
        self.assertEqual(before, after)
        for invalid in ["40", "0,78", "-1,79", "1,2,75", "a,38"]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                module.build(
                    module.parser().parse_args(common + [f"--pp-partition={invalid}"])
                )

    def test_launcher_rejects_wrong_layer_total_before_accessing_gpus(self):
        spec = importlib.util.spec_from_file_location(
            "pp2_launch_total", Path(__file__).with_name("launch.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(json.dumps({"num_hidden_layers": 78}))
            args = module.parser().parse_args(
                [
                    "p",
                    "--target",
                    tmp,
                    "--draft",
                    tmp,
                    "--gpus",
                    "0,1",
                    "--rdma-devices",
                    "mlx5_0",
                    "--pp-partition",
                    "40,39",
                ]
            )
            with (
                patch.object(
                    module.parser().__class__, "parse_args", return_value=args
                ),
                patch.object(module.subprocess, "check_output") as gpu_query,
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(ValueError, "cover all target hidden layers"),
            ):
                module.main()
            gpu_query.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_cuda_graph_replay_restores_new_indices_each_batch(self):
        relay = Relay((40, 38), device="cuda")
        reference = Relay((78,), device="cuda")
        ids = torch.arange(4, device="cuda")
        positions = ids.clone() + 3000
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                relay.run(ids, positions)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = relay.run(ids, positions)
        for offset in [4000, 8000, 12000]:
            positions.copy_(ids + offset)
            relay.models[1].topk_indices_buffer.fill_(-999)
            graph.replay()
            expected = reference.run(ids, positions)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
