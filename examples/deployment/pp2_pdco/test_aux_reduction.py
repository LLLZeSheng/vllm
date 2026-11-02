# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only behavioral tests of the proposed/installed aux capture helper."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]


class Tensor:
    def __init__(self, value):
        self.value = value

    def clone(self):
        return Tensor(self.value)

    def __add__(self, other):
        return Tensor(self.value + other.value)


def helper_namespace():
    source = ROOT / "vllm/models/deepseek_v32/nvidia/model.py"
    nodes = [
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_materialize_aux_hidden_state"
    ]
    assert nodes, "Missing installed auxiliary-state capture helper"
    state = {"world_size": 8, "calls": 0}

    def all_reduce(tensor):
        state["calls"] += 1
        tensor.value *= state["world_size"]
        return tensor

    namespace = {
        "torch": SimpleNamespace(Tensor=Tensor),
        "get_tensor_model_parallel_world_size": lambda: state["world_size"],
        "tensor_model_parallel_all_reduce": all_reduce,
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace
    )
    return namespace["_materialize_aux_hidden_state"], state


class Tests(unittest.TestCase):
    def test_tp8_materializes_global_aux_without_mutating_decoder_partial(self):
        capture, state = helper_namespace()
        partial, residual = Tensor(2), Tensor(10)
        actual = capture(partial, residual, False)
        self.assertEqual(actual.value, 26)  # SUM(eight partials) + residual.
        self.assertEqual(partial.value, 2)  # Normal decoder must reduce this later.
        self.assertEqual(residual.value, 10)
        self.assertEqual(state["calls"], 1)
        self.assertNotEqual(actual.value, partial.value + residual.value)

    def test_tp1_is_exact_original_arithmetic_and_does_not_reduce(self):
        capture, state = helper_namespace()
        state["world_size"] = 1
        self.assertEqual(capture(Tensor(2), Tensor(10), False).value, 12)
        self.assertEqual(state["calls"], 0)

    def test_sequence_parallel_does_not_reduce_token_shards_again(self):
        capture, state = helper_namespace()
        self.assertEqual(capture(Tensor(2), Tensor(10), True).value, 12)
        self.assertEqual(state["calls"], 0)

    def test_embedding_without_residual_is_returned_unchanged(self):
        capture, state = helper_namespace()
        embedding = Tensor(7)
        self.assertIs(capture(embedding, None, False), embedding)
        self.assertEqual(state["calls"], 0)


if __name__ == "__main__":
    unittest.main()
