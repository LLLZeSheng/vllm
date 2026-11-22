# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator: semi-autoregressive parallel drafting.

DSpark drafts a block of ``num_speculative_tokens`` tokens in one parallel pass
(reusing the DFlash machinery: context-KV precompute + a query-block forward),
then injects intra-block dependency with a lightweight sequential Markov head.

Differences from DFlash:
  * Anchor-as-first-prediction: each request emits exactly ``N =
    num_speculative_tokens`` query tokens (anchor + N-1 noise), NOT ``1 + N``.
    Every query position is a prediction (the anchor predicts the first draft
    token), so we sample at all N positions and ``sample_pos = query_pos + 1``
    (standard next-token), whereas DFlash's masks sit AT the predicted position.
    This is the ``sample_from_anchor`` path in the shared prepare-inputs kernel.
    Speculators-format checkpoints instead use the DFlash ``1 + N`` fill-in
    layout (anchor is the bonus token).
  * Sequential Markov sampling: instead of DFlash's single parallel sample, we
    sample left-to-right, adding a prefix-dependent Markov bias derived from the
    previously sampled token at each step.

CUDA graphs (FULL, mirroring DFlash) cover the whole draft step: the parallel
backbone forward AND the sequential Markov sampling.
"""

from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.gqa_context_codec import (
    GQAContextCodec,
    prefill_spans,
)
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.spec_decode.dspark.gqa_context_cache import GQAContextCache

logger = init_logger(__name__)


class DSparkSpeculator(DFlashSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        self.gqa_context_codec: GQAContextCodec | None = None
        spec_config = vllm_config.speculative_config
        assert spec_config is not None
        self.gqa_context_codec_path = spec_config.dspark_gqa_context_codec
        self.gqa_context_cache: GQAContextCache | None = None
        self._gqa_wire_group: int | None = None
        self._gqa_native_groups: list[int] = []
        if self.gqa_context_codec_path is not None:
            if not vllm_config.model_config.enforce_eager:
                raise ValueError("DSpark GQA context codec requires --enforce-eager")
            transfer = vllm_config.kv_transfer_config
            if transfer is not None:
                parallel = vllm_config.parallel_config
                if (
                    transfer.kv_connector != "NixlConnector"
                    or transfer.kv_role not in ("kv_producer", "kv_consumer")
                    or vllm_config.cache_config.enable_prefix_caching
                    or parallel.pipeline_parallel_size != 1
                    or parallel.prefill_context_parallel_size != 1
                    or parallel.decode_context_parallel_size != 1
                    or vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
                ):
                    raise ValueError(
                        "GQA DSpark codec transfer requires NixlConnector, a single "
                        "producer or consumer role, hybrid KV allocation, and "
                        "prefix caching, PP and context parallelism disabled"
                    )

        # Whether to sample from the anchor position. When True, uses anchor-as-first
        # (N slots, each position predicts the next token). When False, uses 1+N
        # fill-in block (anchor is a bonus token).
        self.sample_from_anchor = getattr(
            self.draft_model_config.hf_config, "sample_from_anchor", True
        )
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_steps
        else:
            self.num_query_per_req = 1 + self.num_speculative_steps

        # DSpark consumes mean-pooled target aux hidden states at the target
        # layers, combined to hidden_size via main_proj. Store that combined
        # main_x (hidden_size wide). DSpark does not use the same pre-allocated buffer
        # that DeepSeek-V4's MTP uses.
        draft_hidden = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens, draft_hidden, dtype=self.dtype, device=device
        )

        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )

        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )

        # Reduced-vocab probabilistic drafting only; set in load_draft_model.
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None
        self._draft_topk: int | None = getattr(
            self.draft_model_config.hf_config, "dspark_draft_topk", None
        )

        self.use_confidence_head: bool = False

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        if self.context_kv_only or self.gqa_context_codec_path is not None:
            from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkModel

            if type(model.model) is not Qwen3DSparkModel:
                raise ValueError(
                    "PDCO and the context codec require the GQA DSpark backbone"
                )
        if self.context_kv_only:
            self._trim_context_only_weights(model, target_model)
        if self.gqa_context_codec_path is not None:
            self.gqa_context_codec = GQAContextCodec.from_file(
                self.gqa_context_codec_path,
                hidden_size=model.model.config.hidden_size,
                device=self.device,
            )
            if self.vllm_config.kv_transfer_config is not None:
                from vllm.v1.worker.gpu.spec_decode.dspark.gqa_context_cache import (
                    GQAContextCache,
                )

                layers = [layer.self_attn for layer in model.model.layers]
                windows = {layer.sliding_window for layer in layers}
                if len(windows) != 1 or None in windows:
                    raise ValueError("Codec transfer requires one common SWA window")
                window = windows.pop()
                assert window is not None
                self.gqa_context_cache = GQAContextCache(
                    self.vllm_config,
                    self.gqa_context_codec,
                    tuple(layer.attn.layer_name for layer in layers),
                    window,
                )
        # Reduced draft vocab: probabilistic rejection sampling indexes draft
        # logits by target id, so precompute the draft->target column map and a
        # scratch buffer to scatter logits into target vocab before sampling.
        if self.draft_logits is not None and model.draft_id_to_target_id is not None:
            d2t = model.draft_id_to_target_id
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            # -inf once; the per-step scatter overwrites the draft->target
            # columns. Kept separate from draft_logits to avoid aliasing.
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )
        self.use_confidence_head = (
            self.enable_adaptive_verification
            and model.model.confidence_head is not None
        )
        if self.use_confidence_head:
            # The acceptance estimator is not needed when a trained confidence head
            # is available.
            self.use_acceptance_estimator = False
        return model

    def _trim_context_only_weights(self, draft, target):
        model = draft.model
        if model._fused_kv_weight is None:
            raise ValueError("GQA PDCO requires a fused context KV projection")
        preserved = {id(param) for param in target.parameters()}
        preserved.update(id(param) for param in model.hidden_norm.parameters())
        if model.use_aux_hidden_state:
            preserved.update(id(param) for param in model.fc.parameters())
        removed = 0
        for param in draft.parameters():
            if id(param) not in preserved:
                removed += param.nbytes
                param.data = param.data.new_empty(0)
        logger.info(
            "DSpark PDCO prefill context only. Released %d draft weight bytes.", removed
        )

    def _prepare_prefill_context_states(
        self, states: torch.Tensor, input_batch: InputBatch
    ) -> torch.Tensor:
        codec = self.gqa_context_codec
        if codec is None or not input_batch.has_prefill:
            return states
        spans = prefill_spans(
            input_batch.query_start_loc_np,
            input_batch.prefill_len_np - input_batch.num_computed_prefill_tokens_np,
            input_batch.is_prefilling_np,
        )
        cache = self.gqa_context_cache
        if cache is None:
            return codec.round_trip_prefill(states, spans)

        if self._gqa_wire_group is None:
            group_ids = {
                layer: gid
                for gid, group in enumerate(self.kv_cache_config.kv_cache_groups)
                for layer in group.layer_names
            }
            self._gqa_wire_group = group_ids[cache.layer_name]
            self._gqa_native_groups = [
                group_ids[name] for name in cache.restored_layers
            ]

        transfer = self.vllm_config.kv_transfer_config
        assert transfer is not None
        if transfer.is_kv_producer:
            row = self.draft_kv_cache_group_ids.index(self._gqa_wire_group)
            cache.store(states, self._context_slot_mappings[row], spans)
        else:
            cache.restore(
                self.model,
                input_batch,
                self.block_tables,
                self._gqa_wire_group,
                self._gqa_native_groups,
            )
        return states

    def _sample_logits(
        self,
        logits: torch.Tensor,
        idx_map: torch.Tensor,
        sample_pos: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        self._maybe_predict_acceptance(logits, idx_map, self._step_cols[step])
        if self.draft_logits is None:
            draft_ids = logits.argmax(dim=-1)
            return self.model.map_draft_to_target(draft_ids)

        # Probabilistic sampling and rejection operate in target-vocabulary
        # space. A reduced draft vocabulary is scattered into its target rows.
        if self._d2t_scatter_index is not None:
            assert self._draft_scatter_buf is not None
            buf = self._draft_scatter_buf[: logits.shape[0]]
            buf.index_copy_(1, self._d2t_scatter_index, logits.to(buf.dtype))
            logits = buf

        # sample_pos is the predicted token's position P. Sampling keys a draw
        # by the position before the sampled token, P-1.
        sampled = gumbel_sample(
            logits,
            idx_map,
            self.temperature,
            self.seeds,
            sample_pos - 1,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=self.draft_logits,
            logits_cache_col=self._step_cols[step],
            use_fp64=self.use_fp64_gumbel,
        )
        if self.draft_watermarker is not None:
            sampled = self.draft_watermarker.sample(
                logits, sampled, idx_map, self.temperature
            )
        return sampled

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        if self._draft_topk is not None:
            self._sample_sequential_topk(num_reqs, head_hidden)
            return

        # Sequential Markov sampling over the backbone's output hidden states.
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        # Per-(req, position) head hidden, ordered (req, step).
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        # Draft-vocab logits; sampled ids are remapped to target vocab below.
        base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []

        # Anchor (bonus) token per request = the input id at query offset 0,
        # read via the precomputed persistent index (fixed buffer for capture).
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            if self.use_confidence_head:
                confidence_markov_embeds.append(markov_embed)
            bias = self.model.markov_bias(markov_embed)
            logits_i = base_logits[:, i] + bias
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if self.use_confidence_head:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                num_reqs, n_spec
            )

    def _sample_sequential_topk(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        """Apply the sequential Markov head only to top-k base-logit candidates.

        Candidate selection is done once for all draft positions. At each
        sequential step, the selected logits are corrected in place and every
        other entry is set to ``-inf``. The normal dense sampling and rejection
        paths then consume that truncated distribution unchanged.
        """
        assert self._draft_topk is not None
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_draft_logits(sample_hidden)
        base_logits = base_logits.view(num_reqs, n_spec, -1)
        base_values, draft_indices = base_logits.topk(self._draft_topk, dim=-1)
        # Reuse the dense backbone output as the normal sampler's input. Fill
        # once for all positions, then scatter only the corrected candidates
        # during the sequential loop.
        base_logits.fill_(float("-inf"))
        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            markov_embed = self.model.markov_embed(prev)
            if self.use_confidence_head:
                confidence_markov_embeds.append(markov_embed)
            logits_i = self.model.apply_markov_bias_gathered(
                markov_embed,
                base_logits[:, i],
                base_values[:, i],
                draft_indices[:, i],
            )
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if self.use_confidence_head:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                num_reqs, n_spec
            )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        # Full draft step (captured under CUDA graph): parallel backbone forward
        # then sequential Markov sampling over its hidden state outputs.
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._sample_sequential(num_reqs, head_hidden)
