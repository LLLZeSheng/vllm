# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only worker instrumentation, including physical cache storage sizes."""


class MemoryAudit:
    def memory_audit(self):
        import os

        import regex as re
        import torch

        from vllm.distributed import get_pp_group

        runner = self.model_runner
        draft = runner.get_draft_model()
        target = runner.get_model()

        def storages(tensors):
            by_ptr = {}
            for tensor in tensors:
                storage = tensor.untyped_storage()
                by_ptr[(tensor.device.index, storage.data_ptr())] = storage.nbytes()
            return sum(by_ptr.values())

        target_parameters = list(target.parameters())
        target_buckets = {}
        for name, tensor in target.named_parameters():
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
            key = "layer." + match[1] if match else ".".join(name.split(".")[:2])
            target_buckets.setdefault(key, []).append(tensor)
        draft_parameters = list(draft.parameters()) if draft else []
        registered = {p.untyped_storage().data_ptr() for p in draft_parameters}
        if draft:
            registered.update(p.untyped_storage().data_ptr() for p in draft.buffers())
        context_extras = []
        if draft:
            for key in (
                "_fused_kv_weight",
                "_fused_kv_bias",
                "_k_norm_weights",
                "_rope_cos_sin_cache",
                "_hidden_norm_weight",
            ):
                tensor = getattr(draft.model, key, None)
                if tensor is not None:
                    pointer = tensor.untyped_storage().data_ptr()
                    context_extras.append(
                        dict(
                            name=key,
                            shape=list(tensor.shape),
                            dtype=str(tensor.dtype),
                            storage_bytes=tensor.untyped_storage().nbytes(),
                            shared_with_registered=pointer in registered,
                            pointer=pointer,
                        )
                    )
        target_storage = {p.untyped_storage().data_ptr() for p in target_parameters}
        shared_names = (
            [
                name
                for name, p in draft.named_parameters()
                if p.untyped_storage().data_ptr() in target_storage
            ]
            if draft
            else []
        )
        names = (
            list(draft.get_draft_kv_cache_layer_names())
            if draft is not None and hasattr(draft, "get_draft_kv_cache_layer_names")
            else sorted(
                getattr(
                    getattr(runner, "speculator", None), "draft_attn_layer_names", []
                )
            )
        )
        registry = runner.vllm_config.compilation_config.static_forward_context
        caches = []
        for name in names:
            attn = registry[name]
            tensor = attn.kv_cache
            if isinstance(tensor, list):
                tensor = tensor[0]
            caches.append(
                dict(
                    name=name,
                    shape=list(tensor.shape),
                    dtype=str(tensor.dtype),
                    logical_tensor_bytes=tensor.numel() * tensor.element_size(),
                    storage_bytes=tensor.untyped_storage().nbytes(),
                    storage_offset=tensor.storage_offset(),
                    data_pointer=tensor.data_ptr(),
                    pointer=tensor.untyped_storage().data_ptr(),
                )
            )
        unique_cache = {row["pointer"]: row["storage_bytes"] for row in caches}
        config = getattr(runner, "kv_cache_config", None)
        allocations = []
        for tensor in getattr(config, "kv_cache_tensors", []):
            allocations.append(
                dict(
                    size=tensor.size,
                    layers=tensor.layers,
                    layer_stride=tensor.layer_stride,
                    block_stride=tensor.block_stride,
                    offset=tensor.offset,
                    draft_layers=[name for name in tensor.layers if name in names],
                )
            )
        groups = []
        for group in getattr(config, "kv_cache_groups", []):
            spec = group.kv_cache_spec
            per_layer = getattr(spec, "kv_cache_specs", None)
            per_layer = per_layer or {name: spec for name in group.layer_names}
            details = []
            for name, layer_spec in per_layer.items():
                values = {
                    key: getattr(layer_spec, key, None)
                    for key in (
                        "block_size",
                        "page_size_bytes",
                        "real_page_size_bytes",
                        "sliding_window",
                        "num_kv_heads",
                        "head_size",
                        "head_size_v",
                    )
                }
                details.append(dict(name=name, **values))
            groups.append(
                dict(
                    layer_names=group.layer_names,
                    spec=repr(spec),
                    layer_specs=details,
                    aggregate_page_bytes=spec.page_size_bytes,
                    worst_case_request_pages=(
                        spec.max_memory_usage_bytes(runner.vllm_config)
                        + spec.page_size_bytes
                        - 1
                    )
                    // spec.page_size_bytes,
                    is_draft=bool(set(group.layer_names) & set(names)),
                )
            )
        free, total = torch.accelerator.get_memory_info()
        allocator = torch.accelerator.memory_stats()
        draft_attention = []
        if draft:
            for index, layer in enumerate(getattr(draft.model, "layers", [])):
                attention = getattr(layer, "self_attn", None)
                if attention is not None:
                    draft_attention.append(
                        dict(
                            layer=index,
                            sliding_window=getattr(attention, "sliding_window", None),
                            kv_heads=getattr(attention, "num_kv_heads", None),
                            head_dim=getattr(attention, "head_dim", None),
                        )
                    )
        return dict(
            draft_attention=draft_attention,
            pid=os.getpid(),
            cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
            allocator_stats={
                k: allocator.get(k)
                for k in [
                    "num_alloc_retries",
                    "num_ooms",
                    "inactive_split_bytes.all.current",
                    "inactive_split_bytes.all.peak",
                    "reserved_bytes.all.peak",
                ]
            },
            device=str(torch.accelerator.current_device_index()),
            pp_rank=get_pp_group().rank_in_group,
            worker_local_rank=getattr(self, "local_rank", None),
            data_parallel_rank=runner.vllm_config.parallel_config.data_parallel_rank,
            max_in_flight_tokens=runner.vllm_config.max_in_flight_tokens,
            torch_allocated=torch.accelerator.memory_allocated(),
            torch_reserved=torch.accelerator.memory_reserved(),
            torch_max_allocated=torch.accelerator.max_memory_allocated(),
            device_free=free,
            device_total=total,
            requested_memory=getattr(self, "requested_memory", None),
            available_kv_cache_memory_bytes=getattr(
                self, "available_kv_cache_memory_bytes", None
            ),
            peak_activation_memory=getattr(self, "peak_activation_memory", None),
            cudagraph_memory_estimate=getattr(self, "cudagraph_memory_estimate", None),
            target_parameter_storage_bytes=storages(target_parameters),
            target_parameter_bucket_bytes={
                key: storages(values) for key, values in target_buckets.items()
            },
            draft_parameter_storage_bytes=storages(draft_parameters),
            target_and_draft_unique_parameter_storage_bytes=storages(
                target_parameters + draft_parameters
            ),
            draft_parameter_names_shared_with_target=shared_names,
            draft_buffer_storage_bytes=storages(draft.buffers()) if draft else 0,
            draft_context_extra_tensors=context_extras,
            draft_parameter_names=[n for n, p in draft.named_parameters()]
            if draft
            else [],
            draft_cache_layers=caches,
            draft_unique_storage_bytes=sum(unique_cache.values()),
            physical_shared_pool_bytes=max(
                (row["size"] for row in allocations), default=0
            ),
            kv_allocations=allocations,
            kv_groups=groups,
            num_cache_blocks=getattr(config, "num_blocks", None),
            kv_cache_config=repr(getattr(runner, "kv_cache_config", None)),
        )
