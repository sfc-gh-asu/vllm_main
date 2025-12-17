# vllm/model_executor/weight_streaming.py
# -----------------------------------------------------------------------------
# Unified decoder-weight streaming:
#   - Single-GPU CPU->GPU streaming for full decoder layers (stable pointers).
#   - Multi-GPU MoE expert AllGather into per-slot templates (EP enabled).
#   - Hybrid (Multi-GPU + CPU-resident decoder shards): overlap
#       * MoE AllGather for layer (i + K)
#       * H2D of non-MoE decoder params for layer (i + K + 1)
#
# Public entrypoint:
#   init_decoder_weight_streaming(model, device, vllm_config,
#                                num_slots=5, window_k=3, enable_logs=True)
#
# Assumptions:
#   - Decoder blocks expose `_vllm_layer_index` (standard vLLM).
#   - FusedMoE layers are nested under decoder blocks.
#   - In hybrid mode, MoE params are assembled via AllGather (templates),
#     while non-MoE params of the decoder block stream H2D into per-slot buffers.
# -----------------------------------------------------------------------------

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

import torch
import torch.distributed as dist

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_ep_group
from vllm.model_executor.layers.fused_moe.layer import FusedMoE

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Small utilities for binding tensor views back to module parameters
# -----------------------------------------------------------------------------
def _get_parent_and_leaf(module: torch.nn.Module, dotted: str):
    parent = module
    parts = dotted.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

def _rebind_param_data(module: torch.nn.Module, dotted: str, new_tensor: torch.Tensor):
    parent, leaf = _get_parent_and_leaf(module, dotted)
    if hasattr(parent, "_parameters") and leaf in parent._parameters:
        parent._parameters[leaf].data = new_tensor
    else:
        setattr(parent, leaf, new_tensor)

def _view_from_meta(storage_1d: torch.Tensor,
                    meta: Tuple[torch.dtype, int, int, Tuple[int, ...], Tuple[int, ...]]):
    # meta = (dtype, offset, numel, shape, stride)
    _dt, off, _n, shape, stride = meta
    return storage_1d.as_strided(shape, stride, off)

# -----------------------------------------------------------------------------
# Discovery helpers: decoder blocks and MoE layers & their ancestry
# -----------------------------------------------------------------------------

def _discover_decoder_layers(model: torch.nn.Module) -> List[Tuple[int, torch.nn.Module]]:
    """Find decoder blocks exposing _vllm_layer_index, sorted by index."""
    from vllm.model_executor.models.utils import PPMissingLayer
    ordered: List[Tuple[int, torch.nn.Module]] = []
    for n, m in model.named_modules():
        # logger.info(f"~~~~ vllm/model_executor/weight_streaming.py: _discover_decoder_layers: name: {n}, module: {m}")
        # Todo: the _vllm_layer_index is not a standard vLLM attribute, 
        # it is a custom attribute during the make_layers_with_weight_offloading function
        # in vllm/model_executor/models/utils.py.
        # only works with the custom make_layers_with_weight_offloading function for single GPU.
        # idx = getattr(m, "_vllm_layer_index", None)
        # if idx is not None and not isinstance(m, PPMissingLayer):
        #     ordered.append((idx, m))
        #     logger.info(f"~~~~ vllm/model_executor/weight_streaming.py _vllm_layer_index: _discover_decoder_layers: name: {n}, module: {m}, idx: {idx}")

        # So we use the length of the ordered list as the layer index for generic use.
        if not isinstance(m, PPMissingLayer) and "model.layers." in n and "DecoderLayer" in m.__class__.__name__:
            ordered.append((len(ordered), m))
            # logger.info(f"~~~~ vllm/model_executor/weight_streaming.py by hardcoded name: _discover_decoder_layers: name: {n}, module: {m}, idx: {len(ordered)}")

    ordered.sort(key=lambda x: x[0])
    return ordered

def _collect_moe_under_layer(layer: torch.nn.Module) -> List[Tuple[str, FusedMoE]]:
    """Return [(rel_name, FusedMoE)] for all FusedMoE under this decoder layer."""
    out: List[Tuple[str, FusedMoE]] = []
    for rel_name, mod in layer.named_modules():
        if isinstance(mod, FusedMoE):
            out.append((rel_name, mod))
    return out

def _map_moe_to_layer_index(model: torch.nn.Module, decoder_layers: List[Tuple[int, torch.nn.Module]]) -> Dict[FusedMoE, int]:
    """Map each FusedMoE to its nearest ancestor decoder layer index."""
    layer_by_module: Dict[torch.nn.Module, int] = {}
    for idx, lyr in decoder_layers:
        layer_by_module[lyr] = idx
        for _, child in lyr.named_modules():
            layer_by_module[child] = idx  # mark descendants with same layer idx
    mapping: Dict[FusedMoE, int] = {}
    for _, mod in model.named_modules():
        if isinstance(mod, FusedMoE) and mod in layer_by_module:
            mapping[mod] = layer_by_module[mod]
    return mapping

# -----------------------------------------------------------------------------
# CPU pack (pinned) per decoder layer (optionally excluding MoE subtrees)
# -----------------------------------------------------------------------------

def _make_pinned_flats_for_layer(layer: torch.nn.Module,
                                 include_moe_params: bool) -> tuple[
                                     Dict[torch.dtype, torch.Tensor],  # dtype -> flat(1D, CPU pinned)
                                     Dict[str, Tuple[torch.dtype, int, int, Tuple[int, ...], Tuple[int, ...]]]
                                 ]:
    """
    Build one pinned CPU flat per dtype for this decoder layer, and return
    (pinned_packs, meta) where meta allows reconstructing original views.
    Optionally exclude parameters under FusedMoE submodules (hybrid mode).
    """
    # Build 'skip' prefixes for MoE subtrees
    skip_prefixes: List[str] = []
    if not include_moe_params:
        for rel_name, _ in _collect_moe_under_layer(layer):
            if rel_name:  # non-root
                skip_prefixes.append(rel_name + ".")

    # Push all params to CPU (authoritative storage)
    for p in layer.parameters(recurse=True):
        if p.data.device.type != "cpu":
            p.data = p.data.to("cpu", non_blocking=False)

    # Deterministic order
    named = sorted(list(layer.named_parameters(recurse=True)), key=lambda kv: kv[0])

    # Group by dtype; skip embeddings/lm_head and (optionally) MoE subtrees
    by_dtype: Dict[torch.dtype, List[Tuple[str, torch.Tensor]]] = {}
    for name, t in named:
        if t.numel() == 0:
            continue
        if ".lm_head" in name or ".embed_tokens" in name:
            continue
        if not include_moe_params:
            skip = False
            for pref in skip_prefixes:
                if name.startswith(pref):
                    skip = True
                    break
            if skip:
                continue
        by_dtype.setdefault(t.dtype, []).append((name, t))

    pinned_packs: Dict[torch.dtype, torch.Tensor] = {}
    meta: Dict[str, Tuple[torch.dtype, int, int, Tuple[int, ...], Tuple[int, ...]]] = {}

    for dt, items in by_dtype.items():
        total = sum(t.numel() for _, t in items)
        if total == 0:
            continue
        flat = torch.empty(total, dtype=dt, device="cpu", pin_memory=True)
        off = 0
        for name, t in items:
            n = t.numel()
            if n == 0:
                continue
            flat.narrow(0, off, n).copy_(t.reshape(-1), non_blocking=False)
            meta[name] = (dt, off, n, tuple(t.shape), tuple(t.stride()))
            off += n
        pinned_packs[dt] = flat

    return pinned_packs, meta

def _pack_decoder_layers(decoder_layers: List[Tuple[int, torch.nn.Module]],
                         pin_memory: bool,
                         include_moe_params=True) -> Dict[int, tuple[
                             Dict[torch.dtype, torch.Tensor],
                             Dict[str, Tuple[torch.dtype, int, int, Tuple[int, ...], Tuple[int, ...]]]
                         ]]:
    """
    For each decoder layer index -> (pinned_packs, meta)
    """
    assert pin_memory, "weight streaming pack requires pin_memory=True"
    cpu: Dict[int, tuple[Dict[torch.dtype, torch.Tensor], Dict[str, Tuple]] ] = {}
    for idx, lyr in decoder_layers:
        pinned, meta = _make_pinned_flats_for_layer(lyr, include_moe_params=include_moe_params)
        # Rebind params to CPU views (authoritative host storage)
        for name, m in meta.items():
            dt = m[0]
            cpu_view = _view_from_meta(pinned[dt], m)
            _rebind_param_data(lyr, name, cpu_view)
        cpu[idx] = (pinned, meta)
    gc.collect()
    return cpu

# -----------------------------------------------------------------------------
# Single-GPU / H2D slot offloader for decoder (non-MoE or all in single-GPU)
# -----------------------------------------------------------------------------

@dataclass
class _SlotRuntime:
    active: Dict[torch.dtype, torch.Tensor]
    ev_ready: torch.cuda.Event = field(default_factory=lambda: torch.cuda.Event())
    ev_commit_barrier: torch.cuda.Event = field(default_factory=lambda: torch.cuda.Event())
    owner_layer: Optional[int] = None

class _H2DOffloader:
    def __init__(self, device: torch.device, num_slots: int, window_k: int):
        self.device = device
        self.num_slots = int(num_slots)
        self.window_k = int(window_k)
        self.copy_stream = torch.cuda.Stream()

        self.layer_indices: List[int] = []
        self.pos: Dict[int, int] = {}
        self.layer_to_slot: Dict[int, int] = {}
        self.slots: List[_SlotRuntime] = []
        self.cpu_layers: Dict[int, tuple[Dict[torch.dtype, torch.Tensor], Dict[str, Tuple]]] = {}
        logger.info(f"~~~~ vllm/model_executor/weight_streaming.py: _H2DOffloader init: num_slots: {self.num_slots}, window_k: {self.window_k}")

    def build(self, model: torch.nn.Module,
              ordered_layers: List[Tuple[int, torch.nn.Module]],
              cpu_layers: Dict[int, tuple[Dict[torch.dtype, torch.Tensor], Dict[str, Tuple]]]):

        self.layer_indices = [i for i, _ in ordered_layers]
        self.pos = {idx: k for k, idx in enumerate(self.layer_indices)}
        for idx, _ in ordered_layers:
            self.layer_to_slot[idx] = idx % self.num_slots
        self.cpu_layers = cpu_layers

        # Exemplar per slot to size actives
        exemplar: Dict[int, int] = {}
        for idx in self.layer_indices:
            s = self.layer_to_slot[idx]
            if s not in exemplar:
                exemplar[s] = idx

        for s in range(self.num_slots):
            if s not in exemplar:
                # self.slots.append(_SlotRuntime(active={}))
                slot = _SlotRuntime(active={})
                # Mark slot free for the very first prefetch. Without this, the first
                # copy would wait forever on an unrecorded barrier.
                slot.ev_commit_barrier.record(torch.cuda.current_stream())
                self.slots.append(slot)
                continue
            ex_idx = exemplar[s]
            packs, _ = self.cpu_layers[ex_idx]
            active: Dict[torch.dtype, torch.Tensor] = {}
            for dt, cpu_flat in packs.items():
                n = cpu_flat.numel()
                if n == 0:
                    continue
                active[dt] = torch.empty(n, dtype=dt, device=self.device)
            # self.slots.append(_SlotRuntime(active=active))
            slot = _SlotRuntime(active=active)
            # Mark slot free for the very first prefetch. Without this, the first
            # copy would wait forever on an unrecorded barrier.
            slot.ev_commit_barrier.record(torch.cuda.current_stream())
            self.slots.append(slot)
            logger.info("weight-stream: H2D slot %d sized from layer %d (dtypes=%s)",
                        s, ex_idx, ", ".join(str(d) for d in active.keys()) or "(none)")
        logger.info(f"~~~~ vllm/model_executor/weight_streaming.py: _H2DOffloader build: self.layer_to_slot: {self.layer_to_slot}")

        # Bind decoder params to slot device views
        for idx, mod in ordered_layers:
            sid = self.layer_to_slot[idx]
            packs, meta = self.cpu_layers[idx]
            for name, m in meta.items():
                dt = m[0]
                dev_view = _view_from_meta(self.slots[sid].active[dt], m)
                _rebind_param_data(mod, name, dev_view)

    def _next_k(self, idx: int) -> Optional[int]:
        if not self.layer_indices:
            return None
        pos = self.pos[idx]
        kpos = (pos + self.window_k) % len(self.layer_indices)
        return self.layer_indices[kpos]

    def prefetch(self, layer_idx: int):
        sid = self.layer_to_slot[layer_idx]
        packs, _ = self.cpu_layers[layer_idx]
        slot = self.slots[sid]
        with torch.cuda.stream(self.copy_stream):
            # Do not overwrite while compute still uses this slot.
            self.copy_stream.wait_event(slot.ev_commit_barrier)
            for dt, cpu_flat in packs.items():
                if cpu_flat.numel() == 0: continue
                slot.active[dt].copy_(cpu_flat, non_blocking=True)
            slot.ev_ready.record(self.copy_stream)
            slot.ev_commit_barrier.record(self.copy_stream)
        slot.owner_layer = layer_idx
    
    def pre_layer(self, layer_idx: int):
        sid = self.layer_to_slot[layer_idx]
        torch.cuda.current_stream().wait_event(self.slots[sid].ev_commit_barrier)

    def post_layer(self, layer_idx: int):
        sid = self.layer_to_slot[layer_idx]
        # Set the commit barrier event for the slot with the current stream.
        # Due to the current steam will be used for the next layer computation,
        # So we naturally guarantee the slot is ready for layer prefetch.
        # never switch compute streams, all layer kernels are enqueued onto the same torch.cuda.current_stream() 
        # (the per-thread default stream for that device). That gives the strict in-order execution.
        self.slots[sid].ev_commit_barrier.record(torch.cuda.current_stream())
        nxt = self._next_k(layer_idx)
        if nxt is not None:
            self.prefetch(nxt)

    def prime(self, first_k: int, layer_order: List[int]):
        count = min(first_k, len(layer_order))
        for j in range(count):
            self.prefetch(layer_order[j])
        seen: Set[int] = set()
        for j in range(count):
            sid = self.layer_to_slot[layer_order[j]]
            if sid in seen: continue
            seen.add(sid)
            self.slots[sid].ev_ready.synchronize()
        # Open barriers so the first pass doesn't stall
        cur = torch.cuda.current_stream()
        for sid in seen:
            self.slots[sid].ev_commit_barrier.record(cur)

# -----------------------------------------------------------------------------
# MoE gatherer (EP): per-slot templates, gather experts into template params
# -----------------------------------------------------------------------------
@dataclass
class _Signals:
    ready: torch.cuda.Event = field(default_factory=lambda: torch.cuda.Event(False))
    commit: torch.cuda.Event = field(default_factory=lambda: torch.cuda.Event(False))

class _MoEGatherer:
    def __init__(self, device: torch.device, num_slots: int, window_k: int, enable_logs: bool):
        self.device = device
        self.num_slots = int(num_slots)
        self.window_k = int(window_k)
        self.enable_logs = enable_logs

        self.gather_stream = torch.cuda.Stream()
        self.ep_pg = get_ep_group().device_group
        self.ep_size = get_ep_group().world_size

        # per-slot state
        self.slot_templates: Dict[int, FusedMoE] = {}
        self.signals: Dict[int, _Signals] = {}

        # mapping
        self.layers: List[Tuple[int, torch.nn.Module]] = []  # decoder blocks
        self.layer_to_slot: Dict[int, int] = {}
        self.moe_to_layer: Dict[FusedMoE, int] = {}

        # In HYBRID mode, we must wait for H2D readiness of the MoE source shard.
        self.h2d_offloader: Optional[_H2DOffloader] = None

    def _kinds_shapes_for_full(self, moe: FusedMoE) -> List[torch.Size]:
        shapes = []
        for flat in moe.get_expert_weights():
            shapes.append(torch.Size([moe.global_num_experts, flat.size(1)]))
        return shapes

    def _build_template(self, src: FusedMoE) -> FusedMoE:
        tmpl = FusedMoE(
            num_experts=src.global_num_experts,
            top_k=src.top_k,
            hidden_size=src.hidden_size,
            intermediate_size=src.intermediate_size_per_partition * src.tp_size,
            params_dtype=src.params_dtype,
            reduce_results=src.reduce_results,
            renormalize=src.renormalize,
            use_grouped_topk=src.use_grouped_topk,
            num_expert_group=src.num_expert_group,
            topk_group=src.topk_group,
            quant_config=src.quant_config,
            tp_size=1, ep_size=1, dp_size=1,
            prefix=f"{src.layer_name}.__slot_template__",
            custom_routing_function=src.custom_routing_function,
            scoring_func=src.scoring_func,
            routed_scaling_factor=src.routed_scaling_factor,
            e_score_correction_bias=src.e_score_correction_bias,
            apply_router_weight_on_input=src.apply_router_weight_on_input,
            activation=src.activation,
            enable_eplb=False,
            num_redundant_experts=0,
            has_bias=False,
            is_sequence_parallel=False,
            zero_expert_num=src.zero_expert_num,
            zero_expert_type=src.zero_expert_type,
        ).to(self.device)
        qm = getattr(tmpl, "quant_method", None)
        if qm is not None and hasattr(qm, "process_weights_after_loading"):
            qm.process_weights_after_loading(tmpl)
            if not hasattr(qm, "rocm_aiter_moe_enabled"):
                setattr(qm, "rocm_aiter_moe_enabled", False)
        # For old version of vllm, use ensure_moe_quant_config()
        # tmpl.ensure_moe_quant_config()
        # For v0.12.0 and newer version of vllm, use ensure_moe_quant_config_init()
        tmpl.ensure_moe_quant_config_init()
        assert tmpl.local_num_experts == tmpl.global_num_experts
        return tmpl

    def build(self, model: torch.nn.Module, decoder_layers: List[Tuple[int, torch.nn.Module]]):
        self.layers = decoder_layers
        for idx, _ in self.layers:
            self.layer_to_slot[idx] = idx % self.num_slots

        # map every FusedMoE to its decoder layer idx
        self.moe_to_layer = _map_moe_to_layer_index(model, decoder_layers)

        # exemplar per slot (first MoE found in that slot)
        exemplar_by_slot: Dict[int, FusedMoE] = {}
        for moe, lidx in self.moe_to_layer.items():
            sid = self.layer_to_slot[lidx]
            if sid not in exemplar_by_slot:
                exemplar_by_slot[sid] = moe

        # build templates and signals
        for sid, moe in exemplar_by_slot.items():
            self.slot_templates[sid] = self._build_template(moe)
            self.signals[sid] = _Signals()
            self.signals[sid].commit.record(torch.cuda.current_stream())

        logger.info(f"~~~~ vllm/model_executor/weight_streaming.py: _MoEGatherer build: MoE gatherer built {len(self.slot_templates)} templates (slots={self.num_slots})")

    def prefetch(self, layer_idx: int):
        sid = self.layer_to_slot[layer_idx]
        sig = self.signals[sid]
        # torch.cuda.current_stream().wait_event(sig.commit)
        # Make the NCCL gather stream wait for the template to be reusable
        with torch.cuda.stream(self.gather_stream):
            self.gather_stream.wait_event(sig.commit)
        self._gather_into_template(layer_idx, self.slot_templates[sid], sig)

    def _gather_into_template(self, layer_idx: int, tmpl: FusedMoE, sig: _Signals):
        # gather *all* FusedMoE layers that belong to this decoder layer into this slot template
        # (common case is 1 MoE per layer; if more, we just pick the first by build exemplar)
        # We gather using the shapes implied by tmpl.get_expert_weights()
        # using all_gather_into_tensor into contiguous [E, flat] views.
        with torch.cuda.stream(self.gather_stream):
            # Ensure H2D source bytes are present (HYBRID) before we read them
            if self.h2d_offloader is not None:
                src_sid = self.h2d_offloader.layer_to_slot[layer_idx]
                self.gather_stream.wait_event(self.h2d_offloader.slots[src_sid].ev_ready)
            # find any MoE that corresponds to this layer (we need its local shard views)
            # strategy: pick the first encountered MoE under that decoder layer
            moe_src = None
            for moe, lidx in self.moe_to_layer.items():
                if lidx == layer_idx:
                    moe_src = moe
                    break
            if moe_src is None:
                # no MoE under this layer; just mark ready
                sig.ready.record(self.gather_stream)
                return

            ep_size = self.ep_size
            local_E = moe_src.local_num_experts

            for i, dst in enumerate(tmpl.get_expert_weights()):
                assert dst.dim() == 2 and dst.shape[0] == ep_size * local_E, \
                    f"dst rows {dst.shape[0]} != ep_size*local_E {ep_size*local_E}"
                src = moe_src.get_expert_weights()[i]
                if self.h2d_offloader is not None:
                    assert src.is_cuda, "HYBRID expects MoE local shard to be on GPU (rebuilt via H2D slot)."
                if dst.is_contiguous():
                    dist.all_gather_into_tensor(dst, src, group=self.ep_pg)
                else:
                    # fallback: contiguous scratch then copy_
                    logger.info(f"~~~~ vllm/model_executor/weight_streaming.py: the MoE weight tensor is not contiguous, falling back to contiguous scratch then copy_, the performance will be degraded.")
                    scratch = torch.empty_like(dst, memory_format=torch.contiguous_format)
                    dist.all_gather_into_tensor(scratch, src, group=self.ep_pg)
                    dst.copy_(scratch, non_blocking=True)
            sig.ready.record(self.gather_stream)

    def pre_moe_forward(self, moe_mod: FusedMoE):
        """Install local compute_fn on the real FusedMoE before it runs."""
        layer_idx = self.moe_to_layer[moe_mod]
        sid = self.layer_to_slot[layer_idx]
        sig = self.signals[sid]
        tmpl = self.slot_templates[sid]

        torch.cuda.current_stream().wait_event(sig.ready)

        def compute_fn(hidden_states: torch.Tensor, router_logits: torch.Tensor):
            # For old version of vllm, use ensure_moe_quant_config()
            # tmpl.ensure_moe_quant_config()
            # For v0.12.0 and newer version of vllm, use ensure_moe_quant_config_init()
            tmpl.ensure_moe_quant_config_init()
            return tmpl.quant_method.apply(
                layer=tmpl,
                x=hidden_states,
                router_logits=router_logits,
                top_k=tmpl.top_k,
                renormalize=tmpl.renormalize,
                use_grouped_topk=tmpl.use_grouped_topk,
                global_num_experts=tmpl.global_num_experts,
                expert_map=None,
                topk_group=tmpl.topk_group,
                num_expert_group=tmpl.num_expert_group,
                custom_routing_function=tmpl.custom_routing_function,
                scoring_func=tmpl.scoring_func,
                routed_scaling_factor=tmpl.routed_scaling_factor,
                e_score_correction_bias=tmpl.e_score_correction_bias,
                activation=tmpl.activation,
                apply_router_weight_on_input=tmpl.apply_router_weight_on_input,
                enable_eplb=False,
                expert_load_view=None,
                logical_to_physical_map=None,
                logical_replica_count=None,
            )
        setattr(moe_mod, "full_moe_compute_func", compute_fn)
        setattr(moe_mod, "active_slot", True)
        setattr(moe_mod, "weight_offloading_moe", True)

    def post_moe_forward(self, moe_mod: FusedMoE):
        setattr(moe_mod, "full_moe_compute_func", None)
        setattr(moe_mod, "active_slot", False)
        # Mark the slot template reusable after the MoE forward finishes.
        # Suppose each decoder layer has only one MoE layer, 
        # so the slot template is reusable after the MoE forward finishes.
        layer_idx = self.moe_to_layer[moe_mod]
        sid = self.layer_to_slot[layer_idx]
        self.signals[sid].commit.record(torch.cuda.current_stream()) # Make the NCCL gather stream wait for the template to be reusable

    def prime(self, first_k: int, layer_order: List[int]):
        count = min(first_k, len(layer_order))
        for j in range(count):
            self.prefetch(layer_order[j])
        for j in range(count):
            sid = self.layer_to_slot[layer_order[j]]
            self.signals[sid].ready.synchronize()

# -----------------------------------------------------------------------------
# Combined Orchestrator
# -----------------------------------------------------------------------------

class DecoderWeightStreaming:
    MODE_SINGLE = "single_gpu_h2d"
    MODE_MOE_ONLY = "multi_gpu_moe_only"
    MODE_HYBRID = "multi_gpu_hybrid"

    def __init__(self, model: torch.nn.Module, device: torch.device, vllm_config: VllmConfig, 
                pin_memory: bool, num_slots: int, window_k: int, enable_logs: bool):
        self.model = model
        self.device = device
        self.vllm_config = vllm_config
        self.pin_memory = pin_memory
        self.S = int(num_slots)
        self.K = int(window_k)
        self.enable_logs = enable_logs
        
        # get all the decoder layers, which has attention and moe layers.
        self.decoder_layers: List[Tuple[int, torch.nn.Module]] = _discover_decoder_layers(model)
        self.layer_order = [i for i, _ in self.decoder_layers]
        logger.info(f"~~~~ vllm/model_executor/weight_streaming.py: layer number: {len(self.layer_order)}")
        self.layer_pos: Dict[int, int] = {i: p for p, i in enumerate(self.layer_order)}
        # logger.info(f"~~~~ vllm/model_executor/weight_streaming.py: layer_pos: {self.layer_pos}")

        # detect residency and EP
        self.cpu_src = self._is_decoder_cpu()
        logger.info(f"~~~~ vllm/model_executor/weight_streaming.py: cpu_src: {self.cpu_src}")
        self.ep_size = get_ep_group().world_size if dist.is_initialized() else 1
        logger.info(f"~~~~ vllm/model_executor/weight_streaming.py: ep_size: {self.ep_size}")

        if self.ep_size <= 1:
            self.mode = self.MODE_SINGLE
        else:
            # multi-GPU
            if self.cpu_src:
                self.mode = self.MODE_HYBRID
            else:
                self.mode = self.MODE_MOE_ONLY

        # components
        self.h2d: Optional[_H2DOffloader] = None
        self.moe: Optional[_MoEGatherer] = None

    def _is_decoder_cpu(self) -> bool:
        # check a representative decoder param device
        for _, block in self.decoder_layers:
            for _, p in block.named_parameters(recurse=True):
                # prefer a non-empty param
                if p.numel() == 0: continue
                return (p.device.type == "cpu")
        # fallback: assume CPU if nothing found
        return True

    # ---- hook glue for block (decoder layer) ----
    def _attach_block_hooks(self):
        def pre_factory(layer_idx: int):
            def _pre(mod, args, kwargs):
                if self.h2d is not None:
                    self.h2d.pre_layer(layer_idx)
                return args, kwargs
            return _pre

        def post_factory(layer_idx: int):
            def _post(mod, args, kwargs, output):
                # Mark H2D slot consumed and schedule next H2D
                if self.h2d is not None:
                    self.h2d.post_layer(layer_idx)

                # Hybrid & MoE-only scheduling:
                if self.moe is not None:
                    # Schedule MoE prefetch for i+K
                    nxt_pos = (self.layer_pos[layer_idx] + self.K) % len(self.layer_order)
                    self.moe.prefetch(self.layer_order[nxt_pos])

                # # In HYBRID: schedule H2D for i+(K+1)
                # if self.h2d is not None and self.mode == self.MODE_HYBRID:
                #     nxt_h2d_pos = (self.layer_pos[layer_idx] + self.K + 1) % len(self.layer_order)
                #     self.h2d.prefetch(self.layer_order[nxt_h2d_pos])

                return output
            return _post

        for idx, block in self.decoder_layers:
            block.register_forward_pre_hook(pre_factory(idx), with_kwargs=True)
            block.register_forward_hook(post_factory(idx), with_kwargs=True)

    # ---- hook glue for MoE (pre only in hybrid; pre+post in moe-only) ----
    def _attach_moe_hooks(self):
        if self.moe is None:
            return

        def pre_moe(mod: FusedMoE, args, kwargs):
            self.moe.pre_moe_forward(mod)
            return args, kwargs

        def post_moe(mod: FusedMoE, args, kwargs, output):
            self.moe.post_moe_forward(mod)
            return output

        for _, mod in self.model.named_modules():
            if isinstance(mod, FusedMoE):
                mod.register_forward_pre_hook(lambda m, a, kw: pre_moe(m, a, kw), with_kwargs=True)
                mod.register_forward_hook(lambda m, a, kw, out: post_moe(m, a, kw, out), with_kwargs=True)

    # ---- build components and prime ----
    def build_and_prime(self):
        logger.info(f"~~~~ vllm/model_executor/weight_streaming.py: build_and_prime: mode: {self.mode}")
        if self.mode == self.MODE_SINGLE:
            # Pack full decoder (including MoE) and H2D stream
            logger.info(f"~~~~ vllm/model_executor/weight_streaming.py: mode=SINGLE (H2D only), S={self.S}, K={self.K}")
            cpu_layers = _pack_decoder_layers(self.decoder_layers, pin_memory=self.pin_memory)
            self.h2d = _H2DOffloader(self.device, self.S, self.K)
            self.h2d.build(self.model, self.decoder_layers, cpu_layers)
            self._attach_block_hooks()
            self.h2d.prime(first_k=self.K, layer_order=self.layer_order)
            return


        if self.mode == self.MODE_MOE_ONLY:
            # MoE only; decoder already on GPU
            logger.info(f"~~~~ vllm/model_executor/weight_streaming.py: mode=MOE_ONLY (AllGather only), S={self.S}, K={self.K} (EP={self.ep_size})")
            self.moe = _MoEGatherer(self.device, self.S, self.K, self.enable_logs)
            self.moe.build(self.model, self.decoder_layers)

            self._attach_moe_hooks()
            self._attach_block_hooks()  # lightweight, only for scheduling lookahead from block boundary
            self.moe.prime(first_k=self.K, layer_order=self.layer_order)
            return

        if self.mode == self.MODE_HYBRID:
            # Hybrid: non-MoE H2D + MoE AllGather
            logger.info(f"~~~~ vllm/model_executor/weight_streaming.py: mode=HYBRID (H2D + AllGather overlap), S={self.S}, K={self.K} (EP={self.ep_size})")
            cpu_layers = _pack_decoder_layers(self.decoder_layers, pin_memory=self.pin_memory)
            self.h2d = _H2DOffloader(self.device, self.S, self.K + 1)  # H2D window = K+1
            self.h2d.build(self.model, self.decoder_layers, cpu_layers)
            self.moe = _MoEGatherer(self.device, self.S, self.K, self.enable_logs)
            self.moe.build(self.model, self.decoder_layers)
            # Wire H2D → MoE dependency (HYBRID): AllGather must wait on H2D readiness.
            self.moe.h2d_offloader = self.h2d

            # Hooks: blocks gate H2D; MoE pre hook installs local compute
            self._attach_block_hooks()
            self._attach_moe_hooks()

            # Prime both pipelines
            self.h2d.prime(first_k=self.K + 1, layer_order=self.layer_order)
            self.moe.prime(first_k=self.K, layer_order=self.layer_order)
            return

# -----------------------------------------------------------------------------
# Public entrypoint
# -----------------------------------------------------------------------------

def init_decoder_weight_streaming(model: torch.nn.Module,
                                  device: torch.device,
                                  vllm_config: VllmConfig,
                                  pin_memory: bool,
                                  num_slots: int = 5,
                                  window_k: int = 3,
                                  enable_logs: bool = True):
    """
    Call this right after the GPU Model Runner loads the model.
    Decides mode and installs hooks automatically.

    Args:
      model: self.model
      device: torch.device for this engine process
      vllm_config: runner config (only used for future knobs)
      pin_memory: bool
      num_slots: S slot count (both H2D and MoE templates)
      window_k: K lookahead for MoE; H2D uses K+1 in hybrid
      enable_logs: bool
    """
    mgr = DecoderWeightStreaming(model, device, vllm_config, pin_memory, num_slots, window_k, enable_logs)
    mgr.build_and_prime()
    gc.collect()
    torch.cuda.empty_cache()
    return mgr







# NOTE: The following notes are for the complete development of weight streaming, with all the features:
# logger format change.
# Print the forward batch size with tokens.
# skipping the DeepGEMM warmup when weight offloading is enabled.
# Simulation mode with reuse_first_layer.
# Weight offloading mode.
# Disable KV cache.

# Related files:
# adding new file: 
#   vllm/model_executor/weight_streaming.py
# Modifying the following files:
#   vllm/v1/worker/gpu_model_runner.py
#      contains 3 parts:
#          1. initialize the additional_config
#          2. logging out the scheduling tokens each forward pass
#          3. initialize the weight offloading manager

#   vllm/model_executor/model_loader/base_loader.py
#   vllm/model_executor/model_loader/utils.py
#   vllm/v1/engine/core.py
#   vllm/v1/attention/backends/flash_attn.py
#   vllm/v1/attention/backends/cpu_attn.py
#   vllm/v1/attention/backends/flash_attn.py



# NOTE: Update logger to print the complete function name and path.
# Usage: logger.info("~~~~ ...")
# In vllm/logger.py: change the _FORMAT to the following:
# _FORMAT = (
#    f"{envs.VLLM_LOGGING_PREFIX}%(levelname)s %(asctime)s "
#    "[%(pathname)s:%(lineno)d] %(message)s"
# )




# NOTE: Print the forward batch size with tokens
# In vllm/v1/worker/gpu_model_runner.py:execute_model: add the following logging before the _model_forward call.
# logger.info(f"~~~~ vllm/v1/worker/gpu_model_runner.py:execute_model: num_scheduled_tokens: {num_scheduled_tokens}.")





# NOTE: Skipping the DeepGEMM warmup when weight offloading is enabled.
# Usage: --additional-config '{"skip_deep_gemm_warmup": true}'
# In the end of the GPUModelRunner __init__ function, add the following code:
# self._additional_config = getattr(self.vllm_config, "additional_config", None)
# In vllm/model_executor/warmup/kernel_warmup.py:kernel_warmup: add the following code for skipping the DeepGEMM warmup:
# try:
#     if isinstance(worker.model_runner._additional_config, dict) and (worker.model_runner._additional_config.get("weight_offloading", False) or worker.model_runner._additional_config.get("skip_deep_gemm_warmup", False)):
#         logger.info("~~~~ vllm/model_executor/warmup/kernel_warmup.py:kernel_warmup: Skipping DeepGEMM warmup because weight offloading or skip_deep_gemm_warmup is enabled.")
#         do_deep_gemm_warmup = False
# except Exception:
#     pass




# NOTE: Adding new features of reuse_first_layer and weight offloading.
# Usage: --additional-config '{"reuse_first_layer": true}'; 
# Usage: --additional-config '{"weight_offloading": true}' for single and multi-GPU (H2D is enabled by default), or --additional-config '{"weight_offloading": true, "moe_allgather_only": true}' for multi-GPU only (H2D is disabled).
# In vllm/model_executor/models/utils.py: make_layers function:
# from vllm.config import get_current_vllm_config
# cfg = get_current_vllm_config()
# ac = getattr(cfg, "additional_config", None)
# if isinstance(ac, dict) and ac.get("reuse_first_layer", False):
#     return make_layers_with_first_layer_weights(start_layer, end_layer, num_hidden_layers, layer_fn, prefix)
# if isinstance(ac, dict) and ac.get("weight_offloading", False) and not ac.get("moe_allgather_only", False):
#     return make_layers_with_weight_offloading(start_layer, end_layer, num_hidden_layers, layer_fn, prefix)




# NOTE: Simulation mode with reuse_first_layer.
# Usage: --additional-config '{"reuse_first_layer": true}'
# In vllm/model_executor/models/utils.py, add this new function for reuse_first_layer:
# def make_layers_with_first_layer_weights(
#     start_layer: int,
#     end_layer: int,
#     num_hidden_layers: int,
#     layer_fn: LayerFn,
#     prefix: str,
# ) -> tuple[int, int, torch.nn.ModuleList]:
#     """Make a list of layers with the given layer function, taking
#     pipeline parallelism into account."""
#     def _resolve_parent_and_attr(module: torch.nn.Module, dotted_name: str) -> tuple[torch.nn.Module, str]:
#         parent = module
#         parts = dotted_name.split(".")
#         for p in parts[:-1]:
#             parent = getattr(parent, p)
#         return parent, parts[-1]
#     logger.info(f"~~~~ vllm/model_executor/models/utils.py: only keep first layer weights and reuse it")
#     # logger.info(f"start_layer: {start_layer}, end_layer: {end_layer}")
#     modules = torch.nn.ModuleList([PPMissingLayer() for _ in range(num_hidden_layers)])
#     modules[0] = maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{0}"))
#     base_param_map = dict(modules[0].named_parameters(recurse=True))
#     base_buf_map = dict(modules[0].named_buffers(recurse=True))
#     for idx in range(1, end_layer):
#         layer = maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{idx}"))
#         # Tie parameter storage
#         for name, p0 in base_param_map.items():
#             parent, attr = _resolve_parent_and_attr(layer, name)
#             # Share storage (keep distinct Parameter wrappers)
#             parent._parameters[attr].data = p0.data
#             parent._parameters[attr].requires_grad = False
#         # Tie buffers
#         for name, b0 in base_buf_map.items():
#             parent, attr = _resolve_parent_and_attr(layer, name)
#             parent._buffers[attr] = b0  # point to same tensor
#         modules[idx] = layer
#     # logger.info(f"modules: {modules}")
#     return start_layer, end_layer, modules




# NOTE: Weight offloading mode, turn on in the GPU Model Runner.
# In the end of the GPUModelRunner __init__ function, add the following code:
# self._additional_config = getattr(self.vllm_config, "additional_config", None)




# NOTE: Weight offloading mode, turn on in the GPU Model Runner.
# In the end of the GPUModelRunner load_model function, just after load model and if self.lora_config: 
# if isinstance(self._additional_config, dict) and self._additional_config.get("weight_offloading", False):
#     logger.info(f"~~~~ vllm/v1/worker/gpu_model_runner.py:load_model: weight_offloading is enabled...")
#     logger.info(f"~~~~ vllm/v1/worker/gpu_model_runner.py:load_model: self.vllm_config: {self.vllm_config}.") 
#     logger.info(f"~~~~ vllm/v1/worker/gpu_model_runner.py:load_model: self.vllm_config.parallel_config: {self.vllm_config.parallel_config}.")
#     assert self.vllm_config.parallel_config.tensor_parallel_size == 1 and self.vllm_config.parallel_config.pipeline_parallel_size == 1, "Weight offloading not supported for tensor parallel or pipeline parallel server."
#     from vllm.model_executor.weight_streaming import init_decoder_weight_streaming
#     init_decoder_weight_streaming(model=self.model,
#                 device=self.device,
#                 vllm_config=self.vllm_config,
#                 pin_memory=self.pin_memory,
#                 num_slots=5,
#                 window_k=3)




# NOTE: Weight offloading mode, with a new function for weight offloading.
# usage: --additional-config '{"weight_offloading": true}' for single and multi-GPU, or --additional-config '{"weight_offloading": true, "moe_allgather_only": true}' for multi-GPU only.
# In vllm/model_executor/models/utils.py, add this new function for weight offloading:
# def make_layers_with_weight_offloading(
#     start_layer: int,
#     end_layer: int,
#     num_hidden_layers: int,
#     layer_fn: LayerFn,
#     prefix: str,
# ) -> tuple[int, int, torch.nn.ModuleList]:
#     """Stage-1 for weight offloading:
#     - Move Parameters to CPU (non-pinned). Buffers are left untouched.
#     - Because the tensors and weights might be changed after the weight loaded from disk by postprocessing,
#         so we just move the parameters to CPU first, and then after the postprocessing, 
#         we will have a function (stage-2) to pack the finalized parameters by dtype (or a single uint8 flat tensor) and rebind the parameters to the views.
#     """
#     logger.info(f"~~~~ vllm/model_executor/models/utils.py: make_layers_with_weight_offloading from layer index: {start_layer} to {end_layer} for model: {layer_fn.__class__.__name__} to CPU...")
#     # def _tag(layer: torch.nn.Module, idx: int) -> torch.nn.Module:
#     #     setattr(layer, "_vllm_layer_index", idx)
#     #     setattr(layer, "_vllm_layer_prefix", f"{prefix}.{idx}")
#     #     return layer
#     def _move_params_to_cpu(module: torch.nn.Module) -> None:
#         for p in module.parameters(recurse=True):
#             if p.device.type != "cpu":
#                 p.data = p.data.to("cpu", non_blocking=False)
#     from tqdm import tqdm
#     real_layers: list[torch.nn.Module] = []
#     iterable = range(start_layer, end_layer)
#     if tqdm is not None:
#         iterable = tqdm(
#             iterable,
#             total=end_layer - start_layer,
#             desc=f"Building decoder layers [{prefix}]",
#             dynamic_ncols=True,
#             leave=False,
#         )
#     for idx in iterable:
#         lyr = layer_fn(prefix=f"{prefix}.{idx}")
#         # _tag(lyr, idx)
#         _move_params_to_cpu(lyr)
#         real_layers.append(lyr)
#     modules = torch.nn.ModuleList(
#         [PPMissingLayer() for _ in range(start_layer)]
#         + real_layers
#         + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)]
#     )
#     return start_layer, end_layer, modules




# NOTE: Multi-GPU weight offloading mode, compute the MoE part in each GPU individually, so inception of MoE forward is needed for each GPU.
# usage: --additional-config '{"weight_offloading": true}' for multi-GPU (H2D is enabled by default), or --additional-config '{"weight_offloading": true, "moe_allgather_only": true}' for multi-GPU only (H2D is disabled).
# In vllm/model_executor/layers/fused_moe/layer.py, find the real forward function of MoE, which is forward_impl for qwen3 moe models.
#     In the beginning of the forward_impl function, add the following code for inception of the moe serving forward:
# if getattr(self, "weight_offloading_moe", False) and getattr(self, "active_slot", False) and getattr(self, "full_moe_compute_func", None) is not None:
#     logger.info(f"~~~~ vllm/model_executor/layers/fused_moe/layer.py:FusedMoE forward_impl: weight offloading is enabled, calling function {self.full_moe_compute_func}...")
#     return self.full_moe_compute_func(hidden_states, router_logits)




# NOTE: Weight offloading mode, with whether to pin the postprocessed weights to CPU pinned memory.
# usage: --additional-config '{"weight_offloading": true}' for single and multi-GPU (H2D is enabled by default), or --additional-config '{"weight_offloading": true, "moe_allgather_only": true}' for multi-GPU only (H2D is disabled).
# 1. In vllm/model_executor/model_loader/base_loader.py: load_model function, after self.load_weights(model, model_config) and before process_weights_after_loading(model, model_config):
# By default, the postprocessed weights will be pinned to CPU pinned memory.
# But in case of weight offloading, we will check if weight_offloading is enabled and moe_allgather_only is not enabled;
# weight_offloading is enabled by setting "weight_offloading" to True in additional_config.
# For single GPU, moe_allgather_only is not enabled by default; 
# For multi-GPU, moe_allgather_only is enabled by setting "moe_allgather_only" to True in additional_config.
# If both conditions are met (H2D is enabled), we will not pin the postprocessed weights to CPU pinned memory, 
# because we will pin the postprocessed weights in the weight offloading manager later in GPU Model Runner, 
# so that the unpinned CPU memory for postprocessed weights will be freed after weight offloader initialization.
# additional_config = getattr(vllm_config, "additional_config", None)
# not_pin_postprocessed_weights_to_cpu = bool(isinstance(additional_config, dict) and additional_config.get("weight_offloading", False) and not additional_config.get("moe_allgather_only", False))
# logger.info(f"~~~~ vllm/model_executor/model_loader/base_loader.py: load_model: not_pin_postprocessed_weights_to_cpu is: {not_pin_postprocessed_weights_to_cpu}")
# process_weights_after_loading(model, model_config, target_device, not_pin_postprocessed_weights_to_cpu)

# 2. In vllm/model_executor/model_loader/utils.py: process_weights_after_loading:
#       add the new parameter not_pin_postprocessed_weights_to_cpu: bool = False,
#       in the with device_loading_context(module, target_device, not_pin_postprocessed_weights_to_cpu) block,
#       pass the not_pin_postprocessed_weights_to_cpu to the device_loading_context function.
#       and in the device_loading_context function, if not_pin_postprocessed_weights_to_cpu is True, set the pin_memory to False.
# if not_pin_postprocessed_weights_to_cpu:
#     pin_memory = False




# NOTE: Disable KV cache.
# Usage: --additional-config '{"disable_kv_cache": true}'
# 1. Initialize the additional_config in the vllm/v1/engine/core.py:__init__ function.
# In the vllm/v1/engine/core.py:__init__ function, 
#       before num_gpu_blocks, num_cpu_blocks, kv_cache_config = self._initialize_kv_caches(vllm_config), 
#       add the following code for disabling the KV cache:
# self._additional_config = getattr(self.vllm_config, "additional_config", None)

# 2. Setting the available_gpu_memory to 1 TB hardcoded when disable_kv_cache is enabled, for scheduling.
# In the vllm/v1/engine/core.py:_initialize_kv_caches function, 
#       after assert len(kv_cache_specs) == len(available_gpu_memory)
#       and before kv_cache_configs = get_kv_cache_configs(vllm_config, kv_cache_specs, available_gpu_memory),
#       add the following code for disabling the KV cache:
# if isinstance(self._additional_config, dict) and self._additional_config.get("disable_kv_cache", False):
#     available_gpu_memory = [1024 * 1024 * 1024 * 1024] * len(kv_cache_specs)
#     self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
#     logger.info(f"~~~~ vllm/v1/engine/core.py:_initialize_kv_caches: disable_kv_cache is enabled, so setting available_gpu_memory to 1 TB hardcoded...")

# 3. Skip the KV cache tensor initialization when disable_kv_cache is enabled.
# After setting the available_gpu_memory to 1 TB hardcoded only is not enough, 
# Cause then self.model_executor.initialize_from_config(kv_cache_configs) function will initiate the KV cache tensors,
# So we need to skip the KV cache tensor initialization when disable_kv_cache is enabled.
# In the vllm/v1/worker/gpu_model_runner.py:initialize_kv_cache_tensors function, 
#       update the following code by adding a new condition branch for disabling the KV cache tensor initialization:
# if isinstance(self._additional_config, dict) and self._additional_config.get("disable_kv_cache", False):
#     kv_caches: dict[str, torch.Tensor] = {}
#     for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
#         tensor = torch.zeros((2, 1), dtype=torch.bfloat16, device=self.device)
#         for layer_name in kv_cache_tensor.shared_by:
#             kv_caches[layer_name] = tensor
#     logger.info(f"~~~~ vllm/v1/worker/gpu_model_runner.py:initialize_kv_cache_tensors: len(kv_caches) {len(kv_caches)}")
# else:
#     # original kv cache tensor initialization code

# 4. Turn off the chunked-prefill when disable_kv_cache is enabled.
# In the vllm/v1/engine/core.py:__init__ function, 
#       after if len(kv_cache_config.kv_cache_groups) == 0, means no KV cache is needed, so turning off the chunked-prefill,
#       So we follow the same logic to turn off the chunked-prefill manually, by adding the following code:
# if isinstance(self._additional_config, dict) and self._additional_config.get("disable_kv_cache", False):
#     vllm_config.scheduler_config.enable_chunked_prefill = False
#     logger.info(f"~~~~ vllm/v1/engine/core.py:__init__: disable_kv_cache is enabled, turning off the chunked-prefill.")

# 5. Redirect the attention computation backend API to use reuse the encoding kernel,
#       So that we can skip the KV cache write and read operations, 
#       and directly use the Causal Attention kernel from encoding kernel.
# In the vllm/v1/attention/backends/flash_attn.py:FlashAttentionImpl:__init__ function end,
#       add the following code for initialization the _additional_config:
# self._additional_config = getattr(get_current_vllm_config(), "additional_config", None)
# In the vllm/v1/attention/backends/flash_attn.py:FlashAttentionImpl:forward function,
#       after "if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER)" condition branch function,
#       add a new disable KV cache condition branch,
#       for redirecting the attention computation backend API to reuse the encoding kernel:
# if isinstance(self._additional_config, dict) and self._additional_config.get("disable_kv_cache", False):
#     # Disable the KV cache,
#     # Directly use the causal attention kernel from encoding kernel.
#     return self._forward_prefill_only_attention(
#         query[:num_actual_tokens],
#         key[:num_actual_tokens],
#         value[:num_actual_tokens],
#         output[:num_actual_tokens],
#         attn_metadata, 
#         layer,
#     )
# In the vllm/v1/attention/backends/flash_attn.py, add the new function implementation:
# def _forward_prefill_only_attention(
#     self,
#     query: torch.Tensor,
#     key: torch.Tensor,
#     value: torch.Tensor,
#     output: torch.Tensor,
#     attn_metadata: FlashAttentionMetadata,
#     layer: torch.nn.Module,
# ) -> torch.Tensor:
#     """Forward pass for prefill-only attention.
#     Args:
#         query: shape = [num_prefill_tokens, num_heads, head_size]
#         key: shape = [num_prefill_tokens, num_kv_heads, head_size]
#         value: shape = [num_prefill_tokens, num_kv_heads, head_size]
#         output: shape = [num_prefill_tokens, num_heads, head_size]
#         attn_metadata: Prefill-only attention metadata
#         layer: The attention layer
#     """
#     if self.kv_cache_dtype.startswith("fp8"):
#         dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
#             self.kv_cache_dtype)
#         key = key.view(dtype)
#         value = value.view(dtype)
#     cu_seqlens_q = attn_metadata.query_start_loc
#     cu_seqlens_k = attn_metadata.query_start_loc
#     max_seqlen_q = attn_metadata.max_query_len
#     max_seqlen_k = attn_metadata.max_query_len
#     descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)
#     flash_attn_varlen_func(
#         q=query,
#         k=key,
#         v=value,
#         out=output,
#         cu_seqlens_q=cu_seqlens_q,
#         cu_seqlens_k=cu_seqlens_k,
#         max_seqlen_q=max_seqlen_q,
#         max_seqlen_k=max_seqlen_k,
#         softmax_scale=self.scale,
#         causal=True,                         
#         alibi_slopes=self.alibi_slopes,
#         window_size=self.sliding_window,
#         softcap=self.logits_soft_cap,
#         fa_version=self.vllm_flash_attn_version,
#         q_descale=layer._q_scale.expand(descale_shape),
#         k_descale=layer._k_scale.expand(descale_shape),
#         v_descale=layer._v_scale.expand(descale_shape),
#     )
#     return output