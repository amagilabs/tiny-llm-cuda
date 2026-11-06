# Copyright 2026 AmagiLabs
# SPDX-License-Identifier: Apache-2.0
# Part of tiny-llm-cuda, a fork of https://github.com/skyzh/tiny-llm. New file. Contains code adapted
# from upstream src/tiny_llm_ref/qwen3_week3.py and paged_kv_cache.py (Apache-2.0), applied at import time
# as a monkeypatch; see NOTICE.
"""Week-3 call-site patch: fewer custom-kernel calls per decode step, with the upstream course files left untouched.

Loaded by the same mechanism as the kernel shim: tiny_llm_cuda_kernels/ext_shim.py imports tiny_llm_cuda_kernels,
whose __init__ calls install(). A sys.meta_path hook then patches two upstream modules right after they
finish executing (they import the extension first, so they are not loaded yet at that point):

1. tiny_llm_ref.qwen3_week3.Qwen3MultiHeadAttention.__call__: q/k go through
   tiny_llm_ext_ref._qk_norm_rope4(raw, ..., weight=, eps=), RMSNorm + RoPE in one call, as week 2 does.
   Before: rms_norm(q), rms_norm(k), rope(q), rope(k) = 4 calls/layer. After: 2 (-72 per step).
   One int32 offsets array is shared by q and k (FastRoPE builds a fresh one per call).
   With qk2 on, q and k share one launch (_qk_norm_rope4_pair): 1 call/layer (-36 more per step).
2. tiny_llm_ref.paged_kv_cache.TinyKvPagedPool.write_page_slice: K and V written by one
   paged_cache_update_kv kernel that returns both pools. Before 2 calls/layer, after 1 (-36 per step).
3. tiny_llm_ref.qwen3_week3.Qwen3TransformerBlock.__call__: the residual add h = x + r and
   post_attention_layernorm(h) come from one add_rms_norm kernel that returns both h and norm(h).
   Same kernel-call count (it replaces the rms_norm call), minus one MLX add node and the norm's
   mx.contiguous/astype wrapper nodes per layer.
4. tiny_llm_ref.qwen3_week3.Qwen3ModelWeek3.__call__ (fold): the layer loop carries (h, r) instead of
   out = h + r, and the next layer's input_layernorm (or the final self.norm) gets x = h + r and norm(x)
   from one add_rms_norm kernel. Same call count again (add_rms_norm replaces each input/final rms_norm
   except layer 0's), minus one MLX add node and one norm wrapper per layer. Needs blk (it runs the patched
   block body itself) and falls back to upstream's loop per layer for anything blk would not fuse.
5. tiny_llm_ref.qwen3_week3.Qwen3MLP.__call__ (mlp, OPT-IN, not in "1"): gate and up projections + swiglu
   as one quantized_gate_up_swiglu launch (our W4A16 matvec, 2 MLX QuantizedMatmul + 1 swiglu call -> 1 call
   per layer). Not bitwise vs upstream: upstream's week 3 uses MLX's quantized_matmul, whose summation order
   ours does not reproduce (bitwise vs our own matvec + swiglu instead). "1" stays the bitwise set.

Every patched function falls back to the original upstream function for anything outside the decode fast
path, and refuses to install if the upstream source no longer matches the hash it was written against.
Switches: TL_PATCH_WEEK3=0 | 1 | comma list of qk,qk2,kv,blk,fold in the environment, or set ENABLED_QK /
ENABLED_QK2 / ENABLED_KV / ENABLED_BLOCK / ENABLED_FOLD = False at runtime
(installed, but every call goes to the upstream original), which is what scripts/cuda/check_fusion.py uses
for the patched-vs-unpatched token check. See docs/cuda/DESIGN.md sections 6-8.
"""

import hashlib
import importlib.abc
import importlib.machinery
import inspect
import os
import sys

import mlx.core as mx

# TL_PATCH_WEEK3: "1"/unset = every bitwise part, "all" = "1" + mlp, "0" = nothing, else a comma list:
#   qk  = q/k RMSNorm+RoPE via raw-input _qk_norm_rope4 (one call each)
#   qk2 = with qk, q and k in one launch (_qk_norm_rope4_pair)
#   kv  = one K+V paged cache update
#   blk = residual add + post_attention_layernorm in one kernel (add_rms_norm)
#   fold = with blk, each block's out = h + r folded into the next input_layernorm / final norm
#   mlp  = gate/up projections + swiglu in one of our kernels (not bitwise vs MLX quantized_matmul)
_ENV = os.environ.get("TL_PATCH_WEEK3", "1").strip().lower()
INSTALL = _ENV != "0"
_BITWISE = {"qk", "qk2", "kv", "blk", "fold"}
_PARTS = (_BITWISE if _ENV == "1" else _BITWISE | {"mlp"} if _ENV == "all"
          else {p.strip() for p in _ENV.split(",")})
ENABLED_QK = "qk" in _PARTS
ENABLED_QK2 = "qk2" in _PARTS
ENABLED_KV = "kv" in _PARTS
ENABLED_BLOCK = "blk" in _PARTS
ENABLED_FOLD = "fold" in _PARTS
ENABLED_MLP = "mlp" in _PARTS

# sha256[:16] of inspect.getsource() of the upstream function each patch replaces (upstream fa22518).
EXPECTED = {
    "Qwen3MultiHeadAttention.__call__": "3cda41e07ba86df0",
    "TinyKvPagedPool.write_page_slice": "5db09f52e5a974bf",
    "Qwen3TransformerBlock.__call__": "bb15e5e65329de14",
    "Qwen3ModelWeek3.__call__": "217736ead9e83418",
    "Qwen3MLP.__call__": "7f1a0146ba335df6",
}
STATUS = {}  # target -> "patched" | "skipped: <why>"
COUNTS = {"attn_fast": 0, "attn_pair": 0, "attn_fallback": 0, "kv_fused": 0, "kv_fallback": 0,
          "blk_fused": 0, "blk_fallback": 0, "fold_fused": 0, "fold_final": 0, "fold_layer_fallback": 0,
          "fold_off": 0, "mlp_fused": 0, "mlp_fallback": 0}


def _src_hash(fn):
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()[:16]


def _guard(name, fn):
    want = EXPECTED.get(name)
    got = _src_hash(fn)
    if want is not None and got != want:
        STATUS[name] = f"skipped: upstream source changed ({got} != {want})"
        print(f"[tiny-llm week3 patch] {name}: {STATUS[name]}", file=sys.stderr)
        return False
    return True


# ---------------------------------------------------------------- 1. q/k norm + rope in one call

_OFFSETS = [None, None]  # last (python int offset, int32 [1] array); arrays are immutable, so reuse is safe


def _offsets_1(offset):
    """FastRoPE's offset handling for batch 1, or None if the shape is not the B = 1 fast path."""
    if isinstance(offset, int) and not isinstance(offset, bool):
        if _OFFSETS[0] == offset:
            return _OFFSETS[1]
        arr = mx.full((1,), offset, dtype=mx.int32)
        _OFFSETS[0], _OFFSETS[1] = offset, arr
        return arr
    if isinstance(offset, list):
        return mx.array(offset, dtype=mx.int32) if len(offset) == 1 else None
    if isinstance(offset, mx.array):
        if offset.ndim == 0:
            return mx.broadcast_to(offset.astype(mx.int32), (1,))
        if offset.shape == (1,):
            return offset.astype(mx.int32)
    return None


_WEIGHTS = {}  # id(norm module) -> (weight object, dtype, cast weight)


def _norm_weight(norm, dtype):
    hit = _WEIGHTS.get(id(norm))
    if hit is not None and hit[0] is norm.weight and hit[1] == dtype:
        return hit[2]
    w = norm.weight if norm.weight.dtype == dtype else norm.weight.astype(dtype)
    _WEIGHTS[id(norm)] = (norm.weight, dtype, w)
    return w


def _patch_qwen3(mod):
    from extensions_ref import tiny_llm_ext_ref
    from .week2 import _qk_norm_rope4_pair

    cls = mod.Qwen3MultiHeadAttention
    name = "Qwen3MultiHeadAttention.__call__"
    orig = cls.__call__
    if getattr(orig, "_tl_patched", False) or not _guard(name, orig):
        return
    FastRMSNorm, FastRoPE = mod.FastRMSNorm, mod.FastRoPE
    quantized_linear, paged_attention = mod.quantized_linear, mod.paged_attention
    decode_attention_custom = mod.decode_attention_custom
    scaled_dot_product_attention = mod.scaled_dot_product_attention

    def __call__(self, x, offsets, cache, mask=None):
        B, L, _ = x.shape
        if not (ENABLED_QK and B == 1 and L == 1 and x.dtype == mx.bfloat16
                and type(self.q_norm) is FastRMSNorm and type(self.k_norm) is FastRMSNorm
                and type(self.rope) is FastRoPE and self.head_dim == 128 and self.rope.dims == 128
                and not self.rope.traditional and self.num_heads in (8, 32) and self.num_kv_heads in (8, 32)):
            COUNTS["attn_fallback"] += 1
            return orig(self, x, offsets, cache, mask)
        rope_offsets = _offsets_1(offsets)
        if rope_offsets is None:
            COUNTS["attn_fallback"] += 1
            return orig(self, x, offsets, cache, mask)
        COUNTS["attn_fast"] += 1
        projection_q = quantized_linear(x, self.wq).reshape(B, L, self.num_heads, self.head_dim)
        projection_k = quantized_linear(x, self.wk).reshape(B, L, self.num_kv_heads, self.head_dim)
        dt = projection_q.dtype
        # keep FastRMSNorm's observable dispatch counters as upstream would leave them
        self.q_norm.dispatch_counts["register_cached"] += 1
        self.k_norm.dispatch_counts["register_cached"] += 1
        projection_v = quantized_linear(x, self.wv).reshape(B, L, self.num_kv_heads, self.head_dim)
        rope = self.rope
        if ENABLED_QK2:
            COUNTS["attn_pair"] += 1
            projection_q, projection_k = _qk_norm_rope4_pair(
                projection_q, projection_k, _norm_weight(self.q_norm, dt), _norm_weight(self.k_norm, dt),
                self.q_norm.eps, self.k_norm.eps, rope_offsets, rope.dims, rope.base, rope.traditional)
        else:
            projection_q = tiny_llm_ext_ref._qk_norm_rope4(
                projection_q, rope_offsets, rope.dims, rope.base, rope.traditional,
                weight=_norm_weight(self.q_norm, dt), eps=self.q_norm.eps)
            projection_k = tiny_llm_ext_ref._qk_norm_rope4(
                projection_k, rope_offsets, rope.dims, rope.base, rope.traditional,
                weight=_norm_weight(self.k_norm, dt), eps=self.k_norm.eps)
        projection_q = projection_q.transpose(0, 2, 1, 3)
        projection_k = projection_k.transpose(0, 2, 1, 3)
        projection_v = projection_v.transpose(0, 2, 1, 3)
        # from here on, upstream verbatim
        if self.use_paged_attention:
            metadata = cache.update_and_fetch_paged(projection_k, projection_v, mask_length=L, mask=mask)
            x = paged_attention(
                projection_q,
                metadata.key_pages,
                metadata.value_pages,
                metadata.block_table,
                metadata.context_lens,
                metadata.page_size,
                scale=self.scale,
                mask=metadata.mask,
            )
        else:
            key, value, _, mask = cache.update_and_fetch(projection_k, projection_v, mask_length=L, mask=mask)
            if L <= 8 and key.shape[-2] <= 256:
                x = decode_attention_custom(projection_q, key, value, scale=self.scale, mask=mask)
            else:
                x = scaled_dot_product_attention(projection_q, key, value, scale=self.scale, mask=mask)
        x = x.transpose(0, 2, 1, 3).reshape(B, L, self.num_heads * self.head_dim)
        return quantized_linear(x, self.wo)

    __call__._tl_patched = True
    __call__._tl_orig = orig
    __call__.__doc__ = orig.__doc__
    cls.__call__ = __call__
    STATUS[name] = "patched"


# ---------------------------------------------------------------- 3. residual add + post-attention norm

def _patch_block(mod):
    from .week2 import add_rms_norm

    cls = mod.Qwen3TransformerBlock
    name = "Qwen3TransformerBlock.__call__"
    orig = cls.__call__
    if getattr(orig, "_tl_patched", False) or not _guard(name, orig):
        return
    FastRMSNorm = mod.FastRMSNorm

    def __call__(self, x, offset, cache, mask=None):
        norm = self.post_attention_layernorm
        if not (ENABLED_BLOCK and x.dtype == mx.bfloat16 and type(norm) is FastRMSNorm
                and x.ndim >= 1 and 0 < x.shape[-1] <= 4096):
            COUNTS["blk_fallback"] += 1
            return orig(self, x, offset, cache, mask)
        r = self.self_attn(self.input_layernorm(x), offset, cache, mask)
        if r.shape != x.shape or r.dtype != x.dtype:  # broadcasting add: upstream's remaining lines verbatim
            COUNTS["blk_fallback"] += 1
            h = x + r
            r = self.mlp(norm(h))
            return h + r
        COUNTS["blk_fused"] += 1
        norm.dispatch_counts["register_cached"] += 1  # what FastRMSNorm.__call__ records for dim <= 4096
        # h = x + r and norm(h) from one kernel; both bit-identical to upstream's add -> FastRMSNorm
        h, normed = add_rms_norm(x, r, _norm_weight(norm, x.dtype), norm.eps)
        r = self.mlp(normed)
        return h + r

    __call__._tl_patched = True
    __call__._tl_orig = orig
    __call__.__doc__ = orig.__doc__
    cls.__call__ = __call__
    STATUS[name] = "patched"


# ---------------------------------------------------------------- 4. residual add folded into the next norm

def _patch_model(mod):
    from .week2 import add_rms_norm

    cls = mod.Qwen3ModelWeek3
    name = "Qwen3ModelWeek3.__call__"
    orig = cls.__call__
    if getattr(orig, "_tl_patched", False) or not _guard(name, orig):
        return
    FastRMSNorm, Block, quantized_linear = mod.FastRMSNorm, mod.Qwen3TransformerBlock, mod.quantized_linear

    def _norm_ok(norm, x):
        # add_rms_norm's fast path, and the dim for which FastRMSNorm records "register_cached"
        return type(norm) is FastRMSNorm and x.dtype == mx.bfloat16 and x.ndim >= 1 and 0 < x.shape[-1] <= 4096

    def _fold(norm, h, r):
        """(h + r, norm(h + r)) from one kernel, bitwise equal to upstream's add then FastRMSNorm."""
        norm.dispatch_counts["register_cached"] += 1
        return add_rms_norm(h, r, _norm_weight(norm, h.dtype), norm.eps)

    def _block_split(blk, x, r_prev, offset, cache, mask):
        """Upstream Qwen3TransformerBlock.__call__ (hash-pinned above) without its last line `out = h + r`:
        returns (h, r), out == h + r. With r_prev, the block input x + r_prev and input_layernorm of it come
        from one add_rms_norm kernel; the post-attention add + norm is the blk patch's kernel."""
        if r_prev is None:
            n_in = blk.input_layernorm(x)
        else:
            COUNTS["fold_fused"] += 1
            x, n_in = _fold(blk.input_layernorm, x, r_prev)
        r = blk.self_attn(n_in, offset, cache, mask)
        norm = blk.post_attention_layernorm
        if r.shape != x.shape or r.dtype != x.dtype:  # broadcasting add: upstream's lines
            COUNTS["blk_fallback"] += 1
            h = x + r
            return h, blk.mlp(norm(h))
        COUNTS["blk_fused"] += 1
        h, normed = _fold(norm, x, r)
        return h, blk.mlp(normed)

    def __call__(self, inputs, offset, cache, logits_to_keep=None):
        if not (ENABLED_FOLD and ENABLED_BLOCK and getattr(Block.__call__, "_tl_patched", False)):
            COUNTS["fold_off"] += 1
            return orig(self, inputs, offset, cache, logits_to_keep)
        h = self.embedding(inputs)
        r = None  # pending residual: the hidden state upstream would hold here is h + r
        for layer in range(self.num_hidden_layers):
            blk = self.layers_inner[layer]
            if (type(blk) is Block and _norm_ok(blk.input_layernorm, h) and _norm_ok(blk.post_attention_layernorm, h)
                    and (r is None or (r.shape == h.shape and r.dtype == h.dtype))):
                h, r = _block_split(blk, h, r, offset, cache[layer], "causal")
            else:
                COUNTS["fold_layer_fallback"] += 1
                if r is not None:
                    h, r = h + r, None
                h = blk(h, offset, cache[layer], mask="causal")
        if logits_to_keep is not None:
            if logits_to_keep <= 0:
                raise ValueError("logits_to_keep must be positive")
            # slicing before the add is elementwise-identical to upstream's (h + r)[:, -k:, :]
            h = h[:, -logits_to_keep:, :]
            if r is not None:
                r = r[:, -logits_to_keep:, :]
        if r is not None and _norm_ok(self.norm, h) and r.shape == h.shape and r.dtype == h.dtype:
            COUNTS["fold_final"] += 1
            _, h = _fold(self.norm, h, r)
        else:
            if r is not None:
                h = h + r
            h = self.norm(h)
        # from here on, upstream verbatim
        if self.w_lm_head is not None:
            return quantized_linear(h, self.w_lm_head)
        else:
            return self.embedding.as_linear(h)

    __call__._tl_patched = True
    __call__._tl_orig = orig
    __call__.__doc__ = orig.__doc__
    cls.__call__ = __call__
    STATUS[name] = "patched"


# ---------------------------------------------------------------- 5. gate/up + swiglu in one kernel (opt-in)

def _patch_mlp(mod):
    from .quantized import quantized_gate_up_swiglu

    cls = mod.Qwen3MLP
    name = "Qwen3MLP.__call__"
    orig = cls.__call__
    if getattr(orig, "_tl_patched", False) or not _guard(name, orig):
        return
    quantized_linear = mod.quantized_linear

    def __call__(self, x):
        wg, wu = self.w_gate, self.w_up
        if not (ENABLED_MLP and x.ndim >= 1 and wg.bits == 4 and wu.bits == 4 and wg.group_size == 128
                and wu.group_size == 128 and wg.biases is not None and wu.biases is not None):
            COUNTS["mlp_fallback"] += 1
            return orig(self, x)
        rows = x.size // x.shape[-1] if x.shape[-1] else 0
        act = None
        if 0 < rows <= 8:
            act = quantized_gate_up_swiglu(x.reshape(rows, x.shape[-1]), wg.weight, wg.scales, wg.biases,
                                           wu.weight, wu.scales, wu.biases)
        if act is None:
            COUNTS["mlp_fallback"] += 1
            return orig(self, x)
        COUNTS["mlp_fused"] += 1
        return quantized_linear(act.reshape(*x.shape[:-1], act.shape[-1]), self.w_down)

    __call__._tl_patched = True
    __call__._tl_orig = orig
    __call__.__doc__ = orig.__doc__
    cls.__call__ = __call__
    STATUS[name] = "patched"


def _patch_qwen3_module(mod):
    errors = []
    for patch in (_patch_qwen3, _patch_block, _patch_model, _patch_mlp):
        try:
            patch(mod)
        except Exception as e:  # one failed patch must not stop the other
            errors.append(f"{patch.__name__}: {e!r}")
    if errors:
        raise RuntimeError("; ".join(errors))


# ---------------------------------------------------------------- 2. one K+V paged cache update

def _patch_paged(mod):
    from .paged import paged_cache_update_kv

    cls = mod.TinyKvPagedPool
    name = "TinyKvPagedPool.write_page_slice"
    orig = cls.write_page_slice
    if getattr(orig, "_tl_patched", False) or not _guard(name, orig):
        return

    def write_page_slice(self, page_id, start, key, value):
        if not ENABLED_KV:
            COUNTS["kv_fallback"] += 1
            return orig(self, page_id, start, key, value)
        # validation and storage growth exactly as upstream
        self.validate_page_chunk(key, value)
        if key.shape[2] > self.page_size:
            raise ValueError("Paged K/V writes cannot exceed one physical page")
        if page_id not in self.used_page_ids:
            raise ValueError(f"Page {page_id} is free")
        if page_id < 0 or page_id >= self.num_pages:
            raise ValueError(f"Page {page_id} is out of range")
        end = start + key.shape[2]
        if start < 0 or end > self.page_size:
            raise ValueError("Paged K/V write is outside page storage")
        self._ensure_page_storage(key, value)
        assert self._key_pages is not None
        assert self._value_pages is not None
        H, capacity, D = self._key_pages.shape[1:]
        assert self._value_pages.shape == self._key_pages.shape
        assert capacity == self.page_size
        assert key.shape[:2] == (1, H)
        assert key.shape[3] == D
        assert 0 <= start <= capacity
        if (self._value_pages.dtype != self._key_pages.dtype or value.shape != key.shape
                or value.dtype != key.dtype or mx.default_device() != mx.gpu):
            COUNTS["kv_fallback"] += 1
            self._key_pages = mod.tiny_llm_ext_ref.paged_cache_update(
                self._key_pages, mx.contiguous(key), page_id, start)
            self._value_pages = mod.tiny_llm_ext_ref.paged_cache_update(
                self._value_pages, mx.contiguous(value), page_id, start)
            return
        COUNTS["kv_fused"] += 1
        # cuda_kernel inputs are made row-contiguous by MLX, so no mx.contiguous nodes are needed here
        self._key_pages, self._value_pages = paged_cache_update_kv(
            self._key_pages, self._value_pages, key, value, page_id, start)

    write_page_slice._tl_patched = True
    write_page_slice._tl_orig = orig
    write_page_slice.__doc__ = orig.__doc__
    cls.write_page_slice = write_page_slice
    STATUS[name] = "patched"


# ---------------------------------------------------------------- import hook

TARGETS = {
    "tiny_llm_ref.qwen3_week3": _patch_qwen3_module,
    "tiny_llm_ref.paged_kv_cache": _patch_paged,
}


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        patch = TARGETS.get(fullname)
        if patch is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        exec_orig = spec.loader.exec_module

        def exec_module(module):
            exec_orig(module)
            _apply(fullname, module)

        spec.loader.exec_module = exec_module
        return spec


def _apply(fullname, module):
    try:
        TARGETS[fullname](module)
    except Exception as e:  # a broken patch must never break the import: upstream code still works
        STATUS[fullname] = f"skipped: {e!r}"
        print(f"[tiny-llm week3 patch] {fullname}: {STATUS[fullname]}", file=sys.stderr)


def install():
    if not INSTALL:
        STATUS["install"] = "off (TL_PATCH_WEEK3=0)"
        return
    if not any(isinstance(f, _PatchFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _PatchFinder())
    for fullname in TARGETS:
        module = sys.modules.get(fullname)
        if module is not None:
            _apply(fullname, module)


def patch_status():
    return dict(STATUS)


def source_hashes():
    """Hashes of the upstream originals (for filling EXPECTED after an upstream bump)."""
    out = {}
    for fullname, attr in (("tiny_llm_ref.qwen3_week3", ("Qwen3MultiHeadAttention", "__call__")),
                           ("tiny_llm_ref.qwen3_week3", ("Qwen3TransformerBlock", "__call__")),
                           ("tiny_llm_ref.qwen3_week3", ("Qwen3ModelWeek3", "__call__")),
                           ("tiny_llm_ref.qwen3_week3", ("Qwen3MLP", "__call__")),
                           ("tiny_llm_ref.paged_kv_cache", ("TinyKvPagedPool", "write_page_slice"))):
        fn = getattr(getattr(sys.modules[fullname], attr[0]), attr[1])
        out[".".join(attr)] = _src_hash(getattr(fn, "_tl_orig", fn))
    return out
