---
name: optimize-triton-ascend-grouped-attention-access
description: Optimize irregular KV-cache access in Triton-Ascend grouped decode attention by vector-loading contiguous inner dimensions, assembling discrete outer rows with CANN slice extensions, and transposing UB tiles for tl.dot. Use when an agent ports grouped-query decode attention whose K or V cache addresses are indirect and naive low-axis-discrete loads degrade to scalar memory access.
---

# Optimize Triton-Ascend Grouped Attention Access

## Goal

Preserve vectorized contiguous loads while gathering KV-cache rows selected by an indirect token-location vector.

## Workflow

1. Classify each target tile by which axis is discrete and which axis is contiguous.
2. Load a discrete outer row and its contiguous head dimension as one vector.
3. Assemble BLOCK_N rows into an on-chip tensor with extension.insert_slice.
4. Transpose the assembled K tile when tl.dot requires HEAD_DIM by BLOCK_N.
5. Load V directly as BLOCK_N by VALUE_DIM when its inner value dimension is contiguous.
6. Feed the tiles into numerically stable online softmax across KV splits.

## Implementation Pattern

~~~python
import triton.language.extra.cann.extension as extension

k_rows = tl.zeros((BLOCK_N, BLOCK_DMODEL), dtype=q.dtype)
for i in range(0, BLOCK_N):
    if start_n + i < split_kv_end:
        cache_row = extension.get_element(kv_locations, (i,))
        offsets = (
            cache_row * stride_k_token
            + kv_head * stride_k_head
            + tl.arange(0, BLOCK_DMODEL)
        )
        row = tl.load(K + offsets, mask=tl.arange(0, BLOCK_DMODEL) < key_dim)
        k_rows = extension.insert_slice(
            k_rows, row[None, :], (i, 0), (1, BLOCK_DMODEL), (1, 1)
        )
k = tl.trans(k_rows, (1, 0))
scores = tl.dot(q, k.to(q.dtype))
~~~

## Ascend Guardrails

- Prefer discrete outer rows with contiguous inner-vector loads; avoid forming a tile whose innermost access itself is indirect.
- Use the extension namespace for get_element and insert_slice.
- Mask the final KV tile and padded head dimensions independently.
- Fit assembled K/KPE tiles and the attention accumulator in UB; reduce BLOCK_N or split feature dimensions when necessary.
- Preserve float32 running maxima, sums, and accumulators in the online-softmax update.
- Keep Q-head to KV-head grouping and split boundaries consistent with the original attention contract.

## Verification

Compare with a PyTorch or trusted fused-attention reference across variable sequence lengths, KV splits, grouped-head ratios, optional positional feature dimensions, and partial BLOCK_N tails.
