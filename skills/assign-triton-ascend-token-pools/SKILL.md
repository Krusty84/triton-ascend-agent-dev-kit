---
name: assign-triton-ascend-token-pools
description: Implement and optimize Triton-Ascend request-to-token-pool assignment using int32 metadata, masked prefix-length reduction, and fixed-offset block loops. Use when an agent ports SGLang-style KV cache assignment to Ascend NPU, copies variable request spans from a concatenated cache, or needs to replace loop-carried vector offsets that compile poorly.
---

# Assign Triton-Ascend Token Pools

## Goal

Copy each request's contiguous cache segment into the correct row and token range of a request-to-token table.

## Workflow

1. Launch one program per request.
2. Load kv_start, kv_end, and the destination request-pool row.
3. Sum end-start for all earlier requests to find this request's source offset in the concatenated cache.
4. Use int32 metadata when all lengths, offsets, and pool indices fit.
5. Copy kv_end-kv_start elements in fixed BLOCK_SIZE chunks with derived per-iteration offsets.

## Implementation Pattern

~~~python
prefix_offsets = tl.arange(0, BS_UPPER)
prior = prefix_offsets < pid
starts = tl.load(start_ptr + prefix_offsets, mask=prior, other=0)
ends = tl.load(end_ptr + prefix_offsets, mask=prior, other=0)
source_start = tl.sum(ends - starts, axis=0)

base = tl.arange(0, BLOCK_SIZE)
num_loops = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
for i in range(num_loops):
    load_offsets = source_start + i * BLOCK_SIZE + base
    save_offsets = kv_start + i * BLOCK_SIZE + base
    mask = save_offsets < kv_end
    values = tl.load(out_cache + load_offsets, mask=mask)
    tl.store(token_pool + save_offsets, values, mask=mask)
~~~

## Ascend Guardrails

- Use save_offsets, not the initial block's offsets, when masking every loop iteration.
- Prefer i * BLOCK_SIZE + base over mutating vector offsets across iterations.
- Keep BS_UPPER a safe compile-time bound and mask entries beyond pid.
- Validate 0 <= kv_start <= kv_end <= pool_len and ensure destination request rows are unique or intentionally synchronized.
- Keep int64 when any prefix sum or flattened pool address can exceed int32.

## Verification

Compare with a CPU loop over requests. Include spans longer than BLOCK_SIZE, a partial final chunk, empty spans, nonuniform lengths, and an int32-overflow boundary.
