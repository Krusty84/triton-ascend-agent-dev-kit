---
name: build-triton-ascend-fused-attention
description: Build and adapt a FlashAttention-v2-style fused forward attention kernel for Triton-Ascend using tiled Q/K/V block pointers, online softmax, causal staging, and float32 accumulation. Use when an agent needs forward scaled dot-product attention in BNSD layout on Ascend NPU, must support causal or non-causal masking, or must handle large head dimensions without overflowing on-chip memory.
---

# Build Triton-Ascend Fused Attention

## Goal

Compute softmax(QKᵀ * scale)V without materializing the full attention matrix. Support BNSD tensors shaped Z by H by N_CTX by HEAD_DIM.

## Workflow

1. Validate matching Q, K, and V shapes, compatible strides, supported dtype, and HEAD_DIM in {16, 32, 64, 128, 256}.
2. Choose BLOCK_M for query rows and BLOCK_N for key/value rows. Require N_CTX to be divisible by both in the compact implementation.
3. Map each task to a batch, head, and query-block tuple and create block pointers for Q, K, V, and output.
4. Load one Q block and stream K/V blocks through an online-softmax inner loop.
5. For causal attention, process preceding blocks and the diagonal block as separate stages; apply the triangular mask only to the diagonal stage.
6. Normalize the accumulated output by the running denominator and store it in the output dtype.

## Implementation Pattern

For every query row, initialize m_i=-inf, l_i=1, and a float32 accumulator. For each K/V tile:

~~~text
scores = dot(q, transpose(k)) * scale
scores = scores + causal_mask_if_diagonal
m_new = max(m_i, row_max(scores))
p = exp(scores - m_new)
alpha = exp(m_i - m_new)
l_i = l_i * alpha + row_sum(p)
acc = acc * alpha + dot(cast(p, k.dtype), v)
m_i = m_new
output = acc / l_i
~~~

Use a stage bitmask compatible with STAGE=3 for causal attention and STAGE=1 for non-causal attention. Advance K and V block pointers by BLOCK_N after each iteration.

## Ascend Guardrails

- Compute base offsets from each tensor's own strides; do not assume Q strides also describe K, V, or output.
- Use tl.make_block_ptr with shape, strides, offsets, block_shape, and order consistent with BNSD layout.
- Keep score maxima, denominators, and output accumulators in float32.
- For HEAD_DIM=256, use a global float32 scratch accumulator and CANN extract_slice/insert_slice updates when a full local accumulator exceeds UB capacity.
- Treat the example core count of 20 as target-specific; confirm the available NPU execution resources before fixing a persistent grid size.
- Import triton.language.extra.cann.extension only when the large-head scratch path needs it.
- Expose this as forward-only unless a real backward kernel is implemented.

## Verification

Compare with torch_npu.npu_fusion_attention using input_layout="BNSD", identical scale, and a matching causal mask. Test both causal modes, supported head dimensions, multiple BLOCK_M/BLOCK_N pairs, float16 and bfloat16, and long contexts. Start with atol=rtol=1e-2 and tighten when evidence supports it.
