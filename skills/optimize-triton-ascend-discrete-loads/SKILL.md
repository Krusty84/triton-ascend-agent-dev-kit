---
name: optimize-triton-ascend-discrete-loads
description: Optimize bounded discrete reads in Triton-Ascend by staging a contiguous source vector in UB and selecting elements with tl.gather. Use when an agent implements out = x[idx], ports arbitrary global-memory gathers from GPU, or sees scalarized indirect loads and the complete source domain is small enough to fit on chip.
---

# Optimize Triton-Ascend Discrete Loads

## Goal

Replace repeated indirect global reads with one contiguous global-to-UB transfer followed by on-chip gather.

## Workflow

1. Confirm the source domain length M is bounded and fits the UB budget.
2. Load x[0:M] contiguously into an on-chip vector.
3. Load the N indices and validate both lower and upper bounds.
4. Replace invalid indices with a safe value before tl.gather.
5. Gather along the source axis and store only valid outputs.

## Implementation Pattern

~~~python
source_offsets = tl.arange(0, M)
source = tl.load(x_ptr + source_offsets * stride_x)

out_offsets = tl.arange(0, N)
indices = tl.load(index_ptr + out_offsets * stride_index)
valid = (indices >= 0) & (indices < M)
safe_indices = tl.where(valid, indices, 0)
values = tl.gather(source, safe_indices, axis=0)
tl.store(out_ptr + out_offsets * stride_out, values, mask=valid)
~~~

## Ascend Guardrails

- Stage the whole source only when M plus indices, outputs, padding, and temporaries fit UB.
- Reject invalid indices before launch when the API requires every output element to be defined.
- Keep indices integer and ensure tl.gather supports their dtype.
- Preserve explicit source/index/output strides.
- For large M, partition by index ranges or choose a different algorithm; loading the entire source per program can cost more than indirect access.
- Avoid launching multiple identical programs when the kernel has no program-dependent offsets.

## Verification

Compare with x[idx] for permutations, repeats, sorted and random indices. Add negative and M-or-larger index tests, and benchmark across M/N ratios to identify when UB staging is beneficial.
