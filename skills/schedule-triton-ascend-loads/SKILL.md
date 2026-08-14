---
name: schedule-triton-ascend-loads
description: Reorder independent Triton-Ascend loads to expose overlap with loop-carried stores and reduce dependency stalls. Use when an agent profiles a loop where an early load waits on a previous iteration's store, a later load is independent, or assumes the compiler will automatically reorder source-level memory operations.
---

# Schedule Triton-Ascend Loads

## Goal

Issue independent memory work before a dependency-blocked load so the Ascend pipeline can overlap operations across loop iterations.

## Workflow

1. Draw read/write dependencies for every load and store in one loop iteration and across adjacent iterations.
2. Identify a load blocked by a prior store to the same or possibly aliased address.
3. Find later loads that have no dependency on that store.
4. Move only those independent loads before the blocked load.
5. Preserve calculation and store semantics, then inspect the generated schedule or profiler trace.

## Implementation Pattern

~~~python
for head in range(HEAD_NUM):
    a_ptr = A + head * HEAD_DIM + task * B_DIM + tl.arange(0, B_DIM)
    out_ptr = O + head * HEAD_DIM + task * B_DIM + tl.arange(0, B_DIM)
    index_ptr = B_index + head

    a = tl.load(a_ptr)             # Independent: issue first.
    b_index = tl.load(index_ptr)
    b_ptr = B + b_index
    b = tl.load(b_ptr)             # May wait for the prior iteration's store.

    updated_b = b + tl.sum(a)
    tl.store(out_ptr, a * updated_b)
    tl.store(b_ptr, updated_b)
~~~

## Ascend Guardrails

- Do not reorder loads across writes when pointers may alias.
- Preserve volatile, atomic, synchronization, and externally visible ordering requirements.
- Remember that the compiler preserves relevant source dependencies and may not hoist an independent later load automatically.
- Recalculate liveness and UB pressure; earlier loads stay live longer.
- Treat source reordering as a measured scheduling optimization, not a universal style rule.

## Verification

Compare outputs before and after reordering, including repeated B indices that create loop-carried dependencies. Profile issue overlap and total latency, and reject the transformation if higher UB pressure offsets the gain.
