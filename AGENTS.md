# Triton-Ascend Development Instructions

These instructions apply to all work in this repository and involve writing triton-ascend code to run on the ASCEND NPU.

## Discard the GPU/CUDA mental model

- Forget GPU and CUDA-specific implementation experience when making Triton-Ascend decisions. Treat assumptions derived from CUDA, NVIDIA/AMD GPUs, SMs, warps, shared memory, GPU scheduling, or GPU launch geometry as invalid unless the Triton-Ascend references explicitly establish an equivalent behavior.
- Do not mechanically port GPU Triton code. Preserve only the requested mathematical operation and public contract; derive core selection, grid, work ownership, tiling, memory use, alignment, synchronization, and numeric behavior again from the Triton-Ascend documentation.
- When the input is GPU-oriented code, read [GPU-to-NPU migration](triton-ascend-dev-doc/09-gpu-to-ascend-npu-migration.md) before proposing or editing the implementation.
- Never introduce CUDA imports, CUDA runtime calls, CUDA device names, or CUDA-specific tuning assumptions into Triton-Ascend code unless the user explicitly requests interoperability and the local documentation supports it.

## Authoritative references

- For every Triton-Ascend task, read [the documentation index](triton-ascend-dev-doc.md) first and then load only the files from [triton-ascend-dev-doc/](triton-ascend-dev-doc/) selected by its **Agent loading rules**.
- Treat `triton-ascend-dev-doc.md` and `triton-ascend-dev-doc/` as authoritative for Triton-Ascend behavior. Examples illustrate patterns but do not override explicit constraints in the documentation.
- Do not infer an API signature, supported dtype, hardware capability, memory rule, launch rule, or compiler option. Confirm it in the relevant reference. If it is not documented, ask the user for clarification or an authoritative source.

## Scope discipline

- Implement exactly what the user requested and nothing more.
- Do not add speculative features, optional modes, abstractions, dependencies, configuration, refactors, formatting changes, or unrelated cleanup.
- Preserve the existing public interface and surrounding code unless the requested change requires modifying them.
- Every changed line must be directly attributable to the user request or to the minimum verification needed for that request.
- Match the repository's existing style. Remove only imports, variables, or helpers made unused by your own changes.
- Do not invent missing shapes, dtypes, layouts, strides, tolerances, target hardware, version assumptions, performance goals, or edge-case behavior.

## Uncertainty and clarification

- Before coding, state the operator contract and any assumptions that affect behavior: operation, shapes, dtypes, layouts/strides, target device, accumulation precision, output conversion, and performance requirements.
- If the request or contract is ambiguous, stop and ask the user a focused clarification question. Do not silently choose an interpretation.
- If multiple materially different implementations are valid, briefly present the tradeoff and ask the user to choose unless the local documentation identifies a clear default.
- If a required fact cannot be confirmed from the local references or relevant examples, do not fabricate it. Ask the user for the missing information or source.

## Mandatory uncertainty marker in code

- If any proposed code remains unverified or the agent is not confident that it is correct for the stated Triton-Ascend version, hardware, API, or compiler behavior, place a short comment immediately above the uncertain code.
- The comment text must begin exactly with `ATTENTION!` and briefly explain what is uncertain and what must be verified.
- Use the comment syntax of the target language, for example:

```python
# ATTENTION! Verify this API contract against the installed Triton-Ascend version.
```

- An `ATTENTION!` comment does not replace clarification. Ask the user first when the uncertainty can change the interface, mathematical result, memory safety, supported inputs, or implementation strategy.
- Do not use `ATTENTION!` for ordinary explanations or already verified code.

## Required workflow

1. Restate the requested outcome and define verifiable success criteria.
2. Read `triton-ascend-dev-doc.md` and the minimum task-specific references it selects.
3. Identify ambiguities and ask the user before coding when they affect the result.
4. Implement the smallest change that satisfies the request.
5. Verify correctness in proportion to the change. Prefer an independent Torch reference, representative and boundary shapes, advertised dtypes, and partial-tile cases as applicable.
6. Report what was changed, what was verified, and any remaining `ATTENTION!` items.

Compilation alone is not proof of correctness or performance. Do not claim performance improvement without measurements on the target NPU, after warm-up, with correctness rechecked.
