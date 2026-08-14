# Triton-Ascend Development Documentation for AI Agents

This is the canonical entry point for AI agents that write, review, migrate, or optimize Triton-Ascend kernels. The documents under [`triton-ascend-dev-doc/`](./triton-ascend-dev-doc/) reorganize the source documentation into task-oriented, independently retrievable sections.

The processed documentation is written in English because Triton, PyTorch, compiler APIs, diagnostics, and most model training data use English identifiers. The original documents remain available as provenance; use the processed set as the working guide.

## How an Agent Should Use This Set

1. Load this index first.
2. Load [Execution Model and Architecture](./triton-ascend-dev-doc/02-execution-model-and-architecture.md) before generating a new kernel or changing a GPU kernel's grid.
3. Check [Operator Support Matrix and Constraints](./triton-ascend-dev-doc/12-operator-support-matrix-and-constraints.md) before selecting an operation/dtype combination, then load only the operator and task sections required by the request.
4. For a CUDA/GPU-derived implementation, always load [GPU/CUDA Triton to Ascend NPU Migration](./triton-ascend-dev-doc/09-gpu-to-ascend-npu-migration.md), even when the original Triton kernel compiles unchanged.
5. Treat hardware capacities, product support, and version numbers as documentation-snapshot facts. Confirm them against the installed Triton-Ascend/CANN release when exact compatibility matters.

## Task Router

| User intent or observed signal | Load these sections |
| --- | --- |
| Install or verify the environment | [01 — Quick Start](./triton-ascend-dev-doc/01-quick-start-and-environment.md) |
| Understand grid, cores, UB/L1, SIMD/SIMT, or compilation flow | [02 — Execution Model](./triton-ascend-dev-doc/02-execution-model-and-architecture.md) |
| Write a new kernel from a PyTorch reference | [03 — Common Kernel Workflow](./triton-ascend-dev-doc/03-common-kernel-development.md), then the matching operator guide |
| Element-wise, reduction, gather/scatter, masked update | [04 — Vector Kernels](./triton-ascend-dev-doc/04-vector-kernels.md) |
| `tl.dot`, GEMM, batched GEMM, attention | [05 — Cube Kernels](./triton-ascend-dev-doc/05-cube-kernels.md) |
| Fused matmul + activation/softmax/reduction/layout work | [06 — Cube-Vector Fusion](./triton-ascend-dev-doc/06-cube-vector-fusion.md) |
| UB overflow, poor MTE utilization, padding, irregular access | [07 — Memory and Performance](./triton-ascend-dev-doc/07-memory-tiling-and-performance.md) |
| Tune block sizes or compiler options | [08 — Autotuning](./triton-ascend-dev-doc/08-autotuning.md) |
| Use Ascend extensions, compiler options, or Ascend 950 modes | [10 — Language and Compiler Reference](./triton-ascend-dev-doc/10-language-and-compiler-reference.md) |
| Check whether an operation/dtype is supported or review atomic/gather/transpose constraints | [12 — Operator Support Matrix](./triton-ascend-dev-doc/12-operator-support-matrix-and-constraints.md) |
| Port CUDA/GPU Triton or fix `coreDim` | [09 — GPU/CUDA → Ascend NPU](./triton-ascend-dev-doc/09-gpu-to-ascend-npu-migration.md) |
| Validate results or diagnose compilation/performance failures | [11 — Validation and Troubleshooting](./triton-ascend-dev-doc/11-validation-and-troubleshooting.md) |

## Ordered Contents

1. [Quick Start and Environment](./triton-ascend-dev-doc/01-quick-start-and-environment.md)
2. [Execution Model and Architecture](./triton-ascend-dev-doc/02-execution-model-and-architecture.md)
3. [Common Kernel Development Workflow](./triton-ascend-dev-doc/03-common-kernel-development.md)
4. [Vector Kernel Development](./triton-ascend-dev-doc/04-vector-kernels.md)
5. [Cube Kernel Development](./triton-ascend-dev-doc/05-cube-kernels.md)
6. [Cube-Vector Fusion Development](./triton-ascend-dev-doc/06-cube-vector-fusion.md)
7. [Memory, Tiling, and Performance](./triton-ascend-dev-doc/07-memory-tiling-and-performance.md)
8. [Autotuning and `max_autotune`](./triton-ascend-dev-doc/08-autotuning.md)
9. [GPU/CUDA Triton to Ascend NPU Migration](./triton-ascend-dev-doc/09-gpu-to-ascend-npu-migration.md)
10. [Ascend Language and Compiler Reference](./triton-ascend-dev-doc/10-language-and-compiler-reference.md)
11. [Validation and Troubleshooting](./triton-ascend-dev-doc/11-validation-and-troubleshooting.md)
12. [Triton Operator Support Matrix and Constraints](./triton-ascend-dev-doc/12-operator-support-matrix-and-constraints.md)

## Non-Negotiable Ascend Invariants

- Python tensors and runtime APIs must target `npu`, and `torch_npu` must be imported.
- A Triton program is a block of work, not a CUDA thread. Do not translate CUDA thread/block terminology literally.
- A GPU-sized logical grid may compile on Ascend but launch and initialization overhead can dominate. Prefer a grid close to the relevant physical core count plus an inner strided tile loop, unless independent logical blocks are intentionally folded with Auto-Blockify.
- Vector-only work is organized around Vector Cores. Kernels containing `tl.dot` and CV-fused work are organized around AI/Cube Cores.
- `coreDim` must not exceed `65,535` unless the Auto-Blockify path is enabled and applicable.
- Tile size is constrained by on-chip memory. On Atlas A2, the documented UB capacity is 192 KiB; buffering and intermediate tensors reduce usable capacity.
- Tail-axis transfers are alignment-sensitive: the source guide documents 32-byte alignment for Vector-only work and 512-byte alignment for Cube-Vector work. Short tails may be padded and become disproportionately expensive.
- Contiguous bulk movement to UB followed by local selection is often better than many discrete global-memory accesses.
- On A2/A3, some integer Vector operations fall back to scalar execution. Do not assume CUDA dtype performance carries over.
- A supported Triton operation is not necessarily supported for every dtype. Check the operation/dtype matrix and its operation-specific constraints before generating code.
- Correctness comes before tuning. Validate against a PyTorch NPU reference across tails, small shapes, empty inputs, and representative large shapes.

## Source Coverage Map

| Original source | Processed destination |
| --- | --- |
| [`quick_start.md`](./quick_start.md) | 01 |
| [`architecture_design_and_core_features.md`](./architecture_design_and_core_features.md) | 02, 10 |
| [`programming_guide/index.md`](./programming_guide/index.md) | 02, 03, 07, 08, 11 |
| [`programming_guide/vector_operator.md`](./programming_guide/vector_operator.md) | 04, 07 |
| [`programming_guide/cube_operator.md`](./programming_guide/cube_operator.md) | 05 |
| [`programming_guide/cv_fusion_operator.md`](./programming_guide/cv_fusion_operator.md) | 06, 07 |
| [`autotune_guide.md`](./autotune_guide.md) | 08 |
| [`max_autotune_guide.md`](./max_autotune_guide.md) | 08, 10 |
| [`migration_guide/index.md`](./migration_guide/index.md) | 09 |
| [`migration_guide/architecture_difference.md`](./migration_guide/architecture_difference.md) | 02, 07, 09, 10 |
| [`migration_guide/migrate_from_gpu.md`](./migration_guide/migrate_from_gpu.md) | 09, 11 |
| [`migration_guide/performance_guidelines.md`](./migration_guide/performance_guidelines.md) | 07, 09, 11 |
| [`outline.md`](./outline.md) | 12 |

## Terminology

| Term | Meaning in this documentation |
| --- | --- |
| AI Core / Cube Core | Matrix-oriented compute resource used for `tl.dot`-style work. Source documents sometimes use these terms interchangeably when discussing launch count. |
| Vector Core | Vector/SIMD compute resource for element-wise, reduction, gather/scatter, and post-processing work. |
| CV kernel | One operator combining Cube and Vector work. |
| UB | Unified Buffer, small on-chip memory used by Vector work and data movement. |
| L1 | On-chip storage used by Cube-oriented data flow; capacity is also a tiling constraint. |
| logical block/program | One instance from the Triton launch grid. |
| physical core launch | Launching approximately one persistent Triton program per relevant hardware core and assigning multiple logical tiles inside it. |
| Auto-Blockify | Compiler/runtime folding of a large independent logical grid onto a bounded physical launch with an internal loop. |
