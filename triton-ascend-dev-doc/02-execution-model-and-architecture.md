# Execution Model and Architecture

> Load this section before designing a grid, translating CUDA assumptions, or choosing Vector, Cube, CV, SIMD, or SIMT execution.

## The Mental Model Change

Community Triton code is portable at the language level, but a GPU-optimized launch and memory layout are not automatically NPU-optimized.

| Concern | GPU-oriented expectation | Ascend NPU working model |
| --- | --- | --- |
| Grid | Large logical task space; hardware schedules blocks over SMs | Launch count is closely tied to physical AI/Vector resources; large grids may add repeated dispatch/initialization overhead |
| Compute resources | CUDA cores plus Tensor Cores | Vector Cores for vector/SIMD work; Cube/AI Cores for matrix work; CV kernels use both |
| Program lifetime | Usually one tile per program | Often one program per physical core, processing multiple tiles in an inner strided loop |
| On-chip storage | Registers/shared memory | UB and L1; small capacity directly constrains tile size and buffering |
| Irregular memory | Often tolerated through SIMT threads | SIMD paths can scalarize discrete access; reorganize into contiguous transfers or use Ascend 950 hybrid SIMT when applicable |
| Dtypes | CUDA performance intuition | Some A2/A3 integer Vector operations scalarize; dtype choice must be profiled |
| Matrix work | `tl.dot` maps to Tensor Core paths | `tl.dot` maps toward Cube-oriented lowering and should use Cube-compatible tiling |

Do not equate a Triton program with an individual CUDA thread. A Triton program describes a block/tile of work.

## Hardware Core Selection

The source guide describes one AI Core as containing one Cube Core paired with two Vector Cores. Exact counts vary by product. Query the active device:

```python
import torch_npu
import triton.runtime.driver as driver

device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
num_vectorcore = properties["num_vectorcore"]
num_aicore = properties["num_aicore"]
```

Use:

- `num_vectorcore` for Vector-only kernels;
- `num_aicore` for kernels containing `tl.dot` and for CV-fused kernels;
- a smaller count only when the task count or profiling justifies it.

## Three Grid Strategies

### 1. Conventional logical grid

```python
grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
```

Use this first for correctness and when the logical grid is small. Ascend accepts multidimensional grid syntax, but source guidance says NPU 2D adaptations are merged to 1D; prefer a clearly flattened 1D mapping unless a tested kernel requires otherwise.

### 2. Physical-core launch with an inner work loop

```python
@triton.jit
def kernel(..., n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = tl.cdiv(n_elements, BLOCK_SIZE)

    for tile_id in range(pid, num_tiles, num_programs):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        ...

kernel[(num_vectorcore,)](...)
```

This is the preferred NPU optimization when a GPU-style grid would be far larger than the physical core count. It amortizes program startup and distributes tiles round-robin across cores.

For multidimensional work, flatten the logical tile ID and reconstruct coordinates:

```python
tile_hm = tile_id // num_tiles_m
tile_m = tile_id % num_tiles_m
batch = tile_hm // num_heads
head = tile_hm % num_heads
```

### 3. Auto-Blockify for independent logical blocks

Auto-Blockify folds a large logical grid onto physical cores:

- compile time wraps the kernel body in an internal loop over logical block IDs;
- runtime caps the actual launch at the physical core count;
- workspace is reused per physical core rather than allocated per logical block.

Enable globally:

```bash
export TRITON_ALL_BLOCKS_PARALLEL=1
```

The compiler also exposes a per-kernel `enable_auto_blockify` override; resolution is per-kernel option, then environment variable, then disabled.

Use Auto-Blockify only when logical blocks are order-independent. Rewrite kernels that rely on a particular cross-block execution order or cross-block synchronization schedule.

## Grid Limits

- Without Auto-Blockify, `coreDim` must be at most `UINT16_MAX` (`65,535`).
- A grid within `65,535` can still be much larger than the physical core count and therefore slow.
- Increasing `BLOCK_SIZE` lowers `coreDim` but can cause UB overflow. Use a large outer logical block plus a smaller inner sub-block when both constraints apply.

The compound solution is described in [GPU/CUDA → Ascend NPU Migration](./09-gpu-to-ascend-npu-migration.md#compound-coredim-and-ub-overflow).

## On-Chip Memory and Alignment

UB/L1 capacity limits every live tile, intermediate tensor, reduction buffer, and pipeline buffer. The source documentation gives Atlas A2 UB capacity as 192 KiB (1,572,864 bits). Double/multi-buffering and extra compiler buffers reduce the usable amount.

The source guide documents tail-axis alignment requirements:

- Vector-only (VV): 32 bytes;
- Cube-Vector (CV): 512 bytes.

If a short tail is padded to these boundaries, shapes such as `[2048, 3]` can become unexpectedly expensive. Favor contiguous aligned axes, transpose/reorganize in UB when worthwhile, and profile the result.

## Compiler and Runtime Architecture

Triton-Ascend consists of:

- Ascend language extensions exposed under Triton language facilities;
- an Ascend compiler backend that receives TTIR and lowers toward Linalg/AscendNPU IR and a device object;
- an Ascend driver that connects Triton runtime to CANN/TorchNPU and launches the compiled object.

The main SIMD path is conceptually:

```text
Triton Python → TTIR → structured/unstructured rewrites → Linalg IR
              → AscendNPU/HIVM/HFusion/LLVM lowering → device object
```

This matters because pointer expressions, masks, discrete indices, synchronization, and `tl.dot` follow different lowering paths. A source-level expression that is efficient on GPU can become a scalar loop on NPU.

## Ascend 950 Compilation Modes

Ascend 950 adds SIMT support for discrete/unstructured access:

| `compile_mode` | Behavior | Use when |
| --- | --- | --- |
| `"unstructured_in_simt"` | Default hybrid: structured work stays SIMD; eligible discrete access uses SIMT | General Ascend 950 kernels with irregular load/store |
| `"simd"` | Pure SIMD; unstructured access can lower to scalar loops | Structured, contiguous kernels or controlled comparison |
| `"simt_only"` | Triton IR goes to a pure SIMT path | A kernel is intentionally designed for SIMT; validate all behavior |

```python
kernel[grid](..., compile_mode="unstructured_in_simt")
kernel[grid](..., compile_mode="simd")
kernel[grid](..., compile_mode="simt_only", num_warps=32)
```

In hybrid mode, SIMT is used only at eligible discrete access points (documented rank limit: at most 5); the rest remains on the SIMD/Linalg path. Do not assume this option exists or behaves identically on A2/A3.

## Design Decision Checklist

Before writing the kernel, answer:

1. Is the work Vector-only, Cube, or CV?
2. What physical core count should bound concurrent programs?
3. Is a logical grid sufficient, should the kernel use an inner tile loop, or are blocks independent enough for Auto-Blockify?
4. What tensors and intermediates are live in UB/L1 at once?
5. Is the innermost transfer contiguous and aligned?
6. Can any discrete global-memory access be converted into a bulk load plus UB-local selection?
7. On Ascend 950, does hybrid SIMT improve the truly unstructured portion?

