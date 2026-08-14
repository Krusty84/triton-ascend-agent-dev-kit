---
name: rank-triton-ascend-costmodel
description: Generate TTIR for Triton-Ascend kernel configurations and rank them with the Ascend costmodel backend before empirical autotuning. Use when an agent needs to prefilter a large configuration space, construct costmodel_bench items, bind runtime TTIR arguments or program IDs, or interpret predicted latency values.
---

# Rank Triton-Ascend Configs with Costmodel

## Goal

Predict latency for several compile-time kernel configurations and retain the most promising candidates for real correctness checks and benchmarking.

## Workflow

1. Define the Triton kernel, pointer/runtime signature, and candidate constexpr values.
2. Build a separate ASTSource and TTIR string for every candidate.
3. Enable the costmodel backend in Ascend compiler options.
4. Create one item per candidate with a unique config name, TTIR, and optional arg_bindings.
5. Call costmodel_bench, sort the returned microsecond predictions, and shortlist candidates.
6. Verify and benchmark the shortlist on real NPU inputs.

## Implementation Pattern

~~~python
from triton.backends.ascend.runtime.costmodel_runtime import costmodel_bench
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton.compiler.compiler import make_backend
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir

source = ASTSource(kernel, signature, constants, attrs=None)
backend = make_backend(GPUTarget("npu", "", 32))
options = backend.parse_options({
    "compile_mode": "simd",
    "enable_costmodel_backend": True,
    **source.parse_options(),
})
context = ir.context()
ir.load_dialects(context)
ascend_ir.load_dialects(context)
ttir = str(ast_to_ttir(kernel, source, context, options, {}, {}))

items = [{
    "config": "block256",
    "ttir": ttir,
    "arg_bindings": "arg3=98432,pid_x=0",
}]
latencies_us = costmodel_bench(items)
~~~

## Ascend Guardrails

- Exclude constexpr parameters from the runtime signature and pass their values through ASTSource constants.
- Regenerate TTIR for every constexpr configuration; changing only the item name does not change the compiled program.
- Map argN by the insertion order of runtime arguments in signature. Recheck the index whenever the signature changes.
- Bind pid_x when the kernel uses tl.program_id(0), and bind num_programs_x when it uses tl.num_programs(0).
- Use unique config names because they become keys in the returned dictionary.
- Treat predicted latency as a ranking signal, not proof of correctness or measured hardware performance.

## Verification

Assert that every requested config has a finite returned latency, sort ascending, and compare the top candidates with actual NPU benchmarks. Keep a slower predicted candidate when needed to detect costmodel ranking errors.
