# Metal 2.0 Port Brief — `data_movement/indexed_fill`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `ca0b78e9ad7 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## Op shape

One DeviceOperation, one ProgramFactory, four internal code paths selected inside `create_descriptor` by tensor geometry (not separate factory variants). The `mode` compile-time arg selects the path in the unified reader:

- **Generic 2D-stride** (`MODE_GENERIC`) — interleaved input_a, or any `dim != 0` geometry. Reader + `indexed_fill_writer_strided.cpp` scatter writer.
- **Native** (`MODE_NATIVE`) — `dim==0`, HEIGHT_SHARDED L1, one batch per core. Reader + `indexed_fill_writer.cpp` wait/pop stub; data CB aliased to output.
- **Shard-local, interleaved B** (`MODE_SHARD_LOCAL_INTERLEAVED_B`) and **shard-local, sharded B** (`MODE_SHARD_LOCAL_SHARDED_B`) — `dim==0`, WIDTH/BLOCK_SHARDED L1. Same writer stub; data CB aliased to output.

The per-binding and per-CB dispositions below vary across these paths — apply each in the path it names.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (`IndexedFillProgramFactory::create_descriptor` returns a `ProgramDescriptor`).
- **Op-owned tensors:** none — carried by the target concept without the `WorkloadDescriptor` shape.
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` · other migration-risky pybind. All `no` on the readiness sheet and in the code.

## Construct — to do

**Tensor bindings** (per binding; all four arrive today as `Buffer*` RTAs that the framework registers as `BufferBinding`s — replace each with a typed `TensorParameter` / `TensorBinding`). Classification is config-dependent within the one factory:

- **`batch_ids`** — **Case 1** (via `TensorAccessor`, all paths) → express as `TensorParameter`; kernel builds `TensorAccessor(tensor::name)`. Drop its address RTA and `TensorAccessorArgs` CTAs. (Also drop its 3rd page-size arg — see below.)
- **`input_a`** — **Case 1** in the generic and native paths (accessor `s0`; native does a bulk `s0` read) → `TensorAccessor(tensor::name)`. **Case 2** in the shard-local paths (raw base + device-side offset arithmetic, `indexed_fill_reader.cpp:130-138`) → bind the tensor, pull the base via `get_bank_base_address`, leave the raw arithmetic unchanged.
- **`input_b`** — **Case 1** in the generic / native / shard-local-interleaved-B paths (accessor `s1`). **Case 2** in the shard-local same-sharded-B path (raw base + offset, `indexed_fill_reader.cpp:105-113`) → `get_bank_base_address` bridge, raw walk unchanged.
- **`output`** — **Case 1** in the generic path (accessor `dst` in the strided writer) → `TensorAccessor(tensor::name)`. In the native / shard-local paths the data CB is **borrowed-memory** (globally allocated to `output.buffer()`, `indexed_fill_program_factory.cpp:261`/`:273`); the writer is a wait/pop stub → port via `DataflowBufferSpec::borrowed_from` (mechanical), no output accessor there.

  Because `input_a` and `input_b` are each Case 1 in some paths and Case 2 in others, the same `TensorParameter` is bound once but the kernel-side consumption differs by `mode` — keep the existing `if constexpr` path split; only the *base delivery* changes (typed binding instead of the RTA address), not the per-path arithmetic.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** drop the redundant page-size arg @ `indexed_fill_reader.cpp:49` — `TensorAccessor(batch_ids_args, batch_ids_addr, batch_id_size << 2)` → `TensorAccessor(tensor::name)`. It is Class 2 (inert: the accessor is only read at `page_id = 0`). **Ignore the adjacent comment** (`indexed_fill_reader.cpp:47-48`) claiming the arg guards against a stale `AlignedPageSize` on cache hits: the stride never multiplies a nonzero index, and the Metal 2.0 binding refresh rebuilds the accessor args from the bound base each enqueue, which is exactly the staleness the comment worried about. No `dynamic_tensor_shape`.

**CB endpoints:**

- **Data CB (`cb_index = 0`)** — legal 1:1. Bind the reader **PRODUCER** and the writer **CONSUMER**. In the native / shard-local paths this CB is `borrowed_from(output.buffer())` — carry the borrowed-memory alias through (`DataflowBufferSpec::borrowed_from`); in the generic path it is a plain local 2-page CB.
- **Batch CB (`batch_cb_index = 1`)** — **self-loop** (one toucher: the reader both fills it and reads the staged indices back). Bind the reader as **both PRODUCER and CONSUMER**. All paths.

## Watch for

- **CB endpoints (multi-binding):** none — no hidden second writer, no multi-reader CB, no semaphore-gated co-fill (the op uses no semaphores). Straightforward 1P+1C (data CB) and self-loop (batch CB).
- **Cross-op / shared kernels:** none — all three kernels are owned by this op and instantiated from its own directory; every `#include` is `api/*`. No port-together coupling.
- **RTA varargs:** none — all runtime args are read at fixed constexpr indices; name each. The batch-id-count arg (`batch_id_size`, reader arg 1) is one fixed named field, not a vararg; the reader's internal `for k < batch_id_size` loop scans staged L1 data, not runtime args.
- **The four internal paths share one reader kernel** gated by the `mode` CTA. Keep the `if constexpr` structure; the port changes base delivery (typed bindings) and drops the redundant CTAs/3rd-arg, but must not collapse or reorder the path logic.
