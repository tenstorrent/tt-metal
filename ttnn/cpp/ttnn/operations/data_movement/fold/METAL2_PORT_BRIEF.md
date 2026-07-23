# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/fold`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `b5b801a923d 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (both `MultiCore` and `MultiCoreDRAMFold`).
- **Op-owned tensors:** none.
- **Target concept:** `MetalV2FactoryConcept`.
- **Two factories, three programs.** `Fold` selects `MultiCore` when `is_sharded`, else `MultiCoreDRAMFold` (`fold_device_op.cpp:11-17`). `MultiCoreDRAMFold::create_descriptor` then forks at runtime on input layout into a **tiled** and a **row-major** program with different kernels/CBs (`fold_multi_core_dram_program_factory.cpp:407-419`) — port all three program shapes.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` — all `no` on a cleared op. No smuggled pointers, no op-owned tensors.

## Construct — to do

**Tensor bindings** (per binding — classification differs by factory):

- **`MultiCore` (sharded)** — `input` and `output` are both **clean (borrowed-DFB)**. The input CB `c_0` (`fold_multi_core_program_factory.cpp:63`) and output CB `c_16` (`:79`) set `cb.buffer = src_buffer` / `dst_buffer`. Port each via `DataflowBufferSpec::borrowed_from` the corresponding `TensorParameter`; the kernel's raw `get_read_ptr`/`get_write_ptr` access is unchanged. No `TensorAccessor` here.
- **`MultiCoreDRAMFold` (tiled + row-major)** — `input` and `output` are both **Case 1** (via `TensorAccessor`). Today the base is delivered as a `Buffer*` in the RTA list (a BufferBinding — `fold_multi_core_dram_program_factory.cpp:223-225,236-238,395-396`). Express each as a `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::name)` instead of `TensorAccessor(args, addr)`, and the `Buffer*` RTA + `TensorAccessorArgs(...).append_to(...)` CTA plumbing both disappear. The per-core scalar index/offset args (`block_start_id`, `output_offset`, `src_idx`, `dst_idx`, `src_col_offset`, …) stay as ordinary named RTAs.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — every accessor is already the 2-arg `TensorAccessor(args, addr)` form.

**CB endpoints:**

- **Sharded `MultiCore` — assign 1P+1C** on `c_0` and `c_16`. Both are touched by *two* same-source instances (`writer_cb2s_row_major.cpp` under Writer- and Reader-config, `fold_multi_core_program_factory.cpp:103-122`) via raw pointer with no FIFO ops → role-free. Bind one instance PRODUCER and the other CONSUMER on each CB (cosmetic on Gen1). **Do not** set the multi-binding flag — this is a plain dual-instance work-split, not a genuine ≥3-toucher.
- **DRAM tiled — all legal 1:1.** `c_0`: reader → compute. `c_1`: compute → writer.
- **DRAM row-major — `c_0` legal 1:1** (reader → writer). **`c_1` (scratch) self-loop** in the `!is_l1_aligned` config (writer is the sole, raw-only toucher: `fold_multi_core_dram_program_factory.cpp:318-331`, `writer_cb2dram_for_rm_input.cpp:33`) — bind the writer both PRODUCER and CONSUMER.
  - ⚠ **`c_1` in the `is_l1_aligned` config:** the writer calls `cb_in1.get_write_ptr()` *unconditionally* (`writer_cb2dram_for_rm_input.cpp:33`) but the factory allocates `c_1` only when `!is_l1_aligned`. A DFB the kernel touches must be bound. Resolve by guarding the `get_write_ptr()` inside `if constexpr (!is_l1_aligned)` (the value is dead when aligned) so the touch matches the allocation — then no `c_1` binding is needed in the aligned config. (Do not simply always-allocate `c_1`; guarding is the zero-behavior-change fix.)

## Watch for

- **CB endpoints (dual-instance work-split):** the sharded factory is the classic two-instances-of-one-source shape. Both instances hit every node, so each node genuinely has two touchers per CB — assign 1P+1C, don't hunt for a hidden writer (there is none) and don't over-escalate to multi-binding.
- **Cross-op / shared kernels:** the tiled sub-variant instantiates `untilize/device/kernels/compute/untilize.cpp` by file path (`fold_multi_core_dram_program_factory.cpp:171`). Its Metal 2.0 CB→DFB rewrite is shared with the untilize op — port that shared compute kernel as one unit, not in isolation. It is a thin wrapper over the DFB-aware `compute_kernel_lib::untilize` helper, so the change is small.
- **RTA varargs:** none — all RTAs are fixed-count distinct fields; use named args throughout.
