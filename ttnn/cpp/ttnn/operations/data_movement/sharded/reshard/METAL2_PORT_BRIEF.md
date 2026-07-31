# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/sharded/reshard`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

**Op shape:** one `ReshardDeviceOperation`, eight `descriptor` program-factory variants (five factory types) — `ReshardSameWidthFactory<bool>` (×2), `ReshardSameHeightFactory<bool>` (×2), `ReshardGenericFactory`, `NdReshardCopyPagesFactory`, `NdReshardCopyLocalShardFactory<bool>` (×2). Nine kernels total (3 op-owned + 6 in-family shared). Port all factories together.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (all 8 factories — verified: each exposes `create_descriptor()` returning a `ProgramDescriptor`).
- **Op-owned tensors:** none (a `descriptor` op cannot carry them).
- **Target concept:** `MetalV2FactoryConcept` (no op-owned tensors).
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash (`no` — no `compute_program_hash` in the op) · custom `override_runtime_arguments` / `get_dynamic_runtime_args` (`no`) · pybind `create_descriptor` (`no` — `reshard_nanobind.cpp` binds only `ttnn::reshard`). `Is safe to port? = yes` and `Smuggled pointer = no` on all 8 factory rows.

## Construct — to do

**Tensor bindings** (per binding, per factory — all currently delivered via the `Buffer*`-binding fast-path form; the port replaces these with typed `TensorParameter` / `TensorBinding`):

- `ReshardGenericFactory` / output CB (`output_buffer`) — **clean** (borrowed-memory DFB, `cb.buffer = output_buffer`) → port via `DataflowBufferSpec::borrowed_from`.
- `ReshardGenericFactory` / input tensor — **Case 2** (raw pointer) → bind the tensor, pull the base via `get_bank_base_address`, keep the raw NoC walk unchanged (`.addr = input_shard_addr + addr_offset`, explicit `noc_x/noc_y`, `reshard_reader.cpp:60-67`).
- `ReshardSameWidthFactory` / local CB (`local_buffer`) — **clean** (borrowed-memory DFB).
- `ReshardSameWidthFactory` / remote tensor — **Case 2** (raw `AllocatorBank` addressing) → `get_bank_base_address` bridge, raw walk unchanged.
- `ReshardSameHeightFactory` / local CB (`local_buffer`) — **clean** (borrowed-memory DFB).
- `ReshardSameHeightFactory` / remote tensor — **Case 2** (raw `AllocatorBank` addressing) → `get_bank_base_address` bridge, raw walk unchanged.
- `NdReshardCopyPagesFactory` / input tensor — **Case 1** (via `TensorAccessor`, `nd_reshard_copy_pages_reader.cpp:26`) → express as `TensorParameter`; kernel uses `TensorAccessor(tensor::name)`.
- `NdReshardCopyPagesFactory` / output tensor — **Case 1** (`nd_reshard_copy_pages_writer.cpp:26`) → same.
- `NdReshardCopyLocalShardFactory` / input tensor — **Case 1** (`nd_reshard_copy_local_shards.cpp:44`) → same.
- `NdReshardCopyLocalShardFactory` / output tensor — **Case 1** (`nd_reshard_copy_local_shards.cpp:45`) → same.

Case-2 rule of thumb: bind the tensor, bridge the base via `get_bank_base_address`, and **keep the existing raw arithmetic unchanged** — do not rewrite raw NoC access into `TensorAccessor` iteration.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — every accessor is 2-arg; nothing to drop.

**CB endpoints:**

- **1P+1C assign** (dual-instance work-split — same `kernel_source` in a ReaderConfig + WriterConfig pair over one core range, splitting work by disjoint ranges; bind one instance PRODUCER, the other CONSUMER — cosmetic on Gen1):
  - `ReshardGenericFactory` — output CB (`dst_cb_index = 16`, `output_buffer`), all configs.
  - `ReshardSameWidthFactory` — local CB (`c_0`, `local_buffer`), all configs.
  - `ReshardSameWidthFactory` — scratch CB (`c_1`, `buffer=nullptr`), **only when `unaligned && local_is_output`** (config-dependent existence).
  - `ReshardSameHeightFactory` — local CB (`c_0`, `local_buffer`), all configs.
- **Legal 1:1** (no action) — `NdReshardCopyPagesFactory` CB (`c_0`, `buffer=nullptr`): reader is a locked producer (`reserve_back`/`push_back`), writer a locked consumer (`wait_front`/`pop_front`).
- **No CBs** — `NdReshardCopyLocalShardFactory` (direct `TensorAccessor` + `CoreLocalMem` L1↔L1/DRAM copy).
- No dead CBs to drop; **no multi-binding flag anywhere** (no CB reaches ≥3 touchers or ≥2 locked same-role).

## Watch for

- **CB endpoints (multi-binding):** none. Every two-toucher CB is the dual-instance work-split shape → assign **1P+1C**; do **not** reach for the multi-binding advanced option. There are no hidden-second-writer (semaphore-gated co-fill) shapes — the op uses **no semaphores at all** — and no multi-reader ≥3-toucher shapes.
- **Cross-op / shared kernels (port-together set):** the six shared kernels in `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/` (`reshard_reader.cpp`, `reshard_reader_diff_width.cpp`, `reshard_same_width_reader.cpp`, `reshard_same_width_writer.cpp`, `reshard_same_height_reader.cpp`, `reshard_same_height_writer.cpp`) are file-path instantiated by the Generic/SameWidth/SameHeight factories and are **also borrowed by `experimental/quasar/reshard/`** (a Gen2 port, out of scope here). Their Metal 2.0 CB→DFB / named-token rewrite is a single change — coordinate so the Quasar reshard is not broken; port the shared kernels + both consuming ops as one unit.

  > **⚠ CORRECTION (port, 2026-07-31) — this bullet is stale; do not act on it.**
  > `experimental/quasar/reshard/` is **not** a co-borrower. It carries its own private copies of
  > all nine kernels under `experimental/quasar/reshard/device/kernels/` and instantiates only
  > those paths (verified by extracting every `kernels/` string literal from
  > `experimental/quasar/reshard/device/*.cpp`). A repo-wide search for consumers of the
  > `data_movement/sharded/device/kernels/dataflow/reshard_*` paths returns **only** this op's
  > three factories, so this op is the complete consumer set and the shared kernels were modified
  > **in place** — no fork, no Quasar coordination, no expanded scope. Separately, all five Quasar
  > reshard factories are *already* on `create_program_artifacts`, not the "still legacy device-op"
  > state recorded here.
  > The original text is preserved above as the audit-time record. See
  > `METAL2_PORT_PLAN.md` → *Cross-op kernels* and `METAL2_PORT_REPORT.md` → *Confusion* for the
  > evidence and the fork-vs-in-place decision.
- **RTA varargs (prefer the kernel-side vararg mechanism — do not try to name each):**
  - `reshard_reader.cpp:35` — `for (range_id < num_ranges)` with in-loop `arg_index++` reads; also data-selected reads (`get_arg_val(start_x_index)` / `get_arg_val(core_id_x_index)`, `:41-42,60-61`).
  - `reshard_reader_diff_width.cpp:35` — `for (block_id < num_blocks)` with nested `current_pattern_arg_index++` reads.
  - `reshard_same_width_reader.cpp` / `reshard_same_width_writer.cpp` / `reshard_same_height_reader.cpp` / `reshard_same_height_writer.cpp` — a `get_arg_addr(N)` pointer walked in a runtime-count loop (`num_reads` / `num_segments`); the leading scalars (args 0-4) are nameable, the loop body is varargs.
  - The `nd_reshard_*` kernels read only fixed RTAs + constexpr-offset common args → name each; no varargs.
