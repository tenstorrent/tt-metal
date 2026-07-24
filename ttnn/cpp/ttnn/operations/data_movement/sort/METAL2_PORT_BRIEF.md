# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/sort`

> **Scoped-subset brief.** The audit is RED at op level, but the single blocker (a Device 2.0 holdover) is confined to one factory. This brief covers the **clean two-factory subset** that clears every gate and can be ported now:
> **`SortProgramFactorySingleRowSingleCore`** and **`SortProgramFactorySingleRowMultiCore`**.
> **Out of scope for this brief:** `SortProgramFactoryCrossCoreDataExchange` — blocked on one dead Device-1.0 line (`get_semaphore` at `device/kernels/dataflow/writer_cross_core_data_exchange.cpp:26`); it re-audits and ports after the Device 2.0 team clears that line. Do **not** port `CrossCoreDataExchange` under this brief. The full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared (for the two in-scope factories):** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `44da718b06b 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

> **Before you start — a planning choice to confirm with the requester.** These two factories share one `program_factory_t` variant with the blocked `CrossCoreDataExchange`, so this subset port produces a mixed-concept variant (two `MetalV2FactoryConcept`, one still `WorkloadDescriptor`). Because the blocker is a single dead line, the requester may prefer to have the Device 2.0 team delete it first and port all three factories together at re-audit (likely less total work). Confirm the scoped-subset path before proceeding.

## TTNN factory analysis

Both in-scope factories are current concept `descriptor` with **no** op-owned tensors; they port to `MetalV2FactoryConcept`. Carry forward:

- **Current concept:** `descriptor` (both). `create_descriptor()` returns a `ProgramDescriptor` — `SingleRowSingleCore` at `sort_program_factory.cpp:21`, `SingleRowMultiCore` at `:969`.
- **Op-owned tensors:** none (the `descriptor` concept can't carry them; only the out-of-scope `CrossCoreDataExchange` has one).
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` — all `no` on the readiness sheet and in the code. `Is safe to port? = yes`, `Smuggled pointer = no` (the op uses the framework's `Buffer*`-binding pointer-patching, `Op Classification = PD (pointer-patching)`).

## Construct — to do

**Tensor bindings** (per binding) — **all Case 1** (kernel feeds the base into a `TensorAccessor`; no raw-pointer arithmetic, no borrowed-memory DFB). Each is delivered today as a `Buffer*` runtime arg (`emplace_runtime_args(core, {input_buffer, ...})`) and read in the kernel via `TensorAccessor(args, get_arg_val<uint32_t>(k))`. Port each: declare a `TensorParameter` / `TensorBinding`, build `TensorAccessor(tensor::name)` in the kernel, and delete the address-via-RTA plus its `TensorAccessorArgs<N>()` compile-time plumbing.

- `SingleRowSingleCore`:
  - `input` — **Case 1** (reader, `reader_single_row_single_core.cpp:53`).
  - `index`-output — **Case 1** (reader writes it, `:54`).
  - `value`-output — **Case 1** (writer, `writer_single_row_single_core.cpp:53`).
- `SingleRowMultiCore`:
  - `input` — **Case 1** (coordinator `coordinator_single_row_multi_core.cpp:53`; worker reader `reader_single_row_multi_core.cpp:45`).
  - `value`-output — **Case 1** (coordinator `:54`; worker writer `writer_single_row_multi_core.cpp:43`).
  - `index`-output — **Case 1** (coordinator `:55`; worker reader/writer `reader_single_row_multi_core.cpp:46`, `writer_single_row_multi_core.cpp:44`).

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — every accessor is already 2-arg; nothing to drop.

**CB endpoints:** apply per `(CB, config)`; classify per instantiation (dispositions flip between TILE and ROW_MAJOR). No CB needs the multi-binding advanced option. Full census in `METAL2_PREPORT_AUDIT.md`; the actions:

- **Self-loop** (bind the one compute kernel as PRODUCER *and* CONSUMER):
  - `SingleRowSingleCore`: c_2 `input_tensor_transposed`, c_3 `index_tensor_transposed`, c_6 `synchronization` (both configs); c_0 `input_tensor` and c_10 `rm_post_sort_index` (ROW_MAJOR only).
  - `SingleRowMultiCore` (worker nodes): c_2, c_3 (both configs); c_0, c_1, c_4, c_5 (ROW_MAJOR only). Coordinator node: c_0/c_1 (TILE), c_6/c_7 `rm_coord_*` (ROW_MAJOR) — the coordinator both fills and drains, so self-loop.
- **1P+1C** (ordinary FIFO — one PRODUCER, one CONSUMER; no flag): every remaining live CB (e.g. `SingleRowSingleCore` c_0/c_1/c_4/c_5 in TILE, c_7/c_8/c_9 in RM; `SingleRowMultiCore` worker c_0/c_1/c_4/c_5 in TILE, c_6/c_7/c_8/c_9 in RM).
- **Drop dead / narrow over-scoped CBs** (else the spec validator rejects an unbound DFB — behavior-neutral, a CB no kernel touches has no behavior):
  - `SingleRowSingleCore`, **ROW_MAJOR only:** c_4 `value_tensor` and c_5 `index_tensor_output` have **zero touchers** (RM routes output through c_8/c_9). They are allocated unconditionally at `sort_program_factory.cpp:140-160` — guard their allocation on `!is_row_major`, mirroring the existing `if (is_row_major)` guard on c_7-c_10.
  - `SingleRowMultiCore`: c_0-c_5 are declared on `all_core_set` (coordinator + workers) but the coordinator kernel only touches c_0/c_1 (TILE) or c_6/c_7 (RM), never c_2-c_5. Narrow c_2-c_5 (and c_0/c_1 for the RM instantiation) to the worker `core_range` so the coordinator node carries no unbound DFB.

## Watch for

- **CB endpoints (multi-binding):** none in this subset — no hidden second writer, no multi-reader reaching ≥3 touchers. (For context only: the out-of-scope `CrossCoreDataExchange` has a semaphore-gated cross-core exchange and a stray lookup-table `push_back`; ignore it under this brief.) The only non-obvious CB actions are the self-loops and the dead/over-scoped drops above.
- **Config-dependence:** every CB's disposition can differ between TILE and ROW_MAJOR (and, for the multi-core factory, between the coordinator node and the worker nodes). Build the census per instantiation — do not assume one disposition carries across configs.
- **Cross-op / shared kernels:** none. All kernels are in-directory; the three shared headers (`sort_common.hpp`, `sort_dataflow_common.hpp`, and — `CrossCoreDataExchange`-only — `cross_core_data_exchange_common.hpp`) are the op's own. No port-together coupling with other ops.
- **RTA varargs:** none — every runtime arg is read at a fixed offset and is nameable. Prefer named args throughout; no vararg mechanism is needed.
