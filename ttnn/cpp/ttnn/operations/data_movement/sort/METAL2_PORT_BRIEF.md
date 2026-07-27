# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/sort`

> Audit cleared all gates for **all three** program factories. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `44da718b06b 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

**Scope:** one `DeviceOperation` (`ttnn::prim::SortDeviceOperation`), three program factories selected at runtime by `Wt`:
`SortProgramFactorySingleRowSingleCore`, `SortProgramFactorySingleRowMultiCore`, `SortProgramFactoryCrossCoreDataExchange`.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); all three factories port to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:**
  - `SingleRowSingleCore` — `descriptor` (`create_descriptor()` → `ProgramDescriptor`, `sort_program_factory.cpp:21`).
  - `SingleRowMultiCore` — `descriptor` (`:969`).
  - `CrossCoreDataExchange` — `WorkloadDescriptor`, **secretly SPMD** (one `ProgramDescriptor` replicated across the mesh coords, `:924-930`) → collapses to the single-program concept.
- **Op-owned tensors:**
  - `SingleRowSingleCore`, `SingleRowMultiCore` — none.
  - `CrossCoreDataExchange` — one: the physical-core lookup table, built on cache-miss and parked on `wd.buffers` (`sort_program_factory.cpp:475-497, 915-919`). Carried natively by `MetalV2FactoryConcept`; bind it as a `TensorParameter` (its kernel access is Case 1, below).
- **Target concept:** `MetalV2FactoryConcept` for all three (with op-owned tensors for `CrossCoreDataExchange`).
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` — all `no` on the readiness sheet and in the code. `Is safe to port? = yes`, `Smuggled pointer = no` (the op uses the framework's `Buffer*`-binding pointer-patching, `Op Classification = PD (pointer-patching)`).

## Construct — to do

**Tensor bindings** (per binding) — **all Case 1** across all three factories (the kernel feeds the base into a `TensorAccessor`; no raw-pointer arithmetic, no borrowed-memory DFB). Each is delivered today as a `Buffer*` runtime arg (`emplace_runtime_args(core, {input_buffer, ...})`) and read in the kernel via `TensorAccessor(args, get_arg_val<uint32_t>(k))`. Port each: declare a `TensorParameter` / `TensorBinding`, build `TensorAccessor(tensor::name)` in the kernel, and delete the address-via-RTA plus its `TensorAccessorArgs<N>()` compile-time plumbing.

- `SingleRowSingleCore`: `input` (reader `:53`), `index`-output (reader `:54`), `value`-output (writer `:53`).
- `SingleRowMultiCore`: `input` (coordinator `:53`, reader `:45`), `value`-output (coordinator `:54`, writer `:43`), `index`-output (coordinator `:55`, reader `:46`, writer `:44`).
- `CrossCoreDataExchange`: `input` (reader `:58`), `index`-output (reader `:64`), `value`-output (writer `:45`), and the op-owned `physical_core_lookup_table` (reader `:71`).

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — every accessor is already 2-arg; nothing to drop.

**CB endpoints:** apply per `(CB, config)`; classify per instantiation (dispositions flip between TILE and ROW_MAJOR, and — for the multi-core factory — between the coordinator node and the worker nodes). Full census in `METAL2_PREPORT_AUDIT.md`. Summary of actions:

- **Self-loop** (bind the one compute kernel PRODUCER *and* CONSUMER — or the coordinator kernel, for its CBs):
  - `SingleRowSingleCore`: c_2, c_3, c_6 (both configs); c_0, c_10 (RM only).
  - `SingleRowMultiCore` workers: c_2, c_3 (both); c_0, c_1, c_4, c_5 (RM only). Coordinator: c_0/c_1 (TILE), c_6/c_7 (RM).
  - `CrossCoreDataExchange`: c_2, c_3, c_11 (both); c_0, c_15 (RM only).
- **1P+1C** (one PRODUCER, one CONSUMER; no flag): every remaining live CB — including `CrossCoreDataExchange`'s exchange CBs c_6/c_7 (compute→reader) and c_8/c_9 (reader→compute), which run in both configs.
- **Multi-binding advanced-option flag** — **exactly one CB in the whole op:** `CrossCoreDataExchange` **c_10 `physical_core_lookup_table`**. Two kernels drive the producer cursor: the reader is the genuine producer (`reader_cross_core_data_exchange.cpp:75` reserve, `:216` push, plus self-reads via `get_read_ptr`), and the writer does a lone `push_back(one_tile)` at `writer_cross_core_data_exchange.cpp:102`. Set the multi-binding advanced option on c_10 (records Quasar debt). See Watch-for.
- **Drop dead / narrow over-scoped CBs** (else the spec validator rejects an unbound DFB — behavior-neutral):
  - `SingleRowSingleCore` **RM only:** c_4 `value_tensor`, c_5 `index_tensor_output` have zero touchers (RM uses c_8/c_9). Guard their allocation on `!is_row_major` (they're allocated unconditionally at `sort_program_factory.cpp:140-160`).
  - `CrossCoreDataExchange` **RM only:** c_4 `value_tensor`, c_5 `index_tensor_output` — same dead-in-RM shape; guard on `!is_row_major`.
  - `SingleRowMultiCore`: c_0-c_5 are declared on `all_core_set` but the coordinator kernel only touches c_0/c_1 (TILE) or c_6/c_7 (RM). Narrow c_2-c_5 (and c_0/c_1 for the RM instantiation) to the worker `core_range` so the coordinator node carries no unbound DFB.

## Watch for

- **CB endpoints (multi-binding):** the single flag is `CrossCoreDataExchange` **c_10**. Before setting it, confirm the reader is the sole real producer and the writer's `writer_cross_core_data_exchange.cpp:102` `push_back` is the only extra producer-role access (it is — the writer never `reserve_back`s c_10). That writer `push_back` is vestigial; **do not delete it during the port** (removing a stray FIFO op is a functional change, out of the kernel-side whitelist) — set the multi-binding flag and leave the code as-is. (If the ops team later removes it, c_10 becomes a clean reader self-loop.)
- **Config-dependence:** every CB's disposition can differ between TILE and ROW_MAJOR (and, for the multi-core factory, between coordinator and worker nodes). Build the census per instantiation — do not assume one disposition carries across configs. The aliased CBs in `SingleRowMultiCore` (c_6/c_7 = `rm_coord_*` on the coordinator and `rm_worker_input_*` on the workers) sit on disjoint node sets, so each node sees exactly one instance — keep them as separate per-range `DataflowBufferSpec`s.
- **Cross-op / shared kernels:** none. All kernels are in-directory; the three shared headers (`sort_common.hpp`, `sort_dataflow_common.hpp`, `cross_core_data_exchange_common.hpp`) are the op's own. No port-together coupling with other ops.
- **RTA varargs:** none — every runtime arg is read at a fixed offset and is nameable. Prefer named args throughout; no vararg mechanism is needed.
- **Vestigial CTAs (leave in place):** the cross-core writer declares several unused compile-time args (arg 4 `value_tensor_peer_cb_index`, arg 9 `number_of_cores_used`, arg 10 now only a comment). These are pre-existing; port the kernel's *live* args to named args and don't chase the dead ones into functional changes.
