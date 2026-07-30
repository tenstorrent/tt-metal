# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/normalization/groupnorm`

One `DeviceOperation` in this directory, with three program factories:

- **`ttnn::prim::GroupNormDeviceOperation`** (`device/groupnorm_device_operation.hpp`)
  - `GroupNormShardedProgramFactory` (`device/groupnorm_sharded_program_factory.cpp`)
  - `GroupNormNoMcastProgramFactory` (`device/groupnorm_no_mcast_program_factory.cpp`)
  - `GroupNormMcastProgramFactory` (`device/groupnorm_mcast_program_factory.cpp`)

Factory selection (`device/groupnorm_device_operation.cpp:15-45`): sharded input → sharded factory; otherwise `batch >= num_virtual_rows` → no-mcast, else mcast.

All 16 kernel files under `device/kernels/` are referenced by at least one factory; there is no unreferenced kernel code in the directory. Every kernel the op runs lives in the op's own directory — no kernel file is instantiated by path from another op or shared pool.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `4d6e9518bf5 2026-07-30 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/normalization/groupnorm` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `GroupNormDeviceOperation` → `GroupNormShardedProgramFactory`, `GroupNormNoMcastProgramFactory`, `GroupNormMcastProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | Yes — all 16 kernels are structurally Device 2.0 (`Noc`, `Semaphore<>`, `DataflowBuffer`, `CoreLocalMem`, `UnicastEndpoint` / `MulticastEndpoint`). No legacy free-function data movement anywhere. |
| *Prereqs* — Cross-op escapes | Ok, with one ⭐ flag (`generate_bcast_col_scalar(CircularBuffer, …)`) |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | N/A (absent) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** — all three factory rows |
| *TTNN Readiness* — Concept (current) | `descriptor` (all three factories) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No (no `compute_program_hash` anywhere in the op) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No (`groupnorm_nanobind.cpp` binds only the user-facing op) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | none — no `->address()` expression exists anywhere in the op |
| *Port work* — Tensor bindings (per binding) | `input` clean (sharded) / Case 1 (mcast, no-mcast) · `output` clean (sharded) / Case 1 (mcast, no-mcast) · `gamma` Case 1 · `beta` Case 1 · `input_mask` Case 1 · `negative_mask` Case 1 · `reciprocals` clean |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none — every `TensorAccessor` construction in the op is 2-argument |
| *Port work* — CB endpoints | mostly 1P+1C; several self-loops; **no** multi-binding; **config-scoped dead-CB drops** (see below) |

**CB endpoints** are dispositions, not gates: every out-of-window CB here resolves with a self-loop, a 1P+1C assignment, or a dead-CB drop. No CB in this op needs the multi-binding advanced option — each node runs exactly one reader, one writer and one compute kernel, and no CB is touched by all three in a way that cannot fit 1P+1C.

## Result

**GREEN → brief issued.** All five gates clear: Device 2.0, feature compatibility, TTNN factory concept, offset base pointers, and the `TensorAccessor` 3rd argument.

The op is a large but structurally clean port target: three `descriptor`-concept factories sharing one device-operation, ~20 circular buffers per factory, and a kernel set that was already migrated to Device 2.0 and to `DataflowBuffer` handles. The real cost in the port is breadth (three factories × a welford / non-welford kernel split × several layout configs), not any single blocking construct.

Two things the porter must not skim past, both detailed below: the **config-scoped dead CBs** (several allocated CB indices have zero touchers under specific configurations, and a bindingless DFB is rejected by the spec validator), and the **runtime-arg vararg block** in the mcast reader kernels.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** The readiness sheet carries three rows for `normalization/groupnorm`, one per factory, all with `Concept = descriptor`, `Custom hash = no`, `Runtime-args update (get_dynamic_runtime_args) = no`, `Override runtime args method? = no`, `Pybind descriptor = no`, `Smuggled pointer = no`, `Is safe to port? = yes`, `Is able to port? = yes`, `TensorParameter relaxation = none`, `Porting Target = ProgramSpecFactoryConcept`.

  Cross-check against the code, all clean:
  - `Concept` — all three factories declare `static ProgramDescriptor create_descriptor(...)` ([groupnorm_device_operation.hpp:24-46](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_device_operation.hpp#L24-L46)). No mesh-workload return, no `create()` + `override_runtime_arguments()` pair.
  - `Custom hash` / `get_dynamic_runtime_args` / `override_runtime_arguments` — a grep over the whole op directory for `compute_program_hash`, `get_dynamic_runtime_args`, `override_runtime_arguments`, `WorkloadDescriptor` returns zero hits.
  - `Pybind descriptor` — [groupnorm_nanobind.cpp](ttnn/cpp/ttnn/operations/normalization/groupnorm/groupnorm_nanobind.cpp) contains no `create_descriptor` binding and no `nb::class_` of the device operation.
  - `Op-owned tensors` — no `WorkloadDescriptor`, so structurally impossible; consistent with the sheet's blank cell.
  - **Factory-set match** — three factories in the code, three rows in the sheet, names matching one-to-one. No phantom and no missing row.
  - Cross-column invariants hold (a `descriptor` row with no op-owned tensors and no dynamic-runtime-args hook).

  One naming note, not a conflict: the recipe refers to the column `Override runtime args method? (PD and legacy)`; the sheet's current header reads `Override runtime args method? (PD only)`. Same column, same value (`no`).

- **Device 2.0 (every kernel used):** **GREEN.** All 16 kernel files are structurally Device 2.0:
  - Data movement is exclusively through the `Noc` object (`noc.async_read`, `noc.async_write_multicast`, `noc.async_read_barrier`, `noc.async_write_zeros`) with `UnicastEndpoint` / `MulticastEndpoint` destinations and `CoreLocalMem<uint32_t>` local pointers. A grep for `noc_async_read`, `noc_async_write`, `noc_semaphore_*`, `InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedPow2AddrGen*`, `get_noc_addr_from_bank_id` over `device/kernels/` returns **zero hits**.
  - Semaphores are `Semaphore<>` objects constructed from a semaphore id, e.g. [reader_mcast_sender_unary_sharded_gn_v2.cpp:111-113](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_sender_unary_sharded_gn_v2.cpp#L111-L113). No raw semaphore addresses.
  - CBs are `DataflowBuffer` objects; all FIFO and pointer operations go through methods (`dfb.reserve_back`, `dfb.push_back`, `dfb.wait_front`, `dfb.pop_front`, `dfb.get_write_ptr`, `dfb.get_read_ptr`). No `cb_reserve_back(cb_id, …)` / `get_write_ptr(cb_id)` free-function form anywhere.

  The only CB-index-keyed free functions in the op are ones the Metal 2.0 port itself rewrites, not Device 2.0 holdovers:
  - `get_tile_size(cb_id)` — explicitly sanctioned by the audit recipe.
  - `get_local_cb_interface(cb_id)` — explicitly sanctioned; one site, [groupnorm_zero_fill.hpp:38](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/groupnorm_zero_fill.hpp#L38).
  - `get_dataformat(cb_id)` — four sites (listed under *Misc anomalies*, since all four results are unused). This is **not** treated as a holdover: the port recipe's kernel-side whitelist rule 7 in `metal2_port.md` names `get_dataformat(cb_id)` in the same breath as `get_tile_size(cb_id)` as a port-time rewrite to the `DataflowBuffer` member getter, and the `cb_dfb_api_whitelist.md` §A lists `pack_dst_format` / `unpack_src_format` → `get_dataformat()`. See *Recipe notes* — the audit's sanctioned list should probably name it.
  - `get_tile_address(cb_id, tile_index)`, reached through the in-family donor `get_pointer_to_cb_data` — likewise a port-time rewrite (`cb_dfb_api_whitelist.md` maps `get_pointer_to_cb_data` → `get_tile_address`), not a Device 2.0 gap.

- **Feature compatibility:** every Appendix A entry scanned; none fires.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `global_circular_buffer` field on any `CBDescriptor`, no `remote_index` / `remote_cb_*` / `remote_circular_buffer.h`. All CBs are plain `CBDescriptor`s; the five buffer-backed ones set only `.buffer`. |
  | CBDescriptor `address_offset` (non-zero) | N/A | The field is never mentioned in the op. No `set_address_offset`, no `UpdateDynamicCircularBufferAddress` in any form, no `cb_descriptor_from_sharded_tensor` call. |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` type, no `CreateGlobalSemaphore`, no `global_semaphore.hpp` include. Semaphores are plain `SemaphoreDescriptor`s (two in the sharded and mcast factories, one in the no-mcast factory). |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` is a fixed set of named tensors (`input` plus five `std::optional<Tensor>`) — [groupnorm_device_operation_types.hpp:52-59](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_device_operation_types.hpp#L52-L59) — so the op-level cue does not fire. The decider is also clean: every `get_compile_time_arg_val(...)` in the op takes a literal index, and the rest of the compile-time args are read by name via `get_named_compile_time_arg_val("…")`. No kernel reads a compile-time arg at a runtime-varying index. |

- **CB endpoints (GATE-free):** see the per-factory census below. Every CB resolves to a plain 1P+1C, a self-loop, or (in specific configurations) a dead-CB drop. **No CB in this op requires the multi-binding advanced option.**

  The reason the census stays simple is the kernel layout: every factory places **exactly one reader, one writer and one compute kernel on each node**. Where the same kernel source appears in two `KernelDescriptor`s, the two cover *disjoint* core ranges — mcast-sender cores vs. mcast-receiver cores in the sharded and mcast factories, and `group_1` vs. `group_2` core sets in the no-mcast factory. That is the disjoint-node-set shape (each node sees one instance), **not** the dual-instance work-split, so no node ever has two instances of one kernel source touching a CB.

  I hunted specifically for the hidden second writer (a raw `get_write_ptr` co-fill by a kernel that is not the CB's FIFO producer, gated by a dedicated semaphore). There is one raw-write shape in the op, and it is **not** a co-fill: under `READER_REPACK && UNTILIZE_OUT` the reader kernels raw-write the borrowed output CB `c_16` via `dfb_out0.get_write_ptr()` (e.g. [reader_mcast_sender_unary_sharded_gn_v2.cpp:290](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_sender_unary_sharded_gn_v2.cpp#L290)). In exactly that configuration the compute kernel's untilize target is `c_12` / `c_31` (`repack_out`), not `c_16` — so `c_16` has one toucher, not two. The two semaphores in the op coordinate the mcast reduction, not a CB co-fill.

- **Offset base pointers:** **GREEN.** A grep for `->address()`, `.address()` and `(*buffer).address()` over the entire op directory returns **zero hits**. The op never computes a device address on the host at all: it hands `Buffer*` objects to `KernelDescriptor::RTArgList` and lets the framework inject the base. There is nothing for an offset to be folded into. Types 1 and 2 are therefore structurally absent; Type 3 (`address_offset`) is absent per Appendix A above; Type 4 (`ttnn::narrow`) does not appear. groupnorm is not listed in `2026-07-19_offset_base_pointers.md`, and the scan agrees with that silence.

- **TensorAccessor 3rd argument:** **GREEN — no sites.** Every one of the 21 `TensorAccessor` constructions in the op is the two-argument form `TensorAccessor(args, addr)`. The subject does not fire, so no Class 1/2/3/4/Special classification is needed. groupnorm is not listed in `2026-07-06_tensor_accessor_3rd_arg_triage.md`, consistent with the scan.

  (Note the distinct construct that *is* present and is unrelated to this subject: several kernels compute a per-tile byte count with `get_tile_size(dfb_id)` and pass it as the **transfer size** to `noc.async_read(...)`. That is a read length, not an accessor page-size override.)

## Port-work summary  *(mirrors the brief)*

### Tensor bindings (per binding, per factory)

The op declares one required and five optional input tensors plus one output. Classification differs between the sharded factory (where input and output are borrowed-memory CBs) and the two non-sharded factories (where they travel as `TensorAccessor` bases) — recorded per factory as the recipe's granularity rule requires.

| Binding | Sharded factory | Mcast / No-mcast factories |
|---|---|---|
| `input` | **clean** — borrowed-memory DFB. `CBDescriptor{.buffer = a.buffer()}` on `c_0` at [groupnorm_sharded_program_factory.cpp:837](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_sharded_program_factory.cpp#L837) and `:856`. The reader reads it through `dfb_in0.get_read_ptr()`. Port via `DataflowBufferSpec::borrowed_from`. | **Case 1** — `a.buffer()` pushed into the reader's `RTArgList` ([groupnorm_mcast_program_factory.cpp:1083](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_mcast_program_factory.cpp#L1083), `:1141`; [groupnorm_no_mcast_program_factory.cpp:1340](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_no_mcast_program_factory.cpp#L1340)), consumed as `TensorAccessor(src0_args, src_addr)` ([reader_mcast_sender_unary_gn.cpp:329](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_sender_unary_gn.cpp#L329)). |
| `output` | **clean** — borrowed-memory DFB on `c_16` ([groupnorm_sharded_program_factory.cpp:877](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_sharded_program_factory.cpp#L877)); when `inplace`, `c_16` is a second `CBFormatDescriptor` on the *input's* allocation (`:823-846`), so input and output share one borrowed buffer. | **Case 1**, bound by **two** kernels: the reader (`TensorAccessor(out_args, out_addr)`, [reader_mcast_sender_unary_gn.cpp:419](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_sender_unary_gn.cpp#L419)) and the writer ([writer_unary_gn_rm_gb.cpp:252](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/writer_unary_gn_rm_gb.cpp#L252)). |
| `gamma` | **Case 1** — writer RTA `Buffer*` ([groupnorm_sharded_program_factory.cpp:1235](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_sharded_program_factory.cpp#L1235)) → `TensorAccessor(gamma_args, gamma_addr)`. | **Case 1** — same shape (`:1183` mcast, `:1438` no-mcast). |
| `beta` | **Case 1** (`:1240`). | **Case 1** (`:1188` / `:1443`). |
| `input_mask` | **Case 1** (`:1245`). | **Case 1** (`:1193` / `:1448`). |
| `negative_mask` | **Case 1** (`:1250`) — sharded only. | not supported (`validate_on_program_cache_miss` requires sharded input). |
| `reciprocals` | not used by this factory. | **clean** — borrowed-memory DFB on `c_18` ([groupnorm_mcast_program_factory.cpp:1050](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_mcast_program_factory.cpp#L1050), [groupnorm_no_mcast_program_factory.cpp:1299](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_no_mcast_program_factory.cpp#L1299)). Read by the welford compute kernel via `get_pointer_to_cb_data`. |

**Every one of these is the `Buffer*`-binding form, not the silent-wrong `->address()` form.** The factories push the `Buffer*` object itself into `KernelDescriptor::RTArgList` / `emplace_runtime_args`, which the framework auto-registers as a `BufferBinding` and patches on cache hits. So this is routine port work with no correctness urgency: replace the `BufferBinding` with a typed `TensorParameter` / `TensorBinding` and let the kernel build `TensorAccessor(tensor::name)`.

**Optional-tensor shape the porter must handle.** When an optional tensor is absent the factory pushes the literal `0u` into the same RTA slot and appends a placeholder `TensorAccessorArgs()` — e.g. [groupnorm_sharded_program_factory.cpp:1234-1253](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_sharded_program_factory.cpp#L1234-L1253) and `:597-609`. The kernel then guards its use behind the `fuse_gamma` / `fuse_beta` compile-time flags and the `FUSE_NEGATIVE_MASK` define, so the null accessor is never constructed against. The port needs the binding to be optional in the same way (present-or-absent per program build), not a live binding carrying zero.

### TensorParameter relaxation

None. The sheet lists `none` for all three factories, and the op has no custom hash, so there is nothing to reconcile.

### TensorAccessor 3rd arg

None — no site in the op passes a third argument.

### CB endpoints

Per-node kernel set is always `{1 reader, 1 writer, 1 compute}`, so a CB's census is at most three touchers, and in practice at most two. Dispositions below are per `(CB, config)`.

**Sharded factory** — `all_cores` is the shard grid; kernels are `reader_mcast_{sender,receiver}_unary_sharded_gn_v2` (or the `welford_` variants), `writer_unary_sharded_gn_rm_gb_v2` (or `welford_`), and `groupnorm_sharded_v2` (or `welford_groupnorm_sharded_v2`).

| CB | Role | Touchers | Disposition |
|---|---|---|---|
| `c_0` `in0` (borrowed) | input | compute (tilize source / init); reader raw-reads it only under `READER_REPACK && TILIZE_IN` | **1P+1C** when both touch; **self-loop** otherwise |
| `c_29` | welford fp32 alias of `c_0`, only when `welford_fp32_alias` | compute only | **self-loop** |
| `c_1` `in` (tilized) | interm | compute produces (tilize dest) and consumes | **self-loop** |
| `c_31` | welford fp32 alias of `c_1` | compute only | **self-loop** |
| `c_2` scaler | | writer produces (`calculate_and_prepare_reduce_scaler`), compute consumes | **1P+1C** |
| `c_3` eps | | writer produces (`generate_bcast_col_scalar`), compute consumes | **1P+1C** |
| `c_4` scaler-global (non-welford only) | | writer always produces (the writer's `is_mcast_sender` CTA is hardcoded `1` for every core — [groupnorm_sharded_program_factory.cpp:572](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_sharded_program_factory.cpp#L572)); compute consumes only when `num_cores_per_mcast_group > 1` | mcast: **1P+1C** · single-core group: **self-loop** |
| `c_5` gamma / `c_6` beta / `c_7` input mask / `c_14` negative mask | | writer produces, compute consumes | **1P+1C** each |
| `c_11` repack / `c_12` repack_out (only when `reader_repack_output`) | | reader produces `c_11` & consumes `c_12`; compute consumes `c_11` (tilize source) & produces `c_12` (untilize dest) | **1P+1C** each |
| `c_13` `x` | interm | compute only | **self-loop** |
| `c_17` `ex2pe` | interm | compute only | **self-loop** |
| `c_26` ones | | writer produces, compute consumes | **1P+1C** |
| `c_30` out (only when `untilize_out && !negative_mask`) | interm | compute only | **self-loop** |
| `c_16` out (borrowed) | output | one toucher in every config — reader raw-write under `READER_REPACK && UNTILIZE_OUT`, else compute's untilize/pack target | **self-loop** |
| `c_8` `ex_partial` | reduce | compute produces (`reduce<…, REDUCE_SCALAR>`); reader consumes when mcasting; when `num_cores_per_mcast_group == 1` compute *also* consumes it, because `dfb_ex_global_id` aliases to `c_8` ([groupnorm_sharded_v2.cpp:83](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm_sharded_v2.cpp#L83)) | mcast: **1P+1C** · single-core group: **self-loop** |
| `c_9` `ex` + `c_15` `ex_global` (one descriptor, two format descriptors) | reduce | sender node, mcasting: compute produces `c_9`, reader consumes `c_9`; compute produces & consumes `c_15`. Receiver node: reader produces `c_15`, compute consumes; `c_9` untouched. Single-core group: **both untouched** | mcast sender: **1P+1C** (`c_9`) + **self-loop** (`c_15`) · mcast receiver: **1P+1C** (`c_15`), `c_9` unbound · single-core group: **dead — see below** |
| `c_10` `ex_external` (non-welford only) | reduce | reader produces, compute consumes — both under a `num_mcast_cores > 1` guard | mcast: **1P+1C** · single-core group: **dead — see below** |

**Mcast and no-mcast factories** — kernels are `reader_mcast_{sender,receiver}_unary_gn` (or `welford_`), `writer_unary_gn_rm_gb` (or `welford_`), and `groupnorm.cpp` (or `welford_groupnorm.cpp`). The no-mcast factory runs the *sender* reader on every core and has no receiver kernel.

| CB | Role | Touchers | Disposition |
|---|---|---|---|
| `c_0` `in0` | input | reader produces (or raw-reads under `READER_REPACK && TILIZE_IN`), compute consumes | **1P+1C** |
| `c_19` | welford fp32 alias of `c_0`, only when `welford_fp32_alias` | welford reader produces in lockstep, welford compute consumes | **1P+1C** |
| `c_29` `in` (tilized) | interm | compute produces & consumes; writer declares the id but does not touch it | **self-loop** |
| `c_2` scaler / `c_3` eps / `c_4` scaler-global | | writer produces, compute consumes (`c_4` only on `is_mcast_sender` cores) | **1P+1C** |
| `c_5` gamma / `c_6` beta / `c_28` input mask | | writer produces, compute consumes | **1P+1C** |
| `c_26` repack / `c_31` repack_out (only when `reader_repack_output`) | | reader ↔ compute, as in the sharded factory | **1P+1C** each |
| `c_24` `x`, `c_25` `xmm`, `c_23` `xmm2` / `reread_out`, `c_22` `xmm3` / `reread_write_out`, `c_27` `ex2pe` | interm | compute only (the writer consumes `c_22` when `!UNTILIZE_OUT && !gamma && !beta`) | **self-loop**, except `c_22` in that one config → **1P+1C** |
| `c_8` `ex_partial` (+ `c_21` `ex2_partial`, non-welford) | reduce | compute produces, reader consumes | **1P+1C** |
| `c_10` `ex_external` (non-welford) | reduce | reader produces, compute consumes | **1P+1C** |
| `c_15` `ex_global` (+ `c_14` `ex2_global`, non-welford) | reduce | compute produces on sender cores / reader produces on receiver cores; compute consumes | **self-loop** (sender) · **1P+1C** (receiver) |
| `c_9` `ex` (+ `c_13` `ex2`, non-welford) | reduce | both compute's push and the reader's consume are behind `num_cores_per_mcast_group > 1` / `num_mcast_cores > 1` | mcast: **1P+1C** · **no-mcast factory: dead — see below** |
| `c_18` reciprocals (borrowed, welford + reciprocals only) | input | compute reads via `get_pointer_to_cb_data` | **self-loop** |
| `c_16` out0 | output | `UNTILIZE_OUT && READER_REPACK`: reader raw-writes only. `UNTILIZE_OUT && !READER_REPACK`: compute untilizes into it. `!UNTILIZE_OUT && (gamma||beta)`: compute produces, writer consumes. `!UNTILIZE_OUT && !gamma && !beta`: **untouched — see below** | **self-loop** / **1P+1C** / **dead**, per config |
| `c_30` out (only when `untilize_out`) | interm | compute produces, writer consumes | **1P+1C** |

#### Dead-CB drop candidates (config-scoped) — the porter's most important CB item

A DFB with no producer and no consumer binding cannot be expressed in Metal 2.0 at all, so each of these must be dropped in the configuration where it is dead. All four were established by reading the compile-time guards, not by running the op — the porter should confirm each against the instantiation being built before dropping, and raise anything that does not reproduce.

1. **Sharded factory, `num_cores_per_mcast_group == 1`** (i.e. `use_mcast == false`: one core per batch *and* one core per group): `c_9` + `c_15` (one `CBDescriptor`, [groupnorm_sharded_program_factory.cpp:1066-1082](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_sharded_program_factory.cpp#L1066-L1082)) and `c_10` (`:1052-1063`) have **zero touchers**. The reader's whole mcast block is behind `if constexpr (num_mcast_cores > 1)` ([reader_mcast_sender_unary_sharded_gn_v2.cpp:157](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_sender_unary_sharded_gn_v2.cpp#L157)), compute's `c_9` push and `c_10` reduce are behind `is_mcast_sender and num_cores_per_mcast_group > 1` ([groupnorm_sharded_v2.cpp:286-298](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm_sharded_v2.cpp#L286-L298)), and `dfb_ex_global_id` aliases away to `c_8` in that config.

2. **Sharded factory, mcast receiver cores:** `c_9` is unbound on receiver nodes (the receiver reader never names it; compute's `c_9` block is `is_mcast_sender`-guarded). Because bindings are per `KernelSpec` and the receiver kernels sit on their own core range, this is expressible — the receiver's DFB set simply omits `c_9`'s buffer index. But `c_9` shares a `CBDescriptor` with `c_15`, so the porter must be deliberate about which buffer indices each side binds.

3. **No-mcast factory (always):** `c_9` `ex` and `c_13` `ex2` have zero touchers, for the same reason — every core is its own mcast group of size 1, so `num_mcast_cores > 1` is false everywhere ([reader_mcast_sender_unary_gn.cpp:450](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_sender_unary_gn.cpp#L450), [groupnorm.cpp:374-377](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm.cpp#L374-L377)). They are the second format descriptor of the `c_15` / `c_14` allocations, which stay live.

4. **Mcast and no-mcast factories, `!UNTILIZE_OUT && !gamma && !beta`:** the output CB `c_16` is untouched. Compute's `dfb_out_id` resolves to `c_22` (`reread_write_out`) in that branch ([groupnorm.cpp:171](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm.cpp#L171)) and the writer's `dfb_out_id` resolves to `c_22` as well ([writer_unary_gn_rm_gb.cpp:81](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/writer_unary_gn_rm_gb.cpp#L81)); the tensor write goes out of `c_22` through the output `TensorAccessor`. Nothing names `c_16`. This is the one I would most want a second pair of eyes on, because a dead *output* CB is counter-intuitive — see *Questions for the user*.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints — multi-binding shapes to watch:** none. No CB in this op has three distinct touchers on a node, and no CB has two kernels locked to the same FIFO role. The one raw-write shape (`dfb_out0.get_write_ptr()` in the readers) does not co-exist with a compute write to the same CB in the same configuration — verified above.

- **Cross-op / shared kernels:** the op instantiates **no** kernel file it does not own; every `KernelDescriptor::kernel_source` path is under `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/`. There is therefore no shared-kernel fork question and no sunset list. The coupling that does exist is by `#include` (function-call escape), inventoried in *Team-only* below. One entry is ⭐-flagged: `generate_bcast_col_scalar(CircularBuffer cb, uint32_t scalar)`.

- **RTA varargs (genuine, in five kernels):** the mcast reader kernels take the multicast group's per-core NoC coordinates as a **variable-count runtime-arg block**, read by pointer rather than by name:

  ```
  noc_coord_x = (tt_l1_ptr uint32_t*)(get_arg_addr(17));
  noc_coord_y = (tt_l1_ptr uint32_t*)(get_arg_addr(17 + num_mcast_cores));
  ```

  and then indexed in a loop bounded by `num_mcast_cores` (`noc_coord_x[i + 1]`). Sites:
  - [reader_mcast_sender_unary_sharded_gn_v2.cpp:82-107](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_sender_unary_sharded_gn_v2.cpp#L82-L107), consumed at `:199-209`
  - [welford_reader_mcast_sender_unary_sharded_gn_v2.cpp:76-101](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_reader_mcast_sender_unary_sharded_gn_v2.cpp#L76-L101)
  - [reader_mcast_sender_unary_gn.cpp:165-190](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_sender_unary_gn.cpp#L165-L190)
  - [welford_reader_mcast_sender_unary_gn.cpp:92-117](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_reader_mcast_sender_unary_gn.cpp#L92-L117)

  The count comes from the `num_mcast_cores` compile-time arg, which still varies across instantiations, so this is a genuine vararg per the recipe's shape (a) — reach for the kernel-side vararg mechanism, don't try to name the coordinates.

  **The scalars *before* the block are a separate matter, and they are nameable.** Their legacy positions shift (7…16, or 12…16, or nothing) depending on `has_mcast_first_group` / `has_mcast_last_group`, but each is a distinct field with a stable identity — `mcast_first_group_dest_noc_start_x` and friends. Metal 2.0 addresses named args in a section separate from the varargs, so the shifting legacy offset is irrelevant: name them. What varies is whether the *first-group* and *last-group* field sets are populated at all, which the host already signals through the two boolean args at positions 0 and 1.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ⚠ workable, with one ⭐ flag.**

- No borrowed kernel *files* — the op owns every kernel it instantiates, so no `_metal2` fork question and no cross-op sunset list.
- Function-call escapes are all into `tt_metal` LLK/HAL headers, the official shared kernel library, the second shared pool, or in-family normalization code.
- Every donor signature carries CB handles as `uint32_t` NTTPs or plain `uint32_t` ids — bridgeable by `dfb::name`'s constexpr cast — **except** `generate_bcast_col_scalar`, which takes a `CircularBuffer` by value. That is the ⭐ shape: op-by-op porting plus DFB-replaces-CB on the consumer side leaves no clean per-op story today.

| Op kernel(s) | Donor file | Bucket |
|---|---|---|
| all dataflow + compute kernels | `tt_metal/hw/inc/api/**` (`dataflow_api.h`, `noc.h`, `noc_semaphore.h`, `dataflow_buffer.h`, `circular_buffer.h`, `endpoints.h`, `core_local_mem.h`, `tensor/noc_traits.h`, `numeric/bfloat16.h`, `compute/*.h`), `hostdevcommon/common_values.hpp`, `noc_parameters.h` | 1 — LLK / HAL / firmware. No concern. |
| `compute/groupnorm.cpp`, `compute/groupnorm_sharded_v2.cpp`, `compute/welford_groupnorm.cpp`, `compute/welford_groupnorm_sharded_v2.cpp` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`, `untilize_helpers.hpp` | 2 — shared kernel library |
| `compute/groupnorm.cpp`, `compute/groupnorm_sharded_v2.cpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` | 2 — shared kernel library |
| `dataflow/writer_unary_gn_rm_gb.cpp`, `dataflow/writer_unary_sharded_gn_rm_gb_v2.cpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` | 2 — shared kernel library |
| all four writer kernels | `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` | 3 — second shared pool |
| `compute/welford_groupnorm.cpp` | `ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/memory.h` | 5 — in-family shared |
| `dataflow/reader_mcast_sender_unary_gn.cpp` | `device/kernels/dataflow/groupnorm_zero_fill.hpp` (own directory) | own |
| four welford readers | `device/kernels/dataflow/welford_combine.h` (own directory) | own |
| readers / compute (sharded + mcast) | `device/kernels/groupnorm_constants.hpp` (own directory) | own |

**Per-call detail** for the non-✓ entries:

| Donor function | Shape | Status |
|---|---|---|
| `generate_bcast_col_scalar(CircularBuffer cb, uint32_t scalar)` — `generate_bcast_scalar.hpp:13` | `CircularBuffer` by value | ⭐ **flag.** Called from all four writer kernels as `generate_bcast_col_scalar(CircularBuffer(eps_dfb_id), eps)` — [writer_unary_sharded_gn_rm_gb_v2.cpp:148](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/writer_unary_sharded_gn_rm_gb_v2.cpp#L148), [welford_writer_unary_sharded_gn_rm_gb_v2.cpp:68](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_writer_unary_sharded_gn_rm_gb_v2.cpp#L68), [writer_unary_gn_rm_gb.cpp:156](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/writer_unary_gn_rm_gb.cpp#L156), [welford_writer_unary_gn_rm_gb.cpp:106](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_writer_unary_gn_rm_gb.cpp#L106). The call site currently materialises a `CircularBuffer` from a raw id; in the port there is no id to materialise it from. Needs the cross-team story for `CircularBuffer&`-shaped donors. |
| `compute_kernel_lib::reduce<…, input_dfb_id, scaler_dfb_id, output_dfb_id>` / `tilize<…, input_dfb, output_dfb>` / `untilize<…>` / `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<dfb_id, …>` | `uint32_t` CB id as NTTP | ✓ OK — `dfb::name`'s constexpr cast covers template-parameter position. Note these helpers perform the CB's FIFO operations internally, which is why a naive grep of the compute kernels under-reports touchers. |
| `norm::kernel_util::compute::memory::get_pointer_to_cb_data<To>(uint32_t cb_id, uint32_t tile_index)` — `kernel_util/compute/memory.h:30` | `uint32_t cb_id` | ✓ OK to bridge. Called once, [welford_groupnorm.cpp:247](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/welford_groupnorm.cpp#L247), on the borrowed `reciprocals` DFB. Its body calls `get_tile_address(cb_id, tile_index)`, which the CB→DFB whitelist maps to the `DataflowBuffer` member getter — an in-family donor cleanup, not a blocker. |

### Relaxation candidates

None — the op has no custom hash to mine.

### TTNN factory analysis

- **Current concept:** `descriptor`, all three factories (three `create_descriptor` methods returning `ProgramDescriptor`).
- **Op-owned tensors:** none. No `WorkloadDescriptor`, no `buffers` vector.
- **MeshWorkload need:** none — single-program, `Execution Model = SPMD` per the sheet.
- **Pybind `create_descriptor` / other risky pybind:** none. `groupnorm_nanobind.cpp` binds `ttnn::group_norm` and the two program-config structs only.
- **Custom hash:** absent (default hash).
- **`get_dynamic_runtime_args` / `override_runtime_arguments`:** both absent.
- **Target concept:** `ProgramSpecFactoryConcept`, matching the sheet's `Porting Target` column.

## Misc anomalies  *(team-only, non-gating)*

1. **`bool block_wt_last` is almost certainly meant to be `uint32_t`.** All three factories declare it as `bool` and assign it a tile count:

   ```cpp
   bool block_wt_last = (per_core_Nt + num_groups_per_core - 1) / num_groups_per_core;
   ```

   [groupnorm_sharded_program_factory.cpp:226](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_sharded_program_factory.cpp#L226), [groupnorm_mcast_program_factory.cpp:195](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_mcast_program_factory.cpp#L195), [groupnorm_no_mcast_program_factory.cpp:206](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_no_mcast_program_factory.cpp#L206). The value is then handed to kernels as the `block_w_last` compile-time arg and used in tile arithmetic — `index_g_offset == (per_core_N - block_w_last) ? block_w_last : block_w` ([writer_unary_gn_rm_gb.cpp:250](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/writer_unary_gn_rm_gb.cpp#L250), [groupnorm_sharded_v2.cpp:450](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm_sharded_v2.cpp#L450), and three more sites). The `bool` collapses every non-zero tile count to `1`. Either the kernels are relying on the collapsed value (in which case the type and the name are both misleading) or a real tile count is being lost. Worth an ops-team look; the port must carry the value through unchanged either way.

2. **Four dead `get_dataformat(...)` locals.** `const DataFormat data_format = get_dataformat(dfb_ex_partial_id);` at [reader_mcast_sender_unary_sharded_gn_v2.cpp:132](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_sender_unary_sharded_gn_v2.cpp#L132) and [reader_mcast_receiver_unary_sharded_gn_v2.cpp:49](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_receiver_unary_sharded_gn_v2.cpp#L49); `const DataFormat out_data_format = get_dataformat(dfb_out0_id);` at [reader_mcast_sender_unary_gn.cpp:222](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_sender_unary_gn.cpp#L222) and [reader_mcast_receiver_unary_gn.cpp:140](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_receiver_unary_gn.cpp#L140). None of the four variables is read anywhere in its file. Deleting them is cheaper than porting them to the DFB getter.

3. **`my_x[0]` / `my_y[0]` hardcodes NoC index 0 for self-addressed local transfers.** Every self-read/self-write in the op supplies its own core's coordinates as `{.noc_x = my_x[0], .noc_y = my_y[0], …}` — 14 sites across the readers and writers. But the kernels run on `preferred_noc_for_dram_read` / `preferred_noc_for_dram_write` ([groupnorm_sharded_program_factory.cpp:524-525](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_sharded_program_factory.cpp#L524-L525)), which is not necessarily NOC 0, and `my_x` / `my_y` are per-NoC coordinate arrays. One sibling kernel does it the other way — `{.noc_x = my_x[noc.get_noc_id()], .noc_y = my_y[noc.get_noc_id()], …}` at [welford_reader_mcast_sender_unary_sharded_gn_v2.cpp:142](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_reader_mcast_sender_unary_sharded_gn_v2.cpp#L142) and `:302`. My guess is the `my_x[0]` form is only working because the coordinate spaces happen to coincide for the self case on current silicon, but the inconsistency inside one op is worth resolving. Not porter work — the port copies these lines verbatim.

4. **Dead compile-time arg in the sharded writer.** The sharded factory always appends a `page_size`-ish value at compile-time-arg index 10 ([groupnorm_sharded_program_factory.cpp:583-591](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_sharded_program_factory.cpp#L583-L591)), but both sharded writer kernels skip it — `// compile_time_arg 10: size (unused here)` ([writer_unary_sharded_gn_rm_gb_v2.cpp:43](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/writer_unary_sharded_gn_rm_gb_v2.cpp#L43)). The host still computes `gamma.value().padded_shape()[3] * gamma.value().element_size()` to produce it.

5. **`packer_l1_acc` is destructured from the compute-kernel config in all three factories and never used** (e.g. [groupnorm_sharded_program_factory.cpp:713](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_sharded_program_factory.cpp#L713)). The `ComputeConfigDescriptor` the factories build does not carry it, so a caller setting `packer_l1_acc` gets it silently ignored — while it still participates in the operation-attributes hash through `compute_kernel_config`, so it does change the cache key. That combination (hashed but ignored) is the shape the recipe asks to flag.

## Questions for the user

1. **Is the output CB `c_16` genuinely unused in the non-untilize, no-gamma, no-beta path?** In the mcast and no-mcast factories, `c_16` is allocated unconditionally ([groupnorm_mcast_program_factory.cpp:768-776](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_mcast_program_factory.cpp#L768-L776)), but when `!UNTILIZE_OUT && !fuse_gamma && !fuse_beta` both the compute kernel and the writer resolve their `dfb_out_id` to `c_22` instead ([groupnorm.cpp:171](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm.cpp#L171), [writer_unary_gn_rm_gb.cpp:81](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/writer_unary_gn_rm_gb.cpp#L81)), and I can find no other reference to `c_16` in that configuration. A dead *output* CB is unusual enough that I would rather confirm it than have the porter drop a live allocation. If it is dead, it is also wasted SRAM today.

2. **Confirm the single-core-mcast-group configurations are actually reachable for the sharded factory.** The dead-CB analysis for `c_9` / `c_15` / `c_10` hinges on `use_mcast == false` (`num_cores_per_batch == 1 && num_cores_per_group == 1`, [groupnorm_sharded_program_factory.cpp:359](ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_sharded_program_factory.cpp#L359)). If model configurations never hit that branch, the porter has fewer cases to build; if they do, the three CBs must be dropped there.

## Recipe notes

1. **The sanctioned free-function list should probably name `get_dataformat(cb_id)`.** The *Device 2.0 prerequisite* subject in `metal2_audit.md` enumerates exactly two sanctioned CB-index free functions — `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)` — and tells the auditor that anything else with a wrapper-method replacement and the wrapper in scope is an isolated holdover that **gates the port**. `get_dataformat(cb_id)` matches that description exactly: `DataflowBuffer::get_dataformat()` exists, `CircularBuffer::get_dataformat()` just forwards to the free function (the same forwarding relationship the recipe cites as its reason for sanctioning `get_tile_size`), and the wrapper object is in scope at all four sites in this op. Read literally, that would RED this whole op on four dead lines.

   I resolved it GREEN only after opening the port recipe's kernel-side whitelist rule 7, which names `get_dataformat(cb_id)` alongside `get_tile_size(cb_id)` as a **port-time** rewrite to the DFB member getter — so it plainly belongs on the Device-2.0-sanctioned side of the line. Two suggestions: (a) add `get_dataformat(cb_id)` to the sanctioned list, or better, (b) replace the hardcoded two-item list with a pointer to `cb_dfb_api_whitelist.md` §A/§B, which is the authoritative "these are rewritten at port time, not at Device 2.0 time" table and already covers `get_tile_address` and `get_pointer_to_cb_data` too. As written, the audit doc leads a careful auditor to a false RED, and resolving it required reading a document the audit tells you not to pre-load.

2. **The census under-reports when FIFO operations hide inside shared-library helper templates.** The recognition signals for *CB endpoints* subject in `metal2_audit.md` are all written as direct call shapes (`reserve_back`/`push_back`, `wait_front`/`pop_front`, `get_write_ptr`). In this op the compute kernels perform many of their CB operations *only* through `compute_kernel_lib::reduce<…, in_dfb, scaler_dfb, out_dfb>`, `tilize<…, in_dfb, out_dfb>` and `untilize<…>`, which take the DFB ids as template parameters and do the FIFO work internally. A grep-driven census reads those CBs as having one fewer toucher than they do — for example `c_8` in the sharded factory looks like a consumer-only CB until you notice `reduce<…, dfb_ex_partial_id>` is its producer. The "hunt for the hidden second writer" guidance covers semaphore-gated raw co-fills but not helper-mediated FIFO ops; a sentence pointing at NTTP-carried CB ids in `kernel_lib` helpers would help the next auditor.

3. **A "config-scoped dead CB" is common enough here to deserve its own framing.** The *Dead CB* section says a dead CB should be "exceedingly rare" and pushes hard toward assuming an analysis gap. In this op I found four, and each is dead only under a specific configuration while being live under others — they come from the factory allocating a CB unconditionally while the kernels gate its use behind `num_mcast_cores > 1` or a `do_gamma`/`do_beta`/`UNTILIZE_OUT` branch. That is not really "a CB nobody touches" (the flagrant-waste case the section describes); it is "an allocation the config specialisation outgrew". The distinction matters for the disposition: the porter drops the DFB *in that program build only*, and the same index must still be bound in sibling builds. The recipe's `Classify per instantiation` line implies this, but the Dead-CB section itself reads as if deadness were a property of the op.

4. **Minor: the readiness sheet's column header has drifted from the recipe.** The recipe names `Override runtime args method? (PD and legacy)`; the sheet currently reads `Override runtime args method? (PD only)`. Reading by header name still finds it, but a reader following the recipe verbatim will not match on the parenthetical.
