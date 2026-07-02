# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/normalization/layernorm/`

Single device-operation directory:

- **`LayerNormDeviceOperation`** (`device/layernorm_device_operation.{hpp,cpp}`)
  - `LayerNormMultiCoreProgramFactory` (`device/layernorm_op_multi_core.cpp`) — interleaved / non-sharded path
  - `LayerNormShardedProgramFactory` (`device/layernorm_op_multi_core_sharded.cpp`, + helpers in `device/sharded_layernorm_factory_helpers.{hpp,cpp}`) — sharded path (incl. pre/post all-gather, welford)

`select_program_factory` dispatches to the sharded factory when the input is sharded, else the multi-core factory. The two factories share the device-op class, the kernel-naming scheme, and the in-family kernel pool — single combined report, with per-factory attribution where findings differ.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/normalization/layernorm/` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `LayerNormDeviceOperation` → `LayerNormMultiCoreProgramFactory`, `LayerNormShardedProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (own + donor) | Yes (isolated `get_tile_size(cb_id)` holdovers; route to Device 2.0 track, not the port) |
| *Prereqs* — Cross-op escapes | Ok (shared-lib + in-family only; no cross-family donor) |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — Port Type | `ProgramSpecFactoryConcept` / `MaximizeCacheReuse` → Option 2 |
| *TTNN Readiness* — TTNN infra++ | none (no op-owned tensors; recip LUT is caller-passed) |

Port Type → Option map: **1** = `MinimizeCacheHitCost`; **2** = `MaximizeCacheReuse` basic; **3** = `MaximizeCacheReuse` advanced.

## Result

**GREEN → brief issued.** Both factories are on the `ProgramDescriptor` API; every kernel the op exercises (own + shared-library + in-family) is Device 2.0 compliant; no UNSUPPORTED Appendix-A feature is in use; the factory lands on the implemented `ProgramSpecFactoryConcept` with the default `MaximizeCacheReuse` strategy. Port work is the routine tensor-binding re-expression (buffer-address RTAs → `TensorParameter`/`TensorBinding`) plus translating two LANDED constructs (borrowed-memory DFBs and welford-fp32 aliased CBs). No custom `compute_program_hash` to delete. No subset scoping needed — both code paths clear.

## Gate detail

- **ProgramDescriptor:** GREEN. Both factories populate a `tt::tt_metal::ProgramDescriptor` via `create_descriptor` and use `KernelDescriptor` / `CBDescriptor` / `CBFormatDescriptor` / `SemaphoreDescriptor` (e.g. `layernorm_op_multi_core.cpp:617` onward; `sharded_layernorm_factory_helpers.cpp:782` `add_kernel_descriptors`, `:974` `add_cb_descriptors`). No imperative `host_api.hpp` builder calls (`CreateProgram` / `CreateKernel` / `CreateCircularBuffer` / `SetRuntimeArgs`) anywhere in the op. `<tt-metalium/host_api.hpp>` is included but only for free helpers (`split_work_to_cores`, `pack_two_bfloat16_into_uint32`, `corerange_to_cores`), not the builder.

- **Device 2.0 (every kernel used):** YELLOW within GREEN — *substantively compliant with isolated CB-index-keyed holdovers*. Every dataflow kernel the op instantiates uses the Device 2.0 wrappers consistently: `Noc`, `CircularBuffer`, `Semaphore<>`, `UnicastEndpoint`, `TensorAccessor`, `noc.async_read` / `noc.async_write` / `noc.async_write_multicast`, `sem.up/.wait/.wait_min/.set_multicast`. No Device 1.0 idioms found anywhere — no `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedAddrGenFast` / `InterleavedPow2AddrGen*`, no `get_noc_addr_from_bank_id`, no raw `noc_async_read/write`, no `get_semaphore` / raw `noc_semaphore_*`, no manual CB index management.

  The one residual Device 2.0 cleanup is the `get_tile_size(cb_id)` free function (CB-index-keyed family) used to fetch tile byte sizes. In every site a `CircularBuffer` wrapper for that CB is already in scope, so the mechanical fix is `get_tile_size(cb_id)` → `cb_obj.get_tile_size()` (or pass the wrapper). This does **not** structurally block the port — the tokens attach to the in-scope wrappers — but it is **out of port scope** (kernel-side whitelist forbids the port absorbing Device 2.0 cleanup). Route to the Device 2.0 track.

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `device/kernels/dataflow/reader_unary_interleaved_ln.cpp` | 108, 114, 118, 122 | `get_tile_size(cb_id_in0/gamma/beta/in1)` | `cb_in0`, `cb_gamma`, `cb_beta`, `cb_in1` |
  | `device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb.cpp` | 45, 59, 63, 67 | `get_tile_size(cb_id_*)` | `cb_id_*` wrappers in scope |
  | `device/kernels/dataflow/reader_unary_interleaved_ln_large_tensor.cpp` | 101, 107, 111, 115 | `get_tile_size(cb_id_*)` | wrappers in scope |
  | `device/kernels/dataflow/reader_unary_interleaved_ln_large_tensor_welford.cpp` | 48, 59, 63, 67 | `get_tile_size(cb_id_*)` | wrappers in scope |
  | `device/kernels/dataflow/writer_unary_interleaved_start_id_blocked.cpp` | 25 | `get_tile_size(cb_id_out0)` | `cb_out0` (`:28`) |
  | `device/kernels/dataflow/writer_unary_sharded_ln.cpp` | 53, 79, 93 | `get_tile_size(cb_out/gamma/beta)` | `cb_out_obj`, `cb_gamma_obj`, `cb_beta_obj` (`:48-51`) |
  | `device/kernels/dataflow/writer_unary_sharded_ln_rm_gb.cpp` | 56, 82, 113 | `get_tile_size(cb_out/gamma/beta)` | wrappers in scope |
  | `device/kernels/dataflow/reader_mcast_sender_unary_sharded_ln.cpp` | 93 | `get_tile_size(rms_norm ? cb_ex_partial2 : cb_ex_partial)` | `cb_partial_obj` (`:112`) |
  | `device/kernels/dataflow/reader_mcast_receiver_unary_sharded_ln.cpp` | 88 | `get_tile_size(...)` | `cb_partial_obj` (`:128`) |
  | `device/kernels/dataflow/reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp` | 57 | `get_tile_size(cb_ex_partial2)` | `cb_partial_obj` (`:88`) |
  | `device/kernels/dataflow/reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp` | 54 | `get_tile_size(cb_ex_partial2)` | `cb_partial_obj` (`:124`) |
  | `device/kernels/dataflow/reshard_writer.hpp` | 23 | `get_tile_size(cb_out.get_cb_id())` | `cb_out` (param ref) — id derived from wrapper |

  (Compute kernels — `layernorm.cpp`, `layernorm_welford.cpp`, `layernorm_large_tensor*.cpp`, `layernorm_sharded*.cpp` — use `cb_wait_front` / `cb_pop_front` / `cb_reserve_back` / `cb_push_back` free functions, but compute kernels are **out of Device 2.0 data-movement scope** and out of TensorAccessor scope: they consume from / produce to CBs, not tensor memory.)

- **Feature compatibility:** every Appendix A entry, in order.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `experimental::GlobalCircularBuffer`, no `CBDescriptor::global_circular_buffer` field set anywhere. |
  | Dynamic CircularBuffer (borrowed memory) | GREEN | `CBDescriptor::buffer` set on input/residual/stats/recip/output CBs (both factories) — port uses `borrowed_from`. See heads-up below. |
  | CBDescriptor `address_offset` (non-zero) | N/A | `.address_offset` never set (default 0) on any `CBDescriptor`; no `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor` calls. |
  | Aliased Circular Buffers | GREEN | Multi-element `format_descriptors` on CB 0 / 18 / 19 / 23 (multi-core) and CB 0 / 24 (sharded) under the welford-fp32-alias path — port uses `advanced_options.alias_with`. See heads-up below. |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` type, no `experimental::CreateGlobalSemaphore`. Only plain program-scope `SemaphoreDescriptor`s (sharded factory). |
  | Non-zero semaphore initial value | N/A | All three `SemaphoreDescriptor`s use `.initial_value = 0` (`sharded_layernorm_factory_helpers.cpp` is not the site; `layernorm_op_multi_core_sharded.cpp:219,224,229`). Multi-core factory uses no semaphores. |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | No `ArgConfig::Runtime*` token; all `TensorAccessorArgs(buffer)` are the single-argument static form. |
  | `UpdateCircularBuffer*` | N/A | No `UpdateCircularBufferTotalSize` / `UpdateCircularBufferPageSize` / `UpdateDynamicCircularBufferAddressAndTotalSize`. The factories rebuild the `ProgramDescriptor` per dispatch; no per-execution CB mutation hooks. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` (`LayerNormInputs`) is a fixed tuple of named optional tensors — no `std::vector<Tensor>`. No kernel loops over `get_compile_time_arg_val(i)` with a runtime-varying `i`. |

  No UNSUPPORTED entry fires. Feature compatibility subject verdict: GREEN.

- **Factory concept:** GREEN — `ProgramSpecFactoryConcept`, caching strategy `MaximizeCacheReuse` (default). See the TTNN ProgramFactory section below for the decision-tree path.

## Port-work summary  *(mirrors the brief)*

- **Factory concept:** `ProgramSpecFactoryConcept` · caching strategy `MaximizeCacheReuse` (Option 2). 1:1 with the legacy single-program shape.
- **Tensor bindings** (per binding — Case 1 for buffer-address RTAs, clean for borrowed-memory DFBs):
  - **input (a):** Multi-core factory — **Case 1** (re-express). `a_addr = a.buffer()->address()` (`layernorm_op_multi_core.cpp:187`) pushed into reader RTA arg[0] (`:587`), consumed kernel-side as `TensorAccessor(src0_args, src_addr)` (`reader_unary_interleaved_ln.cpp:51,111`). Sharded factory — **clean** (borrowed-memory DFB: CB 0 `.buffer = a_buffer`, `sharded_layernorm_factory_helpers.cpp:1006`; causal-link gate).
  - **residual (b):** Multi-core — **Case 1** (`b_dram_addr` `:188` → reader arg[8] `:595`). Sharded — **clean** (CB 1 `.buffer = b_buffer`, `:1024`).
  - **gamma:** Both factories — **Case 1**. Multi-core: `gamma_dram_addr` (`:189`) → reader arg[6]. Sharded: `gamma_dram_addr` (`layernorm_op_multi_core_sharded.cpp:157`) → writer RTA (`sharded_layernorm_factory_helpers.cpp:1501`), consumed as `TensorAccessor(gamma_args, gamma_addr)` (`writer_unary_sharded_ln.cpp:80`).
  - **beta:** Both factories — **Case 1** (symmetric to gamma; reader arg[7] multi-core / writer RTA sharded).
  - **stats:** Sharded post-all-gather only — **clean** (CB 7 `.buffer = stats_buffer`, `:1198`).
  - **recip LUT:** Both factories (welford only) — **clean** (CB 25 `.buffer = recip_buffer`; `layernorm_op_multi_core.cpp:832`, `sharded_layernorm_factory_helpers.cpp:1185`). Caller-passed tensor, not factory-allocated.
  - **output:** Multi-core — **Case 1** (`dst_addr = output.buffer()->address()` `:131` → writer arg[0] `:604`, consumed as `TensorAccessor(dst_args, dst_addr)` `writer_unary_interleaved_start_id_blocked.cpp:30`). Sharded — **clean** (CB 16 / CB 17 `.buffer = output_buffer/output_reshard_buffer`, `:1226-1245`).
  - All non-clean bindings are **Case 1**; **no Case 2** — every kernel access is plain page-by-page `TensorAccessor` iteration, no exotic NoC walk or sub-page arithmetic.
- **Custom hash:** none. `LayerNormDeviceOperation` defines no `compute_program_hash` member/override (uses the default reflection-based hash). The `compute_program_hash` static at `layernorm_nanobind.cpp:253` is a Python *test hook* that calls the framework default `ttnn::device_operation::detail::compute_program_hash<...>` — not a custom override; nothing to delete.

## Heads-ups  *(mirrors the brief)*

- **Notable LANDED constructs:**
  - **Borrowed-memory DFB** (Dynamic CircularBuffer): the sharded factory places CBs directly on input/residual/stats/recip/output `Buffer` memory via `CBDescriptor::buffer`; the multi-core factory does the same for the recip LUT (CB 25). Port translation: `DataflowBufferSpec::borrowed_from = <tensor_parameter_name>`. Sites: `sharded_layernorm_factory_helpers.cpp:1006,1024,1032,1185,1198,1226,1245`; `layernorm_op_multi_core.cpp:832`.
  - **Aliased CBs** (welford-fp32 path only): a single `CBDescriptor` carries two `CBFormatDescriptor`s sharing backing SRAM — the primary index plus a `UnpackToDestFp32` alias index (c_29/c_30/c_31). Port translation: one `DataflowBufferSpec` per index, mutually declared via `advanced_options.alias_with`; do **not** split into independent DFBs (shared L1 address is load-bearing). Sites: `layernorm_op_multi_core.cpp:688-693` (c_0→c_29), `:709-714` (c_18→c_30), `:734-739` (c_19→c_31), `:810-815` (c_23→c_29); `sharded_layernorm_factory_helpers.cpp:1007-1012` (c_0→c_29), `:1066-1071` (c_24→c_29). The host also sets matching `unpack_to_dest_mode[c_29/c_30/c_31] = UnpackToDestFp32` on the compute `ComputeConfigDescriptor`.
  - Non-zero sem init / dynamic TA: not present.
- **Cross-op / shared kernels:** see Team-only. No cross-family donor; the in-family `kernel_util/` + `kernel_lib/` + `kernel/` headers form the port-together set. File-path borrows: the multi-core writers `writer_unary_interleaved_start_id_blocked*.cpp` and all kernels are layernorm-owned (in-family), so the file-path coupling is intra-family.
- **RTA varargs:** none. The mcast readers consume NoC-coordinate arrays via `args.insert(... mcast_noc_x ...)` host-side, but the kernels index them with fixed/derived offsets, not a runtime-varying `get_arg_val(i)` loop. (Note: `writer_unary_sharded_ln.cpp:38` reads a variable-length write-back `segment_args` block via `get_arg_addr(9)` + a counted loop — this is a runtime-known segment count, an RTA-vararg-shaped read. It is supported in Metal 2.0; port can keep the counted L1 read or move to the named-RTA endpoint. Non-gating.)

## Team-only

- **TensorAccessor convertibility:** N/A — no Case-2 bindings. All non-clean bindings are Case 1 (convertible by construction).
- **Out-of-directory coupling & donor shape:**
  - **Op-level roll-up: ✓ clean.** All escapes are to shared-lib pools or in-family headers; no cross-family donor, no pre-Device-2.0 donor, no `CircularBuffer&`/`Semaphore`-by-value donor signature that doesn't already cross cleanly.
  - **Donor classes consumed:**
    - `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp`, `reduce_helpers_compute.hpp` — official shared kernel library (class 2; lib team). Device 2.0 clean (no D1.0 idioms).
    - `ttnn/kernel/dataflow/generate_bcast_scalar.hpp` — second shared pool (class 3). Device 2.0 clean.
    - `ttnn/operations/normalization/kernel_util/generic/blocked_range.h`, `.../generic/bit.h`, `.../compute/{memory,numeric,combine_welford}.h` — in-family shared (normalization family). Compute/iteration helpers; not data-movement donors.
    - `layernorm_dataflow_utils.h`, `reshard_writer.hpp`, `layernorm_compute_utils.h` — in-family (layernorm-owned). `reshard_writer.hpp` and `layernorm_dataflow_utils.h` take `Noc&` / `CircularBuffer&` — Shape ✓ excellent (Device 2.0 native).
  - **Per-call shape detail:** the data-movement helpers consumed (`reduce_helpers_dataflow.hpp` reduce-scaler prep, `generate_bcast_scalar`, `layernorm_dataflow_utils::read_block_to_cb` / `push_row_major_blocks_to_cb`, `write_resharded_data`) all take Device 2.0 handles (`Noc&`, `CircularBuffer&`) or CB-index template params (`uint32_t cb_id` — ✓ OK, `dfb::name` constexpr cast handles). No donor takes a raw `uint32_t sem_addr` / `uint64_t` NOC-encoded sem, no `TensorAccessorArgs<N>`-by-value or NTTP-CTA-offset shape, no old-style addr-gen. Nothing sequence-blocks.
  - **Borrowed kernel files (file-path instantiation):** all kernel `.cpp` the factories `CreateKernel`/`kernel_source` are under `layernorm/device/kernels/` (layernorm-owned). No external-pool kernel `.cpp` is instantiated. Port-together set = the layernorm op itself.
- **Relaxation candidates:** none mined — no custom hash to read. (Forward-looking, fallible, for the team only: `validate_on_program_cache_miss` enforces `padded_shape` matching between input and residual and gamma/beta padded-width equality; a future `match_padded_shape_only` relaxation might be safe for some bindings, but this is unverified and the default remains strict.)

## Misc anomalies  *(team-only, non-gating, not porter-actionable)*

- `reader_unary_interleaved_ln.cpp:33` — `arg[4] = packed_one_value` is documented as "legacy; unused, scaler is generated in-kernel." The host still computes and passes `packed_one_value` (`layernorm_op_multi_core.cpp:553-554,591`). Dead RTA on the reader; route to op owner, not the port.
- `layernorm_op_multi_core.cpp:206` — local `std::optional<Tensor> recip_tensor` shadows nothing harmful but is initialized to `nullopt` then reassigned from `tensor_args.recip_tensor` only under `use_welford`; harmless, noted for readers.

## Per-DeviceOperation attribution

Single DeviceOperation (`LayerNormDeviceOperation`); per-factory differences are attributed inline above. Summary of where the two factories diverge:

| Aspect | `LayerNormMultiCoreProgramFactory` (interleaved) | `LayerNormShardedProgramFactory` (sharded) |
|---|---|---|
| input/residual/output binding | **Case 1** (buffer-addr RTA → `TensorAccessor`) | **clean** (borrowed-memory DFB) |
| gamma/beta binding | **Case 1** | **Case 1** |
| stats binding | n/a | **clean** (borrowed-memory DFB; post-all-gather only) |
| recip LUT | clean (borrowed; welford) | clean (borrowed; welford) |
| semaphores | none | 3 × program-scope `SemaphoreDescriptor`, `initial_value=0` |
| aliased CBs | c_0/c_18/c_19/c_23 → c_29/c_30/c_31 | c_0/c_24 → c_29 |
| factory concept | `ProgramSpecFactoryConcept` / `MaximizeCacheReuse` | `ProgramSpecFactoryConcept` / `MaximizeCacheReuse` |

Both factories clear all gates independently — the GREEN is not subset-scoped.

## TTNN ProgramFactory

### Concept chosen
`ProgramSpecFactoryConcept`

### Path through the decision tree
- **Decision 1 (single vs multi-program):** single-program. Each factory's `create_descriptor` returns exactly one `tt::tt_metal::ProgramDescriptor`; neither constructs a `MeshWorkload`. → `ProgramSpecFactoryConcept`.
- **Decision 2 (op-owned resources):** none. Neither factory allocates scratch `MeshTensor`s or `GlobalSemaphore`s in its body. The welford reciprocal LUT is a *caller-passed* `tensor_args.recip_tensor` (threaded through `layer_norm(...)` → `LayerNormInputs`), not factory-created; the output tensor is produced by `create_output_tensors`. The sharded factory's semaphores are plain program-scope `SemaphoreDescriptor`s. → `op_owned_tensors` empty.
- **Decision 3 (tensor-arg relaxations):** strict (default). No deviation during port.
- **Decision 4 (advanced shape):** not needed. No heavy immutable-extraction work that re-running per dispatch would make unacceptable; the factories compute sizing/grid distribution from tensor metadata each call, which the basic shape re-runs cheaply under `MaximizeCacheReuse`.

### Caching strategy
`MaximizeCacheReuse` (default — no op-owned resources, so no `MinimizeCacheHitCost` opt-in required).

### Legacy-to-Metal-2.0 shape change
1:1 with legacy. Single `ProgramDescriptor` per factory maps directly onto a single `ProgramSpec`; no legacy workaround to unwind.

### Stop signals
None.

## Recipe notes

- **Device 2.0 YELLOW-vs-GREEN tier reporting in the status-summary table.** The audit's status-summary row `*Prereqs* — Device 2.0` offers only `Yes / Partial / No`, but the prose tier this op lands in is the explicit *YELLOW carve-out* (substantively compliant, isolated CB-index holdovers, routed to the Device 2.0 track and explicitly **not** port-blocking). "Partial" overstates the blocker (it reads like the RED "broadly uses D1.0 idioms" case); "Yes" understates the residual cleanup. I marked it `Yes (isolated ... holdovers; route to Device 2.0 track)` to preserve the carve-out's meaning, but a dedicated `Yes-with-holdovers` / explicit YELLOW cell value would map the spreadsheet column 1:1 to the three-way Device 2.0 gate outcome the prose defines. Cite: gate table "Device 2.0 (own + donor) | Yes / Partial / No" vs §Prerequisites Check 2 YELLOW.

- **`get_tile_size(cb_id)` is in the named CB-index-keyed family but the migration guide retains it in migrated code.** §Prerequisites Check 2 lists `get_tile_size(cb_id)` as a holdover ("1-line mechanical replacement … `cb_obj.get_read_ptr()`"). But the Device 2.0 migration guide's own *migrated* Example 1 (`device_api_migration_guide.md:551`) keeps `uint32_t tile_size = get_tile_size(cb_id);` verbatim alongside the `CircularBuffer cb(cb_id)` wrapper — i.e. the guide does not show a `cb.get_tile_size()` member form at all. So an auditor cannot tell from the migration guide whether `get_tile_size(cb_id)` is a *true* holdover (with a member-form replacement) or an accepted free function with no wrapper equivalent. I treated it as a holdover per the audit's explicit naming, but the two docs disagree on whether `get_tile_size` has a member form — worth reconciling so the routing (Device 2.0 track vs "leave it") is unambiguous.

- **Borrowed-memory DFB vs Case-1 split across factories for the *same* logical tensor was easy to mis-handle.** The input/residual/output tensors are **clean borrowed-memory DFBs** on the sharded path but **Case-1 buffer-address-RTA** bindings on the interleaved path — the same `TensorParameter` name, opposite classifications, depending on factory. The per-binding-not-per-op rule plus the causal-link gate together resolve this correctly, but the report template's flat per-binding bullet list has no obvious slot for "this binding is clean in factory A and Case 1 in factory B." I used the Per-DeviceOperation attribution table to disambiguate; a note in the TensorAccessor-handling subject that per-binding classification can be *per-factory* within one bundled op would help the next auditor.
