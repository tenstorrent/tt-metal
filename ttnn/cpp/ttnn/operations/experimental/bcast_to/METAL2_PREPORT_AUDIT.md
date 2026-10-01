# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/bcast_to`

- **`BcastToOperation`** (`device/bcast_to_device_operation.hpp:22`)
  - *Direct descriptor* — `create_descriptor` is a static member of the device-op itself (`device/bcast_to_device_operation.hpp:40`, defined at `device/bcast_to_program_factory.cpp:135`). There is no `program_factory_t` and no factory struct; the framework accepts it through the `HasDirectDescriptor` shim (`ttnn/api/ttnn/operation_concepts.hpp:158`).
  - Kernels are picked per `SubtileBroadcastType` (`NONE` / `ROW` / `COL` / `SCALAR`) through `BcastToKernelConfig` (`device/bcast_to_utils.cpp:278-304`). Each config uses one reader, one writer and one compute kernel, all owned by the op (`device/kernels/{dataflow,compute}/*_bcast_to.cpp`). The op has no unreferenced kernel files.

The direct-descriptor `create_descriptor` replaced the legacy `BcastToTileFactory` (`create()` + `override_runtime_arguments()`) in `f5093e705ae` "[Cleanup] Port More Ops to PD (#57409)", committed 2026-09-25.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
*(The recipe tree isn't in this checkout (`Metal_Ports`, branch `edwinlee/PD_Metal_Ports`). The hash comes from the `Port_Recipe` checkout (branch `akertesz/op-porting-recipe`). Its `ai/audit/metal2_audit.md` is byte-identical to the `/localdev/edwinlee/metal2_audit.md` copy this audit followed.)*

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/bcast_to` |
| **Overall** | **GREEN (user override)** — mechanically RED on a stale readiness sheet (spreadsheet-broken gate). The user directed proceeding on the code-side evidence (see Result). Every other gate is clear. |
| **DOps / Factories** | `BcastToOperation` → direct `create_descriptor` (the sheet still lists `BcastToTileFactory`, which no longer exists) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 8 dataflow kernels use `Noc` / `CircularBuffer` / `TensorAccessor`, with no free-function holdovers |
| *Prereqs* — Cross-op escapes | Ok — the only donor is `kernel_lib/eltwise` (shared lib, `uint32_t cb_id` shape ✓). No borrowed kernel files. |
| *Feature Support* — overall | GREEN — no Appendix A signal fires |
| *Feature Support* — Variadic-CTA | Ok — no CTA varargs |
| *TTNN Readiness* — `Is able to port?` (the gate) | Cell reads `yes (with PD step)`. **Overridden by spreadsheet-broken:** the `Concept` column conflicts with the code, and the sheet has a phantom factory row. |
| *TTNN Readiness* — Concept (current) | Sheet: `legacy device-op`. Code: **`descriptor`** (direct descriptor). **Conflict.** |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Custom hash | No — matches the sheet (`no` / backdoor `no`) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No — matches the sheet |
| *TTNN Readiness* — `override_runtime_arguments` | No in the code. The sheet says `n/a`, which fits its stale legacy concept; for the current `descriptor` op it should read `no`. |
| *TTNN Readiness* — Pybind `create_descriptor` | No — `bcast_to_nanobind.cpp:43` binds only the `ttnn.experimental.broadcast_to` function. Matches the sheet. |
| *TTNN Readiness* — Op-owned tensors | No — matches the sheet |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (matches the sheet's `Porting Target`). The port must also convert the direct descriptor to a conventional factory (`ttnn_factory.md` §3). |
| *Port work* — Offset base pointer | none — both address args are bare `Buffer*` |
| *Port work* — Tensor bindings (per binding) | `input` Case 1 · `output` Case 1 |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | none — no accessor passes a 3rd argument |
| *Port work* — CB endpoints | `c_0`: 1:1 legal in every config. `c_1`: **conditional DFB** — dead under `NONE`, 1:1 under `ROW` / `COL` / `SCALAR`. |

## Result

> **User override (2026-10-01).** After reviewing the RED below, the user (Edwin Lee, author of #57409) confirmed that the sheet row is simply out of date and directed the audit to proceed ignoring the spreadsheet-broken gate. The code-side cross-check supports this: the op is on the `descriptor` concept, and every other sheet column that can be checked in code agrees. **`METAL2_PORT_BRIEF.md` was issued on that basis.** The sheet row still needs reconciling with Diego. That is now a bookkeeping item, not a port blocker. The original verdict is kept below for the record.

**Original verdict: RED → blocked on the TTNN factory concept gate (spreadsheet broken), routed to the readiness-sheet owner (Diego).** The op's own code is clear on every other gate.

In plain terms: the readiness sheet still describes `bcast_to` as it was before #57409. That means a `legacy device-op` concept and a `BcastToTileFactory` factory. In the code, that factory is gone and the op is a `descriptor` op. A primary-column conflict plus a phantom factory row is a spreadsheet-broken trigger, so the recipe gates on it rather than proceeding on data it can't trust.

**Path forward:** the sheet owner reconciles the row: `Concept` → `descriptor`, `Factory (variant)` → the direct descriptor, `Override runtime args method?` → `no`, and re-derive `Is able to port?`. Then re-audit. Nothing in this op needs to change. Every other gate is clear, and the porter-facing detail below was run in full (see the scoping note), so a re-audit should be cheap. Once the row is reconciled, it should mostly confirm this report and issue the brief.

Single device-op with a single descriptor, so `RED at op level; no portable subset`. The block covers the whole op, but it lifts without any code change.

**Scoping note (Red-outcome rule, "clears without touching the op's code").** The blocker is cleared on the sheet side, not in the op code, so a re-audit will read the same code. I therefore ran all seven informational subjects instead of skipping them.

## Gate detail

- **TTNN factory concept (`Is able to port?`): RED — spreadsheet broken → readiness-sheet owner to reconcile.** Sheet row (fetched live this session):
  - `Op` = `experimental/bcast_to`, `Device operation` = `BcastToOperation`, `Factory (variant)` = `BcastToTileFactory`
  - `Concept` = `legacy device-op`, `Op Classification` = `Legacy Op`, `Porting Target` = `ProgramSpecFactoryConcept`
  - `Custom hash` = `no`, `Backdoor custom hash` = `no`, `Runtime-args update (get_dynamic_runtime_args)` = `no`, `Override runtime args method? (PD only)` = `n/a`, `Pybind descriptor` = `no`, `Smuggled pointer` = `no`
  - `Known op issues` = *(blank)*, `TensorParameter relaxation` = `none`, `Op-owned tensors?` = `no`, `Diego validation` = `yes`
  - `Is able to port?` = `yes (with PD step)`
  - `Factory definition path` / `Declared in` = `…/bcast_to/device/bcast_to_device_operation.hpp`

  Cross-check against the code:
  - **`Concept`: conflict.** The sheet says `legacy device-op`. The code has a static `create_descriptor` returning `ProgramDescriptor` on the device-op (`device/bcast_to_device_operation.hpp:40-43`, `device/bcast_to_program_factory.cpp:135-224`), with no `create()` and no `override_runtime_arguments()`. That is the `descriptor` concept. `git show f5093e705ae` shows this exact change: it deletes `struct BcastToTileFactory { create(); override_runtime_arguments(); }` and `using program_factory_t = std::variant<BcastToTileFactory>;` and adds `create_descriptor`.
  - **Factory-set match: phantom row.** `BcastToTileFactory` no longer exists anywhere in `ttnn/` (grep returns 0 hits). The descriptor that replaced it has no row of its own.
  - `Custom hash`, `get_dynamic_runtime_args`, `Pybind descriptor`, `Op-owned tensors?`: all agree with the code (no hits for `compute_program_hash` / `attribute_values` / `to_hash` / `get_dynamic_runtime_args` / `nb::class_` in the op directory).
  - `Override runtime args method?` = `n/a`: consistent with the sheet's own stale concept. The code value is `no`.
  - Cross-column invariants: none violated.

  Note on the cell: `yes (with PD step)` is neither `yes` nor `no`. Read literally, it says the PD migration is still pending, which is the same staleness. The verdict above doesn't depend on reading the cell: the primary-column conflict alone is a spreadsheet-broken trigger. (See *Recipe notes*.)

- **Device 2.0 (every kernel used): GREEN.** All 8 dataflow kernels use `Noc noc;`, `CircularBuffer cb_*(id);`, `TensorAccessor(args, addr)`, `noc.async_read(src, cb_src, …)` / `noc.async_write(cb_dst, dst, …)`, `cb.reserve_back` / `push_back` / `wait_front` / `pop_front`, and `cb.get_tile_size()` (a method, not the free function). A grep for `get_read_ptr` / `get_write_ptr` / `noc_async_*` / `*AddrGen*` / `get_semaphore` / `get_local_cb_interface` / `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front` over `device/kernels/` returns nothing. The compute kernels go through `kernel_lib/eltwise` with `uint32_t` buffer ids, which is the normal compute idiom and not a DM holdover. No violations to tabulate.

- **Feature compatibility: GREEN — no gate fired.**

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer`, `global_circular_buffer` field, `remote_index`, or 4-arg `CreateCircularBuffer` |
  | CBDescriptor `address_offset` (non-zero) | N/A | both `CBDescriptor`s (`device/bcast_to_program_factory.cpp:158-166`) leave `address_offset` at its default |
  | GlobalSemaphore | N/A | no semaphores at all |

- **CB endpoints (GATE-free):** `c_0` is a legal 1P+1C in every config. `c_1` is a **conditional DFB**: dead under `NONE`, live 1:1 under `ROW` / `COL` / `SCALAR`. Detail is under Port-work.

- **Offset base pointers: GREEN.** Both address args are bare `Buffer*` with no arithmetic: `input.buffer()` at `device/bcast_to_program_factory.cpp:84` and `output.buffer()` at `:100`. Neither appears in `2026-07-19_offset_base_pointers.md` (outcome: no fold, op not in the tables).

- **TensorAccessor 3rd argument: N/A.** Every accessor is the 2-arg form `TensorAccessor(src_args, src_addr)` / `TensorAccessor(dst_args, dst_addr)` (readers `:31`, writers `:32`/`:33`), so the subject never fires. The op isn't in `2026-07-06_tensor_accessor_3rd_arg_triage.md` either.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings:**
  - `input` — **Case 1.** It reaches the kernel as a `Buffer*` binding in the reader's `emplace_runtime_args` (`device/bcast_to_program_factory.cpp:84`). This is correct on cache hit today, since the framework patches it; it is not the stale-pointer hazard. Every reader feeds it straight into `TensorAccessor(src_args, src_addr)` (`reader_interleaved_*_bcast_to.cpp:13,31`). The `TensorAccessorArgs(input.buffer())` CTA plumbing (`device/bcast_to_program_factory.cpp:173`) goes away.
  - `output` — **Case 1.** Same shape: a `Buffer*` binding at `device/bcast_to_program_factory.cpp:100`, consumed by `TensorAccessor(dst_args, dst_addr)` (`writer_interleaved_*_bcast_to.cpp:13,32-33`). CTA plumbing at `:186`. The output may be a caller-supplied preallocated tensor (`tensor_args.output`, `device/bcast_to_device_operation.cpp:154-156`); either way it is the op's output binding.
- **TensorParameter relaxation:** none (sheet: `none`).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints** (per `(CB, config)`, every node in `all_device_cores` — idle cores run the same kernels with `num_tiles = 0`):
  - `c_0` / `NONE`: reader is the locked producer (`reserve_back` / `push_back`), writer is the locked consumer (`wait_front` / `pop_front`; `writer_cb_id = c_0`, `device/bcast_to_program_factory.cpp:183-184`) → **1P+1C**.
  - `c_0` / `ROW`·`COL`·`SCALAR`: reader is the locked producer, compute is the locked consumer (`ckl::input(dfb_id_src_id, WaitPolicy::PerTile, PopPolicy::PerTile, …)`) → **1P+1C**.
  - `c_1` / `ROW`·`COL`·`SCALAR`: compute is the locked producer (`ckl::output(dfb_id_dst_id, ReservePolicy::PerTile, PushPolicy::PerTile, …)`), writer is the locked consumer → **1P+1C**.
  - `c_1` / `NONE`: **0 touchers**. The writer binds `c_0`, and `compute_interleaved_no_bcast_to.cpp:10-12` has an empty `kernel_main` that never reads its CTAs. → **Conditional DFB: dead under `NONE`, live under `ROW` / `COL` / `SCALAR`. Do not drop it.** The legacy host allocates both CBs unconditionally in a loop (`device/bcast_to_program_factory.cpp:157-167`), so the condition (`subtile_broadcast_type != NONE`) is new host-side structure.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. There is no raw-pointer access, no semaphores, and no dual-instance split. **Do not bind `c_0` / `c_1` to the compute kernel under `NONE`.** The legacy compute CTAs `{c_0, c_1}` (`device/bcast_to_program_factory.cpp:209`) are dead there. Binding `c_0` to compute under `NONE` would make it a phantom third toucher of `c_0` (reader + writer + compute) and push it into multi-binding for no reason.
- **Cross-op / shared kernels:** none. Every kernel `.cpp` is owned by the op and instantiated only by `bcast_to_utils.cpp:306-329`. A filename census (`grep -rl <kernel-filename> ttnn/cpp/ttnn/operations/`) finds no binder outside `experimental/bcast_to/`. There are no intra-op shares either: each config binds its own distinct reader / writer / compute. No `_metal2` fork exists, and none is needed. The port converts these kernels in place. *(Ignore anything under `experimental/quasar/`; there is no bcast_to copy there today.)*
- **RTA varargs:** none. Every kernel reads a fixed `arg_index++` run (reader 13, writer 14, compute 12 args) → name each one.
- **Direct-descriptor shape → give it a conventional factory** (`ttnn_factory.md` §3, "Give a direct-descriptor op a conventional program factory"). `BcastToOperation` has `create_descriptor` as its own member and no `program_factory_t` (`device/bcast_to_device_operation.hpp:40`). Swapping it for `create_program_artifacts` drops `HasDirectDescriptor`, so the port has to introduce a factory struct + `using program_factory_t = std::variant<…>`. Record it under Handoff points.
- **Idle-core RTAs.** Cores outside `core_group_1` / `core_group_2` get all-zero arg vectors of length 13 / 14 / 12 (`device/bcast_to_program_factory.cpp:65-70`), including a 0 in the old address slot. After the port, the address slot becomes the tensor binding and the remaining named args stay zero on idle cores. Keep the per-core counts consistent with the named schema.
- **`unpack_to_dest_mode` is keyed by CB index** (`device/bcast_to_program_factory.cpp:199-203`: `c_0 → UnpackToDestFp32` for 32-bit formats). Carry it onto the `c_0` DFB / compute config in whatever form the recipe uses.
- **Kernels are already part-modernized:** the compute kernels name their CTAs `dfb_id_src_id` / `dfb_id_dst_id`. The dataflow kernels still include `api/dataflow/circular_buffer.h` and use `CircularBuffer`, so the port is the normal CB→DFB + binding-token swap.
- **Program-cache test exists:** `tests/ttnn/nightly/unit_tests/operations/experimental/test_bcast_to.py:137` (`test_bcast_to_program_cache`) moves tensor addresses across cache hits and asserts a single cache entry. It is a good regression check for the binding swap.

## Team-only

- **Out-of-directory coupling & donor shape — op-level roll-up: ✓ clean.**
  - The only escapes are compute-kernel `#include`s of `ttnn/cpp/ttnn/kernel_lib/eltwise/{broadcast/bcast.hpp, api/chain.hpp, api/convenience.hpp}` (shared kernel library, class 2) and `tt_metal/*` / `tools/profiler` headers (class 1).

  | Op kernel | Donor file | Class | Functions used | Shape | Status |
  |---|---|---|---|---|---|
  | `compute_interleaved_{row,col,scalar}_bcast_to.cpp` | `kernel_lib/eltwise/api/chain.hpp` (+ `broadcast/bcast.hpp`, `api/convenience.hpp`) | shared lib | `ckl::eltwise_chain`, `ckl::UnaryBcast`, `ckl::PackTile`, `ckl::input(uint32_t cb_id, …)`, `ckl::output(uint32_t cb_id, …)` (`chain.hpp:356`) | `uint32_t cb_id` (NTTP position) | ✓ |
  | `compute_interleaved_no_bcast_to.cpp` | — (includes `api/compute/*` only; body empty) | — | — | — | ✓ |
  | all 8 dataflow kernels | `api/dataflow/{dataflow_api.h,noc.h,circular_buffer.h}` | `tt_metal` | — | — | ✓ |

  - Per-call detail: omitted (all ✓).
  - **Borrowed kernel files:** none. All 12 kernel `.cpp`s live in this op's `device/kernels/` and are instantiated only here.

- **Relaxation candidates:** none. There is no custom hash to mine.

- **TTNN factory analysis:**
  - Op-owned tensors: none.
  - MeshWorkload need: none (single `ProgramDescriptor`).
  - Pybind `create_descriptor`: none. The nanobind exposes only the function (`bcast_to_nanobind.cpp:43`).
  - Other risky pybind: none.
  - Custom hash: none — the default hash covers `operation_attributes_t{output_shape, memory_config, subtile_broadcast_type}` + `tensor_args_t{input, optional output}`. The comment at `device/bcast_to_device_operation.hpp:37-39` relies on this: every non-address RTA comes from hashed state.
  - `get_dynamic_runtime_args`: none.
  - `override_runtime_arguments`: none (removed in #57409).
  - **Target: `ProgramSpecFactoryConcept`**, via the direct-descriptor → conventional-factory conversion.

## Misc anomalies  *(team-only, non-gating)*

- **Dead RTAs, sent in every config.** All kernels share one 12-arg core layout, plus `src_addr` / `dst_addr` up front and `start_tile_id` at the writer tail, but many slots are read and never used:
  - all writers and all compute kernels: `n_stride`, `c_stride`
  - writers/compute `row`/`col`: `start_t`, `HtWt`
  - writers/compute `no`/`scalar`: `start_th`, `start_tw`
  - readers `row`/`col`: `start_t`
  - readers `no`/`scalar`: `start_th`, `start_tw`
  - readers `row`/`col`/`scalar`: the local `next_channel_shift` is computed and never used (e.g. `reader_interleaved_col_bcast_to.cpp:41`)

  Harmless, but the port will name these and carry them forward. The ops team may want to trim them separately.
- **Empty compute kernel launched under `NONE`.** `compute_interleaved_no_bcast_to.cpp` has an empty `kernel_main`, but it is still built and dispatched on every core of the full compute grid with 12 RTAs per core and `fp32_dest_acc_en` / `unpack_to_dest_mode` config (`device/bcast_to_program_factory.cpp:205-221`). It could be skipped for `NONE`, which is an ops-team optimization and not port work. The empty kernel also includes `api/compute/bcast.h` / `eltwise_binary.h` that it doesn't use.
- **`NONE` config (same H/W, N/C-only broadcast) is untested.** All such cases in `tests/ttnn/unit_tests/operations/eltwise/test_broadcast_to.py:10-17` are commented out (the file notes "broadcast_to op is not a production op"), and the nightly `test_bcast_to.py` covers only row / col / scalar. The `ReaderNoBcast` / `WriterNoBcast` / empty-compute path therefore runs in no active test. (Also surfaced to the porter in the brief, since the port's conditional `c_1` lives on this path.)
- **`auto input = tensor_args.input;`** (`device/bcast_to_program_factory.cpp:139`) copies the `Tensor` handle by value where `const auto&` would do. Trivial.

## Questions for the user

1. **Sheet reconciliation owner/timing:** the row for `experimental/bcast_to` predates #57409 (merged 2026-09-25; you authored it). Do you want to ping Diego to refresh the row (`Concept` → `descriptor`, factory → direct descriptor, override → `no`), or is a sheet refresh for the whole #57409 batch already planned? Once it lands, a re-audit should be a quick confirm plus brief.

## Recipe notes

- **Non-binary `Is able to port?` value.** The cell reads `yes (with PD step)`. The recipe's routing only covers `yes` / `no` ("**`no` blocks the port; `yes` clears this prerequisite**"). It doesn't say how to read a qualified `yes`, especially one that asserts a precondition (a PD step) the code shows is already met. Here the primary-column conflict decided the verdict anyway, but on an op where the PD step really is still pending, `yes (with PD step)` alongside `Concept = legacy device-op` is a combination the routing doesn't address. (I read it as: the sheet's `legacy device-op` concept still gates, so it's effectively `no` until PD lands.) A line in `ttnn_op_porting_readiness.md` listing the qualified values and how each routes would help.
- **Spreadsheet-broken vs. "clears elsewhere" scoping.** The Red-outcome exception lists "an unattributed or held readiness verdict" as clearing elsewhere, but doesn't name the spreadsheet-broken case explicitly. It behaves the same way: the sheet owner fixes it and the op code is untouched. I ran the informational subjects on that basis. Adding "a spreadsheet-broken conflict" to the *Elsewhere → run them* list would remove the judgment call.
- **Brief withheld despite full detail.** Under the recipe, a sheet-only RED still blocks the brief, even though every code-side subject is complete and clean. That's the correct conservative call, but it means the porter brief for an op like this is gated purely on a bookkeeping refresh. Consider whether a "brief issued, pending sheet reconciliation" state is worth having for the spreadsheet-broken case where the code-side cross-check shows the op is *further along* than the sheet says (as opposed to a conflict pointing the other way).
- **Recipe docs not in the op's checkout.** The provenance command assumes the recipe tree is in the same checkout as the op. Here it lives in a sibling worktree (`Port_Recipe`), so I took the hash from there and noted it. A one-line note on what to do when the docs are in a separate checkout would help.
