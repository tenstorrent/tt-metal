# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_mean`

One `DeviceOperation` in this directory, with three program factories nested inside it:

- **`ttnn::operations::moreh::moreh_mean::MorehMeanOperation`** (`device/moreh_mean_device_operation.{hpp,cpp}`)
  - `MorehMeanOperation::MorehMeanWFactory` (`device/moreh_mean_w_program_factory.cpp`) — reduce over the last dim
  - `MorehMeanOperation::MorehMeanHFactory` (`device/moreh_mean_h_program_factory.cpp`) — reduce over the second-to-last dim
  - `MorehMeanOperation::MorehMeanNCFactory` (`device/moreh_mean_nc_program_factory.cpp`) — reduce over any outer dim

Dispatch is by reduced-dim position only (`device/moreh_mean_device_operation.cpp:34-47`): `dim+1 == rank` → W, `dim+2 == rank` → H, otherwise NC. The op is interleaved-only; no sharded path exists.

All eight kernel files in `device/kernels/` are referenced — none is dead. The op **owns every kernel it runs** and file-path-instantiates nothing from a shared pool, which is unusual and materially simplifies the port (see *Team-only*).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `66ac84052d4 2026-07-27 docs(metal_2.0): split the runtime-args porting gate into its two sheet columns`

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_mean` |
| **Overall** | **RED at op level; subset `{MorehMeanHFactory, MorehMeanWFactory}` is clear** |
| **DOps / Factories** | `MorehMeanOperation` → `MorehMeanWFactory` · `MorehMeanHFactory` · **`MorehMeanNCFactory` (blocked)** |
| *Prereqs* — Device 2.0 (every kernel used) | **No (RED)** — exactly **one** violation: a CB-index-keyed free-function holdover at `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp:100`, reached only by `reader_moreh_mean_nc.cpp`. **Isolated holdover, 1-line mechanical fix.** Routed to the Device 2.0 track. See the boundary caveat in *Gate detail* — this one is close enough to the sanctioned list that the maintainer may want to rule on it. |
| *Prereqs* — Cross-op escapes | Ok — every function-call escape is ✓ (donors take `DataflowBuffer` by value or `constexpr uint32_t` DFB ids as NTTPs). **Zero borrowed kernel files** — no port-together set at all. |
| *Feature Support* — overall | **GREEN** — no Appendix A entry fires |
| *Feature Support* — Variadic-CTA | Ok — every `get_compile_time_arg_val` index is a literal or a `constexpr` accessor offset; `tensor_args_t` is a fixed `{const Tensor&, const std::optional<Tensor>&}` pair |
| *TTNN Readiness* — `Is able to port?` (the gate) | **NOT VERIFIED — readiness sheet could not be fetched this run** (claude.ai Google Drive MCP connector unauthorized in this non-interactive session). Every *cheaply-checkable* conjunct was verified against the code and is clean; the sheet-owned `Is safe to port?` axis and `TensorParameter relaxation` are unread. See *Gate detail* and *Questions*. |
| *TTNN Readiness* — Concept (current) | `descriptor` — all three factories expose `static ProgramDescriptor create_descriptor(...)` (`device/moreh_mean_device_operation.hpp:35, 42, 49`) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — not a `WorkloadDescriptor` op |
| *TTNN Readiness* — Is safe to port? | **Unknown** — sheet-owner judgment, not fetchable this run. Code-side note: the op carries **no** `->address()`-in-RTA smuggled pointers (every buffer rides the framework's `Buffer*` binding form), which is the most common source of a `no`. |
| *TTNN Readiness* — Custom hash | No — no `compute_program_hash` in `device/` |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No — hook absent from the device-op |
| *TTNN Readiness* — `override_runtime_arguments` | No — method absent from every factory and from the device-op |
| *TTNN Readiness* — Pybind `create_descriptor` | No — `moreh_mean_nanobind.cpp:19-30` binds only the user-facing `ttnn::moreh_mean` free function |
| *TTNN Readiness* — Op-owned tensors | No — `descriptor` concept, no `buffers` vector |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | **none — cleared.** No `->address()` anywhere; both tensors reach their kernels as bare `Buffer*` RTAs with no arithmetic |
| *Port work* — Tensor bindings (per binding) | **Case 1** for both bindings in every factory (all consumed through a `TensorAccessor`); no Case 2, no borrowed-DFB |
| *Port work* — TensorParameter relaxation | Unknown (sheet unread). No custom hash exists, so no relaxation can be active today |
| *Port work* — TensorAccessor 3rd arg | **none** — every `TensorAccessor` in the op is the 2-arg form; the subject does not fire |
| *Port work* — CB endpoints | **legal 1:1** for `c_0`/`c_1`/`c_2`/`c_16`; **self-loop** for `c_24` (all factories) and `c_25` (W, H); **conditional binding** required for `c_3` and `c_25` — they have **zero touchers** in the `!do_mask` instantiation |

**CB endpoints** are dispositions, not gates. Recorded per `(CB, config)` below. The `c_3` / `c_25` config-dependence is the substantive porter item in this op — Metal 2.0 has a documented mechanism for it (`migration_guide.md` → *Optional resources*), but it requires kernel-side `#ifdef` gating that the current `if` / `if constexpr` guards do **not** satisfy.

---

## Result

**RED at op level; subset `{MorehMeanHFactory, MorehMeanWFactory}` is clear.**

The single blocker is the **Device 2.0 prerequisite**, and it is one line:

`ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp:100` — inside `fill_cb_with_value`:

```cpp
FORCE_INLINE void fill_cb_with_value(DataflowBuffer cb, uint32_t value, int32_t num_of_elems = 1024) {
    cb.reserve_back(1);
    const DataFormat data_format = get_dataformat(cb.get_id());   // <-- holdover
```

A CB-index-keyed free function, called with the id pulled *out of* a Device 2.0 wrapper that is right there, where the wrapper-method replacement `DataflowBuffer::get_dataformat()` already exists (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:241`). The fix is `cb.get_dataformat()`. **Routed to the Device 2.0 migration team.**

Only `reader_moreh_mean_nc.cpp` calls `fill_cb_with_value` (`:32` and `:36`), so only **`MorehMeanNCFactory`** exercises the holdover. The W and H readers `#include` the same header but never instantiate that function, so their factories are clean.

**Path forward:** a one-line change on the Device 2.0 track, then a cheap re-audit and the whole op ports as one unit. In the meantime the H and W factories can be ported today; a scoped brief for them is issued in `METAL2_PORT_BRIEF.md`.

**Two things a reviewer should weigh before acting on the RED** — both spelled out in *Gate detail*:

1. **This holdover may belong on the sanctioned list.** `get_dataformat(cb_id)` is structurally identical to the *sanctioned* `get_tile_size(cb_id)` — both are free functions that the `CircularBuffer` wrapper simply forwards to (`circular_buffer.h:113-115`). I flagged it because the recipe's sanctioned list is explicit and closed and does not include it, and because the Device 2.0 migration guide never mentions `get_dataformat` while it *does* keep `get_tile_size(cb_id)` in its migrated examples. If the maintainer adds `get_dataformat(cb_id)` to the sanctioned list, **this op flips to fully GREEN with no code change.** Logged in *Recipe notes*.
2. **A second, weaker `get_dataformat(id)` site exists on the H path** and I deliberately did *not* gate it: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl:140` uses `get_dataformat(dfb_id)` on a `constexpr` template parameter, with no wrapper object in scope at that call site (the `DataflowBuffer` is constructed 21 lines later). That fails one of the two conditions the recipe requires for a holdover, and the file is kernel_lib (lib-team owned). But if the Device 2.0 team rules `get_dataformat(id)` out categorically, the **H factory joins the RED** and the clean subset shrinks to `{MorehMeanWFactory}` alone.

**One open item that is not a code finding:** the per-factory readiness sheet could not be fetched in this session (connector unauthorized; it cannot be authorized from inside a session). Every conjunct the recipe asks the auditor to cross-check against code was checked and is clean, but the sheet's own `Is safe to port?` call and the `TensorParameter relaxation` column are unread. **The sheet lookup must be completed before the subset port begins** — see *Questions for the user*.

---

## Gate detail

### TTNN factory concept (`Is able to port?`) — **not verified (sheet unavailable)**

The recipe requires this verdict to come from the TTNN team's *"Operations analysis"* sheet, fetched fresh via the claude.ai Google Drive MCP connector. That connector is unauthorized in this session and cannot be authorized from inside one — `ToolSearch` does not resolve `mcp__claude_ai_Google_Drive__download_file_content` at all — and no local CSV exists under `metal_2.0/analyses/`. This is a *fetch failure*, not a "conflict" or a "missing op row"; it is recorded rather than silently resolved either way.

What **was** verified, directly against the code:

| Column | Code evidence | Verdict |
|---|---|---|
| `Concept` | `static tt::tt_metal::ProgramDescriptor create_descriptor(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&)` on all three factories — `device/moreh_mean_device_operation.hpp:35, 42, 49` | `descriptor` |
| `Custom hash` | no `compute_program_hash` in `device/`; the device-op declares exactly `validate_tensors`, `select_program_factory`, `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors` (`device/moreh_mean_device_operation.hpp:57-61`) | `no` |
| `Runtime-args update (get_dynamic_runtime_args)` | hook absent | `no` |
| `Override runtime args method? (PD and legacy)` | no `override_runtime_arguments` anywhere in the op | `no` |
| `Pybind descriptor` | `moreh_mean_nanobind.cpp:19-30` binds only `ttnn::moreh_mean`; no `create_descriptor` binding, no factory/device-op internals exposed | `no` |
| `Secretly SPMD Workload?` | N/A (`descriptor`) | N/A |
| `Op-owned tensors?` | N/A — the `descriptor` concept cannot carry them; no `buffers` vector | `no` |
| **Factory-set match** | three factories in `program_factory_t` (`device/moreh_mean_device_operation.hpp:55`), three factory `.cpp` files, three `create_descriptor` definitions — self-consistent. **Cannot be checked against the sheet's row set** (exactly the staleness check the fetch failure defeats) | unchecked |

Five of six shape conjuncts are code-confirmed clean. The two the sheet alone supplies — `Is safe to port?` and `TensorParameter relaxation` — remain open. Since the op has **no custom hash**, a real relaxation cannot be active today, so the practical residual risk is confined to `Is safe to port?`.

### Device 2.0 (every kernel used) — **RED (one isolated holdover)**

Eleven translation units are exercised across the three factories: eight op-owned kernels plus three shared headers pulled in by them. Ten are Device 2.0 compliant. One is not.

#### The violation

| File | Line | Call | Wrapper in scope | Replacement |
|---|---|---|---|---|
| `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` | 100 | `get_dataformat(cb.get_id())` inside `fill_cb_with_value` | **yes** — `cb`, the `DataflowBuffer` parameter, on the immediately preceding line | `cb.get_dataformat()` — `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:241` |

Owning pool: `ttnn/cpp/ttnn/kernel/dataflow/` — a second shared-kernel pool (donor class 3, treated as shared-lib).

**Sizing for the Device 2.0 team: isolated holdover — one line.** Everything else in `moreh_common.hpp` is fully Device 2.0 (`DataflowBuffer::reserve_back` / `get_write_ptr` / `push_back` throughout, `Noc` objects for zeroing), and `fill_cb_with_value` itself uses wrapper methods on either side of the offending line. The git history makes the provenance unambiguous — the CB→DFB sweep rewrote the parameter type and the id accessor but left the free function standing:

```
af9c372a48c 2026-07-10  [Cleanup] Migrate Moreh Kernels from CircularBuffer to DataflowBuffer (#49430)
-FORCE_INLINE void fill_cb_with_value(CircularBuffer cb, uint32_t value, int32_t num_of_elems = 1024) {
+FORCE_INLINE void fill_cb_with_value(DataflowBuffer cb, uint32_t value, int32_t num_of_elems = 1024) {
     cb.reserve_back(1);
-    const DataFormat data_format = get_dataformat(cb.get_cb_id());
+    const DataFormat data_format = get_dataformat(cb.get_id());
```

`CircularBuffer::get_dataformat()` existed at that time too (`circular_buffer.h:115`), so this was a missed substitution in that sweep rather than a deliberate carve-out.

**Reach: `MorehMeanNCFactory` only.** `fill_cb_with_value` is called at `reader_moreh_mean_nc.cpp:32` (zero tile → `c_1`) and `:36` (scaler tile → `c_2`). The W reader uses `generate_mm_scaler` + `generate_mask_w` instead; the H reader uses `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler` + `generate_mask_h`. All three readers `#include "ttnn/kernel/dataflow/moreh_common.hpp"`, but `fill_cb_with_value` is a `FORCE_INLINE` free function that is never instantiated or emitted in a TU that does not call it — so the W and H kernels do not exercise the violation.

> A reviewer taking a whole-header view (rather than a called-code view) would RED all three factories. I applied the called-code reading because the gate is defined over "whether the op's program factory instantiates or calls into the kernel." Flagging the alternative so the subset can be narrowed on request.

**Co-borrowers of the fix:** `moreh_common.hpp` is shared across the whole `moreh` family, so this one-line change unblocks every moreh op that calls `fill_cb_with_value`, not just this one. That argues for fixing it centrally and early.

#### The near-miss I chose not to gate

`ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl:140` — `constexpr DataFormat data_format = get_dataformat(dfb_id);` inside `prepare_reduce_scaler`, reached from `reader_moreh_mean_h.cpp:33` (**H factory only**).

Not classified as a holdover, for two independent reasons:

1. **The wrapper is not in scope at the call site.** `dfb_id` is a `constexpr` template parameter; the `DataflowBuffer dfb(dfb_id)` object is constructed at `:161`, twenty-one lines later. The recipe's holdover definition requires the Device-2.0 wrapper object to be *already in scope at the call site*, and it is not.
2. **Donor class 2** — `ttnn/cpp/ttnn/kernel_lib/` is the official shared kernel library, which the coupling taxonomy assigns to the lib team internally.

Recorded here so the Device 2.0 team can sweep it alongside the `moreh_common.hpp` fix if they prefer one pass. **If they rule `get_dataformat(id)` out categorically, the H factory joins the RED and the clean subset becomes `{MorehMeanWFactory}` only.**

#### Kernels confirmed Device 2.0 compliant

| Kernel / header | Owner | Used by | Idioms |
|---|---|---|---|
| `device/kernels/reader_moreh_mean_w.cpp` | this op | W | `Noc`, `DataflowBuffer`, `TensorAccessor` |
| `device/kernels/reader_moreh_mean_h.cpp` | this op | H | `Noc`, `DataflowBuffer`, `TensorAccessor` |
| `device/kernels/reader_moreh_mean_nc.cpp` | this op | NC | `Noc`, `DataflowBuffer`, `TensorAccessor` |
| `device/kernels/writer_moreh_mean_unary_interleaved_start_id.cpp` | this op | W, H | `Noc`, `DataflowBuffer`, `TensorAccessor` |
| `device/kernels/writer_moreh_mean_nc.cpp` | this op | NC | `Noc`, `DataflowBuffer`, `TensorAccessor` |
| `device/kernels/moreh_mean_w.cpp` | this op | W | `DataflowBuffer` objects throughout |
| `device/kernels/moreh_mean_h.cpp` | this op | H | `DataflowBuffer` objects + `compute_kernel_lib::reduce` |
| `device/kernels/moreh_mean_nc.cpp` | this op | NC | `DataflowBuffer` objects throughout |
| `ttnn/cpp/ttnn/kernel/dataflow/generate_mm_scaler.hpp` | shared pool (class 3) | W | `DataflowBuffer` + `Noc::async_write_zeros` |
| `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | shared pool (class 3) | W, H, NC | every `*_with_dt` helper takes `DataflowBuffer` by value |
| `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` / `.inl` | kernel_lib (class 2) | H | `constexpr` DFB ids as NTTPs; no CB-index free functions |

**Sanctioned free functions observed — deliberately not flagged**, per the Green bullet's whitelist. Every one is `get_tile_size(cb_id)`, which the Device 2.0 migration guide keeps as a free function in its own migrated examples (`device_api_migration_guide.md:605, 630`):

- `reader_moreh_mean_w.cpp:38`, `reader_moreh_mean_h.cpp:50`, `reader_moreh_mean_nc.cpp:43`
- `writer_moreh_mean_unary_interleaved_start_id.cpp:24`, `writer_moreh_mean_nc.cpp:25`
- `reduce_helpers_dataflow.inl:167, 199`

(The same guide shows `get_write_ptr(cb_id)` → `cb.get_write_ptr()` as a migration at `:338`/`:353`, so that one is *not* sanctioned — and the op has no instances of it.)

### Feature compatibility — **GREEN**

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `global_circular_buffer.hpp` include, no `.global_circular_buffer` field on any `CBDescriptor`, no `remote_index` / `remote_cb_*` idiom, no 4-arg `experimental::CreateCircularBuffer(..., global_cb)`. All 16 `CBDescriptor`s across the three factories are plain L1 CBs — none even sets `.buffer` (the op has no borrowed-memory CB at all). |
| CBDescriptor `address_offset` (non-zero) | N/A | The token `address_offset` does not appear anywhere in the op directory. No `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. |
| GlobalSemaphore | N/A | The op uses **no semaphores of any kind** — `Semaphore`, `GlobalSemaphore`, `CreateSemaphore`, `global_semaphore.hpp` are all absent from the entire directory. |
| Variable-count compile-time arguments (CTA varargs) | N/A | Op-level cue does not fire: `tensor_args_t` is a fixed `{const Tensor& input; const std::optional<Tensor>& output;}` (`device/moreh_mean_device_operation.hpp:26-29`) — an `optional` is a fixed-count slot, not a variable-count container. Kernel-level decider does not fire: every `get_compile_time_arg_val` takes a **literal** index, except two `constexpr` accessor offsets — `get_compile_time_arg_val(src_args.next_compile_time_args_offset())` at `reader_moreh_mean_w.cpp:17` and `reader_moreh_mean_h.cpp:32` — which the entry's guard explicitly excludes ("Args read at **constexpr** offsets — even computed ones — are fixed-count, not this"). |

### CB endpoints (GATE-free) — legal, self-loops, and a config-scoped conditional binding

Counted **per CB, per node, per instantiation**. Structural facts that simplify the census: the op has **no semaphores**, which rules out the hidden-second-writer face (a raw semaphore-gated co-fill cannot exist without a coordinating semaphore); and it never instantiates one kernel source twice over the same core range, which rules out the dual-instance work-split face. The two compute `KernelDescriptor`s in each factory sit on `core_group_1` / `core_group_2`, which `split_work_to_cores_wt_core_range` (`ttnn/cpp/ttnn/operations/moreh/moreh_helper_functions.cpp`) derives from `tt_metal::split_work_to_cores` — disjoint, with `all_cores` as their union — so **every node carries exactly one reader, one writer and one compute instance.**

| Factory | CB | Config | Touchers on a node | Verdict |
|---|---|---|---|---|
| W | `c_0` (input) | all | reader P (`reader_moreh_mean_w.cpp:41,44`) · compute C (`moreh_mean_w.cpp:57,63,76,94,99,120`) | **legal 1:1** |
| W | `c_2` (scaler) | all | reader P — `generate_mm_scaler` reserve/push (`generate_mm_scaler.hpp:13,30`) · compute C (`moreh_mean_w.cpp:36,130`) | **legal 1:1** |
| W | `c_3` (mask_w) | `do_mask_w` | reader P — `generate_mask_w` (`moreh_common.hpp:233,257`) · compute C (`moreh_mean_w.cpp:43,128`) | **legal 1:1** |
| W | `c_3` (mask_w) | **`!do_mask_w`** | **zero** — reader side removed by `#ifdef DO_MASK_W` (`reader_moreh_mean_w.cpp:24-27`); compute side under `if (do_mask_w)` with `do_mask_w` a false `constexpr bool` | **conditional binding** — see below |
| W | `c_24` (accum_dst) | all | compute only — reserve/push at `:67,71`, wait/pop at `:101,122` | **self-loop** |
| W | `c_25` (masked_input) | `do_mask_w` | compute only — reserve/push at `:88,92`, then wait/pop at `:99,120` via `cb_input = cb_masked_input` (`:95`) | **self-loop** |
| W | `c_25` (masked_input) | **`!do_mask_w`** | **zero** | **conditional binding** |
| W | `c_16` (output) | all | compute P (`:114,118`) · writer C (`writer_…_start_id.cpp:28,31`) | **legal 1:1** |
| H | `c_0` | all | reader P (`reader_moreh_mean_h.cpp:57,60`) · compute C | **legal 1:1** |
| H | `c_2` (scaler) | all | reader P — `calculate_and_prepare_reduce_scaler` (`reduce_helpers_dataflow.inl:163,203`) · compute C (`moreh_mean_h.cpp:35,100`) | **legal 1:1** |
| H | `c_3` (mask_h) | `do_mask_h` | reader P — `generate_mask_h` (`moreh_common.hpp:193,217`) · compute C (`moreh_mean_h.cpp:42,98`) | **legal 1:1** |
| H | `c_3` (mask_h) | **`!do_mask_h`** | **zero** — reader side `#ifdef DO_MASK_H`-removed; compute side inside `if constexpr (do_mask_h)`, a language-discarded branch | **conditional binding** |
| H | `c_24` (accum_dst) | all | compute only — produced by `reduce<…, cb_accum_dst>` (`:54`), consumed by `Accumulate::at(cb_accum_dst, …)` (`:84`, `:92`) | **self-loop** |
| H | `c_25` (masked_input) | `do_mask_h` | compute only — reserve/push at `:72,76`, consumed by `reduce<…, cb_masked_input, …>` at `:81` | **self-loop** |
| H | `c_25` (masked_input) | **`!do_mask_h`** | **zero** | **conditional binding** |
| H | `c_16` | all | compute P · writer C | **legal 1:1** |
| NC | `c_0` | all | reader P (`reader_moreh_mean_nc.cpp:52,55`) · compute C (`moreh_mean_nc.cpp:43,52`) | **legal 1:1** |
| NC | `c_1` (zero tile) | all | reader P — `fill_cb_with_value` (`:32`) · compute C — `wait_front` at `moreh_mean_nc.cpp:34` (never popped; a persistent constant tile, which is intentional) | **legal 1:1** |
| NC | `c_2` (scalar) | all | reader P — `fill_cb_with_value` (`:36`) · compute C (`moreh_mean_nc.cpp:35`) | **legal 1:1** |
| NC | `c_24` (intermed0) | all | compute only — reserve/push at `:58,62`, wait/pop at `:45,54,69,79` | **self-loop** |
| NC | `c_16` | all | compute P (`:74,78`) · writer C (`writer_moreh_mean_nc.cpp:29,33`) | **legal 1:1** |

#### The `c_3` / `c_25` conditional binding — the substantive porter item

Both mask-path CBs are allocated **unconditionally** by the host (`moreh_mean_w_program_factory.cpp:80-88` and `:98-106`; `moreh_mean_h_program_factory.cpp:80-88` and `:98-106`) while their *use* is gated by `do_mask_w` / `do_mask_h`. In the no-mask instantiation they therefore have zero touchers. A DFB with no producer and no consumer binding is rejected by the spec validator, so they cannot be carried across as-is.

This is **not** a dead-CB drop — the CBs are genuinely live in the mask configuration. Metal 2.0 has a first-class mechanism for exactly this shape, documented in `migration_guide.md` under *Optional resources*: bind conditionally on the host (omit from `KernelSpec::dfb_bindings` when the path isn't taken) and gate the kernel side at the **preprocessor** level. That doc is explicit that `if constexpr` is *not* sufficient:

> Metal 2.0's `dfb::<name>` namespace is generated from the actual host bindings — `dfb::cb_scaled` exists only when the host actually binds it — and `if constexpr` in non-template `kernel_main` still performs name lookup on the discarded branch, so `if constexpr (false) { … dfb::cb_scaled … }` fails to compile at parse time.

So both the W kernel's plain `if (do_mask_w)` guards (`moreh_mean_w.cpp:42, 74, 127`) **and** the H kernel's `if constexpr (do_mask_h)` guards (`moreh_mean_h.cpp:41, 59, 97`) must become `#ifdef DO_MASK_W` / `#ifdef DO_MASK_H`, and the unconditional aliases and `DataflowBuffer` constructions at `moreh_mean_w.cpp:24-25, 28-29` and `moreh_mean_h.cpp:25-26, 28-29` must move inside those `#ifdef`s. The host already computes the flag and already emits the matching define — but **only to the reader** (`moreh_mean_w_program_factory.cpp:127-129`, `moreh_mean_h_program_factory.cpp:123-125`); the port must also emit it to the compute kernel.

A lower-risk alternative exists and is worth naming: **bind `c_3` and `c_25` unconditionally in both configs**, declaring the bindings even where no runtime traffic flows. That is byte-for-byte what the legacy op does today (the CBs are allocated in both configs), needs no kernel `#ifdef` churn, and keeps the diff minimal — at the cost of leaving a small L1 allocation unused in the no-mask config, exactly as today. Recommended as the default; the conditional form is the cleaner end-state if the porter wants it. Either way the porter must make a deliberate choice here rather than translating the CB list mechanically.

### Offset base pointers — **GREEN, cleared**

The op is **not** in the tables of `analyses/2026-07-19_offset_base_pointers.md` (which carries no `moreh` rows at all), and an independent scan of every address-bearing runtime arg confirms the clean-base finding rather than resting on that absence.

There is **no `->address()` expression anywhere in the op.** Both tensors reach their kernels through the `Buffer*`-binding form — the factory pushes the `Buffer*` object itself into `KernelDescriptor::RTArgList` and the framework auto-registers a `BufferBinding`:

- `moreh_mean_w_program_factory.cpp:205-206` (`input_buf`, `output_buf`), consumed at `:219` and `:222`
- `moreh_mean_h_program_factory.cpp:200-201`, consumed at `:212-214` and `:216`
- `moreh_mean_nc_program_factory.cpp:175-176`, consumed at `:191-199` and `:201`

A `Buffer*` push carries no arithmetic, so there is nowhere for a host-folded offset to hide. Every positional quantity the kernels need is a **separate scalar RTA** consumed as a page index, never added to a base: `tile_offset`, `start_id`, `col_start_tile_id`, `input_tile_stride`, `tile_offset / out_dim_divider`. All page-id arithmetic happens on the device inside the `TensorAccessor` (`reader_moreh_mean_h.cpp:58,61`, `reader_moreh_mean_nc.cpp:50,56`).

Type 3 (`address_offset`) is N/A per Appendix A above; Type 4 (`ttnn::narrow`) does not appear.

### TensorAccessor 3rd argument — **GREEN, subject does not fire**

No accessor in this op passes a third argument. All five construction sites are the 2-arg form:

- `reader_moreh_mean_w.cpp:34` — `TensorAccessor(src_args, src_addr)`
- `reader_moreh_mean_h.cpp:46` — `TensorAccessor(src_args, src_addr)`
- `reader_moreh_mean_nc.cpp:39` — `TensorAccessor(input_args, input_addr)`
- `writer_moreh_mean_unary_interleaved_start_id.cpp:20` — `TensorAccessor(dst_args, dst_addr)`
- `writer_moreh_mean_nc.cpp:21` — `TensorAccessor(output_args, output_addr)`

Consistent with `analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`, which lists `moreh_fold` and `moreh_getitem` but no `moreh_mean` row. Nothing for the porter to drop.

---

## Port-work summary  *(mirrors the brief; scoped to the clean `{H, W}` subset)*

- **Tensor bindings** (per binding) — **every binding is Case 1**; no Case 2, no borrowed-DFB `clean` binding anywhere in the op:

  | Factory | Binding | Delivery today | Kernel consumption | Case |
  |---|---|---|---|---|
  | W | `input` | `Buffer*` RTA 0 @ `moreh_mean_w_program_factory.cpp:219` | `TensorAccessor(src_args, src_addr)` — `reader_moreh_mean_w.cpp:34` | 1 |
  | W | `output` | `Buffer*` RTA 0 @ `moreh_mean_w_program_factory.cpp:222` | `TensorAccessor(dst_args, dst_addr)` — `writer_…_start_id.cpp:20` | 1 |
  | H | `input` | `Buffer*` RTA 0 @ `moreh_mean_h_program_factory.cpp:212-214` | `reader_moreh_mean_h.cpp:46` | 1 |
  | H | `output` | `Buffer*` RTA 0 @ `moreh_mean_h_program_factory.cpp:216` | `writer_…_start_id.cpp:20` | 1 |

  *(Out-of-subset, for the eventual full port: NC's `input` @ `moreh_mean_nc_program_factory.cpp:191-199` → `reader_moreh_mean_nc.cpp:39`, and `output` @ `:201` → `writer_moreh_mean_nc.cpp:21` — both Case 1 as well.)*

  Note on urgency: the `Buffer*` form is the framework's interim binding hack, patched correctly on cache hits today. This is **routine port work, not a correctness hazard** — there are no `->address()`-in-RTA smuggled pointers in this op.

- **TensorParameter relaxation:** none applicable — the op has no custom hash, so no relaxation can be active. *(The sheet's column is unread; see Questions.)*

- **TensorAccessor 3rd arg:** none — no site passes one.

- **CB endpoints:**
  - **self-loop** `c_24` in both W and H (compute-only accumulator), and `c_25` in both W and H under the `do_mask` config (compute-only masked-input staging).
  - **conditional binding** for `c_3` (W, H) and `c_25` (W, H): zero touchers in the `!do_mask` instantiation. Recommended resolution — keep the binding unconditional (matches today's allocation exactly, minimal diff); the `#ifdef`-gated conditional form per `migration_guide.md` → *Optional resources* is the cleaner alternative. See the analysis above.
  - everything else is a plain 1P+1C: `c_0`, `c_2`, `c_16` (and `c_1`, `c_2` in NC).
  - **no multi-binding flag anywhere, no dead-CB drop.**

---

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. No semaphores in the op (rules out the hidden-second-writer face structurally) and no kernel source instantiated twice over one core range (rules out the dual-instance work-split face). The hunt can be skipped.

- **Runtime-selected CB index — a construct to plan for, in two kernels.** Not an audit gate (nothing in Appendix A covers it), but it is the one place a mechanical `dfb::name` substitution will not drop straight in:
  - `moreh_mean_w.cpp:21` declares `auto cb_input = tt::CBIndex::c_0;` as a **mutable** variable, reassigns it to `cb_masked_input` at `:95`, and constructs `DataflowBuffer(cb_input)` inline at `:57, 63, 76, 94, 99, 120`. The variable holds one of two different DFB identities depending on the mask path.
  - `moreh_mean_nc.cpp:48` does the same in miniature: `uint32_t cb_add = enable_reload ? cb_intermed0 : cb_in1;` — selected on a genuine *runtime* bool — then `add_tiles_init_with_dt(dfb_in0_obj, DataflowBuffer(cb_add))` at `:49` and `add_tiles(cb_in0, cb_add, …)` at `:50`.

  Both remain expressible: `dfb::name` accessors implicitly convert to `uint32_t` (`migration_guide.md:579`), so a `uint32_t` variable can still hold either identity and `DataflowBuffer(that_variable)` still constructs. The care needed is that under **conditional** binding, `dfb::cb_masked_input` only exists when the mask path is bound — so the `cb_input = cb_masked_input` assignment must sit inside the same `#ifdef` as the binding.

- **Cross-op / shared kernels: unusually clean — nothing to coordinate.** The op **owns all eight** of its kernel `.cpp` files and file-path-instantiates none from a shared pool, so there is **no port-together set** from file-path coupling. The only escapes are header-level function calls, all with workable shapes (full inventory in *Team-only*). The one caveat is the Device 2.0 fix itself: `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` is shared across the whole `moreh` family, so that one-line change touches many ops (beneficially).

- **RTA varargs:** **none.** Every runtime arg in every kernel is a distinct field at a fixed index. Worth calling out one shape so it is not mis-flagged: `reader_moreh_mean_nc.cpp:12-19` pulls seven args through a running `i++` counter, but it is a **fixed** run at the top of the kernel over a fixed set — the recipe's explicit non-signal ("a fixed run of reads via a running `arg_index++` at the top of the kernel … dissolves into named args"), not a loop. Name all seven. (`ArgFetcher`, the generic running-index helper in `moreh_common.hpp:44-53`, is *not* used by this op.)

---

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ clean** on function-call escape, and **no file-path escape at all**. The only ⭐ in the op is the Device 2.0 gate itself, judged in *Gate detail*.

#### Summary table — function-call escapes (`#include` outside the op directory)

| Op kernel | Donor file | Donor class | Status |
|---|---|---|---|
| `reader_moreh_mean_w.cpp:5` | `ttnn/cpp/ttnn/kernel/dataflow/generate_mm_scaler.hpp` | 3 — second shared-kernel pool | ✓ |
| `reader_moreh_mean_w.cpp:6`, `reader_moreh_mean_h.cpp:5`, `reader_moreh_mean_nc.cpp:6` | `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` | 3 — second shared-kernel pool | ⭐ — carries the Device 2.0 gate (NC only) |
| `reader_moreh_mean_h.cpp:6` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` (+ `.inl`) | 2 — official shared kernel library | ✓ (near-miss noted in *Gate detail*) |
| `moreh_mean_h.cpp:11` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` (+ `.inl`) | 2 — official shared kernel library | ✓ |
| `moreh_mean_w.cpp:12`, `moreh_mean_h.cpp:12`, `moreh_mean_nc.cpp:10` | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | 3 — second shared-kernel pool | ✓ |
| *(all kernels)* | `tt_metal/hw/inc/api/**` — `dataflow_api.h`, `noc.h`, `dataflow_buffer.h`, `noc_traits.h`, `dprint.h`, `compute/{matmul,eltwise_binary,mask,reduce,tile_move_copy,bcast}.h` | 1 — LLK / HAL | ✓ |

#### Per-call detail

| Donor | Function called | Handle shapes in signature | Status |
|---|---|---|---|
| `kernel/dataflow/generate_mm_scaler.hpp:12` | `generate_mm_scaler(DataflowBuffer cb, uint32_t scaler)` | `DataflowBuffer` **by value** | ✓ — porter passes `DataflowBuffer(dfb::name)` (`migration_guide.md:598, 786`) |
| `kernel/dataflow/moreh_common.hpp:223` | `generate_mask_w<T>(DataflowBuffer cb_mask, uint32_t mask_w)` | `DataflowBuffer` by value | ✓ |
| `kernel/dataflow/moreh_common.hpp:183` | `generate_mask_h<T>(DataflowBuffer cb_mask, uint32_t mask_h)` | `DataflowBuffer` by value | ✓ |
| `kernel/dataflow/moreh_common.hpp:98` | `fill_cb_with_value(DataflowBuffer cb, uint32_t value, int32_t num_of_elems)` | `DataflowBuffer` by value | ⭐ signature fine; **body carries the Device 2.0 gate** (`:100`) |
| `kernel_lib/reduce_helpers_dataflow.hpp:85` | `calculate_and_prepare_reduce_scaler<dfb_id, pool_type, reduce_dim, reduce_factor>(uint32_t)` | DFB identity as a `constexpr uint32_t` **NTTP** | ✓ OK — `dfb::name`'s constexpr cast covers template-parameter position |
| `kernel_lib/reduce_helpers_compute.hpp` | `compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, in_dfb, scaler_dfb, out_dfb>(shape, layout, Accumulate::at(cb, iter))` | DFB identities as `constexpr uint32_t` NTTPs; `Accumulate::at` takes a `uint32_t` CB id | ✓ OK |
| `kernel/compute/moreh_common.hpp:28, 35, 42, 121` | `pack_tile_with_dt(uint32_t, DataflowBuffer)`, `copy_tile_init_with_dt(DataflowBuffer, uint32_t)`, `add_tiles_init_with_dt(DataflowBuffer, DataflowBuffer)`, `mul_tiles_bcast_scalar_init_short_with_dt(DataflowBuffer, DataflowBuffer)` | `DataflowBuffer` by value throughout | ✓ |

The recipe's donor-shape table has no `DataflowBuffer` row — only `CircularBuffer` (⭐ flag) and `uint32_t cb_id` (✓ OK). Every donor in this op takes the *post-migration* `DataflowBuffer`, which the shared migration guide shows constructing directly from a `dfb::` accessor, so I scored these ✓. Logged in *Recipe notes*.

**Host-side out-of-directory dependency** (outside the coupling subject's kernel-`#include` scope, noted for completeness): all three factories and the device-op include `ttnn/operations/moreh/moreh_helper_functions.hpp` for `split_work_to_cores_wt_core_range`, `check_tensor`, `validate_input_with_dim`, `validate_output_with_keepdim`; the W and H factories also pull `ttnn/operations/reduction/generic/device/reduce_op.hpp` for `reduce_op_utils::get_defines`. These are ordinary host helpers and translate unchanged.

#### Borrowed kernel files (file-path instantiation)

**None.** All six distinct `kernel_source` paths across the three factories point into `ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/`, and all eight files in that directory are referenced. No shared-pool kernel is instantiated by path, so this op imposes **no port-together coupling** on any other op.

### Relaxation candidates mined from a custom hash

None — the op has no custom hash to mine.

### TTNN factory analysis

Sheet-derived facts could not be retrieved this run (see *Gate detail*). Code-side facts, with evidence:

- **Concept:** `descriptor`, uniformly across all three factories.
- **Op-owned tensors:** none — structurally impossible on the `descriptor` concept, and no `buffers` vector exists.
- **MeshWorkload need:** none — no `WorkloadDescriptor` return anywhere.
- **Pybind `create_descriptor`:** absent (`moreh_mean_nanobind.cpp:19-30`).
- **Other risky pybind:** none — the binding exposes only scalars, an optional output tensor, a memory config and a compute-kernel config.
- **Custom hash:** absent (default hash over `operation_attributes_t`).
- **`get_dynamic_runtime_args`:** absent.
- **`override_runtime_arguments`:** absent.
- **Target concept:** `MetalV2FactoryConcept`, no op-owned tensors.

---

## Misc anomalies  *(team-only, non-gating; route to the ops team — the port does not act on these)*

1. **Dead-but-hashed attribute — `divisor`.** `operation_attributes_t::divisor` (`device/moreh_mean_device_operation.hpp:22`) is pybound (`moreh_mean_nanobind.cpp:27`), threaded through `ttnn::moreh_mean` and `ttnn::prim::moreh_mean`, and included in the default program hash — yet `validate_tensors` hard-rejects any value: `TT_FATAL(operation_attributes.divisor.has_value() == false, "divisor not supported yet.")` (`device/moreh_mean_device_operation.cpp:23`). No factory reads it. It is therefore a permanently-`nullopt` field that still participates in the cache key. Exactly the *"attribute forced or ignored in the factory yet still fed to `compute_program_hash`"* shape.
2. **Suspicious hardcoded constant — the vestigial `NC` loop.** Both the W and H factories pass a literal `1` as the third compute CTA (`moreh_mean_w_program_factory.cpp:166-171` and `:187-192`; `moreh_mean_h_program_factory.cpp:162-167` and `:183-188`). The kernels read it as `NC` (`moreh_mean_w.cpp:18`, `moreh_mean_h.cpp:18`) and wrap their whole body in `for (uint32_t nc = 0; nc < NC; nc++)` — a loop that can only ever execute once. Dead structure left from an earlier work-split; harmless but misleading, and it costs a level of indentation in the two kernels a porter has to read most carefully.
3. **Possible precision asymmetry between the W and H paths under `fp32_dest_acc_en`.** Both factories allocate the `c_24` accumulator with `fp32_dest_acc_en_data_format` (Float32 when enabled) and both set the `FP32_DEST_ACC_EN` define. But **only H** also sets `unpack_to_dest_mode[CBIndex::c_24] = UnpackToDestMode::UnpackToDestFp32` (`moreh_mean_h_program_factory.cpp:152-155`); **W leaves `unpack_to_dest_mode` all-`Default`** (`moreh_mean_w_program_factory.cpp:160`) even though `moreh_mean_w.cpp:104` unpacks that Float32 CB back into DEST via `copy_tile(cb_accum_dst, 0, reduce_dst_idx)`. Same CB, same role, same enable flag, different unpack configuration. **Not verified on hardware** — flagged for the ops team to confirm whether the W path silently loses fp32 accumulation precision, or whether the difference is deliberate.
4. **Inconsistent `constexpr` on CTA reads.** In `moreh_mean_w.cpp:16-18` and `moreh_mean_h.cpp:16-18`, `Ht` / `Wt` / `NC` are read into plain `uint32_t` locals while `origin_W` / `origin_H` on the next line *are* `constexpr`. The non-`constexpr` reads leave `is_w_single_tile` / `is_h_single_tile` (`moreh_mean_w.cpp:52`, `moreh_mean_h.cpp:50`) as runtime booleans over compile-time-known values, so the single-tile branches are not statically folded. Cosmetic for correctness; it also cost this audit a clean compile-time answer on whether `c_24` is referenced in the single-tile configuration (resolved conservatively as a self-loop).
5. **Unused debug include.** `#include "api/debug/dprint.h"` at `reader_moreh_mean_nc.cpp:5` with no `DPRINT` in the file.
6. **Mask-CB aliases declared outside their `#ifdef`.** `reader_moreh_mean_w.cpp:23` and `reader_moreh_mean_h.cpp:40` declare `constexpr uint32_t cb_id_mask_{w,h} = tt::CBIndex::c_3;` immediately *before* the `#ifdef DO_MASK_*` that is the only user, leaving an unused constant in the no-mask build. Trivial today, but it is the exact line the port has to move if the conditional-binding route is taken (see *CB endpoints*), so it is worth fixing at the same time.

---

## Per-DeviceOperation attribution

Not applicable — a single `DeviceOperation` in this directory. The per-**factory** split that matters is captured throughout: `MorehMeanNCFactory` is RED on Device 2.0; `MorehMeanHFactory` and `MorehMeanWFactory` are clear (with the H-factory caveat recorded in *Gate detail*).

---

## Questions for the user

1. **The readiness sheet was not consulted — please complete this lookup before the subset port starts.** The recipe makes the sheet's `Is able to port?` cell the gate verdict, but the claude.ai Google Drive connector is unauthorized in this session and cannot be authorized from inside one (`ToolSearch` does not resolve `mcp__claude_ai_Google_Drive__download_file_content`), and there is no local CSV under `metal_2.0/analyses/`. All five *code-checkable* conjuncts were verified clean; **`Is safe to port?`** and **`TensorParameter relaxation`** are unread, as is the factory-set staleness cross-check. Since the op has no custom hash, a live relaxation is very unlikely; `Is safe to port?` is the one that could still change the verdict. *(This is the second audit in a row blocked on the same thing — worth authorizing once for the whole batch.)*
2. **Should `get_dataformat(cb_id)` be sanctioned alongside `get_tile_size(cb_id)`?** This is the entire RED. The two are structurally identical — both are free functions the `CircularBuffer` wrapper simply forwards to (`circular_buffer.h:113-115`) — and the only reason I flagged one and not the other is that the recipe's sanctioned list names `get_tile_size` explicitly and says nothing about `get_dataformat`. A ruling either way is cheap and decides whether this op is GREEN today or needs a Device 2.0 round-trip. If it stays unsanctioned, please also decide the `reduce_helpers_dataflow.inl:140` case (constexpr id, no wrapper in scope), which determines whether the clean subset is `{H, W}` or `{W}` alone.
3. **For the mask CBs, do you want the minimal-diff or the clean-end-state resolution?** `c_3` and `c_25` have zero touchers in the no-mask configuration. Keeping their bindings unconditional reproduces today's behaviour exactly with no kernel `#ifdef` churn; the conditional form documented in `migration_guide.md` → *Optional resources* is tidier but requires converting the W kernel's `if (do_mask_w)` and the H kernel's `if constexpr (do_mask_h)` guards to `#ifdef`, and emitting `DO_MASK_*` to the compute kernels as well as the readers. I recommended unconditional in the brief; say the word if you'd rather the porter do the fuller job.

---

## Recipe notes

1. **`get_dataformat(cb_id)` looks like a gap in the sanctioned-free-function list, and it is load-bearing here — it is the sole reason this op is RED.** The Device 2.0 subject sanctions `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)` on the stated grounds that "the Device 2.0 migration guide keeps [them] as free functions in its migrated examples," and its own breadcrumb notes that `CircularBuffer::get_tile_size()` merely forwards to the free function. `get_dataformat(cb_id)` satisfies the identical description — `circular_buffer.h:115` is `DataFormat get_dataformat() const { return ::get_dataformat(cb_id_); }`, one line below `get_tile_size()`'s identical forwarder — yet it is absent from the list and absent from the guide. I followed the list literally (RED), because it is written as closed and current, and because too-lenient GREEN sends a porter into an assumption-violation stop at that exact line. But if the intent was "wrapper-forwarding metadata accessors are all sanctioned," the list should say so as a *category* rather than an enumeration; as written, every new forwarder that ships will produce a spurious RED like this one until someone amends the list.
2. **The donor-shape table has no `DataflowBuffer` row.** It covers `CircularBuffer` / `CircularBuffer&` (⭐ flag) and `uint32_t cb_id` (✓ OK), but every donor this op calls takes `DataflowBuffer` **by value** — the post-CB→DFB shape that the migration sweeps are actively producing across the codebase. I scored these ✓ on the strength of `migration_guide.md:598, 786` (`DataflowBuffer dfb(dfb::my_dfb);`), but that required leaving the audit doc to resolve. A row for `DataflowBuffer` / `DataflowBuffer&` would close it, and it will only become more common.
3. **There is no routing for "the readiness sheet could not be fetched."** The recipe enumerates `yes`, `no`, conflict, and missing-op-row, all of which presuppose a successful fetch. Fetch failure is a distinct and, for headless or non-interactive sessions, entirely predictable state — the fetch doc itself says the connector authorizes only in an interactive main session. I recorded it as an explicit unresolved conjunct rather than defaulting to RED (which misroutes a healthy op to a prereq team) or GREEN (which launders an unchecked gate). A one-line preference in the routing list would remove the judgment call. *(Raised on the previous audit as well; repeating because it recurred immediately.)*
4. **The CB-endpoints classification table has no cell for a CB whose touchers vanish under one config.** The table's `0 touchers` row is titled *"Dead CB — allocated, never referenced"* and its resolution is *drop it*, with a long and well-judged warning about over-calling. But `c_3` / `c_25` here are neither dead nor live: they are fully live under `do_mask_*` and have zero touchers otherwise, so "drop" is flatly wrong and "self-loop" does not apply either. The *Classify per instantiation* paragraph implies the right answer, and `migration_guide.md`'s *Optional resources* section supplies the actual mechanism — but the audit doc never points at it. A fifth row ("0 touchers **in some configs only** → conditional binding, see `migration_guide.md` → Optional resources") would land this cleanly, and this shape is common: any op with an optional mask, scaler, or bias CB has it.
5. **The brief-emission rule still contradicts itself on config-scoped gates.** As on the previous audit: the GATE role bullet says a config-scoped gate "still issues a brief for the clean subset," while *Output: the two documents* and the brief section both say the brief is emitted "only on a fully GREEN audit … On any RED there is no brief." I again followed the specific carve-out and issued a subset-scoped brief with a prominent banner. Two audits in a row have hit this, which suggests config-scoped gates are the common case rather than the exception.
6. **Minor — the "called code vs. included header" boundary is undefined for the Device 2.0 gate.** All three readers here `#include` the header containing the violation, but only one *calls* the offending function. I used the called-code reading (the gate is defined over what the factory "instantiates or calls into"), which clears two of three factories — a materially different verdict from the whole-header reading. One sentence fixing the convention would make this reproducible between auditors.
