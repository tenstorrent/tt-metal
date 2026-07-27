# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_sgd`

Single device operation, single program factory (`descriptor` concept):

- **`MorehSgdOperation`** (`device/moreh_sgd_device_operation.{hpp,cpp}`)
  - `create_descriptor` in `device/moreh_sgd_program_factory.cpp` — one `ProgramDescriptor` factory.

Kernels (all owned by the op, under `device/kernels/`, all referenced by the factory):
- `reader_moreh_sgd.cpp` (data movement)
- `writer_moreh_sgd.cpp` (data movement)
- `moreh_sgd.cpp` (compute)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_sgd` |
| **Overall** | **GREEN** (pending readiness-sheet confirmation — see Questions) |
| **DOps / Factories** | `MorehSgdOperation` → `create_descriptor` (`moreh_sgd_program_factory.cpp`) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — kernels already on `DataflowBuffer` / `Noc` / `TensorAccessor`; shared `moreh_common.hpp` helpers take `DataflowBuffer` objects |
| *Prereqs* — Cross-op escapes | Ok — shared-lib pool (`ttnn/cpp/ttnn/kernel/`) only, already Device 2.0; no file-path borrows |
| *Feature Support* — overall | **GREEN** |
| *Feature Support* — GlobalCircularBuffer / address_offset / GlobalSemaphore / CTA-varargs | Ok — none present |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (code cross-check; sheet not fetched — see Questions) |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes — no smuggled raw pointers; tensor buffers ride the sanctioned `Buffer*`-binding form |
| *TTNN Readiness* — Custom hash | No (no `compute_program_hash` override) |
| *TTNN Readiness* — Runtime-args update | No (no `override_runtime_arguments` / `get_dynamic_runtime_args`) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (nanobind binds `&ttnn::moreh_sgd`, not `create_descriptor`) |
| *TTNN Readiness* — Op-owned tensors | No (`descriptor` concept) |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none (clean bases; `Buffer*` delivery, no host-folded offsets) |
| *Port work* — Tensor bindings (per binding) | all Case 1 (`TensorAccessor`) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (no accessor passes a 3rd arg) |
| *Port work* — CB endpoints | 6 legal 1:1 · 4 intermediates self-loop (config-gated) |

## Result

**GREEN → brief issued.** `moreh_sgd` is a clean `descriptor`-concept op whose kernels are already Device 2.0 (they use `Noc`, `DataflowBuffer`, `TensorAccessor`). No feature-compat gate fires, no offset-base-pointer fold, no 3rd-arg page-size override, no custom hash / runtime-args-update / pybind-descriptor. All five tensor bindings are Case 1 (fed to `TensorAccessor`), delivered today via the sanctioned `Buffer*`-binding form. CB endpoints are all resolvable at port time (no gate). Target concept: `MetalV2FactoryConcept`.

**One caveat (not a code blocker):** the authoritative TTNN per-factory readiness sheet was **not fetched** (Drive connector is main-session-only; this audit ran in an agent context). The `Is able to port?` verdict above is a *code cross-check*, not the sheet's cell. Confirm the sheet row before the port session — see Questions.

**Tensorless-dispatch check (orchestrator ask):** **NOT tensorless — safe.** `tensor_args_t` carries two non-optional required input tensors, `param_in` and `grad` (`device/moreh_sgd_device_operation.hpp:27-28`). A device is always available for the MetalV2 factory adapter to source the `MeshDevice` from (`param_in`/`grad`). The three optional tensors (`momentum_buffer_in`, `param_out`, `momentum_buffer_out`) never make dispatch tensorless. No framework BLOCK on this axis.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN by code cross-check. Concept = `descriptor` (`create_descriptor` returns `ProgramDescriptor`, `moreh_sgd_program_factory.cpp:25`). All gate conjuncts confirmed absent in code: no `compute_program_hash` override, no `override_runtime_arguments` / `get_dynamic_runtime_args`, no `create_descriptor` pybind (nanobind binds `&ttnn::moreh_sgd`, `moreh_sgd_nanobind.cpp:22`). `Is safe to port?`: the factory passes tensor buffers as `Buffer*` (`param_in.buffer()`, `grad.buffer()`, `momentum_in_buf`, `param_out.buffer()`, `momentum_out_buf` — `moreh_sgd_program_factory.cpp:266-279`), the framework's sanctioned binding-injection form that re-patches on cache hits (the factory comment at lines 248-253 states this intent), so no smuggled stale-pointer hazard. **Sheet not fetched — confirm the row (Questions).**
- **Device 2.0 (every kernel used):** GREEN. All three kernels are structurally Device 2.0:
  - `reader_moreh_sgd.cpp` — `Noc noc; noc.async_read(...)`, `DataflowBuffer` objects (`reserve_back`/`push_back`), `TensorAccessor(args, addr)`. Only free-function lookup is `get_tile_size(cb_id)` (lines 62-66) — **sanctioned**, not a holdover.
  - `writer_moreh_sgd.cpp` — `Noc noc; noc.async_write(...)`, `DataflowBuffer` (`wait_front`/`pop_front`), `TensorAccessor`. `get_tile_size(cb_id)` (lines 39-42) — sanctioned.
  - `moreh_sgd.cpp` (compute) — `DataflowBuffer` objects throughout; delegates to `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` helpers that take `DataflowBuffer` params (`mul_tiles_to_cb`, `add_tiles_to_cb`, `sub_tiles_to_cb`, `copy_tile_to_cb`). Compute-side `.get_id()` feeding LLK primitives (`mul_tiles`, `pack_tile_with_dt`) and `get_dataformat(cb.get_id())` are ordinary compute-LLK idioms, not data-movement holdovers.
  - Shared donor `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` — `fill_cb_with_value(DataflowBuffer cb, ...)` uses `cb.reserve_back(1)` / `cb.push_back(1)` / `get_dataformat(cb.get_id())`; Device 2.0 native.
- **Feature compatibility:** all Appendix A entries `N/A` — no feature present.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `global_circular_buffer` field, no `remote_index`, plain `CBDescriptor` literals only |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `address_offset` set anywhere (all CBDescriptors default to 0) |
  | GlobalSemaphore | N/A | op uses no semaphores at all |
  | Variable-count compile-time arguments (CTA varargs) | N/A | compute reads a single fixed CTA `get_compile_time_arg_val(0)` (`moreh_sgd.cpp:40`); reader/writer CTAs are fixed `TensorAccessorArgs` runs; `tensor_args_t` is a fixed named-tensor set, no `std::vector<Tensor>` |

- **CB endpoints (GATE-free):** per-node census over `all_cores` (compute is split into `core_group_1`/`core_group_2` but same source and same CB set, so endpoint roles are identical). Ten CBs:
  - `c_0` param_in, `c_1` grad, `c_2` momentum_in — reader **produces**, compute **consumes** → **legal 1:1** each.
  - `c_16` param_out, `c_17` momentum_out — compute **produces** (`sub_tiles_to_cb`→param_out; `copy_tile_to_cb`→momentum_out), writer **consumes** → **legal 1:1** each.
  - `c_24` scalar_args — reader **produces** (`fill_cb_with_value` × 5 → `reserve_back`/`push_back`), compute **consumes** (`wait_front(5)`) → **legal 1:1**.
  - `c_25` tmp1, `c_26` tmp2, `c_27` tmp3, `c_28` tmp4 — **only the compute kernel touches them** (produce + consume within compute) → **single toucher → self-loop** (bind compute as PRODUCER and CONSUMER). See the config note below.
  - No dead CB (every index is referenced under at least one config), no multi-binding.
- **Offset base pointers:** GREEN. Every address delivered to a kernel is a clean base — the factory pushes `Buffer*` values (or `nullptr` for absent optionals), never a `buffer()->address() + <offset>` fold. No Type 1/2 site. (No Type 3 `address_offset`; no Type 4 `narrow`.)
- **TensorAccessor 3rd argument:** GREEN. Every `TensorAccessor` is 2-arg (`TensorAccessor(args, addr)`) — reader (`reader_moreh_sgd.cpp:43,45,49`), writer (`writer_moreh_sgd.cpp:29,34`). No explicit page-size argument anywhere. N/A.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding): all **Case 1** (via `TensorAccessor`) — `param_in` (c_0), `grad` (c_1), `momentum_buffer_in` (c_2, optional), `param_out` (c_16), `momentum_buffer_out` (c_17, optional). Each is delivered today via the `Buffer*`-binding form; express each as a `TensorParameter` / `TensorBinding`, kernel builds `TensorAccessor(tensor::name)`, and the address-via-RTA + `TensorAccessorArgs` plumbing disappears. The optional bindings are gated by the `MOMENTUM` / `MOMENTUM_INITIALIZED` compile defines — keep that conditionality.
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** self-loop `c_25`/`c_26`/`c_27`/`c_28` (single toucher: compute); all others legal 1:1. **Config note:** the four intermediates are touched only under specific compile-define configs (e.g. under the minimal config `weight_decay==0 && momentum==0`, only `c_27` is exercised; `c_25`/`c_26`/`c_28` see zero touches). Classify per `(CB, config)` and self-loop where live; where an intermediate is untouched in a given config it is effectively a 0-toucher for that config. The compute kernel currently constructs all four `DataflowBuffer` wrapper objects unconditionally (`moreh_sgd.cpp:23-30`) regardless of config — factor that into how the per-config bindings are expressed.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no hidden second writer, no multi-reader. All non-1:1 CBs are single-toucher self-loops.
- **Cross-op / shared kernels:** the kernels `#include` the shared `ttnn/cpp/ttnn/kernel/{dataflow,compute}/moreh_common.hpp` pool (function-call escape; no file-path kernel borrows). These helpers already take `DataflowBuffer` objects, so they bind cleanly — but a Metal 2.0 rewrite of any moreh_common helper is a **single shared rewrite** that every moreh op instantiating it must adopt together. See Team-only for the coupling inventory.
- **RTA varargs:** none — reader/writer read a fixed run of distinct named fields via a top-of-kernel `i++` counter (the preferred nameable case, not a vararg loop). The porter names each RTA.

## Team-only

- **Out-of-directory coupling & donor shape:**
  - **Op-level roll-up: ✓ clean.** No borrowed (file-path-instantiated) kernels — the factory instantiates only the op's own `device/kernels/*.cpp`. Function-call escapes go to shared-lib pools only, all Device 2.0 native.
  - **Summary table:**

    | Op kernel | Donor `#include` | Donor class | Status |
    |---|---|---|---|
    | reader / writer / compute | `api/dataflow/{noc,dataflow_buffer,dataflow_api}.h`, `api/tensor/noc_traits.h` | `tt_metal/*` (HAL/LLK) | ✓ no concern |
    | reader / writer | `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` | shared-kernel pool (`ttnn/kernel/`, singular) | ✓ `DataflowBuffer`-param helpers |
    | compute | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | shared-kernel pool | ✓ `DataflowBuffer`-param helpers |

  - **Host-side (informational):** the factory + device-op `#include` `ttnn/cpp/ttnn/operations/moreh/moreh_helper_functions.hpp` (in-family host helpers: `check_tensor`, compute-kernel-config args). Host coupling, not a kernel escape.
  - **Port-together set:** any moreh op that also instantiates `moreh_common.hpp` compute/dataflow helpers shares a Metal 2.0 rewrite of those helpers — sequence that rewrite as one unit across the moreh family.
- **Relaxation candidates:** none (no custom hash to mine).
- **TTNN factory analysis:** concept `descriptor` → target `MetalV2FactoryConcept`; no op-owned tensors (no `buffers` vector — `descriptor` concept can't carry them); no MeshWorkload; no pybind `create_descriptor`; no custom hash; no custom `override_runtime_arguments`. Outputs (`param_out`, `momentum_buffer_out`) are ordinary returned device tensors bound as output `TensorParameter`s, not op-owned.

## Misc anomalies

- **Optional-output shape coupling.** `momentum_buffer_out` is produced only when `momentum != 0` (`moreh_sgd_device_operation.cpp:55-63`), and the writer/compute momentum paths are gated by the `MOMENTUM` define. This is consistent across host + kernel, not a bug — noted so the porter keeps the optional binding + define conditionality aligned.
- No dead RTAs, no dead-but-hashed attributes (no custom hash), no suspicious hardcoded page-size constants observed.

## Questions for the user

1. **Readiness-sheet confirmation (non-blocking):** the authoritative TTNN per-factory *"Operations analysis"* sheet was **not fetched** — the Drive connector authorizes only in the main session and this audit ran in an agent context (the recipe explicitly forbids fetching/cross-checking in a subagent). The GREEN verdict rests on a full **code cross-check** of every `Is able to port?` conjunct (all clear: `descriptor` concept, no custom hash, no runtime-args update, no pybind `create_descriptor`, `Buffer*`-form bindings so `Is safe to port?` holds). Please confirm `moreh_sgd`'s sheet row reads `Is able to port? == yes` (and note any `TensorParameter relaxation` the sheet proposes) before starting the port session. If the sheet has no row or disagrees, that flips to a "spreadsheet-broken" GATE routed to the readiness-sheet owner.

## Recipe notes

- **Subagent vs. authoritative-sheet tension.** The TTNN-factory-concept subject makes the readiness sheet the authoritative gate *and* says "Do NOT fetch or cross-check in a subagent." An audit dispatched to an agent therefore cannot satisfy the gate as written — it can only code-cross-check and defer sheet confirmation to the human. The recipe could state explicitly how to render the verdict in that case (I chose: GREEN-on-cross-check + a non-blocking confirmation Question, rather than a "spreadsheet-broken" GATE, since the op has no row-absence evidence and every code-checkable conjunct is clear).
