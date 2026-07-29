# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/moreh/moreh_norm`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ *(code cross-check only — see below)* · Offset base pointers ✓ · TensorAccessor 3rd arg ✓ *(no sites)*

**Recipe docs:** `9bba65ffd6b 2026-07-29 docs(metal_2.0): follow main's MetalV2FactoryConcept -> ProgramSpecFactoryConcept rename` *(carry this line into the port report's Provenance section)*

> ### ⚠ Read this before you open a file: there are two kernel trees, and one is dead
>
> | Path | Status |
> |---|---|
> | `device/ord_other/moreh_norm_{w,h,nc}/kernels/` — 9 files | **LIVE — port these** |
> | `device/moreh_norm_h/kernels/moreh_norm_h_kernel.cpp` | dead — **do not touch** |
> | `device/moreh_norm_w/kernels/moreh_norm_w_kernel.cpp` | dead — **do not touch** |
> | `device/moreh_norm_other/kernels/moreh_norm_other_kernel.cpp` | dead — **do not touch** |
>
> No factory references the three dead files; only a CMake install glob still copies them. **Two of them share a basename with a live kernel** (`moreh_norm_h_kernel.cpp`, `moreh_norm_w_kernel.cpp`), and the Device 2.0 sweeps migrated them as though live, so they read as current code. A diff against a dead file compiles, installs, and changes nothing. **Every live path contains `ord_other/` — check for it before editing.**

> **One open item before you commit.** The readiness sheet could not be fetched during the audit (Google Drive connector unauthorized in a non-interactive session). All six code-checkable conjuncts of the TTNN factory-concept gate were verified clean, but the sheet-owned **`Is safe to port?`** call is unread. Confirm it reads `yes` for the three factory rows. (No code evidence points at a problem: there is no `->address()`-in-RTA smuggled pointer anywhere in the op.)

**Shape of the job:** three factories, nine live kernels, two tensors, both Case 1 everywhere. No semaphores. No sharding. No borrowed-memory (Buffer-backed) CBs. No shared kernels — the nine live files are op-exclusive, so nothing to co-port. All three factories are the same pipeline (reader fills a `one` tile + streams input tiles → compute applies `f(x)`, accumulates along the reduced dim, reduces → writer drains), which makes the second and third factory largely a repeat of the first.

| Factory | Selected when | Live kernels (all under `device/ord_other/`) |
|---|---|---|
| `ProgramFactoryWOther` (`device/ord_other/moreh_norm_program_factory_w_other.cpp`) | `dim == rank-1` | `moreh_norm_w/kernels/{reader_moreh_norm_w, moreh_norm_w_kernel, writer_moreh_norm_w}.cpp` |
| `ProgramFactoryHOther` (`device/ord_other/moreh_norm_program_factory_h_other.cpp`) | `dim == rank-2` | `moreh_norm_h/kernels/{reader_moreh_norm_h, moreh_norm_h_kernel, writer_moreh_norm_h}.cpp` |
| `ProgramFactoryNCOther` (`device/ord_other/moreh_norm_program_factory_nc_other.cpp`) | otherwise | `moreh_norm_nc/kernels/{reader_moreh_norm_nc, moreh_norm_nc_kernel, writer_moreh_norm_nc}.cpp` |

Each factory owns its own copy of all three kernels — no kernel is shared between factories, so a change in one factory's kernels cannot break another.

**Useful context:** the device op is only ever reached for `p ∈ {0, +INF, -INF}`. The host wrapper (`moreh_norm.cpp:29-60`) sends every other `p` through `moreh_abs_pow` + `moreh_sum`. That is why the `IS_ZERO` / `MINUS_INF` / `REDUCE_OP` define matrix in each factory is exhaustive.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — all three factories expose `static tt::tt_metal::ProgramDescriptor create_descriptor(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&)`. They are **nested structs inside `MorehNormOperation`** (`device/moreh_norm_device_operation.hpp:33-52`); the `program_factory_t` variant at `:54` names them.
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · `get_dynamic_runtime_args` · `override_runtime_arguments` · pybind `create_descriptor` · other migration-risky pybind. All verified in the code; `moreh_norm_nanobind.cpp:38-49` binds only the public `ttnn::moreh_norm` free function, and `device/moreh_norm_device_operation.hpp:56-60` is the complete static-hook set.
- **All three factories convert together** — no mixed-concept transition state. `validate_inputs`, `select_program_factory` (`device/moreh_norm_device_operation.cpp:43-54`), `compute_output_specs`, and `create_output_tensors` are untouched by the port.
- **`tensor_args_t` carries an optional output** (`{const Tensor& input; const std::optional<Tensor>& output;}`). When the caller supplies one, `create_output_tensors` returns it verbatim (`device/moreh_norm_device_operation.cpp:93-101`); otherwise a fresh tensor is allocated. Either way the factory sees a concrete `tensor_return_value_t& output` and binds `output.buffer()` — one `TensorParameter` for the output in both cases.

## Construct — to do

**Tensor bindings** — **both bindings are Case 1** in all three factories. No Case 2 and no borrowed-memory DFB anywhere, so you need no `get_bank_base_address` bridge.

Each tensor reaches its kernel as a typed `Buffer*` pushed into `KernelDescriptor::emplace_runtime_args` (the framework's interim `BufferBinding` mechanism) and is fed straight into a `TensorAccessor`. For each: express it as a `TensorParameter` / `TensorBinding`, have the kernel build `TensorAccessor(tensor::name)`, and delete both the `Buffer*` RTA and the `TensorAccessorArgs(...).append_to(...)` CTA plumbing.

| Factory | Binding | Host sites to remove | Kernel sites to convert |
|---|---|---|---|
| W | `input` | `..._w_other.cpp:264` (RTA arg 0), `:159` (`TensorAccessorArgs`) | `reader_moreh_norm_w.cpp:12` (`input_addr`), `:24` (`input_args`), `:25` (construction) |
| W | `output` | `..._w_other.cpp:274` (RTA arg 0), `:161` | `writer_moreh_norm_w.cpp:14, 23, 24` |
| H | `input` | `..._h_other.cpp:246` (RTA arg 0), `:152` | `reader_moreh_norm_h.cpp:12, 25, 26` |
| H | `output` | `..._h_other.cpp:256` (RTA arg 0), `:154` | `writer_moreh_norm_h.cpp:14, 22, 23` |
| NC | `input` | `..._nc_other.cpp:238` (RTA arg 0), `:139` | `reader_moreh_norm_nc.cpp:12, 24, 25` |
| NC | `output` | `..._nc_other.cpp:248` (RTA arg 0), `:141` | `writer_moreh_norm_nc.cpp:14, 22, 23` |

**The CTA lists empty out completely.** `TensorAccessorArgs` is the *only* compile-time arg in the whole op — every reader and writer reads its accessor args as `constexpr auto args = TensorAccessorArgs<0>()`, and all three compute kernels already set `compile_time_args = {}`. So once the bindings are expressed, each reader/writer has no CTAs left and there is no offset arithmetic to preserve (no `next_compile_time_args_offset()` anywhere in this op).

Note on urgency: the `Buffer*` form is the framework's interim marked-pointer mechanism, patched correctly on cache hits today. **Routine port work, not a correctness hazard** — there are no `->address()`-in-RTA smuggled pointers in this op.

**TensorParameter relaxation:** none. The op has no custom hash, so none can be active. *(Sheet column unread — see the open item. If it names one, stop and re-check it against the hash before applying it.)*

**TensorAccessor 3rd arg:** none — all six constructions are the 2-arg form. Nothing to drop.

**CB endpoints:** all legal 1:1 except the compute-private intermediates. No multi-binding flag, no 1P+1C assignment, **no dead-CB drop**, and — unusually for a masked reduction — **no config-dependence**: every disposition below holds in every instantiation. Self-loop these eight (bind the compute kernel PRODUCER **and** CONSUMER):

| Factory | CB | Role | Why one toucher |
|---|---|---|---|
| W | `c_24` | `f(x)` | compute P `moreh_norm_w_kernel.cpp:58, 92`; C `:97, 108, 112, 134` |
| W | `c_25` | accumulator | compute P `:98, 109, 114, 136`; C `:113, 135`, plus the `reduce<>` input at `:140` |
| W | `c_26` | reduce result | compute P as the `reduce<>` output `:140`; C `:145, 160` |
| H | `c_24` | `f(x)` | compute P `moreh_norm_h_kernel.cpp:57, 92`; C `:97, 108, 113, 135` |
| H | `c_25` | accumulator | compute P `:98, 109, 115, 137`; C `:114, 136`, plus the `reduce<>` input at `:141` |
| H | `c_26` | reduce result | compute P as the `reduce<>` output `:141`; C `:146, 161` |
| NC | `c_24` | `f(x)` | compute P `moreh_norm_nc_kernel.cpp:44, 67`; C `:72, 83, 88, 110` |
| NC | `c_25` | accumulator | compute P `:73, 84, 90, 112`; C `:89, 111, 119, 134` |

`c_0` (reader → compute), `c_1` (reader → compute), `c_2` (reader → compute; W and H only) and `c_16` (compute → writer) are plain 1:1 and need no action.

> **The mask CB (`c_2`) is *not* a config-dependent case here** — do not spend time on it. Both the reader's `generate_mask_*` (`reader_moreh_norm_w.cpp:36-39`, `reader_moreh_norm_h.cpp:37-40`) and the compute's `wait_front`/`pop_front` sit behind plain **runtime** `if (do_mask_*)` guards, never `#ifdef`, so both touchers are compiled in every instantiation and it is a boring 1P+1C throughout. (If you have seen `moreh_mean`, this is the contrast case: there the reader's access *was* `#ifdef`-elided, which forced a self-loop in the no-mask config.)

## Watch for

- **Every kernel derives its CB ids from a runtime counter — you cannot substitute a `dfb::name` token at the declaration site.** All six dataflow kernels open with `uint32_t cb_id{0};` (readers) / `uint32_t cb_id{16};` (writers) followed by `const auto cb_id_x = cb_id++;` — `reader_moreh_norm_w.cpp:19-22`, `reader_moreh_norm_h.cpp:20-23`, `reader_moreh_norm_nc.cpp:20-22`, `writer_moreh_norm_{w,h,nc}.cpp:19-21`. The **NC compute kernel** does the same with `std::uint8_t input_id{tt::CB::c_in0}; const auto cb_x = input_id++;` (`moreh_norm_nc_kernel.cpp:12-29`). These are non-`constexpr`, so the ids stay runtime values feeding the low-level `DataflowBuffer(uint16_t)` ctor, and `dfb::name`'s constexpr `operator uint32_t` (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:47-57, 77`) is what makes the assignment legal. The **W and H compute kernels** use `constexpr auto` for the same idiom (`moreh_norm_w_kernel.cpp:14-35`, `moreh_norm_h_kernel.cpp:14-35`) and *can* take named tokens directly — so the two spellings need different treatment within one port.
- **Same-source compute pair over disjoint core groups — keep two `KernelSpec`s.** Each factory emits two compute `KernelDescriptor`s from one source over disjoint node sets (`..._w_other.cpp:200-229`, `..._h_other.cpp:192-221`, `..._nc_other.cpp:176-205`). Ordinary 1:1, not a dual-instance work-split — no co-fill or co-read to hunt. Here the per-group work count is already an **RTA** (`compile_time_args = {}` on both compute descriptors), so the *demoting-per-group-CTA* anti-pattern cannot arise; just don't collapse the two specs into one.
- **RTA varargs:** none. Every kernel reads its args through a running `i++` over a **fixed** run at the top of `kernel_main` — the non-signal case; name each one. Longest list is the NC reader (`reader_moreh_norm_nc.cpp:11-18`): `input_addr` (→ the `input` binding), `input_is_dram`, `num_output_tiles_per_core`, `tile_offset`, `outer_stride`, `num_inner_tiles`, `num_reduced_tiles_along_dim`.
- **The `*_is_dram` RTA is dead in all six dataflow kernels** (`reader_moreh_norm_w.cpp:13`, `reader_moreh_norm_h.cpp:13`, `reader_moreh_norm_nc.cpp:13`, `writer_moreh_norm_{w,h,nc}.cpp:15`) — read into a `const bool` and never used, because the `TensorAccessor` already knows. **Preserve it** (name it, keep the host passing it); removing it is the ops team's call, logged in the audit's Misc anomalies.
- **The six `get_tile_size(cb_id)` free calls** move onto the DFB object per kernel-side whitelist rule 7 — the wrapper exposes `get_tile_size()` (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:167`). Sites: `reader_moreh_norm_w.cpp:45`, `writer_moreh_norm_w.cpp:30`, `reader_moreh_norm_h.cpp:44`, `writer_moreh_norm_h.cpp:27`, `reader_moreh_norm_nc.cpp:34`, `writer_moreh_norm_nc.cpp:27`.
- **Cross-op / shared kernels:** nothing to co-port. The donor headers all cross cleanly — pass `DataflowBuffer(dfb::name)` or `dfb::name` straight through, no donor-side change needed:
  - `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` → `fill_cb_with_value` (all three readers), `generate_mask_w` (W), `generate_mask_h` (H), the `Scalar` union — `DataflowBuffer` by value.
  - `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` → `compute_kernel_lib::reduce<…>` (W, H compute) — `uint32_t` CB-id NTTPs.
  - `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` → the `*_with_dt` helpers (all three compute kernels) — `DataflowBuffer` by value.
- **Leave the known warts alone; preserve current behavior.** The audit logged several for the ops team, not you: NC's `one` tile (`c_1`) is filled by the reader and waited/popped by compute but its data is never used (`moreh_norm_nc_kernel.cpp:37, 137` — the NC path has no `reduce<>`, so it needs no scaler); the compute kernels use the deprecated `tt::CB` enum rather than `tt::CBIndex`; and `get_floored_p_and_decimal_and_p_is_negative` (`device/moreh_norm_device_operation.cpp:14-22`) is dead code in this op. See `METAL2_PREPORT_AUDIT.md` → *Misc anomalies*.
