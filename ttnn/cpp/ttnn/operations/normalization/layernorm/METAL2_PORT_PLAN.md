# Port Plan — layernorm

Port plan for `ttnn/cpp/ttnn/operations/normalization/layernorm/`, ported from
`ProgramDescriptorFactoryConcept` (`create_descriptor`) to Metal 2.0
(`ProgramSpecFactoryConcept` / `create_program_spec`).

Written during the inventory and planning steps. The op has **two** program
factories under one device-operation; this plan inventories both, then records the
spec plan and the structural findings that bear on completability.

> **Scope note (see METAL2_PORT_REPORT.md → Friction/Open items):** This is an
> unusually large op for a single port pass — two factories (~844 + ~600 + ~1900
> helper lines) and 26 kernel files, with hundreds of legacy-idiom conversion sites.
> Crucially, each factory selects its kernel *source string at runtime* from several
> variants, and the spec's DFB/tensor bindings must match whatever source the factory
> selects — so a factory and **all** kernel variants it can select must port
> **atomically**. This plan documents the full target shape; the report records the
> completability finding.

---

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (both factories define
  `static tt::tt_metal::ProgramDescriptor create_descriptor(...)`).
- Variants: **two factories** under one `program_factory_t` variant on
  `LayerNormDeviceOperation`:
  - `LayerNormMultiCoreProgramFactory` (`device/layernorm_op_multi_core.cpp`) — interleaved.
  - `LayerNormShardedProgramFactory` (`device/layernorm_op_multi_core_sharded.cpp`
    + `device/sharded_layernorm_factory_helpers.{hpp,cpp}`) — sharded (incl. pre/post
    all-gather, welford).
  - `select_program_factory` (in `layernorm_device_operation.cpp`) dispatches to
    sharded when input is sharded, else multi-core.
- Custom `compute_program_hash`: **none** — `LayerNormDeviceOperation` uses the
  default reflection-based hash. (The `compute_program_hash` static at
  `layernorm_nanobind.cpp:253` is a Python *test hook* invoking the framework default;
  not an override — leave it. Audit-confirmed.)

### Non-standard legacy factory signature
Both `create_descriptor` methods take a **fourth parameter**
`const std::optional<CoreRangeSet>& core_range_set = std::nullopt` that the Metal 2.0
`ProgramSpecFactoryConcept` signature does not have. In production the default
(`std::nullopt → default_core_range(device)`) is always used; the explicit override
is exercised only by the `create_descriptor` pybind test hooks
(`layernorm_nanobind.cpp:322,363`). Port resolution: the `create_program_spec`
signature is fixed by the concept to `(attrs, tensor_args, tensor_return_value)`;
the core-range default moves inside the factory body. The pybind `create_descriptor`
hooks lose their target symbol → must be deleted (sanctioned pybind exception).

### Variant: MultiCore (interleaved) factory

Kernel source is selected at runtime (`layernorm_op_multi_core.cpp:483-549`) from:
- **reader**: `reader_unary_interleaved_ln.cpp` (default) |
  `reader_unary_interleaved_ln_rm_gb.cpp` (row-major gamma/beta) |
  `reader_unary_interleaved_ln_large_tensor.cpp` |
  `reader_unary_interleaved_ln_large_tensor_welford.cpp`.
- **writer**: `writer_unary_interleaved_start_id_blocked.cpp` |
  `writer_unary_interleaved_start_id_blocked_rm_output.cpp` (row-major output).
- **compute**: `layernorm.cpp` | `layernorm_welford.cpp` |
  `layernorm_large_tensor.cpp` | `layernorm_large_tensor_welford.cpp`.
  Selection driven by `large_tensor_needed`, `use_welford_and_not_rms_norm`,
  `use_row_major_kernel`, `input_is_row_major`.

#### Kernels (MultiCore)
| unique_id | source (selected) | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | one of 4 reader variants | `all_cores` | block_size, [use_welford], W, TensorAccessorArgs(a/b/gamma/beta), [stick/elem size] | the shared `cb_named_args` block (CB indices) | a_addr, NCHt, Wt, start, packed_one(unused), eps, gamma_addr, beta_addr, b_addr, [H_logical] | none | FUSE_PRE_ADD/FUSE_GAMMA/FUSE_BETA/RMSNORM/TILIZE_IN/ACTIVATION | Reader |
| writer | blocked | rm_output | `all_cores` | block_size, TensorAccessorArgs(output), [elem_size] | `cb_named_args` | dst_addr, Wt, num_tile_rows, start, [H_logical] | none | (none / rm) | Writer |
| compute | one of 4 compute variants | `all_cores` | Wt, block_size, do_gamma, do_beta, fp32_dest_acc_en, then welford-or-not extras (W, TILE_SIZE, rms, fuse) OR (float32_reduction, legacy_rsqrt, W, tile_width) | `cb_named_args` | num_tile_rows_per_core | none | FUSE_PRE_ADD/RMSNORM/TILIZE_IN/UNTILIZE_OUT/ACTIVATION | Compute (math_fidelity, fp32_dest_acc_en, dst_full_sync_en, unpack_to_dest_mode[NUM_CBS], math_approx_mode) |

`cb_named_args` (shared across all three kernels) — the named-CTA CB-index table at
`layernorm_op_multi_core.cpp:448-481`. Maps name → `tt::CBIndex::c_N`. Includes
plain indices (cb_in=c_0, cb_inb=c_1, cb_scaler=c_2, cb_eps=c_3, cb_gamma=c_5,
cb_beta=c_6, cb_out=c_16, cb_ex=c_18, cb_ex2=c_19, cb_xmm2=c_20, cb_ex2pe=c_21,
cb_fusion=c_22, cb_x=c_23, cb_xmm=c_24, cb_reciprocals=c_25, cb_accumulate=c_26,
cb_in_rm=c_27, cb_out_rm=c_28) **plus four computed alias entries**:
- `cb_x_welford` = `welford_fp32_alias ? c_29 : (fuse_pre_add ? c_23 : c_0)`
- `welford_fp32_alias` (a 0/1 *flag*, not a CB index)
- `cb_ex_welford` = `welford_state_fp32_alias ? c_30 : c_18`
- `cb_ex2_welford` = `welford_state_fp32_alias ? c_31 : c_19`
- `welford_state_fp32_alias` (a 0/1 *flag*)

#### CBs (MultiCore) — all `core_ranges = all_cores`, conditionally allocated
| index | size (tiles) | data_format | borrowed | aliased | condition |
|---|---|---|---|---|---|
| c_0 in | in0_t | in_data_format | no | + c_29 (welford_fp32_alias && !fuse_pre_add) | always |
| c_16 out | out0_t | out_data_format | no | no | always |
| c_18 ex | im1_t | cb_data_format | no | + c_30 (welford_state_fp32_alias) | !rms_norm |
| c_2 scaler | in2_t | Float16_b | no | no | !use_welford |
| c_3 eps | in3_t | Float16_b | no | no | always |
| c_19 ex2 | im2_t | cb_data_format | no | + c_31 (welford_state_fp32_alias) | always |
| c_24 xmm | im0_t | cb_data_format | no | no | !rms_norm \|\| fuse_pre_add \|\| large_tensor |
| c_20 xmm2 | im3_t | cb_data_format | no | no | !use_welford |
| c_21 ex2pe | im4_t | cb_data_format | no | no | always |
| c_26 acc | 1 | float32_reduction?F32:cb | no | no | large_tensor && !use_welford |
| c_27 in_rm | in_rm | in_data_format | no | no | input_is_row_major |
| c_28 out_rm | out_rm | out_data_format | no | no | input_is_row_major |
| c_22 fusion | im5_t | cb_data_format | no | no | gamma \|\| beta |
| c_5 gamma | in5_t | gamma_cb_data_format | no | no | gamma |
| c_6 beta | in6_t | beta_cb_data_format | no | no | beta |
| c_23 x (post-add) | im6_t | cb_data_format | no | + c_29 (welford_fp32_alias) | b (fused) && !rms_norm |
| c_1 inb | in1_t | inb_data_format | no | no | b (fused) |
| c_25 reciprocals | recip bytes | Float32 | **recip_tensor->buffer()** | no | use_welford |

#### Semaphores (MultiCore)
none.

#### Tensor accessors (MultiCore)
| host site | originating Tensor | RTA slot | kernel accessor |
|---|---|---|---|
| `:187` a_addr → reader arg[0] | input (a) | reader[0] | `TensorAccessor(src0_args, src_addr)` |
| `:188` b_dram_addr → reader arg[8] | residual (b) | reader[8] | `TensorAccessor(src1_args, b_addr)` |
| `:189` gamma_dram_addr → reader arg[6] | gamma | reader[6] | `TensorAccessor(gamma_args, gamma_addr)` |
| `:190` beta_dram_addr → reader arg[7] | beta | reader[7] | `TensorAccessor(beta_args, beta_addr)` |
| `:131` dst_addr → writer arg[0] | output | writer[0] | `TensorAccessor(dst_args, dst_addr)` |
| (use_welford) recip via borrowed CB c_25 | recip_tensor | n/a (borrowed DFB) | DFB c_25 |

All Case 1 (plain page-by-page iteration). No Case 2.

#### Work split (MultiCore)
- Driver: `split_work_to_cores(requested_cores, num_tile_rows, true /*row_wise*/)`
  at `:197-203`.
- Produces `(num_cores, all_cores, core_group_1, core_group_2,
  num_tile_rows_per_core_group_1, num_tile_rows_per_core_group_2)`.
- **All three kernels use `core_ranges = all_cores`** (one `KernelDescriptor` each).
  The per-group difference (`num_tile_rows_per_core`) is delivered as a **per-core
  RTA**, not a per-group CTA. So there is **no CTA work-split multiplicity** — the
  Demoting-CTA-to-RTA anti-pattern does not apply here; the legacy already uses an
  RTA for the per-core count. One `KernelSpec` per kernel is correct.

### Variant: Sharded factory (inventory summary)
`layernorm_op_multi_core_sharded.cpp` + `sharded_layernorm_factory_helpers.{hpp,cpp}`.
Builds the descriptor via helper functions `add_kernel_descriptors` (`:782`),
`add_cb_descriptors` (`:974`). Sub-paths: standard sharded, pre-all-gather,
post-all-gather, welford; with reshard. Findings from audit:
- **3 program-scope semaphores** (`layernorm_op_multi_core_sharded.cpp:219,224,229`,
  `initial_value=0`). → 3 `SemaphoreSpec`s + `SemaphoreBinding`s on the mcast
  sender/receiver kernels.
- **Borrowed-memory CBs**: c_0 (a_buffer), c_1 (b_buffer), c_7 (stats_buffer,
  post-all-gather), c_16/c_17 (output/output_reshard), c_25 (recip) —
  `sharded_layernorm_factory_helpers.cpp:1006,1024,1032,1185,1198,1226,1245`.
  → `DataflowBufferSpec::borrowed_from`.
- **Case-1 bindings**: gamma, beta (writer RTA → `ta::gamma`/`ta::beta`).
- **Aliased CBs** (welford-fp32): c_0→c_29, c_24→c_29
  (`:1007-1012,1066-1071`).
- **mcast** reader/writer kernels (sender/receiver) + reshard writer.
- **RTA-vararg-shaped read**: `writer_unary_sharded_ln.cpp:38` reads a
  runtime-known-count `segment_args` block via `get_arg_addr(9)` + counted loop.
  Supported; keep as counted L1 read or move to named — non-gating.

Full per-kernel/per-CB sharded inventory is deferred — see report (the sharded
factory + 1900-line helpers is the larger half of the port).

### Cross-op kernels
**None.** All kernel `.cpp` files instantiated by both factories live under
`layernorm/device/kernels/` (layernorm-owned). Shared *headers* come from
`ttnn/cpp/ttnn/kernel_lib/`, `ttnn/kernel/`, and in-family `kernel_util/` — all
Device-2.0-clean and out of porter scope. No fork/in-place cross-op kernel decision
needed.

### Flags
- `reader_unary_interleaved_ln.cpp:33` arg[4] `packed_one_value` is a **dead RTA**
  (documented "legacy; unused, scaler generated in-kernel") but the host still
  computes/passes it (`:553-554,591`). Route to op owner; do not "fix" in port.
- `get_tile_size(cb_id)` holdovers across dataflow kernels (audit table) — Device 2.0
  cleanup track, **not** absorbed by this port.

---

## TTNN ProgramFactory
- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` /
  `MaximizeCacheReuse` (default).
- **Custom `compute_program_hash`**: none.
- **Implementation notes**:
  - `create_descriptor(attrs, args, ret, core_range_set=nullopt)` →
    `create_program_spec(attrs, args, ret)`; the `core_range_set` override drops
    (production always used the default). Two `create_descriptor` pybind hooks
    (`layernorm_nanobind.cpp:322,363`) must be deleted.
  - The factory returns `ttnn::device_operation::ProgramArtifacts{.spec=…,
    .run_params=…}`. Adapter calls it per `mesh_device_operation_adapter.hpp:728`,
    stamps via `MakeProgramFromSpec`, resolves `TensorArgument`s by `MeshTensor`
    identity, refreshes via `UpdateTensorArgs` on cache hit.

## Planned Spec Shape (MultiCore factory)
- **KernelSpecs**: 3 (reader, writer, compute) — one each, source selected at
  construction time as today (Pattern: Multi-variant factories, but realized as a
  single spec with a runtime-chosen source string, not separate specs).
- **DataflowBufferSpecs**: one per allocated CB (see CB table), conditionally added to
  `ProgramSpec::dataflow_buffers` exactly as the legacy conditionally `push_back`s the
  `CBDescriptor`. Borrowed: c_25 (`borrowed_from = RECIP`). Aliased pairs:
  (c_0,c_29), (c_18,c_30), (c_19,c_31), (c_23,c_29) under the welford-fp32 flags —
  one DFB per index, mutual `advanced_options.alias_with`.
- **SemaphoreSpecs**: none.
- **TensorParameters**: input(a), residual(b), gamma, beta, output, recip — declared
  conditionally where the tensor is optional. Bound: a/b/gamma/beta → reader;
  output → writer; recip → borrowed DFB c_25 backing (no kernel `ta::` access — recip
  is consumed via the DFB, so it's a `TensorParameter` referenced by `borrowed_from`
  with **no** `TensorBinding`).
- **WorkUnitSpecs**: 1 — {reader, writer, compute} on `all_cores`.

## Preserved Multiplicity
none — no work-split CTA multiplicity in legacy (per-core count is already an RTA;
all kernels span `all_cores`).

## Dropped Plumbing (MultiCore factory)
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA[0] a_addr (`:587`) | `a.buffer()->address()` | `TensorBinding(INPUT,"input")` → `ta::input` |
| reader RTA[8] b_addr (`:595`) | `b->buffer()->address()` | `TensorBinding(RESIDUAL,…)` (conditional) |
| reader RTA[6] gamma_addr | `gamma->buffer()->address()` | `TensorBinding(GAMMA,…)` (conditional) |
| reader RTA[7] beta_addr | `beta->buffer()->address()` | `TensorBinding(BETA,…)` (conditional) |
| writer RTA[0] dst_addr (`:604`) | `output.buffer()->address()` | `TensorBinding(OUTPUT,"out")` → `ta::output` |
| reader CTA TensorAccessorArgs(a/b/gamma/beta) (`:358-361`) | `TensorAccessorArgs(buf).append_to(cta)` | binding mechanism (auto-packed) |
| writer CTA TensorAccessorArgs(output) (`:378`) | same | binding mechanism |
| reader kernel `TensorAccessorArgs<3>()` chain (`:88-91`) | manual CTA-offset chaining | `TensorAccessor(ta::input)` etc. |
| `cb_named_args` CB-index entries | named CTA = `CBIndex::c_N` | `DFBBinding` → `dfb::name` |
| reader RTA[4] packed_one_value | dead RTA | **leave** (op-owner finding, not port-dropped) |
| positional compute/reader/writer CTAs | `get_compile_time_arg_val(N)` | named CTAs `get_arg(args::name)` |
| reader/writer/compute RTAs | `get_arg_val<uint32_t>(N)` | named RTAs schema + values |

## Applied Patterns
- **Multi-variant factories** — kernel source selected at construction time (realized
  as one spec with a runtime-chosen source string).
- **Aliased DFBs** — welford-fp32: (c_0|c_23)↔c_29, c_18↔c_30, c_19↔c_31. Mutual
  `advanced_options.alias_with`, matching `unpack_to_dest_mode` on the compute spec.
- **Conditional / optional DFB bindings** — many CBs conditional (gamma/beta/inb/x/
  recip/in_rm/out_rm); kernel `#ifdef` gating already present (FUSE_GAMMA etc.) — the
  host conditionally binds and emits the matching define (already done for FUSE_*).
- **Borrowed-memory DFB** — recip c_25 `borrowed_from = RECIP`.
- **Self-loop DFB binding** — likely on compute for accumulator CBs (c_18/c_19/c_24
  read-modify-write); to confirm during construction.

## Deferred / Flagged
- **Kernel-side CB aliasing-by-name** (`cb_x`, `cb_x_welford`, `cb_ex_welford`,
  `cb_ex2_welford`): these are kernel-local names that resolve at compile time to the
  *same* CB index as another binding (or to a real alias index c_29/30/31). In legacy
  this is a named-CTA carrying a CB index computed host-side. In Metal 2.0 there is no
  "named CTA that is a CB index"; CB identity must come from a `DFBBinding` →
  `dfb::name`. The `cb_x = fuse_pre_add ? cb_x_idx : cb_in` fallback has no direct
  Metal 2.0 expression — `dfb::cb_x` only exists if the host binds a DFB named cb_x,
  but on the non-fused path cb_x should *be* dfb::cb_in (same buffer). Candidate
  resolutions and the open question are in the report (this is the highest-value
  structural finding).
- Sharded factory full inventory — deferred (see report).
- `get_named_compile_time_arg_val` is a legacy named-CTA mechanism; Metal 2.0 kernels
  use `get_arg(args::name)` for scalars and `dfb::name` for CB identity. Every
  `get_named_compile_time_arg_val("cb_*")` site must convert to `dfb::*`. See report.
