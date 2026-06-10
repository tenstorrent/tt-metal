# Port Plan — layernorm (`LayerNormDeviceOperation`)

Port plan for `layernorm`, ported from `ProgramDescriptorFactoryConcept` to Metal 2.0
(`ProgramSpecFactoryConcept`). Written during the inventory and planning steps.

> **Scoping headline (read first).** Per the recipe's atomic-unit rule, the port unit is
> *one ProgramFactory + every kernel source it can select at runtime*. For the **multi-core
> (interleaved) factory** that unit is the 843-line `create_descriptor` body **plus 10
> runtime-selectable kernel sources (~3,200 lines)**, all coupled through one shared
> `named_compile_time_args` table and one DFB-binding set, with welford-fp32 aliasing,
> conditional bindings, and same-FIFO aliasing throughout. The **sharded factory** is larger
> still (~2,000 host lines + ~13 sharded/mcast kernels, 3 semaphores, mcast topology). Both
> units exceed a faithful one-pass port. This plan inventories the **multi-core factory**
> completely and lays out the construction blueprint, then records a **grounded stop**
> ([§Grounded stop](#grounded-stop)) — the recipe-sanctioned outcome when a single factory's
> unit is too large to port faithfully in one pass. No code was changed.

This plan covers the **multi-core (interleaved) factory** only. The sharded factory is
enumerated at a coarser grain under [§Sharded factory](#sharded-factory-deferred) as remaining work.

---

## Legacy Inventory — multi-core factory

### Legacy factory shape
- **Concept:** `ProgramDescriptorFactoryConcept` (`create_descriptor` returns
  `tt::tt_metal::ProgramDescriptor`). File: `device/layernorm_op_multi_core.cpp:75`.
- **Variants:** single device-op (`LayerNormDeviceOperation`), two factories in the
  `program_factory_t` variant: `LayerNormMultiCoreProgramFactory` (this plan) and
  `LayerNormShardedProgramFactory` (deferred). `select_program_factory` dispatches sharded
  when the input is sharded, else multi-core.
- **Custom `compute_program_hash`:** **none.** `LayerNormDeviceOperation` defines no override
  (default reflection-based hash). The `compute_program_hash` static at
  `layernorm_nanobind.cpp:253` is a Python test hook calling the framework default — not a
  custom override; **nothing to delete.** (Confirms audit.)
- **Non-standard factory parameter (pybind hook):** `create_descriptor` carries a trailing
  `const std::optional<CoreRangeSet>& core_range_set = std::nullopt` (`.hpp:24`) used in
  production only via `core_range_set.has_value() ? ... : default_core_range(device)`
  (`.cpp:194`). It is driven only by the `create_descriptor` pybind hook
  (`layernorm_nanobind.cpp:322-333`). This is the ttnn-factory doc's **exception 3** (drop the
  parameter, inline the production default, delete the hook).

### Runtime kernel-source selection (the load-bearing inventory fact)

The factory selects its kernel **source file** at runtime from attributes/tensor properties.
This is what makes the unit atomic — all selectable sources flip together.

**Reader** (`.cpp:484-497`), selected by `(large_tensor_needed, use_welford_and_not_rms_norm, use_row_major_kernel)`:
| condition | source |
|---|---|
| `large_tensor_needed && use_welford_and_not_rms_norm` | `reader_unary_interleaved_ln_large_tensor_welford.cpp` |
| `large_tensor_needed && !welford` | `reader_unary_interleaved_ln_large_tensor.cpp` |
| `!large && use_row_major_kernel` (row-major gamma/beta) | `reader_unary_interleaved_ln_rm_gb.cpp` |
| else (default; also handles row-major *input* via `TILIZE_IN` define) | `reader_unary_interleaved_ln.cpp` |

**Writer** (`.cpp:633-637`), selected by `input_is_row_major`:
| condition | source |
|---|---|
| `input_is_row_major` | `writer_unary_interleaved_start_id_blocked_rm_output.cpp` |
| else | `writer_unary_interleaved_start_id_blocked.cpp` |

**Compute** (`.cpp:541-549`), selected by `(large_tensor_needed && (!rm_gb || input_rm), use_welford_and_not_rms_norm)`:
| condition | source |
|---|---|
| large + welford | `layernorm_large_tensor_welford.cpp` (699 lines) |
| large + !welford | `layernorm_large_tensor.cpp` (442 lines) |
| !large + welford | `layernorm_welford.cpp` (430 lines) |
| !large + !welford | `layernorm.cpp` (351 lines) |

**Total selectable sources: 4 readers + 2 writers + 4 compute = 10**, plus shared headers
`layernorm_dataflow_utils.h` (312) and `layernorm_compute_utils.h` (101). All are
layernorm-owned (in-family) — no cross-op kernel. Tests parametrize `use_welford=[True,False]`
and exercise large-tensor + row-major + fused-pre-add + gamma/beta, so **no attribute subset
is untested**: a shippable port of this factory must convert all 10 sources.

### Kernels (multi-core)

The host builds **3 `KernelDescriptor`s** (reader, writer, compute), each with the selected
source above. **All three share one `named_compile_time_args` table** `cb_named_args`
(`.cpp:448-481`) — 24 named CB-index entries plus 4 alias-flag scalars. Several entries are
*conditionally remapped* at host time (`cb_x_welford`, `cb_ex_welford`, `cb_ex2_welford`,
`welford_fp32_alias`, `welford_state_fp32_alias`).

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | (selected, see above) | `all_cores` | `block_size`, `[use_welford if !large]`, `W`, 4×`TensorAccessorArgs(a/b/gamma/beta)`, trailing `elem_size`/`stick_size`/`tile_size` | `cb_named_args` (24+4) | `a_addr, NCHt, Wt, start, packed_one(dead), eps, gamma_addr, beta_addr, b_addr, [H_logical]` | none | `FUSE_PRE_ADD, FUSE_GAMMA, FUSE_BETA, RMSNORM, TILIZE_IN` (conditional) | `ReaderConfigDescriptor{}` |
| writer | (selected) | `all_cores` | `block_size`, `TensorAccessorArgs(output)`, `[elem_size if rm]` | `cb_named_args` | `dst_addr, Wt, num_tile_rows, start, [H_logical]` | none | none | `WriterConfigDescriptor{}` |
| compute | (selected) | `all_cores` | `Wt, block_size, has_gamma, has_beta, fp32_dest_acc_en, {+welford: W,TILE_SIZE,rms,fuse} / {+legacy: float32_reduction,legacy_rsqrt,W,tile_width}` | `cb_named_args` | `num_tile_rows_per_core` | none | `FUSE_PRE_ADD, RMSNORM, TILIZE_IN, UNTILIZE_OUT, ACTIVATION*` (conditional) | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, unpack_to_dest_mode[NUM_CIRCULAR_BUFFERS], math_approx_mode}` |

Notes:
- Reader CTA[1] (`use_welford`) is present only on the non-large path (`.cpp:354-356`) — a
  *positional CTA whose presence depends on the path*, which couples the reader's
  `TensorAccessorArgs<3>()` offset chain to the path.
- `dead RTA`: reader arg[4] `packed_one_value` is documented dead (audit Misc anomaly); the
  port keeps it as the kernel still indexes by position, OR drops it once RTAs go named. Route
  to op owner, not port.

### CBs (multi-core)

Up to **17 CB indices** are configured, most conditionally. One `make_cb_descriptor` lambda
(`.cpp:667`) builds each; the welford-fp32 paths push a *second* `CBFormatDescriptor` onto an
existing descriptor (aliased CB).

| index (CBIndex) | named | total_size | when present | aliased-with | borrowed |
|---|---|---|---|---|---|
| c_0 | cb_in | `in0_t*in_single_tile` | always | c_29 (welford_fp32_alias && !fuse) | — |
| c_1 | cb_inb | `in1_t*inb_single_tile` | `b` (fused) | — | — |
| c_2 | cb_scaler | `in2_t*scaler_tile` | `!use_welford` | — | — |
| c_3 | cb_eps | `in3_t*bf16_tile` | always | — | — |
| c_5 | cb_gamma | `in5_t*gamma_tile` | `gamma` | — | — |
| c_6 | cb_beta | `in6_t*beta_tile` | `beta` | — | — |
| c_16 | cb_out | `out0_t*out_single_tile` | always | — | — |
| c_18 | cb_ex | `im1_t*single_tile` | `!rms_norm` | c_30 (welford_state_fp32_alias) | — |
| c_19 | cb_ex2 | `im2_t*single_tile` | always | c_31 (welford_state_fp32_alias) | — |
| c_20 | cb_xmm2 | `im3_t*single_tile` | `!use_welford` | — | — |
| c_21 | cb_ex2pe | `im4_t*single_tile` | always | — | — |
| c_22 | cb_fusion | `im5_t*single_tile` | `gamma||beta` | — | — |
| c_23 | cb_x | `im6_t*single_tile` | `b && !rms_norm` | c_29 (welford_fp32_alias) | — |
| c_24 | cb_xmm | `im0_t*single_tile` | `!rms_norm||fuse||large` | — | — |
| c_25 | cb_reciprocals | `recip_CB_size` | `use_welford` | — | **recip_tensor->buffer()** |
| c_26 | cb_accumulate | `acc_tile` | `large && !welford` | — | — |
| c_27 | cb_in_rm | `in_rm_size` | `input_is_row_major` | — | — |
| c_28 | cb_out_rm | `out_rm_size` | `input_is_row_major` | — | — |
| c_29/c_30/c_31 | (alias indices) | match primary | welford-fp32 paths | c_0/c_23 ; c_18 ; c_19 | — |

`unpack_to_dest_mode[NUM_CIRCULAR_BUFFERS]` (`.cpp:517`) sets `UnpackToDestFp32` on
c_26 (`float32_reduction`), c_30/c_31 (`welford_state_fp32_alias`), c_29 (`welford_fp32_alias`).

### Semaphores (multi-core)
**none.** (Multi-core factory uses no `SemaphoreDescriptor`. Confirms audit.)

### Tensor accessors (multi-core)
| originating Tensor | host RTA slot | kernel-side construction | classification |
|---|---|---|---|
| input `a` | reader arg[0] (`a_addr`, `.cpp:587`) | `TensorAccessor(src0_args, src_addr)` | **Case 1** |
| residual `b` | reader arg[8] (`b_dram_addr`, `.cpp:595`) | `TensorAccessor(src1_args, b_addr)` | **Case 1** |
| gamma | reader arg[6] | `TensorAccessor(gamma_args, gamma_addr)` | **Case 1** |
| beta | reader arg[7] | `TensorAccessor(beta_args, beta_addr)` | **Case 1** |
| output | writer arg[0] (`dst_addr`, `.cpp:604`) | `TensorAccessor(dst_args, dst_addr)` | **Case 1** |
| recip LUT | — (CB 25 borrowed) | borrowed-memory DFB | **clean** |

All Case 1 (page-by-page iteration). No Case 2. The four reader `TensorAccessorArgs<N>()`
offset-chain through `next_compile_time_args_offset()` (`.cpp:88-91`) — the
`TensorAccessorArgs` plumbing the port collapses into bindings.

### Work split (multi-core)
- Driver: `split_work_to_cores(requested_cores, num_tile_rows, /*row_wise=*/true)` (`.cpp:197-203`),
  `num_tile_rows = NC * Ht`.
- Output: `(num_cores, all_cores, core_group_1, core_group_2, rows_per_g1, rows_per_g2)`.
- **No multi-`KernelDescriptor` work split** — all three kernels are single `KernelDescriptor`
  over `all_cores`; the per-group row count is delivered purely as the compute RTA
  `num_tile_rows_per_core`, *not* as a per-group CTA. → Preserved-Multiplicity is "none"
  (see below); no [demote-CTA-to-RTA](metal2_port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta)
  concern, because the legacy code already uses one descriptor + RTA.

### Cross-op kernels (multi-core)
**none.** All 10 sources + 2 shared headers are layernorm-owned. Donor headers
(`kernel_lib/reduce_helpers_dataflow.hpp`, `kernel/dataflow/generate_bcast_scalar.hpp`,
`kernel_util/generic/*`) are shared-lib / framework — out of porter scope, Device-2.0-clean,
and cross the boundary via `dfb::name → uint32_t` implicit conversion.

### Flags (multi-core)
- `get_tile_size(cb_id)` holdovers in readers/writers (audit table) — **Device 2.0 track, not
  the port.** During conversion these call sites take `dfb::name` directly (implicit
  conversion), so they don't block, but the audit explicitly forbids the port absorbing the
  `cb_obj.get_tile_size()` member-form cleanup.
- Reader arg[4] `packed_one_value` is dead (audit Misc anomaly) — op owner, not port.

---

## TTNN ProgramFactory

- **Concept (inherited from audit):** `ProgramSpecFactoryConcept`.
- **Custom `compute_program_hash`:** none — already default reflection-based hash.
- **Forced device-op-class / pybind edits** (per ttnn-factory doc):
  1. *(exception 3)* Drop `create_descriptor`'s `core_range_set` parameter; inline the
     production default (`default_core_range(device)`) into the `create_program_spec` body;
     delete the `create_descriptor` pybind hooks (`layernorm_nanobind.cpp:322`, and for the
     sharded factory `:363`). Both are pybind test/introspection hooks returning a
     `ProgramDescriptor` — exactly what the port eliminates.
  2. *(exception 2 is N/A — there is no separate `create_program_descriptor` pybind beyond the
     `create_descriptor` hooks above.)*
  3. No custom-hash deletion (none exists).
- **Implementation notes:** the factory must branch internally (Multi-variant pattern) on
  `(large_tensor_needed, use_welford_and_not_rms_norm, use_row_major_kernel, input_is_row_major,
  rms_norm, fuse_pre_add)` to build the per-branch KernelSpec source + bindings + defines.
  Because Metal 2.0 bindings are **per-KernelSpec** (not a shared table), the legacy
  `cb_named_args` shared table dissolves into per-branch `dfb_bindings`/`tensor_bindings` sets.

## Planned Spec Shape (multi-core, construction blueprint)

- **KernelSpecs:** 3 (reader, writer, compute), each with a runtime-selected `.source` and a
  branch-specific binding set. (One KernelSpec per kernel — there is no work-split
  multiplicity to preserve, §Preserved Multiplicity = none.)
- **DataflowBufferSpecs:** one per active CB index (≤17), built conditionally to mirror the
  legacy conditional CB construction. Welford-fp32 alias indices (c_29/c_30/c_31) are
  *additional* DFBs declared with `advanced_options.alias_with` mutually naming their primary
  (c_0/c_23, c_18, c_19) — [Aliased DFBs pattern](metal2_port_patterns.md#pattern-aliased-dfbs-legacy-aliased-cbs).
  c_25 (recip LUT) is `borrowed_from = RECIP_TENSOR`.
- **SemaphoreSpecs:** none.
- **TensorParameters:** up to 5 (`input`, `residual`, `gamma`, `beta`, `output`), each
  conditionally present, all Case 1. recip LUT is a 6th `TensorParameter` backing the borrowed
  c_25 DFB (`borrowed_from`).
- **WorkUnitSpecs:** 1 (`{reader, writer, compute}` over `all_cores`).

### Construction sketch (illustrative — NOT applied)

```cpp
// Per-branch source selection identical to legacy .cpp:484-549, assigned to KernelSpec::source.
// Tensor bindings (Case 1) replace reader arg[0,6,7,8] / writer arg[0]:
KernelSpec reader{
  .unique_id = READER,
  .source = reader_kernel_path,                       // runtime-selected
  .dfb_bindings = { ProducerOf(CB_IN, "cb_in"), /* +cb_inb/gamma/beta conditionally */ },
  .tensor_bindings = { {INPUT,"input"}, /* +residual/gamma/beta conditionally */ },
  .compile_time_args = {{"block_size",block_size},{"W",W}, /* +use_welford on !large path */},
  .runtime_arg_schema = {.runtime_arg_names = {"num_tile_rows","Wt","start_tile_row","eps", ...}},
  .compiler_options = {.defines = reader_defines},     // FUSE_*, RMSNORM, TILIZE_IN
  .hw_config = DataMovementHardwareConfig{.role = RoleHint::READER},
};
// Welford-fp32 aliased DFBs:
DataflowBufferSpec cb_in{.unique_id=CB_IN, ..., .advanced_options={.alias_with={CB_X_WELFORD}}};
DataflowBufferSpec cb_x_welford{.unique_id=CB_X_WELFORD, ..., .advanced_options={.alias_with={CB_IN}}};
// recip LUT borrowed:
DataflowBufferSpec cb_recip{.unique_id=CB_RECIP, ..., .borrowed_from=RECIP_TENSOR};
// compute hw_config via to_compute_hardware_config(compute_kernel_config), then set
// unpack_to_dest_mode entries (c_26/c_29/c_30/c_31) separately (helper does not cover it).
```

### Kernel-side translation (per source, illustrative)
- `get_named_compile_time_arg_val("cb_*")` → drop; `CircularBuffer cb_*(cb_id)` →
  `DataflowBuffer dfb_*(dfb::cb_*)`; LLK/kernel-lib call sites pass `dfb::cb_*` directly.
- `TensorAccessor(src0_args, src_addr)` (+ the `TensorAccessorArgs<N>()` offset chain) →
  `TensorAccessor(ta::input)`; buffer-address RTAs deleted.
- Conditional CBs (`cb_inb`/`cb_gamma`/`cb_beta`/`cb_fusion`) already `#ifdef`-gated by
  `FUSE_*` — keep the gates; the host emits the matching `KernelSpec` defines and conditional
  bindings ([Conditional bindings pattern](metal2_port_patterns.md#pattern-conditional--optional-dfb-bindings)).
- `cb_x_welford` is **same-FIFO-distinct** from cb_in (independent FIFO pointers, separate
  `push_back`) → modeled as an aliased DFB (distinct DFB, shared L1), NOT a handle alias. The
  reader's `cb_x_welford.reserve_back/push_back` stays; the host declares both as a 2-member
  alias clique. (`welford_fp32_alias == 0` collapses `cb_x_welford` to `cb_in` legacy-side —
  in Metal 2.0 this becomes the no-alias branch with no second DFB.)
- `get_named_compile_time_arg_val("welford_fp32_alias")` etc. (alias-flag scalars) → these are
  *named CTAs that control kernel logic*, not CB indices → become `get_arg(args::...)` named
  CTAs (kernel-side whitelist rule 4), or move to a `#define` if used in a conditional-binding
  context.

## Preserved Multiplicity
**none — no work-split multiplicity in legacy.** All kernels are single `KernelDescriptor`
over `all_cores`; the per-group row count is an RTA, not a per-group CTA.

## Dropped Plumbing (multi-core)

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA arg[0] (`a_addr`) | `a.buffer()->address()` | `TensorBinding(input)` |
| reader RTA arg[8] (`b_dram_addr`) | `b->buffer()->address()` | `TensorBinding(residual)` |
| reader RTA arg[6] (`gamma_dram_addr`) | `gamma->buffer()->address()` | `TensorBinding(gamma)` |
| reader RTA arg[7] (`beta_dram_addr`) | `beta->buffer()->address()` | `TensorBinding(beta)` |
| writer RTA arg[0] (`dst_addr`) | `output.buffer()->address()` | `TensorBinding(output)` |
| reader CTA[3..] 4×`TensorAccessorArgs(...).append_to()` + kernel `TensorAccessorArgs<N>()` chain | `TensorAccessorArgs` plumbing | binding mechanism end-to-end |
| writer CTA `TensorAccessorArgs(output)` + kernel `TensorAccessorArgs<1>()` | same | binding |
| all 3 kernels' `named_compile_time_args` CB indices (`cb_*`) | magic CB indices in (named) CTAs | `DFBBinding` per kernel |
| all positional CTAs (`block_size`, `W`, compute args) | positional `compile_time_args` | named CTAs `{{name,value},...}` |
| reader RTA arg[4] `packed_one_value` | dead RTA | drop (or keep; route to op owner) |

## Applied Patterns (multi-core)
- [Multi-variant factories](metal2_port_patterns.md#pattern-multi-variant-factories): branch on
  `(large/welford/rm)` inside `create_program_spec` to select source + bindings + defines.
- [Conditional / optional DFB bindings](metal2_port_patterns.md#pattern-conditional--optional-dfb-bindings):
  `cb_inb`/`cb_gamma`/`cb_beta`/`cb_fusion` gated by `FUSE_*`/gamma/beta defines.
- [Aliased DFBs](metal2_port_patterns.md#pattern-aliased-dfbs-legacy-aliased-cbs): welford-fp32
  c_29/c_30/c_31 aliasing c_0/c_23, c_18, c_19.
- [Pass DFB handles directly to LLKs/kernel-lib](metal2_port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers):
  `dfb::cb_*` into `calculate_and_prepare_reduce_scaler<>`, `generate_bcast_col_scalar`,
  reduce/compute LLKs, and the `get_tile_size(dfb::name)` holdovers.
- [Removing pybound legacy factory entry points](metal2_port_patterns.md#pattern-removing-pybound-legacy-factory-entry-points)
  + ttnn-factory **exception 3** (drop `core_range_set`, delete `create_descriptor` hooks).

## Deferred / Flagged
- **Yellow audit items:** `get_tile_size(cb_id)` holdovers → Device 2.0 track (do not absorb).
- **New finding (planning):** the shared `named_compile_time_args` table across 3
  `KernelDescriptor`s is the structural source of the atomic coupling — see report.
- **Same-FIFO vs aliased-DFB ambiguity** for `cb_x_welford`: resolved as aliased-DFB
  (distinct FIFO, shared L1) per the reader's independent `push_back`; flagged for verification
  if/when ported, because the legacy fallback collapses it to a handle alias when the flag is 0.

---

## Sharded factory (deferred)

Coarse remaining-work enumeration (not fully inventoried — out of this pass's scope):
- Concept: `ProgramDescriptorFactoryConcept`; lands on `ProgramSpecFactoryConcept`.
- Host: `device/layernorm_op_multi_core_sharded.cpp` (466) +
  `device/sharded_layernorm_factory_helpers.{hpp,cpp}` (463 + 1554) ≈ 2,000 lines.
- Kernels: ~13 sharded/mcast sources (reader/writer mcast sender/receiver, pre/post-allgather,
  welford, rm_gb, reshard) — all in-family.
- Semaphores: 3 × program-scope `SemaphoreDescriptor`, `initial_value=0`
  (`layernorm_op_multi_core_sharded.cpp:219,224,229`) → `SemaphoreSpec` + `SemaphoreBinding`.
- Borrowed-memory DFBs: input/residual/stats/output/recip on `Buffer` memory →
  `borrowed_from`. Aliased CBs: c_0/c_24 → c_29.
- RTA-vararg-shaped read: `writer_unary_sharded_ln.cpp:38` (`get_arg_addr(9)` + counted loop) —
  supported; prefer named RTAs.
- `core_range_set` pybind hook + production use (sharded validates shard-spec cores) — same
  exception-3 unwind.

This factory is larger than the multi-core unit and is its own multi-pass effort.

---

## Realized port (this pass — supersedes the prior grounded stop)

The prior dogfood grounded-stopped here on one-pass *size* budget. This pass executed the port
from the blueprint above as an interactive primary session. Status (see `METAL2_PORT_REPORT.md`
for the full record, friction, and handoffs):

- **Host: complete and C++-build GREEN.** `create_descriptor` → `create_program_spec` returning
  `ProgramArtifacts`; the shared `cb_named_args` table dissolved into per-KernelSpec `dfb_bindings`;
  all DataflowBufferSpecs (conditional + welford-fp32 `alias_with` cliques + borrowed recip);
  TensorParameters/Bindings; named CTAs/RTAs; per-node `KernelRunArgs`; `WorkUnitSpec`. The
  `core_range_set` parameter dropped; the multi-core `create_descriptor` pybind hook removed.
  `device/layernorm_device_operation.hpp` declares `create_program_spec`. Verified by a clean
  `ttnncpp` + `unit_tests_ttnn` link.
- **Kernels: base path + RM writer + shared header converted; welford/large readers+computes and
  the rm_gb reader remain** (mechanical, same validated pattern — see the report's checklist).
- **Confirmed against the actual API (no capability gap — the prior stop was a one-shot-vehicle
  limitation, not a real one):** every mechanism the blueprint anticipated is expressible —
  per-KernelSpec DFB bindings, `alias_with`, `borrowed_from`, conditional bindings (with the
  CTA→`#define` promotion the report notes), `to_compute_hardware_config`/`unpack_to_dest_mode`,
  the `DataflowBuffer` kernel-side swap (incl. the `use<READ_PTR>`→bare-DFB-source resolution).

### Binding-model refinements discovered during construction (not in the original blueprint)
- **`cb_in` producer is path-dependent:** reader on the TILE path, but *compute* (via `tilize`) on
  the ROW_MAJOR path — the reader fills `cb_in_rm` there. Bindings branch on `input_is_row_major`.
- **`cb_x_welford` producer moves by path:** reader (non-fused TILE), compute self-loop (fused, or
  ROW_MAJOR non-fused). Alias group stays bound to one kernel set as required.
- **Conditional gamma/beta DFBs need `#define` gates** (`FUSE_GAMMA`/`FUSE_BETA` emitted to the
  compute kernel, not just the reader): legacy gated them with the `do_gamma`/`do_beta` CTAs via
  `if constexpr`, which still name-looks-up the unbound `dfb::cb_gamma`. New
  `WELFORD_FP32_ALIAS` / `WELFORD_STATE_FP32_ALIAS` defines play the same role for the alias DFBs.

### Remaining to reach a fully GREEN device test pass
1. Convert: `layernorm_welford.cpp`, `layernorm_large_tensor.cpp`, `layernorm_large_tensor_welford.cpp`
   (compute); `reader_unary_interleaved_ln_large_tensor.cpp`, `..._large_tensor_welford.cpp`,
   `..._rm_gb.cpp` (readers). (`layernorm_compute_utils.h` needs no change.)
2. Per-path pytest: base → welford → large-tensor → row-major → fused-pre-add → gamma/beta, in
   `tests/ttnn/unit_tests/operations/fused/test_layer_norm.py`. Resolve the borrowed-read-only-DFB
   producer question (recip LUT) if the validator flags it.
