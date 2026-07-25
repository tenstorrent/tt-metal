# Port Plan — moreh_layer_norm

Port plan for `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm`, ported from the TTNN
`ProgramDescriptor` (`descriptor`) concept to Metal 2.0 (`MetalV2FactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (`ProgramFactory::create_descriptor` returns
  `tt::tt_metal::ProgramDescriptor`, `device/moreh_layer_norm_device_operation.hpp:36-40`,
  `device/moreh_layer_norm_program_factory.cpp:28`).
- Variants: single `program_factory_t = std::variant<ProgramFactory>`. One factory, but it
  branches **internally** between a *small* and a *large* algorithm (chosen by L1 fit at
  `program_factory.cpp:159-173`); each branch selects its own reader + compute kernel source. Both
  branches share the one writer kernel. This is runtime kernel-source selection (one factory, two
  reader sources + two compute sources), so all four selectable sources convert together with the
  factory.
- Custom `compute_program_hash`: none — already default reflection-based hash (audit-confirmed,
  grep clean).

*(Target concept `MetalV2FactoryConcept`, chosen during the audit — carried forward in the
[TTNN ProgramFactory](#ttnn-programfactory) section below.)*

### Kernels
Work split: `split_work_to_cores(grid, num_outer)` →
`(num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2)`.
reader + writer run on `all_cores`; compute is emitted as **two** `KernelDescriptor`s of the same
source over the **disjoint** ranges `core_group_1` / `core_group_2`, differing only in the
`num_rows_per_core_group_N` CTA. Per node exactly one compute instance → ordinary 1:1
(preserve-multiplicity), NOT a two-toucher.

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | reader_moreh_layer_norm_{small,large}.cpp | all_cores | `block_size` + `TensorAccessorArgs`(input,gamma,beta) | input_addr, gamma_addr, beta_addr, num_rows_per_core, num_inner, tile_offset, scaler, eps, mask_h, mask_w | GAMMA_HAS_VALUE, BETA_HAS_VALUE, DO_MASK_H, DO_MASK_W, FP32_DEST_ACC_EN (conditional) | ReaderConfigDescriptor{} |
| writer | writer_moreh_layer_norm.cpp | all_cores | `mean_has_value`, `rstd_has_value`, `block_size` + `TensorAccessorArgs`(output,mean,rstd) | output_addr, mean_addr, rstd_addr, num_rows_per_core, num_inner, tile_offset, mean_rstd_height, mean_rstd_width, normalized_dims | (none) | WriterConfigDescriptor{} |
| compute_1 | moreh_layer_norm_{small,large}_kernel.cpp | core_group_1 | num_rows_per_core_group_1, origin_H, origin_W, num_inner, block_size, gamma_has_value, beta_has_value, mean_has_value, rstd_has_value, is_lastdim_layer_norm, is_groupnorm | (none) | REDUCE_OP, REDUCE_DIM, FP32_DEST_ACC_EN (conditional) | ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, math_approx_mode} |
| compute_2 | (same source) | core_group_2 | (same, num_rows_per_core_group_2) | (none) | (same) | (same) |

### CBs
All pushed at base (no `address_offset`), `all_cores`, plain CBs (no GlobalCircularBuffer). Format:
`cb_data_format = datatype_to_dataformat_converter(input.dtype())` for c_0..c_18; intermediates
c_24..c_31 use `intermed_cb_format = fp32_dest_acc_en ? Float32 : cb_data_format`. `push_cb` skips
zero-size (optional) CBs. Tile geometry is default 32×32 (`tile` not set on any format descriptor).

| index | tiles (small / large) | data_format | present when |
|---|---|---|---|
| c_0 input | num_inner / 2·block_size | cb_data_format | always |
| c_1 scaler | 1 | cb_data_format | always |
| c_2 eps | 1 | cb_data_format | always |
| c_3 gamma | 2·block_size | cb_data_format | gamma_has_value |
| c_4 beta | 2·block_size | cb_data_format | beta_has_value |
| c_5 mask_h | 1 | cb_data_format | do_mask_h |
| c_6 mask_w | 1 | cb_data_format | do_mask_w |
| c_16 output | 2·block_size | cb_data_format | always |
| c_17 mean | 1 | cb_data_format | mean_has_value |
| c_18 rstd | 1 | cb_data_format | rstd_has_value |
| c_24 E[x] (also cb_tmp) | 1 | intermed | always |
| c_25 x-E[x] | num_inner / 2·block_size | intermed | always |
| c_26 (x-E[x])² | 1 / 2·block_size | intermed | always |
| c_27 Sum[(x-E[x])²] | 1 | intermed | always |
| c_28 Var[x] | 1 | intermed | always |
| c_29 1/sqrt(Var+eps) | 1 | intermed | always |
| c_30 gamma·+beta | 2·block_size | intermed | gamma_has_value \|\| beta_has_value |
| c_31 Sum[x] | 2 | intermed | always |

### Semaphores
none — this op uses no semaphores.

### Tensor accessors
All Case 1 (via `TensorAccessor`). Address delivered today as a `Buffer*` BufferBinding in the RTA
list, consumed by `TensorAccessor(args, addr)`. Every accessor is the 2-arg form (no 3rd page-size arg).

| host site | originating Tensor | RTA slot (host) |
|---|---|---|
| reader `TensorAccessor(input_args, input_addr)` (`:39`) | input | reader RTA 0 |
| reader `TensorAccessor(gamma_args, gamma_addr)` (`:43`, GAMMA_HAS_VALUE) | gamma | reader RTA 1 |
| reader `TensorAccessor(beta_args, beta_addr)` (`:48`, BETA_HAS_VALUE) | beta | reader RTA 2 |
| writer `TensorAccessor(output_args, output_addr)` (`:121`) | output | writer RTA 0 |
| writer `TensorAccessor(mean_args, mean_addr)` (`:124`) | mean | writer RTA 1 |
| writer `TensorAccessor(rstd_args, rstd_addr)` (`:127`) | rstd | writer RTA 2 |

Compute kernels touch no tensor memory (CB-only) — no tensor bindings there.

### Work split
- Driver: `tt::tt_metal::split_work_to_cores(grid, num_outer)`
- num_cores / all_cores / core_group_1 / core_group_2 / num_rows_per_core_group_1 / num_rows_per_core_group_2

### Cross-op kernels
None by file path — the op owns all five kernel sources in `device/kernels/`. **However** the two
**compute** sources (`moreh_layer_norm_{small,large}_kernel.cpp`) are borrowed **by file path** by a
legacy peer, `moreh_group_norm` (`moreh_group_norm_program_factory.cpp:251-252`), which is *not*
being ported. Converting them in place would break group_norm's JIT build. Per the orchestration
constraint they are **forked** to `_metal2` copies (see Applied Patterns). The reader/writer sources
are used only by this op and are converted in place. (The three shared *header* pools —
`ttnn/kernel/dataflow/moreh_common.hpp`, `ttnn/kernel/compute/moreh_common.hpp`,
`ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` — are Device 2.0 native, take
`DataflowBuffer` / plain-uint args, and are out of the porter's scope; not edited.)

### Flags
- Dead variable `input_data_format` (reader small/large `:32`) — relocated onto the DFB object per
  whitelist rule 7 (`dfb_input.get_dataformat()`), left unused as it was (no cleanup, per scope
  discipline). Dead `offs` (small reader) and `onetile` constants left untouched.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `MetalV2FactoryConcept`.
- **Custom `compute_program_hash`**: none — already default reflection-based hash. No deletion needed.
- **Implementation notes**: single factory, runtime kernel-source selection (small/large × reader/
  compute). `ProgramFactory::create_descriptor` → `create_program_artifacts`. The `program_factory_t`
  variant is unchanged (still one alternative); the framework auto-routes the single alternative to
  the MetalV2 adapter (no `select_program_factory` exists in this op). No pybind `create_descriptor`
  and no pybind factory-hook parameter (nanobind binds only the op function), so **no** device-op-class
  edits are forced.

## Planned Spec Shape

- **KernelSpecs (4)**: `reader` (1), `writer` (1), `compute_g1` + `compute_g2` (2, same source,
  disjoint grids — preserve multiplicity). small vs large picks the reader + compute source at
  `create_program_artifacts` time.
- **DataflowBufferSpecs (up to 18)**: one per allocated CB (c_0..c_6, c_16..c_18, c_24..c_31).
  Optional ones (gamma c_3, beta c_4, mask_h c_5, mask_w c_6, mean c_17, rstd c_18, gamma_beta c_30)
  are added only when present — matching the legacy `push_cb` skip-zero-size behavior. `entry_size`
  = tile_size(fmt); `num_entries` = the legacy tile count; `data_format_metadata` = the CB's format;
  `tile_format_metadata` left unset (all default 32×32).
- **SemaphoreSpecs**: none.
- **TensorParameters (up to 6)**: input (always), gamma/beta (optional), output (always),
  mean/rstd (optional).
- **WorkUnitSpecs (1 or 2)**: WU_g1 = {reader, writer, compute_g1} @ core_group_1; WU_g2 =
  {reader, writer, compute_g2} @ core_group_2 (only when core_group_2 non-empty). reader/writer land
  on the union (all_cores); each compute on its group.
- **Op-owned tensors**: none.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| compute_desc_1 (core_group_1) + compute_desc_2 (core_group_2), source moreh_layer_norm_{small,large}_kernel_metal2.cpp | compute_g1, compute_g2 | WU_g1, WU_g2 | all compute DFBs (c_0 input CONSUMER, c_1 scaler CONSUMER, c_2 eps CONSUMER, c_3 gamma CONSUMER, c_4 beta CONSUMER, c_5 mask_h CONSUMER, c_6 mask_w CONSUMER, c_16 out PRODUCER, c_17 mean PRODUCER, c_18 rstd PRODUCER, c_24..c_31 self-loop P+C) — each compute KernelSpec binds the same roles over its disjoint grid; the shared input/scaler/etc DFBs get one PRODUCER (reader) + two CONSUMER KernelSpecs (compute_g1/g2) over non-overlapping nodes → legal per the DFB per-node invariant, no multi-binding flag |

reader binds c_0 PRODUCER, c_1/c_2 PRODUCER (fill), c_3/c_4 PRODUCER (opt), c_5/c_6 PRODUCER (opt).
writer binds c_16 CONSUMER, c_17/c_18 CONSUMER (opt).

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA 0 (input_addr) | `input.buffer()` BufferBinding + `TensorAccessor(input_args, input_addr)` | `TensorParameter input` + `TensorBinding` + `TensorAccessor(tensor::input)` |
| reader RTA 1 (gamma_addr) | gamma `Buffer*` | `TensorParameter gamma` (opt) + `TensorAccessor(tensor::gamma)` |
| reader RTA 2 (beta_addr) | beta `Buffer*` | `TensorParameter beta` (opt) + `TensorAccessor(tensor::beta)` |
| reader CTA `TensorAccessorArgs`(input,gamma,beta) + kernel `TensorAccessorArgs<N>()` chain | layout plumbing | binding mechanism (dropped end-to-end) |
| writer RTA 0/1/2 (output/mean/rstd addr) | `Buffer*` | `TensorParameter` output/mean/rstd + `TensorAccessor(tensor::*)` |
| writer CTA `TensorAccessorArgs`(output,mean,rstd) | layout plumbing | binding mechanism |
| reader/writer/compute CB-index constants (`tt::CBIndex::c_N`) | magic CB index | `dfb::<name>` binding handle |
| writer CTA mean_has_value / rstd_has_value | positional CTA bool | `#define MEAN_HAS_VALUE` / `RSTD_HAS_VALUE` (compiler_options.defines) |
| compute CTA gamma/beta/mean/rstd_has_value | positional CTA bool | `#define GAMMA_HAS_VALUE / BETA_HAS_VALUE / MEAN_HAS_VALUE / RSTD_HAS_VALUE` (+ `GAMMA_OR_BETA` for c_30) |
| compute do_mask_h/do_mask_w gating of mask CBs | computed constexpr bool + `if` | `#define DO_MASK_H / DO_MASK_W` (host already emits to reader; now also to compute) + `#ifdef` |
| all positional CTAs (reader block_size; compute num_rows/origin_H/W/num_inner/block_size/is_lastdim/is_groupnorm; writer block_size) | positional | named CTAs (`args::name`) |
| all positional RTAs | positional | named RTAs (`args::name`) |

No semaphore-ID RTAs (op has no semaphores). No page-size 3rd-arg CTAs. No raw pointers (all Case 1).

## Applied Patterns

- **Shared-kernel fork** ([Caution: Modifying a shared dataflow kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)):
  the two compute sources are borrowed by legacy `moreh_group_norm`, so they are forked to
  `moreh_layer_norm_{small,large}_kernel_metal2.cpp` and the factory repoints at the forks. The
  legacy originals stay untouched for group_norm.
- **Preserve-multiplicity work split** — two compute KernelSpecs over disjoint `core_group_1/2`,
  per-group `num_rows_per_core` CTA preserved; NOT demoted to RTA.
- **Multiple KernelSpecs on one DFB endpoint (disjoint nodes)** — compute_g1 + compute_g2 both bind
  the reader-fed DFBs as CONSUMER over non-overlapping grids (legal, no multi-binding flag).
- **Self-loop DFB binding** — c_24..c_31 (intermediates) bound PRODUCER+CONSUMER on each compute
  KernelSpec.
- **Conditional / optional DFB bindings** — gamma/beta/mask_h/mask_w/mean/rstd/gamma_beta DFBs and
  the gamma/beta/mean/rstd/output tensor bindings are declared conditionally on the host, matched by
  `#define`s, and `#ifdef`-gated in the kernels (aliases, DFB/accessor construction, and every
  referencing expression incl. the `cb_gamma_beta_or_out` / `cb_outg` file-scope ternaries).
- **Multi-source runtime selection** — small/large algorithm picks reader + compute source inside
  `create_program_artifacts`.

## Deferred / Flagged

- New findings during planning: none that block. The `reduce_helpers_compute.hpp` helper takes CB
  ids as `uint32_t` **non-type template parameters** (`reduce<..., cb_xsum, cb_scaler, cb_ex>`);
  `dfb::name` (a `constexpr DFBAccessor` with a `constexpr operator uint32_t`,
  `dataflow_buffer.h:55`) is a valid converted-constant-expression NTTP, so it flows in unchanged.
  Noted in the report as a success/confirmation.
