# Port Plan — `layernorm` (`LayerNormMultiCoreProgramFactory`)

Port plan for `ttnn/cpp/ttnn/operations/normalization/layernorm`, ported from the
`ProgramDescriptor` API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

**Scope of this pass: `LayerNormMultiCoreProgramFactory` only** (the interleaved / non-sharded
path), together with the ten kernel entry points it can select.
`LayerNormShardedProgramFactory` stays on `ProgramDescriptorFactoryConcept` and is untouched;
the two factories share no kernel source, so the op builds and runs with one factory on each
concept. What the sharded pass still has to do is recorded in
`METAL2_PORT_REPORT.md`.

---

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` — `create_descriptor` returns a
  `tt::tt_metal::ProgramDescriptor`
  ([device/layernorm_device_operation.hpp:20-25](device/layernorm_device_operation.hpp#L20-L25)).
- Variants: single `DeviceOperation` (`LayerNormDeviceOperation`) with **two** program factories in
  its `program_factory_t` variant, selected on `input.is_sharded()`
  ([device/layernorm_device_operation.cpp:18-24](device/layernorm_device_operation.cpp#L18-L24)).
  This plan covers `LayerNormMultiCoreProgramFactory`.
- Factory methods live in a **`program_factory_t` variant member struct**, not on the
  device-operation itself — so [ttnn_factory exception 3](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/ttnn_factory.md)
  (direct-descriptor conversion) does **not** apply. The port is a method swap inside the
  existing struct.
- Custom `compute_program_hash`: **none** — the device-operation defines no override and no
  backdoor `attribute_values` / `to_hash`. (The `compute_program_hash` nanobind at
  [layernorm_nanobind.cpp:252-263](layernorm_nanobind.cpp#L252-L263) forwards to the framework
  default; it is a pybind of internals, not a custom hash.)
- **Non-standard factory parameter**: `create_descriptor` takes a fourth
  `const std::optional<CoreRangeSet>& core_range_set = std::nullopt`
  ([device/layernorm_device_operation.hpp:24](device/layernorm_device_operation.hpp#L24)),
  reachable only through the pybind hook. This is the case
  [ttnn_factory exception 2](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/ttnn_factory.md)
  names explicitly. See [TTNN ProgramFactory](#ttnn-programfactory) below.

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's
TTNN factory analysis section. Carried forward below.)*

### Runtime kernel-source selection

The factory selects its kernel sources at run time on three independent axes, so the port's true
unit is this factory **plus all ten entry points**. Host selection sites:
[device/layernorm_op_multi_core.cpp:483-496](device/layernorm_op_multi_core.cpp#L483-L496) (reader),
[:635-639](device/layernorm_op_multi_core.cpp#L635-L639) (writer),
[:540-548](device/layernorm_op_multi_core.cpp#L540-L548) (compute).

Shorthand used throughout this plan:

| tag | source | selected when |
|---|---|---|
| `R_STD` | `dataflow/reader_unary_interleaved_ln.cpp` | `!large_tensor && !use_row_major_kernel` |
| `R_LT` | `dataflow/reader_unary_interleaved_ln_large_tensor.cpp` | `large_tensor && !welford` |
| `R_LTW` | `dataflow/reader_unary_interleaved_ln_large_tensor_welford.cpp` | `large_tensor && welford` |
| `R_RMGB` | `dataflow/reader_unary_interleaved_ln_rm_gb.cpp` | `!large_tensor && use_row_major_kernel` |
| `W_STD` | `dataflow/writer_unary_interleaved_start_id_blocked.cpp` | `!input_is_row_major` |
| `W_RM` | `dataflow/writer_unary_interleaved_start_id_blocked_rm_output.cpp` | `input_is_row_major` |
| `C_STD` | `compute/layernorm.cpp` | `!large_tensor && !welford` |
| `C_LT` | `compute/layernorm_large_tensor.cpp` | `large_tensor && !welford` |
| `C_W` | `compute/layernorm_welford.cpp` | `!large_tensor && welford` |
| `C_LTW` | `compute/layernorm_large_tensor_welford.cpp` | `large_tensor && welford` |

Here `welford` is the factory's `use_welford_and_not_rms_norm`. `validate_on_program_cache_miss`
rejects `use_welford && RMSNORM` ([device/layernorm_device_operation.cpp:262-270](device/layernorm_device_operation.cpp#L262-L270)),
so `use_welford` implies `!rms_norm` and the two spellings coincide.

`large_tensor_needed` can only become true under `!use_row_major_kernel || input_is_row_major`
([:288](device/layernorm_op_multi_core.cpp#L288)), so the compute selector's extra conjunct at
[:541](device/layernorm_op_multi_core.cpp#L541) is redundant and reader/compute always agree on the
large-tensor axis.

**Producer roles move between kernels across paths.** Under `TILIZE_IN` (`input_is_row_major`) the
input DFB `c_0` is filled by *compute* (`tilize_block`) rather than by the reader, and `c_16` is
drained by *compute* (`pack_untilize_block`) rather than by the writer. Only `C_STD` / `C_LT`
carry those blocks. The endpoint table below is therefore keyed per config, not per CB.

### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `R_STD` / `R_LT` / `R_LTW` / `R_RMGB` (see table above) | `all_cores` | `R_STD`/`R_RMGB`: `block_size`, `use_welford`, `W`, `TensorAccessorArgs`×4, trailing size arg. `R_LT`/`R_LTW`: `block_size`, `W`, `TensorAccessorArgs`×4, trailing size arg ([:352-373](device/layernorm_op_multi_core.cpp#L352-L373)) | `cb_named_args` (24 entries, [:447-480](device/layernorm_op_multi_core.cpp#L447-L480)) | per core: `a.buffer()`, `NCHt`, `Wt`, `reader_start`, `packed_one_value`, `eps`, `gamma_buffer`, `beta_buffer`, `b_buffer`, [`H_logical`] ([:587-601](device/layernorm_op_multi_core.cpp#L587-L601)) | none | `FUSE_PRE_ADD`, `FUSE_GAMMA`, `FUSE_BETA`, `RMSNORM`, `TILIZE_IN` ([:387-411](device/layernorm_op_multi_core.cpp#L387-L411)) | **O2** (field absent; DM default) | `ReaderConfigDescriptor{}` ([:631](device/layernorm_op_multi_core.cpp#L631)) |
| writer | `W_STD` / `W_RM` | `all_cores` | `block_size`, `TensorAccessorArgs`×1, [`elem_size_bytes`] ([:376-381](device/layernorm_op_multi_core.cpp#L376-L381)) | `cb_named_args` (same 24) | per core: `output.buffer()`, `Wt`, `num_tile_rows`, `writer_start`, [`H_logical`] ([:605-613](device/layernorm_op_multi_core.cpp#L605-L613)) | none | none | **O2** (field absent; DM default) | `WriterConfigDescriptor{}` ([:644](device/layernorm_op_multi_core.cpp#L644)) |
| compute | `C_STD` / `C_LT` / `C_W` / `C_LTW` | `all_cores` | non-welford: `Wt`, `block_size`, `do_gamma`, `do_beta`, `fp32_dest_acc_en`, `float32_reduction`, `legacy_rsqrt`, `W`, `tile_width`. welford: `Wt`, `block_size`, `do_gamma`, `do_beta`, `fp32_dest_acc_en`, `W`, `TILE_SIZE`, `rms_norm`, `fuse_pre_add` ([:500-511](device/layernorm_op_multi_core.cpp#L500-L511)) | `cb_named_args` (same 24) | per core: `num_tile_rows_per_core` ([:614](device/layernorm_op_multi_core.cpp#L614)) | none | `FUSE_PRE_ADD` (only when `!use_welford`), `RMSNORM`, `TILIZE_IN`, `UNTILIZE_OUT`, activation defines ([:387-420](device/layernorm_op_multi_core.cpp#L387-L420)) | **O3** (field absent; `ComputeConfigDescriptor` default) | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, unpack_to_dest_mode, math_approx_mode}` ([:654-659](device/layernorm_op_multi_core.cpp#L654-L659)) |

`grep -n opt_level device/layernorm_op_multi_core.cpp` prints **nothing** — the field is absent on
all three descriptors, so it resolves to `O2` on the two DM kernels and **`O3` on the compute
kernel**. The compute `KernelSpec` therefore needs an explicit
`compiler_options.opt_level = KernelBuildOptLevel::O3`.

*(`opt_level` is the legacy kernel's **resolved** level, not what the source literally says.)*

### CBs

All 21 `CBDescriptor`s are built through the `make_cb_descriptor` lambda over `all_cores`
([:666-679](device/layernorm_op_multi_core.cpp#L666-L679)); allocation sites are
[:681-833](device/layernorm_op_multi_core.cpp#L681-L833). None sets `.tile`, so no
`tile_format_metadata` carries over. No `GlobalCircularBuffer` anywhere.

| index | total_size | core_ranges | data_format | page_size | allocated when |
|---|---|---|---|---|---|
| `c_0` | `in0_t * in_single_tile_size` | `all_cores` | `in_data_format` | `in_single_tile_size` | always |
| `c_29` (2nd format descriptor on `c_0`) | — shares `c_0`'s allocation | `all_cores` | `in_data_format` | `in_single_tile_size` | `welford_fp32_alias && !fuse_pre_add` |
| `c_16` | `out0_t * out_single_tile_size` | `all_cores` | `out_data_format` | `out_single_tile_size` | always |
| `c_18` | `im1_t * single_tile_size` | `all_cores` | `cb_data_format` | `single_tile_size` | `!rms_norm` |
| `c_30` (2nd format descriptor on `c_18`) | — shares `c_18`'s allocation | `all_cores` | `cb_data_format` | `single_tile_size` | `welford_state_fp32_alias` |
| `c_2` | `in2_t * scaler_tile_size` | `all_cores` | `Float16_b` | `scaler_tile_size` | `!use_welford` |
| `c_3` | `in3_t * bfloat16_tile_size` | `all_cores` | `Float16_b` | `bfloat16_tile_size` | always |
| `c_19` | `im2_t * single_tile_size` | `all_cores` | `cb_data_format` | `single_tile_size` | always |
| `c_31` (2nd format descriptor on `c_19`) | — shares `c_19`'s allocation | `all_cores` | `cb_data_format` | `single_tile_size` | `welford_state_fp32_alias` |
| `c_24` | `im0_t * single_tile_size` | `all_cores` | `cb_data_format` | `single_tile_size` | `!rms_norm \|\| fuse_pre_add \|\| large_tensor` |
| `c_20` | `im3_t * single_tile_size` | `all_cores` | `cb_data_format` | `single_tile_size` | `!use_welford` |
| `c_21` | `im4_t * single_tile_size` | `all_cores` | `cb_data_format` | `single_tile_size` | always |
| `c_26` | `large_tensor_acc_tile_size` | `all_cores` | `float32_reduction ? Float32 : cb_data_format` | same as total | `large_tensor && !use_welford` |
| `c_27` | `in_rm_size` | `all_cores` | `in_data_format` | `in_single_tile_size` | `input_is_row_major` |
| `c_28` | `out_rm_size` | `all_cores` | `out_data_format` | `out_single_tile_size` | `input_is_row_major` |
| `c_22` | `im5_t * single_tile_size` | `all_cores` | `cb_data_format` | `single_tile_size` | `gamma \|\| beta` |
| `c_5` | `in5_t * gamma_single_tile_size` | `all_cores` | `gamma_cb_data_format` | `gamma_single_tile_size` | `gamma` |
| `c_6` | `in6_t * beta_single_tile_size` | `all_cores` | `beta_cb_data_format` | `beta_single_tile_size` | `beta` |
| `c_23` | `im6_t * single_tile_size` | `all_cores` | `cb_data_format` | `single_tile_size` | `fuse_pre_add && !rms_norm` |
| `c_29` (2nd format descriptor on `c_23`) | — shares `c_23`'s allocation | `all_cores` | `cb_data_format` | `single_tile_size` | `welford_fp32_alias && fuse_pre_add` |
| `c_1` | `in1_t * inb_single_tile_size` | `all_cores` | `inb_data_format` | `inb_single_tile_size` | `fuse_pre_add` |
| `c_25` | `reciprocal_CB_size_bytes` | `all_cores` | `Float32` | `reciprocal_CB_size_bytes` | `use_welford`; **`.buffer = recip_tensor->buffer()`** (borrowed memory) |

`c_25` is the factory's only borrowed-memory CB ([:824-832](device/layernorm_op_multi_core.cpp#L824-L832));
`address_offset` is never set anywhere in this op, so it is a plain base-only borrow.

### Semaphores

none — the interleaved factory builds no `SemaphoreDescriptor`.

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| [:357](device/layernorm_op_multi_core.cpp#L357) `TensorAccessorArgs(a.buffer())` | `tensor_args.input` | reader slot 0 (`a.buffer()`, [:588](device/layernorm_op_multi_core.cpp#L588)) |
| [:358](device/layernorm_op_multi_core.cpp#L358) `TensorAccessorArgs(b->buffer())` | `tensor_args.residual_input_tensor` | reader slot 8 (`b_buffer`, [:596](device/layernorm_op_multi_core.cpp#L596)) |
| [:359](device/layernorm_op_multi_core.cpp#L359) `TensorAccessorArgs(gamma->buffer())` | `tensor_args.weight` | reader slot 6 (`gamma_buffer`, [:594](device/layernorm_op_multi_core.cpp#L594)) |
| [:360](device/layernorm_op_multi_core.cpp#L360) `TensorAccessorArgs(beta->buffer())` | `tensor_args.bias` | reader slot 7 (`beta_buffer`, [:595](device/layernorm_op_multi_core.cpp#L595)) |
| [:377](device/layernorm_op_multi_core.cpp#L377) `TensorAccessorArgs(output.buffer())` | `tensor_return_value` | writer slot 0 (`output.buffer()`, [:606](device/layernorm_op_multi_core.cpp#L606)) |

All five are **Case 1** (the kernel builds a `TensorAccessor` and uses its page-access methods); no
kernel does raw base-address arithmetic, so `get_bank_base_address` is never needed. No accessor in
the factory passes a third (page-size) constructor argument.

`recip_tensor` is a sixth tensor, but it reaches the device **only** as `c_25`'s borrowed backing
memory — never through an accessor and never as an address RTA.

### Work split

- Driver: `split_work_to_cores(requested_cores, num_tile_rows, /*row_wise=*/true)`
  ([:196-202](device/layernorm_op_multi_core.cpp#L196-L202)), where
  `num_tile_rows = NC * Ht` and `requested_cores` is `core_range_set` or `default_core_range(device)`.
- num_cores: `num_cores`
- core_group_1: `core_group_1`, count_per_core: `num_tile_rows_per_core_group_1`
- core_group_2: `core_group_2`, count_per_core: `num_tile_rows_per_core_group_2`

The per-group counts are delivered as **per-core RTAs**, not as per-group CTAs: the factory emits a
single `KernelDescriptor` per role over `all_cores` and varies `num_tile_rows_per_core` in the RTA
list ([:567-617](device/layernorm_op_multi_core.cpp#L567-L617)). There is therefore no
multi-`KernelDescriptor` work split to preserve.

### Shared kernels

**none.** All ten sources this factory binds live in the op's own directory, and
`grep -rln` over `ttnn/cpp/ttnn/operations/`, `tests/` and `models/` finds no consumer outside
`ttnn/cpp/ttnn/operations/normalization/layernorm/`. The op's *other* factory (sharded) binds a
disjoint set of sources, so there is no intra-op sharing either, and no `_metal2` fork exists
beside any of them.

Two in-op kernel **headers** are shared between the two factories —
[device/kernels/dataflow/layernorm_dataflow_utils.h](device/kernels/dataflow/layernorm_dataflow_utils.h)
and [device/kernels/layernorm_scaler_tiles.h](device/kernels/layernorm_scaler_tiles.h) (also
included by `reader_mcast_sender_unary_sharded_ln.cpp` / `reader_mcast_receiver_unary_sharded_ln.cpp`).
Both are already parameterized on `DataflowBuffer&` / `TensorAccessor` templates with no CB ids and
no argument reads, so **neither needs a functional change** and the sharded factory is unaffected.
The `cb` → `dfb` name sweep does rename four helpers in `layernorm_dataflow_utils.h`
(`read_block_to_cb` → `read_block_to_dfb` and the three row-major siblings) plus two in
`layernorm_compute_utils.h`; the sharded kernels call none of them, using only
`compute_single_stage_noc_addrs` / `compute_two_stage_noc_addrs` from the shared header.
[device/kernels/compute/layernorm_compute_utils.h](device/kernels/compute/layernorm_compute_utils.h)
is interleaved-only and likewise needs no change.

Function-call escapes out of the op directory (all cleared by the audit):

- ⭐ [ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp](../../../../kernel/dataflow/generate_bcast_scalar.hpp)
  takes the legacy `CircularBuffer`. A `_metal2` fork already exists beside it
  ([generate_bcast_scalar_metal2.hpp](../../../../kernel/dataflow/generate_bcast_scalar_metal2.hpp),
  `generate_bcast_col_scalar(DataflowBuffer&, uint32_t)` at line 13). **Rung 1 — reuse the existing
  fork**; no new file, no pointer comment (the original already has consumers and is not ours to
  annotate). Its parameter is a non-const reference, so each call site must pass a named local.
  Call sites in this factory's kernels: `R_STD` [:152](device/kernels/dataflow/reader_unary_interleaved_ln.cpp#L152),
  `R_LT` [:140](device/kernels/dataflow/reader_unary_interleaved_ln_large_tensor.cpp#L140),
  `R_LTW` [:73](device/kernels/dataflow/reader_unary_interleaved_ln_large_tensor_welford.cpp#L73),
  `R_RMGB` [:112](device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb.cpp#L112).
- ✓ `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` and `.../reduce_helpers_compute.hpp` take
  the DFB identity as a `uint32_t` non-type template parameter — `dfb::name`'s constexpr conversion
  covers template-argument position. No donor change.
- ✓ `ttnn/cpp/ttnn/operations/normalization/kernel_util/**` (in-family): `DataflowBuffer&` or
  `uint32_t cb_id` parameters throughout. No change. This includes
  `compute/memory.h`'s `get_pointer_to_cb_data(uint32_t, ...)`, which the Welford compute kernels
  reach for the reciprocal LUT — `dfb::reciprocals` converts and flows straight through
  (audit Question 1, answered: the `uint32_t cb_id` donor shape stands).

### Flags

- **No unreferenced kernel files.** All ten sources this factory can select are reachable.
- **Dead legacy plumbing found during the inventory** (each is read by *no* kernel this factory
  binds; all four are recorded again in `METAL2_PORT_REPORT.md`):
  - Reader RTA slot 4, `packed_one_value` ([:592](device/layernorm_op_multi_core.cpp#L592), computed
    at [:552-553](device/layernorm_op_multi_core.cpp#L552-L553)). The reader kernels' own header
    comments label it *"legacy; unused, scaler is generated in-kernel"*.
  - The reader's trailing size CTA in three of its four branches
    ([:365-373](device/layernorm_op_multi_core.cpp#L365-L373)): only the `input_is_row_major` branch
    (`a.element_size()`) is ever read, and only by `R_STD` / `R_LT` under `TILIZE_IN`. The
    `gamma_stick_size` / `beta_stick_size` / `tile_size` branches feed nothing.
  - Compute CTA slot 7 on the Welford path, `rms_norm`
    ([:504](device/layernorm_op_multi_core.cpp#L504)): neither `C_W` nor `C_LTW` reads index 7, and
    the value is a compile-time `false` on that path anyway.
  - `cb_fusion` (`c_22`) is declared but never used by `C_LTW`
    ([compute/layernorm_large_tensor_welford.cpp:357](device/kernels/compute/layernorm_large_tensor_welford.cpp#L357)),
    which routes gamma/beta staging through `cb_xmm` instead. `c_22` is still *allocated* in that
    config, so it is a config-local dead CB rather than a dead allocation.
- **Two legacy-broken configurations, preserved as-is.** Neither is guarded by a `TT_FATAL`, and
  both hang or produce garbage today. They are called out here because they are the reason several
  DFB endpoints below are declared unconditionally rather than following the kernel's actual use:
  1. `use_welford && input_is_row_major` — the factory emits `TILIZE_IN` / `UNTILIZE_OUT` and
     allocates `c_27` / `c_28`, but neither Welford compute kernel contains a tilize or untilize
     block, so nothing fills `c_0` and nothing fills `c_28`. (This is the audit's misc anomaly 1,
     which covers the large-tensor half; the non-large-tensor half behaves the same way.)
  2. `input_is_row_major && use_row_major_kernel && !large_tensor` — `R_RMGB` is selected, which has
     no `TILIZE_IN` branch, so it reads a row-major tensor as tiles and never fills `c_27`, while
     `C_STD` waits on `c_27`.
  The op's own tests skip both combinations
  ([tests/ttnn/nightly/unit_tests/operations/fused/test_layernorm.py:384-386](../../../../../../tests/ttnn/nightly/unit_tests/operations/fused/test_layernorm.py#L384-L386)).

---

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`. The ported-from factory has no
  `override_runtime_arguments`, so the framework owns the cache-hit binding refresh and the factory
  writes one method, `create_program_artifacts`.
- **Custom `compute_program_hash`**: none — default reflection-based hash. Nothing to preserve.
- **Implementation notes**:
  - The legacy fourth parameter `core_range_set` is **dropped** and its production default,
    `default_core_range(device)`, is **inlined** at the work-split site
    ([:193](device/layernorm_op_multi_core.cpp#L193)), per ttnn_factory exception 2. To be precise
    about why, since this plan's first revision claimed the signature could not carry it: nothing
    technically prevented keeping it. `ProgramSpecFactoryConcept` tests only for the presence of
    `create_program_artifacts` and the adapter calls it with exactly three arguments, so a defaulted
    fourth parameter would have compiled unnoticed. It is dropped because its only caller was the
    nanobind being deleted below. `default_core_range` itself survives as a static member (the
    factory body calls it) and its nanobind stays.
  - The multi-core `create_descriptor` nanobind
    ([layernorm_nanobind.cpp:320-346](layernorm_nanobind.cpp#L320-L346)) is **deleted** — it would
    otherwise reference a vanished symbol. This is a user-visible API surface change with real
    downstream consumers; both the removal and the consumers are recorded in
    `METAL2_PORT_REPORT.md` under Handoff points. The sharded factory's `create_descriptor` nanobind
    stays, because that factory is not ported in this pass.
  - The factory's `create_descriptor` declaration is replaced by `create_program_artifacts` in
    [device/layernorm_device_operation.hpp](device/layernorm_device_operation.hpp); the sharded
    factory's declaration is untouched. A `program_factory_t` variant is valid with its factories on
    different concepts.

## Planned Spec Shape

Default is 1:1 with legacy. Naming convention: the legacy `cb_*` names lose the prefix, since the
`dfb::` namespace already carries the role (`cb_in` → `dfb::in`, `cb_x_welford` → `dfb::x_welford`).

- **KernelSpecs** — three, one per legacy `KernelDescriptor`: `READER`, `WRITER`, `COMPUTE`. Each
  points at whichever of its runtime-selected sources the factory picked. No multiplicity to
  preserve (see Work split above).
- **DataflowBufferSpecs** — one per legacy `CBDescriptor`, **plus one extra per additional
  `buffer_index`** on the four aliased descriptors: `IN`, `INB`, `SCALER`, `EPS`, `GAMMA`, `BETA`,
  `OUT`, `EX`, `EX2`, `XMM2`, `EX2PE`, `FUSION`, `X`, `XMM`, `RECIPROCALS`, `ACCUMULATE`, `IN_RM`,
  `OUT_RM`, `X_WELFORD`, `EX_WELFORD`, `EX2_WELFORD` — each declared under the same condition as its
  legacy CB. `RECIPROCALS` carries `borrowed_from = RECIP`.
- **SemaphoreSpecs** — none.
- **TensorParameters** — six: `INPUT`, `RESIDUAL`, `GAMMA_T`, `BETA_T`, `OUTPUT`, `RECIP`.
  `RECIP` is declared for the borrowed-memory DFB only and is bound by no kernel, which the
  validator permits for a `borrowed_from` target. The other five are declared under the same
  condition as their legacy `TensorAccessorArgs` (residual / gamma / beta are optional).
- **WorkUnitSpecs** — one, `{READER, WRITER, COMPUTE}` over `all_cores`.
- **Op-owned tensors** — none. Every tensor the factory touches arrives through `tensor_args` /
  `tensor_return_value`, `recip_tensor` included.

### Planned DFB endpoint bindings

Re-derived by counting the kernels that touch each buffer, rather than copied from the brief. `P` = PRODUCER,
`C` = CONSUMER, `self` = the same kernel bound both ways.

Two host-side predicates drive the rows that move:

```
compute_tilizes    = input_is_row_major && !use_welford   // C_STD / C_LT carry TILIZE_IN / UNTILIZE_OUT
reader_fills_in    = !compute_tilizes                     // otherwise compute fills IN via tilize_block
```

| DFB | declared when | endpoints | resolution |
|---|---|---|---|
| `IN` (`c_0`) | always | `compute_tilizes` ? compute **self** : reader P + compute C | 1P+1C, or self-loop on the row-major path |
| `INB` (`c_1`) | `fuse_pre_add` | reader P + compute C | 1P+1C |
| `SCALER` (`c_2`) | `!use_welford` | reader P + compute C | 1P+1C |
| `EPS` (`c_3`) | always | reader P + compute C | 1P+1C (compute waits, mostly never pops — intentional reuse) |
| `GAMMA` (`c_5`) | `gamma` | reader P + compute C | 1P+1C |
| `BETA` (`c_6`) | `beta` | reader P + compute C | 1P+1C |
| `OUT` (`c_16`) | always | `input_is_row_major` ? compute **self** : compute P + writer C | self-loop on the row-major path (compute re-reads it for `pack_untilize_block`), else 1P+1C |
| `EX` (`c_18`) | `!rms_norm` | compute **self** | self-loop |
| `EX2` (`c_19`) | always | compute **self** | self-loop |
| `XMM2` (`c_20`) | `!use_welford` | compute **self** | self-loop |
| `EX2PE` (`c_21`) | always | compute **self** | self-loop |
| `FUSION` (`c_22`) | `gamma \|\| beta` | compute **self** | self-loop (unused by `C_LTW`; see Flags) |
| `X` (`c_23`) | `fuse_pre_add && !rms_norm` | compute **self** | self-loop |
| `XMM` (`c_24`) | `!rms_norm \|\| fuse_pre_add \|\| large_tensor` | compute **self** | self-loop |
| `RECIPROCALS` (`c_25`) | `use_welford` | compute **self** | self-loop; sync-free (raw pointer, no FIFO ops) → role-free, labels cosmetic |
| `ACCUMULATE` (`c_26`) | `large_tensor && !use_welford` | compute **self** | self-loop |
| `IN_RM` (`c_27`) | `input_is_row_major` | reader P + compute C | 1P+1C; **both declared unconditionally** under the gate |
| `OUT_RM` (`c_28`) | `input_is_row_major` | compute P + writer C | 1P+1C; **both declared unconditionally** under the gate |
| `X_WELFORD` (`c_29`) | `welford_fp32_alias` | `fuse_pre_add` ? compute **self** : reader P + compute C | alias of `X` (fused) or `IN` (non-fused) |
| `EX_WELFORD` (`c_30`) | `welford_state_fp32_alias` | compute **self** | alias of `EX` |
| `EX2_WELFORD` (`c_31`) | `welford_state_fp32_alias` | compute **self** | alias of `EX2` |

**No multi-binding flag is set anywhere in this factory**, and no DFB is both self-looped and
multi-bound. Every buffer has either a single touching kernel (self-loop) or two (1P+1C).

`IN_RM` and `OUT_RM` are the one place the declaration is wider than the kernel's actual use: in the
two legacy-broken row-major configs (Flags above) the selected kernel never references the token.
Declaring the conditional-side endpoint unconditionally is the sanctioned resolution for asymmetric
conditional use; it keeps the spec valid, keeps the L1 footprint byte-identical, and leaves the
broken configs behaving exactly as they do today. The alternative — dropping the DFB — would change
the L1 layout of a config that is merely broken, not absent.

### Alias groups

Four `advanced_options.alias_with` cliques, each a legacy `CBDescriptor` carrying two
`CBFormatDescriptor`s. Every group is conditional; when its gate is false the second index does not
exist and the kernel-side name falls back to the primary DFB's handle (same-FIFO aliasing, not
`alias_with`).

| group | gate | legacy site |
|---|---|---|
| `IN` ↔ `X_WELFORD` | `welford_fp32_alias && !fuse_pre_add` | [:687-692](device/layernorm_op_multi_core.cpp#L687-L692) |
| `X` ↔ `X_WELFORD` | `welford_fp32_alias && fuse_pre_add` | [:809-814](device/layernorm_op_multi_core.cpp#L809-L814) |
| `EX` ↔ `EX_WELFORD` | `welford_state_fp32_alias` | [:708-713](device/layernorm_op_multi_core.cpp#L708-L713) |
| `EX2` ↔ `EX2_WELFORD` | `welford_state_fp32_alias` | [:733-738](device/layernorm_op_multi_core.cpp#L733-L738) |

All three legality rules hold by construction: each pair mutually names the other (a two-member
clique), both members take their size from the one legacy `CBDescriptor::total_size`, and every DFB
in the factory targets the same node set (`all_cores`, one work unit).

### `unpack_modes`

Legacy fills a `vector<UnpackToDestMode>` of size `NUM_CIRCULAR_BUFFERS`, indexed by CB id, with
`Default` everywhere except three sites ([:516-535](device/layernorm_op_multi_core.cpp#L516-L535)):
`c_26` under `float32_reduction`, `c_30` / `c_31` under `welford_state_fp32_alias`, and `c_29` under
`welford_fp32_alias`. Those become `UnpackMode::UnpackToDest` keyed by DFB name; everything else is
`UnpackMode::UnpackToSrc`.

The Metal 2.0 validator additionally **requires** an explicit entry for every `Float32` DFB the
compute kernel *consumes* when `enable_32_bit_dest` (= `fp32_dest_acc_en`) is true. With
`fp32_dest_acc_en` on, `cb_data_format` is `Float32`, so that sweeps in most of the intermediates.
The port therefore walks the compute `KernelSpec`'s own CONSUMER bindings and emits
`UnpackToSrc` for each `Float32` DFB that is not one of the four `UnpackToDest` sites — deriving
each value from the legacy vector rather than guessing.

Each `UnpackToDest` entry is gated on the **same condition as its binding**, which is stricter than
the legacy vector write. `ACCUMULATE` is the case where the two differ: legacy set its slot from
`float32_reduction` alone, so under `use_welford && fp32_dest_acc_en && !legacy_reduction` it wrote
an entry for a buffer index it had not allocated. The validator rejects that outright
(*"unpack_modes entry references DFB 'accumulate', which the kernel does not bind"*), so the port
gates the entry on `large_tensor_needed && !use_welford && float32_reduction`.

*Watch item:* under `welford_fp32_alias && fuse_pre_add && !fp32_dest_acc_en`, `X_WELFORD` aliases
`X`, whose format is `cb_data_format` = `Float16_b`, yet legacy sets `UnpackToDestFp32` on it. The
Gen1 validator rejects `UnpackToDest` on a ≤16-bit format. If that fires it is a legacy
misconfiguration surfacing, not a port defect — stop and report rather than silently changing the
mode.

## Preserved Multiplicity

none — no work-split multiplicity in legacy. The factory emits one `KernelDescriptor` per role over
`all_cores` and varies the per-group tile-row count through per-core runtime args, so there is no
per-group CTA to preserve and no second `KernelSpec` of any source.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| [:588](device/layernorm_op_multi_core.cpp#L588) reader RTA slot 0 | `a.buffer()` pushed as `Buffer*` into `RTArgList` | `TensorParameter INPUT` + `TensorBinding{INPUT, "src"}` on `READER` |
| [:596](device/layernorm_op_multi_core.cpp#L596) reader RTA slot 8 | `b_buffer` | `TensorParameter RESIDUAL` + `TensorBinding{RESIDUAL, "src_b"}` on `READER` |
| [:594](device/layernorm_op_multi_core.cpp#L594) reader RTA slot 6 | `gamma_buffer` | `TensorParameter GAMMA_T` + `TensorBinding{GAMMA_T, "gamma"}` on `READER` |
| [:595](device/layernorm_op_multi_core.cpp#L595) reader RTA slot 7 | `beta_buffer` | `TensorParameter BETA_T` + `TensorBinding{BETA_T, "beta"}` on `READER` |
| [:606](device/layernorm_op_multi_core.cpp#L606) writer RTA slot 0 | `output.buffer()` | `TensorParameter OUTPUT` + `TensorBinding{OUTPUT, "dst"}` on `WRITER` |
| [:357-360](device/layernorm_op_multi_core.cpp#L357-L360) reader CTAs | `TensorAccessorArgs(...).append_to(reader_compile_time_args)` ×4, consumed kernel-side by the `TensorAccessorArgs<3>()` → `next_compile_time_args_offset()` chain | binding mechanism end-to-end; the kernel collapses to `TensorAccessor(tensor::src)` etc. |
| [:377](device/layernorm_op_multi_core.cpp#L377) writer CTAs | `TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args)`, consumed by `TensorAccessorArgs<1>()` | `TensorAccessor(tensor::dst)` |
| [:447-480](device/layernorm_op_multi_core.cpp#L447-L480) all three kernels' `named_compile_time_args` | 21 CB-index entries (`cb_in`, `cb_inb`, … `cb_ex2_welford`) carried as named CTAs | one `DFBBinding` per DFB per kernel; the kernel reads `dfb::in` etc. |
| [:472](device/layernorm_op_multi_core.cpp#L472), [:479](device/layernorm_op_multi_core.cpp#L479) | named CTAs `welford_fp32_alias`, `welford_state_fp32_alias` — booleans that gate a conditionally-present CB index | `KernelSpec::compiler_options.defines` `WELFORD_FP32_ALIAS` / `WELFORD_STATE_FP32_ALIAS`, gating the kernel-side handle alias and its uses |
| [:352-373](device/layernorm_op_multi_core.cpp#L352-L373) reader positional CTAs | `{block_size, [use_welford], W, …, trailing size}` | named: `block_size`, `W`, `elem_size_bytes` (row-major input only). `use_welford` is **not** carried across: it gates `SCALER`, a conditionally-present buffer, so it becomes the `USE_WELFORD` define instead |
| [:376-381](device/layernorm_op_multi_core.cpp#L376-L381) writer positional CTAs | `{block_size, …, [elem_size_bytes]}` | named: `block_size`, `elem_size_bytes` (row-major output only) |
| [:587-613](device/layernorm_op_multi_core.cpp#L587-L613) reader / writer RTA slot 3 | one positional slot whose meaning depends on the selected source (`start_tile_row` in the merged readers, `tile_offset` in the two legacy tile readers; likewise for the writer) | named after the host's own variable, `reader_start` / `writer_start` — the only name accurate for every source the factory can select |
| [:500-511](device/layernorm_op_multi_core.cpp#L500-L511) compute positional CTAs | 9 positional values | named: `Wt`, `block_size`, `do_gamma`, `do_beta`, `fp32_dest_acc_en`, then `float32_reduction` / `legacy_rsqrt` / `W` / `tile_width` (non-welford) or `W` / `tile_width` / `fuse_pre_add` (welford) |
| [:592](device/layernorm_op_multi_core.cpp#L592) reader RTA slot 4 | `packed_one_value` (with its `bfloat16(1)` / `pack_two_bfloat16_into_uint32` computation at [:552-553](device/layernorm_op_multi_core.cpp#L552-L553)) | **dropped** — read by no reader kernel |
| [:365-373](device/layernorm_op_multi_core.cpp#L365-L373) reader trailing CTA | `gamma_stick_size` / `beta_stick_size` / `tile_size` branches | **dropped** — read by no reader kernel; only the `input_is_row_major` branch survives, as the named `elem_size_bytes` |
| [:504](device/layernorm_op_multi_core.cpp#L504) compute CTA slot 7 | `rms_norm` on the Welford path | **dropped** — read by neither Welford compute kernel |

No semaphore-ID RTA and no page-size third-argument CTA exists in this factory.

## Applied Patterns

- [Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding)
  — the ten compute-private intermediates (`EX`, `EX2`, `XMM2`, `EX2PE`, `FUSION`, `X`, `XMM`,
  `ACCUMULATE`, `EX_WELFORD`, `EX2_WELFORD`), plus `OUT` on the row-major path and `IN` when compute
  tilizes.
- [Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb)
  — `RECIPROCALS`, read only through a raw base pointer with no FIFO ops, by the one compute kernel.
- [Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings)
  — `INB` / `X` (`FUSE_PRE_ADD`), `GAMMA` / `BETA` / `FUSION` (`FUSE_GAMMA` / `FUSE_BETA`),
  `EX` (`RMSNORM`), `IN_RM` / `OUT_RM` (`TILIZE_IN` / `UNTILIZE_OUT`),
  `X_WELFORD` (`WELFORD_FP32_ALIAS`), `EX_WELFORD` / `EX2_WELFORD` (`WELFORD_STATE_FP32_ALIAS`),
  `SCALER` / `XMM2` (`USE_WELFORD`), `ACCUMULATE` (`LARGE_TENSOR`).
  Three of those gates are **CTA-to-define promotions**: `do_gamma` / `do_beta` reach conditional DFB
  names from `if constexpr` branches in every compute kernel, and `fuse_pre_add` does the same in the
  two Welford compute kernels (which today receive no `FUSE_PRE_ADD` define at all —
  [:389-391](device/layernorm_op_multi_core.cpp#L389-L391) emits it only when `!use_welford`).
  The defines are emitted to **every** kernel that references the gated name, not just the reader.
- [Aliased DFBs](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-aliased-dfbs-legacy-aliased-cbs)
  — the four `alias_with` cliques above.
- [Same-FIFO aliasing](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-same-fifo-aliasing-one-dfb-multiple-kernel-side-names)
  — the *path-dependent* variant, five kernel-side names that resolve to another DFB's handle rather
  than to a DFB of their own: `cb_xmm` → `cb_in` under `RMSNORM && !FUSE_PRE_ADD` (`C_STD`);
  `cb_x` → `cb_in` / `cb_xmm` depending on `FUSE_PRE_ADD` / `RMSNORM` (`C_STD`, `C_LT`, `C_W`);
  `cb_x_welford` → `cb_x` when `WELFORD_FP32_ALIAS` is off; `cb_ex_welford` / `cb_ex2_welford` →
  `cb_ex` / `cb_ex2` when `WELFORD_STATE_FP32_ALIAS` is off. Each becomes an `#ifdef`-gated
  `constexpr` handle alias with **one** `DataflowBuffer` object per FIFO — the objects the legacy
  kernels construct a second time on the same FIFO are gated out along with the alias.
- [Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers)
  — `reduce_init`, `pack_tile`, `add_tiles`, `transpose_tile`, `compute_kernel_hw_startup`,
  `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<dfb::scaler, …>` (template-argument
  position), `kutil::compute::memory::get_pointer_to_cb_data(dfb::reciprocals, 0)`.
- [Removing pybound legacy factory entry points](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-removing-pybound-legacy-factory-entry-points)
  — the multi-core `create_descriptor` nanobind, together with the ttnn_factory exception 2 unwind
  of its `core_range_set` parameter.

## Deferred / Flagged

- **New findings during planning:**
  - The two legacy-broken row-major configurations under Flags above. They are the reason `IN_RM` /
    `OUT_RM` declare both endpoints unconditionally. Neither is guarded by a `TT_FATAL`; the port
    does not add one.
  - `cb_fusion` (`c_22`) is allocated but untouched in the `large_tensor && use_welford &&
    (gamma || beta)` configuration. The port keeps the allocation and self-loops the DFB on the
    compute kernel so the L1 footprint is unchanged; dropping the allocation is an op-owner decision.
  - The `unpack_modes` watch item above (`UnpackToDest` on a `Float16_b` DFB when
    `welford_fp32_alias && fuse_pre_add && !fp32_dest_acc_en`).
  - `Wt` and `eps` are per-core runtime args whose value is identical on every node, so they are
    really common runtime args. The port keeps them as RTAs — converting changes dispatch semantics
    and is a separate cleanup.
