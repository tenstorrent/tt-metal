# Port Plan — `layernorm`

Port plan for `ttnn/cpp/ttnn/operations/normalization/layernorm`, ported from the
`ProgramDescriptor` API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

The op has two program factories and they were ported in two passes, one factory each. The two
passes are written up separately below:

- [Part 1 — `LayerNormMultiCoreProgramFactory`](#part-1--layernormmulticoreprogramfactory) (the
  interleaved / non-sharded path), with the ten kernel entry points it can select.
- [Part 2 — `LayerNormShardedProgramFactory`](#part-2--layernormshardedprogramfactory) (the sharded
  path, all three distributed-norm stages), with the thirteen kernel entry points it can select.

---

# Part 1 — `LayerNormMultiCoreProgramFactory`

**Scope of this pass: `LayerNormMultiCoreProgramFactory` only** (the interleaved / non-sharded
path), together with the ten kernel entry points it can select.
At the time of this pass `LayerNormShardedProgramFactory` stayed on
`ProgramDescriptorFactoryConcept` and was untouched; the two factories share no kernel source, so
the op built and ran with one factory on each concept.

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

- ⭐ [ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp](../../../kernel/dataflow/generate_bcast_scalar.hpp)
  takes the legacy `CircularBuffer`. A `_metal2` fork already exists beside it
  ([generate_bcast_scalar_metal2.hpp](../../../kernel/dataflow/generate_bcast_scalar_metal2.hpp),
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

---

# Part 2 — `LayerNormShardedProgramFactory`

**Scope of this pass: `LayerNormShardedProgramFactory` only**, together with the thirteen kernel
entry points it can select across its three `distributed_norm_stage` values. With this pass both
factories are on `ProgramSpecFactoryConcept` and the op has no legacy `create_descriptor` left.

Shorthand used throughout Part 2:

| tag | meaning |
|---|---|
| **ND** | `distributed_norm_stage == NOT_DISTRIBUTED` |
| **PRE** | `distributed_norm_stage == PRE_ALL_GATHER` |
| **POST** | `distributed_norm_stage == POST_ALL_GATHER` |
| `welford` | the program config's `use_welford` (only ever true under ND, see below) |
| `writes_back` | `POST && !skip_write_back` — the one config that reshards the output |

---

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` — `create_descriptor` returns a
  `tt::tt_metal::ProgramDescriptor`
  ([device/layernorm_device_operation.hpp:29-36](device/layernorm_device_operation.hpp#L29-L36) as of
  the Part 1 pass).
- Variants: one `DeviceOperation` with two factories; this part covers the second,
  `LayerNormShardedProgramFactory`, selected when `tensor_args.input.is_sharded()`.
- Factory methods live in a `program_factory_t` variant member struct, so ttnn_factory exception 3
  (direct-descriptor conversion) does not apply.
- Custom `compute_program_hash`: **none** (same as Part 1).
- **Non-standard factory parameter**: `create_descriptor` takes a fourth
  `const std::optional<CoreRangeSet>& core_range_set = std::nullopt`. Unlike the multi-core factory,
  the sharded factory uses it only to *validate* that the shard grid's whole multicast bounding box
  lies inside it ([device/layernorm_op_multi_core_sharded.cpp:41-70](device/layernorm_op_multi_core_sharded.cpp#L41-L70));
  it never chooses cores. **The invoker's decision for this pass: keep the parameter**, as a
  defaulted fourth parameter on `create_program_artifacts`, so the validation survives. See
  [TTNN ProgramFactory](#ttnn-programfactory-1) below.
- The factory body is thin; the bulk lives in
  [device/sharded_layernorm_factory_helpers.cpp](device/sharded_layernorm_factory_helpers.cpp) /
  [.hpp](device/sharded_layernorm_factory_helpers.hpp), which this pass rewrites in place (the
  helpers are private to this factory — nothing else includes them).

### Runtime kernel-source selection

`KernelPaths::get` ([device/sharded_layernorm_factory_helpers.cpp:461-492](device/sharded_layernorm_factory_helpers.cpp#L461-L492))
selects on three axes: stage (ND / PRE / POST), `use_row_major_kernel` (row-major gamma or beta,
writer only), and `use_welford` (compute only, ND only). Thirteen entry points:

| tag | source | selected when |
|---|---|---|
| `RS_ND` | `dataflow/reader_mcast_sender_unary_sharded_ln.cpp` | ND |
| `RR_ND` | `dataflow/reader_mcast_receiver_unary_sharded_ln.cpp` | ND |
| `RS_PRE` | `dataflow/reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp` | PRE |
| `RR_PRE` | `dataflow/reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp` | PRE |
| `RS_POST` | `dataflow/reader_mcast_sender_unary_sharded_ln_post_allgather.cpp` | POST |
| `RR_POST` | `dataflow/reader_mcast_receiver_unary_sharded_ln_post_allgather.cpp` | POST |
| `W_STD` | `dataflow/writer_unary_sharded_ln.cpp` | ND or POST, `!use_row_major_kernel` |
| `W_RMGB` | `dataflow/writer_unary_sharded_ln_rm_gb.cpp` | ND or POST, `use_row_major_kernel` |
| `W_PRE` | `dataflow/writer_unary_sharded_ln_pre_all_gather.cpp` | PRE |
| `C_ND` | `compute/layernorm_sharded.cpp` | ND, `!use_welford` |
| `C_NDW` | `compute/layernorm_sharded_welford.cpp` | ND, `use_welford` |
| `C_PRE` | `compute/layernorm_sharded_pre_allgather.cpp` | PRE |
| `C_POST` | `compute/layernorm_sharded_post_allgather.cpp` | POST |

Three in-directory kernel **headers** are also part of the unit:
[dataflow/reshard_writer.hpp](device/kernels/dataflow/reshard_writer.hpp) (bound by `W_STD` and
`W_RMGB` only — the same-named header in
`ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/` is that op's own
separate copy, not this one),
[dataflow/col_mask_dataflow.h](device/kernels/dataflow/col_mask_dataflow.h) (bound by all three
writers) and [dataflow/layernorm_dataflow_utils.h](device/kernels/dataflow/layernorm_dataflow_utils.h)
(shared with the Part 1 factory; the sharded readers use only `compute_single_stage_noc_addrs` /
`compute_two_stage_noc_addrs` from it, and this pass leaves it unchanged).

**Config combinations excluded upstream**, relied on below:

- `use_welford && rms_norm` is rejected by `validate_on_program_cache_miss`
  ([device/layernorm_device_operation.cpp:262-270](device/layernorm_device_operation.cpp#L262-L270)),
  so `use_welford` implies `!rms_norm`.
- `use_welford` needs a reciprocal LUT tensor, which the factory `TT_FATAL`s on
  ([device/layernorm_op_multi_core_sharded.cpp:192](device/layernorm_op_multi_core_sharded.cpp#L192)).
  Neither `layer_norm_pre_all_gather` nor `layer_norm_post_all_gather` supplies one on the sharded
  path, so **PRE and POST imply `!use_welford`** in practice, and `use_welford` only ever reaches ND.
- A non-rectangular shard grid is rejected for PRE and POST
  ([device/layernorm_device_operation.cpp:194-196](device/layernorm_device_operation.cpp#L194-L196)),
  and `inactive_cores` is non-empty only for a non-rectangular grid, so **the idle-core kernel
  triple only ever appears under ND**. That matters because two of the three PRE/POST reader sources
  carry no `IDLE_CORE` guard.
- PRE and POST both require `a.padded_shape()[-2] == tile_height`, so `block_ht == 1` there, which
  makes `num_cores_all_to_all == 1` unless the two-stage reduce is on. That is why
  `all_to_all_cores == sender_cores` in every non-two-stage PRE / POST program.

### Kernels

Seven `KernelDescriptor`s (plus a three-descriptor idle triple), built by `add_kernel_descriptors`
([device/sharded_layernorm_factory_helpers.cpp:897-1113](device/sharded_layernorm_factory_helpers.cpp#L897-L1113)).
Compile-time args come from `CompileTimeArgs::build`
([:613-807](device/sharded_layernorm_factory_helpers.cpp#L613-L807)), runtime args from
`RuntimeArgsResult::build` ([:1704-1747](device/sharded_layernorm_factory_helpers.cpp#L1704-L1747)).

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader_sender | `RS_ND` / `RS_PRE` / `RS_POST` | `sender_cores` | 20 values ([:623-643](device/sharded_layernorm_factory_helpers.cpp#L623-L643)) | `reader_cb_named_args` (8 CB indices, [:904-913](device/sharded_layernorm_factory_helpers.cpp#L904-L913)) | 4 mcast coords, 2 grid-relative coords, then `num_x` + `num_y` NOC coords | none | `FUSE_PRE_ADD`, `FUSE_GAMMA`, `FUSE_BETA` | **O2** (absent; DM default) | `DataMovementConfigDescriptor{RISCV_0, reader_noc, DM_DEDICATED_NOC}` |
| reader_receiver_all_to_all | `RR_ND` / `RR_PRE` / `RR_POST` | `all_to_all_workers_except_sender` | 17 values, `is_all_to_all_worker = 1` ([:646-663](device/sharded_layernorm_factory_helpers.cpp#L646-L663)) | same 8 | 3 flags/offsets, 2 grid-relative coords, then `num_x` + `num_y` NOC coords | none | same as sender | **O2** | same as sender |
| reader_receiver | same source | `not_all_to_all_workers` | 17 values, `is_all_to_all_worker = 0`, `num_x = num_y = 1` ([:666-683](device/sharded_layernorm_factory_helpers.cpp#L666-L683)) | same 8 | same shape, 1 + 1 NOC coords | none | same as sender | **O2** | same as sender |
| writer_sender | `W_STD` / `W_RMGB` / `W_PRE` | `all_to_all_cores` | 5 values + 2 `TensorAccessorArgs` blocks + `[stick_size]` + 2 f32 flags + 3 write-back values ([:686-727](device/sharded_layernorm_factory_helpers.cpp#L686-L727)) | `writer_cb_named_args` (7 CB indices + `logical_K`, `block_w`, `use_welford`) + `is_all_to_all_worker = 1` | 3 packed scalars, 2 addresses, `width_shard_tile_start_id`, then the write-back block (2 scalars + 3 per segment) | none | `RMSNORM`, `SKIP_WRITE_BACK`, `DO_COL_MASK` | **O2** | `DataMovementConfigDescriptor{RISCV_1, writer_noc, DM_DEDICATED_NOC}` |
| writer_receiver | same source | `not_all_to_all_workers` | same, `is_all_to_all_worker = 0`, plus a trailing dead `use_welford` ([:728](device/sharded_layernorm_factory_helpers.cpp#L728)) | same + `is_all_to_all_worker = 0` | same | none | same | **O2** | same |
| compute_all_to_all | `C_ND` / `C_NDW` / `C_PRE` / `C_POST` | `all_to_all_cores` | 14 values (+6 more under `use_welford`), `is_all_to_all_worker = 1` ([:732-746](device/sharded_layernorm_factory_helpers.cpp#L732-L746)) | `compute_cb_named_args` (19 CB indices + `welford_fp32_alias`) | 1 value, +3 (+1 under POST) when all-to-all, then 3 Welford values | none | `FUSE_PRE_ADD`, `RMSNORM`, `DO_COL_MASK`, activation defines | **O3** (absent; `ComputeConfigDescriptor` default) | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, unpack_to_dest_mode, math_approx_mode}` |
| compute_not_all_to_all | same source | `not_all_to_all_workers` | same, `is_all_to_all_worker = 0` | same 20 | 1 value, then 3 Welford values | none | same | **O3** | same |
| idle reader / writer / compute | `RR_*` / `W_*` / `C_*` | `inactive_cores` | the receiver / not-all-to-all lists | same | none | none | receiver defines + `IDLE_CORE` | O2 / O2 / **O3** | receiver configs |

`grep -n opt_level` over
[device/layernorm_op_multi_core_sharded.cpp](device/layernorm_op_multi_core_sharded.cpp) and
[device/sharded_layernorm_factory_helpers.cpp](device/sharded_layernorm_factory_helpers.cpp) prints
**nothing** — the field is absent on all ten descriptors, so it resolves to `O2` on the DM kernels
and **`O3` on all three compute descriptors** (`compute_all_to_all`, `compute_not_all_to_all`, and
the idle one). Each compute `KernelSpec` therefore needs an explicit
`compiler_options.opt_level = KernelBuildOptLevel::O3`.

**Both DM configs are custom, not the reader / writer defaults.** Legacy pairs the *reader* with
`RISCV_0` and the *writer* with `RISCV_1` — the opposite processor assignment from Metal 2.0's
reader (`RISCV_1`/`NOC_0`) and writer (`RISCV_0`/`NOC_1`) defaults. So neither
`create_reader_datamovement_config` nor `create_writer_datamovement_config` reproduces this op;
both kernels get a hand-built `DataMovementGen1Config` with the legacy triple copied verbatim,
including the `preferred_noc_for_dram_read` / `_write` NOC choice and its POST-with-write-back
override to `NOC_0` / `NOC_1`
([device/layernorm_op_multi_core_sharded.cpp:265-270](device/layernorm_op_multi_core_sharded.cpp#L265-L270)).

### CBs

All CB descriptors are built by `add_cb_descriptors`
([device/sharded_layernorm_factory_helpers.cpp:1115-1417](device/sharded_layernorm_factory_helpers.cpp#L1115-L1417)).
None sets `.tile`, so no `tile_format_metadata` carries over. No `GlobalCircularBuffer` anywhere.

| index | legacy name(s) | total_size | core_ranges | data_format | page_size | allocated when |
|---|---|---|---|---|---|---|
| `c_0` | `cb_in0` | `in0_CB_size` | `all_cores` | `in_data_format` | `in_single_tile_size` | always; **`.buffer = a`** |
| `c_29` | `cb_x_welford` (2nd format descriptor on `c_0`) | shares `c_0` | `all_cores` | `in_data_format` | `in_single_tile_size` | `welford_fp32_alias && !has_b` |
| `c_1` | `cb_in1` | `in1_CB_size` | `all_cores` | `in_data_format` | `in_single_tile_size` | `has_b`; **`.buffer = b`** |
| `c_14` | `cb_in_pre_add` | `in1_CB_size` | `all_cores` | `in_data_format` | `in_single_tile_size` | `has_b && PRE`; **`.buffer = a`** |
| `c_5` | `cb_gamma` | `in5_CB_size` | `all_cores` | `gamma_cb_data_format` | `gamma_single_tile_size` | `has_gamma` |
| `c_6` | `cb_beta` | `in6_CB_size` | `all_cores` | `beta_cb_data_format` | `beta_single_tile_size` | `has_beta` |
| `c_24` | `cb_x` / `cb_x2` / `cb_ex_sqr` | `x_CB_size` | `all_cores` | `cb_data_format` | `single_tile_size` | always |
| `c_29` | `cb_x_welford` (2nd format descriptor on `c_24`) | shares `c_24` | `all_cores` | `cb_data_format` | `single_tile_size` | `welford_fp32_alias && has_b` |
| `c_18` | `cb_xmm` / `cb_fusion` | `xmm_CB_size` | `all_cores` | `cb_data_format` | `single_tile_size` | always |
| `c_8` | `cb_ex_partial` | `ex_partial_CB_size` | `all_cores` | `cb_data_format` | `single_tile_size` | `!rms_norm` |
| `c_9` | `cb_ex` | `ex_CB_size` | `all_cores` | `cb_data_format` | `single_tile_size` | `!rms_norm` |
| `c_10` | `cb_ex_external` | `ex_external_CB_size` | `all_cores` | `cb_data_format` | `single_tile_size` | `!rms_norm` |
| `c_2` | `cb_scaler` / `cb_in_2` | `in2_CB_size` | `all_cores` | `Float16_b` | `bfloat16_tile_size` | `!use_welford` |
| `c_14` | `cb_mask_scratch` | `xmm_CB_size` | `all_cores` | `cb_data_format` | `single_tile_size` | `!use_welford && do_legacy_layernorm_col_mask` |
| `c_19` | `cb_col_mask` | `col_mask_gen_CB_size_bytes` | `all_cores` | `Float16_b` | `bfloat16_tile_size` | `!use_welford && do_col_mask` |
| `c_3` | `cb_eps` | `in3_CB_size` | `all_cores` | `Float16_b` | `bfloat16_tile_size` | `!use_welford` |
| `c_4` | `cb_scaler_global` | `scaler_global_tile_size` | `all_cores` | `Float32` or `Float16_b` | same as total | `!use_welford` |
| `c_11` | `cb_ex_partial2` | `ex_partial_CB_size` | `all_cores` | `cb_data_format` | `single_tile_size` | `!use_welford` |
| `c_12` | `cb_ex2` | `ex_CB_size` | `all_cores` | `cb_data_format` | `single_tile_size` | `!use_welford` |
| `c_13` | `cb_ex_external2` | `ex_external_CB_size` | `all_cores` | `cb_data_format` | `single_tile_size` | `!use_welford` |
| `c_20` | `cb_ex2pe` / `cb_reciprocal` | `ex2pe_CB_size` | `all_cores` | `cb_data_format` | `single_tile_size` | `!use_welford` |
| `c_15` | `cb_ex_global` | `ex_global_CB_size` | `all_cores` | `cb_data_format` | `single_tile_size` | always |
| `c_22` | `cb_transpose` | `ex_global_CB_size` | `all_cores` | `cb_data_format` | `single_tile_size` | `use_welford` |
| `c_25` | `cb_reciprocals` | `reciprocal_CB_size_bytes` | `all_cores` | `Float32` | `reciprocal_CB_size_bytes` | `use_welford`; **`.buffer = recip`** |
| `c_7` | `cb_stats` | `stats_cb_size` | **`sender_cores`** | `stats_cb_data_format` | `stats_single_tile_size` | POST; **`.buffer = stats`** |
| `c_21` | `cb_stats_reduced` | `stats_reduced_cb_size` | **`sender_cores`** | `cb_data_format` | `single_tile_size` | POST |
| `c_19` | `cb_var` | `ex_global_CB_size` | **`sender_cores`** | `cb_data_format` | `single_tile_size` | POST |
| `c_16` | `cb_out` | `out_CB_size` | **`sender_cores`** under PRE, else `all_cores` | `out_data_format` | `out_single_tile_size` | always; **`.buffer = output`** unless `writes_back` |
| `c_17` | `cb_out_resharded` | `out_reshard_CB_size` | `all_worker_and_storage_cores` | `out_data_format` | `out_single_tile_size` | `writes_back`; **`.buffer = output`** |

A post-pass then widens every **non**-buffer-backed CB whose range is exactly `all_cores` to
`mcast_dest_cores` when `inactive_cores` is non-empty
([:1410-1416](device/sharded_layernorm_factory_helpers.cpp#L1410-L1416)).

`address_offset` is never set, so every buffer-backed CB is a plain base-only borrow.

### Semaphores

| id | core_type | core_ranges | initial_value |
|---|---|---|---|
| 0 (`reduce_sender`) | `WORKER` | `mcast_dest_cores` | 0 |
| 1 (`reduce_receiver`) | `WORKER` | `mcast_dest_cores` | 0 |
| 2 (`reduce_second_stage`) | `WORKER` | `mcast_dest_cores` | 0 |

([device/layernorm_op_multi_core_sharded.cpp:232-246](device/layernorm_op_multi_core_sharded.cpp#L232-L246).)
Note the kernel-side naming is crossed relative to the ids: the readers' CTA slot 0 carries
`reduce_receiver_semaphore_id` (= 1) and slot 1 carries `reduce_sender_semaphore_id` (= 0).

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| [device/sharded_layernorm_factory_helpers.cpp:692, 702](device/sharded_layernorm_factory_helpers.cpp#L692-L702) `TensorAccessorArgs(gamma_buffer)` | `tensor_args.weight` | writer slot 3 (`gamma_dram_addr`, [:1697](device/sharded_layernorm_factory_helpers.cpp#L1697)), with an explicit `buffer_binding` on the same index ([:816-834](device/sharded_layernorm_factory_helpers.cpp#L816-L834)) |
| [:693, 703](device/sharded_layernorm_factory_helpers.cpp#L693-L703) `TensorAccessorArgs(beta_buffer)` | `tensor_args.bias` | writer slot 4 (`beta_dram_addr`, [:1698](device/sharded_layernorm_factory_helpers.cpp#L1698)), likewise bound |

Both are **Case 1**. No accessor in the factory passes a third (page-size) constructor argument.

The remaining five tensors reach the device without an accessor:

- `input`, `residual_input_tensor`, `stats`, `recip_tensor` — **clean**: each *is* a borrowed-memory
  CB (`c_0` / `c_14`, `c_1`, `c_7`, `c_25`).
- `output` — **clean** in every config except `writes_back`, where it is **Case 2**: `c_17` carries
  no data, the writer reads only its base address
  ([device/kernels/dataflow/reshard_writer.hpp:39](device/kernels/dataflow/reshard_writer.hpp#L39))
  and issues its own remote NOC writes to the storage cores.

### Work split

n/a — there is no `split_work_to_cores` call. The sharded factory derives its core ranges from the
input's shard spec (`GridParams` / `WorkerDistribution` / `CoreRanges`,
[device/sharded_layernorm_factory_helpers.cpp:112-414](device/sharded_layernorm_factory_helpers.cpp#L112-L414)),
and the several descriptors per role cover **disjoint** node sets. Per node there is exactly one
reader, one writer and one compute kernel — the disjoint-node-sets case, not a dual-instance work
split.

### Shared kernels

**none.** All thirteen sources this factory binds live in the op's own directory and no other op
instantiates any of them; the Part 1 factory binds a disjoint set of sources. Two in-directory
headers are shared with the Part 1 factory —
[dataflow/layernorm_dataflow_utils.h](device/kernels/dataflow/layernorm_dataflow_utils.h) and
[layernorm_scaler_tiles.h](device/kernels/layernorm_scaler_tiles.h) — and this pass changes neither.
[dataflow/reshard_writer.hpp](device/kernels/dataflow/reshard_writer.hpp) *is* modified, but its only
two consumers are this factory's two write-back writers (the identically-named header under
`experimental/ccl/rms_allgather/.../dataflow/` is that op's own separate file, resolved from its own
directory, and carries a different function).

Function-call escapes out of the op directory (all cleared by the audit):

- ⭐ [ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp](../../../kernel/dataflow/generate_bcast_scalar.hpp)
  takes the legacy `CircularBuffer`. **Rung 1 — reuse the existing `_metal2` fork**
  ([generate_bcast_scalar_metal2.hpp](../../../kernel/dataflow/generate_bcast_scalar_metal2.hpp)),
  exactly as Part 1 did. Call sites in this factory's kernels:
  [writer_unary_sharded_ln.cpp:71](device/kernels/dataflow/writer_unary_sharded_ln.cpp#L71) and
  [writer_unary_sharded_ln_rm_gb.cpp:74](device/kernels/dataflow/writer_unary_sharded_ln_rm_gb.cpp#L74).
  With those two converted, **this op no longer binds the legacy header at all**.
- ✓ `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` and `.../reduce_helpers_compute.hpp` take
  the DFB identity as a `uint32_t` non-type template parameter — `dfb::name` converts in
  template-argument position. No donor change.
- ✓ [ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp](../../../kernel/dataflow/moreh_common.hpp)'s
  `generate_mask_w<T>(DataflowBuffer, uint32_t)`, reached from
  [col_mask_dataflow.h](device/kernels/dataflow/col_mask_dataflow.h). Already on `DataflowBuffer`.
- ✓ `ttnn/cpp/ttnn/operations/normalization/kernel_util/**` (in-family): `DataflowBuffer&` or
  `uint32_t cb_id` parameters throughout, including `compute/memory.h`'s
  `get_pointer_to_cb_data(uint32_t, ...)` that the Welford compute kernel uses for the reciprocal LUT
  (audit Question 1, answered: the `uint32_t cb_id` donor shape stands).

### Flags

- **No unreferenced kernel files.** All thirteen sources are reachable.
- **Config-dead CBs.** The brief states "No dead CBs anywhere in this op." Re-deriving the census
  per `(CB, stage)` contradicts that for PRE and POST: several CBs the legacy code allocates
  unconditionally are touched by no kernel in those stages. Full list, each verified by grepping
  every kernel the stage can select:
  - **PRE**: `c_3` (`cb_eps`), `c_8`, `c_9`, `c_10`, `c_15`, `c_18`, `c_20`. `c_9` even has a
    declared-but-unused kernel-side alias
    ([compute/layernorm_sharded_pre_allgather.cpp:64](device/kernels/compute/layernorm_sharded_pre_allgather.cpp#L64)),
    as does `c_14`
    ([dataflow/reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp:49](device/kernels/dataflow/reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp#L49),
    the audit's misc anomaly 5).
  - **POST**: `c_8`, `c_9`, `c_10`, `c_11`, `c_13`, `c_20`, and `c_1` when a residual is supplied.
    `c_9`, `c_20` and `c_1` again have declared-but-unused aliases
    ([compute/layernorm_sharded_post_allgather.cpp:62, 68, 73](device/kernels/compute/layernorm_sharded_post_allgather.cpp#L62-L73)).
  - **PRE**, with gamma or beta supplied through the `prim` entry point rather than
    `layer_norm_pre_all_gather`: `c_5` / `c_6`, which no PRE kernel reads.
  A bindingless DFB is rejected by the validator, so each of these is dropped per the recipe's
  dead-CB rule. Recorded in `METAL2_PORT_REPORT.md`.
- **`c_2` is single-ended under POST.** The writer fills it (`prepare_reduce_scaler`) and the POST
  compute kernel never reads it. Not dead — it gets a self-loop on the writer.
- **Dead legacy plumbing** (read by no kernel this factory binds):
  - The writer's `gamma_stick_size` / `beta_stick_size` CTA
    ([:706-712](device/sharded_layernorm_factory_helpers.cpp#L706-L712)). Both writer sources skip
    that slot: `W_STD` reads `beta_args.next_compile_time_args_offset() + 2 … + 4` and `W_RMGB` reads
    `+ 1 … + 5`, and the slot exists only in the row-major case, where `W_RMGB`'s `+ 1` starts past
    it.
  - The trailing `use_welford` appended to `args.writer_receiver` only
    ([:728](device/sharded_layernorm_factory_helpers.cpp#L728)) — the audit's misc anomaly 4. The
    value is already at positional index 4 and again as a named CTA.
  - The pre-all-gather writer's RTA slots 2, 3 and 4 (`eps_u`, `gamma_dram_addr`,
    `beta_dram_addr`) — the audit's misc anomaly 3. `W_PRE` reads only slots 0, 1 and 5. These do not
    vanish outright: the same `build_writer_args` feeds all three writers, and slots 3 / 4 become the
    gamma / beta `TensorBinding` in the two writers that do read them. Under PRE the port simply does
    not declare the names.
- **One legacy read the port cannot reproduce.** `W_STD` / `W_RMGB` compile their write-back block
  under `#ifndef SKIP_WRITE_BACK`, i.e. whenever `!skip_write_back` — but `build_write_back_args`
  emits the runtime args that block reads (`num_segments_to_write_back`, `storage_core_start_offset`
  and the segment array) only under POST
  ([:1641-1646](device/sharded_layernorm_factory_helpers.cpp#L1641-L1646)). So an **ND** program with
  an output shard spec that differs from the input's — reachable, since nothing validates them equal
  — reads runtime args past the end of its own list and then writes through an unconfigured `c_17`.
  Metal 2.0 has no way to read an argument the host never declared, so the port makes the two
  conditions agree: the write-back is compiled when `POST && !skip_write_back`, exactly when its
  arguments exist. Detailed with its consequences in `METAL2_PORT_REPORT.md`.

---

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`. The ported-from factory has no
  `override_runtime_arguments`, so the framework owns the cache-hit binding refresh and the factory
  writes one method, `create_program_artifacts`.
- **Custom `compute_program_hash`**: none — default reflection-based hash. Nothing to preserve.
- **Implementation notes**:
  - The legacy fourth parameter `core_range_set` is **kept**, as
    `create_program_artifacts(attributes, tensor_args, tensor_return_value, core_range_set = std::nullopt)`.
    `ProgramSpecFactoryConcept` only tests for the presence of `create_program_artifacts` and the
    adapter calls it with exactly three arguments, so a defaulted fourth parameter compiles and is
    simply never supplied by the framework path. This is the invoker's decision, taken so the
    containment validation the parameter guards
    ([device/layernorm_op_multi_core_sharded.cpp:41-70](device/layernorm_op_multi_core_sharded.cpp#L41-L70))
    stays in the tree; it differs from Part 1, where the equivalent parameter genuinely chose cores
    and was dropped. It leaves the sharded factory with no caller for the parameter, which is a
    known cost, not an oversight.
  - The sharded `create_descriptor` nanobind
    (the `create_descriptor` static method on the class registration now at
    [layernorm_nanobind.cpp:339](layernorm_nanobind.cpp#L339)) is **deleted** — it would
    otherwise reference a vanished symbol. The `nb::class_<LayerNormShardedProgramFactory>`
    registration itself **stays** (now with no methods) because `select_program_factory` returns that
    type to Python and nanobind needs it registered to convert the variant. User-visible API change,
    recorded in `METAL2_PORT_REPORT.md` under Handoff points.
  - The factory's `create_descriptor` declaration is replaced by `create_program_artifacts` in
    [device/layernorm_device_operation.hpp](device/layernorm_device_operation.hpp). With Part 1
    already converted, the op then has no `create_descriptor` anywhere and
    `<tt-metalium/program_descriptors.hpp>` drops out of the header.
  - `sharded_layernorm_factory_helpers.{hpp,cpp}` are rewritten in place rather than forked: they are
    private to this factory (nothing else includes them), and the descriptor-shaped structs
    (`KernelConfig`, `CBConfig`, `CompileTimeArgs`, `RuntimeArgsResult`) are the descriptor plumbing
    the port replaces.

## Planned Spec Shape

Naming: the legacy `cb_*` names lose the prefix, since the `dfb::` namespace already carries the
role. Where one legacy index carries two unrelated buffers in different stages, each gets its own
name (`c_14` → `in_pre_add` / `mask_scratch`; `c_19` → `col_mask` / `var`; `c_20` → `ex2pe` /
`reciprocal`).

- **KernelSpecs** — up to ten, one per legacy `KernelDescriptor`:
  `READER_SENDER`, `READER_RECEIVER_ALL_TO_ALL`†, `READER_RECEIVER`†, `WRITER_SENDER`,
  `WRITER_RECEIVER`†, `COMPUTE_ALL_TO_ALL`, `COMPUTE_NOT_ALL_TO_ALL`†,
  `IDLE_READER`†, `IDLE_WRITER`†, `IDLE_COMPUTE`† († = built only under the same condition as its
  legacy descriptor).
- **DataflowBufferSpecs** — one per legacy `CBDescriptor` that some kernel in the selected config
  binds, plus one extra for the second `buffer_index` of the two aliased descriptors. Full list with
  allocation conditions in [Planned DFB set](#planned-dfb-set) below.
- **SemaphoreSpecs** — three: `REDUCE_SENDER`, `REDUCE_RECEIVER`, `REDUCE_SECOND_STAGE`, all with
  `target_nodes = mcast_dest_cores`.
- **TensorParameters** — seven: `INPUT`, `RESIDUAL`, `GAMMA_T`, `BETA_T`, `STATS`, `RECIP`, `OUTPUT`,
  each declared under the same condition as the buffer or accessor that uses it.
- **WorkUnitSpecs** — up to four, one per distinct (kernel set, node set):
  `sender` on `sender_cores`, `all_to_all_except_sender` on `all_to_all_workers_except_sender`,
  `not_all_to_all` on `not_all_to_all_workers`, `inactive` on `inactive_cores`.
  `WRITER_SENDER` and `COMPUTE_ALL_TO_ALL` belong to the first two, so their derived node set is
  `all_to_all_cores`.
- **Op-owned tensors** — none.

### Planned DFB set

Re-derived by counting the kernels that touch each buffer **per stage**, rather than copied from the
brief. `P` = PRODUCER, `C` = CONSUMER, `self` = the same kernel bound both ways.

| DFB | index | declared when | ND endpoints | PRE endpoints | POST endpoints |
|---|---|---|---|---|---|
| `in0` | `c_0` | always (borrowed `INPUT`) | compute **self** | compute **self** | compute **self** |
| `in1` | `c_1` | `has_b && !POST` (borrowed `RESIDUAL`) | compute **self** | compute **self** | — dropped |
| `in_pre_add` | `c_14` | `PRE && has_b` (borrowed `INPUT`) | — | compute **self** | — |
| `scaler` | `c_2` | `!welford` | writer P + compute C | writer P + compute C | writer **self** (single-ended) |
| `eps` | `c_3` | `!welford && !PRE` | writer P + compute C | — dropped | writer P + compute C |
| `scaler_global` | `c_4` | `!welford` | writer P + compute C | writer P + compute C | writer P + compute C |
| `gamma` | `c_5` | `has_gamma && !PRE` | writer P + compute C | — dropped | writer P + compute C |
| `beta` | `c_6` | `has_beta && !PRE` | writer P + compute C | — dropped | writer P + compute C |
| `stats` | `c_7` | `POST` (borrowed `STATS`) | — | — | compute **self** |
| `ex_partial` | `c_8` | `!rms_norm && ND` | compute P + reader C | — dropped | — dropped |
| `ex` | `c_9` | `!rms_norm && ND` | compute P + reader C | — dropped | — dropped |
| `ex_external` | `c_10` | `!rms_norm && ND` | reader P + compute C | — dropped | — dropped |
| `ex_partial2` | `c_11` | `!welford && !POST` | compute P + reader C | compute P + reader C | — dropped |
| `ex2` | `c_12` | `!welford` | compute P + reader C | compute P + reader C | compute **self** |
| `ex_external2` | `c_13` | `!welford && !POST` | reader P + compute C | reader P + compute C | — dropped |
| `mask_scratch` | `c_14` | `!welford && do_legacy_layernorm_col_mask` | compute **self** | — | — |
| `ex_global` | `c_15` | `!PRE` | reader P + compute C | — dropped | reader P + compute C |
| `out` | `c_16` | always (borrowed `OUTPUT` unless `writes_back`) | compute **self** | compute **self** | compute **self**, or compute P + writer C when `writes_back` |
| `xmm` | `c_18` | `!PRE` | compute **self** | — dropped | compute **self** |
| `col_mask` | `c_19` | `!welford && do_col_mask` | writer P + compute C | writer P + compute C | — (`do_col_mask` excludes POST) |
| `var` | `c_19` | `POST` | — | — | compute **self** |
| `ex2pe` | `c_20` | `!welford && ND` | compute P + reader C | — dropped | — dropped |
| `stats_reduced` | `c_21` | `POST` | — | — | compute P + reader C |
| `transpose` | `c_22` | `welford` | compute **self** | — | — |
| `x` | `c_24` | always | compute **self** | compute **self** | compute **self** |
| `reciprocals` | `c_25` | `welford` (borrowed `RECIP`) | compute **self** (raw pointer, sync-free) | — | — |
| `x_welford` | `c_29` | `welford_fp32_alias` | compute **self**; alias of `in0` (`!has_b`) or `x` (`has_b`) | — | — |
| ~~`out_resharded`~~ | `c_17` | **never** — replaced by the `OUTPUT` `TensorBinding` (Case 2) | | | |

**No multi-binding flag is set anywhere in this factory.** The brief named five buffers for it
(`c_8` under ND+welford, `c_9` under ND, `c_12` under ND, `c_16` under `writes_back`, `c_21` under
POST), all the same shape: one kernel `reserve_back`/`push_back`es the buffer and then `wait_front`s
its own result, and a second kernel also `wait_front`s it. Every one of them is a **1P+1C**: the
pushing kernel takes the producer side and the second toucher the consumer side, and the first
kernel's read-back of its own result needs no endpoint of its own. That is what the self-audit's
"never stack a self-loop with the flag" check is pointing at, and the framework enforces it — a
self-loop plus a third endpoint is rejected outright, with the reason spelled out ("when a DFB is
self-looped, every same-side binding must come from a self-loop participant"). Recorded in
`METAL2_PORT_REPORT.md`.

Every DFB is therefore either a self-loop (one touching kernel) or 1P+1C (two touching kernels, one
each side).

**Two endpoints are declared wider than the kernel's actual use**, which is the sanctioned resolution
for an asymmetric conditional touch. Under PRE the combine writes into `ex2` or `out` and runs only on
the gathering cores, but both buffers exist on every node the reader spans, so every compute instance
declares its producer side. And `out` under PRE has no reader at all — the statistics leave the device
through the all-gather — so compute holds both of its ends.

### Placement notes

Metal 2.0 derives a DFB's node set from its bound kernels, so the legacy core-range choices have to
be reproduced through the bindings:

- The **idle triple** exists to give the inactive cores inside the multicast bounding box the CB
  configuration legacy installed there with its post-pass widening. Each idle kernel binds the same
  set its active counterpart does — its source is the same file with `IDLE_CORE` added, and the body
  after the early `return` still compiles, so every `dfb::` name it mentions needs a binding. Net
  effect: every non-borrowed `all_cores` DFB lands on `mcast_dest_cores`, exactly as legacy. Borrowed
  DFBs are widened too, which legacy did not do — but a borrowed DFB takes no per-core allocation
  (its address is the tensor's), so the SRAM layout is unchanged and only a CB-config entry appears
  on cores that never read it. Same mechanism, and same harmlessness, as legacy's own `c_17` entry on
  the storage cores.
- The four **`sender_cores`-only** CBs (`c_7`, `c_21`, `c_19`-as-`var`, and PRE's `c_16`) are named by
  compute kernels that run on the wider `all_to_all_cores`. Every reference to them sits inside an
  `is_allgather_worker` guard, so the port promotes that CTA to an `IS_ALLGATHER_WORKER` define and
  `#ifdef`-gates the aliases, which keeps them bound on `COMPUTE_ALL_TO_ALL` alone. Their node set is
  then `all_to_all_cores`, which **equals `sender_cores`** in every non-two-stage PRE / POST program
  (see the excluded-combinations note above). Under a two-stage reduce it is wider, so `c_21` and
  `c_19`-as-`var` gain an allocation on the non-sender all-to-all cores — a configuration where the
  legacy kernels were already touching CB indices their own core had no configuration for. Recorded
  in `METAL2_PORT_REPORT.md`.
- `IS_ALLGATHER_WORKER` is needed anyway: the two compute `KernelSpec`s have genuinely different
  runtime-arg schemas (the all-to-all one carries `num_rows_per_all_to_all_worker`,
  `use_two_stage_reduce`, `is_second_stage_reader` and, under POST, `num_distributed_blocks`), and a
  named argument must exist at compile time even where legacy guarded the read with
  `is_allgather_worker ? get_arg_val(1) : 0`.

### Alias groups

One `advanced_options.alias_with` clique, conditional, with the primary chosen by the fused flag:

| group | gate | legacy site |
|---|---|---|
| `in0` ↔ `x_welford` | `welford_fp32_alias && !has_b` | [:1148-1153](device/sharded_layernorm_factory_helpers.cpp#L1148-L1153) |
| `x` ↔ `x_welford` | `welford_fp32_alias && has_b` | [:1207-1212](device/sharded_layernorm_factory_helpers.cpp#L1207-L1212) |

Both legality rules hold by construction: the pair mutually names the other, both take their size
from the one legacy `CBDescriptor::total_size`, and both are bound only by the compute kernels, so
they target the same node set.

`c_0` and `c_14` under PRE-with-residual both borrow the **input** tensor — that is how the pre-add
result is written back over `a`'s own shard. Two DFBs borrowing the same `TensorParameter` expresses
it directly; it is not an `alias_with` group.

### `unpack_modes`

Legacy fills a `vector<UnpackToDestMode>` of size `NUM_CIRCULAR_BUFFERS` with `Default` everywhere
except `c_29` under `welford_fp32_alias`
([:1073-1078](device/sharded_layernorm_factory_helpers.cpp#L1073-L1078)). That becomes a single
`{X_WELFORD, UnpackMode::UnpackToDest}` entry under the same gate.

The Metal 2.0 validator additionally **requires** an explicit entry for every `Float32` DFB a compute
kernel *consumes* when `enable_32_bit_dest` (= `fp32_dest_acc_en`) is true. With that flag on,
`cb_data_format` is `Float32`, which sweeps in most intermediates, so the port walks each compute
`KernelSpec`'s own CONSUMER bindings and emits `UnpackToSrc` for every `Float32` DFB that is not the
alias — deriving each value from the legacy vector's `Default` rather than guessing.

## Preserved Multiplicity

none — no work-split multiplicity in legacy. The several `KernelDescriptor`s per role cover disjoint
node sets (one reader / writer / compute per node), which is ordinary 1:1 placement expressed as
several `KernelSpec`s, not a dual-instance work split. Same-source specs over disjoint node sets each
bind one role legally, which is what lets `scaler_global` take a PRODUCER binding from both
`WRITER_SENDER` and `WRITER_RECEIVER`.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| [device/sharded_layernorm_factory_helpers.cpp:1697](device/sharded_layernorm_factory_helpers.cpp#L1697) writer RTA slot 3 | `gamma_dram_addr`, with a `buffer_binding` on the same index ([:816-834](device/sharded_layernorm_factory_helpers.cpp#L816-L834)) | `TensorParameter GAMMA_T` + `TensorBinding{GAMMA_T, "gamma"}` on the writers |
| [:1698](device/sharded_layernorm_factory_helpers.cpp#L1698) writer RTA slot 4 | `beta_dram_addr`, likewise bound | `TensorParameter BETA_T` + `TensorBinding{BETA_T, "beta"}` |
| [:821-834](device/sharded_layernorm_factory_helpers.cpp#L821-L834) `bind_writer_gamma_beta` | patches those two slots on cache hits | **deleted** — the typed binding does the refresh |
| [:692-693, 702-703](device/sharded_layernorm_factory_helpers.cpp#L692-L703) writer CTAs | `TensorAccessorArgs(gamma_buffer).append_to(...)` ×2 per writer, consumed by the `TensorAccessorArgs<5>()` → `next_compile_time_args_offset()` chain | binding mechanism end-to-end; the kernels collapse to `TensorAccessor(tensor::gamma)` / `(tensor::beta)` |
| [device/kernels/dataflow/reshard_writer.hpp:39](device/kernels/dataflow/reshard_writer.hpp#L39) | `dfb_out_resharded.get_write_ptr()` on the buffer-backed `c_17` | `TensorParameter OUTPUT` + `TensorBinding{OUTPUT, "dst"}`; the kernel pulls the base with `TensorAccessor(tensor::dst).get_bank_base_address()` and the write-back arithmetic is unchanged. `c_17` gets no DFB. |
| [:904-963](device/sharded_layernorm_factory_helpers.cpp#L904-L963) all three named-CTA tables | 34 CB-index entries across reader / writer / compute | one `DFBBinding` per DFB per kernel |
| [:623-683](device/sharded_layernorm_factory_helpers.cpp#L623-L683) reader positional CTAs (slots 0, 1, 16 / 14) | `reduce_receiver_semaphore_id`, `reduce_sender_semaphore_id`, `reduce_second_stage_semaphore_id` | `SemaphoreBinding`s; the kernels read `Semaphore<>(sem::reduce_receiver)` etc. |
| [:623-643](device/sharded_layernorm_factory_helpers.cpp#L623-L643) remaining reader positional CTAs | 17 positional values | named: `num_blocks`, `block_h`, `block_h_size_bytes`, `num_all_to_all_workers_first_stage`, `num_tiles_per_worker`, `num_tiles_per_worker_bytes`, `num_tiles_per_worker_last`, `num_tiles_per_worker_last_bytes`, `row_major`, `num_x`, `num_y`, `use_two_stage_reduce`, `num_blocks_first_stage`, `num_blocks_second_stage`, `num_mcast_dests`, `is_all_to_all_worker`. `rms_norm` and `use_welford` are **not** carried across: each gates a conditionally-present buffer index, so they become the `RMSNORM` / `USE_WELFORD` defines |
| [:686-728](device/sharded_layernorm_factory_helpers.cpp#L686-L728) writer positional CTAs | 5 + accessor blocks + flags + 3 write-back values | named: `is_all_to_all_worker`, `block_w`, `gamma_is_float32`, `beta_is_float32`, `worker_core_stride_w_bytes`, `storage_core_stride_w_bytes`, `block_ht`, `logical_K`. `fuse_gamma` / `fuse_beta` / `use_welford` become the `FUSE_GAMMA` / `FUSE_BETA` / `USE_WELFORD` defines (each gates a conditionally-present buffer) |
| [:706-712](device/sharded_layernorm_factory_helpers.cpp#L706-L712) writer CTA | `gamma_stick_size` / `beta_stick_size` | **dropped** — read by neither writer source |
| [:728](device/sharded_layernorm_factory_helpers.cpp#L728) writer_receiver trailing CTA | a second `use_welford` | **dropped** — read by nothing |
| [:732-746](device/sharded_layernorm_factory_helpers.cpp#L732-L746) compute positional CTAs | 14 values (+6 under `use_welford`) | named: `is_top_row`, `num_blocks_first_stage`, `block_h`, `block_w`, `subblock_w`, `num_subblocks_w`, `num_tiles_per_block`, `float32_dtype`, `float32_reduction`, `legacy_rsqrt`, `num_blocks_second_stage`, and under Welford `tile_width`, `last_tile_w`, `W`, `eps`, `per_core_recip_lut_size`, `last_block_wt`. `do_gamma` / `do_beta` / `is_all_to_all_worker` / `welford_fp32_alias` become the `FUSE_GAMMA` / `FUSE_BETA` / `IS_ALLGATHER_WORKER` / `WELFORD_FP32_ALIAS` defines |
| [:1560-1572, 1599-1611](device/sharded_layernorm_factory_helpers.cpp#L1560-L1611) reader RTA tail | `num_x` X-coords then `num_y` Y-coords, read via `get_arg_addr(6)` cast to an `L1Ptr` | `advanced_options.num_runtime_varargs`; the kernels fill a small local array from `get_vararg(i)` and keep `compute_single_stage_noc_addrs` / `compute_two_stage_noc_addrs` unchanged |
| [:1641-1676](device/sharded_layernorm_factory_helpers.cpp#L1641-L1676) writer RTA tail | write-back segments, 3 args each, count at slot 6, read via `get_arg_addr(8)` | `num_runtime_varargs`; the writer fills a local array from `get_vararg(i)` and keeps `write_resharded_data`'s `segment_args` pointer parameter |

No page-size third-argument CTA exists in this factory.

**Vararg counts are per node**, because the reader's block is `num_x + num_y` (with `num_x = num_y =
1` on the not-all-to-all descriptor) and the writer's is `2 + 3 * num_segments`, where the segment
count is computed per core by `build_write_back_args`. `num_runtime_varargs` is a per-kernel scalar,
so the writer uses `advanced_options.num_runtime_varargs_per_node` (the deprecated per-node override)
for its ragged block. The readers' counts are uniform per `KernelSpec`, so the scalar suffices there.

## Applied Patterns

- [Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding)
  — the compute-private intermediates in every stage (`x`, `xmm`, `transpose`, `var`, `mask_scratch`),
  plus the borrowed inputs `in0`, `in1`, `in_pre_add`, `stats` and `out` where compute is the only
  toucher, and `scaler` on the writer under POST.
- [Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb)
  — `reciprocals` (raw pointer, no FIFO ops) and `scaler` under POST (filled, never drained).
- [Two-toucher DFB → assign 1P+1C](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split)
  — the reduce pipeline's partial / external / global buffers.
- [Aliased DFBs](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-aliased-dfbs-legacy-aliased-cbs)
  — the one conditional `in0`/`x` ↔ `x_welford` clique.
- [Same-FIFO aliasing](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-same-fifo-aliasing-one-dfb-multiple-kernel-side-names)
  — the *path-dependent* variant, several kernel-side names resolving to another DFB's handle rather
  than to a DFB of their own: `dfb_fusion` → `xmm`; `dfb_xmm_id` → `in0` under
  `RMSNORM && !FUSE_PRE_ADD`; `dfb_in_id` → `in0` / `x` / `in_pre_add` by stage and fused flag;
  `dfb_im_id` / `dfb_outgamma_id` → `x` / `xmm` / `out`; `dfb_x_welford_id` → the primary when the
  alias is off; `dfb_x2_id` → `x`. Each becomes a `constexpr` handle alias, `#ifdef`-gated where the
  branch it selects can be absent.
- [Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings)
  — `in1` / `in_pre_add` (`FUSE_PRE_ADD`), `gamma` / `beta` (`FUSE_GAMMA` / `FUSE_BETA`),
  `scaler` / `eps` / `scaler_global` / `ex_partial2` / `ex2` / `ex_external2` / `ex2pe`
  (`USE_WELFORD`), `ex_partial` / `ex` / `ex_external` (`RMSNORM`), `col_mask` / `mask_scratch`
  (`DO_COL_MASK`), `x_welford` (`WELFORD_FP32_ALIAS`), `out` on the writer (`SKIP_WRITE_BACK`),
  `stats` / `stats_reduced` / `var` and PRE's `out` (`IS_ALLGATHER_WORKER`).
  Five of those gates are **CTA-to-define promotions**: `rms_norm`, `use_welford`, `do_gamma` /
  `do_beta`, `welford_fp32_alias` and `is_allgather_worker`.
- [Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers)
  — `reduce_init`, `reduce_tile`, `pack_tile`, `add_tiles`, `transpose_tile`,
  `compute_kernel_hw_startup`, `reconfig_data_format*`,
  `dataflow_kernel_lib::prepare_reduce_scaler<dfb::scaler, …>` and
  `compute_kernel_lib::reduce<…, dfb::x, …>` (template-argument position),
  `norm::kernel_util::compute::mask_block_in_place`, and
  `get_pointer_to_cb_data<recip_lut_t>(dfb::reciprocals, 0)`.
- [Caution: Avoid varargs](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary)
  — five genuine variable-count blocks retained as varargs; every other argument is named.
- [Removing pybound legacy factory entry points](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-removing-pybound-legacy-factory-entry-points)
  — the sharded `create_descriptor` nanobind.

## Deferred / Flagged

- **New findings during planning:**
  - The thirteen config-dead CBs under PRE and POST, contradicting the brief's "No dead CBs anywhere
    in this op." Dropped per the recipe's dead-CB rule; each recorded in the report.
  - The ND-with-reshard write-back that reads runtime args the host never emits (Flags above). The
    port narrows the compile gate to `POST && !skip_write_back` because Metal 2.0 cannot express the
    out-of-range read. This is the one place the port does not reproduce legacy behavior, and it is
    written up prominently in the report.
  - `c_2` is filled by the writer and read by nobody under POST — a single-ended buffer, self-looped
    rather than dropped.
  - The brief's five multi-binding buffers are all 1P+1C, not flags. The first draft of this plan
    read the census the brief's way and the framework rejected it; the corrected reading is above and
    the episode is written up in the report.
  - `eps_u` and the packed `cinv` / `winv` scalars are per-node runtime args whose value is identical
    on every node, so they are really common runtime args. The port keeps them as RTAs — converting
    changes dispatch semantics and is a separate cleanup.
