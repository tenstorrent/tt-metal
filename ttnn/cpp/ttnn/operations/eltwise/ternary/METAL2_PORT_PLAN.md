# Port Plan — `eltwise/ternary`

Port plan for `eltwise/ternary` (`TernaryDeviceOperation` → `TernaryProgramFactory`), ported from the `ProgramDescriptor`
API (`create_descriptor` + an imperative `override_runtime_arguments`) to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

Base: `anasuya/metal2_port_eltwise_ternary` @ `b90203d162f` (`origin/main` `3f57b50ff00` plus recipe docs). Line numbers
below are this base's. Paths are relative to `ttnn/cpp/ttnn/operations/eltwise/ternary/` unless they start with `ttnn/` or
`tt_metal/`.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept`. `TernaryProgramFactory::create_descriptor` returns a `ProgramDescriptor`
  (`device/ternary_device_operation.hpp:50-53`, body `device/ternary_program_factory.cpp:997-1541`). The factory also
  has an imperative `override_runtime_arguments(Program&, …)` (`.hpp:59-64`, body `.cpp:1543-1649`).
- Where the methods live: in the nested struct `TernaryProgramFactory`, the one alternative of
  `program_factory_t = std::variant<TernaryProgramFactory>` (`.hpp:49-67`). Not the direct-descriptor shape:
  device-op exception 3 does not apply. No `select_program_factory`.
- Variants: single factory. It selects its kernel sources at runtime on four axes (variant TTT / TTS / TST × broadcast
  type × op type × dtype). See *Kernels* and *Runtime kernel-source selection*.
- Custom `compute_program_hash`: present at `device/ternary_device_operation.cpp:582-637`, layered over the backdoor
  `operation_attributes_t::to_hash()` at `:261-271` — left intact.

### Kernels

One `KernelDescriptor` per role. All three run on `all_device_cores` (= `operation_attributes.worker_grid`).

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | one of 11, see below (`:1322`) | `all_device_cores` | TTS / TST: `[c_0, c_1, TA(predicate)…, TA(tensor operand)…, has_sharding]` (`:1274-1295`). TTT: `[c_0, c_1, c_2, TA(predicate)…, TA(value_true)…, TA(value_false)…, has_sharding]` (`:1296-1309`) | none | 27 per core (`kNumReaderArgs`, `:648`), rows from `build_per_core_runtime_args` (`:701-960`); slots 0-2 are `Buffer*` bindings on work cores (`:713-722`, `:1503-1517`); noop cores get all-zero rows (`:784-791`) | the `TensorAccessorArgs` runtime words, `ArgConfig::RuntimeTensorShape` (empty for interleaved tensors; tensor shape in pages for sharded ones) | `reader_defines` (`:1198-1268`): `FILL_TILE_WITH_FIRST_{COLUMN,ROW,ELEMENT}[_B,_C]`, `FILL_WITH_VALUE*` (unused), `SRC_BCAST_A/B/C`, `SRC_SHARDED_A/B/C`, ROW_COL only `SRC_ROW_BCAST_*`, `SRC_SCALAR_*` (`:1211-1252`), TTT only `BCAST_LLK` (`:1255-1268`) | unset → **O2** | `ReaderConfigDescriptor{}` → (RISCV_1, NOC_0, DEDICATED) |
| writer | `device/kernels/dataflow/ternary_writer_nobcast.cpp` | `all_device_cores` | `[c_3, TA(output)…, has_sharding]` (`:1331-1335`) | none | 11 per core (`:649`); slot 0 = output `Buffer*`; slot 10 = padding (`:914`) | `TensorAccessorArgs` runtime words of the output | `DST_SHARDED` (`:1336-1337`) | unset → **O2** | `WriterConfigDescriptor{}` → (RISCV_0, NOC_1, DEDICATED) |
| compute | one of 10, see below (`:1461-1465`) | `all_device_cores` | `[num_tiles_per_cycle = 1, scalar_is_true_value]` (`:1389-1394`) | none | 4 per core: `{num_tiles, freq, counter, scalar_arg}` (`:943`) | none | `kernel_defines`: `BCAST_A/B/C` (`:1397-1427`), `TERNARY_SFPU_OP_INIT/FUNC` (`:1431-1439`), `FILL_LLK`, `FILL_WITH_VALUE_INT/FLOAT` (`:1442-1451`), `ADDCMUL_DATA_FORMAT` (`:1453-1460`) | unset → **O3** (compute default) | `ComputeConfigDescriptor{.fp32_dest_acc_en, .unpack_to_dest_mode}` (`:1470-1473`), everything else default (Style B) |

**Runtime kernel-source selection** (`device/ternary_op_utils.cpp:192-433`, `device/ternary_program_factory.cpp:1254-1268`,
`:1453-1462`). Every selectable source flips with the factory.

- Reader (11): TTT NONE / OUTER → `ternary_reader_nosubtilebcast_ttt.cpp`; TTS / TST NONE → `ternary_reader_nobcast_tst_tts.cpp`;
  TTS / TST OUTER → `tst_tts_reader_outer_bcast.cpp`; TTS / TST SCALAR_A / SCALAR_B → `tst_tts_reader_scalar_bcast.cpp`;
  TTT SCALAR → `ternary_reader_scalar_ttt.cpp`; TTT COL → `ternary_reader_colbcast_ttt.cpp`; TTS / TST COL →
  `tts_tst_reader_col_bcast.cpp`; TTT ROW → `ternary_reader_rowbcast_ttt.cpp`; TTS / TST ROW → `tts_tst_reader_row_bcast.cpp`;
  TTT ROW_COL → `ternary_reader_row_col_bcast_ttt.cpp`; TTS / TST ROW_COL → `tts_tst_reader_row_col_bcast.cpp`.
- Writer (1): `ternary_writer_nobcast.cpp` for every configuration.
- Compute (10): `ternary_sfpu_no_bcast_ttt.cpp` (TTT NONE / OUTER, and ROW without the LLK broadcast; WHERE, LERP, MAC,
  SNAKE_BETA); `ternary_sfpu_col_scalar_bcast_ttt.cpp` (TTT COL / SCALAR / ROW_COL); `ternary_sfpu_row_bcast_ttt.cpp`
  (TTT ROW with `is_llk_bcast`, non-addc ops); `ternary_sfpu_no_bcast_tts_tst.cpp`; `ternary_sfpu_col_scalar_bcast_tts_tst.cpp`;
  `ternary_addc_ops_sfpu.cpp` (ADDCMUL / ADDCDIV NONE / OUTER / non-LLK ROW); `ternary_addc_ops_sfpu_bcast.cpp` (addc
  SCALAR / COL / ROW_COL); `ternary_addc_ops_fpu_rowbcast.cpp` (addc ROW with `is_llk_bcast`); `ternary_addcmul_int_sfpu.cpp`
  and `ternary_addcmul_int_sfpu_bcast.cpp` (ADDCMUL with an INT32 / UINT32 `dtype`).
- Kernel-side header: `compute/ternary_sfpu_copy_init.hpp` (included by `ternary_sfpu_col_scalar_bcast_ttt.cpp` and
  `ternary_addc_ops_sfpu_bcast.cpp`).
- DFB roles do not move between kernels across paths: the reader always produces c_0..c_2, compute always consumes them
  and produces c_3, and the writer consumes c_3 unless `DST_SHARDED`. c_4..c_6 are compute-only.

### CBs

Every CB spans `all_device_cores`, one `CBFormatDescriptor` each, `tile` never set. `num_tiles_per_cb = 2` (`:1065`).

| index | total_size | core_ranges | data_format | page_size | tile (if set) | backing `buffer` |
|---|---|---|---|---|---|---|
| c_0 (predicate) | pred tile × (pred shard volume, else 2) | all | predicate dtype | pred tile | — | `predicate.buffer()` iff `predicate_sharded` (`:1070-1080`) |
| c_1 (TTT / TTS value_true; TST value_false) | tile × (shard volume, else 2) | all | that tensor's dtype | tile | — | iff that tensor is sharded (`:1086-1123`) |
| c_2 (TTT value_false) | tile × (shard volume, else 2) | all | value_false dtype | tile | — | iff sharded (`:1125-1136`) |
| c_4 / c_5 / c_6 (TTT ∧ ROW_BCAST) | 2 tiles of predicate / value_true / value_false format | all | same as c_0 / c_1 / c_2 | tile | — | none (`:1141-1169`) |
| c_3 (output) | out tile × (output shard volume, else 2) | all | output dtype | out tile | — | `output.buffer()` iff `output_sharded` (`:1171-1183`) |

`*_sharded` = `has_sharding` ∧ that slot has a shard volume (`:1053-1061`). No GlobalCircularBuffer, no `address_offset`.

### Semaphores
none

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `device/ternary_program_factory.cpp:1279-1280` (TTS), `:1290-1291` (TST), `:1300-1301` (TTT) | predicate (`input_tensor_a`) | reader 0 (`:500`, `:864`; `Buffer*` at `:713`) |
| `:1281-1283` (TTS) | value_true (`input_tensor_b`) | reader 1 (`:501`, `:720`) |
| `:1292-1294` (TST) | value_false (`input_tensor_c`) | reader 1 (`:501`, `:720`) |
| `:1302-1304` (TTT) | value_true | reader 1 (`:865`, `:715`) |
| `:1305-1307` (TTT) | value_false | reader 2 (`:866`, `:716`) |
| `:1333-1334` | output | writer 0 (`:904`, `:722`) |

Kernel side: 28 `TensorAccessor(args, addr)` constructions, all two-argument (audit). TTS / TST reader slot 2 is a plain
`0u` (`:502`), read by all six TTS / TST readers into `src2_addr` and never used.

### Work split
- Driver: `compute_core_partition` (`:549-629`).
  - Natively sharded: `core_group_1` = the shard grid; cores from `grid_to_cores_with_noop` (`:593-605`); per-core tile
    counts come from the shard-shape generators (`:805-820`).
  - Interleaved, single range from (0,0): `split_work_to_cores(compute_with_storage_grid, num_output_tiles, row_major)`
    (`:606-616`).
  - Otherwise: `split_work_to_cores(all_device_cores, num_output_tiles, row_major)` (`:617-627`).
- num_cores: `num_cores_total` = every core of `worker_grid`. Cores outside the groups are noop cores with zero rows.
- core_group_1 / core_group_2 with `num_tiles_per_core_group_1/2`. No per-group CTA: one `KernelDescriptor` per role.

### Shared kernels
none. The sweep (`grep -rl <file> ttnn/cpp/ttnn/operations/`, with `ternary_reader_nobcast_ttt.cpp` → binary_backward
as the positive control) finds every bound kernel referenced only from this op's `device/ternary_op_utils.cpp`.
`compute/ternary_sfpu_copy_init.hpp` is included only by this op's two compute kernels. No `_metal2` fork exists or is
needed; all 22 kernels convert in place.

### Flags
- **Not bound by this op, left untouched:** `compute/ternary_addc_ops_fpu.cpp` and `compute/ternary_addc_ops_fpu_bcast.cpp`
  (bound nowhere), `dataflow/ternary_reader_nobcast_ttt.cpp` (lent to `eltwise/binary_backward`,
  `binary_backward_program_factory.cpp:122`).
- **Unreachable selections, converted anyway** (one factory flips atomically): the `ComputeRowBcastAddcOp` →
  `ternary_addc_ops_sfpu.cpp` branch (`device/ternary_op_utils.cpp:417-419`); every broadcast reader's
  `SRC_SHARDED_* = 1` branch (native sharding implies broadcast type NONE).
- **Dead reads in the kernels:** reader slot 2 in the six TTS / TST readers (always 0); slots 19 and 26 in the five
  TTS / TST broadcast readers; writer slot 10 is padding the writer never reads.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `CustomProgramSpecFactoryConcept` — the ported-from factory has an
  `override_runtime_arguments`.
- **Custom `compute_program_hash`**: present at `device/ternary_device_operation.cpp:582-637` (+ `to_hash()` at
  `:261-271`) — leave intact.
- **Implementation notes**:
  - `create_program_artifacts` builds the spec. One helper builds the complete `ProgramRunArgs` (every
    `TensorParameter`'s `TensorArgument` plus every named RTA on every core of the worker grid) from the existing
    `build_per_core_runtime_args`. The miss path returns it with the spec; the translated `override_runtime_arguments`
    returns the same thing on every hit. That keeps the legacy "one builder for miss and hit" property.
  - Refresh set, inherited from the legacy override (`:1543-1649`): it re-wrote all 27 / 11 / 4 slots of every core
    (addresses included) and re-pointed every tensor-backed CB. → `tensor_args` for every `TensorParameter`, and every
    named RTA of all three kernels on every core. It did **not** touch the common runtime args (`:1555-1570`). Metal 2.0
    has no op-declared CRTAs here. There is nothing to leave out.
  - Header: the two method declarations change shape (`ProgramArtifacts create_program_artifacts(…)`;
    `ProgramRunArgs override_runtime_arguments(…, coord)` without the `Program&`). The legacy descriptor includes in
    `ternary_device_operation.hpp` are replaced by the Metal 2.0 ones. No pybind change: `ternary_nanobind.cpp` binds no
    factory entry point.

## Planned Spec Shape

- **KernelSpecs** (1:1 with legacy):
  - `reader`: the selected reader source; DM, `create_reader_datamovement_config()` (= RISCV_1 / NOC_0 / dedicated, the
    legacy resolved triple); no `opt_level` (legacy O2 = Metal 2.0 default); defines = `reader_defines` unchanged.
  - `writer`: `ternary_writer_nobcast.cpp`; `create_writer_datamovement_config()` (= RISCV_0 / NOC_1 / dedicated);
    defines = `{DST_SHARDED}`.
  - `compute`: the selected compute source; `ComputeHardwareConfig` built directly (Style B, see *Hardware config*);
    `compiler_options.opt_level = O3`; defines = `kernel_defines` unchanged.
- **DataflowBufferSpecs** (sizes and formats exactly as legacy; `data_format_metadata` = legacy `data_format` on every DFB,
  since compute binds all of them; `tile_format_metadata` unset, as legacy `tile` was unset). Creation order: `in_a`,
  `in_b`, (`in_c`), `out`, then any staging DFB, created in the branch that decides the LLK row broadcast. Slots follow
  creation order; the kernels name every DFB by token. The order has no effect:
  - `in_a` (c_0): `entry_size` = predicate tile, `num_entries` = predicate shard volume or 2; `borrowed_from = predicate`
    when `predicate_sharded`. Reader PRODUCER, compute CONSUMER.
  - `in_b` (c_1): value_true (TTT, TTS) or value_false (TST); borrowed from that `TensorParameter` when sharded. Reader
    PRODUCER, compute CONSUMER.
  - `in_c` (c_2, TTT only): value_false; borrowed when sharded. Reader PRODUCER, compute CONSUMER.
  - `bcast_a` / `bcast_b` / `bcast_c` (c_4 / c_5 / c_6): **conditional** — created and bound only when `is_llk_bcast`
    holds and that operand is row-broadcast (`BCAST_A/B/C` = 1). Compute self-loop (PRODUCER + CONSUMER). 2 tiles of the
    matching input format.
  - `out` (c_3): compute PRODUCER. `DST_SHARDED = 0`: writer CONSUMER. `DST_SHARDED = 1`: `borrowed_from = output`,
    compute self-loop (PRODUCER + CONSUMER), writer not bound.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `predicate`; `value_true` (TTT, TTS); `value_false` (TTT, TST); `output`. Each declared from its
  tensor's `tensor_spec()` with `.relaxations = {.dynamic_tensor_shape = true, .relax_logical_rank = true}`,
  unconditionally (relaxation doc §2, validated by the audit). No `match_page_size`, no `match_padded_shape_only`.
  - Bindings (kernel-side accessor names are slot names, because TTS and TST share sources): reader `src_a` → predicate,
    `src_b` → value_true (TTT, TTS) or value_false (TST), `src_c` → value_false (TTT); writer `dst` → output. Bound
    unconditionally, as legacy appended every `TensorAccessorArgs` unconditionally; the kernels keep constructing the
    accessor only under their existing `SRC_SHARDED_*` / `DST_SHARDED` guards.
- **WorkUnitSpecs**: one, `ternary`, kernels `{reader, writer, compute}`, `target_nodes = worker_grid`.
- **Op-owned tensors**: none.

**Named arguments.** Each role has one fixed name table, shared by every source of that role. The legacy builder rows
and their slot numbering stay; slots that were addresses or padding have no name and are not emitted.

| role | legacy slot → name |
|---|---|
| reader RTA | 0, 1, 2 → *(TensorBindings `src_a` / `src_b` / `src_c`; the TTS / TST slot-2 `0u` goes with them)*; 3 `num_tiles`; 4 `start_id`; 5 `nD_stride`; 6 `d_stride`; 7 `n_stride`; 8 `c_stride`; 9 `D`; 10 `N`; 11 `C`; 12 `Ht`; 13 `Wt`; 14 `cND`; 15 `nD_stride_b`; 16 `d_stride_b`; 17 `n_stride_b`; 18 `c_stride_b`; 19 `src_b_num_tiles`; 20 `nD_stride_c`; 21 `d_stride_c`; 22 `n_stride_c`; 23 `c_stride_c`; 24 `src_c_num_tiles`; 25 `dst_shard_width`; 26 `src_a_num_tiles` |
| writer RTA | 0 → *(TensorBinding `dst`)*; 1 `dst_num_tiles`; 2 `start_tile_id`; 3 `dst_shard_width`; 4 `D`; 5 `N`; 6 `C`; 7 `Ht`; 8 `Wt`; 9 `cND`; 10 → *(padding, dropped)* |
| compute RTA | 0 `num_tiles`; 1 `tile_freq`; 2 `tile_start`; 3 `scalar_arg` |
| reader / writer CTA | `has_sharding` |
| compute CTA | `num_tiles_per_cycle`, `scalar_is_true` |

- Every reader `KernelSpec` declares all 24 reader names, whichever source is selected: legacy pushed the full 27-word
  row to every reader on every core, and the values (zeros included) stay exactly as the builder produced them. A
  source reads only its subset. Names follow the dominant kernel variable for each slot; kernels whose local differs
  keep their local name (`const uint32_t true_nD_stride = get_arg(args::nD_stride_b);`).

## Preserved Multiplicity

none — no work-split multiplicity in legacy (one `KernelDescriptor` per role over one core range set).

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA slot 0 (`device/ternary_program_factory.cpp:500`, `:864`; `Buffer*` `:713`) | `predicate.buffer()->address()` | `TensorBinding{predicate, "src_a"}` |
| reader RTA slot 1 (`:501`, `:865`; `:715`, `:720`) | value_true / value_false address | `TensorBinding{value_true or value_false, "src_b"}` |
| reader RTA slot 2 (`:866`; `:716`) | TTT value_false address | `TensorBinding{value_false, "src_c"}` |
| reader RTA slot 2 (`:502`) | TTS / TST `0u` placeholder for the absent third tensor | dropped with the address slots (never read for anything) |
| writer RTA slot 0 (`:904`; `:722`) | `output.buffer()->address()` | `TensorBinding{output, "dst"}` |
| writer RTA slot 10 (`:914`) | padding `0u` | dropped |
| `TernaryPerCoreArgs::BufferSlot`, `reader_buffer_slots` / `writer_buffer_slots` (`:666-677`, `:711-722`), `emplace_with_buffers` (`:1487-1501`) | `Buffer*` re-wrap of the address slots | gone with the address slots |
| legacy override CB re-point (`:1606-1648`) | `UpdateDynamicCircularBufferAddress` by CB index | `borrowed_from` DFBs, refreshed through the same `tensor_args` |
| reader CTAs 0-1 (TTS / TST), 0-2 (TTT) (`:1276-1277`, `:1288-1289`, `:1297-1299`) | CB indices c_0 / c_1 (/ c_2) | `DFBBinding`s `in_a` / `in_b` / `in_c` |
| `tts_tst_reader_row_bcast.cpp:47-48` | hardcoded `tt::CBIndex::c_0` / `c_1` (the CTAs were ignored) | `dfb::in_a` / `dfb::in_b` |
| writer CTA 0 (`:1331`) | `c_3` | `DFBBinding` `out` |
| compute kernels | hardcoded `tt::CBIndex::c_0`..`c_6` | `dfb::in_a` / `in_b` / `in_c` / `out` / `bcast_a/b/c` |
| `TensorAccessorArgs(...).append_to(...)` (`:1279-1307`, `:1333-1334`) and kernel `TensorAccessorArgs<N, M>()` chains | positional accessor CTAs / CRTAs | binding mechanism (`TensorAccessor(tensor::src_a)` …) |
| `has_sharding` positional CTA after the accessor args (`:1284`, `:1295`, `:1308`, `:1335`) | `get_compile_time_arg_val(srcN_args.next_compile_time_args_offset())` | named CTA `has_sharding` |
| compute CTAs 0, 1 (`:1393-1394`) | positional | named CTAs `num_tiles_per_cycle`, `scalar_is_true` |
| all remaining RTAs | `get_arg_val<uint32_t>(N)` | named, per the table above |
| page-size 3rd accessor argument | — | none present |
| semaphore IDs | — | none present |

## Applied Patterns

- Multi-variant factories: variant / broadcast / op branching stays inside `create_program_artifacts`.
- Conditional / optional resource bindings: `bcast_a/b/c` on the LLK row-broadcast compute kernels, gated by the existing
  `BCAST_A/B/C` defines (`ternary_addc_ops_fpu_rowbcast.cpp` needs the gate added; `ternary_sfpu_row_bcast_ttt.cpp` has
  it); the writer's `out` binding gated by the existing `DST_SHARDED` define (the writer's `DataflowBuffer` moves inside
  `#if !DST_SHARDED`).
- Sync-free and single-ended CBs → self-loop DFB: `out` on compute when `DST_SHARDED = 1` (packer into the resident output
  shard, nothing drains it); `bcast_a/b/c` (compute produces and consumes them).
- Two-toucher → 1P+1C (census re-derived, matches the brief): `in_a`, `in_b`, `in_c` (reader FIFO producer, compute FIFO
  consumer), `out` when `DST_SHARDED = 0` (compute producer, writer consumer). No multi-binding.
- Borrowed-memory DFBs: `in_a/b/c`, `out` when their tensor is natively sharded.
- Pass DFB handles directly to LLKs and kernel-lib helpers; the `.id`/temporary-wrapper anti-pattern removed from the four
  `process_tile` helpers. Same-FIFO aliasing, path-dependent variant: the "effective input" handle in the two LLK
  row-broadcast compute kernels (`dfb::bcast_a` when `BCAST_A`, else `dfb::in_a`).

## Deferred / Flagged

- **`CircularBuffer::get_tile_size()` in the 8 fill-helper readers maps to `DataflowBuffer::get_tile_size()`, not
  `get_entry_size()`.** The brief suggested `get_entry_size()` and asked for a whitelist check. The legacy member is
  `::get_tile_size(cb_id)` = `unpack_tile_size[cb_id]` (`tt_metal/hw/inc/api/dataflow/circular_buffer.h:113`,
  `dataflow_api.h:282-290`); whitelist §A maps that to the member `get_tile_size()`, which reads the same array by the DFB
  id (`dataflow_buffer.h:274-280`). The JIT fills that array at each DFB's slot for every kernel, DM included, when the DFB
  carries `data_format_metadata` (`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3236-3253`). The legacy line is
  `const`, not `constexpr`. The member getter applies. Numerically equal to `get_entry_size()` here either way.
- **`ternary_sfpu_copy_init.hpp`**: `copy_init_shared_with_c0<cb>()` hardcodes `tt::CBIndex::c_0`. The reference buffer
  becomes a second template parameter and the callers pass tokens (`dfb::in_a`, `dfb::in_b`, `dfb::in_c`); the file-scope
  `unpack_src_format[tt::CBIndex::c_0]` becomes `unpack_src_format[dfb::in_a]`. Both are `constexpr` sites (rule 7's
  token form), recorded in the report.
- **Override guards.** The legacy override's two `TT_FATAL`s check the cached `Program` (runtime-arg vector width per
  core; one tensor CB index per globally allocated CB). The Metal 2.0 override does not see the `Program`. The plan
  carries the first one's intent (every core of the worker grid gets a full set of args) and records the second as
  subject-deleted; both are called out in the report for review.
- **Recipe drift seen while planning:** the in-place recipe names `ComputeGen1Config` / `DataMovementGen1Config` /
  `create_reader_datamovement_config(device->arch())`; this tree has `ComputeHardwareConfig` (+ optional `config_1xx`),
  `DataMovementHardwareConfig::config_1xx`, and `create_reader_datamovement_config(bool)`. Same values; report Friction.
