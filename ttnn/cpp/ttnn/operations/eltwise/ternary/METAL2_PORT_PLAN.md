# Port Plan — eltwise/ternary

Port plan for `ttnn/cpp/ttnn/operations/eltwise/ternary/`, ported from the `ProgramDescriptor` API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

Line numbers are those of the pre-port tree: `anasuya/metal2_ternary_port` at `21d0f3c7ba9` (*Pre-port fixes*) on
`origin/main` `d764c9f5ce2`. Kernel paths are relative to `device/kernels/`.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept`. `TernaryProgramFactory::create_descriptor` returns a `ProgramDescriptor`
  (`device/ternary_program_factory.cpp:997-1541`), and `override_runtime_arguments` re-applies the per-dispatch state on
  a cache hit (`:1543-1649`).
- Where the factory methods live: in a conventional `program_factory_t` variant,
  `std::variant<TernaryProgramFactory>` (`device/ternary_device_operation.hpp:49-67`). Not the direct-descriptor shape, so
  exception 3 does not apply.
- Variants: one factory. It serves every op type (WHERE, LERP, ADDCMUL, ADDCDIV, SNAKE_BETA, MAC), every variant (TTT,
  TTS, TST) and every broadcast type, by choosing kernel sources at run time (see *Runtime kernel-source selection*).
- Custom `compute_program_hash`: present at `device/ternary_device_operation.cpp:592-653`, layered over
  `operation_attributes_t::to_hash()` (`:264-274`) — left intact.

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's TTNN factory analysis
section. Carried forward in the [TTNN ProgramFactory](#ttnn-programfactory) section below.)*

### Kernels

One `KernelDescriptor` per kernel role, each over `all_device_cores` (= `operation_attributes.worker_grid`). No
work-split multiplicity: per-core counts travel as RTAs.

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | one of 11 (below) | `all_device_cores` | TTS/TST: `c_0, c_1`, 2 × `TensorAccessorArgs(RuntimeTensorShape)`, `has_sharding` (`:1274-1295`). TTT: `c_0, c_1, c_2`, 3 × accessor args, `has_sharding` (`:1296-1308`) | none | 27 per core (`kNumReaderArgs`, `:648`); slots 0-2 are tensor base addresses | the accessor args' runtime words (`reader_common_runtime_args`) | dataflow fill defines, `SRC_BCAST_A/B/C`, `SRC_SHARDED_A/B/C`, `SRC_ROW_BCAST_*` / `SRC_SCALAR_*` (ROW_COL), `BCAST_LLK` (TTT) | O2 (unset) | `ReaderConfigDescriptor{}` → RISCV_1, NOC_0, dedicated |
| writer | `dataflow/ternary_writer_nobcast.cpp` | `all_device_cores` | `c_3`, `TensorAccessorArgs(output, RuntimeTensorShape)`, `has_sharding` (`:1331-1335`) | none | 11 per core; slot 0 is the output base address, slot 10 is padding the kernel never reads | accessor runtime words | `DST_SHARDED` (`:1337`) | O2 (unset) | `WriterConfigDescriptor{}` → RISCV_0, NOC_1, dedicated |
| compute | one of 10 (below) | `all_device_cores` | `num_tiles_per_cycle = 1`, `scalar_is_true_value` (`:1392-1394`) | none | 4 per core: `num_tiles, freq, counter, scalar_arg` | none | `BCAST_A/B/C`, `TERNARY_SFPU_OP_INIT/FUNC`, `FILL_LLK`, `FILL_WITH_VALUE_*`, `ADDCMUL_DATA_FORMAT` | O3 (unset; `ComputeConfigDescriptor` resolves to O3) | `ComputeConfigDescriptor{.fp32_dest_acc_en, .unpack_to_dest_mode}` (`:1470-1473`); every other field default |

*(`grep -n opt_level` on the factory prints nothing: every kernel resolves to its type's default.)*

### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) | backing buffer |
|---|---|---|---|---|---|---|
| c_0 | pred tile × (pred shard volume, else 2) | `all_device_cores` | predicate | pred tile size | unset | predicate buffer when `has_sharding` ∧ predicate sharded (`:1069-1080`) |
| c_1 | b tile × (b shard volume, else 2) | same | value_true (TTT/TTS), value_false (TST) | b tile size | unset | that tensor when natively sharded (`:1082-1124`) |
| c_2 | false tile × (shard volume, else 2) | same | value_false | tile size | unset | value_false when natively sharded; TTT only (`:1126-1136`) |
| c_4 / c_5 / c_6 | tile × 2 | same | predicate / value_true / value_false | tile size | unset | none; allocated whenever TTT ∧ ROW_BCAST (`:1141-1169`) |
| c_3 | out tile × (out shard volume, else 2) | same | output | out tile size | unset | output buffer when natively sharded (`:1171-1183`) |

No GlobalCircularBuffer, no multi-element `format_descriptors` (no aliased CBs).

### Semaphores
none

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `ternary_program_factory.cpp:1279-1283` (TTS), `:1290-1294` (TST), `:1300-1307` (TTT) | predicate; value_true (TTS/TTT); value_false (TST/TTT) | reader 0 / 1 / 2 (`:500-501`, `:864-866`; re-wrapped as `Buffer*` at `:1483-1533`) |
| `:1333-1334` | output | writer 0 (`:904`) |

All 28 kernel-side constructions take two arguments (no third page-size argument).

### Work split
- Driver: `compute_core_partition` (`:549-629`): the shard grid when `has_sharding`; otherwise
  `split_work_to_cores(compute_with_storage_grid | all_device_cores, num_output_tiles, row_major)`.
- num_cores: every core of the worker grid gets a row. Cores outside both core groups get all-zero rows (noop).
- core_group_1 / core_group_2 and their tile counts ride the per-core RTA `num_tiles`; there is one
  `KernelDescriptor` per role, so nothing varies per group at compile time.

### Shared kernels
none. Census `grep -rl <file> ttnn/cpp/ttnn/operations tests/ttnn` over all 26 files under `device/kernels/`: no bound
kernel or header is named outside `eltwise/ternary/`. Positive control: `dataflow/ternary_reader_nobcast_ttt.cpp` hits
`eltwise/binary_backward/device/binary_backward_program_factory.cpp:122`. That file is *lent* to binary_backward and not
bound by ternary, so the port does not touch it. No `_metal2` fork exists or is needed.

Out-of-op headers the kernels include (callees, not shared kernels): `binary_ng/.../dataflow/fill_tile_utils.hpp`
(8 readers; raw L1 pointer helpers), `binary_ng/.../compute/eltwise_utils_common.hpp` / `eltwise_utils_sfpu.hpp`
(5 compute kernels; included, unused), `ttnn/cpp/ttnn/kernel_lib/eltwise` (`ternary_addcmul_int_sfpu.cpp`).

### Runtime kernel-source selection
`TernaryKernelConfig` (`ternary_op_utils.cpp:192-379`), `get_kernel_file_path` (`:381-422`),
`overwrite_compute_kernel_name_and_defines` under `is_llk_bcast` (`ternary_program_factory.cpp:219-247`, applied at
`:1255-1268`) and `override_addcmul_compute_kernel` for INT32/UINT32 addcmul (`:1453-1462`). Selection axes: variant ×
broadcast type × op type × dtype (LLK row broadcast needs all-BF16 inputs and output; `is_fpu` for addc ops; int
addcmul). Every source the factory can bind (22):

- Readers (11): `ternary_reader_nosubtilebcast_ttt.cpp` (TTT NONE/OUTER), `ternary_reader_colbcast_ttt.cpp`,
  `ternary_reader_rowbcast_ttt.cpp`, `ternary_reader_row_col_bcast_ttt.cpp`, `ternary_reader_scalar_ttt.cpp`,
  `ternary_reader_nobcast_tst_tts.cpp` (TTS/TST NONE), `tst_tts_reader_outer_bcast.cpp`, `tst_tts_reader_scalar_bcast.cpp`,
  `tts_tst_reader_col_bcast.cpp`, `tts_tst_reader_row_bcast.cpp`, `tts_tst_reader_row_col_bcast.cpp`.
- Writer (1): `ternary_writer_nobcast.cpp`.
- Compute (10): `ternary_sfpu_no_bcast_ttt.cpp`, `ternary_sfpu_col_scalar_bcast_ttt.cpp`, `ternary_sfpu_row_bcast_ttt.cpp`,
  `ternary_sfpu_no_bcast_tts_tst.cpp`, `ternary_sfpu_col_scalar_bcast_tts_tst.cpp`, `ternary_addc_ops_sfpu.cpp`,
  `ternary_addc_ops_sfpu_bcast.cpp`, `ternary_addc_ops_fpu_rowbcast.cpp`, `ternary_addcmul_int_sfpu.cpp`,
  `ternary_addcmul_int_sfpu_bcast.cpp`; plus the op-local header `compute/ternary_sfpu_copy_init.hpp`.

Per-path DFB roles: on every path the reader produces c_0..c_2 and compute consumes them. c_3 is compute → writer, or a
compute-only sink when the output is natively sharded. c_4..c_6 are touched only by `ternary_sfpu_row_bcast_ttt.cpp` and
`ternary_addc_ops_fpu_rowbcast.cpp`, and only for an operand whose `BCAST_*` is 1; there compute both fills and drains them.

### Flags
- Unreferenced kernel files: `compute/ternary_addc_ops_fpu.cpp`, `compute/ternary_addc_ops_fpu_bcast.cpp` (bound by
  nothing). Not ported, not audited.
- `dataflow/ternary_reader_nobcast_ttt.cpp` is bound by binary_backward only (lent). Not ported.
- The recipe's `cb`-name sweep therefore cannot reach zero over the op directory: those three files keep their legacy
  names, and `device/ternary_device_operation.cpp:8` includes `ttnn/operations/cb_utils.hpp` (device-op class,
  off-limits).
- The TTS/TST readers read RTA 2 (`src2_addr`) into a local they never use; the host writes 0 there (`:502`).

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `CustomProgramSpecFactoryConcept`. The ported-from factory has an
  `override_runtime_arguments`; the port translates it to return `ProgramRunArgs`.
- **Custom `compute_program_hash`**: present at `device/ternary_device_operation.cpp:592-653` (over `to_hash()` at
  `:264-274`) — leave intact.
- **Implementation notes**: the device-op header changes only inside `TernaryProgramFactory` (method signatures). No
  pybind exposure of `create_descriptor` exists, so no binding is removed.

## Planned Spec Shape

- **TensorParameters** (one per io tensor, each with the analysis-doc relaxation
  `{.dynamic_tensor_shape = true, .relax_logical_rank = true}`, unconditionally):
  `predicate`, `value_true` (TTT, TTS), `value_false` (TTT, TST), `output`.
- **DataflowBufferSpecs** (one per legacy CB; spec name = kernel accessor name). Declared in legacy CB-index order
  (`in_a`, `in_b`, `in_c`, `out`, then the conditional staging DFBs). DFB device slots follow declaration order, so
  for TTT they equal the legacy CB indices whenever all or none of the staging DFBs exist. No kernel depends on a
  slot number after the port.

  | DFB | legacy | entry_size / num_entries / format | borrowed_from | endpoints |
  |---|---|---|---|---|
  | `in_a` | c_0 | as legacy | `predicate` when `SRC_SHARDED_A` | reader P, compute C |
  | `in_b` | c_1 | as legacy | `value_true` (TTT/TTS) / `value_false` (TST) when `SRC_SHARDED_B` | reader P, compute C |
  | `in_c` | c_2 (TTT) | as legacy | `value_false` when `SRC_SHARDED_C` | reader P, compute C |
  | `bcast_a/b/c` | c_4/5/6 | as legacy | — | compute P + C (self-loop); spec and binding only when live |
  | `out` | c_3 | as legacy | `output` when `DST_SHARDED` | compute P, writer C; compute P + C when `DST_SHARDED` |

  `tile_format_metadata` stays unset, as the legacy `tile` field was.
- **KernelSpecs**: `reader`, `writer`, `compute` — one per legacy `KernelDescriptor`, same sources, same defines.
  - reader binds `tensor::in_a/in_b/in_c` only for slots whose `SRC_SHARDED_*` is 0.
  - writer binds `dfb::out` and `tensor::out` only when `DST_SHARDED` is 0.
  - hw_config: `create_reader_datamovement_config()` / `create_writer_datamovement_config()` (values equal the legacy
    triples).
  - compute: `ComputeHardwareConfig{.enable_32_bit_dest = fp32_dest_acc_en, .unpack_modes = ...}` (Style B: the op sets
    a Metal config directly, so no TTNN helper), `opt_level = O3`.
- **SemaphoreSpecs**: none.
- **WorkUnitSpecs**: one, `{reader, writer, compute}` on `all_device_cores`.

## Preserved Multiplicity

none — no work-split multiplicity in legacy (one `KernelDescriptor` per role; per-group counts are RTAs).

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA 0 (`ternary_program_factory.cpp:500`, `:864`) | predicate `buffer()->address()`, re-wrapped as `Buffer*` (`:713`) | `TensorBinding(predicate, "in_a")` when `SRC_SHARDED_A=0`; nothing when 1 (the borrowed `in_a` carries the buffer) |
| reader RTA 1 (`:501`, `:865`) | value_true (TTT/TTS) / value_false (TST) address | `TensorBinding(..., "in_b")` when `SRC_SHARDED_B=0` |
| reader RTA 2, TTT (`:866`) | value_false address | `TensorBinding(value_false, "in_c")` when `SRC_SHARDED_C=0` |
| reader RTA 2, TTS/TST (`:502`) | constant 0 for the absent operand, read into an unused local | dropped (no tensor exists) |
| reader CTAs 0-1 / 0-2 (`:1276-1277`, `:1288-1289`, `:1297-1299`) | CB ids | `DFBBinding(in_a / in_b / in_c, PRODUCER)` |
| reader CTA accessor blocks + CRTAs (`:1279-1307`) | `TensorAccessorArgs(buffer, RuntimeTensorShape).append_to(...)`; kernel `TensorAccessorArgs<N, M>()` chains | the tensor bindings; `TensorAccessor(tensor::in_x)` |
| reader trailing CTA (`:1284`, `:1295`, `:1308`) | positional `has_sharding` | named CTA `has_sharding` |
| reader RTAs 3-26 | positional | named: `num_tiles, start_id, nD_stride, d_stride, n_stride, c_stride, D, N, C, Ht, Wt, cND, nD_stride_b, d_stride_b, n_stride_b, c_stride_b, src_num_tiles_b, nD_stride_c, d_stride_c, n_stride_c, c_stride_c, src_num_tiles_c, dst_shard_width, src_num_tiles` |
| `tts_tst_reader_row_bcast.cpp:47-48` | hardcoded `tt::CBIndex::c_0` / `c_1` (CTAs 0-1 ignored) | `dfb::in_a` / `dfb::in_b` |
| writer RTA 0 (`:904`) | output address | `TensorBinding(output, "out")` when `DST_SHARDED=0` |
| writer RTA 10 (`:914`) | padding, never read | dropped |
| writer CTA 0 + accessor block + trailing CTA (`:1331-1335`) | CB id, accessor args, `has_sharding` | `DFBBinding(out, CONSUMER)` when `DST_SHARDED=0`; tensor binding; named CTA `has_sharding` |
| writer RTAs 1-9 | positional | named: `num_tiles, start_id, dst_shard_width, D, N, C, Ht, Wt, cND` |
| compute CTAs (`:1393-1394`) | positional `num_tiles_per_cycle`, `scalar_is_true_value` | named `num_tiles_per_cycle`, `scalar_is_true` |
| compute RTAs (`:943`) | positional | named `num_tiles, tile_freq, tile_start, scalar_arg` |
| every compute kernel and `ternary_sfpu_copy_init.hpp` | hardcoded `tt::CBIndex::c_0..c_6`; `dfb.get_id()` at LLK sites | `dfb::` tokens at construction and at LLK / kernel-lib call sites |
| `process_tile(uint32_t …cb_id, …)` helpers (four compute kernels) | CB ids passed in as parameters, `DataflowBuffer` built from them | parameters dropped; the helper builds its `DataflowBuffer`s from the tokens |
| `unpack_to_dest_mode` (`:1357-1387`) | `vector<UnpackToDestMode>(NUM_CIRCULAR_BUFFERS)` by CB id | `unpack_modes` keyed by DFB name; an `UnpackToDest` entry for each legacy `UnpackToDestFp32` slot (in_a/in_b/in_c/out), legacy `Default` omitted |
| `TernaryPerCoreArgs::{reader,writer}_buffer_slots`, `is_work_core`, `emplace_runtime_args` re-wrap (`:665-683`, `:1483-1533`) | descriptor-only `Buffer*` binding plumbing | gone |
| override (`:1575-1648`) | `GetRuntimeArgs` copy-in; `UpdateDynamicCircularBufferAddress` per borrowed CB | `kernel_run_args` for all three kernels; `tensor_args` for every TensorParameter |
| c_4 / c_5 / c_6 where no kernel touches them (`:1141-1169`) | allocated for every TTT ∧ ROW_BCAST | no spec |

## Applied Patterns

- *Two-toucher DFB → assign 1P+1C* (`port_patterns.md`): `in_a`, `in_b`, `in_c` (reader PRODUCER, compute CONSUMER, real FIFO);
  `out` when `DST_SHARDED=0` (compute PRODUCER, writer CONSUMER).
- *Sync-free and single-ended CBs → self-loop DFB* (`port_patterns.md`): `out` when `DST_SHARDED=1` (the packer writes into the
  resident output shard and nothing drains it; the writer's body is compiled out).
- *Self-loop DFB binding* (`port_patterns.md`): `bcast_a/b/c` — compute fills each with `unary_bcast` and drains it into
  `copy_tile`.
- *Conditional / optional resource bindings* (`port_patterns.md`):
  - `bcast_a/b/c`: spec and binding exist only under TTT ∧ ROW_BCAST ∧ an LLK row-broadcast compute kernel ∧ that
    operand's `BCAST_*` = 1. The existing `BCAST_*` defines gate them kernel-side. `ternary_addc_ops_fpu_rowbcast.cpp`
    builds them unconditionally today, so those constructions move under the matching `#if`.
  - Reader tensor bindings follow `SRC_SHARDED_*`. The kernels already compile the accessor out under those defines.
  - Writer `dfb::out` / `tensor::out` follow `DST_SHARDED`. `ternary_writer_nobcast.cpp` builds `dfb_out` outside
    `#if !DST_SHARDED` today, so that construction moves inside.
  - Host side, each binding is read back from the define value the kernel receives, so the two share one condition.
- Borrowed-memory DFBs: `in_a/in_b/in_c/out` take `borrowed_from` exactly where the legacy CB had `.buffer` set.
- *Pass DFB handles directly to LLKs and kernel-lib helpers* (`port_patterns.md`): all compute kernels, including
  `ckl::input(dfb::in_a, …)` (kernel_lib has a `DFBBindingToken` overload) and `copy_init_shared_with_c0<dfb::in_b>()`.
- Rule 7 `constexpr` exception: `init_once` and `copy_init_shared_with_c0` read `unpack_*[...]` in constant expressions,
  so they keep the array form indexed by the token (`unpack_src_format[dfb::in_a]`).

## Deferred / Flagged

- **`ternary_sfpu_copy_init.hpp` hardcodes `tt::CBIndex::c_0`.** Settled with the invoker (2026-10-27): replace it in
  place with `dfb::in_a` (rule 2), keeping the helper's signature. Both including kernels bind `in_a`. Without the swap
  the helper would compare against whichever DFB the framework puts in slot 0.
- **Cache-hit refresh of the accessor runtime words.** The legacy override deliberately left the
  `TensorAccessorArgs` common runtime args alone (`:1555-1570`). In Metal 2.0 a tensor binding's runtime-field words
  (the sharded shape in pages) travel with its address, so supplying the `TensorArgument` that refreshes the address
  re-emits them too. The hash pins those words (`sharded_tensor_shape_in_pages`, `ternary_device_operation.cpp:640-643`),
  so the re-emitted values equal the cached ones. Recorded, not a stop.
- **Validation forcing.** The recipe's `skip_validation` edits in `tt_metal/impl/` are not applied. Hit-path checks are
  turned on with `TTNN_CONFIG_OVERRIDES='{"validate_program_args": true}'`; the miss path always validates.
- No new feature gate, no Case 2 binding, no varargs, no GlobalCircularBuffer, no semaphore.
