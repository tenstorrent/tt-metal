# Port Plan — nlp_create_qkv_heads

Port plan for `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads`, ported from the
`ProgramDescriptor` host API (`ProgramDescriptorFactoryConcept`, both factories) to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

Both factories (`Interleaved`, `Sharded`) are ported in this change. The port sits on top of the
pre-port offset-split commit (`83142201337`, "pass bare shard bases to the Sharded kernel, add offsets
on device"), which the audit's `Sharded` GREEN was contingent on.

## Legacy Inventory

*Observed against `device/nlp_create_qkv_heads_program_factory.cpp` at `83142201337` (post offset-split).*

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `create_descriptor` returns `tt::tt_metal::ProgramDescriptor`,
  plus a `void override_runtime_arguments(Program&, ...)` on both factories.
- Where the methods live: in the `program_factory_t` variant (`std::variant<Interleaved, Sharded>`,
  `device/nlp_create_qkv_heads_device_operation.hpp:41-69`). Not the direct-descriptor shape.
- Variants: `Interleaved` (input Q interleaved), `Sharded` (input Q sharded); selected by
  `select_program_factory` (`device/nlp_create_qkv_heads_device_operation.cpp:287-294`).
- Custom `compute_program_hash`: none — default reflection-based hash (no `attribute_values` / `to_hash`).

### Variant: Interleaved

#### Kernels
| unique_id (legacy push order) | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs (per core) | CRTAs | defines | opt_level (resolved) | config |
|---|---|---|---|---|---|---|---|---|---|
| compute g1 (only if `transpose_k_heads`) | `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp` (**borrowed**) | `core_group_1` | `[NHtWt = num_blocks_per_core_group_1 * kv_num_tiles]` | none | none | none | none | **O3** (descriptor `opt_level` absent → compute default) | `ComputeConfigDescriptor{.fp32_dest_acc_en = dtype == FLOAT32}`; all other fields default (HiFi4, approx off, dst_full_sync off, no unpack_to_dest_mode, bfp8_pack_precise off) |
| compute g2 (only if `transpose_k_heads && core_group_2.num_cores() > 0`) | same | `core_group_2` | `[NHtWt = num_blocks_per_core_group_2 * kv_num_tiles]` | none | none | none | none | O3 | same |
| reader | `device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads.cpp` (own) | `all_cores` | `[q_num_tiles, kv_num_tiles, TensorAccessorArgs(in0), TensorAccessorArgs(in1 or nullptr placeholder)]` | none | `[in0 Buffer*, in1 Buffer* or literal 0, num_blocks, in0_tensor_tile_id = num_blocks_written*in0_w_tiles, in1_tensor_tile_id = num_blocks_written*in1_w_tiles]` | none | `TRANSPOSE_K_HEADS`?, `READ_FROM_INPUT_TENSOR_KV`?, `KV_TIED`? | O2 | `ReaderConfigDescriptor{}` → RISCV_1 / NOC_0 / DM_DEDICATED_NOC |
| writer | `device/kernels/dataflow/writer_tm_tile_layout_nlp_create_qkv_heads.cpp` (own) | `all_cores` | `[q_out_h_tiles, q_out_w_tiles, q_out_HtWt, num_q_heads (q_out_c), num_kv_heads (kv_out_c), TAA(q), TAA(k), TAA(v)]` | none | `[q Buffer*, k Buffer*, v Buffer*, num_blocks, q_out_h_dim, q_out_tensor_tile_id, k_out_tensor_tile_id, v_out_tensor_tile_id]` | none | `TRANSPOSE_K_HEADS`? | O2 | `WriterConfigDescriptor{}` → RISCV_0 / NOC_1 / DM_DEDICATED_NOC |

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile | condition | buffer |
|---|---|---|---|---|---|---|---|
| 1 (src1, "qv") | `4 * single_tile_size` | `all_cores` | `cb_data_format` (input dtype) | `single_tile_size` | unset | always | none (allocated) |
| 0 (src0) | `4 * single_tile_size` | `all_cores` | same | same | unset | `transpose_k_heads` only | none |
| 16 (out) | `4 * single_tile_size` | `all_cores` | same | same | unset | `transpose_k_heads` only | none |

No GlobalCircularBuffer; no `address_offset`; single-element `format_descriptors` (no aliased CBs).
Kernel-side same-FIFO aliasing of index 1 under `!TRANSPOSE_K_HEADS` (reader `cb_qv`/`cb_k` both index 1;
writer likewise).

#### Semaphores
none

#### Tensor accessors
| host site | originating Tensor | RTA slot (host) | kernel site |
|---|---|---|---|
| `:166` `TensorAccessorArgs(in0_buffer)` | `input_tensor_q` | reader slot 0 (`:311`) | `reader:26,36` `s0` |
| `:168-169` `TensorAccessorArgs(in1 or nullptr)` | `input_tensor_kv` (optional) | reader slot 1 (`:312-316`; literal 0 when absent) | `reader:39-40` `s1` under `READ_FROM_INPUT_TENSOR_KV` |
| `:179` `TensorAccessorArgs(q_buffer)` | output q | writer slot 0 (`:325`) | `writer:32,42` `sq` |
| `:180` `TensorAccessorArgs(k_buffer)` | output k | writer slot 1 (`:326`) | `writer:33,43` `sk` |
| `:181` `TensorAccessorArgs(v_buffer)` | output v | writer slot 2 (`:327`) | `writer:34,44` `sv` |

All two-argument constructions (no page-size third argument).

#### Work split
- Driver: `split_work_to_cores(compute_with_storage_grid_size, num_blocks)` with
  `num_blocks = B * 1 * S / TILE_HEIGHT` (`build_interleaved_work_split`, `:57-82`).
- Yields `(num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2)`.
- Core enumeration order for RTAs: `CoreCoord{i / num_cores_y, i % num_cores_y}` for `i < num_cores` (`:74-76`).

#### Legacy override (`:633-661`)
Address-only: re-applies reader slots 0/1 (`input_q`; `input_kv` only when present) and writer slots 0/1/2
(`q`, `k`, `v`) on every core. No other RTA, no CB re-pointing.

### Variant: Sharded

#### Kernels
| unique_id (legacy push order) | source | core_ranges | CTAs (positional) | RTAs (per core) | defines | opt_level | config |
|---|---|---|---|---|---|---|---|
| reader (instance R) | `device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp` (own) | `q_cores` | `[c_16 (q out), c_17 (k out)]` | 18 fixed slots (`:436-455`): `head_size, per_risc0_out_q_heads, per_core_in_q_heads, remote_q_head_start_idx, q_x, q_y, q_buffer_addr, q_offset=0, read_kv_heads, per_core_out_kv_heads, per_core_in_kv_heads, remote_kv_head_start_idx, kv_x, kv_y, kv_buffer_addr, k_section_offset, k_num_tiles, num_cores_x` then `noc_x_coords[num_cores_x]`, `noc_y_coords[num_cores_y]` | none | O2 | `ReaderConfigDescriptor{}` |
| writer (instance W, **same source**) | same | `q_cores` | `[c_16 (q out), c_18 (v out)]` | same vector with slots 1/3/4/5 = risc1 Q values, slot 7 = `per_risc0_out_q_heads * head_size`, slot 15 = `v_section_offset` on kv cores (`:466-482`) | none | O2 | `WriterConfigDescriptor{}` |

Both instances run on every `q_cores` node (dual-instance work-split). The kernel touches the K/V CB only
where `read_kv_heads` (`i < k_cores.num_cores()`, `:432`), i.e. exactly on `k_cores`.

#### CBs (all borrowed-memory)
| index | total_size | core_ranges | data_format | page_size | buffer |
|---|---|---|---|---|---|
| 16 (q out) | `q_num_tiles * single_tile_size` | `q_cores` | `cb_data_format` | `single_tile_size` | output q (`:523`) |
| 17 (k out) | `k_num_tiles * single_tile_size` | `k_cores` | same | same | output k (`:539`) |
| 18 (v out) | `v_num_tiles * single_tile_size` | `v_cores` (== `k_cores`) | same | same | output v (`:555`) |

No GlobalCircularBuffer; three-arg `UpdateDynamicCircularBufferAddress` (no `address_offset`).

#### Semaphores
none

#### Tensor accessors
none in the kernel (raw `UnicastEndpoint` NoC walk over remote L1 shards). Address-valued RTAs: slot 6
(`input_tensor_q` shard base) and slot 14 (`input_tensor_kv` shard base, or `input_tensor_q`'s when fused).

#### Work split
n/a (no `split_work_to_cores`). `num_cores = max(q_cores, k_cores)`, `cores = grid_to_cores(num_cores,
bbox_x, bbox_y, row_major=true)` over the `q_cores` bounding box (`:405-408`).

#### Legacy override (`:599-629`)
Re-applies slots 6/14 on both instances on every core, and re-points the three borrowed CBs to the three
output buffers. No other RTA.

### Shared kernels
| source | shape | other binders (census: `grep -rl transpose_wh.cpp ttnn/cpp/ttnn/operations/`) | `_metal2` sibling in the original's dir? | rung |
|---|---|---|---|---|
| `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp` | **borrowed** (shared pool) | `nlp_create_qkv_heads_boltz` (`…_boltz_program_factory.cpp:169,179`), `nlp_create_qkv_heads_vit` (`…_vit_program_factory.cpp:103,111`), `split_query_key_value_and_split_heads` (`…_program_factory.cpp:127`) | **no** (`ls ttnn/cpp/ttnn/kernel/compute/` has no `transpose_wh_metal2.cpp`) | **2 — create the fork** `ttnn/cpp/ttnn/kernel/compute/transpose_wh_metal2.cpp` + pointer comment in the original |

A same-body Metal 2.0 fork exists at a different path
(`ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh_metal2.cpp`,
`dfb::cb_in`/`dfb::cb_out`, `NHtWt` as a named RTA). It fails the locational rung-1 test and takes
`NHtWt` as an RTA, so it is **not** bound. The op's three own dataflow kernels are bound by no other op
(`grep -rl` over `ttnn/cpp/ttnn/operations/` returns only this op's factory) — converted in place.

### Flags
- No unreferenced kernel files in the op directory.
- `tensor_args.input_tensor_kv` is copied into a `std::optional<const Tensor>` local in the Interleaved
  factory (`:91`). A `TensorArgument` must reference the framework-visible tensor, so the port reads the
  optional through the reference instead of the copy.
- Sharded: `num_cores = max(q_cores, k_cores)` while both instances run over `q_cores`; equal by
  validation (`num_q_heads >= num_kv_heads`). Not a port concern.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `CustomProgramSpecFactoryConcept` — both factories carry an
  `override_runtime_arguments`. The method's shape changes (`ProgramRunArgs` return, no `Program&`);
  its body becomes a `tensor_args` echo of exactly the set the legacy override refreshed.
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: no pybound `create_descriptor`, no op-owned tensors, no device-op-class edits
  beyond the factory struct declarations in `device/nlp_create_qkv_heads_device_operation.hpp`.

## Planned Spec Shape

### Variant: Interleaved

- **KernelSpecs**: `reader`, `writer`; `compute_g1` (if `transpose_k_heads`) and `compute_g2` (if
  `transpose_k_heads && core_group_2` non-empty), both sourcing the new fork
  `ttnn/cpp/ttnn/kernel/compute/transpose_wh_metal2.cpp`.
  - reader: named CTAs `q_num_tiles`, `kv_num_tiles`; named RTAs `num_blocks`, `in0_tensor_tile_id`,
    `in1_tensor_tile_id`; DFB `qv` PRODUCER (accessor `qv`), DFB `k_in` PRODUCER (accessor `k`, only if
    `transpose_k_heads`); tensors `input_q` (accessor `input_q`), `input_kv` (accessor `input_kv`, only if
    present); defines `TRANSPOSE_K_HEADS`, `READ_FROM_INPUT_TENSOR_KV`, `KV_TIED` as today; hw
    `create_reader_datamovement_config(arch)`; opt_level default (O2 == legacy).
  - writer: named CTAs `q_out_h_tiles`, `q_out_w_tiles`, `q_out_HtWt`, `q_out_c`, `kv_out_c`; named RTAs
    `num_blocks`, `q_out_h_dim`, `q_out_tensor_tile_id`, `k_out_tensor_tile_id`, `v_out_tensor_tile_id`;
    DFB `qv` CONSUMER (accessor `qv`), DFB `k_out` CONSUMER (accessor `k`, only if `transpose_k_heads`);
    tensors `q`, `k`, `v`; define `TRANSPOSE_K_HEADS`; hw `create_writer_datamovement_config(arch)`.
  - compute_gN: named CTA `NHtWt`; DFB `k_in` CONSUMER (accessor `in`), DFB `k_out` PRODUCER (accessor
    `out`); hw `ComputeGen1Config{.enable_32_bit_dest = (dtype == FLOAT32)}`, plus
    `unpack_modes = {{k_in, UnpackToSrc}}` when `enable_32_bit_dest` (legacy `unpack_to_dest_mode` empty
    → Default → UnpackToSrc; the Float32 DFB makes the entry mandatory); `compiler_options.opt_level = O3`.
- **DataflowBufferSpecs**: `qv` (legacy index 1; always), `k_in` (index 0) and `k_out` (index 16) only
  when `transpose_k_heads`. Each `entry_size = single_tile_size`, `num_entries = 4`,
  `data_format_metadata = cb_data_format`. No borrowed memory, no aliasing.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `input_q`, `input_kv` (only if present), `q`, `k`, `v` — from each tensor's
  `tensor_spec()`, strict matching.
- **WorkUnitSpecs**: `g1` = `{reader, writer[, compute_g1]}` over `core_group_1`; `g2` =
  `{reader, writer[, compute_g2]}` over `core_group_2` when non-empty. (`all_cores == g1 ∪ g2`.)
- **Override**: `tensor_args = {input_q, [input_kv], q, k, v}`, `kernel_run_args` empty.

### Variant: Sharded

- **KernelSpecs** (one source, four specs — instance × node set):
  - `reader_kv` / `writer_kv` — the R / W instances on `k_cores` (the cores that hold a K/V output shard).
  - `reader_q` / `writer_q` — the R / W instances on `q_cores \ k_cores` (Q output only).
  - All four: tensor `input_q` (accessor `input_q`); DFB `q_out` (accessor `q_out`; R instances PRODUCER,
    W instances CONSUMER); named RTAs `head_size`, `num_q_heads`, `num_q_heads_per_core`,
    `remote_q_head_start_idx`, `start_q_x`, `start_q_y`, `q_offset`, `num_x`; runtime varargs
    `num_cores_x + num_cores_y` (the NoC x-coordinate table then the y-coordinate table); hw reader /
    writer default config by instance.
  - `*_kv` only: define `READ_KV_HEADS`; DFB `kv_out` self-loop (R → `k_out`, W → `v_out`; PRODUCER +
    CONSUMER on one accessor `kv_out`); tensor `input_kv` (accessor `input_kv`) + define
    `READ_FROM_INPUT_TENSOR_KV`, both only when `input_tensor_kv` is present; extra named RTAs
    `num_kv_heads`, `num_kv_heads_per_core`, `remote_kv_head_start_idx`, `start_kv_x`, `start_kv_y`,
    `kv_section_offset`, `num_kv_tiles`.
- **DataflowBufferSpecs**: `q_out` (index 16, `num_entries = q_num_tiles`, `borrowed_from = q`), `k_out`
  (index 17, `k_num_tiles`, `borrowed_from = k`), `v_out` (index 18, `v_num_tiles`, `borrowed_from = v`);
  `entry_size = single_tile_size`, `data_format_metadata = cb_data_format`.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `input_q`, `input_kv` (only if present), `q`, `k`, `v` (the last three are
  borrow-only — no `TensorBinding`, legal per the borrowed-from exception).
- **WorkUnitSpecs**: `kv_cores` = `{reader_kv, writer_kv}` over `k_cores`; `q_only_cores` =
  `{reader_q, writer_q}` over `q_cores.subtract(k_cores)`, omitted when empty
  (`num_kv_heads == num_q_heads`).
- **Override**: `tensor_args = {input_q, [input_kv], q, k, v}`, `kernel_run_args` empty (the borrowed
  DFBs re-resolve their base from the `q`/`k`/`v` arguments — the translation of the three
  `UpdateDynamicCircularBufferAddress` calls).

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| Interleaved compute g1, g2 of `transpose_wh.cpp` (`:192-208`) | `compute_g1`, `compute_g2` of `transpose_wh_metal2.cpp`, CTA `NHtWt` differing | `g1` (core_group_1), `g2` (core_group_2) — disjoint | `k_in` CONSUMER (both), `k_out` PRODUCER (both): legal disjoint-node multi-KernelSpec endpoints, no flag |
| Interleaved reader, writer (one descriptor each over `all_cores`) | one `reader`, one `writer` | listed in both `g1` and `g2` | — |
| Sharded reader-config + writer-config of the sharded kernel over `q_cores` (`:561-577`) | `reader_kv` + `writer_kv` over `k_cores`; `reader_q` + `writer_q` over `q_cores \ k_cores` | `kv_cores`, `q_only_cores` — disjoint | `q_out`: R instances PRODUCER, W instances CONSUMER (1P+1C per node, two-toucher work-split); `k_out` self-loop on `reader_kv`; `v_out` self-loop on `writer_kv` |

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory `:311` / reader RTA slot 0 | `in0_buffer` (`Buffer*`) | `TensorBinding{input_q, "input_q"}` on reader; kernel `TensorAccessor(tensor::input_q)` |
| factory `:312-316` / reader RTA slot 1 | `in1_buffer` or literal `0` | `TensorBinding{input_kv, "input_kv"}` on reader, only when present; kernel `TensorAccessor(tensor::input_kv)` under `READ_FROM_INPUT_TENSOR_KV`; the literal-0 dead slot vanishes |
| factory `:166` / reader CTA tail | `TensorAccessorArgs(in0_buffer).append_to` | binding mechanism; kernel drops `TensorAccessorArgs<2>()` (`reader:26`) |
| factory `:167-169` / reader CTA tail | `TensorAccessorArgs(in1 or nullptr)` placeholder | binding mechanism; kernel drops `TensorAccessorArgs<in0_args.next_compile_time_args_offset()>()` (`reader:39`) |
| factory `:325-327` / writer RTA slots 0-2 | `q_buffer`, `k_buffer`, `v_buffer` (`Buffer*`) | `TensorBinding{q,"q"}`, `{k,"k"}`, `{v,"v"}` on writer; kernel `TensorAccessor(tensor::q|k|v)` |
| factory `:179-181` / writer CTA tail | three `TensorAccessorArgs(...).append_to` | binding mechanism; kernel drops `TensorAccessorArgs<5>()` and the two chained offsets (`writer:32-34`) |
| reader CTA slots 0-1 (`:162-165`) | positional `q_num_tiles`, `kv_num_tiles` | named CTAs of the same names |
| reader RTA slots 2-4 (`:317-319`) | positional | named RTAs `num_blocks`, `in0_tensor_tile_id`, `in1_tensor_tile_id` |
| writer CTA slots 0-4 (`:172-178`) | positional | named CTAs `q_out_h_tiles`, `q_out_w_tiles`, `q_out_HtWt`, `q_out_c`, `kv_out_c` |
| writer RTA slots 3-7 (`:328-332`) | positional | named RTAs `num_blocks`, `q_out_h_dim`, `q_out_tensor_tile_id`, `k_out_tensor_tile_id`, `v_out_tensor_tile_id` |
| compute CTA slot 0 (`:191, :201`) | positional `NHtWt` | named CTA `NHtWt` (stays a CTA, per KernelSpec) |
| `reader:28-33`, `writer:36-41` | `constexpr uint32_t cb_id_qv = 1; cb_id_k = 0/1/16` magic indices | `dfb::qv`; `dfb::k` under `TRANSPOSE_K_HEADS`, else the `qv` object itself (same-FIFO alias) |
| `transpose_wh.cpp:13-17,28,31` | `tt::CBIndex::c_0`, `c_16` literals | `dfb::in`, `dfb::out` in the fork |
| Sharded factory `:443` / slot 6 | `q_buffer_addr = input_tensor.buffer()->address()` | `TensorBinding{input_q, "input_q"}` on all four specs; kernel `TensorAccessor(tensor::input_q).get_bank_base_address()` (Case 2 bridge) |
| Sharded factory `:451` / slot 14 | `kv_buffer_addr` (KV base, or Q base when fused) | `TensorBinding{input_kv, "input_kv"}` on the `*_kv` specs when present; kernel `TensorAccessor(tensor::input_kv).get_bank_base_address()` under `READ_FROM_INPUT_TENSOR_KV`, else reuses the Q base |
| Sharded factory `:558-559` / CTA slots 0-1 | `q_output_cb_index`, `k/v_output_cb_index` | `DFBBinding`s `q_out` and `kv_out` |
| Sharded RTA slot 8 (`:445`) / `sharded:25` | `read_kv_heads` scalar | define `READ_KV_HEADS` on the `*_kv` specs (host-deterministic per core: `i < k_cores.num_cores()`); the RTA is dropped |
| Sharded RTA slots 0-7, 9-13, 15-17 | positional | named RTAs (see Planned Spec Shape) |
| Sharded RTA slots 18… (`:456-457`) / `sharded:28-29` | coordinate arrays read via `get_arg_addr(18)` / `get_arg_addr(18 + num_x)` | runtime varargs (`num_runtime_varargs = num_cores_x + num_cores_y`), read via `get_vararg(i)` / `get_vararg(num_x + i)` — retained vararg use, reported |
| Sharded override `:626-628` | three-arg `UpdateDynamicCircularBufferAddress` | `TensorArgument`s for `q`, `k`, `v` (borrowed DFBs re-resolve) |
| factory `:23-37`, `:50-54`, `:607-610` | positional kernel/CB/slot index bookkeeping | gone with named specs |
| `reader:46-47`, `writer:49-50` | `get_tile_size(cb_id)` (legacy `const`) | `dfb.get_tile_size()` member getter |

## Applied Patterns

- [Same-FIFO aliasing, path-dependent variant](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-same-fifo-aliasing-one-dfb-multiple-kernel-side-names):
  reader/writer `cb_k` is index 1 (the `qv` FIFO) under `!TRANSPOSE_K_HEADS` and index 0/16 otherwise →
  one `qv` binding, `#ifdef`-gated `k` binding, one `DataflowBuffer` object per FIFO (the K handle is a
  reference to the `qv` object on the non-transpose path).
- [Conditional / optional resource bindings](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-resource-bindings):
  `k_in`/`k_out` DFBs and the compute specs gated on `transpose_k_heads` (define `TRANSPOSE_K_HEADS`
  already exists); `input_kv` tensor gated on presence (define `READ_FROM_INPUT_TENSOR_KV`, already exists
  on Interleaved, newly emitted on Sharded); Sharded `kv_out` DFB + KV RTAs gated per node set via the
  promoted define `READ_KV_HEADS` (an RTA gate whose value is host-deterministic per core).
- [Two-toucher DFB → assign 1P+1C](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split):
  Sharded `q_out` — R and W instances both raw-write `get_write_ptr() + q_offset` on every node → R
  PRODUCER, W CONSUMER, no flag.
- [Sync-free and single-ended CBs → self-loop DFB](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb):
  Sharded `k_out` (R only: `reserve_back`/`push_back`, nothing consumes) and `v_out` (W only) — DM
  self-loops, legal on Gen1 (Quasar-uplift debt).
- [Anti-pattern guard: Demoting per-group CTA to RTA](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta):
  two compute `KernelSpec`s keep `NHtWt` a CTA.
- [Caution: Porting a shared kernel, rung 2](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel):
  fork `transpose_wh_metal2.cpp` beside the shared-pool original; pointer comment in the original.
- [Pass DFB handles directly to LLKs](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers):
  `compute_kernel_hw_startup(dfb::in, dfb::out)`, `transpose_init(dfb::in)`, `transpose_tile(dfb::in, …)`,
  `pack_tile(0, dfb::out)` in the fork.
- [Caution: Avoid varargs](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary):
  the Sharded coordinate tables are a genuine data-indexed collection (`noc_x[q_x]`, `q_x` advanced by the
  data walk) with a runtime count → retained as varargs; every other slot is named.

## Deferred / Flagged

- **Fork binding vocabulary vs the brief.** The brief asked for `dfb::cb_in` / `dfb::cb_out` in the new
  fork to match the landed `data_movement/transpose` fork. The recipe's self-audit forbids `cb` in any DFB
  name the port introduces (it names `dfb::cb_in0` as the costly miss), and the recipe outranks the brief,
  so the fork uses `dfb::in` / `dfb::out`. Reported as friction.
- **`READ_KV_HEADS` promotion is an RTA→define promotion**, one step beyond the catalog's "promote a CTA
  gate" wording. Judged inside whitelist rule 6's intent (a host-known condition that selects a binding
  moves to a define) because the value is a per-core host constant (`i < k_cores.num_cores()`), and the
  two-WorkUnit split is what makes a per-node define expressible. Reported as friction / recipe gap.
- The Sharded factory's per-program constants (`head_size`, `num_q_heads_per_core`, `num_x`, …) are
  per-node RTAs with one value on every node — CRTA candidates for a later cleanup, **not** converted here.
- **New finding during verification (not planning): the legacy sharded kernel reads one row past the end
  of its NoC coordinate table, in two places.** (1) After the last head on the last core of the last
  source row, the source-core advance wraps `q_y` / `kv_y` to `num_cores_y` and refreshes the
  coordinates from `in0_mcast_noc_y[q_y]`. (2) The writer-config instance on a core whose Q output holds
  one head reads zero Q heads, but the host hands it the coordinates the *next* instance would start from
  (one row past the table on the last core), and the pre-loop lookup reads them anyway. Both are raw L1
  pointer reads past the runtime-arg area, silent in legacy because neither value is consumed. The port's
  `get_vararg(num_x + y)` routes the same reads through `get_arg_addr`, which Watcher bounds-checks, so
  the `num_kv_heads == num_q_heads` sharded configs trip `DebugAssertRtaOutOfBounds`. Resolved the way
  the audit resolved the offset fold: a **separate pre-port kernel commit** (guard the advance with
  `q + 1 < num_q_heads` / `kv + 1 < num_kv_heads`, guard the pre-loop lookup with `num_q_heads > 0`;
  semantically no-ops), verified against the legacy host code before the port was re-applied on top.
  Recorded prominently in the report; the invoker may drop that commit and take `Interleaved` only.
- Every descriptor the legacy factory uses maps onto the audit's Appendix A scan; no other new findings.
