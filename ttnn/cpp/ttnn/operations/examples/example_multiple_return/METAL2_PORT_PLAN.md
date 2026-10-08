# Port Plan — `examples/example_multiple_return`

Port plan for `examples/example_multiple_return`, ported from the ProgramDescriptor API (direct-descriptor shape) to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept`, in the **direct-descriptor shape**. `create_descriptor` is a static member of `ExampleMultipleReturnDeviceOperation` (`device/example_multiple_return_device_operation.hpp:72`). There is no `program_factory_t`, and the framework reaches the op through the `DirectDescriptorFactory` shim. The body is in `device/single_core_program_factory.cpp:12`. This forces `ttnn_factory.md` exception 3.
- Variants: single.
- Custom `compute_program_hash`: none. The op uses the default reflection hash over `operation_attributes_t` (`attribute`, `some_other_attribute`, `return_output1`, `return_output2`) plus the tensor args.

*(The Metal 2.0 factory concept comes from the audit. It is carried forward in [TTNN ProgramFactory](#ttnn-programfactory) below.)*

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` (borrowed) | `all_cores` = `{(0,0)}` | `TensorAccessorArgs(*src_buffer)` (slots 0..) | none | `src_buffer` (Buffer*), `num_tiles_per_core`, `num_tiles_written` | none | none | O2 (unset, DM default) | `ReaderConfigDescriptor{}` → RISCV_1 / NOC_0 / dedicated |
| writer | `examples/example_multiple_return/device/kernels/writer_multiple.cpp` (own) | `{(0,0)}` | `[0] output_cb_index (=2)`, then `TensorAccessorArgs(dst_buffer1)`, `TensorAccessorArgs(dst_buffer2)` (either may be `nullptr`) | none | `dst_buffer1` (Buffer* or null), `dst_buffer2` (Buffer* or null), `num_tiles_per_core`, `num_tiles_written` | none | none | O2 (unset, DM default) | `WriterConfigDescriptor{}` → RISCV_0 / NOC_1 / dedicated |
| compute | `eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` (borrowed) | `{(0,0)}` | none | none | `num_tiles_per_core` | none | none (`SFPU_OP_CHAIN_0` left undefined) | **O3** (unset on a `ComputeConfigDescriptor`) | `ComputeConfigDescriptor{.math_fidelity = HiFi4, .math_approx_mode = false}`, all other fields default (`fp32_dest_acc_en = false`, `dst_full_sync_en = false`, `bfp8_pack_precise = false`, `unpack_to_dest_mode` empty) |

`grep -n opt_level single_core_program_factory.cpp` prints nothing.

### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` | `2 * single_tile_size` | `{(0,0)}` | `datatype_to_dataformat_converter(input.dtype())` | `single_tile_size` | not set |
| `c_2` | `2 * single_tile_size_output` | `{(0,0)}` | `datatype_to_dataformat_converter(output_dtype)` (dtype of whichever output exists) | `single_tile_size_output` | not set |

No `.buffer`, no `address_offset`, no aliasing, no GlobalCircularBuffer.

Endpoint census (re-derived from the kernels):
- `c_0`: the reader FIFO-produces it (`reserve_back` / `push_back`, fork `:42,45`). Compute FIFO-consumes it (`wait_front` / `pop_front`, `eltwise_sfpu.cpp:31,45`). That is 1P+1C.
- `c_2`: compute FIFO-produces it (`reserve_back` / `push_back`, `:32,46`). The writer FIFO-consumes it (`wait_front` / `pop_front`, `writer_multiple.cpp:32,44`). That is 1P+1C.

### Semaphores
none

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `single_core_program_factory.cpp:71` (CTA) / `:117` (RTA) | `input_tensor` | reader RTA 0 (`src_buffer`) |
| `single_core_program_factory.cpp:82` (CTA) / `:118` (RTA) | `tensor_return_value[0]` (optional) | writer RTA 0 (`dst_buffer1`; null when absent) |
| `single_core_program_factory.cpp:83` (CTA) / `:118` (RTA) | `tensor_return_value[1]` (optional) | writer RTA 1 (`dst_buffer2`; null when absent) |

Device side: reader `TensorAccessor(src_args, src_addr)` (legacy reader `:30`). Writer `TensorAccessor(dst1_args, dst_addr1)` and `TensorAccessor(dst2_args, dst_addr2)` (`writer_multiple.cpp:27-28`). The writer also reads `dst_addr1` / `dst_addr2` as **presence flags** (`:34`, `:39`).

### Work split
- Driver: `split_work_to_cores({1, 1}, num_tiles)`.
- num_cores: 1.
- core_group_1: `{(0,0)}`, count_per_core: `num_tiles`.
- core_group_2: empty.

The per-core count reaches every kernel as an RTA, so there is no per-group CTA and no KernelSpec multiplicity.

### Shared kernels
- **Reader** `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` is *borrowed*. A `_metal2` fork exists beside it, so this is **rung 1 (reuse)**. Fork vocabulary: `dfb::in` (bound as PRODUCER), `tensor::src`, RTAs `args::num_pages` and `args::start_id`, and `#ifdef BACKWARDS`, which this op does not set. Page size comes from `dfb.get_entry_size()`, which equals the legacy `fifo_page_size` of `c_0` (`single_tile_size`). Legacy binders still on the original: `examples/example` (single and multi core), `experimental/transformer/nlp_create_qkv_heads_falcon7b`, `reduction/topk` (`topk_route_prep_program_factory.cpp`).
- **Compute** `eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` is *borrowed*. There is no `_metal2` sibling in this checkout, so this is **rung 2 (create the fork)**: create `eltwise_sfpu_metal2.cpp` beside it and add the pointer comment to the original. Vocabulary from the kernel itself: `dfb::in` (legacy `cb_input = c_0`), `dfb::out` (`cb_output = c_2`), and named arg `num_tiles`. `SFPU_OP_CHAIN_0` is kept, and this op leaves it undefined. Legacy binders: `eltwise/unary` (`unary_op_utils.cpp:1212` → `unary_program_factory.cpp`), `examples/example` (single and multi core), and `tests/ttnn/unit_tests/gtests/test_generic_op.cpp`. (The census also hits `tests/tt_metal/**` and `tests/tt_eager/ops/test_sfpu.cpp`, which carry their own same-named or path-built test copies. They are not op factories.)
  - *Concurrent fork:* the unmerged branch `origin/anasuya/metal2_port_eltwise_unary` (`229aab0b082`) creates the same `eltwise_sfpu_metal2.cpp` with the same names (`dfb::in`, `dfb::out`, `args::num_tiles`). This port takes that branch's fork and pointer comment **byte-for-byte** so the eventual merge is a clean identical add/add, per the shared-kernel Caution's "concurrent ports collide by design".
- **Writer** `device/kernels/writer_multiple.cpp` is op-owned. `grep -rl writer_multiple.cpp ttnn/cpp/ttnn/operations/` hits only this op's factory, so it is **not lent** and is converted in place.

### Flags
- `writer_multiple.cpp:9` includes `api/debug/dprint.h` without using it. Left as is (pre-existing, not port work).
- No unreferenced kernel files in the op directory.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`.
- **Custom `compute_program_hash`**: none (default reflection hash).
- **Implementation notes**: direct-descriptor shape, so per exception 3 add the nested `struct ExampleMultipleReturnProgramFactory { static ProgramArtifacts create_program_artifacts(...); };` and `using program_factory_t = std::variant<ExampleMultipleReturnProgramFactory>;`, replacing the static `create_descriptor`. The body stays in `single_core_program_factory.cpp`. There is no pybound `create_descriptor` to remove (`example_multiple_return_nanobind.cpp` binds only the composite).

## Planned Spec Shape

- **KernelSpecs** (one per legacy `KernelDescriptor`):
  - `reader`: source `reader_unary_interleaved_start_id_metal2.cpp`. DFB binding `src0` → accessor `in`, PRODUCER. Tensor binding `src` → accessor `src`. RTAs `num_pages`, `start_id`. `create_reader_datamovement_config()`. Default O2.
  - `writer`: source `writer_multiple.cpp` (converted in place). DFB binding `output` → accessor `out`, CONSUMER. Tensor bindings `dst1` → accessor `dst1` **only when output 0 is present** (`return_output1`), and `dst2` → accessor `dst2` **only when output 1 is present** (`return_output2`). Defines `RETURN_OUTPUT1` / `RETURN_OUTPUT2` on the same conditions. RTAs `num_tiles`, `start_id`. `create_writer_datamovement_config()`. Default O2.
  - `compute`: source `eltwise_sfpu_metal2.cpp` (new fork). DFB bindings `src0` → accessor `in` CONSUMER and `output` → accessor `out` PRODUCER. RTA `num_tiles`. `ComputeHardwareConfig{.fpu_math_fidelity = HiFi4, .sfpu_precision_mode = Precise}` (Style B; other fields default, matching the legacy defaults). **`opt_level = O3`** set explicitly.
- **DataflowBufferSpecs**:
  - `src0` (legacy `c_0`, name from the legacy `src0_cb_index`): `entry_size = single_tile_size`, `num_entries = 2`, `data_format_metadata = dfb_data_format`.
  - `output` (legacy `c_2`, name from the legacy `output_cb_index`): `entry_size = single_tile_size_output`, `num_entries = 2`, `data_format_metadata = dfb_data_format_output`.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `src` (always), `dst1` (only when `tensor_return_value[0]` is present), `dst2` (only when `tensor_return_value[1]` is present). Names follow the kernels' own vocabulary (`src_addr`, `dst_addr1`, `dst_addr2`).
- **WorkUnitSpecs**: one, `{reader, writer, compute}` on `all_cores`.
- **Op-owned tensors**: none.

## Preserved Multiplicity

none — no work-split multiplicity in legacy (one core, one descriptor per kernel).

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory `:71`, reader CTAs | `TensorAccessorArgs(*src_buffer).append_to(...)` | `TensorBinding{src → src}` (fork already uses `TensorAccessor(tensor::src)`) |
| factory `:117`, reader RTA 0 | `src_buffer` (Buffer*) | `TensorBinding{src → src}` + `tensor_args[src]` |
| factory `:117`, reader RTAs 1,2 | positional `num_tiles_per_core`, `num_tiles_written` | named RTAs `num_pages`, `start_id` (the fork's names) |
| factory `:81`, writer CTA 0 | `output_cb_index` (= `c_2`) | `DFBBinding{output → out, CONSUMER}` |
| factory `:82-83`, writer CTAs | `TensorAccessorArgs(dst_buffer1/2).append_to(...)` (`nullptr` when absent) | conditional `TensorBinding{dst1 → dst1}` / `{dst2 → dst2}` |
| factory `:118`, writer RTAs 0,1 | `dst_buffer1`, `dst_buffer2` (Buffer* or null; also the kernel's presence flag) | conditional `TensorBinding` + `RETURN_OUTPUT1` / `RETURN_OUTPUT2` defines; kernel `#ifdef`s replace `if (dst_addrN != 0)` |
| factory `:118`, writer RTAs 2,3 | positional `num_tiles_per_core`, `num_tiles_written` | named RTAs `num_tiles`, `start_id` |
| factory `:122`, compute RTA 0 | positional `num_tiles_per_core` | named RTA `num_tiles` |
| `eltwise_sfpu.cpp:20-21` | hardcoded `tt::CBIndex::c_0` / `c_2` | `dfb::in` / `dfb::out` (fork) |
| legacy reader `:21` | hardcoded `cb_id_in0 = 0` | `dfb::in` (fork, already done) |

## Applied Patterns

- `port_patterns.md` — Conditional / optional resource bindings: `tensor::dst1` / `tensor::dst2` on the writer are gated by `RETURN_OUTPUT1` / `RETURN_OUTPUT2`. The presence flags come from `operation_attributes_t`, which the default hash covers, so each presence combination is already its own cached program.
- `port_patterns.md` — Caution: Porting a shared kernel: rung 1 for the reader, rung 2 for compute.
- `port_patterns.md` — Pass DFB handles directly to LLKs: `compute_kernel_hw_startup`, `copy_init`, `copy_tile`, `pack_tile` in the compute fork.

## Deferred / Flagged

- The legacy presence test is a runtime `dst_addrN != 0`. The port turns it into a compile-time gate keyed on `return_outputN`. These agree as long as a present output never gets buffer address 0. The allocator places buffers above the reserved bank base, so a live output never sits at 0 in practice, and the gate preserves behaviour. Recorded here because it is the one place the port changes *how* a decision is made.
