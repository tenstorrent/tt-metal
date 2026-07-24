# Port Plan — integral_image (`experimental::reduction::integral_image`)

Port plan for `integral_image`, ported from the `descriptor` (ProgramDescriptorFactoryConcept, direct
`create_descriptor`) API to Metal 2.0 `MetalV2FactoryConcept`.
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — direct `create_descriptor` on the device-op struct
  (`HasDirectDescriptor`; no `program_factory_t`), `intimg_program_factory.cpp:67`.
- Variants: single (one config: interleaved, fixed 2×4 core grid `CORES_X=2 × CORES_Y=4`).
- Custom `compute_program_hash`: none — already default reflection-based hash (audit confirmed).

*(Target concept `MetalV2FactoryConcept`, inherited from the audit; see [TTNN ProgramFactory](#ttnn-programfactory).)*

### Kernels (all three owned by the op; all bound by the single factory; they flip together)
| unique_id | source | core_ranges | CTAs (positional) | RTAs | config |
|---|---|---|---|---|---|
| reader | `kernels/intimg_reader.cpp` | 2×4 `{{0,0},{1,3}}` | 18 CTAs (0-8 CB idx, 9-17 scalars) + `TensorAccessorArgs(src)` @18 + `TensorAccessorArgs(dst)` @next | `{src_buffer}` @0 (per core, all cores) | `ReaderConfigDescriptor{}` (reader default: RISCV_1/NOC_0) |
| compute | `kernels/intimg_compute.cpp` | 2×4 | 18 CTAs (0-8 CB idx, 9-17 scalars) | none | `ComputeConfigDescriptor{.math_fidelity=HiFi4, .fp32_dest_acc_en=<fp32>, .math_approx_mode=false}` |
| writer | `kernels/intimg_writer.cpp` | 2×4 | same 18 CTAs + both `TensorAccessorArgs` blocks | `{dst_buffer}` @0 (per core, all cores) | `WriterConfigDescriptor{}` (writer default: RISCV_0/NOC_1) |

Positional CTA slots 0-17 (compute list, `intimg_program_factory.cpp:113-131`):
0 START, 1 INPUT, 2 ACC, 3 CUMSUM_STAGE_0, 4 CUMSUM_STAGE_1, 5 CUMSUM_STAGE_2, 6 OUTPUT,
7 AXIS_2_BUFFER, 8 AXIS_3_BUFFER (all CB indices) — 9 tile_height, 10 tile_width, 11 block_depth,
12 num_channels(=`input_shape[3]`), 13 input_height(=`[2]`), 14 input_depth(=`[1]`),
15 num_batches(=`[0]`), 16 cores_x, 17 cores_y. Dataflow kernels append the two `TensorAccessorArgs` blocks.

In-directory shared headers (owned by the op, edited freely — all three kernels flip together, so no fork needed):
`kernels/common.hpp` (RAII `ReadCBGuard`/`WriteCBGuard`, `std_type_t`, tile-index math), `kernels/common_dataflow.hpp`
(`write_to_dram`/`load_from_dram`, the reader/writer `IntImgCTAs` + `get_ctas()`).

### CBs (9; `make_cb`, `intimg_program_factory.cpp:35-49,96-110`)
`entry_size = tt::tile_size(dataformat(dtype))`; `data_format = dataformat(dtype)`; no `.tile` set (default 32×32).
| CB (idx) | num_entries | data_format | tile |
|---|---|---|---|
| START (0) | 2 | dtype fmt | default |
| INPUT (1) | 48 (BLOCK_DEPTH) | dtype fmt | default |
| ACC (2) | 2 | dtype fmt | default |
| CUMSUM_STAGE_0 (3) | 48 | dtype fmt | default |
| CUMSUM_STAGE_1 (4) | 48 | dtype fmt | default |
| CUMSUM_STAGE_2 (5) | 48 | dtype fmt | default |
| OUTPUT (6) | 48 | dtype fmt | default |
| AXIS_2_BUFFER (7) | 2 | dtype fmt | default |
| AXIS_3_BUFFER (8) | 48 | dtype fmt | default |

No GlobalCircularBuffer. `AXIS_3_BUFFER_1` (`:111`) is a dead comment, not a CB.

### Semaphores
None — the op uses no semaphores.

### Tensor accessors
- input tensor — reader `TensorAccessor(ctas.input_args, input_base_addr)` (`intimg_reader.cpp:53`); base via `{src_buffer}` RTA @0. **Case 1**.
- output tensor — writer `TensorAccessor(ctas.output_args, output_base_addr)` (`intimg_writer.cpp:64`), used for BOTH output writes (`write_to_dram`) AND the cross-row readback (`receive_upper_block`→`load_from_dram`, `:31`); base via `{dst_buffer}` RTA @0. **Case 1**, one binding covers both directions.

### Work split
None. All three kernels are placed over the same fixed 2×4 `CoreRangeSet` (`intimg_program_factory.cpp:92`); no `split_work_to_cores`; no per-group CTA multiplicity. Every node runs reader+compute+writer.

### Cross-op kernels
None — op owns all three kernels; every `#include` resolves to `tt_metal/*` or an in-directory header.

### Runtime kernel-source selection
None — one fixed source per kernel descriptor.

## Plan the spec

### TTNN ProgramFactory
- **Concept (inherited from audit):** `MetalV2FactoryConcept`.
- **Custom `compute_program_hash`:** none.
- **Implementation notes:** the op is `HasDirectDescriptor` today (no `program_factory_t`). The Metal 2.0 adapter's
  `DirectDescriptorFactory` fallback wraps `create_descriptor`, not `create_program_artifacts`, so a bare
  `create_program_artifacts` on the op struct would NOT be picked up. The port therefore introduces a nested
  `struct ProgramFactory { static ProgramArtifacts create_program_artifacts(...); }` and
  `using program_factory_t = std::variant<ProgramFactory>;` (single-alternative → framework auto-selects it,
  no `select_program_factory` needed), and deletes `create_descriptor`.

### Planned Spec Shape (1:1 with legacy)
- **KernelSpecs:** 3 — READER (DM), COMPUTE, WRITER (DM). One per legacy `KernelDescriptor`.
- **DataflowBufferSpecs:** 9, one per legacy CB (names above), `data_format_metadata = dtype fmt`, `tile_format_metadata` unset.
- **SemaphoreSpecs:** none.
- **TensorParameters:** 2 — `input` (bound by reader), `output` (bound by writer). `.spec` from each tensor's `tensor_spec()`.
- **WorkUnitSpecs:** 1 — {READER, COMPUTE, WRITER} on `NodeRange{{0,0},{CORES_X-1,CORES_Y-1}}`.
- **Op-owned tensors:** none.

### Preserved Multiplicity
None — no work-split multiplicity in legacy (single fixed grid, one KernelDescriptor per source).

### Dropped Plumbing
- **Buffer-address RTAs → `TensorBinding`:** reader `{src_buffer}` RTA @0 (`intimg_program_factory.cpp:145`) →
  `TensorParameter input` + reader `TensorBinding{input,"input"}`; kernel `get_arg_val<uint32_t>(0)`
  (`intimg_reader.cpp:50`) dropped. Writer `{dst_buffer}` RTA @0 (`:169`) → `TensorParameter output` + writer
  `TensorBinding{output,"output"}`; `intimg_writer.cpp:62` dropped. Both kernels end up with **zero** RTAs.
- **`TensorAccessorArgs(...).append_to(...)`:** `intimg_program_factory.cpp:133,134` dropped; kernel
  `TensorAccessorArgs<18>()`/`TensorAccessorArgs<...offset>()` (`common_dataflow.hpp:70-71`) dropped;
  `TensorAccessor(ctas.input_args, addr)` → `TensorAccessor(tensor::input)`, `...output_args...` → `tensor::output`.
- **Magic CB indices in CTAs → `DFBBinding`:** compute CTA slots 0-8 (and the dataflow copies) → 9 `DFBBinding`s
  across the three kernels (endpoint census below). Kernel `ctas.*_cb` field reads → `dfb::*` tokens.
- **Positional CTAs → named CTAs:** the 9 scalar CTAs (slots 9-17) → named `compile_time_args`
  `{tile_height, tile_width, block_depth, num_channels, input_height, input_depth, num_batches, cores_x, cores_y}`,
  emitted on all three kernels; kernels read via `get_arg(args::name)`.
- **Page-size 3rd accessor arg:** none (both accessors 2-arg).
- **Semaphore-ID RTAs:** none.

### CB endpoint dispositions (re-derived from the kernel-touch census)
Every node runs reader+compute+writer, so the census is program-wide:
- START (0): reader PRODUCER (zero-fill), compute CONSUMER → **1P+1C**.
- INPUT (1): reader PRODUCER (`load_from_dram`), compute CONSUMER → **1P+1C**.
- ACC (2): compute only (produce+consume) → **self-loop** (compute PRODUCER + CONSUMER).
- CUMSUM_STAGE_0/1/2 (3/4/5): compute only → **self-loop**.
- OUTPUT (6): compute PRODUCER, writer CONSUMER (`write_to_dram`) → **1P+1C**.
- AXIS_2_BUFFER (7): compute only → **self-loop**.
- AXIS_3_BUFFER (8): writer PRODUCER (readback `load_from_dram`), compute CONSUMER → **1P+1C**.

Matches the brief. No dead CBs, no multi-binding flag, no aliasing, no borrowed DFBs.

### Applied Patterns
- **Self-loop DFB binding:** ACC, CUMSUM_STAGE_0/1/2, AXIS_2_BUFFER on the compute KernelSpec (PRODUCER **and** CONSUMER).
- **Multi-kernel-same-tensor:** the output `TensorParameter` is bound only by the writer (compute never touches
  output-tensor memory), so a single writer `TensorBinding` covers both the output write and the cross-row readback.
- **`hw_config` — DM defaults:** reader/writer resolved triples are the reader/writer defaults →
  `ttnn::create_reader_datamovement_config(arch)` / `create_writer_datamovement_config(arch)`.
- **`hw_config` — compute Style B (Metal `ComputeConfig` set directly, literal values):** build `ComputeGen1Config`
  directly — `fpu_math_fidelity=HiFi4`, `sfpu_precision_mode=Precise` (legacy `math_approx_mode=false`),
  `enable_32_bit_dest=fp32_dest_acc_en`; `double_buffer_dest`/`bfp_pack_precision_mode` left at defaults (legacy unset).
- **`unpack_modes` required-entry rule:** when `fp32_dest_acc_en` (Float32 input) the compute kernel consumes
  Float32 DFBs with `enable_32_bit_dest=true`, so the validator requires an explicit entry for each **consumed**
  Float32 DFB. Legacy left `unpack_to_dest_mode` default (=`UnpackToSrc`), so add `UnpackToSrc` for the 8 consumed
  DFBs {START, INPUT, ACC, CUMSUM_STAGE_0/1/2, AXIS_2_BUFFER, AXIS_3_BUFFER} (OUTPUT is producer-only). For bf16
  input `enable_32_bit_dest=false` and no entries are required (unpack_modes empty).

### Deferred / Flagged
- **Reader `get_dataformat` metadata move (whitelist rule 7 friction):** the brief asks for
  `dfb::input.get_dataformat()`, but `get_dataformat()` is a member of `DataflowBuffer`, not of the `dfb::` token
  (`DFBAccessor`), and the reader uses it in a `constexpr` template-argument (`std_type_t<...>`); `DataflowBuffer`'s
  constructor is not `constexpr`, so no object-getter spelling works in that context. Resolved by the sanctioned
  rule-2 shim — pass `dfb::input` (implicit→`uint32_t`) to the still-`constexpr` free function
  `get_dataformat(...)` (`dataflow_api.h:300`, reads the same `unpack_src_format[]` slot the getter would). See
  the port report.
- Misc pre-existing anomalies (reader `tile_width` vs writer/compute `tile_height` for the same quantity; dead
  `num_batches` loops; the dead `AXIS_3_BUFFER_1` comment) are out of scope — routed to the port report, not touched.
