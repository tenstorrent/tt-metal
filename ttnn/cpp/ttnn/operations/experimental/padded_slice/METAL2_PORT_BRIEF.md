# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/padded_slice`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Precondition:** the gates clear on `main` **plus pre-port fix `f574e0b3430`** (PR #62499,
merged into `main` as `0226df45c4a` on 2026-10-26), which `main` now carries. Do not port
from a tree without it: the RM reader's base address is a `base + offset` fold there and its non-aligned
path is Device 1.0. Do not re-do or absorb the fix in the port.

**Recipe docs:** `aab653ac9b9 2026-10-27 docs(metal_2.0): import the op-porting recipe docs for the padded_slice audit` *(carry this line into the port report's Provenance section; it imports `akertesz/op-porting-recipe` @ `4bd4bf42bfe`)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). Carry them forward:

- **Current concept:** `descriptor` (both factories; PR #57550). Factories live in the `program_factory_t`
  variant (`device/padded_slice_device_operation.hpp:25`) — no direct-descriptor conversion needed.
- **Op-owned tensors:** none.
- **Target concept:**
  - `PaddedSliceRMProgramFactory` → **`CustomProgramSpecFactoryConcept`** — ported-from
    `override_runtime_arguments` at `device/padded_slice_rm_program_factory.cpp:411` (declared `.hpp:20`).
    It refreshes exactly two things per dispatch: the output CB's backing buffer (`:88-89`) and the reader's
    base-address RTA (`:434`). Translate it to a `ProgramRunArgs` carrying `TensorArgument`s for `INPUT` and
    `OUTPUT`, and nothing else. Its guards (`:419-423`) move with it.
  - `PaddedSliceTileProgramFactory` → **`ProgramSpecFactoryConcept`** (no override; the framework refreshes
    the bindings).
- **Custom `compute_program_hash`:** none (default reflection hash).
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a `TensorParameter relaxation`
  that is neither `none` nor an analysis pointer · `get_dynamic_runtime_args`. A pybound
  `create_descriptor` is also absent (nothing to delete in `padded_slice_nanobind.cpp`).

## Construct — to do

**Tensor bindings** (per binding):

- RM `input` — **Case 1** (via `TensorAccessor`, `kernels/dataflow/padded_slice_reader_rm_interleaved_start_id.cpp:45`)
  → `TensorParameter` / `TensorBinding`; kernel uses `TensorAccessor(tensor::src)`. Drops: reader RTA slot 0
  (`padded_slice_rm_program_factory.cpp:226`, `:434`) and the `TensorAccessorArgs` CTA tail (`:361`,
  kernel `TensorAccessorArgs<3>()` at `:33`).
- RM `output` — **clean** (borrowed-memory CB, `padded_slice_rm_program_factory.cpp:339`) →
  `DataflowBufferSpec::borrowed_from = OUTPUT`; the override's `UpdateDynamicCircularBufferAddress` (`:88-89`)
  becomes the `OUTPUT` `TensorArgument`.
- Tile `input` — **Case 1** (`Buffer*` delivered at `padded_slice_tile_program_factory.cpp:44`, consumed by
  `TensorAccessor(src_args, src_addr)` at `kernels/dataflow/padded_slice_reader_tiled_interleaved_start_id.cpp:33`)
  → `TensorParameter` / `TensorBinding`. Drops: the `Buffer*` RTA helper (`:38-50`) and the CTA tail (`:518`,
  kernel `TensorAccessorArgs<1>()` at `:32`).
- Tile `output` — **clean** (borrowed-memory CB, `padded_slice_tile_program_factory.cpp:470`) →
  `borrowed_from = OUTPUT`.

**TensorParameter relaxation:** `none` (both factories). Keep strict.

**TensorAccessor 3rd arg:** drop `padded_stick_size` from the RM reader's accessor at `:45` (Class 2 —
interleaved, correct magnitude, and the default hash pins the width per cache entry). The host emits it as
reader RTA slot 1 (`padded_slice_rm_program_factory.cpp:115`, common args) solely for the accessor; that
emission drops with it. The Tile reader passes no 3rd argument.

**CB endpoints:**

- RM `output` (CB 0, `:339`) — **1P+1C** in both writer configs: reader PRODUCER; writer CONSUMER, whether
  it is the op's pad writer (`writer_unary_sharded_padded_rm.cpp`, raw `get_read_ptr` + NoC self-reads, no
  FIFO ops) or the borrowed plain writer (`writer_unary_sharded_metal2.cpp`, `wait_front`/`pop_front`). With
  the pad writer the FIFO is fill-only (the output is resident); do not add a pop.
- RM `temp_pad` (CB 1, `:344`) — **self-loop** on the pad writer; **conditional DFB** — only when
  `pad_output_row` (`:341-346`). Bind it only on the pad-writer `KernelSpec`, which is the only one selected
  under that condition, so no `#ifdef` is needed in the kernel.
- RM `non_aligned` scratch (CB 2, or 1 when there is no `temp_pad`; `:353`) — **self-loop** on the reader;
  **conditional DFB** — only when `is_non_aligned` (`:349-355`). The kernel already gates every use under
  `if constexpr (is_non_aligned)` (`reader:90`); the binding is conditional on the host and the constexpr
  alias is `#ifdef`-gated per *Pattern: Conditional / optional resource bindings*. The index-reuse trick at
  `:346` ("so that CBs are continuous") simply disappears with named DFBs.
- Tile `input` (CB 0, `:450`) — legal 1:1: reader PRODUCER, compute CONSUMER.
- Tile `untilized` (CB 1, `:457`) — legal 1:1: compute PRODUCER, writer CONSUMER. The compute kernel names it
  twice (`out_cb_id0`, `out_cb_id1`, CTAs 1–2 at `:490-496`) — one DFB, two kernel-side names (*Pattern:
  Same-FIFO aliasing*).
- Tile `output` (CB 2, `:470`) — **self-loop** on the writer (raw NoC self-reads into it, no FIFO ops).
- Tile `padding` (CB 3, `:478`) — **self-loop** on the writer.

No multi-binding flag anywhere; no dead CB.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:**
  - `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` (RM,
    `!pad_output_row`) → **`_metal2` fork already exists** at
    `…/sharded/device/kernels/dataflow/writer_unary_sharded_metal2.cpp` — bind it, don't re-fork. Its
    vocabulary is your constraint: `dfb::out` bound CONSUMER, `args::num_units` (RTA). Other legacy binders
    (**sunset list, not authorization to convert in place**; issue #52228):
    `data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.cpp`,
    `data_movement/untilize/device/factories/untilize_multi_core_input_and_output_nd_shard_type_and_shard_spec_identical_program_factory.cpp`.
  - `ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/compute/pack_untilize.cpp` (Tile compute)
    → **no fork yet — this port creates it** beside the original as `pack_untilize_metal2.cpp`, plus the
    pointer comment in the original (rung 2; the only sanctioned write outside this op's directory). Name
    the bindings for the kernel's own vocabulary (input / output DFB, `total_blocks`), not for padded_slice.
    Other legacy binder: `sliding_window/halo/device/untilize_with_halo_program_factory.cpp` (**sunset
    list**). The kernel's donor calls (`compute_kernel_lib::untilize_init` / `untilize` / `untilize_uninit`,
    `compute_kernel_hw_startup`) take CB ids as `uint32_t` / template parameters — pass `dfb::name` straight
    in. Note it includes `untilize_helpers.hpp` twice (`:7`, `:12`); copy it as is, don't tidy.
  - **Lent host helpers — leave them in place.** `device/slice_cb_descriptor.hpp` (a `CBDescriptor` builder)
    and `device/padded_slice_utils.hpp` are included by the three `experimental/slice_write` factories. Your
    factories stop using `slice_cb_descriptor.hpp`, but the file stays: the "no `CBDescriptor` in the op
    directory" sweep will hit it — record it as a lent header, not a leftover. `padded_slice_utils.hpp` is
    plain host geometry and stays in use.
- **RTA varargs:** three `num_dims`-long blocks in each DM kernel, read through `get_arg_addr` and indexed in
  `j < num_dims` loops — genuine varargs:
  - RM reader `:24` (`get_arg_addr(11)`): `num_unpadded_sticks`, `num_padded_sticks`, `id_per_dim`;
  - Tile reader `:21` (`get_arg_addr(6)`): the same three;
  - Tile writer `:34-36` (`get_arg_addr(5)`, `5 + num_dims`, `5 + 2*num_dims`): `output_coord`,
    `output_start_in_input`, `output_end` (`num_dims` is a CTA there — still a vararg block).
  Every other RTA is a distinct field → name it.
- **Mutable arg memory (all three DM kernels).** The kernels write back into their vararg blocks:
  `id_per_dim[j]++` (RM reader `:140`, `:212`; Tile reader `:106-114`) and `output_coord[…] +=` (Tile
  writer `:196-202`). Metal 2.0 varargs are read-only. `data_movement/slice`'s ported RM reader seeds a
  `Scratchpad<uint32_t>` from the vararg block and advances that
  (`slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:47-53` on `main`; `ScratchpadSpec` +
  `scratchpad_bindings` in `slice_program_factory_rm.cpp`). Do the same for each of the three.
- **`experimental::CB` is `CircularBuffer`.** All four op kernels take the wrapper through
  `pool/device/kernels/experimental_device_api.hpp` (`:24`); rule 1 turns it into `DataflowBuffer`. Keep the
  include — the writers and the RM reader use its `experimental::local_addr` with `UnicastEndpoint` for
  L1-to-L1 NoC copies, which stay as they are (transfers are not rewritten in a port).
- **Compute `hw_config` / `opt_level` (Tile):** Style B — `ComputeConfigDescriptor{.fp32_dest_acc_en = false,
  .dst_full_sync_en = false}` (`padded_slice_tile_program_factory.cpp:506`) → `enable_32_bit_dest = false`,
  `double_buffer_dest = true`, fidelity at default. **No `opt_level` is set anywhere** (grep returns nothing
  on either factory), so the compute spec resolves to O3 and must say `O3` explicitly; both DM specs resolve
  to O2 (defaults). DM configs are `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` → the
  arch-agnostic reader / writer helpers.
- **`constexpr` metadata:** Tile writer `:38` `constexpr uint32_t tile_size = get_tile_size(cb_out_id)` keeps
  the free-function form on the token (`get_tile_size(dfb::out)`); Tile reader `:31` is a runtime `const`
  → DFB getter. RM reader `:100` `get_local_cb_interface(cb).fifo_page_size` is a sanctioned free function;
  use the DFB's page-size getter from the whitelist if one is listed, else keep the token form.
- **Runtime kernel-source selection (RM writer):** `pad_output_row` picks the op's pad writer or the borrowed
  plain writer (`padded_slice_rm_program_factory.cpp:376-385`). Both entry points convert with the factory;
  the writer `KernelSpec`'s bindings differ by source (`dfb::out` only for the plain fork; `dfb::out` +
  `dfb::temp_pad` + the pad CTAs for the pad writer).
- **Tests (no-regression baseline, confirm with the invoker):**
  `tests/ttnn/unit_tests/operations/data_movement/test_padded_slice.py` (both factories; cache-hit test for
  each; `-k 'not quasar'` on Gen1), nightly
  `tests/ttnn/nightly/unit_tests/operations/data_movement/test_slice_for_conv.py` (height / block / width
  sharded, RM and TILE, `padded_slice` at `:381`, `:478`, `:567`). Production callers:
  `sliding_window/op_slicing/op_slicing.cpp:484`, `conv/conv_transpose2d/conv_transpose2d.cpp`,
  `experimental/ccl/strided_reduce_scatter_async`; `models/demos/stable_diffusion_xl_base/tests/test_sdxl_perf.py`.
