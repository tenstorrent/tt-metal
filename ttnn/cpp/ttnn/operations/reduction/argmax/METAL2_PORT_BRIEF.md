# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/reduction/argmax`

> **Scope: `ArgMaxDeviceOperation` only** — its `ArgMaxSingleCoreProgramFactory` and
> `ArgMaxMultiCoreProgramFactory`, and the four kernels they instantiate. This DeviceOperation
> cleared every gate.
>
> **`ArgMaxNCDeviceOperation` / `ArgMaxNCProgramFactory` is OUT OF SCOPE and must not be
> touched.** It is still on the legacy imperative `host_api.hpp` builder (`legacy device-op`
> concept), so it is blocked pending its `ProgramDescriptor` migration. Leave
> `argmax_nc_device_operation.*`, `argmax_nc_program_factory.cpp`, `kernels/reader_argmax_nc.cpp`,
> `kernels/writer_argmax_nc.cpp`, and `kernels/argmax_nc_compute.cpp` exactly as they are. The
> user-facing `ttnn::argmax` facade will dispatch to a ported device-op and a legacy one
> side by side; that is expected.
>
> Full record — including the gated half — is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ (both in-scope factories)
· Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `b73b958088a 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to
`ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — `create_descriptor()` returning a `ProgramDescriptor`,
  declared at `device/argmax_device_operation.hpp:17` (single-core) and `:22` (multi-core).
- **Op-owned tensors:** none.
- **Target concept:** **`ProgramSpecFactoryConcept`** (the base concept — no
  `override_runtime_arguments` exists on either in-scope factory, so the framework refreshes the
  tensor bindings on a cache hit and each factory writes one method).
- **`program_factory_t` variant:** `std::variant<ArgMaxSingleCoreProgramFactory,
  ArgMaxMultiCoreProgramFactory>` @ `device/argmax_device_operation.hpp:31`. Both factories are
  selected at runtime by `select_program_factory` (`argmax_device_operation.cpp:76`) via
  `uses_multicore_path`, so **both must be ported together** — the variant admits no half state.
- **No pybound `create_descriptor`** to delete: `argmax_nanobind.cpp` binds only the user-facing
  `ttnn::argmax`, so this port carries **no user-visible API change**.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a non-`none`
  `TensorParameter relaxation` (the sheet reads `none`) · `get_dynamic_runtime_args` (the
  deprecated hook — absent from both device-ops). This op *also* happens to carry no custom
  `compute_program_hash`, no `override_runtime_arguments`, and no pybound `create_descriptor` —
  but note none of those three would have gated anyway.

## Construct — to do

**Tensor bindings** (per binding — identical in both factories, and in all three of the
single-core factory's kernel configs):

- `input` — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`;
  the kernel builds `TensorAccessor(tensor::<name>)` instead.
- `output` — **Case 1** (via `TensorAccessor`) → same.

Today both arrive through the `Buffer*`-binding form in its MeshTensor spelling —
`reader_desc.emplace_runtime_args(core, {input, output})` at
`device/argmax_single_core_program_factory.cpp:205` and
`device/argmax_multi_core_program_factory.cpp:394` / `:424`. The framework auto-registers those as
`BufferBinding`s and patches them on cache hits, so this is **routine port work, not a correctness
hazard** — it is already correct today, and the typed binding supersedes the mechanism.

Both of these disappear on the host side when you bind the tensors:

- `TensorAccessorArgs(input).append_to(ctime_args)` / `TensorAccessorArgs(output).append_to(...)`
  @ `argmax_single_core_program_factory.cpp:182,183` and
  `argmax_multi_core_program_factory.cpp:373,374`.
- Kernel-side, the paired `TensorAccessorArgs<N>()` /
  `next_compile_time_args_offset()` plumbing: `reader_argmax_interleaved.cpp:41,42`;
  `reader_argmax_interleaved_multicore.cpp:299,300`; `reader_argmax_tile_layout.cpp:49,50`;
  `reader_argmax_tile_layout_h.cpp:44,45`. Note the two TILE readers hardcode the accessor's
  starting CTA offset as `num_c_time_args` (`= 13` @ `reader_argmax_tile_layout.cpp:40`, `= 11` @
  `reader_argmax_tile_layout_h.cpp:39`) — those constants go away with the accessor args, and the
  remaining CTAs become named.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — no accessor in the op passes a 3rd argument. All ten
construction sites are the 2-arg form, so there is nothing to drop and no
`dynamic_tensor_shape` to set.

**CB endpoints:** **self-loop on all six**, i.e. bind the single touching kernel as *both*
PRODUCER and CONSUMER. Every CB in scope has exactly **one** toucher per node, and that toucher
is **sync-free** — a raw `get_write_ptr()` peek with no FIFO ops anywhere in these four kernels
(no `reserve_back` / `push_back` / `wait_front` / `pop_front`). No 1P+1C assignment, no
multi-binding advanced option, no dead-CB drop, no conditional DFB.

| Factory | `(CB, config)` | Disposition |
|---|---|---|
| single-core | `(c_0 src, RM)` · `(c_1 dst, RM)` | self-loop |
| single-core | `(c_0 src, TILE-W)` · `(c_1 dst, TILE-W)` | self-loop |
| single-core | `(c_0 src, TILE-H)` · `(c_1 dst, TILE-H)` | self-loop |
| multi-core | `(c_0 src)` · `(c_1 dst)` · `(c_2 red_idxs)` · `(c_3 red_vals)` | self-loop |

**`get_dataformat(<cb_idx>)` → move onto the bound object.** Five sites query the buffer's data
format through the CB-index free function; the port moves them to the DFB member, which is
`constexpr` and so survives the NTTP uses:

| File | Line |
|---|---|
| `device/kernels/reader_argmax_interleaved.cpp` | `54` |
| `device/kernels/reader_argmax_interleaved_multicore.cpp` | `317`, `334` |
| `device/kernels/reader_argmax_tile_layout.cpp` | `63` |
| `device/kernels/reader_argmax_tile_layout_h.cpp` | `53` |

Each result must stay a compile-time constant — they feed `get_default_value<fmt>()`,
`compare_values<fmt>(…)`, `process_input_tile<T, fmt>(…)`,
`find_argmax_from_intermediate_outputs<n, fmt>(…)`, and a `static_assert` @
`reader_argmax_interleaved_multicore.cpp:341`. `DataflowBuffer::get_dataformat()` is `constexpr`
(`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:279`), so this works; the `CircularBuffer`
member is **not** `constexpr`, which is why this move belongs to the port and not to Device 2.0.

**Two DFB specs share one buffer index, over disjoint core ranges — reproduce this faithfully.**
The multi-core factory declares `c_0` **twice**:
`argmax_multi_core_program_factory.cpp:218` over `cores0` sized
`round_up_to_mul32(red_dim_units0 * input_unit_size)`, and `:231` over `cores1` sized from
`red_dim_units1` — **guarded by `if (num_cores1 > 0)`**. On the default (no `sub_core_grids`) path
those two sizes genuinely **differ**, because `split_work_to_cores` gives the two groups different
per-core block counts. Keep both specs, keep the size difference, and keep the second one
conditional.

**Two `KernelDescriptor`s share one `kernel_source` — this is the disjoint-node shape, not a
work-split.** `argmax_multi_core_program_factory.cpp:376` and `:408` both instantiate
`reader_argmax_interleaved_multicore.cpp`, but over `cores0` and `cores1` respectively — disjoint
sets, so each node sees exactly **one** instance. Do not read this as a dual-instance work-split
and do not reach for the multi-binding option; the per-node census really is 1. The second
`KernelSpec`, like the second `c_0` spec, is conditional on `num_cores1 > 0`.

## Watch for

- **CB endpoints (multi-binding):** none — no CB reaches two touchers on a node. Nothing to hunt.
- **The multi-core kernel requires `c_2` / `c_3` to land at the *same* L1 address on every node.**
  A worker computes the reducer's destination from its **own** base pointer —
  `red_idx_cb.get_write_ptr() + core_id * red_idx_size_per_core`, then writes to
  `{.noc_x = reduce_core_x, .noc_y = reduce_core_y, .addr = red_idx_cb_local_addr}`
  (`reader_argmax_interleaved_multicore.cpp:326-339`, `:416-427`, `:467-478`). Legacy honours
  that: `ProgramImpl::allocate_circular_buffers` assigns **one** address per CB object, taken as
  the max region-end across every core range it spans
  (`tt_metal/impl/program/program.cpp:1719-1751`), so a CB over `all_cores` is uniform even
  though the two `c_0` specs differ in size between the groups. **Confirm the DFB allocator keeps
  the same property** rather than assuming it — if a DFB were placed per-core-range, these
  cross-core writes would silently land at the wrong offset. Wrong numerics, no assertion. Worth
  an explicit numerical check on the multi-core path with an odd `red_dim` (so the two groups get
  different sizes) plus `sub_core_grids` unset.
- **Cross-op / shared kernels:** **no borrowed kernel files** — this op owns all four in-scope
  kernels, and no other *op* instantiates any of them. But `reader_argmax_interleaved.cpp` has one
  out-of-directory **consumer**: `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:126`
  file-path-instantiates it from a hand-built `ProgramDescriptor`, with the CTA layout (indices
  0-7 plus two `TensorAccessorArgs` blocks) and the RTA pair
  (`src_buffer->address()`, `dst_buffer->address()`) hardcoded at `:105-121`. Rewriting the kernel
  changes exactly that contract, and `generic_op` / `ProgramDescriptor` **cannot** supply Metal
  2.0 named bindings — so the test cannot simply be re-pointed. **No `_metal2` fork exists beside
  any argmax kernel today.** Confirm the chosen resolution with the user before you touch that
  kernel (audit *Questions* #1): either fork to `reader_argmax_interleaved_metal2.cpp` and leave
  the test on the legacy copy (then it is a **sunset** item, not authorization to convert in
  place), or rewrite in place and migrate the test off `generic_op`. Either way, run that gtest.
- **RTA varargs:** none — prefer named RTAs throughout. Every in-scope kernel reads a fixed set of
  args at constant indices: 2 for each single-core reader (`src_base_addr`, `dst_base_addr`), and
  7 for the multi-core reader (`src_base_addr`, `dst_base_addr`, `core_id`, `src_offset`,
  `red_dim_offset`, `src_read_size`, `red_dim_units_this_core` @
  `reader_argmax_interleaved_multicore.cpp:219-233`). No counted loop, no in-loop `arg_index++`,
  no data-selected index. **No CTA varargs either** — every `get_compile_time_arg_val` call in
  the op uses a literal index, so every CTA gets a name.
- **The four kernels are at *different* modernization levels — the work is asymmetric.**
  `reader_argmax_tile_layout.cpp:58,59` and `reader_argmax_tile_layout_h.cpp:51,54` already hold
  **`DataflowBuffer`**, so for those two the port is a binding-layer change (name the DFB, drop
  the CTA index) rather than an object rewrite. `reader_argmax_interleaved.cpp:49,50` and
  `reader_argmax_interleaved_multicore.cpp:311-314` are still on **`CircularBuffer`** and need the
  full CB→DFB swap — including the `use<CircularBuffer::AddrSelector::WRITE_PTR>(cb)` views
  (`reader_argmax_interleaved.cpp:87,100`; `reader_argmax_interleaved_multicore.cpp:417,423,451,
  468,474,499`) and dropping `#include "api/dataflow/circular_buffer.h"`
  (`reader_argmax_interleaved.cpp:7`, `reader_argmax_interleaved_multicore.cpp:8`). That include
  is legitimate today — the kernels really do use the wrapper — so it goes away as part of the
  CB→DFB swap, not as a stray cleanup.
- **`constexpr`-vs-`const` decides token form site by site.** Every DFB in these kernels comes
  from a `constexpr uint32_t` CTA feeding a **non**-`constexpr` wrapper object (e.g.
  `constexpr uint32_t src_dfb_idx = get_compile_time_arg_val(0);` then
  `DataflowBuffer src_dfb(src_dfb_idx);` @ `reader_argmax_tile_layout.cpp:18,58`), while the
  `get_dataformat` results must remain compile-time constants. Decide per site, not uniformly.
- **The dst DFB is reached by raw pointer, not by the object, in the two TILE readers.** They take
  `dst_dfb.get_write_ptr()` (`reader_argmax_tile_layout.cpp:66`,
  `reader_argmax_tile_layout_h.cpp:55`), hand it to `OutputContext`, and the actual NoC write goes
  through a `CoreLocalMem<uint32_t>` built on that address inside `write_to_output`
  (`kernels/argmax_tile_layout.hpp:329-345`). The binding is still required — the kernel touches
  the DFB — but you will not find the DFB object at the write site, so bind from the
  `get_write_ptr()` peek and leave the `CoreLocalMem` walk alone.
- **`c_1` in the multi-core factory looks per-core-conditional and is not.** Its comment says
  *"This CB is only used in the reduction core"* (`reader_argmax_interleaved_multicore.cpp:239`),
  but `dst_cb.get_write_ptr()` @ `:322` executes unconditionally on every node. It is live
  everywhere — do **not** drop it or make its DFB spec conditional.
- **`experimental/quasar/reduction/` exists. Stay out of it.** There is a quasar copy of the
  reduction family. It is a deliberately hacky shortcut port — not a precedent, not a naming
  source, not a fork to reuse, and its kernels carry idioms this recipe forbids. Do not read it,
  and do not let anything from it into the diff.
- **Baseline before the first kernel edit.** Kernels are JIT-compiled from the working tree at
  test time, so capture the pre-port numerical baseline (all four configs: RM last-dim,
  RM reduce-all, TILE dim=W, TILE dim=H — plus the multi-core path with and without
  `sub_core_grids`) **before** editing any kernel file.
- **Known latent issues in these files that are NOT yours to fix** (they route to the ops team;
  see the audit's *Misc anomalies*): dead CTA index 4 in the multi-core reader; dead CTA index 3
  in both TILE readers; the `(bool)`-narrowed `reduce_core_id` @
  `reader_argmax_interleaved_multicore.cpp:273`; the dummy `(0,0)` core-range CTAs for the
  single-group case. **Do not scoop these up** — but do notice that the two dead CTAs will show
  up as CTAs with no kernel-side reader when you convert to named args. Leave them present and
  named (or raise them), rather than silently deleting them as part of the port.
