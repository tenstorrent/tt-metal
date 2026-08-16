# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/uniform`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `086a669ff5e 2026-08-15 docs(metal_2.0): two porter-facing gaps a blind cold read turned up` *(carry this line into the port report's Provenance section)*

> **Before you start — confirm the hold has lifted.** The readiness sheet's `Is able to port?` cell for `uniform`
> currently reads `no`. That is **not** an op defect: it is a deliberate family-wide hold on ops targeting
> `CustomProgramSpecFactoryConcept`, whose recipe support is newly added and still being tested. This audit was run
> with the gate treated as `yes` on the recipe maintainer's explicit instruction. **Check with the maintainer that the
> hold is lifted before porting.** Everything below is unaffected by the hold — it is a verdict on the code, and the
> code clears every gate.

## Scope

One device operation, one factory. `create_descriptor` and `override_runtime_arguments` both live on the device-op
itself — there is no separate `ProgramFactory` class.

- `device/uniform_program_factory.cpp` — `create_descriptor` @ `:107`, `override_runtime_arguments` @ `:213`
- `device/uniform_device_operation.hpp` — declarations, and the backdoor hash @ `:28-29`
- `device/kernels/writer_uniform.cpp` (writer / DM), `device/kernels/compute_uniform.cpp` (compute)

`uniform` is in-place: `create_output_tensors` returns the input tensor (`device/uniform_device_operation.cpp:31-35`),
so the op has exactly **one** tensor.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to
`CustomProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — `create_descriptor()` returning a `ProgramDescriptor` @
  `device/uniform_program_factory.cpp:107`
- **Op-owned tensors:** none
- **Target concept:** `CustomProgramSpecFactoryConcept` — selected by `Override runtime args method? == yes`.
  `UniformDeviceOperation::override_runtime_arguments` @ `device/uniform_program_factory.cpp:213-247` (declared
  `device/uniform_device_operation.hpp:51-56`) is **translated** into a method returning a `ProgramRunArgs`, not
  deleted.
- **Backdoor custom hash — present, and load-bearing. Leave it exactly as it is.**
  `attribute_names` / `attribute_values` @ `device/uniform_device_operation.hpp:28-29` list only `memory_config` and
  `compute_kernel_config`, deliberately excluding `from`, `to`, and `seed` from the program hash. That exclusion is
  safe *only because* `override_runtime_arguments` re-applies all three on every cache hit. The two are a matched
  pair — if your translated override stops writing any of those three slots, the hash exclusion becomes a silent
  correctness bug. Verify the translated method still writes all of `seed`, `f2u_from`, `f2u_to`.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): a non-`none` `TensorParameter relaxation` ·
  `get_dynamic_runtime_args` (removed from this op by #50338, 2026-07-30). A custom hash, an
  `override_runtime_arguments`, and a pybound `create_descriptor` are **not** in this list: none of them gate, and any
  may be present on a cleared op. Here: no `compute_program_hash`, no pybound `create_descriptor`, and an
  `override_runtime_arguments` that is the target-concept signal above.

## Construct — to do

**Read the shared-kernel item under *Watch for* first.** Neither kernel may be converted in place.

**Tensor bindings** (per binding):

- `output` — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`; kernel uses
  `TensorAccessor(tensor::<name>)`.

  What disappears when you do:
  - Host, cache miss: the `Buffer*` inside
    `writer_desc.emplace_runtime_args(core, {output.buffer(), tile_offset, units_per_core})` @
    `device/uniform_program_factory.cpp:204` (the framework's `BufferBinding` interim hack — superseded by the typed
    binding).
  - Host, cache hit: `writer_args[0] = out_addr` @ `device/uniform_program_factory.cpp:243`, and the
    `output.buffer()->address()` that feeds it @ `:228`.
  - Host, CTAs: `TensorAccessorArgs(output.buffer()).append_to(writer_ct_args)` @ `:163`.
  - Kernel: `uint32_t dst_addr = get_arg_val<uint32_t>(0)` @ `device/kernels/writer_uniform.cpp:19`,
    `constexpr auto dst_args = TensorAccessorArgs<2>()` @ `:17`, and the two-arg
    `TensorAccessor(dst_args, dst_addr)` @ `:24`.

  The writer's remaining two RTAs (`start_id`, `num_tiles`) shift down by one slot — they stay as named args.

**TensorParameter relaxation:** none — the only value that reaches a brief.

**TensorAccessor 3rd arg:** none. The op's single accessor is already two-arg
(`device/kernels/writer_uniform.cpp:24`); there is no page-size override to drop.

**CB endpoints:**

- **`c_0` dst** (`device/uniform_program_factory.cpp:144`) — **self-loop.** The writer is the only toucher:
  `cb_dst.reserve_back(1)` @ `device/kernels/writer_uniform.cpp:32`, `cb_dst.get_write_ptr()` @ `:33`,
  `cb_dst.push_back(1)` @ `:78`. Bind the writer **PRODUCER and CONSUMER** on this DFB. Legal on Gen1 for a DM kernel.
  Same disposition under both dtype configs.
- **`c_24` intermed** (`device/uniform_program_factory.cpp:133`) — **already legal 1:1.** Compute is the locked
  producer (`reserve_back` @ `compute_uniform.cpp:31`, `push_back` @ `:41`), the writer the locked consumer
  (`wait_front` @ `writer_uniform.cpp:36`, `get_read_ptr` @ `:38`, `pop_front` @ `:49`/`:64`). Bind one PRODUCER, one
  CONSUMER. No action beyond the ordinary CB→DFB translation.

No multi-binding advanced option anywhere. No dead CB — do **not** drop `c_0` even though its *memory* goes unused in
the FLOAT32 configuration (the audit records that as a team-only anomaly, not port work): the writer still reserves,
peeks and pushes it in both configs, and its `fifo_page_size` is read on every iteration.

## Watch for

- **CB endpoints (multi-binding):** none. No CB on any node has ≥3 touchers or two kernels locked to the same FIFO
  role. The hidden-second-writer hunt is negative by construction — the op declares **no semaphores at all**, so the
  semaphore-gated raw co-fill shape cannot occur here.

- **Cross-op / shared kernels — the one non-routine thing in this port.** Both kernels are **lent**: they sit in
  `uniform`'s own directory, inside your writeable surface, but `rand` binds them by file path. Converting either in
  place breaks `rand`.

  | Kernel | Also bound by | `_metal2` fork? |
  |---|---|---|
  | `device/kernels/writer_uniform.cpp` | `rand` — `ttnn/cpp/ttnn/operations/rand/device/rand_program_factory.cpp:28`, used at `:165` | **No fork yet — this port creates the first** |
  | `device/kernels/compute_uniform.cpp` | `rand` — `ttnn/cpp/ttnn/operations/rand/device/rand_program_factory.cpp:29`, used at `:181` | **No fork yet — this port creates the first** |

  Take **rung 2** of *Caution: Porting a shared kernel* (`port_patterns.md`): copy each to
  `writer_uniform_metal2.cpp` / `compute_uniform_metal2.cpp` **beside the originals**, in `uniform`'s own directory;
  convert the copies; point your `KernelSpec::source` at them; leave the originals untouched apart from the pointer
  comment. Both forks are checked in with the port.

  `{rand}` is a **sunset list, not authorization to convert the kernels in place** — and `rand` sits under the same
  family-wide hold as `uniform`, so it cannot co-migrate today regardless. Because `rand` inherits your binding names
  at sunset, **name the bindings for the kernel's role vocabulary, not `uniform`'s locals** — the kernels' own words
  (`dst_addr` → `tensor::dst`, `intermed`, `dst`) rather than anything `uniform`-specific.

- **RTA varargs:** none — prefer named RTAs throughout. Every arg in both kernels is a distinct field at a constant
  index; there is no counted loop and no data-selected index. Names are already legible from the kernel locals:
  - writer (`device/kernels/writer_uniform.cpp:19-21`) → `dst_addr` *(becomes the tensor binding)*, `start_id`,
    `num_tiles`
  - compute (`device/kernels/compute_uniform.cpp:13,18,19,21,22`) → `seed`, `f2u_from`, `f2u_to`, `start_id`,
    `num_tiles`

- **One metadata lookup to resolve, not swap blind.** `get_local_cb_interface(dst_cb_id).fifo_page_size` @
  `device/kernels/writer_uniform.cpp:26` is sanctioned Device 2.0, and kernel-side whitelist rule 7 moves such lookups
  onto the DFB object. But `DataflowBuffer` (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:167+`) exposes a **tile**
  metadata set — `get_tile_size` / `get_tile_r_dim` / `get_tile_c_dim` / `get_tile_hw` / `get_tile_num_faces` — with
  no direct `fifo_page_size` analog. For this op the CB's page size *is* its tile size (`page_size = dtype_tile_size`
  @ `device/uniform_program_factory.cpp:150`), so `dfb::dst.get_tile_size()` should be the equivalent — but that is an
  inference from the descriptor, not an API identity. The value is used as the NOC write size on every iteration
  (`:45`, `:69`), so confirm it before swapping.
