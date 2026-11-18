# Post-Port Report — all_to_all_dispatch

Op: `ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch`, on top of `63a9895942d` (the Metal 2.0 port).

**Sentinel set** (used for every pass; this host cannot open a multi-device mesh, and on one device the op `TT_FATAL`s in
`get_neighbors`, so no pytest runs here): a throw-away gtest (not committed) on the 2-chip mock fabric fixture
`ProgramSpecFabricTestGen1`, which builds the op's per-coordinate `ProgramSpec` (same kernels, DFBs, scratchpads,
bindings, named args, defines, GlobalSemaphore parameters, four fabric connection specs) for five configs — Linear 1x2
axis 1 first device FullPacket, Linear 2x1 axis 0 second device PageByPage, Ring 1x4 axis 1 second device FullPacket and
PageByPage, Linear 1x2 no axis second device PageByPage — runs `MakeProgramFromSpec` and JIT-compiles it; two more cases
(one per mode) run `SetProgramRunArgs` and then `UpdateProgramRunArgs` with fresh tensors (the no-override cache-hit
refresh); plus the existing `CPU_FabricConnectionBindingJITSmoke`. **8 passed** on `63a9895942d`.

## Pass 1 — Convert sync-free DFBs to their proper shapes (`style/sync_free_dfbs.md`)

### Outcome

**APPLIED (2 sites)**, plus **one DFB left as is with a feature request** (sync-free, but two kernels on one node; see
below).

### Sites

- `packet_header` — writer only. Before: DFB spec `device/all_to_all_dispatch_program_factory.cpp:306` (at `63a9895942d`),
  self-loop bindings `:403-412`; kernel `writer_all_to_all_dispatch.cpp:165,202` — one `get_read_ptr()` captured as the
  base of three packet headers, no credit call anywhere. After: `ScratchpadSpec` (`.cpp:308`, `size_per_node =
  dfb_sizes[5]` = `3 * packet_header_size`, the DFB's entries × entry size), `ScratchpadBinding` (`.cpp:485`);
  kernel `Scratchpad<uint8_t> packet_header(scratch::packet_header)`, base
  `reinterpret_cast<uint32_t>(&packet_header[0])`.
- `send_preparation_buffer` — writer only. Before: DFB spec `.cpp:313`, self-loop bindings `:413-422`; kernel
  `writer…:166,168-170,232` — `get_write_ptr()` captured, the handle passed to `detail::zero_buffer_async` (which calls
  `Noc::async_write_zeros`, no credit call; followed into `internal/tt-1xx/noc_zero_l1.inl`), then raw `uint8_t*` element
  reads and writes. After: `ScratchpadSpec` (`.cpp:313`, `size_per_node = dfb_sizes[3]` = `num_devices *
  tokens_per_core`), `ScratchpadBinding` (`.cpp:490`); kernel `Scratchpad<uint8_t>
  send_preparation_buffer(scratch::send_preparation_buffer)`, zeroed through the same helper (now taking the
  scratchpad) and indexed with `operator[]` — the index expressions (`writer…:252-253`) are textually unchanged.

Survey (spec → binding kernels → the six credit methods on that handle, then opaque uses of the handle):

| DFB | Binding kernels | Credit calls | Verdict |
|---|---|---|---|
| `input` | reader PRODUCER, writer CONSUMER | reader `reserve_back` / `push_back`; writer `wait_front` / `pop_front` | synchronized |
| `indices` | reader PRODUCER, writer CONSUMER | same | synchronized |
| `mapping` | reader PRODUCER, writer CONSUMER | reader `reserve_back` / `push_back`; writer `wait_front` / `pop_front` | synchronized |
| `packet_header` | writer PRODUCER + CONSUMER | none; handle only used for `get_read_ptr()` | **sync-free, regular-backed → Scratchpad** |
| `send_preparation_buffer` | writer PRODUCER + CONSUMER | none; handle passed to `zero_buffer_async` → `Noc::async_write_zeros` (no credit call) | **sync-free, regular-backed → Scratchpad** |
| `metadata_buffer` (FullPacket only) | writer PRODUCER, reader CONSUMER | none; writer `get_read_ptr()` (loopback write, NoC destination for remote sends), reader `get_write_ptr()` (source of the copy to the metadata tensor) | **sync-free, but two kernels on one node → not convertible (feature request)** |

`data_format_metadata` (RawUInt32, UInt8) of the two converted DFBs is consulted by nothing — every use is an address
grab, a NoC zero, or a raw element access — so it drops. `borrowed_from` is unset on every DFB of this op.

### Feature request — Scratchpad shared by kernels on the same node

> **Feature request — Scratchpad shared by kernels on the same node.**
> Op / factory: `ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch` / `AllToAllDispatchSparse`
> (`device/all_to_all_dispatch_program_factory.cpp`, the `metadata_buffer` DFB, declared only when
> `impl == FullPacket`).
> The DFB `metadata_buffer` is sync-free and regular-backed, so it is a scratchpad by nature, but it is bound by `writer`
> and `reader`, which share nodes (both run on every sender core) — and a `ScratchpadSpec` may be bound by multiple
> `KernelSpec`s only over disjoint node sets. What the kernels use it for: the writer of every device writes its tokens'
> metadata into this buffer, locally through a NoC loopback write and on the other devices of the dispatch group through
> fabric writes addressed at the same L1 offset; after the completion semaphore reaches its count, the reader of the
> same core copies the whole buffer to the metadata output tensor. The ordering between the two kernels comes from the
> GlobalSemaphore, not from the DFB.
> Left as a DFB; no workaround applied.

### Verification

Sentinel set green before (8 passed, on `63a9895942d`) and after (8 passed). The sentinel spec was updated to the
converted shape (two `ScratchpadSpec`s and `ScratchpadBinding`s instead of the two self-loop DFBs) and JIT-compiles the
converted writer in both modes.

### Repaired because the change falsified it

- `detail::zero_buffer_async` took `const DataflowBuffer&`; there is no DFB any more, so it takes
  `const Scratchpad<uint8_t>&` (the body is unchanged apart from the parameter name).
- Kernel local `send_preparation_buffer_address` and the raw `uint8_t* send_preparation_buffer` alias of it: the raw
  pointer was the only way to reach the DFB's memory; the scratchpad named `send_preparation_buffer` now serves the same
  element accesses, so both declarations are gone and the uses read the same.
- Host DFB names `packet_header_dfb` / `send_preparation_buffer_dfb` → `packet_header_scratchpad` /
  `send_preparation_buffer_scratchpad` (they name scratchpad specs now).

### Noticed, not done

- **Recipe/API drift:** `Scratchpad::get_base_address()` is named by the recipe but absent on this branch
  (`api/scratchpad.h`); the packet-header base is `reinterpret_cast<uint32_t>(&x[0])`, as in `point_to_point` and
  `all_to_all_combine`. The recipe's warning that `&scratch[0]` is a `T*` holds; the cast is what makes it a `uint32_t`.
- **`operator[]` is bounds-checked** under `ASSERT` (watcher / debug builds); the raw pointer was not. No functional
  change in release builds; not presented as a safety improvement.
- **No DM self-loop is left** after this pass: the two self-loops of the port had no credit calls at all, so they were
  this pass's sites rather than the DM self-loop pass's.
- **Packet-header region size.** `3 * packet_header_size` bytes for three headers — exactly what is used.

## Pass 2 — Replace fake-FIFO DM self-loops with a scratchpad (`semantic/dm_self_loop_dfbs.md`)

### Outcome

**NO SITES FOUND.** No DFB is bound by one DM kernel as both `PRODUCER` and `CONSUMER`. Nothing was changed.

### Sites

None. Survey of every `DataflowBufferSpec` and its `DFBBinding`s after pass 1 (`2de9087cf79`):

| DFB | PRODUCER | CONSUMER | Self-loop? |
|---|---|---|---|
| `input` (`device/all_to_all_dispatch_program_factory.cpp:285`) | reader | writer | no |
| `indices` (`:292`) | reader | writer | no |
| `mapping` (`:299`) | reader | writer | no |
| `metadata_buffer` (`:320`, FullPacket only) | writer | reader | no |

The port had two DM self-loops (`packet_header`, `send_preparation_buffer` on the writer). Neither made a credit call,
so pass 1 (sync-free) converted both to scratchpads; there is nothing left for this pass. No compute kernel exists.

### Verification

Nothing was changed, so the before and after trees are the same tree: sentinel set green (8 passed, on `2de9087cf79`).

### Noticed, not done

- **Quasar note.** `metadata_buffer` is the last DFB with no credit calls in this op; it is a two-kernel DFB (reader and
  writer on one node), so it is neither this pass's target nor convertible by pass 1 (see the pass 1 feature request).

## Pass 3 — Add the Gen2 hardware configs (`semantic/gen2_hardware_configs.md`)

### Outcome

**APPLIED (2 sites).** Both kernels had a hand-written Gen1-only DM config (shape 4, data movement). Each now also carries
the default-constructed Gen2 config.

### Sites

- `device/all_to_all_dispatch_program_factory.cpp:459` — reader `hw_config` (Gen1: `RISCV_1`, `NOC_1`, `DM_DEDICATED_NOC`).
- `device/all_to_all_dispatch_program_factory.cpp:541` — writer `hw_config` (Gen1: `RISCV_0`, `NOC_0`, `DM_DEDICATED_NOC`).

Survey (`grep -rn "hw_config\|to_compute_hardware_config\|Gen1Config\|1XXConfig\|2XXConfig\|std::get<\|std::get_if<\|holds_alternative\|disable_dfb_implicit"`
over the op directory): the two `hw_config`s above and nothing else — no compute kernel, no helper-built config, no field
set after the fact, no `disable_dfb_implicit_sync*` anywhere in the op.

**Shape on this branch.** As for `all_to_all_combine`: the recipe describes `hw_config` as a variant over
`DataMovementGen1Config` / `DataMovementGen2Config`, chosen by an `arch` branch. On this branch
`DataMovementHardwareConfig` is one struct with two optional members, `config_1xx` and `config_2xx`, and the target
architecture picks one at program construction. So the mechanical equivalent of "default-construct the Gen2 config" is
`.config_2xx = DataMovementHardwareConfig::DataMovement2XXConfig{}` next to the existing `.config_1xx`, with no `arch`
branch — what the arch-agnostic TTNN helpers do (`datamovement_kernel_config.hpp`). The Gen1 initializers are textually
unchanged (two added lines in the diff, nothing removed). `disable_dfb_implicit_sync_for_all` /
`disable_dfb_implicit_sync_for` are not set. No `unpack_modes` marker: both sites are DM.

### Verification

Sentinel set green before (8 passed, on `cbc77b0e1c2`) and after (8 passed); the sentinel spec carries the same
`config_2xx`. This exercises only the Gen1 path; nothing here checks the Gen2 values (none were authored — the Gen2
config is default-constructed).

### Noticed, not done

- **Implicit sync on Quasar.** The reader moves one row-major page per DFB entry into `input` / `indices` with explicit
  `reserve_back` / `push_back`, and fills `mapping` with `mapping_pages` reads under one `reserve_back(mapping_pages)`;
  the writer consumes with `wait_front` / `pop_front`. `metadata_buffer` has no credit calls and is filled by fabric
  writes from other devices. Whether this op needs `disable_dfb_implicit_sync_for_all` (or a per-DFB opt-out for
  `metadata_buffer`) on Quasar is a Quasar-uplift decision for the op owner; this pass does not set it.
- **Recipe drift.** The recipe's variant-and-`arch`-branch shape does not match this branch's `config_1xx` /
  `config_2xx` struct.
