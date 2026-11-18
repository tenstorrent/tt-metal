# Post-Port Report — all_to_all_async

Op: `ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async`, on top of `e5e5b7cdc18` (the Metal 2.0 port).

**Sentinel set** (used for every pass; this host cannot open a multi-device mesh and the op needs a ring of at least 3
devices, so no pytest runs here): a throw-away gtest (not committed) on the 2-chip mock fabric fixture
`ProgramSpecFabricTestGen1`, which builds the op's per-coordinate `ProgramSpec` (same four kernels, DFBs, bindings, named
args, GlobalSemaphore parameter, `forward` / `backward` fabric connection specs) for four ring positions — first device /
in_dim 2 / both connections, second device / in_dim 3 / both, last device / forward only, third device / backward only —
runs `MakeProgramFromSpec` and JIT-compiles it; a fifth case runs `SetProgramRunArgs` and then `UpdateProgramRunArgs`
with the override's run args; plus the existing `CPU_FabricConnectionBindingJITSmoke`. **6 passed** on `e5e5b7cdc18`.

## Pass 1 — Convert sync-free DFBs to their proper shapes (`style/sync_free_dfbs.md`)

### Outcome

**NO SITES FOUND.** The op has three `DataflowBufferSpec`s per program, and each one has credit calls on its handle in a
binding kernel. Nothing was changed.

### Sites

None. The survey (spec → binding kernels → credit calls on that handle):

| DFB | Spec | Binding kernels | Credit calls |
|---|---|---|---|
| `sender_data` | `device/all_to_all_async_program_factory.cpp:92` | `sender_reader` (PRODUCER, `:332`), `sender_writer` (CONSUMER, `:388`) | `interleaved_all_to_all_reader.cpp:75,89` (`reserve_back` / `push_back`); `interleaved_all_to_all_writer.cpp:144,186` (`wait_front` / `pop_front`) |
| `packet_header` | `:100` | `sender_writer` (PRODUCER + CONSUMER, `:378,383`) | `interleaved_all_to_all_writer.cpp:83,85,86,88,89,91` (`reserve_back` / `push_back`, three times) |
| `receiver_data` | `:114` | `receiver_reader` (PRODUCER, `:507`), `receiver_writer` (CONSUMER, `:457`) | `interleaved_all_to_all_receiver_reader.cpp:65,75,90,115`; `interleaved_all_to_all_receiver_writer.cpp:51,67,78,90` |

No DFB is bound conditionally. No DFB handle or id goes into a helper, an RAII guard or a template argument; the
`DataflowBuffer` objects are passed only to `Noc::async_read` / `async_write` as endpoints.

### Verification

Nothing was changed, so the before and after trees are the same tree: sentinel set green (6 passed).

### Noticed, not done

- **`packet_header` is a DM self-loop with FIFO calls** (`reserve_back` / `push_back`, never `wait_front` /
  `pop_front`). It is not sync-free, so it is not a site here; it is a candidate for the semantic pass
  `semantic/dm_self_loop_dfbs.md` (pass 2).
- **Quasar implicit sync.** `ttnn::create_reader_datamovement_config()` / `create_writer_datamovement_config()` take
  `disable_dfb_implicit_sync_for_all` (default `false`). Whether to set it is a Quasar-uplift decision for the op owner;
  it has no effect on Gen1.

## Pass 2 — Replace fake-FIFO DM self-loops with a scratchpad (`semantic/dm_self_loop_dfbs.md`)

### Outcome

**APPLIED (1 site).** The packet-header DM self-loop DFB is now a scratchpad. The op has no DM self-loop DFB left.

### Sites

- `packet_header` — sender writer (`device/kernels/interleaved_all_to_all_writer.cpp:77, 83-99`; host
  `device/all_to_all_async_program_factory.cpp:42, 98-103` spec, `:382-388` binding). Before: three times
  `reserve_back(1)` → `get_write_ptr()` captured into `packet_header_buffer_addr_forward` / `_backward` / `_seminc` →
  `push_back(1)`; never waited or popped. After: `Scratchpad<uint8_t> packet_header(scratch::packet_header)`; a byte
  index `packet_header_wr` replays the write pointer: each address is `base + packet_header_wr` at the point of the old
  `get_write_ptr()`, and each of the first two `push_back(1)` becomes `packet_header_wr += packet_header_entry_elems`
  with the FIFO's `ASSERT` / equality wrap. `size_per_node = get_tt_fabric_packet_header_size_bytes() *
  (PACKET_HEADER_BUFFER_SIZE * 2)` (the DFB's `entry_size * num_entries`).

Survey per the recipe: bound by one DM kernel (`sender_writer`) as PRODUCER and CONSUMER, no other binder; every use is on
the covered list (`reserve_back`, `get_write_ptr`, `push_back`; the addresses go to `reinterpret_cast<volatile
PACKET_HEADER_TYPE*>` and to `send_payload_flush_blocking_from_address`); no helper, guard or template takes the handle;
`borrowed_from` unset; no `dfb_run_overrides` (the op has none). `data_format_metadata` (`RawUInt32`) is consulted by
nothing — every use is an address grab — so it drops.

**Stride.** The index is read after it advances (the second and third `get_write_ptr()` follow a `push_back`), so a
stride is needed. The spec's `entry_size` is `tt::tt_fabric::get_tt_fabric_packet_header_size_bytes()`. The kernel
already uses `sizeof(PACKET_HEADER_TYPE)` as the size of each header (`send_payload_flush_blocking_from_address(…,
sizeof(PACKET_HEADER_TYPE))`). The two are the same quantity by construction: the host function returns
`sizeof(<header type>)` for the header type that the fabric config selects (`FabricContext::compute_packet_header_size_bytes`
→ `visit_packet_header_type(…) { return sizeof(Header); }`, plus `sizeof(UDMControlFields)` in UDM mode, where
`fabric_edm_packet_header.hpp` `static_assert`s `sizeof(UDMHybridMeshPacketHeader)` equals that sum), and the kernel's
`PACKET_HEADER_TYPE` is that header type, selected from the same config through `ROUTING_MODE` /
`FABRIC_1D_PKT_HDR_EXTENSION_WORDS`. So the pass reuses `sizeof(PACKET_HEADER_TYPE)` and adds no compile-time arg.
**This is the one judgment in the pass; the launcher may prefer to review it** (see Noticed).

**Cleanup applied:** item 1 — the third `push_back(1)` advance is dead (nothing reads the index after it), so it is not
written. No other cleanup.

**`T` choice.** The kernel only takes addresses (it casts them to `volatile PACKET_HEADER_TYPE*` itself), so
`T = uint8_t` and the index is a byte offset. The base is `reinterpret_cast<uint32_t>(&packet_header[0])`, the form the
earlier passes on this branch use (`Scratchpad::get_base_address()`, named by the recipe, does not exist here).

### Verification

Sentinel set green before (6 passed, on `e5e5b7cdc18`; only clang-format reflow and the pass-1 report since) and after
(6 passed). The sentinel spec was updated to the converted shape (a `ScratchpadSpec` and a `ScratchpadBinding` on the
sender writer instead of the self-loop DFB) and was run in an empty JIT cache, so all four writer variants were compiled
against the converted kernel.

### Repaired because the change falsified it

- Kernel comment `// packet header dfb` → `// packet header scratchpad: the forward, backward and seminc headers, one after
  another` (there is no DFB any more).
- Host comment `// Create dataflow buffers for sender cores` → `// Create the dataflow buffer and the packet header
  scratchpad for sender cores` (the helper now creates a scratchpad too).

### Noticed, not done

- **The wrap checks never fire.** The region holds 16 headers and the kernel uses 3, so the two `if (wr == size)` resets
  are inert. They are kept because the recipe's cleanup list does not cover a live-but-inert wrap.
- **Oversized packet-header region.** 16 headers reserved (`PACKET_HEADER_BUFFER_SIZE * 2`), 3 used; legacy sizing,
  carried as is.
- **Stride judgment.** If the recipe owner reads "same expression" strictly, `sizeof(PACKET_HEADER_TYPE)` vs
  `get_tt_fabric_packet_header_size_bytes()` is a stop, not a reuse. The argument for reuse is above; the alternative
  is to leave the self-loop (Quasar debt).
- **Recipe/API drift:** `Scratchpad::get_base_address()` is named by the recipe but absent here.

## Pass 3 — Add the Gen2 hardware configs (`semantic/gen2_hardware_configs.md`)

### Outcome

**NO SITES FOUND.** All four `hw_config`s come from the arch-agnostic TTNN DM helpers (shape 1), and nothing sets a field
on them afterwards. Nothing was changed.

### Sites

None. Survey (`grep -rn "hw_config\|to_compute_hardware_config\|Gen1Config\|1XXConfig\|std::get<\|std::get_if<\|holds_alternative"`
over the op directory, 13 `.cpp`/`.hpp` files):

| KernelSpec | `hw_config` | Shape |
|---|---|---|
| `sender_reader` | `ttnn::create_reader_datamovement_config()` (`device/all_to_all_async_program_factory.cpp:366`) | 1 |
| `sender_writer` | `ttnn::create_writer_datamovement_config()` (`:426`) | 1 |
| `receiver_writer` | `ttnn::create_writer_datamovement_config()` (`:493`) | 1 |
| `receiver_reader` | `ttnn::create_reader_datamovement_config()` (`:549`) | 1 |

No compute kernel, no hand-written `DataMovement1XXConfig`, no `std::get` / `std::get_if` / `holds_alternative`.

### Verification

Nothing was changed, so the before and after trees are the same tree: sentinel set green (6 passed, on `3c26d153570`).

### Noticed, not done

- **Helper signature drift.** The recipe spells the helpers `create_reader_datamovement_config(arch)`; on this branch they
  take no arch and always fill both `config_1xx` and `config_2xx`
  (`ttnn/cpp/ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp`). Same outcome: shape 1, no work.
- **`disable_dfb_implicit_sync_for_all`** stays at its default (`false`) on all four kernels. Whether a Quasar bring-up
  needs it for these kernels is for the op owner.
