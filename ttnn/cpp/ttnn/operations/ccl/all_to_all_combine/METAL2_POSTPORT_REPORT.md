# Post-Port Report — all_to_all_combine

Op: `ttnn/cpp/ttnn/operations/ccl/all_to_all_combine`, on top of `8e3a6c01aa8` (the Metal 2.0 port).

**Sentinel set** (used for every pass; this host cannot open a multi-device mesh, and on one device the op `TT_FATAL`s in
`get_neighbors`, so no pytest runs here): a throw-away gtest (not committed) on the 2-chip mock fabric fixture
`ProgramSpecFabricTestGen1`, which builds the op's per-coordinate `ProgramSpec` (same kernels, DFBs, bindings, named args,
GlobalSemaphore parameters, four fabric connection specs) for three configs — Linear 1x2 axis 1 first device, Linear 2x1
axis 0 second device (locally reduced), Ring 1x4 axis 1 second device — runs `MakeProgramFromSpec` and JIT-compiles it; a
fourth case runs `SetProgramRunArgs` and then `UpdateProgramRunArgs` with fresh tensors (the no-override cache-hit
refresh); plus the existing `CPU_FabricConnectionBindingJITSmoke`. **5 passed** on `8e3a6c01aa8`.

## Pass 1 — Convert sync-free DFBs to their proper shapes (`style/sync_free_dfbs.md`)

### Outcome

**NO SITES FOUND.** Every `DataflowBufferSpec` has credit calls on its handle in a binding kernel. Nothing was changed.

### Sites

None. The survey (spec → binding kernels → credit calls on that handle):

| DFB | Spec | Binding kernels | Credit calls |
|---|---|---|---|
| `data` | `device/all_to_all_combine_program_factory.cpp:162` | reader PRODUCER (`:252`), writer CONSUMER (`:326`) | `reader_all_to_all_combine_metal2.cpp:112,122`; `writer_all_to_all_combine.cpp:226,284` |
| `mapping` | `:169` | reader PRODUCER + CONSUMER (`:232`, `:237`) | `reader_all_to_all_combine_metal2.cpp:87,89` (`reserve_back` / `push_back`) |
| `local_experts` | `:175` | reader PRODUCER (`:242`), writer CONSUMER (`:310`) | `reader…metal2.cpp:83,93`; `writer…cpp:209,294` |
| `metadata` | `:182` | reader PRODUCER (`:247`), writer CONSUMER (`:305`) | `reader…metal2.cpp:95,129`; `writer…cpp:216,292` |
| `packet_header` | `:188` | writer PRODUCER + CONSUMER (`:316`, `:321`) | `writer_all_to_all_combine.cpp:186,192` (`reserve_back(3)` / `push_back(3)`) |

No DFB is bound conditionally. The only handles passed to a function are `dfb_metadata` / `dfb_data` as `Noc::async_read`
/ `async_write` operands (`reader…metal2.cpp:99,116`; `writer…cpp:243`), which take the address and make no credit call.
`mapping` and `packet_header` are self-loops with real credit calls (way 1 of the recipe), so they are not sites here; they
are the targets of the DM self-loop pass.

### Verification

Nothing was changed, so the before and after trees are the same tree: sentinel set green (5 passed).

### Noticed, not done

- **DM self-loops `mapping` and `packet_header`** — fake FIFOs (reserve + push once, never waited or popped); left for
  `semantic/dm_self_loop_dfbs.md`.
- **Packet-header DFB size.** `packet_header` reserves 3 × `CLIENT_INTERFACE_SIZE` (3280 B) entries but the writer uses
  only three `sizeof(PACKET_HEADER_TYPE)` headers from its base (`writer_all_to_all_combine.cpp:183-191`). Legacy sizing,
  carried as is.

## Pass 2 — Replace fake-FIFO DM self-loops with a scratchpad (`semantic/dm_self_loop_dfbs.md`)

### Outcome

**APPLIED (2 sites).** Both DM self-loop DFBs are now scratchpads. The op has no DM self-loop DFB left.

### Sites

- `mapping` — reader (`reader_all_to_all_combine_metal2.cpp:78,88`; host `device/all_to_all_combine_program_factory.cpp:183-186`,
  binding `:244-251`). Before: `reserve_back(1)`, `get_write_ptr()` captured into `mapping_buffer_addr`, `push_back(1)`; the
  captured address is then handed to the op-local helper `get_device_expert_indices` as the NoC read destination and for raw
  reads. After: `Scratchpad<uint8_t> mapping(scratch::mapping)`; the address is the scratchpad base. `size_per_node =
  aligned_mapping_page_size_bytes * 1` (the DFB's `entry_size * num_entries`).
- `packet_header` — writer (`writer_all_to_all_combine.cpp:180,187`; host `:187-190`, binding `:315-322`). Before:
  `reserve_back(3)`, `get_read_ptr()` captured as the base, three header pointers at `base + i * sizeof(PACKET_HEADER_TYPE)`,
  `push_back(3)`; never waited or popped. After: `Scratchpad<uint8_t> packet_header(scratch::packet_header)`; the base is the
  scratchpad base, the three header pointers are computed as before. `size_per_node = CLIENT_INTERFACE_SIZE * num_headers`.

Survey per the recipe: both DFBs are bound by exactly one DM kernel as PRODUCER and CONSUMER; every use is on the covered
list (FIFO calls and one getter, captured before the only `push_back`); `borrowed_from` unset; no `dfb_run_overrides` (the op
has none). No index is read after it advances, so there is no stride and no wrap: both indices stay 0 and the calls simply
disappear. `data_format_metadata` (mapping dtype / UInt32) is consulted by nothing — every use is an address grab — so it
drops.

**`T` choice.** Both kernels only take the base address and do byte arithmetic on it (the reader passes a `uint32_t` to its
helper; the writer adds `i * sizeof(PACKET_HEADER_TYPE)`), so `T = uint8_t`. For `packet_header`, `T = volatile
PACKET_HEADER_TYPE` is not possible as is: the region is `3 * 3280` bytes, which is not a whole number of headers for every
packet-header size, and the `Scratchpad` constructor `ASSERT`s `size_in_bytes % sizeof(T) == 0`.

**Base address.** The recipe names `Scratchpad::get_base_address()`; it does not exist on this branch (`api/scratchpad.h`
has `operator[]`, `local_mem()`, `size()`, `size_in_bytes()`). The base is taken as `reinterpret_cast<uint32_t>(&x[0])`, the
same form the `point_to_point` DM self-loop pass on this branch uses (`writer_send_metal2.cpp`).

### Verification

Sentinel set green before (5 passed, on `a09cd2c5228`) and after (5 passed). The sentinel spec was updated to the converted
shape (two `ScratchpadSpec`s and `ScratchpadBinding`s instead of the two self-loop DFBs) and JIT-compiles both converted
kernels.

### Repaired because the change falsified it

- `writer_all_to_all_combine.cpp:183-185` comment: "get_read_ptr stays pinned (nothing pops this scratch DFB)" — there is
  no DFB and no `get_read_ptr` any more; now says the scratchpad base is taken once.
- Host binding comment "packet headers, reserved once and never popped by the writer" → "packet headers" (nothing is
  reserved or popped).
- Host local `mapping_data_format` (only consumer was the deleted DFB spec's `data_format_metadata`) — deleted; it would
  also have been an unused variable under `-Werror`.

### Noticed, not done

- **The reader fork's interface changed.** `reader_all_to_all_combine_metal2.cpp` now binds `scratch::mapping` instead of
  `dfb::mapping`. It has no other consumer yet; when `moe_expert_token_remap` ports, it must declare a `mapping`
  scratchpad (and size it from its own mapping page).
- **Oversized packet-header region.** 9840 bytes for three headers; legacy sizing, carried as is (the pass carries the DFB's
  bytes across, it does not resize).
- **Recipe/API drift:** `Scratchpad::get_base_address()` is named by the recipe but absent here (see above).

## Pass 3 — Add the Gen2 hardware configs (`semantic/gen2_hardware_configs.md`)

### Outcome

**APPLIED (2 sites).** Both kernels had a hand-written Gen1-only DM config (shape 4, data movement). Each now also carries
the default-constructed Gen2 config.

### Sites

- `device/all_to_all_combine_program_factory.cpp:289` — reader `hw_config` (Gen1: `RISCV_1`, `NOC_1`, `DM_DEDICATED_NOC`).
- `device/all_to_all_combine_program_factory.cpp:357` — writer `hw_config` (Gen1: `RISCV_0`, `NOC_0`, `DM_DEDICATED_NOC`).

Survey (`grep -rn "hw_config\|to_compute_hardware_config\|Gen1Config\|1XXConfig\|2XXConfig\|std::get<\|std::get_if<\|holds_alternative\|disable_dfb_implicit"`
over the op directory): the two `hw_config`s above and nothing else — no compute kernel, no helper-built config, no field
set after the fact, no `disable_dfb_implicit_sync*` anywhere in the op.

**Shape on this branch.** The recipe describes `hw_config` as a variant over `DataMovementGen1Config` /
`DataMovementGen2Config`, chosen by an `arch` branch. On this branch `DataMovementHardwareConfig` is one struct with two
optional members, `config_1xx` and `config_2xx`, and the target architecture picks one at program construction
(`data_movement_hardware_config.hpp`). So the mechanical equivalent of "default-construct the Gen2 config" is
`.config_2xx = DataMovementHardwareConfig::DataMovement2XXConfig{}` next to the existing `.config_1xx`, with no `arch`
branch. This is also what the arch-agnostic TTNN helpers do (`datamovement_kernel_config.hpp`: `config_2xx` with
`disable_dfb_implicit_sync_for_all = false`, which is the default). The Gen1 initializers are textually unchanged (two
added lines in the diff, nothing removed). `disable_dfb_implicit_sync_for_all` / `disable_dfb_implicit_sync_for` are not set.
No `unpack_modes` marker: both sites are DM, so there is nothing for it to mark.

### Verification

Sentinel set green before (5 passed, on `9582d0fc6ce`) and after (5 passed); the sentinel spec carries the same
`config_2xx`. This exercises only the Gen1 path; nothing here checks the Gen2 values (none were authored — the Gen2 config
is default-constructed).

### Noticed, not done

- **Implicit sync on Quasar.** Both kernels move one row-major page (a "stick") per DFB entry with explicit
  `reserve_back`/`push_back` and `wait_front`/`pop_front`. The helper header says DM kernels doing many sub-tile transfers can
  stall the Gen2 implicit credit accounting. Whether this op needs `disable_dfb_implicit_sync_for_all` on Quasar is a
  Quasar-uplift decision for the op owner; this pass does not set it.
- **Recipe drift.** The recipe's variant-and-`arch`-branch shape does not match this branch's `config_1xx` / `config_2xx`
  struct; worth a note in the recipe.
