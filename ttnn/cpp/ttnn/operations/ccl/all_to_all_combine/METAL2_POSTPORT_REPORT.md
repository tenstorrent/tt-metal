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
