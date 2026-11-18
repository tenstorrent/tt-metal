# Post-Port Report — all_gather_async

Op: `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async`, on top of `11174c4e133` (the Metal 2.0 port). The passes
cover the ported factory, `LlamaShardedMeshWorkloadFactory`, and its two kernels. `DefaultMeshWorkloadFactory` and
`AllGatherViaBroadcastFactory` are not ported (they still build `ProgramDescriptor`s), so the passes do not apply to them.

**Sentinel set** (used for every pass; this host cannot open a multi-device mesh, so no pytest runs here): a throw-away
gtest (not committed) on the 2-chip mock fabric fixture `ProgramSpecFabricTestGen1`, which builds the llama-sharded
per-coordinate `ProgramSpec` (same two kernels, DFBs, bindings, named args, padded varargs, GlobalSemaphore parameters,
`forward` / `backward` fabric connection specs) for three line-of-4 positions — first device / forward only / barrier /
1 link / mult+silu shape; second device / both / no barrier / 2 links / post-SDPA shape; last device / backward only /
barrier / 2 links / mult+silu shape — runs `MakeProgramFromSpec` and JIT-compiles it in an empty JIT cache; a fourth case
runs `SetProgramRunArgs` and then `UpdateProgramRunArgs` with the override's run args; plus the existing
`CPU_FabricConnectionBindingJITSmoke`. **5 passed** on `11174c4e133`.

## Pass 1 — Convert sync-free DFBs to their proper shapes (`style/sync_free_dfbs.md`)

### Outcome

**NO SITES FOUND.** The factory has two `DataflowBufferSpec`s per program, and each one has credit calls on its handle in
a binding kernel. Nothing was changed.

### Sites

None. The survey (spec → binding kernels → credit calls on that handle):

| DFB | Spec | Binding kernels | Credit calls |
|---|---|---|---|
| `data` | `device/all_gather_async_llama_sharded_program_factory.cpp:173` | `reader` (PRODUCER, `:210`), `writer` (CONSUMER, `:248`) | `llama_shapes_sharded_reader.cpp:56, 68` (`reserve_back` / `push_back`); `llama_shapes_sharded_writer.cpp:164, 193` (`wait_front` / `pop_front`) |
| `packet_header` | `:182` | `writer` (PRODUCER + CONSUMER, `:254, 259`) | `llama_shapes_sharded_writer.cpp:101, 103, 104, 106, 107, 109` (`reserve_back` / `push_back`, three times) |

No DFB is bound conditionally. No DFB handle or id goes into a helper, an RAII guard or a template argument; `dfb0` and
`dfb_packet_header` are used only through their own methods. No `pages_reservable_at_back` / `pages_available_at_front`.

### Verification

Nothing was changed, so the before and after trees are the same tree: sentinel set green (5 passed).

### Noticed, not done

- **`packet_header` is a DM self-loop with FIFO calls** (`reserve_back` / `push_back`, never `wait_front` / `pop_front`).
  It is not sync-free, so it is not a site here; it is the target of `semantic/dm_self_loop_dfbs.md` (pass 2).
- **`data` holds every page of a worker** (`num_entries = input_pages / num_links + 1`, the legacy "small shapes"
  sizing), so the reader never blocks on `reserve_back` in practice. It is still a real FIFO between two kernels.
