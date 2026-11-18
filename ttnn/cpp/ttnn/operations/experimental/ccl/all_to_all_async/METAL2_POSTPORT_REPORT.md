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
