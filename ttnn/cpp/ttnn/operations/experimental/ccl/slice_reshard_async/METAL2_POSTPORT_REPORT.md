# Post-Port Report — slice_reshard_async

Op: `ttnn/cpp/ttnn/operations/experimental/ccl/slice_reshard_async`, on top of `7ba0b188e3f` (the Metal 2.0 port).

**Sentinel set** (used for every pass; this host cannot open a multi-device mesh and the op `TT_FATAL`s on a 1-device
ring, so no pytest runs here): a throw-away gtest (not committed) on the 2-chip mock fabric fixture
`ProgramSpecFabricTestGen1`, which builds the op's per-coordinate `ProgramSpec` (same kernels, DFB, bindings, named args,
GlobalSemaphore parameters, fabric connection) for the first, middle and last device of a ring, runs `MakeProgramFromSpec`
and JIT-compiles it; a fourth case runs `SetProgramRunArgs` and then `UpdateProgramRunArgs` with the override's run args;
plus the existing `CPU_FabricConnectionBindingJITSmoke`. **5 passed** on `7ba0b188e3f`.

## Pass 1 — Convert sync-free DFBs to their proper shapes (`style/sync_free_dfbs.md`)

### Outcome

**NO SITES FOUND.** The op has one `DataflowBufferSpec` per program, and both of its binding kernels make credit calls on
it. Nothing was changed.

### Sites

None. The survey (spec → binding kernels → credit calls on that handle):

| DFB | Spec | Binding kernels (per core) | Credit calls |
|---|---|---|---|
| `sender` | `device/slice_reshard_async_program_factory.cpp:144` | `reader_<i>` (PRODUCER, `:197-199`), `writer_<i>` (CONSUMER, `:264-266`) | `minimal_default_reader.cpp:68,74,81,83,92,98` (`reserve_back` / `push_back`); `minimal_default_writer.cpp:115,131,149,169,180,186` (`wait_front` / `pop_front`) |

No DFB is bound conditionally. The DFB id goes into one helper, `zeroPad` (`minimal_default_reader.cpp:29-35`), which only
zero-fills the entry the caller reserved; the caller makes the credit calls around it.

### Verification

Nothing was changed, so the before and after trees are the same tree: sentinel set green (5 passed).

### Noticed, not done

- **Quasar implicit sync.** `ttnn::create_reader_datamovement_config()` / `create_writer_datamovement_config()` take
  `disable_dfb_implicit_sync_for_all` (default `false`). Their header says DM kernels doing many sub-tile ("stick") NOC
  transfers stall the 2xx implicit credit accounting. Both kernels of this op move one stick per DFB entry. Whether to set
  the flag is a Quasar-uplift decision for the op owner; it has no effect on Gen1.
