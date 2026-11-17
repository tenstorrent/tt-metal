# Post-Port Report — point_to_point

Op: `ttnn/cpp/ttnn/operations/point_to_point`, on top of `a54fb44aba9` (the Metal 2.0 port).

## Pass 1 — Convert sync-free DFBs to their proper shapes (`style/sync_free_dfbs.md`)

### Outcome

**NO SITES FOUND.** The op has seven `DataflowBufferSpec`s over three programs. Each one has at least one of the six
credit methods called on its handle in a binding kernel, so none is sync-free. Nothing was changed.

### Sites

None. The survey, per DFB (spec → binding kernels → credit calls on that handle):

| Program | DFB | Spec | Binding kernels | Credit calls |
|---|---|---|---|---|
| send | `sender` | `device/host/send_program_factory.cpp:69` | `reader_unary_interleaved_start_id_gen_metal2` (PRODUCER), `writer_send_metal2` (CONSUMER) | `reader_unary_interleaved_start_id_gen_metal2.cpp:29,32`; `writer_send_metal2.cpp:69,95` |
| send | `packet_header` | `device/host/send_program_factory.cpp:75` | `writer_send_metal2` (PRODUCER + CONSUMER) | `writer_send_metal2.cpp:42,44,98,100` |
| send | `packet` | `device/host/send_program_factory.cpp:82` | `writer_send_metal2` (PRODUCER + CONSUMER) | `writer_send_metal2.cpp:52,54` |
| receive | `packet_header` | `device/host/receive_program_factory.cpp:68` | `reader_receive_metal2` (PRODUCER + CONSUMER) | `reader_receive_metal2.cpp:40,42` |
| receive | `packet` | `device/host/receive_program_factory.cpp:75` | `reader_receive_metal2` (PRODUCER + CONSUMER) | `reader_receive_metal2.cpp:59,93` |
| receive | `receiver` | `device/host/receive_program_factory.cpp:81` | `reader_receive_metal2` (PRODUCER), `writer_unary_interleaved_start_id_gen_metal2` (CONSUMER) | `reader_receive_metal2.cpp:69,91`; `writer_unary_interleaved_start_id_gen_metal2.cpp:26,29` |
| local copy | `pages` | `device/host/local_copy_program_factory.cpp:50` | `reader_unary_interleaved_start_id_gen_metal2` (PRODUCER), `writer_unary_interleaved_start_id_gen_metal2` (CONSUMER) | same lines as the two unary kernels above |

No DFB is bound conditionally, and no DFB handle or id goes into an RAII guard or a template argument.

### Verification

Nothing was changed, so the before and after trees are the same tree. The sentinels are green on it:

- `tests/ttnn/unit_tests/operations/point_to_point/test_point_to_point_same_device.py` — 8 passed (local-copy program, on
  hardware).
- A throw-away gtest (not committed) on the 2-chip mock fabric fixture `ProgramSpecFabricTestGen1` built the send and
  receive `ProgramSpec`s with the same kernels, DFBs, bindings, named args, GlobalSemaphore parameter and fabric
  connection as the factories, and JIT-compiled them — 2 passed. This host cannot run the send and receive programs.

### Noticed, not done

- **The four scratch self-loops are sites for the semantic pass `semantic/dm_self_loop_dfbs.md`.** Send `packet_header`,
  send `packet`, receive `packet_header`, receive `packet` are DM self-loops whose FIFO calls only track an address. The
  port report already lists them as Quasar debt (the Gen2 validator rejects DM self-loops). That pass needs its own run
  with someone at the keyboard. The recipe order (this pass first) is satisfied.
- **Receive `packet` is a NOC operand, not only an address grab.** `reader_receive_metal2.cpp:75` passes the `dfb_packet`
  handle as the destination of `noc.async_read(packet_buffer, dfb_packet, …)`. The semantic pass must check that a
  `Scratchpad` (or its base address) is an accepted destination for that `Noc` call before it converts this DFB. Also,
  this DFB's `reserve_back` (`:59`) and `push_back` (`:93`) enclose the whole page loop, not one use.
- **The `packet_header` DFBs reserve more than they use.** Both specs reserve
  `num_packet_headers_storable * buffering_factor` = 4 entries. The send kernel takes 2 entries and the receive kernel
  takes 1. A future scratchpad conversion carries the reserved bytes across (per the recipe), so the oversize stays
  unless the op owner shrinks it.
- **The factory comment "Scratch DFB for …"** (`send_program_factory.cpp:81`, `receive_program_factory.cpp:74`) already
  calls the `packet` DFBs scratch. It becomes accurate after the semantic pass converts them.
