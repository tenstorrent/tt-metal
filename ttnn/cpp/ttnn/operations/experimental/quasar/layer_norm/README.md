# `ttnn.experimental.quasar.layer_norm`

Quasar port of `ttnn/cpp/ttnn/operations/normalization/layernorm/` (namespace `ttnn::prim::qsr`,
host wrapper `ttnn::operations::experimental::quasar::layer_norm`). The program-config types are
the mainline ones, re-exported through `device/layernorm_types_qsr.hpp`, so the Python call
signature is identical to `ttnn.layer_norm`.

Validated on craq-sim (8x4 grid) with `models/demos/stable_diffusion_xl_base/quasar/tests/ops/test_layer_norm.py`:

| path | kernels | status |
|---|---|---|
| interleaved (`LayerNormDefaultProgramConfig`, `legacy_reduction=True`) | `reader_unary_interleaved_ln`, `writer_unary_interleaved_start_id_blocked`, `compute/layernorm` | passes |
| block sharded (`LayerNormShardedMultiCoreProgramConfig`, single-stage reduce) | `reader_mcast_{sender,receiver}_unary_sharded_ln`, `writer_unary_sharded_ln`, `compute/layernorm_sharded` | passes |
| large-tensor, row-major gamma/beta, Welford, pre/post all-gather, two-stage reduce | cloned, unchanged | untested; still use the draining helpers described below |

Mainline `ttnn.layer_norm` hangs on the simulator for both validated paths, so every change
below is a Quasar requirement, not a port artifact.

## Quasar `DataflowBuffer` rules the kernels had to follow

`~DataflowBuffer()` drains on Quasar (`tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl`,
`finish_impl`): a data-movement handle spins until the consumer acked every entry it pushed, an
UNPACK / PACK handle spins until the buffer's tile counter reads empty. The layernorm kernels keep
several buffers resident for the whole kernel (reduce scaler, eps, gamma, beta, column mask) and
pop them at the very end or never, so any handle that goes out of scope earlier deadlocks:

1. **Helper-owned handles.** `dataflow_kernel_lib::prepare_reduce_scaler` /
   `calculate_and_prepare_reduce_scaler` build a local handle, so the reader / writer hung right
   after pushing the scaler. `device/kernels/dataflow/layernorm_reduce_scaler_qsr.{hpp,inl}` is a
   copy with `*_into(DataflowBuffer&, ...)` variants that fill a caller-owned handle. The column
   mask generator (`col_mask_dataflow.h`) got the same treatment.
2. **Temporaries.** `DataflowBuffer(id).pop_front(n)` at the end of `compute/layernorm.cpp`
   (gamma / beta) never returned on UNPACK / PACK. The shared eltwise chain
   (`ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.inl`, `emit_wait/pop/reserve/push`) and the shared
   reduce helper (`ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl`) also built per-call
   handles; both are patched under `#ifdef ARCH_QUASAR` to hold them in a union whose destructor
   does not run. The reduce helper's `accum_dfb` aliases buffer 0 when accumulation is off, so its
   drain waited on the input buffer.
3. **Resident buffers at kernel exit.** `device/kernels/ln_dfb_qsr.h` defines `LN_RESIDENT_DFB`,
   a handle whose destructor does not run (plain `DataflowBuffer` on WH/BH). Every handle in the
   compute kernels and the resident constants in the reader / writer use it. The next program
   re-initializes the tile counters, so leaving a buffer occupied at exit is harmless.
4. **Pops that are no-ops.** An UNPACK `pop_front` on a buffer the host did not bind the compute
   kernel to as a consumer returns without touching the counter (`tensix_trisc_mask` check). The
   sharded compute kernel pops `ex2` itself in the single-stage reduce, but the host binds the
   reader as its consumer, so the reader's draining handle on `ex2` spun forever; it is resident now.
5. **POP after WAIT with no UNPACR in between** (TEN-4746) is ordered with `dummy_unpack(id)`
   before each standalone pop in `compute/layernorm.cpp`.

## Host-side changes

* `layernorm_op_multi_core.cpp`: reader / writer use
  `create_{reader,writer}_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true)`, as
  the other Quasar factories do.
* `sharded_layernorm_factory_helpers.cpp`: mainline leaves implicit DFB sync ON for the mcast
  reader / writer and documents that as required by the multicast. The receivers push their mcast
  input explicitly after the sender's semaphore, so it is not; and the implicit NOC path ignores
  `offset_bytes` (the partial gathers rely on it) and treats the sender's multicast *read* of the
  global buffer as a consumer pop although the compute kernel is the consumer. All three DM kernels
  now set `config_2xx.disable_dfb_implicit_sync_for_all`.
