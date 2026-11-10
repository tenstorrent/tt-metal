# Lego: fused programs composed from op blocks

Lego builds one fused program per device out of *blocks*: logical parts of existing ttnn ops (a matmul's weight
reader, a reduce-scatter's writer, an RMSNorm's statistics compute), ported with as few changes as possible. The host
code in this package places blocks on cores and RISCs, allocates their circular buffers, semaphores and L1, and wires
their hand-offs. The generated kernels contain no logic of their own: each one is a list of blocks.
`TPBoundary` (`tp_boundary.py`) is the first op built this way: the tensor-parallel layer boundary
`matmul -> reduce-scatter + residual -> all-gather -> RMSNorm -> matmul` as a single `ttnn.generic_op` per device.

- [Blocks](#blocks)
- [Composer](#composer)
- [TPBoundary](#tpboundary)
- [Constraints](#constraints)
- [Tests and profiling](#tests-and-profiling)

## Blocks

A block is a class template in `ttnn/cpp/ttnn/operations/experimental/lego/kernels`:

```cpp
template <typename A> struct Block {
    void init();      // read args, build accessors and fabric connections
    void run();       // the work
    void teardown();  // barriers, connection close, semaphore resets
};
```

`A` is a generated struct of `static constexpr` members: the block's compile-time values and the offsets of its
slices of the kernel's runtime args, common runtime args and tensor-accessor compile-time args. Blocks read their
own slice through `lego::rt<A>`, `lego::crt<A>` and `lego::rt_index<A>` (`kernels/lego.hpp`), never by absolute
index, so any number of blocks can share a kernel.

| Blocks | Ported from | Hooks added for composition |
|---|---|---|
| `mm/in0.hpp`, `mm/in1.hpp` (`lego::mm::In0`, `In1`) | `minimal_matmul` dataflow kernels | per-core N range, K read in all-gather arrival order with L1-counter waits, row-scale read, M batches with per-batch signals |
| `mm/compute.hpp` (`lego::mm::Compute`) | `minimal_matmul` compute (SwiGLU in the K loop) | row-scale epilogue for plain and SwiGLU outputs (`mm/swiglu_scaled_sfpu.hpp`) |
| `rs/reader.hpp`, `rs/writer.hpp`, `rs/reduction.hpp` (`lego::rs::*`) | `reduce_scatter_minimal_async`, ring | residual add in the last step, second output (the all-gather's slot), output-ready signals, batched traversals, mux or direct fabric connection |
| `ag/reader.hpp`, `ag/writer.hpp` (`lego::ag::*`) | `all_gather_async`, minimal default | slice or row-group arrival signals, optional local-slice write, mux or direct fabric connection |
| `fabric/mux.hpp`, `fabric/mux_client.hpp` | `tt_fabric_mux.cpp` | program-local L1 for the mux |
| `norm/rms_dataflow.hpp`, `norm/rms_compute.hpp` (`lego::norm::*`) | `rmsnorm_distributed` pre- and post-all-gather | row strides (a part of a row), one read barrier per row part |
| `common/tile_writer.hpp` | tile writers of the norm ops | row strides |
| `sync/signal.hpp`, `sync/wait.hpp`, `sync/barrier.hpp` | | stage hand-offs: semaphore increments on target cores, waits, barriers |

## Composer

`compose.py`:

- `LegoProgram.place(cores, Block(...))` puts a block on one RISC (`"dm0"`, `"dm1"` or `"compute"`) of each core.
  On a core, a RISC runs its blocks' `init()`s, then their `run()`s, then their `teardown()`s, in placement order.
- `LegoProgram.cb(...)` and `LegoProgram.sem(...)` allocate per core: blocks on one core never collide, and two blocks
  share a CB only when the caller passes the same index to both. A CB can live in an `Arena` (one L1 tensor sharded
  over the cores) at a byte offset, so blocks whose lifetimes do not overlap reuse the same L1.
- `LegoProgram.set_noc(cores, risc, noc)` sets a RISC's NoC per core.
- Cores whose RISCs carry the same block lists share one generated kernel. The first compute block of a kernel runs
  the hardware start-up.
- `LegoProgram.descriptor()` returns a `ttnn.ProgramDescriptor`. `mesh_descriptor()` combines one per device for
  `ttnn.generic_op`.

The `place_*` helpers build blocks for one op part and return the cores or plan they used:
`mm.place_matmul`, `ccl.place_reduce_scatter`, `ccl.place_all_gather`, `ccl.place_mux`,
`norm.place_rms_stats`, `norm.place_rms_scale`, `norm.place_rmsnorm`, and the hand-offs in `sync.py`
(`place_handoff_multi`: consumers wait until every producer core has passed a point; `place_local_handoff`: one RISC
waits for another on the same core).

## TPBoundary

`TPBoundary(mesh, eps, boundary, S, n_out, cfg=None)` returns `(h, out)` from `op(x, inp, w1, w2)`, per device:

- `x` `[1,1,S,4096/ring]`: residual stream slice.
- `inp` `[1,1,S,K/ring]`: input of the row-parallel matmul.
- `w1` `[K/ring, 4096]` and `w2` `[4096, N]`: the model's prefill weight tensors, unchanged (block-float, DRAM
  interleaved). With SwiGLU (`boundary="A"`) `w2` interleaves gate and up tile by tile.
- `h` = `x + reduce_scatter(inp @ w1)`, the new residual.
- `out` = `rmsnorm(all_gather(h)) @ w2`, with SwiGLU for boundary A.

The RMSNorm weight is folded into `w2`, as the model does. The norm is applied as a per-row scale in MM2's
epilogue (`(r x) W = r (x W)`). MM2 therefore consumes the gathered, unnormalized `H` slice by slice as the
all-gather delivers it, while the statistics run beside it.

Core map on a 12x10 Blackhole grid:

| Rows | Prefill, S >= 512 | Prefill, S < 512 | Decode (S = 32) |
|---|---|---|---|
| 0-1 (aux: HiFi4, fp32 accumulation) | RS with mux (5 workers per link and direction), AG with mux (2), statistics + statistics all-gather | same | RS and AG with 1 direct worker per link and direction; statistics from the gathered `H` |
| 2-9 (matmul: LoFi) | MM1, then MM2 on 12x8; MM2's weights reuse MM1's arena bytes, 2 K blocks deep | MM1 split by N over four 12x2 grids; MM2 on 12x4 (two 12x4 grids split by N for A); MM2's weights prefetched as deep as the arena allows | MM1 split by N over eight 12x1 grids; MM2 over the first five (A) or four (B) 12x1 grids with all its weights in their own L1 |

`prefill_config` and `decode_config` hold these settings; `BoundaryConfig` documents every field.

Stage order on each device: MM1 writes `P` → the RS reduces `P` over the ring, adds `x`, writes `h` and this
device's slot of `H`, and signals MM2's injectors → the AG fills `H` and signals each injector as its rows of a slice
land → MM2 runs its K blocks in arrival order and, after its first N block, reads the row scales. The RS muxes
close before the AG muxes connect, and the statistics all-gather starts after the AG has released the fabric.

## Constraints

- Compute config is per kernel. Blocks that need different math fidelity or fp32 accumulation must sit on different
  cores. The aux rows run the model's norm config (HiFi4, fp32); the matmul rows run LoFi.
- A core has 16 semaphores. The statistics all-gather therefore runs on aux cores the RS and AG do not use.
- An N split must cut at whole per-core chunks (`place_matmul(n_range=...)`). A core whose range starts past its
  end would underflow the in1 reader's block size.
- Cached programs keep the arena buffer alive (it backs CBs). Clear the program cache before freeing an op whose
  successor needs that L1.

## Tests and profiling

- `tests/ttnn/unit_tests/operations/experimental/lego/test_tp_boundary.py` runs both boundaries at S = 32 (decode),
  128 and 1024 on a 1x4 ring (`FABRIC_1D_RING`) against fp32 torch on the same quantized weights.
- `LEGO_ZONES=1` in the host environment compiles a device-profiler zone around each block's `run()` (the
  `LEGO_ZONE` macro in `kernels/lego.hpp`), for per-block timelines with the device profiler.
