# all_gather_rms_norm — LTX accuracy investigation: the Q/K head-split path is the sole bad actor

**Status:** Block-norm fused path is a **byte-perfect** drop-in for the wan baseline. The model accuracy
regression when fusing comes **only from the Q/K (head-split, `num_heads > 1`) norms.** Holding this finding
here so we don't re-litigate it.

Context: replacing the wan `wan_fused_rmsnorm_pre/post_allgather` path in LTX-2
(`models/tt_dit/layers/normalization.py::DistributedRMSNorm.forward`) with the fused
`ttnn.all_gather_rms_norm`, on `bh_2x4sp1tp0` (mesh (2,4), sp_axis=1, tp_axis=0, TP=2). A/B'd via env:
`LTX_FUSED_AGRMS=1` fuses the block norms (`num_heads==1`); `LTX_FUSED_AGRMS_QK=1` additionally fuses the
Q/K norms (`num_heads==16`). Proxy metric: output `.mp4` file size (good ≈ 3.2 MB, garbage ≈ 12 MB).

## The two fixes/finds, in order

### 1. Block-norm accumulation order (FIXED → bit-identical to wan)
The fused op and wan compute the same RMS-norm math but **accumulated x² in a different order**, and float
add is non-associative, so the fp32 sum differed by ~1 ulp and rounded to an adjacent bf16 value on ~0.1%
of elements (per-norm diff: max 0.0625, mean 5e-5, PCC 1.0). Root cause:
- **wan** (`fused_distributed_rmsnorm/.../compute/rmsnorm_pre_allgather.cpp`): computes x² per column-tile
  and **L1-accumulates all Wt tiles element-wise onto a single tile** (`pack_tile<true>` +
  `llk_pack_reconfig_l1_acc(1)`), *then* one `REDUCE_ROW`. → sums **across tiles first**, then columns.
- **ours** (`all_gather_rms_norm/.../compute/all_gather_rms_norm_compute.cpp::reduce_x2`, before the fix):
  stored the Wt x² tiles separately and let the reduce library sum **per-tile columns first, then tiles**.

Ruled out as *not* the difference: `use_legacy_rsqrt` (both `false`); scaler placement (wan scales
`1/total_W` in *post*, we scale in *pre*) — **bit-identical** because `total_W` is a power of two
(4096, 128) so the multiply is an exact exponent shift.

**Fix:** rewrote `reduce_x2` to mirror wan exactly (L1-accumulate x² into one tile, then one `REDUCE_ROW`).
Result on the model (`DUMP_NORM_DIR` dump, idx 0–11): **12/12 norm outputs bit-identical to wan**
(`max = 0.00000`, in and out), block *and* Q/K alike.

### 2. Bisect: only the Q/K path corrupts the model
Same prompt/seed, byte counts of the produced video:

| Run | env | video bytes |
|---|---|---|
| wan baseline | (none) | 3,220,993 |
| **block-fused, Q/K = wan** | `LTX_FUSED_AGRMS=1` | **3,220,993** ← byte-identical to wan |
| both fused | `LTX_FUSED_AGRMS=1 LTX_FUSED_AGRMS_QK=1` | 12,383,333 |

**Conclusion (the thing to hold): the block-norm fused path is a byte-perfect drop-in; the entire
divergence comes from the Q/K head-split fused path alone.**

## What is (and isn't) different about the Q/K norms
- **Reduction math is identical** to the block norm: `reduce_factor = shape[-1] * ring_size` (full gathered
  width) regardless of `num_heads`; wan pools all heads into one E[x²] too. So Q/K is *not* a per-head
  reduction — `num_heads` is layout-only.
- The **only** Q/K-specific code is the **writer's head-split output scatter**
  (`all_gather_rms_norm_writer.cpp`): `out_id = h*per_head_stride + gr*head_dim_tiles + e`, with
  `per_head_stride = m_tiles*head_dim_tiles`, `head_dim_tiles = Wt/num_heads`. For `num_heads==1` this
  collapses to a contiguous write (the block path). `max out_id = num_heads*m_tiles*head_dim_tiles - 1` =
  output tile count − 1 → **in-bounds** for the shapes checked.

## Open thread — RESOLVED at the op level: the op is byte-perfect in isolation (so it's a side-effect)

### 3. Op-level Q/K parity sweep PASSES across the entire LTX shape space (2026-06-04)
Ran `test_distributed_rmsnorm_fused.py::test_distributed_rmsnorm_fused_parity -k qk` on `(2,4)` —
**all 16 Q/K cases pass byte-perfect.** This builds the norm exactly as LTX does, feeds an *enumerated*
input (each width-tile a distinct int → any misplacement is exact, not hidden in noise), and asserts an
**element-wise allclose** (not just PCC, which stays ~1.0 through layout bugs) plus the on-device SPEC:
- `max_abs = 0.0000`, `allclose = True`, per-head[row0,col0] order **element-for-element identical** to wan.
- SPEC **identical** wan vs op: shape / padded_shape / dtype=BFLOAT16 / layout=TILE /
  mem=INTERLEAVED,DRAM — for every (dim∈{4096,2048}, axis∈{sp1tp0 TP2/H16, sp0tp1 TP4/H8},
  seq∈{512,4k,16k}). **`seq16k` exercises the multi-row-per-worker path** (4096 seq/device = 128
  tile-rows > worker cores → `gr++` across rows + the `done` back-pressure semaphore) — still byte-perfect.

This matches the static read of the kernels: the Q/K writer scatter formula is *identical* to wan
(`writer.cpp:121-122` ≡ `rms_post_allgather_writer.cpp:34-47`), `m_tiles` is the global tile-row count
(`program_factory.cpp:144` → `:353`), and both ops default output mem_config to `input.memory_config()`.
So at the op boundary the Q/K fused output is provably identical to wan — values, ordering, shape, padding,
layout, dtype.

### Why the model fingerprint can never catch this
`DistributedRMSNorm._dump_norm` (incl. the `poschk` position-sensitive checksum) compares the **norm output**
gathered to torch with the same composer for both paths. Since that output is byte-identical, sum/sumsq AND
poschk all match — exactly the observed "bit-identical through idx 0–280". The fingerprint only ever watches
*norm outputs*; it is structurally blind to a side-effect on a **downstream non-norm tensor**.

### Conclusion → it is a side-effect (option 2), not a wrong norm output
With the op byte-perfect for arbitrary input (enumerated *and* random), a late data-dependent norm divergence
(option 1) is implausible. The 12 MB video must come from the Q/K op's *execution* corrupting something the
norm-output comparison can't see. The only num_heads-dependent runtime behavior is the writer's DRAM write
**pattern**: block writes `Wt` contiguous pages/row; Q/K writes the same `Wt` pages **scattered** across the
16 head regions (all in-bounds) — 16 strided NoC transactions instead of one contiguous burst, which can
shift dispatch/teardown timing.

### 4. Neighbor-tensor diff in the model: NOT bad input — the *following* ring-joint SDPA is corrupted (2026-06-04)
Instrumented `attention_ltx.py` with an env-gated `DUMP_ATTN_FP` fingerprint (sum/sumsq/poschk, gathered
to torch) at the tensors *downstream* of the Q/K norm, and A/B'd wan vs `LTX_FUSED_AGRMS_QK=1` on the real
model (`bh_2x4sp1tp0`, reduced 256x256x9 for speed). First self-attention block:

| idx | tag | wan vs fused |
|---|---|---|
| 0 | norm_q | identical |
| 1 | norm_k | identical |
| 2 | rope_q | identical |
| 3 | rope_k | identical |
| 4 | **v_in** | **identical** (byte-exact) |
| 5 | **sdpa_out** | **DIVERGE ~14% sum / ~21% poschk** |

**All three SDPA inputs (q, k, v) are byte-identical, yet `sdpa_out` is wrong → it is NOT bad input.**
At `bh_2x4sp1tp0` (sp=4, blackhole, no mask) the first self-attn SDPA is
`ring_joint_scaled_dot_product_attention` — itself a **fabric/CCL op**. The fused norm op leaves
fabric/EDM/global-semaphore state that this following fabric op inherits → the SDPA's own fabric
all-gather of K/V (or its EDM connection) is corrupted. This is the **improper-teardown** branch
(cf. the earlier `[[project_agrms_mux_corruption]]` mux↔EDM teardown). It explains QK-only cleanly:
the Q/K-norm site is the *only* norm site immediately followed by a fabric op (block norms feed a matmul).

The op-level parity test misses this twice over: its input is **constant across sequence rows** (so a row
bug is invisible) AND it runs the op **in isolation** with no following fabric op (so a teardown side-effect
can't manifest). The model neighbor-diff on real, row-varying data is the honest test.

Next (localize inside the ring SDPA + fix the fused op's fabric teardown):
- Dump the ring SDPA's `persistent_output_buffer_k/v` (its fabric-gathered K/V) wan-vs-fused. If the
  *gathered* K/V is corrupt while the *input* k/v is clean → the corruption is the ring SDPA's own fabric
  gather running on the poisoned EDM.
- Cheap discriminator: a device sync / fabric barrier between the fused norm and the ring SDPA — does it
  clear `sdpa_out`? (in-flight vs persistent EDM state.)
- Inspect the fused op's gather-core teardown: `close_connections(fabric_connection)` + barrier ordering
  in `all_gather_rms_norm_gather.cpp`, and the global-semaphore lifecycle.
- If needed, DPRINT in the ring-joint SDPA gather/reader kernel to see the first corrupt tile.

### 5. ROOT CAUSE (high confidence): the gather kernel closes the fabric without draining the EDM (2026-06-04)
Confirmed via two independent angles:

**(a) Sync test rules out "in-flight/ordering".** Forcing `ttnn.synchronize_device` between the fused QK
norm and the ring SDPA leaves `sdpa_out` diverging by the *identical* ~14%/21%. A completion-sync doesn't
re-init the EDM, so a dirty EDM left by teardown survives it → persistent fabric-state corruption.

**(b) wan's CCL all_gather drains before close; the fused gather does not.** Compared
`all_gather_async/.../llama_shapes_sharded_writer.cpp` (what the wan DistributedRMSNorm uses) to
`all_gather_rms_norm_gather.cpp`:
- **wan teardown**: per-connection `wait_for_empty_write_slot()` + `send_payload_flush_blocking_from_address()`
  → `fabric_connection.close()` → `noc_async_full_barrier()`. Plus a cross-device `barrier_sem` rendezvous
  at *start* (`use_barrier_sem`).
- **fused gather teardown** (`all_gather_rms_norm_gather.cpp:102-116`): `noc_async_writes_flushed()` (LOCAL
  NoC flush only — does NOT wait for the EDM to forward/consume the packet) → `close_connections()`
  immediately → `noc_async_write_barrier()` (write-only). **No `wait_for_empty_write_slot` after the last
  send, no flush_blocking, no full barrier.** The `barrier_semaphore` the program factory allocates is
  explicitly UNUSED (`(void)barrier_semaphore`), so there's no device rendezvous either.

The linear experimental fabric API (`fabric_multicast_*_with_state`) does `wait_for_empty_write_slot()`
*before* each send, but nothing drains the channel *after* the final send → `close_connections` can run
while the last packets are still in the EDM, leaving its write-pointer/credit state mid-stream. The next op
to open a connection on that routing plane — the ring-joint SDPA — inherits the dirty EDM and its K/V
fabric-gather is corrupted (clean q/k/v inputs, wrong output).

**Proposed fix:** before `close_connections`, drain every active sender —
`fabric_connection.for_each([](Sender& s, uint32_t, uint32_t){ s.wait_for_empty_write_slot(); })` (mirror
wan), and consider `noc_async_full_barrier()` instead of write-only, and/or wiring up the unused
`barrier_semaphore` for a start/end device rendezvous. **Verify:** re-run the model fused-QK neighbor diff;
`sdpa_out` (idx 5) should become byte-identical to wan.

### 6. Drain-before-close fix does NOT help — and the divergence is BIT-IDENTICAL across variants → deterministic state, not a race (2026-06-04)
Added wan's drain to the gather kernel: `fabric_connection.for_each(... wait_for_empty_write_slot())` +
`noc_async_full_barrier()` before/after `close_connections` (all_gather_rms_norm_gather.cpp).
**Cleared the gather kernel cache to force a real recompile** (first attempt was a stale cache hit — 937/939
hits, zero gather compiles; the recompile is also why the next run hit the 300s pytest timeout, but it
dumped 4280 lines first). Result: `sdpa_out` still diverges by the **exact same** `sum rel=1.38e-01`,
`sumsq 3.48e-02`, `poschk 2.09e-01`.

**The divergence is bit-identical across no-fix / `synchronize_device` / drain-fix.** A timing race would
vary run-to-run; bit-identical means the corruption is **deterministic** — the fused op leaves a *fixed
wrong state* that the ring-joint SDPA inherits, OR there's a deterministic resource collision. This
**reframes** finds #4/#5: it is NOT an EDM in-flight/drain race. Candidates for the deterministic shared
state between the fused norm op and the following ring SDPA:
  1. a **global semaphore** left at a fixed non-zero value (the op's `_agrms_sem` / out_ready / the unused
     `barrier_semaphore`) aliasing the SDPA's `ag_ping_pong_semaphore`;
  2. the EDM **write-counter/credit** persisted by `close_start` and inherited (not reset) by the SDPA's
     `open` on the same routing plane → fixed slot offset in the gather;
  3. a **persistent buffer** overlap (the designated-gather relay region at `allocator L1 base`, or
     `gathered_stats`/`preallocated_stats`) with the SDPA's ping-pong K/V buffer.

(The drain change is harmless+more-correct so it's left in, but it is NOT the fix.)

Next: instrument the ring-joint SDPA directly (the original ask) — dump its gathered K/V
(`persistent_output_buffer_k/v`) and DPRINT the gather's semaphore/credit at entry — to see WHERE its output
goes wrong; and audit the fused op's global-semaphore / relay-L1 / gathered-stats allocation for a
deterministic collision with the ccl_manager's ping-pong SDPA resources.

### 7. DPRINT in the ring-SDPA sync: NOT a premature/stale-semaphore read (2026-06-04)
DPRINT'd the ring-joint SDPA's gather sync (`fused_op_receiver.hpp::get_next_ring_id_and_sync`, printing
the AllGather signal-semaphore `have` vs the `need` threshold before `noc_semaphore_wait_min`). First
(corrupted) self-attn, every reader core:
```
dir=0 need=0 have=0 PREMATURE   <- ring_iter 0 reads LOCAL K/V (need=0, benign)
dir=0 need=2 have=0 wait        <- ring_iter 1 properly WAITS for the gather (have=0 < need=2)
```
The semaphore starts at 0 and correctly waits for both ring writes → **the gather is properly
synchronized; the stale/aliased-semaphore hypothesis (find #6 candidate 1) is RULED OUT.** So the
corruption is in the gathered K/V *data* (fabric all-gather transfer, despite clean source + correct sync)
or in the SDPA compute. Next: dump the ring-SDPA `persistent_output_buffer_k/v` (gathered K/V) wan-vs-fused
— corrupt gathered K/V ⇒ fabric transfer; clean ⇒ compute. (DPRINT left in `fused_op_receiver.hpp`, inert
unless `TT_METAL_DPRINT_CORES` is set.)

### 8. DEFINITIVE LOCALIZATION: the ring-SDPA fabric all-gather writes K/V to WRONG ADDRESSES (2026-06-04)
Dumped the ring-joint SDPA's gathered K/V (`persistent_output_buffer_k/v`) wan-vs-fused, first self-attn:

| tensor | wan | fused |
|---|---|---|
| `gathered_k` | sum=2.5e3, max=7.56, min=-6.43 (normal normed K) | sum=-3.2e4, **max=69.5, min=-50.5** |
| `gathered_v` | sum=2.4e5, max=71, min=-50.5 (normal V) | sum=**1.6e9**, sumsq=**6.3e17**, **max=3.96e8** |

Source k/v (`norm_k`,`v_in`) are byte-identical and sync is correct (find #7), yet the gathered buffers are
**garbage**: fused `gathered_k` carries `min=-50.5` — exactly wan's `gathered_v` min — i.e. **V-range data bled
into the K buffer**, and `gathered_v` exploded to ~4e8 (uninitialized/mis-addressed memory). This is
**K/V cross-contamination + mis-addressed writes**: the ring-SDPA all-gather's *payload* writes land at the
wrong addresses when the fused QK norm precedes it. Deterministic.

So the bespoke `all_gather_rms_norm` leaves **fabric routing / packet-header / EDM-channel state** that the
next fabric op (the ring-SDPA all-gather) inherits, mis-addressing its writes. The wan baseline uses the
standard CCL `all_gather_async` (clean teardown) so the SDPA gather is clean. The partial drain fix (find #6:
`wait_for_empty_write_slot` + `noc_async_full_barrier`) was INSUFFICIENT — the leaked state is not just
in-flight packets.

Prime suspects for the leaked state (next to investigate/fix):
- **PacketHeaderPool**: the gather kernel `set_state`s packet headers (dst addr / payload size) via
  `fabric_multicast_*_with_state` from `MEM_PACKET_HEADER_POOL_BASE`; if the SDPA's fabric kernels reuse
  stale header slots the dst addresses are wrong.
- The **unused `barrier_semaphore`**: wan does a start/end device rendezvous (`use_barrier_sem`); the fused
  op allocates it but never uses it → no cross-device fabric quiesce.
- EDM **connection config** not reset for the next opener on the same routing plane.
Fix direction: replicate wan's FULL teardown/handshake (start barrier + flush_blocking + full barrier), or
ensure the packet-header pool / EDM channel is reset before the op completes.

### Earlier framing (now superseded by find #4 above):
1. **Neighbor-tensor diff** — dump the tensor *downstream* of the Q/K norm (RoPE output / SDPA input) for wan
   vs fused-QK and diff; a side-effect corrupts a neighbor, not the norm output.
2. **Bounds/sentinel guard** in the writer (`ASSERT(out_id < num_heads*m_tiles*head_dim_tiles)` + sentinel-fill
   the output buffer) to rule in/out an out-of-output write.
3. **Concurrency** — re-run fused-QK; if a teardown/fabric race, narrow the gather core's
   `close_connections`/barrier ordering vs the scattered writer drain. (Note: this op uses fabric, so
   `TT_METAL_SLOW_DISPATCH_MODE=1` may not be viable — verify before relying on it.)

## How to reproduce
```bash
# block-only (matches wan byte-for-byte):
LTX_FUSED_AGRMS=1 SEED=0 OUTPUT_PATH=/tmp/blockonly.mp4 \
  pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx_fast_av.py -k bh_2x4sp1tp0 -s
# both (garbage video, isolates Q/K):
LTX_FUSED_AGRMS=1 LTX_FUSED_AGRMS_QK=1 SEED=0 OUTPUT_PATH=/tmp/both.mp4 \
  pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx_fast_av.py -k bh_2x4sp1tp0 -s
# per-norm fingerprint (compare wan vs fused, find first divergence):
DUMP_NORM_FP=/tmp/fp.txt ...  # + raise --timeout; diff fp_wan.txt vs fp_fused.txt by (idx,tag)
```
`tt-smi -r` before runs; the first JIT run may time out (re-run uses the cache).
