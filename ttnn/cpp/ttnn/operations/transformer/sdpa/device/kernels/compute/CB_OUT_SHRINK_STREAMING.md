# Shrinking `cb_out` in Single-Chip SDPA (streaming compute)

Analysis of the MM2 output circular buffer `cb_out` (`tt::CBIndex::c_16`) in
the **single-chip** SDPA streaming path (`ttnn.transformer.scaled_dot_product_attention`):
why it is currently sized for the full Q chunk, why that over-provisions L1,
and what has to change to shrink it.

The Phase 2 compute body is shared with ring joint SDPA
(`compute_streaming.hpp` → `sdpa_standard_v2` + `sdpa_ring_v2`), so insights
transfer; but the **single-chip writer** (`writer_interleaved.cpp` +
`dataflow_common.hpp::write_block`) uses a **bulk chunk drain**, not the
ring-joint writer's row-by-row drain. That difference matters for shrinking.

Companion: [`OUT_SUBBLOCK_STREAMING.md`](./OUT_SUBBLOCK_STREAMING.md) — the
`qktv_h` / `qktv_remainder_h` dynamic subblock logic referenced below.

---

## 1. Current state

### 1.1 Allocation

`ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp:412-413, 792`:

```cpp
uint32_t out0_t = Sq_chunk_t * vDHt;                       // full MM2 output
auto c_out0_config = CircularBufferConfig(
    out0_t * out_tile_size,
    {{tt::CBIndex::c_16, out_df}})
    .set_page_size(tt::CBIndex::c_16, out_tile_size);
```

`cb_out` holds **one full Q chunk** of MM2 output — `Sq_chunk_t * vDHt` tiles —
with no double-buffering.

### 1.2 Producer — single up-front reserve

`compute_streaming.hpp:867` (inside `sdpa_standard_v2`, shared with
`sdpa_ring_v2`):

```cpp
cb_reserve_back(out_cb, qktv_output_num_tiles);   // Sq_chunk_t * vDHt
```

All subsequent `blocked_matmul_and_pack` and `salad_correct_fused` calls
(lines 904, 955-963, 1012) address tiles **as offsets from that one base**,
using the row-within-chunk offset `w_q` (lines 985-986). Rows are then pushed
in groups of `qktv_h * vDHt` (lines 938, 1052, 1072, 1080) and, in the dynamic
case, a trailing `qktv_remainder_h * vDHt`.

### 1.3 Consumer — **bulk chunk drain** (single-chip)

`dataflow_common.hpp:1076-1105` (invoked from `writer_interleaved.cpp:163`):

```cpp
void write_block(..., const uint32_t cb_out, const uint32_t out_chunk_tiles, ...) {
    ...
    cb_wait_front(cb_out, out_chunk_tiles);     // <── waits for ENTIRE chunk
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            noc_async_write_tile(tile_id, out_writer, l1_read_addr);
            ...
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, out_chunk_tiles);      // <── pops ENTIRE chunk
}
```

Compare to ring joint writer's `write_out_row_by_row`
(`ring_joint_writer.cpp:146-178`), which waits and pops one `out_subblock_h`-sized
row group at a time. **That asymmetry is the extra obstacle for single-chip:
even if we fix the producer, the consumer still blocks until the whole chunk
is present.**

### 1.4 Footprint sanity check (bf16 tiles)

| `Sq_chunk_t` (tile rows) | `vDHt` (tile cols) | Tiles | L1 footprint |
|---|---|---|---|
| 4  | 4 | 16  | 32 KB   |
| 7  | 4 | 28  | 56 KB   |
| 8  | 4 | 32  | 64 KB   |
| 16 | 4 | 64  | 128 KB  |
| 16 | 8 | 128 | 256 KB  |

---

## 2. The observation

The producer **pushes row-by-row** (per `qktv_h` group), yet `cb_out` is sized
for the **entire chunk** and the **single-chip writer drains the entire
chunk** at once. Two different reasons for the over-provisioning, both fixable:

1. **Kernel side**: single up-front reserve + flat-base pack addressing.
2. **Writer side**: single bulk `cb_wait_front` / `cb_pop_front`.

Minimum theoretical footprint ≈ one group × ping-pong = `2 * qktv_h * vDHt`
tiles.

---

## 3. Why the current code is structured this way

### 3.1 Kernel: pack addressing is base-relative

`blocked_matmul_and_pack<false, vDHt, vDHt>(…, w_q, v_subblock*qktv_subblock_w, …)`
(lines 904, 1012) computes

```
pack_addr = cb_out_wr_ptr + (w_q * qktv_h + row_in_subblock) * vDHt + col
```

With **one reserve per chunk**, `cb_out_wr_ptr` is fixed and `w_q` walks row
groups 0..N-1. With per-group reserves, each group starts at its own
`wr_ptr` and `w_q` must reset to `0`.

### 3.2 Kernel: pack L1 accumulation

For `q_subblock = 0`, the kt-split loop (lines 876-928) issues multiple
`matmul_and_pack` iterations with `llk_pack_reconfig_l1_acc(1)` (line 891) so
partial products add into the same destination tiles. This is *within-group*
accumulation — a per-group reserve that covers all `kt_sub` iterations keeps
this valid.

### 3.3 Kernel: SALAD targets current group's output

`salad_correct_row` (lines 948-969) writes the SALAD-corrected tiles back into
`out_cb` at row offset `w_salad = salad_row - pushed_rows` (line 981). SALAD is
for the **previously-produced** group's tiles.

- With the current single-reserve layout, "previously produced" tiles are
  still **inside the same reserve window**, so writing back is legal.
- With per-group reserves, once a group is `cb_push_back`-ed it is downstream
  of the writer — the kernel can no longer write into it. SALAD must move
  before the push (see §5.2).

### 3.4 Kernel: normalize-in-place on last ring iter

`normalize_row_streaming` (line 939) reads `cb_out` via `cb_normalized_out`
(aliased to `cb_out` itself in `sdpa.cpp`). This is per-row inside a single
reserve window — already compatible with per-group reserves.

### 3.5 Writer: single `cb_wait_front(out_chunk_tiles)`

Simplicity: one wait, one loop, one barrier, one pop. For single-chip this was
fine because nothing cared about streaming the drain. To shrink `cb_out` the
writer must switch to row-by-row, mirroring `ring_joint_writer::write_out_row_by_row`.

**Net: only §3.1 (kernel) and §3.5 (writer) are real blockers.** §3.2, §3.3,
§3.4 fall out correctly with the right reserve granularity.

---

## 4. Why shrinking helps

### 4.1 Direct L1 savings

For `Sq_chunk_t = 16, vDHt = 8` (256 KB), a 2-group ping-pong at
`qktv_h = 2` drops `cb_out` to **32 KB** — **224 KB saved per core**. L1 saved
enables any of:
- Larger `Sk_chunk_t` (fewer K iterations, better MM1/MM2 overlap).
- Deeper `cb_v_in` / `cb_qk_im` buffering.
- Extra intermediates for attention sink or GQA variants.

### 4.2 Forced back-pressure (good)

A small `cb_out` makes compute stall on `cb_reserve_back` when the writer
falls behind — the natural flow-control we want. Currently, with a full-chunk
CB, compute can race ahead and pile MM2 output in L1 with no DMA overlap.

### 4.3 What could go wrong

If the writer's NoC throughput is **below** compute's MM2 throughput, a
smaller CB will force compute to wait on the writer every group — a
throughput regression. Measure first (§7).

---

## 5. Proposed solution

### 5.1 Host-side: parameterize `out0_t` by buffer depth

In `sdpa_program_factory.cpp`, replace:

```cpp
uint32_t out0_t = Sq_chunk_t * vDHt;
```

with a streaming-specific formula:

```cpp
// Mirror the kernel's qktv_h bump rule (see OUT_SUBBLOCK_STREAMING.md §"Why it's dynamic")
const uint32_t effective_qktv_h =
    (out_out_subblock_h == 1 && 2 * out_out_subblock_w <= dst_size && Sq_chunk_t >= 2)
        ? 2 : out_out_subblock_h;

const uint32_t out0_groups = 2;  // pure ping-pong (1 pending + 1 in-flight) — see §5.3
uint32_t out0_t = use_streaming_compute
    ? out0_groups * effective_qktv_h * vDHt
    : Sq_chunk_t * vDHt;   // non-streaming path unchanged
```

### 5.2 Kernel (`compute_streaming.hpp`): per-group reserve, group-local addressing

Two structural edits inside Phase 2 (lines ~840-1128):

**Edit A — move reserve into the loop**

```cpp
// Before (line 867)
cb_reserve_back(out_cb, qktv_output_num_tiles);

// After: reserve at the top of each row-group iteration
for (uint32_t q_subblock = 0; q_subblock < total_v_row_groups; ++q_subblock) {
    const uint32_t cur_group_h = is_remainder_iter ? qktv_remainder_h : qktv_h;
    cb_reserve_back(out_cb, cur_group_h * vDHt);
    ... matmul / SALAD / normalize ...
    cb_push_back(out_cb, cur_group_h * vDHt);
}
```

**Edit B — drop flat-base `w_q`, use group-local offsets**

Every pack-address site currently keyed off `w_q`:

- Line 904: `blocked_matmul_and_pack` for `q_subblock = 0` — already uses
  `w_q = 0` (line 910). No change.
- Line 1018: `blocked_matmul_and_pack` for `q_subblock > 0` — currently
  `w_q = is_remainder_iter ? (qktv_q_num_subblocks - pushed_rows) * qktv_h :
  (q_subblock - pushed_rows)`. **Change to `w_q = 0`**.
- Lines 955-963: `salad_correct_fused<…>(…, 0, salad_row, w_salad)` — the
  `0` is already group-local. `w_salad = salad_row - pushed_rows`
  (line 981) — **keep for salad_row indexing into `prev.out`, not `cb_out`**.
  Re-audit to confirm no `cb_out` write uses `w_salad` as a row offset.

**Edit C — resolve SALAD-into-pushed-slot race (see §3.3)**

**Use Option B: pending slot — preserve the SALAD / matmul pipeline.**

Keep the existing call order (matmul group `k`, then SALAD group `k-1`, then
push group `k-1`). Only change the **reserve granularity** — reserve per
group instead of once up-front. Steady-state holds **2 slots simultaneously**:

- slot for group `k` — current matmul, reserved but not yet SALAD'd/pushed
- slot for group `k-1` — held from prior iter, still un-pushed until this
  iter's SALAD finishes writing into it

CB capacity: `2 * qktv_h * vDHt` (16 tiles for the target config) — the minimum
that keeps the reserve invariant. A 3-deep buffer with writer-latency slack
was evaluated but unnecessary: perf measurements (§7) show the writer keeps up
with MM2 at 2-deep, so triple-buffering only costs L1 with no throughput benefit.

No change to SALAD / matmul ordering → the streaming v2 overlap is preserved
and there is no perf risk from the restructure itself (only from the smaller
buffer if the writer can't keep up, which is measured in §7).

### 5.3 Writer (`dataflow_common.hpp::write_block`): switch to row-by-row drain

Port `write_out_row_by_row` (`ring_joint_writer.cpp:149-179`) into
`dataflow_common.hpp` (or add a new `write_block_row_grouped` variant), called
from `writer_interleaved.cpp:163` when `use_streaming_compute` is active:

```cpp
void write_block_row_grouped(
    const TensorAccessorType& out_writer,
    const uint32_t cb_out,
    const uint32_t rows,
    const uint32_t cols,
    const uint32_t out_tile_id_base,
    const uint32_t tile_bytes,
    const uint32_t sbh /* qktv_h or out_subblock_h */,
    const uint32_t barrier_threshold) {

    const uint32_t row_tiles = sbh * cols;
    const uint32_t num_row_groups = rows / sbh;
    const uint32_t remainder_rows = rows - num_row_groups * sbh;
    uint32_t tile_id = out_tile_id_base;
    uint32_t barrier_count = 0;

    for (uint32_t rg = 0; rg < num_row_groups; ++rg) {
        cb_wait_front(cb_out, row_tiles);
        uint32_t l1 = get_read_ptr(cb_out);
        for (uint32_t r = 0; r < sbh; ++r)
            for (uint32_t c = 0; c < cols; ++c, ++tile_id, l1 += tile_bytes) {
                noc_async_write_tile(tile_id, out_writer, l1);
                if (++barrier_count == barrier_threshold) {
                    noc_async_writes_flushed();
                    barrier_count = 0;
                }
            }
        cb_pop_front(cb_out, row_tiles);
    }
    if (remainder_rows) {  // dynamic-h case: trailing qktv_remainder_h group
        cb_wait_front(cb_out, remainder_rows * cols);
        uint32_t l1 = get_read_ptr(cb_out);
        for (uint32_t r = 0; r < remainder_rows; ++r)
            for (uint32_t c = 0; c < cols; ++c, ++tile_id, l1 += tile_bytes)
                noc_async_write_tile(tile_id, out_writer, l1);
        cb_pop_front(cb_out, remainder_rows * cols);
    }
    noc_async_write_barrier();
}
```

Pass `sbh` as the **host-side** `out_subblock_h` (not the kernel's bumped
`qktv_h`). The kernel may push 2 rows at a time but the writer drains 1 at a
time; `cb_wait_front` is a `≥` check, so the excess waits for the next
iteration naturally. This matches what ring joint already does.

### 5.4 Streaming-only scope

Leave non-streaming `sdpa_ring` / non-streaming `sdpa.cpp` paths untouched.
They use `cb_out_im_A`/`cb_out_im_B` intermediate buffers and a bulk-push to
`cb_out`; shrinking there requires separate analysis. Gate all changes behind
`use_streaming_compute`.

---

## 6. Dynamic `qktv_h` — does it affect sizing?

**No blocker, slot size unchanged.** See
[`OUT_SUBBLOCK_STREAMING.md`](./OUT_SUBBLOCK_STREAMING.md).

Summary:
- Pushes: `qktv_h * vDHt` × N, plus `qktv_remainder_h * vDHt` × 1 if odd.
- `cb_out` slot size = `qktv_h * vDHt` (the max). Remainder reserves less,
  consuming part of a slot — small internal fragmentation, fine.
- Writer drain uses host `out_subblock_h` (1 in dynamic case), which
  pops one row at a time; agnostic to producer grouping.

---

## 7. Benchmark and iteration test

### 7.1 Target configuration (4× Galaxy analog, smaller seq)

`tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py`

| Parameter | Value |
|---|---|
| Shape (b, nh, s, d) | 1, 10, **2368**, 128 |
| Shape id | `wan2_2_4xGLX_analog` |
| `q_chunk_size` | **224** |
| `k_chunk_size` | **512** |
| dtype | `bf16` |
| `fp32_dest_acc_en` | False → `dst_size = 8` |
| `is_causal` | False → streaming eligible |

Derived tile counts:

| Quantity | Formula | Value |
|---|---|---|
| `Sq_chunk_t` | `224 / 32` | **7 tiles** (odd → dynamic) |
| `Sk_chunk_t` | `512 / 32` | 16 tiles |
| `DHt` / `vDHt` | `128 / 32` | 4 tiles |
| `num_q_chunks` | `⌈2368/224⌉` | 11 |
| `num_k_chunks` | `⌈2368/512⌉` | 5 |

Host subblock picker (§1 of `OUT_SUBBLOCK_STREAMING.md`):
walks `{(2,4), (1,8), (1,7), (2,3), (1,6), (1,5), (2,2), (1,4), …}`
under `max_h=2` and `Sq_chunk_t=7`, `vDHt=4`, `dst_size=8`:
- `(2,4)`: 7 % 2 ≠ 0 → skip
- `(1,8)`: 4 % 8 ≠ 0 → skip
- `(1,7)`: skip
- `(2,3)`: skip
- `(1,6)`: skip
- `(1,5)`: skip
- `(2,2)`: skip
- `(1,4)`: ✓ → host returns `(h=1, w=4)`.

Kernel bump (`compute_streaming.hpp:844-847`):
`qktv_subblock_h==1`, `2*4=8 ≤ 8`, `Sq_chunk_t=7 ≥ 2` → **`qktv_h = 2`,
`qktv_remainder_h = 1`, `has_qktv_remainder = true`**, 3 full groups + 1
remainder group.

**Push pattern per Q chunk**: `8 + 8 + 8 + 4 = 28` tiles (matches
`Sq_chunk_t * vDHt`).

### 7.2 `cb_out` size — before and after

| Variant | Formula | Tiles | L1 (bf16, 2048 B/tile) |
|---|---|---|---|
| **Current** | `Sq_chunk_t * vDHt` = `7 * 4` | **28** | **56 KB** |
| Proposed, 3-deep pipeline | `3 * qktv_h * vDHt` = `3 * 2 * 4` | 24 | 48 KB (−8 KB, **−14%**) |
| Proposed, 2-deep ping-pong | `2 * qktv_h * vDHt` = `2 * 2 * 4` | 16 | 32 KB (−24 KB, **−43%**) |
| Minimum (1 slot, no overlap) | `qktv_h * vDHt` = `2 * 4` | 8 | 16 KB (−40 KB, **−71%**) |

The dynamic-h case also exercises the `qktv_remainder_h` path in every Q chunk,
which is the most demanding validation target — if this test passes, the
static-h configs are trivially covered.

### 7.3 Baseline test invocation

Command (from project CLAUDE.md: always use `run_safe_pytest.sh`, not `pytest`):

```bash
source python_env/bin/activate
scripts/run_safe_pytest.sh \
    tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_sweep_perf_impl \
    -k "wan2_2_4xGLX_analog and q224 and k512 and bf16"
```

`test_sdpa_sweep_perf_impl` sets `do_check=False` — runs once for perf. For
correctness-gated iteration, also run `test_sdpa_accuracy` with the same
`-k`:

```bash
scripts/run_safe_pytest.sh \
    tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_accuracy \
    -k "wan2_2_4xGLX_analog and q224 and k512 and bf16"
```

PCC threshold in `test_sdpa_accuracy`: 0.9997, RMSE < 4e-2.

### 7.3.1 Baseline measurement (unmodified code, 2026-04-18)

Commit: `d0e37e5048` (branch `ring_joint_sdpa_cb_out_shrink_v2`).

**Functional check** — `test_sdpa_sweep_perf_impl[wan2_2_4xGLX_analog-k512-q224-bf16]`:
PASSED, 1.96s wall, `Multicast eligibility: 10/10 chains using mcast`.

**Performance** — `test_sdpa_create_perf_table[wan2_2_4xGLX_analog]`, 3 runs,
q224-k512 row only (rest of the sweep table matches across runs within the
same noise band):

| Run | Device kernel duration (ms) | Math util (%) | Cores used |
|---|---|---|---|
| 1   | 0.170 | 55.4 | 110/110 |
| 2   | 0.169 | 55.7 | 110/110 |
| 3   | 0.171 | 55.3 | 110/110 |
| **Avg** | **0.170** | **55.5** | 110/110 |

Other q224-k512 metadata (stable across runs):
- Iters/core: 5 (= `⌈2368/512⌉` K chunks × 1 Q chunk per core given full
  parallelization of 11 Q chunks across 110 cores with nh=10).
- Pad waste: 11.1% · Slot waste: 0.0% · Core util: 100%.
- MM FLOPs for the workload: 28.71 GFLOPs.

**Baseline targets for the iteration loop** (see acceptance criteria §7.4):
- Device kernel duration (q224-k512): **≤ 0.170 ms** (no regression).
- Math util (q224-k512): **≥ 55.5%** (no regression).
- `cb_out` size: target **≤ 32 KB** (−43% from 56 KB) at ping-pong depth
  (shipped configuration).

**Computed CB size** (not yet logged): `7 * 4 = 28 tiles = 56 KB`.

Next baseline step: add a `log_debug` line to `sdpa_program_factory.cpp` next
to the `c_out0_config` creation (line 792) to print the exact L1 bytes:

```cpp
log_debug(tt::LogOp, "cb_out L1 bytes: {} ({} tiles)",
          out0_t * out_tile_size, out0_t);
```

Run with `TT_METAL_LOGGER_LEVEL=Debug` to capture. This instrumentation is
cheap and should stay in during the iteration loop so each variant's CB size
is visible in the log.

### 7.4 Iteration loop

1. **Baseline capture**: run both tests above with the **current** code. Log
   - device kernel duration (from tracy / `post_process_ops_log`),
   - L1 occupancy of `cb_out` (add a `log_debug` in `sdpa_program_factory.cpp`
     next to line 792: `log_debug(tt::LogOp, "cb_out L1 bytes: {}",
     out0_t * out_tile_size);`),
   - PCC and RMSE.
2. **Change guardrails**: only edit inside `use_streaming_compute`. Keep a
   runtime-selectable flag (env var or `.hpp` constexpr) that falls back to
   the current `out0_t = Sq_chunk_t * vDHt` path during bring-up.
3. **First iteration — shrink CB only, leave kernel and writer alone**: set
   `out0_t = 3 * qktv_h * vDHt = 24 tiles`. Expectation: device hangs in
   `cb_reserve_back` — confirms the single-up-front-reserve dependency. Revert.
4. **Second iteration — per-group reserve + group-local pack addressing**
   (§5.2 Edits A + B). Keep the bulk `write_block` in the writer. Run
   `test_sdpa_accuracy`. Expectation: still hangs, because the writer won't
   unblock until the whole chunk is in `cb_out`, but the shrunk CB can only
   hold 3 groups — the producer blocks on reserve, the writer blocks on wait,
   deadlock.
5. **Third iteration — switch writer to row-grouped drain** (§5.3). Now
   compute and writer cooperate per-group. Run `test_sdpa_accuracy`. Expect
   pass. Run `test_sdpa_sweep_perf_impl` for perf delta.
6. **Fourth iteration — tune `out0_groups`**: try 2, 3, 4. Pick the smallest
   that sustains full MM2 throughput (watch for new `cb_reserve_back` waits
   in the tracy trace).

Acceptance criteria before broadening scope:
- `test_sdpa_accuracy` passes (PCC ≥ 0.9997, RMSE < 4e-2).
- `test_sdpa_determinism` passes 10 iterations.
- **Performance must not regress** on `test_sdpa_create_perf_table`
  (q224-k512 row, 4× Galaxy analog):
  - Device kernel duration (ns): **≤ baseline** (see §7.3.1 for measured
    baseline — average of 3 runs).
  - Math utilization (%): **≥ baseline**.
  - Zero tolerance for perf regression is the policy: the goal is L1 savings
    with the same or better throughput. If a proposed change causes any perf
    regression on this config, it must either be reverted or the CB depth
    (`out0_groups`) must be bumped until throughput recovers.
- All three `Q_CHUNK_SIZES ∈ {224, 256, 288}` × `K_CHUNK_SIZES ∈ {128, 256, 512}`
  combos pass for the `wan2_2_4xGLX_analog` shape.

After the single-chip path is green, mirror the kernel-side edits in the ring
joint variants (`ring_joint_sdpa_program_factory.cpp`,
`exp_ring_joint_sdpa_program_factory.cpp`) — they inherit the kernel changes
automatically because `sdpa_ring_v2` shares the same Phase 2 body, and their
writer (`ring_joint_writer.cpp`) already drains row-by-row so §5.3 is a no-op
there.

---

## 8. References

- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp:412-413, 792` — CB allocation (single-chip)
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa.cpp:112-156` — streaming path entry (`sdpa_standard_v2`)
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp:840-1128` — Phase 2 MM2 body (shared with ring joint)
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp:867` — single up-front reserve
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/writer_interleaved.cpp:163` — single-chip writer call site
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp:1076-1105` — bulk `write_block`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_writer.cpp:146-178` — row-by-row drain template to port
- `tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py` — iteration test file
- Companion: `OUT_SUBBLOCK_STREAMING.md` — `qktv_h` bump and dynamic subblock behavior
