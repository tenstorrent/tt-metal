# Verification Report: moe_fused_swiglu

Device **blackhole_p150**, `compute_with_storage_grid_size() = 11 x 10 = 110` cores. Every golden
cell ran on **110 of 110 cores** (`device_num_cores` in `verifier_report.json`), i.e. the design's
"no single-core path, no idle sub-grid" holds as built.

---

## HEADLINE FINDING — the golden PCC gate sits ABOVE the format floor it grades

`supported_fail = 44/45`, every one `numerical-precision`, `pcc = 0.9789-0.9796` against a gate of
`0.98`. That is **not** a kernel defect and **not** closable by any kernel:

I measured the *unbeatable format floor* independently of the implementation — the identical torch
fp32 chain with only the mandated `bfloat4_b` weight quantization applied
(`from_torch`/`to_torch` round-trip, the recipe `helpers.py:152-158` itself prescribes), on the
**golden suite's own fixture and seed**:

| emb | cap | count | floor (bfp4 weights) | + contractual bfp8 `h`/out | vs gate 0.98 |
|---|---|---|---|---|---|
| 6144 | 1024 | 32 | 0.979673 | 0.979613 | **UNREACHABLE** |
| 6144 | 1024 | 255 | 0.979798 | 0.979740 | **UNREACHABLE** |
| 6144 | 2048 | 64 | 0.979731 | 0.979668 | **UNREACHABLE** |
| 6144 | 2048 | 511 | 0.979766 | 0.979705 | **UNREACHABLE** |
| 6144 | 5120 | 160 | 0.979983 | 0.979927 | **UNREACHABLE** |
| 6144 | 5120 | 1279 | 0.979737 | 0.979681 | **UNREACHABLE** |
| 7168 | 1024 | 32 | 0.980193 | 0.980126 | reachable by 1.3e-4 |
| 7168 | 1024 | 255 | 0.979893 | 0.979836 | **UNREACHABLE** |
| 7168 | 2048 | 64 | 0.979813 | 0.979748 | **UNREACHABLE** |
| 7168 | 2048 | 511 | 0.979896 | 0.979834 | **UNREACHABLE** |
| 7168 | 5120 | 160 | 0.979921 | 0.979851 | **UNREACHABLE** |
| 7168 | 5120 | 1279 | 0.979746 | 0.979688 | **UNREACHABLE** |

11 of the 12 (emb, capacity, fill) combinations have a ceiling **below** the gate, and the 12th
leaves 1.3e-4 of PCC for the whole kernel (three matmuls, a transcendental, a cross-core reduce, an
all-gather and a bfp8 requantization). `bfloat4_b` is a sign + 3 magnitude bits against a shared
exponent — ~6.5 % per element on `randn` weights, ~22 % RMS through three matmuls in series, which
is exactly the `rms = 0.221-0.230` the harness records. `PCC = 0.98` corresponds to a 20 % residual,
i.e. the gate is *inside* the format's own noise and its exact position is fixture noise.

**Action requested of the user (I must not edit the golden suite):** relax `_PCC_GATE` in
`eval/golden_tests/moe_fused_swiglu/feature_spec.py:279` and `helpers.py:159` to sit below the
measured floor (e.g. **0.975**), or quantize the reference weights to bfp4 before comparing. Until
then all 44 cartesian+loose correctness cells report red for a reason that has nothing to do with
this op. Route via `/golden-tests`.

**Scale-vs-precision triage (done before routing anything to a precision refinement).** PCC is
*modestly* below target (0.979, not >=0.999) and the got/true ratio spread is
`median 1.036-1.045, p5 ~= -0.32, p95 ~= 2.40` — an enormously broad spread centred near 1.0, which
is broadband quantization noise. It is **not** the high-PCC + high-RMS "tight cluster at a non-1.0
constant" signature of a uniform scale/structural bug, so this is genuinely precision, not a scale
error. The three deterministic structural tests (`test_moe_fused_swiglu_debug.py`: all-ones,
hidden-identity, emb-contraction) pass exactly, which independently clears the Kg row split, the
reduce tree, the h all-gather round order, the phase-2 K indexing and the Ne output split.

---

## Code Review

### Prompt rules (`eval/prompts/moe_fused_swiglu.txt` § Rules) — all MUST items verified

| Rule | Verdict |
|---|---|
| The count MUST come from the device tensors, no host readback | ✓ reader NoC-reads `idx` then `counts[idx[local_expert_id]]` into unpushed scratch CBs and publishes `{count, M_t, m_blocks}` to an L1 mailbox; no Python ever sees the count. `test_moe_fused_swiglu_reads_the_indirected_count` exercises the indirection trap. |
| ONE device program per call, ONE expert | ✓ one `ProgramDescriptor`, one `generic_op` dispatch. |
| `h` must not reach DRAM or be observable | ✓ `h` exists only in `cb_h_local` / `cb_h`; no tensor, no memory_config, not in the signature. |
| Rows past `count` UNDEFINED: not zeroed, not contaminating a real row | ✓ writer breaks at `row >= m_t`; nothing reduces across tokens. The suite's `100.0` sentinel in the padding rows would collapse PCC on any leak — PCC sits at the format floor. |
| ROW_MAJOR activation: tilize MUST be fused in-kernel; entry point must not call to_layout/tilize/pad/slice | ✓ compute calls `tilize<KR_PAD, cb_x_in, cb_x_tiles, ..., OutputPolicy::CallerOwned>` directly into the reader-reserved resident slot; the entry point contains no manipulation op. |
| The SwiGLU MUST be fused | ✓ `add_bias_bcast_rows<..., SiluActivation>` (SiLU on the packer thread) + FPU `mul`, same program. |
| gate and up MUST share one activation transport | ✓ one resident `cb_x_tiles`, `InputPolicy::WaitAndRetainOnLastBlock` on both matmuls, one explicit `pop_front` at the end of the M-block. Host-asserted (`num_k_blocks_gu != 1` raises). |
| No internal config or geometry search | ✓ every knob is a host constant; the `MOE_SWIGLU_*` env vars are documented `/perf-measure` ablation hooks with fixed defaults. |
| Report perf as utilization AND device ns with the structure | ✓ table below + `changelog.md`. |

Soft guidance not followed (advisory only, no change made): the prompt suggests configuring the
tilize DEST remap once and passing `RemapMode::AssumeConfigured` **for a tilize in a hot loop**. Here
the tilize runs at most `n_inject = 1` time per M-block on 8 of 110 cores (a few hundred cycles
against a 220 us kernel), so the condition barely applies; splitting the call into a prologue
`Configure` + hot-loop `AssumeConfigured` would add a second code path for no measurable gain. Left
as-is deliberately.

### Registry conformance — clean, nothing to auto-fix

`INPUT_TAGGERS` (3 taggers, all `(inputs, axes)` signatures), `SUPPORTED` (5 axes: every axis the
kernel gates on plus every tagger key), `EXCLUSIONS = []`, `validate()` (structural `ValueError`s ->
per-axis `UnsupportedAxisValue` -> cell-level `ExcludedCell`, in that order) all present and
correctly wired; `validate()` is the entry point's first statement. **`INVALID` is NOT declared in
the op file** ✓. `tag_fill` matches `feature_spec.classify_fill` verbatim. `fill` is correctly
carried as observed-but-uncheckable (it derives from a device-resident value; `validate()` is
host-side and forbidden from reading `counts`). `xpass_drift = 0` and `xfail_wrong_mode = 0`, so
there is no drift to fix.

### INVALID audit (`eval/golden_tests/moe_fused_swiglu/feature_spec.py`)

`INVALID = []`, and that is **correct** for this op — I checked each authoring rule rather than the
emptiness:
- *Single-tensor coupling*: no entries, so no cross-tensor coupling mistake is possible.
- *Universe-must-change*: `input_format` already collapses the activation's dtype x layout cross to
  the two physically real combinations, so the canonical `bf8b + ROW_MAJOR` entry is **not
  applicable here** (that cell is not expressible in this axis set — it is neither `"bf16_rm"` nor
  `"bfp8_tile"`). Nothing else is structurally impossible: every tensor is DRAM interleaved, and
  every `(capacity, fill)` pair is realisable because `fill` is defined relative to `capacity`.
- *No "my kernel can't do this yet" entries* ✓ (that would be EXCLUSIONS, which is also empty and
  honestly so).
- Norm-like weight-canonicalization rules do not apply (no optional weight axis).

### Design conformance (`op_design.md` §1 Blocking Model)

Confirmed as built: **Hn** across the 11 grid columns (`HN_PAD`), **Kg** across the 10 grid rows
(`KR_PAD`/`kr(y)`) combined by a binary reduce tree, **Ne** across all 110 cores (`ec(i)`), **Kh**
sequential per core (`HGROUPS` K-blocks), **M** the sequential outer loop (`M_BLOCK`), `x`
row-multicast with a rotating injector, `h` grid-wide multicast fused into the phase-2 K stream. Both
dataflow halves are batched (reader: W_gate + W_down issued as multi-transaction batches behind one
barrier; writer: W_up on NoC1 — the dual-NoC split — and coalesced multi-tile output writes). Grid
is fully occupied. `TensorAccessor` everywhere for addressing, `void kernel_main()`,
`api/dataflow/dataflow_api.h` includes: all correct.

Deviations found. Those already documented in `changelog.md` §2 (grid from the device, `M_BLOCK = 8`,
`num_k_blocks = 1`, `SEM_GO`/`SEM_DATA` instead of level parity, caller-owned direct tilize into the
reader-reserved x slot, `cb_*_acc`/`cb_*_send`, per-tile-row `add_bias_bcast_rows`,
`KBlockInnerDimFn` instead of a zero-fill) I re-verified and accept. Three are **not** in that list:

1. **The runtime `m_tiles` shrink was never implemented — a performance-conformance bug.**
   `op_design.md` §3 specifies `m_tiles = min(M_BLOCK, M_t - b*M_BLOCK)`; all three kernels instead
   iterate the constant `M_BLOCK`, so a `count = 128` dispatch (`M_t = 4`) does **8** tile-rows of
   x-multicast rounds, gate/up matmul, reduce payload and `down` matmul — twice the necessary work,
   and 8x at `count = 32`. The measurement proves it: count 128 = **221.0 us** vs count 256 =
   **227.1 us** (2.6 % apart for half the tokens), while the graded targets are 91.8 us vs 108.0 us
   (15 % apart). This is the single largest perf item at the graded low-count end and it is a
   correctness-neutral change, so it is **Refinement 1**. Not fixed in this pass: it touches the
   trip counts and matmul shapes of all three kernels plus a CB-granularity constraint
   (`m_eff` must divide `M_BLOCK` or a partial push can straddle a CB's FIFO end), and it needs a
   device-ns measurement to close — i.e. exactly a refinement, not a verifier edit.
2. **Phase 2 uses 3 of 8 DEST tiles.** `OUT_SUBBLOCK_H = 1` is a single knob shared by both matmuls;
   `HN_PAD = 6` pins gate/up at height 1, but the `down` matmul's `out_subblock_w = ec` is 2-3, so
   its sub-block is 2-3 tiles against a `DEST_AUTO_LIMIT` of 8. `matmul_output_subblock` measured
   that the win tracks sub-block *size*. The fix is a *separate* height knob for the `down` matmul —
   filed as a lever in Refinement 2 (it wants a measurement, and it interacts with R1's runtime
   `m_eff`).
3. **`DataReadySignal::Counter` (design §4.2/§4.4) is not usable** — two independent `mcast_pipe`
   bugs, see below. Flag is shipped; the difference is real serialisation.

### Blocking-model fidelity — knobs, DRY, and what I fixed

Every block factor is a named parameter with one definition and everything derived from it; no CB
page count is a function of `capacity` or `count`. Two exceptions found and **fixed**:

- **`cb_w_down` was sized to a whole-op dimension.** Its page count was `HGROUPS * HN_PAD * EC_MAX`
  — `HGROUPS * HN_PAD` *is* `HID_T`, the entire `down` contraction extent (111.4 KB of a 1427 KB
  budget). Replaced with a real depth knob: `depth_wd = max(DEPTH_WD, wd_ahead + 1)` K-blocks, with
  the FIFO-wrap precondition derived and asserted host-side (single-block pushes can never straddle;
  the `wd_ahead`-block batch reserve only stays inside the buffer when the depth divides the
  per-M-block push count, so the `WD_AHEAD > 1` ablation falls back to the full stream and stays
  legal). A/B measured over the 9 loose cases:

  | `DEPTH_WD` | cb_w_down L1 | sum device ns (9 cells) | vs the old whole-extent sizing |
  |---|---|---|---|
  | 3 | 30.4 KB | 6 220 416 | +1.68 % |
  | **5 (shipped)** | **50.6 KB** | **6 126 822** | **+0.15 % (noise)** |
  | 7 | 70.9 KB | 6 150 023 | +0.53 % |
  | 11 (= old) | 111.4 KB | 6 117 596 | 0 |

  Shipped at 5: **60.8 KB of L1 freed for no measurable time** (the 7-vs-5 inversion sets the noise
  floor at ~0.4 %). `MOE_SWIGLU_DEPTH_WD` keeps it a live knob.
- **48 KB of unreachable L1 on the `bfp8_tile` path.** `cb_x_in` serves only the row-major
  DRAM-read -> fused-tilize path; tiled input lands directly in the resident slot. It is one page in
  the tiled configuration. The legacy-named `cb_x_stage` is now a format-independent 64-byte
  compute-to-reader completion channel rather than a tile payload buffer.

Per-core L1 after the fixes (measured by building the real descriptor, emb 7168 / capacity 5120):
**1267.9 KB** (`bf16_rm`) and **1199.6 KB** (`bfp8_tile`) of the 1427.1 KB budget — 159 KB / 227 KB
of slack, up from ~78 KB.

`KB1_FRACTION = 1` makes `cb_w_gate`/`cb_w_up` hold a full per-row K extent (155 KB each, 310 KB
together, the largest fixed cost). That is *not* a collapsed knob — `KR_PAD` is the row split's block
factor, `KB1_FRACTION` is live, and a host `RuntimeError` names exactly what else must change
(the second-CB copy of `x`) if it is turned. It is, however, the L1 that Refinement 3 has to spend.

### DRY fixes (single source of truth)

- **The bank-run coalescing existed in six places**: `remap_n` and `run_len` duplicated verbatim in
  the reader and the writer, and the same run-enumeration `while` loop copied four times (W_gate,
  W_down x2, W_up, plus the output write). A `WRUN` turn had to land consistently in all six. Now one
  definition: `kernels/moe_fused_swiglu_bank_runs.hpp`, `BankRuns<REMAP, NUM_BANKS, WRUN>::{remap,
  run, read, write}`, instantiated once per kernel from that kernel's CT args. Verified the header is
  actually compiled into both dataflow TUs (deliberate `#error` probe: brisc **and** ncrisc both
  failed on it, and the JIT cache correctly invalidates on header content).
- **The mailbox word layout was three sets of bare literals** (`mbox[0..3]` in reader, compute and
  writer). Now `kernels/moe_fused_swiglu_common.hpp` — include-free so the compute TU can use it —
  defines `MBOX_COUNT / MBOX_M_T / MBOX_M_BLOCKS / MBOX_READY` once.
- `dest_limit = 8` inlined in the descriptor -> module-level `DEST_AUTO_LIMIT_TILES` with the
  half-sync / `fp32_dest_acc_en=False` derivation recorded next to it.
- A variable named `counter` actually holding `McastDataReady.Flag` -> `data_ready_signal`.

### `mcast_pipe` Counter path: one bug fixed upstream, one documented as still open

The design asks for `DataReadySignal::Counter` on both collectives; the build used `Flag` because
Counter hung. I re-tested Counter on device after fixing the bug the implementer identified, and it
still hangs — for a **second, independent** reason. Both are now recorded where they can be acted on:

1. **FIXED** (`ttnn/cpp/ttnn/kernel_lib/mcast_pipe.inl::signal_ready_`): the multicast atomic
   increment was handed `mcast_dests`, which is the INCLUDE-source (loopback) fan-out, while
   `noc_semaphore_inc_multicast` is unconditionally exclude-source ("the multicast sender cannot be
   part of the multicast destinations", `dataflow_api.h`). `fence_()`'s non-posted
   `async_atomic_barrier()` therefore waited forever for one ack from a destination never addressed.
   It now always passes `num_dests_excl_`. No in-tree op used Counter, so nothing can regress.
2. **STILL OPEN** (documented at `mcast_pipe.hpp`'s `DataReadySignal` and at this op's emission
   site): `send_data_()` issues the data multicast with `NOC_CMD_VC_LINKED` and relies on the
   *signal* to terminate the linked chain. A linked chain is only released by an UNLINKED transaction
   **on the same command buffer** — the Flag signal is a multicast write on `NCRISC_WR_CMD_BUF`
   (terminates it), but the Counter signal is a multicast atomic on `write_at_cmd_buf`, a different
   buffer, so the path stays reserved and the sender's next write spins in `noc_cmd_buf_ready()`
   forever. Triage confirmed exactly that: round-0's `h` sender stuck in
   `noc_async_write_multicast_loopback_src -> noc_cmd_buf_ready` while all 110 readers sat in the `h`
   `wait_min`. Making Counter usable needs the Counter path to send unlinked plus an **acked** write
   barrier before the atomic (the linked chain is what currently enforces data-before-signal
   ordering), which trades a flag-reset round trip for a write-ack round trip — a helper-level change
   that must be measured, not guessed. Filed as a lever in Refinement 2. The op ships on `Flag`.

### Helper usage

Compute is helper-first throughout: `compute_kernel_hw_startup<SrcOrder::Reverse>`,
`ActivationInitHelper<SILU>::init()`, `tilize<>`, `matmul_block<>` (x3 shapes, `packer_l1_acc` +
caller-owned pack target on phase 2, `KBlockInnerDimFn` shrinking both ragged K tails),
`add_bias_bcast_rows<..., SiluActivation>` (SiLU on the packer thread — free), `add`/`mul`/`copy`
from `eltwise_convenience`. The in-place `add` (output CB == input CB) is legal here because the
chain pops inputs in the compute phase *before* reserving the output in the pack phase, so the read
and write pointers advance in lockstep through a single-slot CB — verified in `eltwise_chain.inl`.
`mcast_pipe` is used for both real broadcasts. The four raw-dataflow deviations (bank-run
read/write, unicast+counting-semaphore reduce tree, L1 mailbox for a scalar loop bound on all three
TRISCs) each have a concrete named cost that the helper imposes and are documented at the site and
in `op_design.md` §6 — all four re-checked and accepted. `pack_reconfig_l1_acc(0)` after the phase-2
matmul is load-bearing (the helper leaves packer L1 accumulation enabled) and correctly present.

### CB sync

Every CB's push count equals its wait count per M-block; I walked all 19. The two that took real
checking: `cb_x_tiles` (one slot of `M_BLOCK*KR_PAD`, so the multicast landing address is identical
on every core in the row, pushed once after the last round and popped once by compute after *both*
matmuls) and `cb_reduce_*_in` (one slot, so a child's own write pointer *is* the parent's landing
address; the `SEM_GO` invite is the flow control that stops a child overwriting an unconsumed slot).
`count == 0` gives `m_blocks == 0` uniformly on all 110 cores: no CB traffic, no collective, no
semaphore, and the golden zero-count cell passes in 6 us.

---

## Precision Baseline

`tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_moe_fused_swiglu_precision_baseline.py`
(4 shapes x 2 activation formats). Errors are absolute on an output whose elements are ~1e5, so the
relative RMS and the ratio spread are the meaningful columns.

| format | emb | cap | count | PCC | Max Abs Err | Mean Abs Err | Rel RMS Err | bfp4 floor | kernel dPCC |
|---|---|---|---|---|---|---|---|---|---|
| bf16_rm | 6144 | 1024 | 32 | 0.979919 | 2.288e5 | 3.495e4 | 0.2163 | 0.980485 | 0.000566 |
| bf16_rm | 7168 | 1024 | 255 | 0.979089 | 2.543e5 | 4.057e4 | 0.2217 | 0.979718 | 0.000628 |
| bf16_rm | 7168 | 2048 | 512 | 0.979098 | 2.548e5 | 4.050e4 | 0.2213 | 0.979746 | 0.000648 |
| bf16_rm | 7168 | 5120 | 256 | 0.979047 | 2.691e5 | 4.058e4 | 0.2219 | 0.979725 | 0.000678 |
| bfp8_tile | 6144 | 1024 | 32 | 0.979912 | 2.206e5 | 3.457e4 | 0.2140 | 0.980485 | 0.000572 |
| bfp8_tile | 7168 | 1024 | 255 | 0.979067 | 2.543e5 | 4.015e4 | 0.2195 | 0.979718 | 0.000651 |
| bfp8_tile | 7168 | 2048 | 512 | 0.979093 | 2.630e5 | 4.008e4 | 0.2190 | 0.979746 | 0.000653 |
| bfp8_tile | 7168 | 5120 | 256 | 0.979074 | 2.527e5 | 4.014e4 | 0.2194 | 0.979725 | 0.000650 |

got/true ratio spread: `median 1.036-1.045`, `p5 ~= -0.32`, `p95 ~= 2.40` on every row.

**Assessment.** Error is broadband and format-dominated: `rel_rms ~= sqrt(2(1-PCC))` on every row,
and the ratio spread is enormous around a near-1.0 median — quantization noise from bfp4 weights,
not a scale or structural error. The two activation formats agree to 5e-5 of PCC (the fused tilize
adds nothing measurable), and PCC is flat across `emb`, `capacity`, `count` and `m_blocks` (no
accumulation-order drift with more work). The **kernel-attributable** shortfall against the
unbeatable floor is a remarkably consistent **5.7e-4 to 6.8e-4**, of which ~6e-5 is the contractual
bfp8 `h`/output; the rest is the bf16 DEST/L1 accumulation chain plus bfp8 for `x`, the gate/up
accumulators and the reduce payload. `fp32_dest_acc_en` is pinned False by both the prompt and the
harness, so ~4e-4 is the realistic recoverable amount (Refinement 4).

**Recommended tolerances.** For this op's own tests: `PCC >= measured_bfp4_floor - 0.0015` (what the
baseline asserts — a regression tripwire against the ceiling, not against an arbitrary constant),
which is `PCC >= 0.978` in absolute terms. `rtol ~= 0.3` element-wise is meaningless here (the ratio
p5/p95 straddle zero); grade on PCC + relative RMS only. The golden suite's `0.98` should become
**<= 0.975** (see the headline finding).

---

## Verifier CLI Summary

`eval/eval_test_runner.sh eval/golden_tests/moe_fused_swiglu/ <results>` then
`python3 -m eval.verify_supported <results> ttnn.operations.moe_fused_swiglu`
(artifact: `verifier_report.json`, copied next to this report).

- supported_pass: **1**
- xfail_expected: **0** — expected and correct: `SUPPORTED == TARGET` on all five axes, so no cell
  is outside SUPPORTED. There is nothing to xfail and no generality gap to queue.
- invalid_skipped: **0** (`INVALID = []`, audited above)
- infeasible_skipped: 0
- **supported_fail: 44** — all `numerical-precision`, `pcc 0.9789-0.9796` vs a gate of 0.98 that
  sits above the measured format floor (headline finding). Kept failing deliberately: the PCC/RMS
  *is* the signal, and hiding these behind EXCLUSIONS or a shape tagger would delete the only
  measurement that shows how close the kernel is to its ceiling. Routed to Refinement 4 for the
  ~4e-4 the kernel actually owns.
- xpass_drift: **0** ✓
- xfail_wrong_mode: **0** ✓
- supported_marked_xfail: 0 ✓
- No hangs, no OOM, no compilation failures, no inf/NaN anywhere.

## Measured performance (shipped config, from the same golden run)

`util = dram_read_bytes / (512e9 * device_kernel_ns * 1e-9)`, reads only.

| format | emb | cap | count | read MB | device ns | util | target ns / util | best-measured ns / util |
|---|---|---|---|---|---|---|---|---|
| bf16_rm | 7168 | 5120 | 128 | 26.61 | 221 006 | 0.235 | 91 800 / 0.566 | 102 000 / 0.509 |
| bf16_rm | 7168 | 5120 | 256 | 28.44 | 227 123 | 0.245 | 108 000 / 0.514 | 120 000 / 0.463 |
| bf16_rm | 7168 | 5120 | 512 | 32.11 | 439 863 | 0.143 | 161 816 / 0.388 | 179 795 / 0.349 |
| bf16_rm | 7168 | 1024 | 256 | 28.44 | 226 772 | 0.245 | 108 000 / 0.514 | 120 000 / 0.463 |
| bf16_rm | 7168 | 2048 | 256 | 28.44 | 227 776 | 0.244 | 108 000 / 0.514 | 120 000 / 0.463 |
| bfp8_tile | 7168 | 5120 | 256 | 26.72 | 218 430 | 0.239 | 108 000 / 0.483 | 120 000 / 0.435 |
| bf16_rm | 6144 | 5120 | 256 | 24.38 | 212 463 | 0.224 | report only | — |
| bf16_rm | 7168 | 5120 | 5120 | 98.17 | 4 334 803 | 0.044 | report only | — |
| bf16_rm | 7168 | 1024 | 0 | — | 6 026 | — | must not hang | ✓ |

Structure measured: `grid 11x10`, `M_BLOCK 8`, `KGROUPS 10`, `HGROUPS 11`, `KR_PAD 23`, `HN_PAD 6`,
`EC_MAX 3`, `KB1 = KR_PAD` (`num_k_blocks = 1`), `WRUN 8`, `WD_AHEAD 1`, `DEPTH_WD 5`, `DEPTH_W 2`,
`DEPTH_H 3`, `OUT_SUBBLOCK_H 1`, LoFi + `math_approx_mode` + 16-bit DEST.

Capacity costs nothing (226 772 / 227 776 / 227 123 ns at capacity 1024 / 2048 / 5120, same count),
which is the one cross-capacity requirement the prompt names. `count = capacity` is a reported-only
finding at util 0.044 (10 sequential M-blocks, weights re-read 10x).

---

## Recommendations

1. **Fix the golden PCC gate first** (headline finding). Every correctness signal for this op is
   currently saturated red for a format-floor reason; nothing downstream can be graded honestly
   until the gate sits below 0.9797.
2. **Refinement order is value- and dependency-driven, not axis-driven** — `TARGET - SUPPORTED` is
   empty, so the whole queue is measured perf work (plus one precision entry). Refinement 1 (honour
   the runtime token count) must land before 2 and 3: both of those restructure loops whose trip
   counts R1 changes, and doing them first means redoing them.
3. **`M_BLOCK = 16` is NOT the knob turn `changelog.md` §5 calls it.** I did the arithmetic: the
   M-scaled CBs (`cb_x_tiles` 195.5 + `cb_reduce_*` 102 + `cb_h` 153 + `cb_out_tiles` 51 +
   `cb_*_acc`/`*_send`/`silu`/`h_local` 306 + `cb_out_interm` 48) total ~855 KB, so doubling
   `M_BLOCK` needs +855 KB against 159 KB of slack. It is reachable only after the 310 KB in
   `cb_w_gate`/`cb_w_up` is broken up (i.e. after `KB1_FRACTION < 1`, which needs the second-CB copy
   of `x`). Corrected in the queue so nobody spends a pass discovering that.
4. **L1 headroom, for planning:** 159 KB free on the `bf16_rm` path. Double-buffering `cb_x_tiles`
   for M-block pipelining costs +195 KB, so it needs `DEPTH_W: 2 -> 1` (frees 155 KB) as well —
   which is exactly the trade Refinement 3 should measure, since `DEPTH_W = 2` currently buys little
   (the reader reserves the next block's weights only after the previous M-block's phase 2).
5. **Per-core-range CBs would free another ~102 KB** (`cb_gate_send`/`cb_up_send` are dead on the 11
   reduce roots; `cb_gate_silu`/`cb_h_local` are dead on the other 99 cores). **Do not do this
   naively**: heterogeneous CB sets per core would let the L1 allocator place the *shared* CBs at
   different addresses on different cores, and `cb_x_tiles` / `cb_h` / `cb_reduce_*_in` all require
   a byte-identical landing address across cores. Only worth attempting with an explicit
   address-uniformity check.
6. **The reduce tree fan-in is serialised by construction** and this is not visible in any single
   ablation: the parent invites child `c`, waits for its data, *then* invites child `c+1`, because
   `cb_reduce_*_in` is a single slot and every child writes the CB base. A root has up to 4 children
   -> up to 4 sequential 102 KB round trips per M-block. Giving the CB `MAX_CHILDREN` slots and each
   child its slot index as a runtime arg makes the fan-in parallel; costed and filed in Refinement 2.
7. `WRUN` (3 %), `SKIP_COMPUTE` (13 %), no-handshake (5 %) and `WD_AHEAD` (negative) are all measured
   small, so ~85 % of the time is the *serial composition* of the collectives — treat any new perf
   idea that does not shorten that chain as low-yield on this op.
8. Kept as known issues, not queued: the `make_mailbox` per-call L1 allocation + 7 KB host->device
   zeroing (required by the magic-word protocol, negligible but it is host overhead on every
   invocation), and the `tilize` `AssumeConfigured` advisory above.
