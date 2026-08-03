# moe_fused_swiglu — race-condition audit + interleaved determinism stress (round 17)

Round 17 added **three new cross-agent protocols** to an op that was already built almost entirely
out of collectives. This is the audit of every synchronisation object in the shipped configuration,
plus the empirical result that goes with it.

Shipped configuration audited: `WD_SPLIT=3`, `HPOSTED=1`, `REDUCE_MECH=dest_acc`, `HSLOT=1`,
`HSEND=reader`, `HACK_AHEAD=2`, `DEPTH_H=3`, `REDUCE=scatter`, `SCATTER_NOC=split`,
`W_RESIDENT=WD_RESIDENT=1`, `WD_AHEAD=1`, `GU_CHUNKS=3`, `XPRIO=XSTAGE_FIRST=XSTAGE_STAGGER=1`.

---

## 1. Empirical result

`tests/.../test_moe_fused_swiglu_determinism_stress.py` — many shapes interleaved in a
pseudo-random order, each compared **bitwise** against its own first run, device-side.

| run | grid | shapes | dispatches | seed | result |
|---|---|---|---|---|---|
| interleave | 110 cores (default) | 14 + zero-count | **5 000** | 0 | bitwise identical |
| interleave | **11x8** (the perf config) | 14 + zero-count | **20 000** | 7 | bitwise identical |
| multi-M-block soak | 11x8 | 6 + zero-count | **20 000** | 13 | bitwise identical |
| long soak | 11x8 | 14 + zero-count | **50 000** | 31 | bitwise identical |
| post-restore re-verify | 11x8 | 14 + zero-count | **5 028** | 99 | bitwise identical |
| | | | **100 028 total** | | **0 divergences** |

Plus the pre-existing single-shape `test_moe_fused_swiglu_determinism.py` (10 iters x 5 cells) and
golden 45/45 on the default path.

**Why interleaving and not just repetition.** A race whose outcome is pinned by a steady-state
pipeline reproduces bit-for-bit forever, because every dispatch finds the grid in the same rhythm.
The interleave changes `m_eff` (1/2/4/8), `m_blocks` (1/2/4), activation format, `emb` and capacity
between consecutive dispatches, so no two dispatches see the same preceding state. It also
interleaves a **count == 0** dispatch, where every core does zero M-blocks — no CB traffic, no
collective, no semaphore — so any state a previous shape left behind must survive being stepped
over.

**The compare is not vacuous.** Each shape's reference is asserted finite and non-zero, and the
first *and* last element of the compared region are independently poisoned and required to fire the
marker (`assert_marker_can_fire`). The sibling file separately proves the merge/attribution path
survives a planted divergence. The compare is sliced to `ceil_tile(count)` rows because rows beyond
that are undefined by contract and hold stale DRAM.

### 1.1 Negative controls — three REAL races injected into the shipped op, all detected

A determinism test that has never failed proves very little, so three genuine synchronisation
defects were compiled into the op and the stress re-run. **All three failed the run**; the tree was
restored and re-verified green (golden 45/45 + 5 028 dispatches) afterwards.

| injected defect | what it breaks | detected by |
|---|---|---|
| drop `noc_semaphore_wait_min(sem_data_ptr, data_arrivals)` in `reader_reduce` | compute reads the landing CBs before the column's 8 contributors have written | liveness guard, `max\|out\| = inf` |
| same wait, but `data_arrivals - 1` | **7 of 8** contributors synchronised, one races | liveness guard, `inf` |
| drop `noc_semaphore_wait(hf, VALID)` in phase 2 | a core consumes a `cb_h` round before it lands | liveness guard, `inf` |

**A property of this op worth recording: a desynchronised read here does not drift, it explodes.**
Every one of the three produced `inf`, not a plausible-but-different number — including the subtle
7-of-8 variant, and including runs warmed up so the stale bytes were real h data from a previous
dispatch rather than uninitialised L1. The reason is the block-float transport: `h` is bfp8 and
`W_down` is bfp4, so a misaligned or half-written read corrupts a SHARED EXPONENT, which scales a
whole 16-element datum rather than perturbing one mantissa. The `down` matmul then accumulates it.

Consequently the harness's `assert_reference_is_live` is the first-line detector for this op's race
class and the bitwise marker is the backstop for anything subtler. Both are assertions in the same
test and either one fails the run, so the coverage claim is unaffected — but it does mean the marker
path itself is exercised by the corner-poisoning control (which runs on real op output on every
single run, for every shape) rather than by the injected races.

`MOE_STRESS_WARMUP` was added to the harness for these controls: it dispatches each shape N times
before capturing its reference. Default 0, because a correct op must be identical from its very
first dispatch — which is exactly what the 95 000 green dispatches show.

---

## 2. Inventory — every cross-agent synchronisation object

All are **monotone**: incremented, never decremented, never reset inside a dispatch, and always
compared with `wait_min` against a running local total. That is what makes them immune to arrival
order. They are re-zeroed by the runtime at each launch, which the interleave exercises hard.

| object | direction | reset discipline |
|---|---|---|
| `SEM_GO` | every core -> its whole column (invite) | monotone, `invites += KGROUPS` |
| `SEM_DATA` | contributor -> worker (gather landed) | monotone, `data_arrivals += KGROUPS * (1 or 2)` |
| `SEM_HSLICE` | worker -> column root (h slice landed) | monotone, `h_arrivals += sl_w` |
| `SEM_H_FREE` | receiver -> round r's sender (slot reserved) | monotone, `+= NUM_CORES` per send |
| `SEM_H_RDY_BASE + s` | round sender -> all cores (data ready) | **the one non-monotone object** — a VALID/INVALID flag per `cb_h` slot |
| `SEM_XSTAGED` | reader -> writer, SAME core | monotone, `b + 1` |
| `SEM_WDSPLIT` | writer -> reader, SAME core (**new**) | monotone, K-blocks completed since op start |
| L1 mailbox | reader -> writer + all 3 TRISCs | write-once per dispatch, MAGIC-stamped |

The per-slot VALID flag is the only reset-based signal, and its safety argument is unchanged from
Perf 4: rounds `r` and `r+DEPTH_H` share a cell and are ordered by the ack itself — a core acks slot
`s` only after `cb_reserve_back` proves it consumed round `r`, which is strictly after it put that
cell back to INVALID. `HACK_AHEAD` is clamped to `DEPTH_H` in the reader for exactly this reason.

---

## 3. The three round-17 mechanisms

### 3.1 `WD_SPLIT` — the writer reads part of W_down into a CB the reader owns

**Two RISC-Vs write into `cb_w_down`; only ONE touches its state.** The reader is the sole caller of
`cb_reserve_back` / `cb_push_back`. The writer performs raw `noc_async_read` into the region and
holds no CB state at all, so the single-producer invariant (§4 trap 5 of the plan) is preserved by
construction rather than by convention.

**Address derivation.** The writer addresses K-block `r` at
`cb_w_down_base + r * HN_PAD * EC_MAX * BFP4_TILE`, off a base captured before anything touches the
CB. Correct because `WD_RESIDENT` forces `depth_wd == HGROUPS`, so slot `r` permanently holds
K-block `r`. **Enforced, not assumed** — the host sets `wd_split = 0` unless
`WD_RESIDENT and depth_wd == HGROUPS` (descriptor line ~1214), and there is a separate pre-existing
`RuntimeError` if `HGROUPS % depth_wd != 0`.

**Why no flow control is needed.** Under `WD_RESIDENT` every W_down DRAM read happens at `b == 0`.
At `b == 0` the CB is empty, so nothing has been pushed, so compute cannot be reading any slot — all
HGROUPS slots are dead space the writer may fill freely. At `b > 0` the writer performs no reads at
all (early return, publishing `(b+1) * HGROUPS`). There is therefore no window in which the writer
can overwrite live data.

**The completion signal is mandatory and is the one real hazard.** `noc_async_read_barrier()` is
PER-RISC-V: the reader's barrier proves nothing about NOC_1's reads of the same K-blocks. Publishing
without the gate would hand the `down` matmul a half-written weight tile — no hang, wrong numbers.
`SEM_WDSPLIT` closes it. The reader's `wd_done` running total is exactly `b * HGROUPS` at the top of
block `b` because the reader pushes exactly HGROUPS blocks per M-block
(`WD_AHEAD` prologue + `HGROUPS - WD_AHEAD` in-loop), so `wait_min(wd_done + n)` before pushing `n`
is the tightest legal gate.

**Intra-core visibility.** `SEM_WDSPLIT` is a plain volatile store read by `noc_semaphore_wait_min`,
whose poll loop calls `invalidate_l1_cache()` every iteration (`dataflow_api.h:1965`); on Blackhole
that is a `fence` and L1 is write-through. Identical pattern to `SEM_XSTAGED`, which has shipped
since Perf 3.

**Transaction ids.** Block `r` takes trid `r+1`; the host falls back to a single blanket barrier
unless `HGROUPS <= 15` (`NOC_MAX_TRANSACTION_ID`), so two K-blocks can never alias one id and
publish bytes still in flight.

### 3.2 `HPOSTED` — posted h multicast payload, non-posted linked flag

**The ordering claim.** The payload is issued `posted=true, linked=true` on
`NOC_MULTICAST_WRITE_VC`; the VALID flag is a non-posted multicast WRITE on the same VC that
terminates the link. The flag therefore cannot overtake the payload on the wire, and the receiver's
existing `wait(VALID)` remains a true arrival proof.

**This adds no new assumption.** The SHIPPED non-posted Flag path already relies on precisely this
and nothing else — `mcast_pipe.inl::send_data_` takes **no barrier** on the Flag branch and its own
comment says "LINKED ONLY FOR THE Flag SIGNAL. The link is terminated by the following signal
mcast". Posting the data changes only whether **acks return to the sender**; it does not change wire
order or receiver-visible ordering. So the safety argument for `HPOSTED=1` is the same argument that
already had to hold for `HPOSTED=0`.

**Sender-side reuse.** `noc_async_writes_flushed()` after the flag waits on NON-posted writes sent,
which covers the flag — and the flag is the last non-posted write, so the subsequent
`set(INVALID)` cannot race its own multicast source read. The posted payload's source (`hdst`, the
`cb_h` slot) is not rewritten until that slot is re-reserved, which requires the `SEM_H_FREE` ack
round, which requires receivers to have consumed it.

**Fan-out.** `hrect.area() - 1` (exclude-source), matching `SenderPipe::num_dests_excl_`; the sender's
own copy comes from the reader's self-copy, so `src == dst` and the send is genuinely exclude-source
rather than a loopback (§4 trap 4).

### 3.3 `REDUCE_MECH=dest_acc` — DEST accumulation in the slice reduce

**No cross-core exposure.** The fold is pure compute. Its inputs are the landing CB, and the reader
does not push that CB until `SEM_DATA >= data_arrivals` proves every contributor has landed.

**Order-independence is structural.** Contributors are folded by SLOT INDEX (`c = 0, 1, 2, ...`
addressing `c * n` in the landing CB), never by arrival. Float addition is not associative, so this
is the property that makes the op reproducible at all — and it is unchanged from `addchain`.

**Sticky state is cleared.** `acc_to_dest` is a latched math-config bit; `fold_dest` clears it with
`add_tiles_init(IN, IN, false)` before returning, and sets its own formats
(`reconfig_data_format` + `pack_reconfig_data_format`) **before** its inits. Both are load-bearing:
omitting the formats produced `pcc = 1.000000` with `inf` (see `ROUND17_LOG.md`).

---

## 4. Deadlock / ordering analysis of the new writer-reader coupling

`WD_SPLIT` adds a reader-waits-on-writer edge to an op that previously only had
writer-waits-on-reader (`XPRIO`). Both directions now exist, so the cycle has to be ruled out.

```
writer:  out_drain -> W_up [waits SEM_XSTAGED >= b+1] -> wd share [SETS SEM_WDSPLIT] ->
         scatter [waits SEM_GO] -> hslice [waits cb_h_slice] -> out_issue
reader:  x stage [SETS SEM_XSTAGED] -> W_gate -> reduce [SENDS SEM_GO, waits SEM_DATA] ->
         wd_wait [waits SEM_WDSPLIT] -> phase 2
```

The writer sets `SEM_WDSPLIT` **before** it waits on anything the reader produces after
`SEM_XSTAGED`, and the reader sets `SEM_XSTAGED` early in the block. So the edge
reader->`SEM_WDSPLIT`->writer->`SEM_XSTAGED`->reader is not a cycle: the writer's set is reachable
from a point the reader has already passed. Confirmed empirically by 95 000 dispatches with no hang,
including 4-M-block shapes where the pattern repeats.

(The rejected `WD_WPLACE=scatter` placement moves the set past the scatter and hslice, which is
still acyclic but measured +10-13 % — see `ROUND17_LOG.md`.)

---

## 5. What this does and does NOT prove

**Proves, jointly with the golden suite:** the op computes the *right* answer (golden, 45/45 against
torch) and the *same* answer across 95 000 dispatches spanning 14 shapes, 4 `m_eff` regimes, 3
`m_blocks` regimes, 2 formats, 2 `emb` values, 2 grids and 4 interleave orders.

**Does not prove:**

1. **The HPOSTED wire-ordering property itself.** It is an architectural guarantee about linked
   transactions on one VC. No amount of black-box testing establishes it; the strongest true
   statement is that the shipped non-posted path already depends on the identical guarantee, so
   `HPOSTED` does not widen the exposure. If that guarantee is ever found not to hold, `HPOSTED=0`
   is a one-env-var revert and the Flag path needs re-examining regardless.
2. **Races whose outcome is fixed by structure rather than timing.** Interleaving varies the state a
   dispatch *starts* from; it does not materially perturb NoC arbitration *within* a dispatch. A
   defect that always resolves the same way would evade both the bitwise compare and PCC.
3. **Anything outside the shipped knob set.** The test refuses to run with `MOE_SWIGLU_*` overrides
   unless `MOE_DET_ALLOW_KNOBS=1`, and the non-default knobs (`HSPLIT`, `SILU_FUSE`, `SCATTER_ROT`,
   `SCATTER_NOBAR`, `DOWN_OUT`, `H_DTYPE=bfp4`) are **not** covered here. `H_DTYPE=bfp4` is known
   broken and default-off.

---

## 6. Residual risks worth a comment in future review

* **`WD_SPLIT`'s slot derivation is coupled to the per-M-block push count.** The host checks
  `HGROUPS % depth_wd == 0` and `depth_wd == HGROUPS`, but a future change to how many W_down blocks
  the reader pushes per M-block would silently invalidate the writer's `r * WD_BLOCK_TILES` address
  without tripping either guard.
* **The writer's `cb_w_down` / `cb_h` base capture assumes it never pushes those CBs.** True today
  and the reason the derivation works; it is a convention, not something the CB API enforces.
* **`fold_dest`'s `acc_to_dest` clear is on the fall-through path only.** An early `return` added to
  that function later would leak a latched accumulate bit into every subsequent FPU op.
* **`DEPTH_H = 4` is L1-dead, not semaphore-dead.** The `SEM_H2_RDY_BASE` block is now allocated only
  when `HSPLIT` is on, precisely so it does not consume the budget; re-enabling `HSPLIT` re-imposes
  the 16-semaphore ceiling.

## 7. Reproduction

```bash
# the default: ~5000 dispatches interleaved over 14 shapes, shipped defaults, 110 cores
scripts/run_safe_pytest.sh --run-all \
  tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_moe_fused_swiglu_determinism_stress.py

# the configuration the perf numbers are quoted at, harder and longer
MOE_DET_ALLOW_KNOBS=1 MOE_SWIGLU_GRID=11x8 MOE_STRESS_ITERS=50000 MOE_STRESS_SEED=31 \
  scripts/run_safe_pytest.sh --run-all <this test>

# focus the soak on the multi-M-block shapes
MOE_DET_ALLOW_KNOBS=1 MOE_SWIGLU_GRID=11x8 MOE_STRESS_ITERS=20000 MOE_STRESS_SEED=13 \
  MOE_STRESS_SHAPES="7168,1024,257,bfp8_tile;7168,1024,512,bf16_rm;7168,1024,1024,bfp8_tile" \
  scripts/run_safe_pytest.sh --run-all <this test>
```
