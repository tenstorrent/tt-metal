# Operation Requirements: qkv_proj_heads_rope

## Definition
- **Formula**: `P = x @ W + b` over token rows; columns split as `col(t, a, d) = t*768 + a*64 + d`;
  `V = P[.., t=2]`; for t in {q, k}: `out = P_t * cos + rotate_half(P_t) * sin` (NeoX rotate-half,
  full head width D = 64, cos/sin row s for position s).
- **PyTorch Reference**:
  ```python
  def qkv_proj_heads_rope_ref(x, w, b, cos, sin, num_heads=12):
      B, _, S, H = x.shape
      D = H // num_heads
      p = x.reshape(B, S, H).double() @ w.double() + b.double().reshape(-1)
      p = p.reshape(B, S, 3, num_heads, D).permute(2, 0, 3, 1, 4)
      c, s = cos[0, 0, :S].double(), sin[0, 0, :S].double()
      rot = lambda t: t * c + torch.cat([-t[..., D // 2:], t[..., :D // 2]], dim=-1) * s
      return rot(p[0]), rot(p[1]), p[2]
  ```
- **Import Path**: `from ttnn.operations.qkv_proj_heads_rope import qkv_proj_heads_rope`
- **Function Signature**:
  `qkv_proj_heads_rope(input_tensor, weight, bias, cos, sin, *, num_heads=12, compute_kernel_config=None, memory_config=None) -> (q, k, v)`,
  x `(B,1,S,768)`, weight `(768,2304)`, bias `(2304,)|(1,2304)`, cos/sin `(1,1,S_tab,64)`, each output `(B,12,S,64)`.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory - the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, ... (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements - a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Golden harness caveat**: as committed, `eval/golden_tests/qkv_proj_heads_rope/helpers.py` passes `packer_l1_acc` to `ttnn.ComputeConfigDescriptor` (TypeError on every `test_op` cell) and `test_regression.py` passes a tuple as `tolerance`. See `verification_report.md`. Until `/golden-tests` fixes them, judge `Done when` on a run with that two-line local patch, and never commit it.

### [x] Phase 0 - Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment = tile_aligned only
- **SUPPORTED placement**: memory_layout = INTERLEAVED; buffer_type (x) = L1; out_buffer_type = L1; table_buffer_type = L1
- **SUPPORTED precision**: fp32_dest_acc_en = True (default HiFi3; any caller fidelity honored)
- **Cores**: 2D grid `grid_m x grid_n` from the selection function, 108/110 on p100a (regime R1, unicast reads)
- **Golden baseline**: 25 supported_pass (17 `test_op` + 8 loose), 175 xfail_expected, 0 loud (harness-patched run)
- **Perf**: 200-283 us on the eight flagged rows vs stock 75-246 us. Slower than stock everywhere; R1 is DM-bound on the x unicast re-read.

### [x] Refinement 1 - x multicast within each M-group (regime R3) + `buffer_type = DRAM`

**Goal**: add `ttnn.BufferType.DRAM` to `SUPPORTED["buffer_type"]` (x placement) through the
design's deferred regime R3: each M-group's `grid_n` cores share x rows, so one core reads each x
K-block (`block_m*k_block` tiles) from its tier and multicasts it into the same
`cb_activation_block` slot on the rest of the group. x then crosses DRAM (or the L1 banks) once, not
`grid_n` times (9x at 8x512, 36x at 2x512). The same channel serves L1-resident x, where the
unicast re-read is today's binding resource. The compute kernel, the CB contracts and the selection
function are unchanged (`op_design.md`, Dataflow Strategy, "R3 x multicast").

**Verifier notes**: scheme-change (a new cross-core topology). It stands alone and is filed first as
the hardest generality item. No implementation skill covers multicast; use the multicast helper
(host `ttnn.Mcast` + `attach()`, kernel `McastArgs` `sender(noc).send()` /
`receiver(noc).receive(round)`, round = K-block) with its own semaphores. Measured how-to:
`ttnn/ttnn/operations/examples/master.md` entries `shared_input_reuse` (T3) and `mcast_topology`
(T2). Receiver-set shape: under the current logical order an M-group is strided by `grid_m`. Either
use the irregular receiver set or reorder logical cores so M-groups are consecutive. The second
breaks the N-group-consecutive layout Refinement 3's W multicast wants, so pick the ordering with
both channels in mind and record it in `op_design.md`. Rotate the injector across the group (catalog:
a single injector serializes). The table reads (4 tiles/row) may ride the same channel; that is
optional. Evidence: x in DRAM is already numerically correct under R1 but costs +13% at 8x512
(317 vs 279 us) and +17% at 2x512 (`l1_ledger.md`, measured placement sensitivity). Shipping DRAM x
without the multicast is the `[~]` escape hatch only, with the multicast filed as `Refinement 1b`.

**Done when**: every `tile_aligned, buffer_type=DRAM, out_buffer_type=L1, table_buffer_type=L1`
golden cell passes; device ns at 8x512 with x in DRAM is within 5% of x in L1; 8x512 with x in L1
improves on the 279 us R1 baseline (the ledger ablation says x reads are most of the K-loop); no
regression on 2x512 / 4x512 / 8x384 (the h_c = 1 / 2 / 4 grid classes); acceptance + golden green;
`l1_ledger.md` data-movement budget and regime table updated (R3 built).

**Outcome**: R3 built with a fixed staggered sender per M-group (rotating senders need rectangular groups; consecutive M-groups on the 11-wide grid are not). x in L1: 8x512 279 -> 265 us, 4x512 215 -> 145 us, 2x512 206 -> 93 us, 8x384 252 -> 252 us (flat). x in DRAM: 8x512 272 us (+2.5% vs L1), 2x512 92 us. 17/17 DRAM-x golden cells pass. Ablation at 8x512: stubbing W reads gives 134 us, stubbing the x path gives 277 us, so the W unicast re-read (12 DRAM crossings) now binds. That is Refinement 3 (R2). Core order is now M-group-consecutive; R2 needs per-N-group `Mcast` objects or a rectangular grid constraint (see `op_design.md`, Work distribution).

### [x] Refinement 2 - Non-aligned S (`h_non_aligned`) + DRAM outputs and tables

**Goal**: add `"h_non_aligned"` to `SUPPORTED["alignment"]`, and `ttnn.BufferType.DRAM` to
`SUPPORTED["out_buffer_type"]` and `SUPPORTED["table_buffer_type"]`. For non-aligned S, the writer
zeroes rows `[S % 32, 32)` of every output tile on the last tile-row of each image (`st == St - 1`)
in L1 before the NoC write (faces `2*(row/16) + {0,1}`, row `row % 16`). Pad rows of q, k and v must be
exactly 0 and finite whatever x's padding or the tables' rows beyond S hold (prompt rule: MUST). The
geometry is already `ceil`-based. The probe shows valid rows correct at 2x37, 13x300, 16x8 and
4096x1; only the zeroing is missing (v padding non-zero now).

**Implementation skill**: /memory-layouts

**Verifier notes**: knob-turns, so all three values are bundled. The DRAM output/table placements
need no kernel change (TensorAccessor is placement-agnostic): probed correct, and traffic-neutral
under R1 (275 / 281 us vs 279 us at 8x512). The skill applies to the non-aligned rule (zero the
last-tile padding in the writer); there is no placement skill. Order after Refinement 1 so DRAM
cells are covered on every placement axis at once. The writer pad-zero sits in the per-row output
path, which is identical under R1 and R3. Cover the multi-block path (`num_blocks_this_core > 1`,
e.g. B=4096, S=1; regime R4 territory) and the golden `test_padded_rows_zero` (x and table padding
= inf) in the acceptance test.

**Done when**: every golden cell in TARGET passes (all alignments x all placements); the
`test_regression.py` padded-row and large-mean cases pass; no device-ns regression on 8x512
tile-aligned all-L1.

**Outcome**: writer zeroes rows `[S % 32, 32)` of each output tile on `st == St - 1` (NoC zero writes, compiled out when S is aligned). Golden `test_golden.py` + `test_regression.py`: 210/210 pass. 8x512 aligned all-L1: 268 us vs 265 us post-R1 (noise band; aligned path unchanged).

### [x] Refinement 3 - W multicast within each N-group (regime R2) on the flagged 8x512 profile

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags 8x512 (x, tables, q/k/v in L1, weight/bias in DRAM,
tables cut to S, default compute config, bf16 TILE tile-aligned) as the primary perf target:
stock 226.3 us, target <= 90 us. Its whole config is already in SUPPORTED. Under R1, W + bias cross
DRAM `grid_m` = 12 times (42.5 MB vs 3.7 MB minimum). Build the design's deferred regime R2: one
DRAM read of each W K-block (`k_block*N_c` tiles, bias as K tile 24) per N-group, multicast to the
group's `grid_m` cores with a rotating sender. Patterns: `master.md` `shared_input_reuse` (T3),
`mcast_topology`, `bank_stagger`. One T3 lever, one phase. No SUPPORTED change. Secondary targets:
the other 4096-token flagged rows (16x256, 32x128; target 90 us) and 8x384 (target 68 us).

**Done when**: measured device ns on 8x512 improves over the post-Refinement-2 baseline, moving
toward 90 us; the soft precision gate (PCC >= 0.9999, rel RMS <= 4e-3 per output) holds; golden
green; no regression across the guard set (one representative per grid class h_c = 1 / 2 / 4 x
x in L1 / DRAM x non-aligned S).

**Outcome**: R2 built on the writer (NOC_1), one `Mcast` + writer descriptor per N-group, folded core layout so both channels' groups are <= 2 rectangles on the 12x9 grid. Device us (x in L1): 8x512 268.7 -> 123.7, 16x256 269.8 -> 123.9, 32x128 269.9 -> 124.1, 8x384 253.0 -> 92.4; 8x512 x in DRAM 125.3. 6x18 (4x512 145, 8x256 145) has no <= 3-rectangle layout and stays R1; 3x36 (2x512 95, 4x256 97) measured slower with W multicast (105-129 us) and is gated off (`W_MCAST_MIN_GROUP = 4`). Bottleneck now at 8x512: stubbing the matmul drops 123.7 -> 66.8 us, so the matmul (HiFi3, 1x4 subblocks, 2200 tile-products/core) is the larger share and the rest (DM, epilogue, writes) ~67 us; that is Refinement 4's knobs (subblock shape, rotary init hoist, L1 overlap). Next for 6x18: a layout with 18-core M-groups in <= 3 rectangles and 6-core N-groups (not attempted: the folded scheme needs grid_n <= grid extent).

### [ ] Refinement 4 - Block / compute knob co-tune on the flagged 4096-token rows

**Type**: perf

**Goal**: once both multicasts land, the op leaves the x-re-read bound, and the compute and pipeline
lamps of `op_design.md` (Perf lamps) become reachable. Target the flagged 8x512 (then 16x256,
32x128, 8x384) toward their goals with the cheaper levers, co-tuned and each kept only on a
measured win:
- hoist the rotary chain's init/reconfig to once per block (`InitReconfigOwner::Caller`; today it
  runs per row x head);
- subblock shape (L6: `block_m` 12/10 split for 2x2 subblocks, or `dst_full_sync_en`);
- L1 overlap (L1: 2 blocks per core, so block 1's matmul overlaps block 0's writes; free once W is
  multicast);
- `k_block` (L5: ragged 8 or 13 with `KBlockInnerDimFn`);
- q/k vs v load balance (L3: cost-weighted M split);
- NoC placement of the W stream (L8).

Catalog: `master.md` `matmul_output_subblock`, `compute_block_size`, `double_buffer`,
`trid_double_issue`, `noc_placement`, `compute_fusion` (T1/T2). No SUPPORTED change.

**Done when**: measured device ns improves on 8x512 (toward 90 us) with the soft precision gate
holding, golden green, and no regression across the guard set (one representative per grid class x
placement x alignment).

### [ ] Refinement 5 - Small-token flagged rows (2048 / 1024 tokens)

**Type**: perf

**Goal**: the flagged rows 4x512 / 8x256 (stock about 124 us, target 46 us) and 2x512 / 4x256
(stock 75-80 us, target 28 us) run on `Mt` = 64 / 32 tile-rows. The selection function picks
h_c = 2 / 1 (grids 6x18 / 3x36). The ingress model behind that choice assumed both multicasts (lamp
L4), and the fixed per-call costs (pipeline fill, one K-loop, epilogue init) weigh most here.
Re-tune the selection function against measured device ns (head-group size, multicast group size
L2, occupancy slack), and attack fill/drain latency with the relevant `master.md` patterns
(`compute_block_size`, `double_buffer`, `bank_stagger`, `width_split`). No SUPPORTED change.

**Done when**: measured device ns improves on 2x512 and 4x256 (toward 28 us) and on 4x512 / 8x256
(toward 46 us), every flagged row beats stock, the soft precision gate holds, golden green, and no
regression on the 4096-token rows or the guard set.
