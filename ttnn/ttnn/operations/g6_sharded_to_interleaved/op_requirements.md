# Operation Requirements: g6_sharded_to_interleaved

## Definition
- **Formula**: `out[i] = input_tensor[i]` for every logical index i, bit for bit. Output page
  `p = a * C + b` holds the input tile in tile row a and tile column b (C = tile columns of the 2-D view
  whose rows are all dims but the last). Output: same shape and dtype, TILE, interleaved in
  `memory_config.buffer_type`, a new buffer. The input is unchanged.
- **PyTorch Reference**: `def ref(input_tensor, memory_config=None): return torch.clone(input_tensor)`
  (for bfloat8_b the reference is the host readback of the device input as stored).
- **Import Path**: `from ttnn.operations.g6_sharded_to_interleaved import g6_sharded_to_interleaved`
- **Function Signature**:
  `g6_sharded_to_interleaved(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor`.
  Both arguments are positional and have no defaults. There is no compute configuration, no dtype
  parameter and no output parameter (fixed by the contract).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED in_scheme**: ["width_sharded"]
- **SUPPORTED buffer**: ["dram"]
- **Regime**: `owner_writes`. Each shard core writes its resident shard straight to its interleaved
  pages: one NoC write per tile, zero reads, one barrier. `cb_input_shard` is zero-copy on the input.
  There is no reader and no compute kernel.
- **Cores**: the input's own shard grid. p2: 32 cores, 1 range. p1: 103 cores, 2 ranges.
- **Compute config**: none. The op does no arithmetic.
- **Golden baseline**: 13 / 17 cells passing (13 supported_pass, 4 xfail_expected, 0 loud) per
  `verifier_report.json`
- **Device baseline** (kernel ns, 1 lane): p2 1494.5, p1 2110. Trace replay ns/op: p2 2845, p1 5952.
  Numbers and method are in `verification_report.md`.

### [x] Refinement 1 — Claim the Panoptic production contract: bfloat8_b, height_sharded, L1 output

**Goal**: add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`, `"height_sharded"` to `SUPPORTED["in_scheme"]`
and `"l1"` to `SUPPORTED["buffer"]`. After this, every TARGET value is in SUPPORTED. The op then runs
the full config of the perf-flagged case `LOOSE_CASES[0]`: (1, 128, 256, 32) bfloat8_b TILE, height
sharded (320, 32) on (0,0)-(10,8) + (0,9)-(3,9), row-major, written to L1 interleaved. All three values
are knob-turns in the design's "TARGET axes as blocking" table. Page size is a CT arg read off the
dtype, the block rectangle comes from the shard spec, and the bank set comes from the output
`TensorAccessorArgs`. No kernel change is expected and no EXCLUSIONS entry is allowed.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- Scope of the skill: only the `SUPPORTED["dtype"]` widening and its precision-matrix contract, which
  is **exact** for both dtypes (`helpers.TOLERANCES`). Do **not** expose `compute_kernel_config` or any
  intermediate-CB format: the op has no compute kernel, and the contract fixes the two-argument
  signature. bfloat8_b tiles must move as raw 1088 B pages with their shared exponents (contract MUST).
  The writer already does this.
- `in_scheme` and `buffer` are memory placement, which has no skill. The native path already exists:
  `cb_input_shard` placed on the sharded input via `ttnn.cb_descriptor_from_sharded_tensor` (local
  shard, zero copy), and the output bank set from `TensorAccessorArgs(output)`. Keep that path and do
  not re-read the local shard through an accessor.
- Bundled as one refinement on purpose. Every height_sharded golden cell is also bf8b and L1, so any
  split leaves a value with no cell to prove it, and the perf-flagged case needs all three. The
  unlock is already built to the performance-conformance bar: it fills all 103 shard cores and issues
  every write before a single barrier.
- The evidence that this is a pure claim: `test_g6_sharded_to_interleaved_impl_debug.py` passes
  bit-exact through `_dispatch` on p1, the p1 other spec, p2-to-L1, a height-sharded padded-image
  case, and all knob sets. The precision baseline is also bit-exact on both p1 specs. Once p1 is in
  SUPPORTED, the precision baseline switches to the public entry point by itself.
- BB-05: for an L1 output the host L1 bound (`ceil(T / n_l1_banks) * page_size <= 1347776`, otherwise
  `RuntimeError`) is the refusal path. Every INPUTS shape fits even at one bank, so no golden cell
  exercises it.
- The four-chip p2 mesh cannot be tested on a one-chip board. Do not make it part of this refinement.

**Done when**: `verify_supported` reports supported_pass 17, xfail_expected 0, and every loud
category 0. The four former xfail cells (p1 production, p1 loose, p1 other spec, p2 to L1) pass bit
for bit. The acceptance test is 17 passed / 0 xfailed. impl_debug, trace replay and the precision
baseline stay green. `SUPPORTED == TARGET` and `EXCLUSIONS == []`.

### [x] Refinement 2 — Speed up the perf-flagged profile: the p1 write issue path

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` flags Panoptic-DeepLab case 42 as the mandatory perf target:
(1, 128, 256, 32) bfloat8_b, height sharded (320, 32) on 2 ranges / 103 cores, written to L1
interleaved, exact gate. Its goal is a device kernel time below the stock incumbent's **2.05 us**. The
roofline figure the lead gave is 0.00 us and unusable. Phase 0 measures **2110 ns** (median, 1 lane),
so the op does not yet meet the goal. Speed up the per-core write issue path with the patterns in
`ttnn/ttnn/operations/examples/master.md`. No SUPPORTED change. Relevant levers, cheapest first:
- **L1 bank spread (`bank_stagger`, ⭐⭐).** Core k writes pages `10k..10k+9`, so at write step j all
  103 cores hit banks `(10k + j) mod 110`: only 11 of the 110 L1 banks, about 9.4 writers each. Rotate
  each core's issue order so the concurrent writes spread over all banks. Any rotation offset that
  separates the cores sharing `10k mod 110` (for example by `k // 11`) spreads them. The implementer
  measured an earlier stagger slower (2282 and 2360 ns against 2118), but it took a modulo on every
  row. Measure a modulo-free form: a host-supplied start offset and a two-segment walk.
- **Incremental bank walk** (the design's second lamp, ⭐). Replace the per-page
  `TensorAccessor::get_noc_addr(page)` div/mod with a bank index that advances by 1 and wraps, and an
  offset that advances by `page_size` on each wrap. The implementer's ablation puts the upper bound
  at about 112 ns.
- **Write lanes** (`split_reader` / `noc_placement`, ⭐ knob): already measured, and not open on its
  own. Kernel time drops 2110 to 1875 ns, but trace replay rises 5952 to 6229 ns/op. The NoC0 lane is
  the long pole (1873 against 1462 ns). Turn it on only if the trace-replay gate below holds.

**Verifier notes**: requires Refinement 1, because the flagged config must be in SUPPORTED, and
measure it through the public entry point. Measure the exact flagged config, never a bf16 or
width-sharded stand-in. Use `test_g6_sharded_to_interleaved_perf.py` (`test_eager_calls` under
`--profile` for kernel ns, `test_trace_replay_ns_per_op` for replay ns/op). Any rotation offset or
lane choice is a host-derived per-core value or knob. It must stay in the program memo key (BB-07)
and be visible in `describe_plan` (BB-08). Refinement 3 may later derive these per-core values on
device, so keep the offset a function of the shard index k.

**Done when**: p1 device kernel ns improves beyond the 2–3% noise band from 2110 ns (target: under
2050 ns). The output stays bit-exact. p1 trace-replay ns/op does not regress from about 5952. The
golden suite is green. There is no regression on the guard set: p2 (bf16 width to DRAM, 1494.5 ns
kernel / about 2845 ns/op replay) and p2 to L1 (bf16 width to L1).

**Outcome**: p1 kernel 2119 → 1712 ns (median of 1632, 1711.5 and 1714.5 over three runs; −19%, under the 2050 ns
target) with `default_knobs` choosing 2 lanes + `bank_stagger` (L1 destination, ≥ 8 shard pages). p1 replay is
5952 → 5964 ns/op (flat, +0.2%). Output is bit-exact. The guard set runs its unchanged 1-lane plan (p2 1499.5 ns /
2842 ns/op, p2 to L1 1271 ns / 2812 ns/op, both within noise). The stagger is only a win on two NoCs (1 NoC: 2239
vs 2139 ns). `num_write_nocs = 2` is parked as a knob (one kernel, dynamic NoC: 1804 ns; costs ~16 cycles/write
over dedicated mode). The incremental bank walk is capped at ~30 ns by ablation and not built (it needs bank-level
addressing that `TensorAccessor` lacks). The p1 wall is now kernel start skew (~620 cycles) plus NoC0-lane
barrier tails on cores at physical x=1–2, y=3–5. Next: pick each core's NoC by position rather than by lane. Not
tried here because this refinement's levers were exhausted and that is a new traffic-placement scheme.

### [ ] Refinement 3 — Speed up the perf-flagged profile: per-call dispatch cost on p1

**Type**: perf

**Goal**: on the same flagged case (`LOOSE_CASES[0]`, p1 to L1), a traced model pays **5952 ns per
call against a 2110 ns kernel**, so about 3.8 us is dispatch and firmware. Of that, about 2.5 us is
more than p2 pays (1.35 us) across 71 more cores and one more range. FW duration exceeds kernel
duration by 930 ns on p1 against 520 ns on p2. The op-side driver is the per-core unique RT args:
6 words x 103 cores, written core by core on every replay. Common RT args are written once per range.
Cut the per-call cost without growing the kernel. The candidate lever (⭐⭐⭐, a host/kernel contract
change) is to derive each core's block, and Refinement 2's rotation offset, on device from the core's
logical coordinates plus common RT args: ordered shard ranges, orientation, shard and tensor tile
extents. This removes per-core RT args, or shrinks them to one word. First confirm the headroom with
an ablation, for example the same program with its per-core RT args zeroed and a stubbed kernel,
before building it. No SUPPORTED change.

**Verifier notes**: ordered after Refinement 2, because the derivation has to reproduce whatever
per-core offset Refinement 2 keeps. The shard-to-core map must stay exactly the buffer's own walk:
ranges in their given order, each walked row-wise or column-wise by orientation (the design's "Shard
-> core" row). Cores at k >= num_shards stay idle and still drain with the barrier (BB-03). After
this lands, re-measure `num_write_lanes = 2`: with dispatch cost lower, its 235 ns kernel gain may
show in replay. Keep the kernel text small (BB-04, BB-18) and the zones in place (BB-21).

**Done when**: p1 trace-replay ns/op improves beyond the noise band from about 5952 ns, with p1
device kernel ns not regressing from its post-Refinement-2 value and the output bit-exact. The golden
suite is green. There is no regression on the guard set (p2 to DRAM and p2 to L1, kernel ns and
replay ns/op). The multi-range and column-major specs (the p1 and p2 other specs, the cache-order
loose case) still pass bit for bit.
