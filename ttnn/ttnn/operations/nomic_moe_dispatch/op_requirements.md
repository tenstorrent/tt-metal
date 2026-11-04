# Operation Requirements: nomic_moe_dispatch

## Definition
- **Formula**: `out[0, 0, r, :] = table[0, 0, gather_index[0, r], :]` for r in 0 .. R−1, tilized, bit-exact (no arithmetic)
- **PyTorch Reference**:
  ```python
  def nomic_moe_dispatch_ref(table: torch.Tensor, gather_index: torch.Tensor) -> torch.Tensor:
      # table (1, 1, T, 768) bf16; gather_index (1, R) int, entries in [0, T)
      rows = gather_index.reshape(-1).long()
      return table.reshape(-1, 768)[rows].reshape(1, 1, rows.numel(), 768)
  ```
  (golden: `check_gather` in `eval/golden_tests/nomic_moe_dispatch/helpers.py`, bit-for-bit)
- **Import Path**: `from ttnn.operations.nomic_moe_dispatch import nomic_moe_dispatch`
- **Function Signature**:
  ```python
  nomic_moe_dispatch(
      table: ttnn.Tensor,          # (1, 1, T, 768) bf16 ROW_MAJOR interleaved, 1000 <= T <= 4096
      gather_index: ttnn.Tensor,   # (1, R) uint32 ROW_MAJOR, DRAM interleaved (one page), R % 32 == 0, 2240 <= R <= 8512
      *,
      memory_config: ttnn.MemoryConfig | None = None,             # output buffer: L1 (default) or DRAM
      compute_kernel_config: ttnn.ComputeConfigDescriptor | None = None,
  ) -> ttnn.Tensor                 # (1, 1, R, 768) bf16 TILE in output_memory_config(R, device, buffer_type)
  ```
  Also exported: `output_memory_config(rows, device, buffer_type)`, `default_compute_kernel_config()`, `circular_buffer_bytes(rows)`.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Bit-exactness is a hard constraint for every refinement**: the golden check is a bit-for-bit compare, and the prompt's Rules (one `generic_op`, index read on device, never CT args or cached state, no ttnn op sequencing) apply to every new regime.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED placement axes**: table_buffer = [l1], output_buffer = [l1]
- **SUPPORTED shape-derived axes**: window ∈ {aligned, ragged}
- **Regime**: R1 `local_tilize_two_noc`. The output is ND-sharded in the op's own L1 layout (shard = 32 rows × w tiles, round robin over 110 cores). Two reader RISCs (rows 0-15 on NOC0, 16-31 on NOC1) do stateful one-packet gathers, and the fast tilize writes straight into the core's own shard. There is no writer.
- **Cores**: all 110 (11×10); per-core work derived on device.
- **Compute config**: `default_compute_kernel_config()` (fast tilize needs `dst_full_sync_en=False`); values are independent of the config, and both non-default configs are measured bit-exact.
- **Golden baseline**: 29 / 62 cells passing (`supported_pass` 29, `xfail_expected` 33, loud categories 0)
- **Perf (device kernel ns, flagged shape R=8416)**: 9.62 us. Acceptance 15.0, stretch 11.0, incumbent 21.85.

### Deferred (nomic-opt, 2026-11-04): DRAM output (regime R3 `output_dram`)

> Not queued. The model places Xg in L1 at every window this op takes (1000 to 4096 tokens) and keeps the incumbent routed_combine wherever it asks for DRAM; the run's scope is the model's configs only (no generality work). The refinement runner skips this heading on purpose.

**Goal**: add `"dram"` to `SUPPORTED["output_buffer"]`. This moves the 11 `table_buffer=l1, output_buffer=dram` golden cells from xfail to pass. Xg is written DRAM-interleaved in TILE layout (`output_memory_config(R, device, DRAM)` already returns `ttnn.DRAM_MEMORY_CONFIG`). This is a **scheme-change**, so it stands alone. Writes now take a NoC and a RISC:
- compute packs into a ring `cb_output_tiles` (≥ 2 blocks) instead of an output-backed CB;
- a writer kernel `nomic_moe_dispatch_writer.cpp` takes RISCV_1 / NOC1 and writes each block's `w` tiles to their DRAM pages (tile id = `tile_row·24 + piece·w + j`), batched per block with one barrier (or a trid ring), never one barrier per tile;
- the NOC0 reader reads all 32 rows of a block.

**Implementation skill**: /interleaved-parallel

**Verifier notes**:
- **The L1 regime must stay untouched.** It is the perf target. Select the regime on the host from `memory_config.buffer_type`, as distinct kernel sources or CT branches. Do not run a shared code path that pessimizes R1.
- **The work split is not pinned by the L1 plan in this regime.** Xg is in DRAM, so the selection rule's +8 KB budget does not apply. Choose `w` and the per-core assignment by measurement. The measured R1 data says k = 2 (768 B pieces) is the cheapest per row, and whole rows (k = 1) drain at 1-2 blocks a core. The incumbent's notes say 8-tile units with 2 slots were best for *its* one-reader design.
- **L1 footprint.** The output ring is new L1, so `circular_buffer_bytes` must report it. It currently takes only `rows`. Extend it backward-compatibly, e.g. `circular_buffer_bytes(rows, buffer_type=L1)`, and keep the validate() budget check per regime. Add the new CB rows to `l1_ledger.md`, with a data-movement budget row for the +R·1536 B of DRAM writes.
- Ordered before Refinement 2 because it is the structural change (hardest first), and Refinement 2's dram/dram cells need it.
- The incumbent routed_combine serves DRAM Xg today. Record this regime's device-ns against it at R = 2272 / 4320 / 8416 in the changelog, so the model can adopt it per shape where it is faster.

**Done when**: all 11 `output_buffer=dram, table_buffer=l1` golden cells pass bit-exactly. `verify_supported` has 0 loud categories. The acceptance and regression tests are still green. The L1-regime device-ns at R = 8416 / 4320 / 2272 does not regress.

### Deferred (nomic-opt, 2026-11-04): DRAM table (regime R4 `table_dram`)

> Not queued, for the same reason: the dispatch table is in L1 for every one-window layer in this op's range, and the incumbent serves the DRAM case.

**Goal**: add `"dram"` to `SUPPORTED["table_buffer"]`. This moves the remaining 22 xfail cells to pass: `table_buffer=dram` with `output_buffer=l1` (11) and with `output_buffer=dram` (11). It is a **knob-turn** on the reader's bank flavor. The per-read address becomes `dram_bank_to_noc_xy[noc][bank]` / `bank_to_dram_offset[bank]` with `bank = id mod NUM_DRAM_BANKS` (a constant divisor), replacing the L1 table. It applies to every reader of both output regimes (R1's two readers and Refinement 1's reader). DRAM alignment holds: pieces are multiples of 256 B at offsets that are multiples of 256 B, in a 1536 B page.

**Verifier notes**:
- There is no skill match (the bank-flavor switch is reader-internal). It is filed standalone only because its natural bundle, Refinement 1, is a scheme-change that stands alone.
- Make the bank flavor a single CT arg derived on the host from `table.memory_config().buffer_type`. Do not fork a second copy of the gather body: `nomic_moe_dispatch_gather.hpp` is shared by both readers on purpose.
- **Re-measure `reads_in_flight` for this regime.** It was flat from 8 to 16 on the L1 table, but DRAM latency is higher and the table is spread over far fewer banks, so the knee will move. Keep it a per-regime plan knob, not a global constant change. Also check the NoC choice for DRAM reads against the catalog's `noc_placement` entry. The incumbent serves DRAM tables today; record device-ns against it in the changelog.
- `validate()` must refuse nothing new here except what stays out of range (T > 4096 multi-window tables remain out of the op's range).

**Done when**: all 22 `table_buffer=dram` golden cells pass bit-exactly. `verify_supported` shows 62 / 62 `supported_pass`, 0 `xfail_expected`, 0 loud categories. The L1/L1 regime does not regress at R = 8416 / 4320 / 2272.

### [x] Refinement 3 — Speed up the perf-flagged profile: gather_index fetch topology + issue stagger

**Type**: perf

**Goal**: the target is the `feature_spec.LOOSE_CASES` entry flagged `# attention: PERF FOCUS`: T = 4096, R = 8416 (8×512), bf16, `table_buffer=l1`, `output_buffer=l1`, default compute config, real-text routing, 96 KB of every bank reserved. Its extras give `achievable_ns = 21850` at 1350 MHz and `minimum_expected_speedup = 1.457`, i.e. a 15.0 us ceiling (11.0 us stretch). It is at **9.62 us** today; reduce it further. The lever is the gather_index fetch, the design's `deferred` broadcast row: gather_index does not vary along `column`, yet every one of the k piece cores of a tile row reads the same ids, and all ~220 reader RISCs queue 64 B requests on **one** DRAM bank. Candidate topologies, to be built and measured, not inferred:
- (a) R7 `ids_sibling_forward`: the piece-0 core reads a tile row's ids once and forwards them 1 hop, under a semaphore, into the siblings' existing ids scratch;
- (b) read the whole gather_index once (33 KB at R = 8416) and multicast it to the grid with `ttnn.Mcast` / `McastArgs` (catalog `shared_input_reuse`, `mcast_topology`), so every core's ids are local.

Patterns: `ttnn/ttnn/operations/examples/master.md` entries `shared_input_reuse` (⭐⭐⭐), `mcast_topology`, `bank_stagger`. No SUPPORTED change.

**Verifier notes**:
- **A cheaper ids path must ship with an explicit per-core issue stagger.** The verifier's ablation stubbed the ids fetch with local ids. R = 2272 / 4320 sped up by ~1.7 / ~2.5 us, but **R = 8416 slowed from 9.66 to 12.8-14.3 us**. At the largest R the ids round trip is what spreads the grid's first row reads; removed naively, 220 RISCs burst at once and congest. Likewise, raising `ids_lookahead` to 2 / 4 regressed R = 8416 to 11.1 / 12.1 us. Stagger the start (e.g. start each core at block `c mod num_blocks`, or a per-core delay in the first block's issue) and measure the pair together.
- `reads_in_flight` is flat from 8 to 16 at every perf shape. Do not spend this phase on it.
- This is a scheme-change (T3), so it is a whole phase on its own. Refinement 4's path-aware NoC split needs both readers to see all 32 ids of a block, which is cheap only once ids are local. That is why this phase comes first.
- If the multicast form is chosen, the ids scratch grows from a 2-page ring to the full index (R·4 B ≤ 34 KB, one copy a core shared by both readers). Update `circular_buffer_bytes` and `l1_ledger.md`. The CB total stays ≤ ~157 KB, under the 256 KiB cap, but check it against the model's L1 plan (`l1_reserved_kb`).

**Done when**: measured device-ns improves on the flagged R = 8416 shape (moving it further below 9.62 us toward the read-bandwidth floor), bit-exactness still holds, the golden suite is green, and nothing regresses across the config-spanning guard set: one representative per distinct kernel path × placement. That is k ∈ {2, 3, 4, 6} (e.g. R = 8416, 4320, 7392, 2656), the two guard loose cases R = 2272 / 4320 (expected to *improve*, since the ids fetch is their critical path), (no DRAM regime: Refinements 1 and 2 were deferred).

**Outcome**: built a third topology, R10 `ids_row_mcast`. Each grid row's x = 0 core reads its row's contiguous round span of ids and multicasts it along the row, with the per-core issue stagger and a k = 2 bytes-in-flight cap. Results (device us, mean of 5 calls, same harness): R = 8416 9.81 → 9.13-9.22 (loose case 9.12; was 9.62 at verification); guards 2272 4.61 → 3.73-3.75, 4320 7.02 → 6.94-7.03 (parity), 7392 14.86 → 13.98, 2656 8.89 → 8.56; no shape slower (6368 at parity). Both candidate forms were built and measured first and lost at R = 8416: the whole index multicast to the grid (14.4 us; a 7 KB grid multicast costs ~1-1.25 us under load and the sender becomes the tail) and grid round 0 plus per-reader fetches (9.84). Bottleneck now: the start (~1.3 us, one DRAM round trip on the gather_index bank plus a row multicast) and the per-block gather (~1.1 us a block on the NOC0 reader, ~40% longer than NOC1's at k = 3). Next: Refinement 4's lamp L1 (path-aware / rebalanced NOC0 / NOC1 row split) and the L7 tail tilize, plus the k = 2 NoC collapse mechanism, which is understood only empirically (the stagger and the 6-read cap avoid it).

### [x] Refinement 4 — Speed up the perf-flagged profile: NoC / bank contention levers

**Type**: perf

**Goal**: the target is the same flagged loose case (R = 8416, table and Xg in L1, the full contract above). After Refinement 3, attack the row-read contention it exposes. All of these are knob-turns on the R1 block surface, bundled because each one is cheap:
- **Lamp L1, path-aware row-to-NoC split.** Read each row on the RISC whose NoC has the shorter response path from the row's bank, instead of the positional 16 / 16 halves. The shorter of the two torus paths averages about a third fewer hops. A host-built per-core bank mask, or an in-kernel test on `l1_bank_to_noc_xy`, picks the NoC; both readers then walk all 32 ids and skip the other's rows.
- **Lamp L6, issue order.** Start each core at a different block or row offset (`bank_stagger`), if Refinement 3's stagger did not already settle it.
- **Lamp L3 / L7 tail co-tune.** Tune `row_slots` and the last block's tilize width (two half-width tilize calls) if the profile shows the final tilize on the tail.

Patterns: `ttnn/ttnn/operations/examples/master.md` entries `noc_placement`, `bank_stagger`, `split_reader`. No SUPPORTED change.

**Verifier notes**:
- This depends on Refinement 3, because lamp L1's cost of every reader seeing all 32 ids disappears once ids are local.
- The aliased-CB lockstep must survive an uneven split. Both readers still push exactly `w` pages per block, so a per-row NoC choice changes only which RISC writes which rows of the slot, never the push quanta.
- Per-row cost tracks k (k = 2 ≈ 1.14 ns/row, k = 3 ≈ 1.5-1.7, k = 4 ≈ 2.0 measured). Check the levers at k = 4 too (R = 7392, 14.9 us), where 384 B pieces make contention worst, but the target and gate stay at R = 8416.

**Done when**: measured device-ns improves on the flagged R = 8416 shape versus the post-Refinement-3 number, bit-exactness and the golden suite hold, and nothing regresses across the config-spanning guard set (one per k ∈ {2, 3, 4, 6}, the R = 2272 / 4320 guard loose cases).

**Outcome**: R = 8416 9.17 → 8.46-8.49 us (eval harness, mean of 5; loose case 8.46, was 9.12). Per call (median of 14, Tracy) it went 8.65 → 7.97. Guards per call: 4320 6.45 → 5.30, 7392 13.40 → 9.56, 2656 8.02 → 5.72; 2272 at parity (3.29 → 3.31). Ablation showed the readers were **RISC-issue-bound, not NoC-bound**: with every row read stubbed, a block body still took ~0.85 of ~1.1 us. So the first lever was the per-row issue path: a compile-time NoC, credit-batched in-flight accounting, and non-volatile ids. Faster issue brought the congestion collapse to k = 3, and the contention levers then paid off: one ~4 KB bytes-in-flight rule, the L6 issue order re-measured ((5, 1) at k = 2, (7, 1) otherwise), and the L1 sender-column rebalance (the ids senders' reader sheds 2 rows to the other reader).
- The L1 path-aware split is built, but parked (`ROW_SPLIT_PATH = False`). Its per-block owner pass sits on the issue path and costs more (+1.4-3 us) than the assignment changes (−0.6 / +0.14 us at equal cost).
- L3 / L7 were not built: `row_slots` is already the whole assignment, and splitting the last block's tilize would double that block's reads on an issue-bound RISC.
- Bottleneck now: per-row issue (~45 cycles a row) plus the ~1.3 us start and the 0.31 us tail tilize. The collapse cliff caps reads in flight just above the current values, and one collapsed call (k = 3) was seen in ~140 since.
- Next: hide the address math (precompute a block's sources and coordinates in the in-flight wait slots), which would also make the path-aware split free; understand the collapse mechanism itself. Not done here because each needs a restructured issue loop and its own collapse-rate study.
