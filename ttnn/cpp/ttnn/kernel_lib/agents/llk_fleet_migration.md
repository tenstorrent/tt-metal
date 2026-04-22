# LLK Fleet Migration Skill

Migrate a family of compute kernels to `compute_kernel_lib` helpers systematically.
Operates on all kernels under `ttnn/cpp/ttnn/operations/` (or a scoped subdirectory)
and drives the Survey → Classify → Gap Map → Fix → Sweep cycle.

---

## Entry

Invoked as `/fleet-migration [op-dir]`.
`op-dir` is optional; omit to scan all operations.

**First action**: read `ttnn/cpp/ttnn/kernel_lib/migration_log.jsonl` for a state record.

```json
{"type": "state", "phase": 1, "phase_name": "classify", "progress": {"done": 12, "total": 125}, "ts": "..."}
```

- If a state record exists → resume at the recorded phase and progress.
- If no state record → this is a fresh run; start at Phase 0.
- If the file doesn't exist yet → create it during Phase 0.

---

## JSONL file locations (all under `ttnn/cpp/ttnn/kernel_lib/`)

| File | Role | Written by |
|------|------|-----------|
| `coverage.jsonl` | One record per kernel: status + raw patterns | Phase 0 (regenerated) |
| `migration_log.jsonl` | Append-only: attempts + H decisions + state | Phases 1, 4, and checkpoints |
| `feature_gap_map.jsonl` | One record per gap: description + blocked kernels + status | Phases 2, 3 |

---

## JSONL schemas

### coverage.jsonl — one line per kernel
```json
{
  "type": "kernel",
  "path": "normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp",
  "status": "MIGRATED",
  "helpers": ["binary_op"],
  "raw_remaining": [],
  "updated": "2026-04-22"
}
```
`status` ∈ `MIGRATED` | `PARTIAL` | `RAW`
`helpers` = names actually included (`binary_op`, `sfpu`, `reduce`, `tilize`, `untilize`, `dest`)
`raw_remaining` = patterns still raw (`add_tiles`, `mul_tiles`, `bcast_tiles`, `sfpu_raw`, `binary_dest_reuse`)

### migration_log.jsonl — append-only, multiple record types

**Attempt record** (written after each kernel migration or classification):
```json
{
  "type": "attempt",
  "kernel": "normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp",
  "status": "MIGRATED",
  "commit": "a77de3b3793",
  "blockers": [],
  "stages_migrated": ["all"],
  "stages_blocked": [],
  "tests_passed": 1080,
  "notes": "",
  "ts": "2026-04-22"
}
```
`status` ∈ `MIGRATED` | `PARTIAL` | `BLOCKED` | `DEFERRED`
`blockers` = list of gap IDs blocking remaining stages

**State record** (upserted, not appended — replace the most recent state record):
```json
{
  "type": "state",
  "phase": 1,
  "phase_name": "classify",
  "progress": {"done": 12, "total": 125},
  "ts": "2026-04-22T10:31:00"
}
```
Update this record at the start and end of every phase and after each kernel in Phase 1/4.

**H checkpoint record** (written when human decides at a checkpoint):
```json
{
  "type": "h_checkpoint",
  "phase": 2,
  "decision": "approved",
  "approved_gaps": ["GAP-1", "GAP-4"],
  "deferred_gaps": ["GAP-2", "GAP-3"],
  "notes": "GAP-2 too risky without CB API changes",
  "ts": "2026-04-22T11:00:00"
}
```

### feature_gap_map.jsonl — one line per gap, replaced not appended
```json
{
  "type": "gap",
  "id": "GAP-1",
  "name": "Absolute tile index on A/B operand",
  "description": "binary_op only supports sequential tile indices; kernels use index = w + subblock_offset",
  "blocked_kernels": [
    "normalization/layernorm/device/kernels/compute/layernorm.cpp"
  ],
  "n_blocked": 40,
  "complexity": "Medium",
  "fix_approach": "Add cb_tile_idx runtime field to binary_op B side (DestReuseOp precedent exists)",
  "status": "OPEN",
  "fixed_in": null,
  "priority": 1
}
```
`complexity` ∈ `Low` | `Medium` | `High` | `VeryHigh`
`status` ∈ `OPEN` | `FIXED` | `DEFERRED`
`priority` = rank by ROI (1 = highest); recomputed each Phase 2

---

## Phase 0 — Survey

**Goal**: regenerate `coverage.jsonl` from the current state of the codebase.

1. Grep for all `.cpp` files under the target directory containing eltwise FPU or SFPU patterns:
   - FPU: `add_tiles`, `mul_tiles`, `sub_tiles`, `_tiles_bcast`, `binary_dest_reuse_tiles`
   - SFPU: `sfpu_helpers`, `sfpu_chain(`, `sfpu_pipeline`, and raw SFPU tile ops (`exp_tile`, `rsqrt_tile`, `tanh_tile`, etc.)
2. For each found file, check which helper headers it includes.
3. Classify:
   - Includes `binary_op_helpers` or `sfpu_helpers` AND no raw patterns → `MIGRATED`
   - Includes either helper AND has raw patterns → `PARTIAL`
   - Neither helper → `RAW`
4. Write `coverage.jsonl` (overwrite entirely).
5. Write state record: `{"type":"state","phase":0,"phase_name":"survey","progress":{"done":1,"total":1},...}`.
6. Print summary: `Survey complete: N kernels (M MIGRATED, P PARTIAL, Q RAW)`.

→ Advance to Phase 1.

---

## Phase 1 — Classify

**Goal**: for every `RAW`/`PARTIAL` kernel in `coverage.jsonl`, audit and record blockers.

Update state: `{"phase":1,"phase_name":"classify","progress":{"done":0,"total":N}}`.

For each unprocessed kernel (no `attempt` record in `migration_log.jsonl`):

1. Read the kernel file.
2. Apply the audit checklist (from `llk_helpers_hq.md` Step 1):
   - Raw LLK calls → classify as covered / chain-primitive / not-covered
   - CB lifecycle per operand → map to Load/BinaryInputPolicy
   - Control-flow shape → single chain / branched chains / split windows
3. Run the blockers checklist (from `llk_helpers_hq.md` Policy Mapping section):
   - Non-zero absolute tile index? → GAP-1
   - Cumulative wait? → GAP-2
   - Non-sequential output pack? → GAP-3
   - Asymmetric wait/process/pop? → GAP-4
   - In-place CB capacity=1? → GAP-5
   - Runtime fill_tile constant? → GAP-9 (FIXED if FillScalar added)
   - Missing chain element? → note which element
   - Mid-chain reinit needed? → GAP-14
4. Determine: **fully migratable now**, **partially migratable**, or **fully blocked**.
5. If fully or partially migratable: migrate clean stages now (HQ Steps 3–5), commit, write `attempt` record with `status: MIGRATED` or `PARTIAL`.
6. If blocked: write `attempt` record with `status: BLOCKED` and `blockers` list.
7. Update state progress counter after each kernel.

When all kernels processed → advance to Phase 2.

---

## Phase 2 — Gap Map

**Goal**: aggregate blockers into a ranked gap map and get human approval on what to fix.

1. Read all `attempt` records from `migration_log.jsonl` with `status: BLOCKED`.
2. Aggregate blocker IDs → for each gap ID, collect the list of blocked kernels.
3. Compute ROI priority: `priority = n_blocked / complexity_score` where
   `Low=1, Medium=2, High=4, VeryHigh=8`.
4. Write `feature_gap_map.jsonl` (replace existing).
5. Present the ranked gap table to the human:

```
Gap map (ranked by ROI):
  GAP-1  [OPEN] Absolute tile index on A/B  — blocks 40 kernels  Complexity: Medium
  GAP-4  [OPEN] Asymmetric wait/process/pop — blocks  9 kernels  Complexity: Medium
  GAP-2  [OPEN] Cumulative wait policy      — blocks 15 kernels  Complexity: High
  ...
  GAP-12 [FIXED] TanhDerivative element     — blocks  0 kernels  fixed_in: 31f3de5
```

**H CHECKPOINT — Phase 2**:
- Ask human: "Which gaps should be fixed this round? Which to defer or accept permanently?"
- Human responds with approved list and any notes.
- Write `h_checkpoint` record with `approved_gaps`, `deferred_gaps`, `notes`.
- Update `status: DEFERRED` for deferred/accepted gaps in `feature_gap_map.jsonl`.

→ Advance to Phase 3 for each approved gap.

---

## Phase 3 — Fix Gaps

**Goal**: fix each approved gap; validate the fix.

For each gap in `approved_gaps` from the Phase 2 checkpoint:

### Fast path — use when ALL of these hold:
- Fix adds only new chain element(s) inheriting `UnaryOp`/`BinaryOp`/`TernaryOp` CRTP
- No new policies, no CB lifecycle changes, no API surface changes
- ≤ 30 lines total across `.hpp` + `.inl`
- A direct LLK API exists to wrap (no compound logic)

Fast path steps:
1. Add element struct declaration to appropriate `sfpu_*.hpp`
2. Add `#include` for the LLK header
3. Add `init()` + `call()` implementation to `sfpu_*.inl`
4. `./build_metal.sh`
5. One targeted device test calling the new element directly in a chain
6. Update gap `status: FIXED`, `fixed_in: <commit>` in `feature_gap_map.jsonl`

### Full pipeline path — use when fast path criteria are not met:
1. Invoke `llk_helpers_pipeline.md` as a skill in update mode
2. The pipeline handles proposal → validation → implementation
3. After pipeline completes: update gap `status: FIXED` in gap map

**H CHECKPOINT — Phase 3 (per gap)**:
After fix is committed and tested, present to human:
```
GAP-N fixed in <commit>. Tests: X passed.
Authorize sweep for kernels blocked by this gap? [yes/no/notes]
```
Write `h_checkpoint` record with decision.
If not authorized: stop, do not sweep, leave in current state.

→ After all approved gaps fixed and authorized: advance to Phase 4.

---

## Phase 4 — Sweep

**Goal**: migrate every kernel that is now fully unblocked.

Update state: `{"phase":4,"phase_name":"sweep","progress":{"done":0,"total":N}}`.

1. Read `migration_log.jsonl` for all `BLOCKED` attempt records.
2. For each kernel: check if all its `blockers` are now `FIXED` in `feature_gap_map.jsonl`.
3. For each newly-unblocked kernel: run HQ Steps 3–5 (write migration → test → commit).
4. Write new `attempt` record with `status: MIGRATED` or `PARTIAL` (if some stages still blocked by other gaps).
5. Update `coverage.jsonl` status for that kernel.
6. Update state progress after each kernel.

One commit per kernel. Bisectability requirement: never batch multiple kernels in one commit.

→ When all unblocked kernels processed: advance to Phase 5.

---

## Phase 5 — Iterate

**Goal**: decide whether to continue or stop.

1. Re-read `feature_gap_map.jsonl`. Count OPEN gaps by complexity bucket.
2. Present stopping condition check to human:

```
Remaining open gaps:
  Low complexity:    0  gaps  blocking  0 kernels
  Medium complexity: 2  gaps  blocking 15 kernels
  High complexity:   3  gaps  blocking 55 kernels

Continue? (back to Phase 2 with updated priorities) [yes/no]
```

- If yes: write state `{"phase":2,...}`, go to Phase 2.
- If no: write state `{"phase":5,"phase_name":"done",...}`, print final summary and stop.

**Final summary format**:
```
Fleet migration complete.
  MIGRATED:  N kernels  (M in this run)
  PARTIAL:   P kernels  (Q newly partial)
  BLOCKED:   R kernels  (blocked by OPEN gaps)
  Commits:   S

Open gaps remaining:
  GAP-X  [OPEN]  blocks N kernels  Complexity: Medium
  ...
```

---

## First run: convert existing .md files

If `coverage.jsonl`, `migration_log.jsonl`, or `feature_gap_map.jsonl` are absent but the
corresponding `.md` files exist in `kernel_lib/`, convert them before starting Phase 0:

- `kernel_eltwise_coverage.md` + `kernel_helper_coverage.md` → `coverage.jsonl`
- `binary_migration_log.md` → `migration_log.jsonl`
- `feature_gap_map.md` → `feature_gap_map.jsonl`

After converting, delete the `.md` source files. The `.jsonl` files are the source of truth.

---

## Relation to other skills/docs

| This phase needs | Where |
|-----------------|-------|
| Per-kernel audit checklist | `llk_helpers_hq.md` → "Kernel Migration Steps" Step 1 |
| Policy mapping (CB lifecycle → policy) | `llk_helpers_hq.md` → "Policy Mapping" section |
| Blockers checklist | `llk_helpers_hq.md` → "Blockers Checklist" section |
| Adding a chain element (fast path) | `llk_helpers_conventions.md` §5 |
| Full helper update (pipeline path) | `llk_helpers_pipeline.md` update mode |
