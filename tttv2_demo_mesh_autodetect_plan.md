# TTTv2 Demo `MESH_DEVICE` Auto-Detection + PR #51184 Review Resolution

Status: **PLAN ONLY — do not implement Part 2 in PR #51184.**

Two scopes:

- **Part 1** — resolve all reviewer comments on
  [PR #51184](https://github.com/tenstorrent/tt-metal/pull/51184) (llama3_8b only).
- **Part 2** — reusable playbook to roll the same `MESH_DEVICE` auto-detection out to the
  remaining TTTv2 demos in a follow-up PR.

---

## Background / why

PR #51184 added a nested `case "{sku}" in ... esac` block to
`tests/pipeline_reorg/models_e2e_tests.yaml` so the Llama 3.1-8B entry could set
`MESH_DEVICE=N150` vs `MESH_DEVICE=T3K` per SKU. mtairum's review (CHANGES_REQUESTED):

> "Why are the changes in this yaml so complex? This needs to be simplified. If it means
> that we have to split into 2 jobs, so be it."
> "This should not be here." (the per-SKU `MESH_DEVICE` case)
> "We either skip TTTv2 on BH (at the pytest level), or we split. I prefer pytest,
> because that's model owner responsibility and a fix for it won't affect infra files."

Everything `MESH_DEVICE` encodes is discoverable from hardware on the Python side:

| Fact | Hardware source | Already used by |
|---|---|---|
| Cluster SKU (`wh_n150`, `wh_llmbox_perf`, `bh_p150`, ...) | `ttnn.cluster.get_cluster_type()` via `models/demos/utils/device_sku.py::get_current_device_sku_name()` | `trace_region_sizes.py` (prescribed path for full-device demos) |
| Physical mesh shape | `ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()` | `models/common/tests/conftest.py:156` (fixture pre-check) |
| Blackhole skip | `ttnn.device.is_blackhole()` | `models/common/tests/conftest.py:130-131` — **all TTTv2 demos already skip on BH at fixture setup** |

Additional facts that shape the plan:

- `MESH_DEVICE` is read **only** in the demo files themselves. Zero references in
  `models/common/llm_runtime/` or `models/common/models/llama3_8b/`. vLLM/production
  paths are untouched by this plan.
- `models/model_trace_region_sizes.yaml` is keyed by canonical SKU names
  (`wh_n150`, `wh_llmbox_perf`, ...), and `resolve_trace_region_size()` runs keys
  through `normalize_sku`, so both `T3K` and `wh_llmbox_perf` resolve.
- In `llama3_8b/demo.py`, `MESH_DEVICE` feeds exactly 3 things: `mesh_shape`,
  the trace-region lookup, and the parametrization id (which appears in pytest node
  ids, e.g. `test_llama3_8b[wormhole_b0-performance-ci-b1-DP-4-T3K]`).

---

## Part 1 — PR #51184 (llama3_8b)

### 1.1 `models/common/tests/demos/llama3_8b/demo.py`: auto-detect when `MESH_DEVICE` is unset

Replace the module-level env read (current lines 324-330) with detection that prefers
the env var (keeps the vLLM/sibling-demo convention and local submesh experiments) and
falls back to the attached cluster:

```python
def _detect_mesh_device() -> tuple[str, tuple[int, int]]:
    """(name, mesh_shape) from MESH_DEVICE if set, else from the attached cluster."""
    env = os.environ.get("MESH_DEVICE", "").strip().upper()
    by_name = {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (4, 8)}
    if env:
        shape = by_name.get(env)
        if shape is None:
            raise ValueError(f"Unsupported MESH_DEVICE={env!r}; use N150, N300, T3K, or TG.")
        return env, shape
    try:
        sku = get_current_device_sku_name()  # models.demos.utils.device_sku
    except Exception as e:
        raise RuntimeError(
            "No TT cluster detectable at collection time; set MESH_DEVICE explicitly or run on a TT host"
        ) from e
    name = {"wh_n150": "N150", "wh_n300": "N300", "wh_llmbox_perf": "T3K", "wh_galaxy_perf": "TG"}.get(sku)
    if name is None:
        raise RuntimeError(f"Unsupported cluster SKU={sku!r} for this demo")
    return name, by_name[name]


mesh_device_name, mesh_device_shape = _detect_mesh_device()
```

Notes:

- Detection failures are **hard errors, not skips**: a silent skip turns a
  misconfigured runner (driver/UMD issue, wrong SKU, typo'd `MESH_DEVICE`) into a green
  CI job that ran zero tests. All three failure exits (invalid env value, undetectable
  cluster, unsupported SKU) raise at collection → pytest reports ERROR and the job
  fails. Verified safe: no pipeline collects `models/common/tests/demos/` broadly —
  every CI invocation (`models_e2e_tests.yaml`, `release_tests.yaml`,
  `t3k_e2e_tests.yaml`) names a demo file explicitly on a hardware runner, so the raise
  can only fire when someone explicitly points pytest at the demo on a broken or
  unsupported host.
- The `try/except` still wraps the UMD query, but only to re-raise with actionable
  context (set `MESH_DEVICE` or run on a TT host).
- Returned names are identical to today's env values, so parametrized node ids
  (`...-T3K`, `...-TG`) are unchanged and the yaml's explicit node-id selectors in the
  DP entries keep matching.
- `resolve_trace_region_size("llama3.1-8b", mesh_device_name)` needs no change
  (`normalize_sku` handles `N150`/`T3K`).
- On BH runners, detection yields e.g. `bh_p150` → not in the map → collection error.
  Irrelevant to CI: the split yaml never invokes the TTTv2 demo on BH. If someone
  misconfigures an entry to do so, a loud failure beats a silent skip. (The conftest
  fixture's `is_blackhole()` skip stays as the backstop for the non-demo unit tests
  under `models/common/tests/`.)

### 1.2 `tests/pipeline_reorg/models_e2e_tests.yaml`: split Llama 3.1-8B wh/bh, drop all branching

Follows the existing `main` precedent: `Qwen3-32B e2e tests` (wh, TTTv2) +
`Qwen3-32B e2e tests (Blackhole)` (bh, TTTv1). Duplicate `model:` values across entries
are already the norm (`qwen3-32b`, and `llama3.1-8b-dp` via the P300 LFC entry).

**Entry 1 — Wormhole (TTTv2), no `case`, exports on one line:**

```yaml
# TTTv2 substitution (Wormhole only — TTTv2 Llama-3.1-8B has no Blackhole support yet).
# Blackhole SKUs keep the TTTv1 legs in the "(Blackhole)" entry below; the split entries
# are reunified under #52330 once TTTv2 supports Blackhole.
- name: Llama 3.1-8B e2e tests
  cmd: |
    export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct HF_HUB_OFFLINE=1 TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/tttv2/meta-llama/Llama-3.1-8B-Instruct SAMPLING_MODE=on_device_topk PIPELINE_READBACK=1
    # Match TTTv1 token-matching coverage.
    pytest --timeout 420 models/common/tests/demos/llama3_8b/demo.py -k "performance and token-accuracy-repeat_batch-1-prefetcher-off"
    # Match both TTTv1 ci-eval-32 runs (TTTv2 has no DRAM prefetcher, so all ids are prefetcher-off).
    pytest --timeout 600 models/common/tests/demos/llama3_8b/demo.py -k "performance and eval-32-repeat_batch-3-prefetcher-off-perf-report-off"
    pytest --timeout 600 models/common/tests/demos/llama3_8b/demo.py -k "performance and eval-32-repeat_batch-1-prefetcher-off-perf-report-on"
    # Keep this TTTv1 leg until TTTv2 supports native HF-layout RoPE (#51374).
    pytest --timeout 420 models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-token-matching" --use_hf_rope
  model: llama3.1-8b
  model_family: Llama
  skus:
    wh_n150:
      timeout: 11
      tier: 1
    wh_llmbox_perf:
      timeout: 11
      tier: 2
  owner_id: U03PUAKE719 # Miguel Tairum Cruz
  team: models
```

**Entry 2 — Blackhole (TTTv1, verbatim from `main`'s original cmd):**

```yaml
# TTTv1 legs preserved until TTTv2 Llama-3.1-8B supports Blackhole; then reunified under #52330.
- name: Llama 3.1-8B e2e tests (Blackhole)
  cmd: |
    export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/meta-llama/Llama-3.1-8B-Instruct
    pytest --timeout 420 models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-token-matching"
    # The DRAM prefetcher does not yet support more than one repeat batch (issue #47820):
    # cover the eval repeat-batch loop with the prefetcher DISABLED (perf reporting skipped
    # so only the prefetcher run sets perf), and cover the prefetcher path with a SINGLE
    # repeat batch (this run reports/validates perf). See #47820 for the multi-batch work.
    pytest --timeout 600 models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-eval-32" --skip_perf_report
    pytest --timeout 600 models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-eval-32" --use_prefetcher True --repeat_batches 1
    pytest --timeout 420 models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-token-matching" --use_hf_rope
  model: llama3.1-8b
  model_family: Llama
  skus:
    bh_p150:
      timeout: 11
      tier: 1
    bh_quietbox_2:
      timeout: 20
      tier: 2
  owner_id: U03PUAKE719 # Miguel Tairum Cruz
  team: models
```

**DP entry** — same split. wh entry keeps only the TTTv2 lines, minus `MESH_DEVICE=T3K`
(now auto-detected); node-id selectors unchanged:

```yaml
# TODO: Move DP tests to sweep pipelines when those come online
- name: Llama 3.1-8B data-parallel e2e tests
  cmd: |
    export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct HF_HUB_OFFLINE=1 TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/tttv2/meta-llama/Llama-3.1-8B-Instruct SAMPLING_MODE=on_device_topk PIPELINE_READBACK=1
    pytest --timeout 1000 "models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[wormhole_b0-performance-ci-b1-DP-4-T3K]"
    pytest --timeout 1000 "models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[wormhole_b0-performance-ci-b1-DP-8-T3K]"
  model: llama3.1-8b-dp
  model_family: Llama
  skus:
    wh_llmbox_perf:
      timeout: 20
      tier: 2
  owner_id: U03PUAKE719 # Miguel Tairum Cruz
  team: models

# bh_quietbox_2 is 4-chip: the DP-8 case is auto-skipped when the chip count is below the
# DP factor, so it effectively runs the DP-4 case only. Migrated from the legacy
# blackhole_demo_tests.yaml DP entries. TTTv1 until TTTv2 supports Blackhole (#52330).
- name: Llama 3.1-8B data-parallel e2e tests (Blackhole)
  cmd: |
    export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/meta-llama/Llama-3.1-8B-Instruct
    pytest models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-b1-DP-4 or performance-ci-b1-DP-8" --timeout 1000
  model: llama3.1-8b-dp
  model_family: Llama
  skus:
    bh_quietbox_2:
      timeout: 20
      tier: 2
  owner_id: U03PUAKE719 # Miguel Tairum Cruz
  team: models
```

**Galaxy DP entry** — already single-SKU; just drop `MESH_DEVICE=TG` from its export line.

Per-SKU `timeout:`/`tier:` values are unchanged, so `.github/time_budget.yaml` sums are
unaffected.

Why split rather than one entry + pytest skips (mtairum's first choice): the TTTv1 legs
must run on BH but not WH (TTTv2 replaces them there). A single entry would either run
TTTv1 legs on WH too (duplicated coverage, blown 11-min timeout) or require arch gates
inside TTTv1's `simple_text_demo.py` — worse than two entries. The split is the only
option that removes *all* branching, and mtairum explicitly approved it.

### 1.3 Reference the tracking issue — no new issue needed

mtairum: "Please add a comment with the issue number to the file, so we can track this."
Use the existing [#52330](https://github.com/tenstorrent/tt-metal/issues/52330)
("Unify the test cases in model_e2e_tests.yaml") as the tracker for merging the split
entries back once TTTv2 supports Blackhole. It is already substituted into the yaml
comments above. (#48496 is generic TTTv2-CI tracking and #51374 is RoPE-only; neither
fits, which is why #52330 exists.)

### 1.4 `models/common/llm_runtime/lane_group.py`: kill the Cycode re-fire

Cycode flags `setattr(primary, attribute, ...)` (line 742) as "unsanitized external
input in code generation" — Critical — and re-fires on every push (3× so far). False
positive (`attribute` is a hardcoded literal at all 4 call sites), but repo precedent
(`perf_utils.py:503`, `validate_perf_targets.py:501`) is to restructure + comment:

```python
def _attach_failures(primary: BaseException, failures: Sequence[BaseException], attribute: str) -> None:
    if not failures:
        return
    previous = tuple(getattr(primary, attribute, ()))
    try:
        # Static assignments only — Cycode SAST flags dynamic setattr as code injection.
        if attribute == "cleanup_failures":
            primary.cleanup_failures = previous + tuple(failures)
        else:
            primary.lane_failures = previous + tuple(failures)
    except BaseException:
        pass
```

Also dismiss the finding as false-positive in the Cycode dashboard so it stays quiet.

### 1.5 Review-thread replies

| Thread | Reply | Resolve? |
|---|---|---|
| adrian: "Enum is not imported" (`executor.py:39`) | Fixed in `dd8cacaec75` — `Mode` is imported from `tt.common` again; file no longer defines it locally. | Yes |
| adrian: "math is not imported neither" (`executor.py:82`) | Same commit — `math` is no longer used in this file. | Yes |
| cycode ×3 | False positive (`attribute` is a hardcoded literal at all call sites); refactored to static assignment anyway. Dismissed in dashboard. | Yes |
| mtairum: yaml complexity (file-level) | Split into wh/bh entries per Qwen3-32B precedent; reunification tracked by #52330, referenced in the entry comments. | After push |
| mtairum: "can't we do a check at the python script level instead?" | Done: the demo now auto-detects mesh/SKU from hardware (`get_current_device_sku_name()`), so no yaml branching remains — and running it on an unsupported cluster is a loud collection error, not a silent skip. Chose split over a single entry because the TTTv1 legs must not run on WH. | After push |
| mtairum: "This should not be here." / "1 liner" ×2 | All `case` blocks gone; exports are one line; RoPE comment is one line. | After push |

### 1.6 Part-1 verification

1. `python3 -m py_compile models/common/tests/demos/llama3_8b/demo.py models/common/llm_runtime/lane_group.py`
2. On N150 and T3K, with `MESH_DEVICE` **unset**: `pytest --collect-only models/common/tests/demos/llama3_8b/demo.py -q` → node ids identical to today (mesh-name suffix preserved); `pytest ... -k "token-accuracy and performance"` runs the same cases.
3. With `MESH_DEVICE=N150` set on N150 → identical behavior (override path).
4. yaml sanity: parse `tests/pipeline_reorg/models_e2e_tests.yaml`; confirm no other entries changed (`git diff` shows only the llama3.1-8b blocks).
5. Re-run the affected CI legs (models-e2e tier-1/2 on wh_n150, wh_llmbox_perf, bh_p150, bh_quietbox_2).

---

## Part 2 — Follow-up PR: roll out to the remaining TTTv2 demos

**Do not include in PR #51184.** One PR touching all demos below (or one per demo if
review load prefers); no yaml entry changes are needed for single-SKU entries beyond
deleting `MESH_DEVICE=...` from the export line.

### 2.1 Per-demo recipe

1. Add `_SKU_TO_MESH_NAME = {"wh_n150": "N150", "wh_n300": "N300", "wh_llmbox_perf": "T3K", "wh_galaxy_perf": "TG"}`
   and a `_detect_mesh_name()` helper (env var wins; else `get_current_device_sku_name()`;
   detection failure / unknown SKU / invalid env value all **raise**, never skip — see
   Part 1.1 notes for the rationale and safety check).
2. Replace every `os.environ.get("MESH_DEVICE", "")` read with the helper. Two code
   shapes exist:
   - `_ttnn_mesh_device_param_from_env()` style (most demos): rename to
     `_ttnn_mesh_device_param()`, source the name from the helper.
   - Inline style (llama3_8b): Part 1.1 is the reference.
3. Replace the parametrization id source `ids=[os.environ.get("MESH_DEVICE", "mesh")...]`
   with the detected name, or node ids silently change to `mesh` and any `-k`/node-id
   selectors drift.
4. Delete `MESH_DEVICE=...` from the demo's yaml entry cmd (and from the module
   docstring usage examples, or rewrite them to mention the override).
5. Keep each model's own shape map as the source of truth for shape — see pitfalls.

### 2.2 Inventory

| Demo | Code shape | Mesh map | yaml entry / SKUs | Node-id selectors in yaml? | Notes |
|---|---|---|---|---|---|
| `llama3_8b` | inline | N150, N300, T3K, TG(4,8) | llama3.1-8b (n150, t3k); -dp (t3k); -dp-galaxy (TG) | **yes** (`...-T3K`, `...-TG`) | Part 1 of this plan; uses `resolve_trace_region_size` |
| `qwen25_7b` | env-func | N150, N300, T3K, TG(8,4) | qwen2.5-7b (wh_n300) | no | T3K/TG listed only so module imports there; skips at model build (`_skip_unless_heads_divide_mesh`). N150x4 deliberately absent. |
| `qwen2_7b` | env-func | N150, N300, T3K, TG(8,4) | (not in e2e yaml yet; #50879) | no | Same head-divisibility pattern as qwen25_7b. |
| `qwen3_32b` | env-func | T3K only | qwen3-32b (wh_llmbox_perf); wh/bh already split | no | Auto-detect on T3K runner returns T3K; everywhere else module-skips (as today with env unset). |
| `qwen25_coder_32b` | env-func | T3K only | qwen2.5-coder-32b (wh_llmbox_perf) | no | Same as qwen3_32b. |
| `qwen25_72b` | env-func | T3K only | entry commented out (CI hang) | no | Apply anyway so re-enabling needs no demo change. |
| `mistral_7b` | env-func | N150 (+N300?) | mistral-7b (wh_n150); note its 2nd leg overrides `TT_CACHE_PATH` inline for TTTv1 eval | no | Keep the inline TTTv1 `TT_CACHE_PATH` override line as-is. |
| `llama32_1b` | env-func | N150 (+?) | llama3.2-1b (wh_n150) | no | |
| `llama32_3b` | env-func | N150 (+?) | llama3.2-3b (wh_n150) | no | |
| `llama33_70b` | env-func | T3K?/TG? | not substituted in e2e yaml (TTTv1 still) | no | Low priority; confirm map before touching. |
| `phi4` | env-func | N150 (+?) | not in e2e yaml | no | Low priority. |
| `deepseek_r1_distill_qwen_14b` | env-func | N150 (+?) | not in e2e yaml | no | Low priority. |

### 2.3 Pitfalls (learned while scoping)

- **TG orientation discrepancy:** llama3_8b maps `TG: (4, 8)`; qwen25_7b / qwen2_7b /
  phi4 map `TG: (8, 4)`. Physical Galaxy is 8x4 — one side is likely a latent bug.
  **Do not unify the shape maps as part of this rollout**; auto-detection selects the
  *name*, each demo keeps its own map. Investigate/fix the orientation separately.
- **Hardcoded `trace_region_size: 50_000_000`** in the env-func demos (llama3_8b already
  uses the centralized `resolve_trace_region_size`). Out of scope here; optional later
  cleanup is to migrate all demos to the centralized yaml.
- **Listed-but-skipped meshes:** qwen25_7b/qwen2_7b list T3K/TG so the module imports on
  those hosts and skips cleanly at model build. With auto-detect, running on such a host
  now reaches the model-build skip instead of the env-unset module skip — same outcome,
  slightly more collection work. Preserve the map entries; don't "clean them up".
- **BH behavior changes from skip to error:** an unmapped BH SKU now raises at
  collection (by design — loud failure on misconfiguration). The fixture-level
  `is_blackhole()` skip (`models/common/tests/conftest.py:130`) stays as the backstop
  for non-demo unit tests; demos never reach it on BH.
- **Collection-time hardware query** is wrapped in `try/except`, but only to re-raise
  with context — never to skip. Safe because no pipeline collects `demos/` broadly;
  every CI invocation names the demo file on a hardware runner (verified across
  `tests/pipeline_reorg/*.yaml`).
- **`MESH_DEVICE` stays as an override** — it is the vLLM/sibling-demo convention and is
  how developers pin a submesh config locally. This plan removes the *requirement*, not
  the variable.

### 2.4 Per-demo verification

1. `pytest --collect-only <demo> -q` with `MESH_DEVICE` unset on the demo's target SKU →
   node ids byte-identical to `git stash` baseline (diff the two collection outputs).
2. Smoke the cheapest case (`-k` the smallest config) on the target runner.
3. Confirm a loud collection **error** (not a silent skip, not a hang) on: a BH runner,
   and a WH SKU the demo doesn't support; and confirm the error message names the fix
   (set `MESH_DEVICE` / supported values).
4. `git diff tests/pipeline_reorg/models_e2e_tests.yaml` shows only `MESH_DEVICE=...`
   deletions (plus, for llama3_8b in Part 1, the entry split).
