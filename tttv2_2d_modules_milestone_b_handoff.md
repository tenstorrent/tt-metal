# TTTv2 Milestone B Handoff — Qwen Split and Static Follow-Through

Written 2026-08-21. Companion to `tttv2_2d_modules_milestone_b_work_log.md`
(what was done and why) and `tttv2_2d_modules_plan.md` (the gated plan).

> **Status, 2026-08-21 (second session).** Task 1 is done: Qwen owns its graph
> and `galaxy/dense_transformer.py` is deleted (work log, Checkpoint 7). Task 2
> is done: the RoPE decode contract and the Qwen ring widths were traced against
> the TTNN op sources (Checkpoint 8, both clear — no code change), and the
> step-2 hardware test exists but has never been executed (Checkpoint 9). Still
> nothing executed anywhere in this work stream. What is left is hardware:
> Milestone B sequence steps 2, 3, 5, 6, 7, plus the new paged-decode
> page-table gap recorded at the end of Checkpoint 8. The invariants at the
> bottom of this file still apply.

## Assume no hardware and no `ttnn`

This checkout cannot import `ttnn` (`ModuleNotFoundError: ttnn._ttnn`), so
**pytest cannot run at all** — not even the host-only suites, because every test
module imports `ttnn` transitively. Plan the next session around static work.

What you *can* verify locally:

```bash
# Syntax.
python -m py_compile models/common/models/galaxy/*.py \
    models/common/models/llama33_70b_galaxy/*.py \
    models/common/models/qwen3_32b_galaxy/*.py \
    models/common/tests/models/*/*.py

# Boundary invariants required by the plan.
git status --porcelain | grep '_1d\.py'      # must be empty
git status --porcelain | grep 'llm_runtime'  # must be empty

# Line length (black, 120 columns) — no formatter is installed here.
python - <<'PY'
import pathlib
for p in pathlib.Path("models/common").rglob("*.py"):
    for i, line in enumerate(p.read_text().splitlines(), 1):
        if len(line) > 120:
            print(f"{p}:{i} ({len(line)})")
PY
```

Do **not** claim any test result. Nothing in this work stream has ever been
executed.

## State on disk

Milestone B sequence steps 1 and 4 (provider adaptors + one-layer-capable tensor
models) are written for both models. Steps 2, 3, 5, 6, 7 need a WH Galaxy mesh.

```text
models/common/models/galaxy/            # shared, topology-owned, model-neutral
  ccl.py resources.py                   # pre-existing (Milestone A)
  recipes.py        (8,4) geometry, core sets, placements, program configs
  plans.py          GalaxyResourcesConfig union + select_galaxy_resource
  collectives.py    Attention2D adapters, LMHead2D column all-reduce
  prefetch.py       sender/receiver mapping, address placement, prefetch producer
  kv_contract.py    per-layer paged KV metadata + PagedKVCacheManager view
  dense_transformer.py   DEPRECATED — Qwen's only remaining dependency (deleted in Checkpoint 7)

models/common/models/llama33_70b_galaxy/   # split: owns its own graph
  weight_utils.py model.py hf_adaptor.py __init__.py

models/common/models/qwen3_32b_galaxy/     # NOT split: still uses the shared graph
  weight_utils.py model.py hf_adaptor.py __init__.py

models/common/tests/models/galaxy/test_recipes.py, test_plans.py
models/common/tests/models/llama33_70b_galaxy/test_model_host.py
models/common/tests/models/qwen3_32b_galaxy/test_model_host.py
```

Two 2D module contracts were corrected (see the work log, Checkpoint 4):
`Attention2D`'s `wo` shape and `LMHead2D`'s accepted activation width. Both have
focused host tests; neither has been re-run on hardware.

## Task 1 (next session) — split Qwen the way Llama was split

Mirror `models/common/models/llama33_70b_galaxy/model.py`. It is the reference
implementation; read it end to end first.

### Exactly what Qwen borrows from the shared graph today

```text
qwen3_32b_galaxy/model.py       GalaxyDensePrecision, GalaxyDenseTransformer2D,
                                GalaxyDenseWeights, assemble_galaxy_dense_model
qwen3_32b_galaxy/hf_adaptor.py  GalaxyDenseLayerWeights, GalaxyDensePrecision,
                                GalaxyDenseWeights
tests/.../qwen3_32b_galaxy/     GalaxyDenseLayerWeights, GalaxyDenseWeights
```

Replace each with a package-owned equivalent named after the model
(`Qwen3_32BGalaxyPrecision`, `Qwen3_32BGalaxyLayerWeights`,
`Qwen3_32BGalaxyWeights`, `Qwen3_32BGalaxyLazyLayerWeights`,
`Qwen3_32BGalaxyLazyWeights`, `Qwen3_32BGalaxyBlockConfig`,
`Qwen3_32BTransformerBlock2D`, `Qwen3_32BGalaxyTransformer2DConfig`,
`build_qwen3_32b_galaxy_lazy_weights`,
`build_qwen3_32b_galaxy_transformer_2d_config`, and
`build_qwen3_32b_galaxy_model` owning the construction order).
`Qwen3_32BGalaxyTransformer2D` becomes a direct `LightweightModule`, not a
subclass of anything shared.

### Where Qwen must differ from the Llama copy

| Concern | Llama | Qwen |
| --- | --- | --- |
| Q/K normalization | none — dropped entirely | **keep**: per-head `RMSNorm2DConfig` with `RMSNorm2DGeometry.HEAD_LOCAL`, DRAM in/out, wired into `Attention2DConfig.q_norm_config` / `k_norm_config`. `Attention2D` rejects any other geometry. |
| `n_heads * head_dim` | `== dim` (8192), asserted | `!= dim` (8192 vs 5120). Do **not** copy Llama's `__post_init__` assertion. `wo` is `[8192, 5120]`. |
| Lazy layer weights | 5 projections + 2 norms | plus `q_norm`, `k_norm` |
| Prefetch registration | 5 per layer | still 5 (`wqkv, wo, w1, w3, w2`) — Q/K norms are not prefetched |
| Precision | BFP8 MLP weights | `mlp_w1_w3_dtype` / `mlp_w2_dtype` are **bfloat16** in the accuracy recipe (`QWEN3_32B_GALAXY_ACCURACY`), matching the qualified MLP2D Qwen recipe |
| RoPE | `rope_theta=500000`, llama3 scaling factor 8.0 | `rope_theta=1000000`, no scaling (`rope_scaling_factor=None`, `original_context_len=None`) |
| Vocabulary | 128256, padded 128256 | 151936, padded 152064 |
| Layers / eps | 80 / 1e-5 | 64 / 1e-6 |
| HF revision | none | keep `DEFAULT_HF_REVISION` pin |
| Fused QKV bias | no path | keep the conversion in `weight_utils` (it detects a biased checkpoint) but **reject** it during lazy-weight resolution, as `dense_transformer._reject_qkv_bias` does today, with the same explanation |

Everything else — placements, collectives, prefetch, KV contract, the residual
convention, the `_relocate` / `_release_unless` helpers, the graph methods — is a
faithful copy of the Llama file.

### Finish the task

1. Delete `models/common/models/galaxy/dense_transformer.py` once nothing imports
   it (`grep -rln dense_transformer models/common`; the Llama test mentions the
   string inside an assertion, which is expected).
2. Update `models/common/models/qwen3_32b_galaxy/__init__.py` exports the way
   Llama's was updated.
3. In `models/common/tests/models/qwen3_32b_galaxy/test_model_host.py`: switch to
   the package-owned weight types, and copy Llama's two structural tests —
   `test_prefetch_registration_is_ordered_per_layer` and
   `test_package_owns_its_graph_and_imports_no_model_named_implementation`
   (including the `galaxy.dense_transformer` absence assertion). Add a
   Qwen-specific test that the Q/K norm configs resolve to `HEAD_LOCAL` geometry
   with width `head_dim`.
4. Append a checkpoint to `tttv2_2d_modules_milestone_b_work_log.md` and update
   the modularity-scorecard "Boundary leakage" row (it currently records Qwen as
   the open follow-up).
5. Update `models/common/modules/README.md` if the wording about the shared layer
   stops being accurate.

## Task 2 (optional, static) — reduce hardware-debug risk

These are static-only and would each shorten the first Galaxy session:

- **Write the step-2 hardware test now, unrun.** A one-layer decode/prefill PCC
  test built through
  `llama33_70b_galaxy.from_pretrained(..., n_layers=1, prefill_sequence_lengths=(128,))`
  against an independent HF reference. Model it on
  `models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py`
  (fixture style, `device_params` with `DispatchCoreAxis.COL` and
  `FabricConfig.FABRIC_1D_RING`, PCC >= 0.99, explicit teardown).
- **Trace the RoPE decode contract on paper.** This is the highest-risk unknown
  (gap 2 in the work log). `ttnn.experimental.rotary_embedding_llama` decode mode
  requires height-sharded Q/K, `cos.logical_shape()[1] == batch`, and
  `cos.shard_spec().shape[0] == TILE_HEIGHT`; the validation lives in
  `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/rotary_embedding_llama_device_operation.cpp`.
  Confirm on paper that `RotarySetup2D.decode_forward` output and the fused
  create-QKV-heads output satisfy it for `users_per_column = 8`, and record the
  answer. If they cannot, decide between the fused path
  (`use_qk_fused_rotary=True`, which needs `rows = 16`) and a config change
  before burning mesh time.
- **Audit the Qwen decode ring widths** (gap 4) the same way: the scattered W1/W3
  placement is padded to 960 columns for both models while the resource key uses
  the logical width (960 Llama / 800 Qwen). Confirm against
  `ttnn.experimental.llama_reduce_scatter` / `llama_rs_matmul` which width the op
  reports, and align `GalaxyDenseGeometry.decode_reduce_scatter_width` if the
  reasoning says otherwise.

## Do not attempt without hardware

- Any claim about numerics, PCC, KV-cache correctness, or performance.
- Retuning the qualified recipes in `recipes.py` on intuition. Every formula
  there is a port of a hardware-qualified Milestone A test recipe; changing one
  invalidates recorded evidence. If a change looks necessary, record the argument
  in the work log and leave the code alone.
- Milestone C work (executors, generators, demos, vLLM routing). The plan gates
  it behind Milestone B's exit gate.

## Invariants to preserve

1. Zero changes to `models/common/modules/**/*_1d.py`.
2. Zero changes to `models/common/llm_runtime/**`.
3. No model package imports another model-named package
   (`models.demos`, `models.tt_transformers`, `models.common.models.<other>`).
4. Existing Milestone A host tests keep their expectations. One fixture was
   corrected (`test_attention_2d.py`'s `wo` shape, which was not a realizable
   geometry); do not extend that to loosening real assertions.
5. `Prefetcher2D` is the resource root: register weights explicitly in per-layer
   issue order, seal, then create the Galaxy CCL owner. Never reorder those.
