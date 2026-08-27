# TTTv2 2D Modules Milestone B Work Log

> **Current state (2026-08-24).** All seven Milestone B sequence steps now have
> code. Steps 1 and 4 landed in Checkpoints 1-9; steps 2, 3, 5, 6 and 7 are
> covered by Checkpoints 9-12. **Nothing in this branch has ever run**, on
> hardware or otherwise. Checkpoint 12 is the suggested order for the first
> Galaxy session; Checkpoint 11 lists what is most likely to break.

## Checkpoint 1 - Scope

- Date: 2026-08-21.
- Requested scope: Milestone B **sequence steps 1 and 4 only** from
  `tttv2_2d_modules_plan.md` — the Llama-3.3-70B and Qwen3-32B provider adaptors
  and their one-layer-capable tensor-model reconstructions.
- Deliberately **not** in scope for *this checkpoint*: steps 2, 3, 5, 6, and 7
  (one-block and full-model numerical validation, direct demos,
  paged/prefix/concat-32/sampling coverage). All of those require a WH Galaxy
  `(8, 4)` mesh, and all are delivered as unrun code in Checkpoints 9-12.
- Environment: laptop with no Tenstorrent hardware. `import ttnn` fails here
  (`ModuleNotFoundError: ttnn._ttnn`), so **nothing in this checkpoint was
  executed** — not even the host-only tests. Every claim below is static design
  evidence: contract reading, shape algebra, and syntax checks
  (`python -m py_compile`, clean).
- Executors, generators, demos, and vLLM routing remain unwritten by design;
  they are Milestone C.

## Checkpoint 2 - What was added

### Shared Galaxy production layer (`models/common/models/galaxy/`)

Model-neutral, topology-owned, outside any model-named package. Existing files
(`ccl.py`, `resources.py`) are unchanged.

| File | Ownership |
| --- | --- |
| `recipes.py` | `(8, 4)` constants, core sets, `GalaxyDenseGeometry`, memory/program-config recipes, resolved decode and prefill placements |
| `plans.py` | The `GalaxyResourcesConfig` union of every collective the qualified modules issue, plus `select_galaxy_resource` |
| `collectives.py` | `GalaxyAttentionCollectives` (Attention2D low-level adapters), `GalaxyColumnAllReduce` (LMHead2D collective), runtime batch-offset tensors |
| `prefetch.py` | Canonical sender/receiver mapping, address placement, decode prefetch producer |
| `kv_contract.py` | Per-layer paged KV metadata plus the narrow view the common `PagedKVCacheManager` already accepts |
| `dense_transformer.py` | **Deprecated.** A shared graph + config assembly, kept only until Qwen is split. Llama no longer uses it. *(Deleted in Checkpoint 7.)* |

### Model packages

```text
models/common/models/llama33_70b_galaxy/{__init__,weight_utils,model,hf_adaptor}.py
models/common/models/qwen3_32b_galaxy/{__init__,weight_utils,model,hf_adaptor}.py
```

Each package owns the checkpoint contract, the precision recipe, and provider
key/layout conversion. Neither imports any model-named package (asserted by a
source scan in its host tests).

**Graph ownership.** The first draft put the decoder layer, the tensor model,
and the module-config assembly in a shared `dense_transformer.py`, with each
product model as a thin subclass. The plan permits that (generic helpers outside
model-named directories) but does not require it, and it reads against TTTv2's
"library, not framework" principle and against the plan's per-package language
("builds only the new 2D module configs"). Llama has therefore been split: its
`model.py` now owns its precision recipe, host/lazy weight types, every 2D
module config, `Llama33_70BTransformerBlock2D`,
`Llama33_70BGalaxyTransformer2D`, and the construction order. It borrows only
topology-neutral machinery from the shared Galaxy layer.

The split also removed generic scaffolding Llama does not need: no Q/K
normalization, no fused QKV bias, and `n_heads * head_dim == dim` is now an
asserted invariant rather than a general case.

Qwen still composes the shared `dense_transformer.py`; splitting it the same way
is the immediate follow-up, after which that file is deleted.

### Tests

```text
models/common/tests/models/galaxy/test_recipes.py
models/common/tests/models/galaxy/test_plans.py
models/common/tests/models/llama33_70b_galaxy/test_model_host.py
models/common/tests/models/qwen3_32b_galaxy/test_model_host.py
```

Plus focused additions to `tests/modules/attention/test_attention_2d.py` and
`tests/modules/lm_head/test_lm_head_2d.py` for the two module contract changes
below.

## Checkpoint 3 - Construction order

The prefetcher is the resource root, and `GalaxyResources` refuses to borrow an
unsealed prefetcher context, so the order is forced and now explicit in
`assemble_galaxy_dense_model`:

1. resolve geometry and decode placements;
2. resolve the Galaxy collective-resource policy (`build_galaxy_resources_config`);
3. build every `LazyWeight` (`build_galaxy_dense_lazy_weights`) — no device work;
4. materialize and register the prefetched decode weights in per-layer issue
   order (`wqkv, wo, w1, w3, w2`, matching the legacy Galaxy stack) and seal;
5. create the Galaxy CCL/subdevice owner over the sealed prefetcher; and
6. assemble module configs and construct the tensor model.

Registration is explicit and ordered; nothing scans the model graph, and the
prefill context intentionally carries no global CB.

## Checkpoint 4 - Shared/module files changed, and why config alone was insufficient

| File | Change | Why |
| --- | --- | --- |
| `modules/attention/attention_2d.py` | `wo` source shape is now `(n_heads * head_dim, dim)` instead of `(dim, dim)`; added the matching row-divisibility check | Qwen3-32B decouples `head_dim` from the hidden size (64 heads x 128 = 8192 vs dim 5120), so the previous contract could not express the real checkpoint. For Llama (and any model where the two coincide) the check is byte-identical. |
| `modules/attention/attention_2d.py` | `wqkv`/`wo` lazy-weight resolution no longer swaps `weight_memory_config` and `wo_weight_memory_config` | Latent defect: the exact-policy validator pairs them the other way. Only reachable when a caller leaves a weight's `memory_config` unset, which no qualified path did. |
| `modules/lm_head/lm_head_2d.py` | A device activation may carry the column-local hidden width (`dim / 4`) as well as the full `dim` | The module's own mapper shards the LM-head hidden dimension over columns, so a device activation from the column-sharded residual stream is `dim / 4` wide. Only host `LazyWeight` inputs carry the full width, which is why the hardware qualification never hit this. |

No `llm_runtime` file was touched, and no `*_1d.py` module implementation file
was touched. The batched-prefill policy delivered in Milestone A already covers
what these models need; step 1/4 required no runtime change at all.

One existing host fixture was corrected rather than re-expected:
`tests/modules/attention/test_attention_2d.py` declared 64 heads of dimension
128 with a square `(5120, 5120)` WO, which is not a realizable geometry. It now
uses `(8192, 5120)`, the true Qwen3-32B shape, and two new tests pin both the
decoupled and the square case.

## Checkpoint 5 - Gaps that need the WH Galaxy mesh

Ordered by risk. None of these can be closed on this machine.

1. **Nothing was executed.** The four new host suites and the two updated module
   suites have never run. Run them first; they need no hardware.
2. **RoPE composed with Attention2D is unqualified.** *(Shape contract traced on
   paper and confirmed in Checkpoint 8; the numerics are still unqualified.)* The Milestone A attention
   hardware test used an identity rotary, and `RotarySetup2D` was qualified
   standalone. `GalaxyAttentionCollectives.rotary` now issues the production
   `rotary_embedding_llama` calls (non-fused decode by default, fused available
   through `use_qk_fused_rotary`). Decode-mode RoPE requires the Q/K heads to be
   height-sharded with `cos.logical_shape()[1] == batch`; that is what
   `RotarySetup2D` produces for `users_per_column = 8`, but the pairing is
   unproven. Expect this to be the first hardware failure point.
3. **Real Qwen3-32B attention geometry is unqualified.** The recorded Milestone A
   Qwen attention evidence used a 40-head fixture so that `n_heads * head_dim`
   equalled `dim`. The model package builds the real 64-head geometry, which is
   what the relaxed WO contract exists for.
4. **Qwen decode ring widths.** *(Confirmed correct against the TTNN op source in
   Checkpoint 8; no change needed.)* The scattered W1/W3 *placement* is padded to the
   24-core ring (960 columns for both models, identical to the qualified Llama
   recipe), while the resource *key* uses the logical width TTNN reports (960 for
   Llama, 800 for Qwen). If the Qwen decode all-gather cannot find its resource,
   this pair is the first thing to inspect.
5. **Residual placement moved to the norm's own grid.** Attention and MLP decode
   outputs are placed on `RMSNorm2D`'s default two-wide `x=2..3` grid so the
   fused residual norm consumes them without a relocation. The module tests used
   `x=1..2` for their own outputs, so the combination is unproven.
   *(Corrected by job0/reconcile: the premise that `x=1` owns the fused stats
   circular buffer is dead. Milestone A defect D1 established the opposite - the
   stats buffer must sit on the **first core of the norm input shard grid**,
   which is now `(2, 0)`, because `fused_rms_minimal` creates its stats CB there
   and binds it to that tensor's L1 address. `RMSNorm2D._require_fused_stats_placement`
   raises on any other placement. `distributed_norm_stats_memory_config` now
   derives the origin from the residual placement instead of naming `(1, 0)`, and
   neither model passes `decode_stats_memcfg` any more.)*
6. **`semaphore_cores` is one set per mode.** Decode uses the full worker
   envelope, a superset of the norm grid the RMSNorm2D hardware test used. If the
   fused norm rejects it, split the plan rather than moving the norm shards.
7. **Sampling2D wiring is intentionally absent.** *(Closed in Checkpoint 10:
   `GalaxyColumnUserSelector` performs the per-column user selection and each
   model exposes `sample_decode`. The selection itself is unqualified.)* The
   sampler expects logits with users sharded over columns; the decode graph keeps
   the physical batch replicated across columns.
8. **A fused QKV bias is unsupported.** Neither target checkpoint has one
   (`attention_bias=False` is in both contracts). `Attention2D` validates a bias
   against the projection's DRAM-sharded weight placement, which a bias vector
   cannot satisfy; supporting one needs a bias placement field on the module
   config. Llama's package has no bias path at all; the shared graph Qwen still
   uses rejects one explicitly.
9. **Qwen is not split yet.** *(Closed in Checkpoint 7: Qwen owns its graph and
   `dense_transformer.py` is deleted.)* It composes the deprecated shared
   `dense_transformer.py`. Splitting it mirrors Llama's `model.py` and then
   deletes that file.

## Checkpoint 6 - Suggested first commands on the Galaxy host

*(Superseded by Checkpoint 12, which covers the complete sequence. Kept for the
record of what the second session recommended.)*

```bash
# 1. Host-only, no hardware required.
pytest models/common/tests/models/galaxy \
       models/common/tests/models/llama33_70b_galaxy \
       models/common/tests/models/qwen3_32b_galaxy -v

# 2. Milestone A regression for the two changed modules.
pytest models/common/tests/modules/attention/test_attention_2d.py \
       models/common/tests/modules/lm_head/test_lm_head_2d.py -v

# 3. Recorded Milestone A hardware paths, unchanged by this work.
pytest models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py -v

# 4. Milestone B step 2, written statically in Checkpoint 9 and never executed.
pytest models/common/tests/models/llama33_70b_galaxy/test_model_wh_galaxy.py -v
```

Step 2 of the Milestone B sequence (one-block decode/prefill PCC) still needs a
new hardware test; it should build the model through
`llama33_70b_galaxy.from_pretrained(..., n_layers=1, prefill_sequence_lengths=(128,))`
and compare against an independent HF reference. *(Written in Checkpoint 9 for
Llama and in Checkpoint 10 for Qwen; both construct the model directly so they
can pass `paged_attention_config=None`.)*

## Checkpoint 7 - Qwen split, shared graph deleted

Date: 2026-08-21 (second session). Still no hardware and still no `ttnn` import
in this checkout, so **nothing below was executed**. Evidence is static:
`python -m py_compile` (clean), a 120-column scan (clean), a source scan for the
boundary invariants (clean), plus two mechanical AST checks described below.

`models/common/models/qwen3_32b_galaxy/model.py` now owns the Qwen graph the way
Llama owns its own: precision recipe, host and lazy weight types, every 2D module
config, the decoder layer, the tensor model, and the construction order.
`Qwen3_32BGalaxyTransformer2D` is a direct `LightweightModule`.

| Concern | Llama | Qwen as delivered |
| --- | --- | --- |
| Q/K normalization | none | `_head_local_norm_config` builds `RMSNorm2DGeometry.HEAD_LOCAL` norms (DRAM in/out, no `tt_ccl`) wired into `Attention2DConfig.q_norm_config` / `k_norm_config` |
| `n_heads * head_dim` | `== dim`, asserted | 8192 vs `dim` 5120; no assertion, `wo` is `[8192, 5120]` |
| Lazy layer weights | 5 projections + 2 norms | plus required `q_norm`, `k_norm` |
| Prefetch registration | 5 per layer | still 5 (`wqkv, wo, w1, w3, w2`); the 128-element Q/K norms are not ring operands |
| Precision | BFP8 MLP | `Qwen3_32BGalaxyPrecision` defaults `mlp_w1_w3_dtype`/`mlp_w2_dtype` to bfloat16 (the accuracy recipe *is* the default, as for Llama); the performance recipe drops them to BFP8 + LoFi FF1/FF3 |
| RoPE | theta 500000, llama3 factor 8.0 | theta 1000000, `rope_scaling_factor=None`, `original_context_len=None`; the scaling parameters are gone from the config builder because there is nothing to pass |
| Vocabulary / layers / eps | 128256 / 80 / 1e-5 | 151936 padded 152064 / 64 / 1e-6 |
| HF revision | none | `DEFAULT_HF_REVISION` pin kept |
| Fused QKV bias | no path at all | `weight_utils` still packs one when a checkpoint carries it; `_reject_qkv_bias` refuses it during lazy-weight resolution with the same explanation the shared graph used |

Everything else — placements, collectives, prefetch order, KV contract, the
`(x, h)` residual convention, `_relocate` / `_release_unless`, and every graph
method — is a faithful copy. That was verified mechanically rather than by eye: a
normalized diff (model-name tokens rewritten to a common placeholder) between
the two `model.py` files shows only the rows above plus docstrings and constants.

Also changed:

- `models/common/models/galaxy/dense_transformer.py` **deleted** after confirming
  the only remaining occurrences of the string are the two tests that assert its
  absence.
- `qwen3_32b_galaxy/hf_adaptor.py` converts into `Qwen3_32BGalaxyWeights` /
  `Qwen3_32BGalaxyLayerWeights` and resolves `Qwen3_32BGalaxyPrecision`.
- `qwen3_32b_galaxy/__init__.py` exports the package-owned graph types, mirroring
  Llama's export surface.
- `models/common/modules/README.md`: the shared Galaxy layer no longer "composes
  the modules"; each product package owns its graph.

Tests (`models/common/tests/models/qwen3_32b_galaxy/test_model_host.py`), all
unrun: switched to the package-owned weight types, plus Llama's two structural
tests (`test_prefetch_registration_is_ordered_per_layer`,
`test_package_owns_its_graph_and_imports_no_model_named_implementation`
including the `galaxy.dense_transformer` absence assertion) and two
Qwen-specific ones — `test_qk_norms_resolve_to_head_local_geometry` (geometry is
`HEAD_LOCAL`, weight width is `head_dim`, no CCL borrowed, DRAM placements) and
`test_lazy_weights_reject_a_fused_qkv_bias`.

Two mechanical checks stand in for the tests that cannot run here:

1. every keyword passed to `Attention2DConfig`, `MLP2DConfig`, `RMSNorm2DConfig`,
   `LMHead2DConfig`, `Embedding2DConfig`, `RotarySetup2DConfig`,
   `Sampling2DConfig`, and `LazyWeight` exists as a field on that class (AST
   comparison against the module sources) — 0 unknown keywords across both model
   packages;
2. every `from X import n` in the changed files resolves to a top-level name in
   `X` — 0 unresolved (submodule imports excepted, which the checker does not
   model).

## Checkpoint 8 - Two static audits that close Checkpoint 5 gaps 2 and 4

### Decode RoPE contract (gap 2): satisfied for `users_per_column = 8`

Traced from the op's own validation, not from intuition. Decode-mode
`rotary_embedding_llama`
(`ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/rotary_embedding_llama_device_operation.cpp:84-127`)
requires: `input.logical_shape()[0] == 1`; input, cos, sin and the transformation
matrix all `HEIGHT_SHARDED` and bfloat16; `batch = input.logical_shape()[1]` no
larger than the core count; `cos.logical_shape()[1] == batch`;
`cos.shard_spec()->shape[0] == TILE_HEIGHT`; and a transformation matrix with
leading dims `1, 1` and shard shape exactly `(TILE_HEIGHT, TILE_WIDTH)`.

What the Galaxy graph actually produces:

- `all_reduce_create_qkv_heads` computes its own output specs
  (`.../all_reduce_create_qkv_heads/device/all_reduce_create_qkv_heads_device_operation.cpp:190-245`):
  with `slice_size = users_per_column = 8` the batch dimension **is** 8, so
  `q = [1, 8, n_heads/8, head_dim]` and `k = [1, 8, n_kv_heads/8, head_dim]`,
  each height-sharded with shard `(TILE_HEIGHT, head_dim)` over the first eight
  cores of the grid the recipe's `attention_heads_memcfg` supplies (Q takes
  cores 0-7, K the next eight, V the next eight — which is why the qualified
  recipe hands it a 32-core grid rather than an 8-core one).
- `RotarySetup2D.decode_forward` (`models/common/modules/rope/rope_2d.py:97-131`,
  `224-237`, `336-341`) embeds 8 index rows and reshapes to logical
  `(8, 1, head_dim)` padded `(8, TILE, head_dim)`, then unsqueezes to 4D, so
  `cos.logical_shape()[1] == 8` and the shard is `(TILE, head_dim)` on the
  8-core `batch_grid`. The decode transformation matrix is
  `get_rot_transformation_mat(dhead=TILE).repeat(1, 1, 8, 1)` sharded
  `(TILE, TILE)` on the same grid.

Every clause matches: `batch = 8` on both sides, `cos.shard_spec()[0] = 32`,
trans-mat leading dims `1, 1` with a `(32, 32)` shard, and all four tensors are
bfloat16 because the fused head creation is asked for
`dtype=precision.decode_activation_dtype` (bfloat16). `head_dim = 128` also
stays inside the `head_dim <= 128 || !fp32_dest_acc_en` clause. So the default
non-fused decode path needs no config change, and the fused-QK alternative
(`use_qk_fused_rotary=True`, 16 rows) is not required. What remains unproven is
numerical, not structural.

### Qwen decode ring widths (gap 4): the recipe already matches the op

`llama_reduce_scatter` derives its output width in
`ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/llama_reduce_scatter_device_operation.cpp:60-85`:

```
final_width = (input_width % input_shard_width) ? padded_input_width / ring_devices
                                                : input_width / ring_devices
```

with `padded_input_width` the input width rounded up to a whole number of tiles
per input core. `llama_rs_matmul` reuses the same function for its reduce-scatter
output (`.../llama_reduce_scatter_matmul/device/rs_matmul_op.cpp:53-71`), so both
decode call sites agree.

Substituting the recipe's own placement (`mlp_w1_w3_output_memcfg`: 24 receiver
cores, shard width `pad_ring_width(local_hidden_dim) / 24 = 160`):

| Model | `local_hidden_dim` | padded | `width % shard` | op reports | `decode_reduce_scatter_width` |
| --- | --- | --- | --- | --- | --- |
| Llama-3.3-70B | 3584 | 3840 | 64 (non-zero) | 3840 / 4 = **960** | 960 |
| Qwen3-32B | 3200 | 3840 | 0 | 3200 / 4 = **800** | 800 |

`GalaxyDenseGeometry.decode_reduce_scatter_width` reproduces that branch exactly,
so the Qwen decode all-gather key (800) is the width TTNN will report and the
Llama key (960) is unchanged. `input_width % ring_devices == 0` also holds for
both (the op asserts it). **No code change**, and the handoff's suggested
alignment of that property is not needed.

One residual observation, deliberately left alone: the scattered *placement*
`mlp_reduce_scatter_memcfg` is sized from the padded width (960 → 30 cores of one
tile) for both models, so for Qwen five of those cores hold no shard of the
800-wide result. That is the same over-provisioning the qualified Llama recipe
already relies on elsewhere — its own `mlp_w1_w3_output_memcfg` gives 24 cores to
a tensor whose 3584 logical columns pad to 23 shards of 160 — so it is not
evidence of a defect, and per the handoff no qualified recipe was retuned. If the
Qwen decode MLP does fail on the mesh, the one-line experiment is to derive
`reduce_scatter_cores` from `decode_reduce_scatter_width` instead of
`decode_reduce_scatter_padded_width` in
`models/common/models/galaxy/recipes.py`; that is byte-identical for Llama (both
are 960) and yields 25 cores for Qwen.

### New gap found while auditing: paged decode page-table rows *(closed in Checkpoint 10)*

`Attention2D._validate_page_table` (`models/common/modules/attention/attention_2d.py:655-672`)
is called from `decode_forward` with `users = range(max_batch_size)`, so it
demands a page table with **more than 31 rows**, while the Galaxy decode SDPA
batch is one mesh column's eight users (`current_positions` is accepted at
`users_per_column` width, which is what the qualified Milestone A attention test
passed). The Milestone A hardware test ran a *contiguous* cache
(`KVCacheBinding` with `metadata=None`), so the paged decode path has never been
exercised on 2D and this row contract has never been reconciled. `from_pretrained`
always installs a paged config, so the first paged Galaxy decode is likely to hit
this. Recorded, not changed.

## Checkpoint 9 - Unrun step-2 hardware test

`models/common/tests/models/llama33_70b_galaxy/test_model_wh_galaxy.py` builds a
one-layer Llama-3.3-70B through the package's own
`build_llama33_70b_galaxy_model`, prefills 128 tokens for one column-local user,
takes one decode step at position 128, and compares logits against the same
single HF layer (PCC >= 0.99). It skips unless the checkpoint resolves, uses the
`DispatchCoreAxis.COL` + `FabricConfig.FABRIC_1D_RING` device parameters and the
`(8, 4)` mesh fixture, and tears down explicitly.

It deviates from the handoff's sketch in two deliberate ways, both documented in
its module docstring: it constructs the model directly instead of through
`from_pretrained` so it can pass `paged_attention_config=None` (see the page-table
gap above), and it truncates one loaded HF checkpoint to a single layer to serve
both the TT weight conversion and the reference, instead of loading 140 GB twice.
The docstring states plainly that the file has never been executed and lists the
four assumptions most likely to be wrong.

## Checkpoint 10 - Remaining Milestone B sequence (steps 3, 5, 6, 7)

Date: 2026-08-24 (third session). Still a laptop, still no `ttnn`, so **nothing
below was executed**. Evidence is static as before: `python -m py_compile`
(clean across every changed file), a 120-column scan (clean), the two mechanical
AST checks from Checkpoint 7 re-run over the new files (0 unknown config
keywords, 0 unresolved imports beyond two known false positives — tuple-unpacked
constants and a namespace package), a boundary scan (0 model-named imports, 0
cross-package imports), and two op-source reductions recorded below.

Scope covered this session:

| Plan step | Delivered |
| --- | --- |
| 3. Full Llama model and direct demo | `llama33_70b_galaxy/demo.py`, `tests/.../test_full_model_wh_galaxy.py` |
| 5. One Qwen block, decode and prefill | `tests/models/qwen3_32b_galaxy/test_model_wh_galaxy.py` |
| 6. Full Qwen model and direct demo | `qwen3_32b_galaxy/demo.py`, `tests/.../test_full_model_wh_galaxy.py` |
| 7. Paged KV, prefix cache, concat-32, device sampling, long context | Runner, collectives, recipes and the tests listed below |

### Two defects found by reading the op sources, both fixed

Neither was reachable from any qualified path, and neither is expressible
through configuration, which is why both are module changes rather than model
changes.

1. **Decode page-table rows** (the gap recorded in Checkpoint 8).
   `Attention2D.decode_forward` validated the page table against
   `users = range(max_batch_size)`, demanding more than 31 device-local rows.
   `paged_scaled_dot_product_attention_decode`
   (`.../sdpa_decode/device/sdpa_decode_device_operation.cpp:247`) requires
   `page_table_shape[0] == B` with `B = q_shape[1]`, and `paged_update_cache`
   (`.../update_cache/paged_update_cache_device_operation.cpp:170`) requires the
   same. On Galaxy that batch is one mesh column's eight users. Decode now has
   its own validator requiring `users_per_column` rows, or that batch repeated
   once per core for the L1-sharded table layout the legacy stack uses. Prefill
   keeps the by-user contract `paged_fill_cache` needs.
2. **Chunked-SDPA page-table batch.** The prefill table must carry one row per
   filled user, but `chunked_scaled_dot_product_attention`
   (`.../sdpa/device/sdpa_device_operation.cpp:261`) also requires
   `page_table_shape[0] == B`, which is one for a single-row prefill. The two
   requirements are incompatible in one tensor, so the module now slices the
   addressed user's row for SDPA only, exactly as the legacy stack does with its
   one-row-per-column table. A concatenated prefill already matches and is passed
   through.

Both are pinned by host tests in `tests/modules/attention/test_attention_2d.py`.

### Concat-32 prefill

Implemented without touching `Attention2D`, by using the extension point the
module already provides: the injected collectives receive the recipe identity,
so `GalaxyAttentionCollectives.reduce_qkv` splits the reduced projection into
one row per user and `reduce_output` merges the rows back into the residual
stream's single token stream. Everything between them — head creation, RoPE,
the per-row causal SDPA, the per-user paged fill, the WO matmul — already
accepts a 32-row batch.

The collective *resources* need no new family either: every prefill key is
derived from `math.prod(shape[:-1])`, which is `32 * length` on both sides of
the reshape, so registering the prefill collectives at the total token count
covers concatenated prefill exactly. `GalaxyDenseGeometry` gained
`batched_prefill_sequence_lengths` (per-row lengths) and
`collective_prefill_token_counts` (the union that drives registration), and
`resolve_galaxy_prefill_placements` gained the 32-row projection program configs
plus one chunk-aligned SDPA config for the prefix-cached family.

### Device sampling: the column user selector

`LMHead2D` decode logits carry all 32 users on every column; `Sampling2D`
consumes one column's eight, with its top-k/top-p/temperature/seed buffers
sharded the same way. TTNN has no per-column slice, so
`GalaxyColumnUserSelector` performs the selection as a matmul against a one-hot
selector whose *rows differ per column*: the host source is `I(32)` sharded over
columns on the user axis, so column `c` holds rows `8c .. 8c + 7`. The product is
an exact row gather, not an arithmetic mix.

This is new, unqualified composition. It has a standalone hardware test
(`tests/models/galaxy/test_column_user_selector_wh_galaxy.py`) precisely so the
first hardware session can qualify it in seconds instead of inside a 70B demo.
If it fails, the fallback is host sampling, which every other test already uses.

### The direct runner

`models/common/models/galaxy/direct_runner.py` owns the mechanical parts of
driving either model before the Milestone C executors exist: paged KV
allocation, both page-table layouts, position and token staging, last-token
extraction, sampling, chunked prefill, teacher forcing, and deterministic
teardown. It is model neutral and imports no model-named package.

Paged block ownership is static: active slot `u` owns
`[u * blocks_per_user, (u + 1) * blocks_per_user)`. Because the decode graph
always runs the full physical batch, **every idle slot gets its own sink block**
so its unavoidable KV writes can neither touch an active slot's pages nor race
another idle slot. That is what makes the batch-1 long-context smokes possible:
128K context needs one user's blocks plus 31 single-block sinks, not 32 users'
worth of 128K.

The two tables are deliberately asymmetric. The prefill table is right-padded to
eight int32 entries because chunked SDPA reads it in 32-byte sticks; the decode
table is not, because the paged decode SDPA derives its KV length from the row
width and padding would claim more cached context than a slot owns. For the
default 2048-token geometry the widths coincide at 64, so the difference only
appears in short-context configurations — which is why it has its own host test.

### New model surface

Both packages gained the same three methods, in the same place, with identical
bodies: `project_prefill_logits` (normalize the whole token stream, then select
one token per prefill row and project each — mirroring what the legacy stack
does, because `LMHead2D` cannot consume a row count below the physical batch),
`select_decode_column_users`, and `sample_decode`. Parameters gained
`batched_prefill_sequence_lengths` and `chunked_prefill_sequence_lengths`, both
empty by default so no existing construction changes shape.

### Files added

```text
models/common/models/galaxy/direct_runner.py
models/common/models/galaxy/direct_demo.py
models/common/models/llama33_70b_galaxy/demo.py
models/common/models/qwen3_32b_galaxy/demo.py
models/common/tests/models/galaxy/galaxy_hardware.py
models/common/tests/models/galaxy/test_direct_runner.py
models/common/tests/models/galaxy/test_collectives.py
models/common/tests/models/galaxy/test_column_user_selector_wh_galaxy.py
models/common/tests/models/llama33_70b_galaxy/test_full_model_wh_galaxy.py
models/common/tests/models/qwen3_32b_galaxy/test_model_wh_galaxy.py
models/common/tests/models/qwen3_32b_galaxy/test_full_model_wh_galaxy.py
```

## Checkpoint 11 - What is still unqualified, ordered by risk

Everything from Checkpoint 5 that needed a mesh still needs one. New entries:

1. **Nothing has been executed**, including the host-only suites, which need no
   hardware and should run first.
2. **The column user selector.** A matmul-based per-column row gather is a
   composition nobody has run. Qualify it standalone before any device sampling
   result is believed.
3. **Concat-32 prefill.** The reshape is a view only if `ttnn.reshape` keeps the
   buffer when the last dimension is unchanged and the split falls on tile
   boundaries. Every batched row length is a multiple of the 128-token chunk
   alignment, so the tile condition holds; the view assumption does not have a
   test.
4. **Chunked prefill at scale.** The long-context smokes chain 64 chunks at
   128K. Each chunk re-reads the whole cached prefix, and the KV pool is roughly
   2.7 GB per device for Llama. An allocation failure there is a capacity result,
   not a defect.
5. **The accuracy gates are eager.** 511 decode steps of an 80-layer model with
   no trace. Expect minutes, not seconds.
6. **`project_prefill_logits` runs the final norm over the whole token stream**
   before selecting last tokens, because the distributed-statistics gather is
   keyed by that geometry. For a 32-row concatenated prefill that is 32x more
   norm work than the result needs. Correct, not fast; Milestone C should revisit.

## Checkpoint 12 - Suggested order on the Galaxy host

```bash
# 1. Host-only, no hardware. Everything below assumes these pass.
pytest models/common/tests/models/galaxy \
       models/common/tests/models/llama33_70b_galaxy/test_model_host.py \
       models/common/tests/models/qwen3_32b_galaxy/test_model_host.py -v

# 2. Milestone A regression for the changed modules.
pytest models/common/tests/modules/attention/test_attention_2d.py \
       models/common/tests/modules/lm_head/test_lm_head_2d.py -v

# 3. The cheapest new hardware step: qualify the column user selector alone.
pytest models/common/tests/models/galaxy/test_column_user_selector_wh_galaxy.py -v

# 4. One block per model, against a truncated HF reference.
pytest models/common/tests/models/llama33_70b_galaxy/test_model_wh_galaxy.py \
       models/common/tests/models/qwen3_32b_galaxy/test_model_wh_galaxy.py -v

# 5. Full models. Start with the smoke, then the gates.
pytest models/common/tests/models/llama33_70b_galaxy/test_full_model_wh_galaxy.py \
       -v -k "prefill_and_first_decode_token or repeated_requests"
pytest models/common/tests/models/llama33_70b_galaxy/test_full_model_wh_galaxy.py -v

# 6. Direct demos, including concat-32 prefill and device sampling.
#    LLAMA33_70B_GALAXY_DEMO_LAYERS=1 iterates in minutes.
pytest models/common/models/llama33_70b_galaxy/demo.py -v
pytest models/common/models/qwen3_32b_galaxy/demo.py -v
```

## Modularity scorecard

| Required item | Evidence | Assessment |
| --- | --- | --- |
| New 2D/model files added | Seven shared Galaxy files (`dense_transformer.py` was deleted in Checkpoint 7; `direct_runner.py` and `direct_demo.py` added in Checkpoint 10), ten model-package files including both demos, and eleven test files | Within Milestone B boundaries |
| Existing shared files changed | Three 2D module changes across two files (two from Checkpoint 4, one from Checkpoint 10) plus their host tests; three shared Galaxy files extended (`recipes.py`, `plans.py`, `collectives.py`); zero runtime files | Every module change is a contract correction required to issue a valid TTNN call |
| Why config alone was insufficient | `Attention2D` hard-coded a square WO, validated the decode page table against the physical batch instead of the device-local SDPA batch, and passed a by-user page table to an op that requires Q's batch; `LMHead2D` hard-coded a full-width activation. None is expressible through configuration | Minimal, backward-compatible relaxations |
| 1D module implementation files changed | `git status` shows no `models/common/modules/**/*_1d.py` change | Required value met |
| Default runtime behavior changed | No `models/common/llm_runtime/**` file changed in any checkpoint | Required value met |
| 1D regression suites run | None — `ttnn` is not importable in this environment | Outstanding; must run on the Galaxy host |
| Common-code topology assumptions discovered | Two, both 2D-module-local and both found by reading the TTNN op sources rather than by running anything | Recorded and corrected |
| Boundary leakage | Concat-32 needed no `Attention2D` change: the injected collectives already receive the recipe identity, and the collective resource keys are invariant to the reshape. Device sampling needed no module change either — the selector is a model-layer collaborator. Both packages still own their graphs; the boundary scan reports zero model-named and zero cross-package imports | Closed |

## Checkpoint 13 - Rebased onto the final Milestone A tree; C1-C10 closed (job0/reconcile, 2026-08-26)

Host-only. Full account in `tttv2_milestone_b_evidence/reconcile/REPORT.md`.

- Rebased onto Milestone A tip **`bc6ad03bfc2`** (re-derived at run time; the reconciliation report's
  A-side SHAs no longer exist, the branch having been rebased). Merge base `de4c8f4e659` unchanged.
  Final stack, contract-defect commit isolated at the base as the brief requires:
  `a38cc7bf506` (D5/C4/C5) -> `35fe6f34115` (the Milestone B commit) -> `c8c96558ad2` (C1) ->
  `52def65194c` (C6/C9/C2 + the Milestone A status-page corrections). Pre-rebase commit kept as the
  local tag `mb-prerebase-backup`.
- **Zero code conflicts.** Two document conflicts, not one: `MILESTONE_A_STATUS.md` (predicted;
  auto-merged silently, overridden by taking A's side wholesale) and `models/common/modules/README.md`
  (not predicted; resolved by keeping A's closing sentence and B's two additive paragraphs).
- **C1 was real and would have aborted every layer.** Fixed, but not by the brief's literal recipe:
  `plans.py:165` needs a memory config to size the persistent all-gather buffer that
  `_require_fused_stats_placement` inspects, so the function could not simply be deleted. It now
  *derives* the stats origin from the decode residual placement instead of *naming* a core, which
  gives the same guarantee - plan buffer and module-resolved placement are both functions of one
  residual placement - without breaking allocation. Neither model passes `decode_stats_memcfg` any
  more. Guard test in both host suites; it fails against the pre-fix state.
- **D5/C3 was half wrong, and that is the more useful finding.** The swapped `resolve_lazy_weight`
  arguments are real, but unreachable: `_require_exact_weight_policy` runs first and rejects any
  weight whose `memory_config` is not already its own config field, and `resolve_lazy_weight` only
  fills `None` fields. Probed both orderings with two genuinely different configs and got identical
  results. The fix is kept because the code was wrong; no test claims to fail without it, and the
  gate that makes it unreachable is now pinned instead. Recorded as latent in `MILESTONE_A_STATUS.md`.
- C4, C5 isolated with tests that do fail without their change. C2 verified and pinned. C6 promoted
  into `GalaxyModePlan` validation with an explicit `allow_narrow_semaphore_cores` opt-out, because a
  blanket rule would have rejected the fused-RMS narrowing A qualified on hardware. C9 de-duplicated,
  proven field-identical. C7/L3 updated to point at the ring/`gather_in0` recipe, wired but
  unqualified. C8/L1 recorded, not redesigned.
- **Checkpoint 5 item 1 is now closed: the four new host suites and the two updated module suites have
  been executed.** They found four real defects, all test-side against correct production behaviour,
  all fixed: the stale `prefill_wo` fixture the C4 contract rejects (this failed in the Milestone B
  tree too), a `SimpleNamespace` mock that a pybind11 binding rejects, a wrong expected rejection
  message, and a `rank-2` rejection case that built a rank-2 table and so never tested the rank check.
- Host gates at `52def65194c`: modules+galaxy `300 passed`; both model host suites `59 passed`;
  `llm_runtime` `1032 passed, 1 skipped`; module host-only set `260 passed`. Boundary greps all empty.
- **Three unintended device touches, all disclosed in the report §7.** Two were caused by the brief's
  own gate commands collecting device suites (`tests/models/galaxy` pulls in
  `test_column_user_selector_wh_galaxy.py`; `tests/modules` pulls in the 1D hardware matrix, which
  Milestone A P4 deliberately routes off this host). Both were killed; one cost a
  `tt-smi -glx_reset`, which reported `Re-initialized 32 boards`. **The mesh is clean and free.**
  Host-only selections need `--ignore-glob="*_wh_galaxy*.py"` *and* explicit file lists under
  `tests/modules`.
- Still open: the 1D device matrix (Milestone A P4, not this host); five pre-existing host failures in
  packages neither milestone owns, proven independent of this job; and everything in Milestone B
  remains unqualified on hardware.

## Checkpoint 14 — `mb-llama`: first silicon, nine defects, and a dead mesh (2026-08-27)

Plan steps 1-3 for Llama-3.3-70B on WH Galaxy `(8, 4)`. Full account:
`tttv2_milestone_b_evidence/llama/REPORT.md`; handoff:
`tttv2_milestone_b_briefs/job1_completion_handoff.md`.

- **Step 1 host: PASS.** New `test_hf_conversion_host.py`, 9 tests, 3 fresh processes. Proves
  numerically - not by shape - that `reverse_permute` composed with the interleaved rotation the
  device kernel performs is the *same operator* as the HF layout composed with `rotate_half`, at
  `head_dim 128` against the real Llama-3.3 scaled rotary; that the Meta tables carry the scaled
  frequencies pair-duplicated and that llama3 scaling is actually applied; and that converted
  attention/MLP/LM-head weights reproduce the unmodified HF modules at PCC >= 0.9999. The rotation is
  read out of `get_rot_transformation_mat`, the matrix the kernel is handed, so the host reference
  cannot drift from the device one. **This closes the RoPE-convention half of the author's ranked
  risk #1.**
- **Step 1 device: PASS.** One-layer model with real layer-0 weights constructs, seals the
  prefetcher, resolves both CCL contexts, binds/unbinds a KV cache and tears down cleanly in 109 s.
  First Milestone B code ever to run on hardware. **C1/D1 holds at real scale** - it is a hard
  `ValueError` at construction and it did not fire. Run once; reported as one pass, not as
  three-run evidence.
- **Steps 2 and 3: NOT REACHED. No PCC, no accuracy, no demo output exists.** The night went on
  making the decode graph execute at all.
- **Nine defects, eight fixed, all one root cause:** a decode-mode program touching cores the
  sub-device manager does not own. Measured partition: workers `x=1..3` and `x=5..6`, senders `x=0`
  and `x=4`, and **8 cores in no sub-device at all**. The worker envelope is **not contiguous**, so
  its bounding box spans the sender column - and `ttnn.reduce_scatter` uses exactly that bounding
  box. `ttnn::prim::copy`, `ttnn.typecast`'s fallback and the generic reshard use the whole grid.
  Fixed: the rope prefill clone (rope_2d), the rope `batch_grid` (recipes), the embedding decode
  placement (Llama model), `RMSNorm2D` **deallocating its own return value** (latent since A;
  nanobind hands back a new wrapper for the same tensor, so `is not` could not tell "no copy" from
  "copy"), `dense_matmul_program_config`'s missing `allowed_worker_cores`, the attention all-reduce
  op choice, `_relocate`'s three unsafe spellings, and the shared all-reduce buffer dtype.
- **L3 is NOT closed.** The brief's premise was half wrong: the Milestone B recipes moved the *MLP*
  to the ring/`gather_in0` form and left **both attention decode matmuls on the dense `(7,1)` grid
  L3 names**. Confining it with `allowed_worker_cores` makes it legal but leaves three worker
  columns, and its circular buffers then clash with the decode activations there (D-B9, open). The
  structural fix is the ring form; `attention_qkv_collective_input_memcfg` is already shaped for
  those 24 cores, so the design anticipated it.
- **L1 was NOT measured** - the 80-layer model was never built. Job 0's O5 stands unchanged.
- **One change in the tree has never run on hardware**: `in0_block_w` halved in
  `dense_matmul_program_config`, the candidate fix for D-B9. Host-green, device-unverified, flagged
  in the report and handoff. Everything else here ran or is host-only.
- **`BLOCKED (infra)`: the mesh died.** Recurring ETH heartbeat timeouts at mesh open on one ASIC
  (`87032054158471220`), then `Read 0xffffffff over PCIe ID 17`, then `tt-smi -ls` aborting with zero
  boards, then `tt-smi -glx_reset` failing with `[Errno 6] ... '/dev/tenstorrent/7'` - it cannot
  reset the mesh because the node it needs is gone. Two recovery attempts used, both failed. Needs an
  IPMI power cycle or a reboot. 25 device processes and 23 resets this session.
- Host gate at the final code state, driver-free selection: **390 passed, 0 failures**. Standard
  selection: `13 failed, 385 passed`, and all 13 are `test_plans.py` failing to open the dead driver
  - it was **398 passed** one code change earlier on a working mesh. `test_plans.py` is recorded as
  NOT RUN at the final state, by infrastructure failure. Boundary greps for `_1d.py`, `llm_runtime`
  and `qwen` are all empty. No test was deleted, `xfail`ed, skipped or relaxed.

## 2026-08-27 — `mb-qwen` (plan steps 4–6, Qwen3-32B): BLOCKED (infra), host work delivered

- **`BLOCKED (infra)`. Not one device test ran.** The mesh is worse than `mb-llama` left it:
  **eleven** of 32 boards are off the PCIe bus (`0–7, 10, 11, 14`), not one. Both permitted recovery
  attempts were spent and both failed — `tt-smi -glx_reset` cannot start (it must open the missing
  `/dev/tenstorrent/7`), and `tt-smi -r` reset the 21 visible devices then failed re-initialising
  with `Read 0xffffffff over PCIe ID 17`. Neither path can recover a board that is not on the bus.
  Needs an IPMI tray power cycle or a host reboot.
- **`ls /dev/tenstorrent | wc -l` is not a mesh health check on this host.** It returned 32
  throughout while eleven nodes were stale. `ls /sys/class/tenstorrent | wc -l` returned the true 21.
  The house-rules run procedure opens with the misleading one; worth correcting.
- **Second, independent blocker: Qwen3-32B's weights are not on this machine.** The HF cache entry is
  config-only (12 KB). `/proj_sw/user_dev/hf_data` has Llama-3.3-70B but no Qwen. So the full model,
  the demo and the accuracy gate were unreachable even setting the mesh aside.
- **No Qwen PCC, no accuracy number, no demo output exists.** The finish condition was not met.
- **The 64-head decoupled geometry is now qualified on host — it previously had no evidence of any
  kind.** Attention rebuilt from the converted tensors alone (`wqkv`, `wo`, `q_norm`, `k_norm`, Meta
  RoPE tables) reproduces unmodified HF `Qwen3Attention` at PCC ≥ 0.9999, on a fixture with the real
  decoupled ratio rather than Milestone A's square 40-head one. The `wo` pairing
  `(local_attention_dim 1024, local_dim 1280)` is correct and pinned. **Still unqualified on
  silicon**; job 0's O4 stands for the device half.
- **The trap this model hides:** `local_qkv_size == local_dim == 1280` for Qwen3-32B, so a
  fused-QKV-vs-residual width confusion is **shape-invisible**. `local_attention_dim` (1024) is the
  width that differs.
- **Q/K norm, host half done.** `reverse_permute_1d` is proved to be the same permutation
  `reverse_permute` applies to the projection rows, and head-local RMSNorm reproduces HF's
  `Qwen3RMSNorm` at PCC ≥ 0.9999 for both norms. The first version of that test derived its
  permutation from the function under test and was therefore a tautology; it was rewritten to state
  the permutation independently. **No Qwen Q/K-norm number from silicon exists anywhere** — D2's
  defect was that the path aborted before producing one, and it still has not run.
- **Ring widths: the brief was right, and the reason is exact divisibility.** Ring shard is 160;
  Qwen's `local_hidden_dim` 3200 is a multiple of it so the logical width scatters (**key 800**,
  placement 960), Llama's 3584 is not so the padded width scatters (key 960). Arithmetic, not a
  defect. Device-unverified.
- **Risk 4 resolved:** the real `config.json` declares `attention_bias: false`. No contract change
  needed, and a test now asserts it plus every contract field (it runs, since the config is cached
  even though the weights are not).
- **Two placement defects ported from Llama, both UNVERIFIED on device:** Qwen's `_relocate` was
  still the full-grid three-argument `to_memory_config`, and its embedding decode output was
  interleaved `L1_MEMORY_CONFIG`. Both were what job 1 found on silicon for Llama; this package had
  carried them unchanged. Pinned by three new tests — the embedding one was confirmed to fail
  against the unfixed code.
- **`test_model_host.py` never built the transformer config**, so module-to-module placement wiring
  had zero host coverage. It builds fine against the existing `MagicMock(spec=ttnn.MeshDevice)` and
  the decode placements resolve to real `MemoryConfig`s — cheap coverage for the class of defect
  that produced two of this milestone's nine.
- **No shared Galaxy code was touched**, so Llama's evidence is not invalidated and its device gates
  did not need re-running (they could not have been).
- Host gates: **410 passed, 0 failed** driver-free. The brief's gate as written is *not* host-only —
  `models/common/tests/modules` collects device suites, giving 289 device-open errors with the mesh
  down (0 failed). Boundary greps for `_1d.py`, `llm_runtime` and the Llama import are all empty.
- **Deliberate omission:** no new device test files were written, though deliverable 2 asked for
  them. Unrun device code would invite `mb-coverage` to trust it. Recorded as a decision, not an
  oversight.
- No test was deleted, `xfail`ed, skipped or relaxed. No threshold was tuned.

## 2026-08-27 — `mb-coverage` (plan step 7): mesh still dead, step 7 taken as far as the host allows

Full account: `tttv2_milestone_b_evidence/coverage/REPORT.md`.
Handoff: `tttv2_milestone_b_briefs/job3_completion_handoff.md`.

- **The mesh never came back.** Same eleven boards off the PCIe bus as `mb-qwen`
  (`0 1 2 3 4 5 6 7 10 11 14`), same `Read 0xffffffff over PCIe ID 17`, `ttnn` cannot open a
  cluster at all. **No recovery attempt was spent** — two jobs had already used all four and proved
  neither reset path can recover a board that is not on the bus. Logged at the `tt-smi` level and
  at the pytest level.
- **Three consecutive device jobs, zero numerical results from silicon.** No PCC, no accuracy
  number, no demo output, no functional smoke, for either model. The two accuracy gates the brief
  asked to *re-measure* had never been measured by anyone, at any tree — there was nothing to
  compare against.
- **Qwen stays `BLOCKED (upstream)`**: weights still absent (~65 GB), so even a healthy mesh does
  not unblock it. Scope was not halved — the Qwen device coverage was written alongside Llama's.
- **All five areas were attacked at the level that is decidable without a mesh** — block ownership,
  the two page-table layouts and their mappers, the planned tokens/tables/source rows of a
  concatenated prefill, the chunk-table arithmetic, and the exact values `Sampling2D` ships to its
  per-slot buffers. **162 new host tests, identical across three fresh processes.**
- **D-C1 (correctness).** A prefill-shaped page table fed to decode is **accepted**, not rejected.
  `_validate_decode_page_table` discriminates on row count alone and allows any multiple of
  `users_per_column`, because an L1-sharded table repeats the batch per core; the replicated prefill
  table's device-local view is 32 rows and `32 == 4 * 8`. Shape cannot separate the two — placement
  can, and is never read. The step-7 gate asking for rejection **cannot be met by the current
  contract**. Not fixed: an existing 2D module test asserts the 32-row acceptance, and changing that
  expectation is the boundary violation both briefs say to report instead.
- **D-C2 (contract conflict).** "Moving a request to a different slot does not change its stream" is
  **false**: `_seed_digest` is `blake2b("sampling2d:{seed}:{slot}")`. That is deliberate — it stops
  32 slots sharing one seed from collapsing onto one token, which is also now pinned by a test — so
  it is a serving-contract decision, not a bug, and it was measured rather than "fixed".
- **F-C1 (premise correction).** **Llama has no vocabulary padding.** 128256 is already a multiple
  of `8 vocab shards * 32`, so `padded_vocab_size == vocab_size` and `invalid_vocab_mask is None`.
  The padded-vocab gate is *vacuous* for Llama; only Qwen pads (128 ids, masked to bf16 min, proved
  unsampleable at four temperatures on host).
- **G-C1 / G-C2 / G-C3.** Concat-32 needs all 32 slots and does not compose with the `active_slots
  < 32` sink-block mechanism; an empty row is rejected one call too late; and the
  `"chunk_page_table requires a prefix/chunked recipe"` guard is unreachable because a chunk table
  alone already selects the chunked recipe.
- **F-C2 (test infra).** `tests/models/galaxy/test_plans.py` looks host-only but needs a cluster:
  `ttnn.SubDevice` constructs the `MetalContext`. 13 of the 18 baseline host failures are this.
- **The other 5 baseline host failures are `reconcile`'s O2**, re-measured here and proved
  mechanically independent of Milestone B — nothing under `llm_runtime` or any 1D package has
  changed since `bc6ad03bfc2`. The exit-gate line is FAIL as written; the owner is not Milestone B.
- **Long context: capacity accounted rather than smoked.** Per device, replicated pool: 4K 0.14 GiB,
  32K 0.73 GiB, 128K **2.72 GiB** of KV for Llama (2.17 for Qwen), plus 130 MiB of RoPE tables and
  64 chunked-prefill graphs at 128K. Against ~2.3 GiB of weights on a 12 GB device that should fit;
  the risk is fragmentation, not the total.
- **L1 confirmed on the host.** `Prefetcher2D.cleanup()` clears `_global_cb` without handing it to
  `deallocate`, so `owned_resources == ()` is true while the CB is still resident; two owners
  allocate two and free neither. The OOM itself needs real L1. Whether the ordering contract is
  workable at model scale is still unmeasured — the 80-layer model has never been built.
- **No implementation file was changed.** Only tests and evidence. Every finding either needs a mesh
  to validate a fix or needs a product decision first. Both boundary greps stay empty across all 190
  changed paths.
- **Device tests were committed despite never running** (17 Llama, 16 Qwen), against `mb-qwen`'s
  advice, because the gaps are now specific enough that prose would have to be re-derived. Both
  files say "This file has never been executed" in their docstring, with the date and the reason;
  both were verified to collect and nothing more.
- **One host assumption a mesh must check:** `step7_harness.py` models a distributed tensor's
  `.shape` as the *shard* shape, read out of `TensorToMesh::Impl::create_tensor`, not measured.
  D-C1 rests on it; the one-line check is in the handoff.
- No test was deleted, `xfail`ed, skipped or relaxed. No threshold was tuned.

## 2026-08-27 — `mb-signoff` (job 4): the exit gate is NOT PASSED

Host-only, no device taken. Commit read: `9d3ec5799ef`. Evidence:
`tttv2_milestone_b_evidence/signoff/`.

- **Verdict: Milestone B does not pass its exit gate.** 3 of 9 lines PASS (all three mechanical
  boundary checks), 1 PARTIAL, **4 NOT REACHED**, 1 FAIL. Written up in
  `models/common/models/MILESTONE_B_STATUS.md`, with the verdict in the first screen rather than
  after the evidence.
- **The cause is infrastructure, not code**, and the page says so in those words — the distinction
  decides who has to act. Mesh re-checked here at 03:34Z: `ls /sys/class/tenstorrent | wc -l` → 21,
  missing boards `0 1 2 3 4 5 6 7 10 11 14`, unchanged since `mb-qwen`. All four permitted recovery
  attempts were spent by earlier jobs; none was spent here.
- **No numerical result from silicon exists for either model, at any tree, from any job.** No PCC,
  no accuracy figure, no demo output, no functional smoke. Three device jobs each had a night.
  Recorded as "never measured", not as a judgement about what the gates would have shown.
- **Nothing was quoted.** Every mechanical line was re-run here. All three boundary greps empty
  (`_1d.py`, `llm_runtime`, model-named imports) over all 228 changed paths.
- **The regression gate independently reproduced `mb-coverage`'s number**: `18 failed, 2121 passed,
  2059 skipped, 3276 deselected, 351 errors in 1045.84s` against its `... 1048.36s`. Two jobs, two
  processes, identical counts. `HF_HOME` was exported, so the real-checkpoint tests ran.
  Decomposition checked, not accepted: 13 are `F-C2` (`test_plans.py` needs a cluster), 5 are `O2`,
  and all 351 errors are cluster-open.
- **`O2` re-proved mechanically here**: the five failing test files are **byte-identical** to the
  Milestone A tip and `models/common/llm_runtime` is byte-identical (`git diff` → 0 lines). New
  finding: **Milestone A's own 1263-test gate never collected them.** Read out of its log
  (`host01_integrated_gate.log`) — it collected `tests/llm_runtime/`, `tests/modules/` and, under
  `tests/models/`, only `tests/models/galaxy/`. Milestone B is the first milestone to measure that
  exit-gate line, and it was red the first time anyone looked.
- **`L3` corrected in `MILESTONE_A_STATUS.md`, because silicon disproved it.** That page claimed
  Milestone B had moved the decode QKV and `wo` projections to the ring form. Re-verified here at
  `recipes.py:708,711`: both are still `dense_matmul_program_config`; only the MLP moved. `D-B5` is
  right and L3 is **still open**. Surgical edits only — the L3 paragraph, the `D-B` and `D-C`
  deferrable rows (both re-routed to Milestone C), the `L1` mechanism now confirmed on host, and the
  Qwen decoupled-geometry gap now half-closed on host.
- **A documentation defect fixed in `modules/README.md`**: it asserted "A table sized to the full
  physical batch is the prefill layout and decode rejects it." **`D-C1` disproves that** — decode
  accepts it. The README now carries the gap instead of the false contract. Also updated: the L3
  paragraph, the post-record contract amendments, and a new Milestone B section for the two Galaxy
  model packages that states plainly that none of it is qualified on hardware.
- **Two findings routed to a human, not filed as bugs.** `D-C1` (is a decode page table
  discriminated by shape or by placement? — shape cannot do it) and `D-C2` (is a sampling seed
  per-request or per-(request, slot)? — the design and the step-7 gate are in direct conflict).
  Both need a decision before Milestone C builds serving on top of them.
- **Modularity scorecard: the boundaries held.** Zero 1D implementation files changed; `llm_runtime`
  **byte-identical**, not merely behaviour-preserving; no model-named package imported. 17 new
  implementation files (+7841), 6 shared files changed (+289/−20, 3.5% of the implementation diff),
  26 new test files (+9345) — more test code than implementation code. Four topology assumptions
  were discovered in shared code, all on silicon, all now derived values or explicit parameters.
  Recorded as an independent finding: **the boundaries holding and the model tests not running are
  two separate results, and the plan asks for both.**
- **`tttv2_milestone_c_brief.md` written, and explicitly not an authorisation to start.** It carries
  what C inherits working (with commands), what it inherits broken (with evidence), the three items
  routed to it by name — `L1`, `D-A`, and the CCL/`tt_ccl.py` merge evaluation — and the paired
  TTTv1/TTTv2 performance methodology, so C stands the harness up first rather than retrofitting it.
  Three known performance debts are listed against those thresholds up front.
- No device work attempted. No test written, deleted, `xfail`ed, skipped or relaxed. No threshold
  touched. This job changed **no implementation file** — four markdown files only.

## `mb-llama`, attempt 2 — 2026-08-27, WH Galaxy `(8, 4)`, recovered mesh

Attempt 1 ended `BLOCKED (infra)` with board 7 off the PCIe bus. The machine was
power-cycled out of band between the attempts, so attempt 2 could do device work.
It re-verified the mesh first rather than trusting the handoff:
`tt-smi -ls` enumerated 32 boards including board 7, and
`test_partition_wh_galaxy.py` passed 5/5 on device with a clean 32-chip open and
close (`tttv2_milestone_b_evidence/llama/logs2/a2_00_partition.log`). The
partition numbers were re-measured and are identical to attempt 1's.

- **`D-B9` is CLOSED, and the fix was attempt 1's own untested hypothesis.**
  `in0_block_w` from `gcd(k_tiles, 8)` to `gcd(k_tiles, 4)` in
  `dense_matmul_program_config` works on hardware: the
  `Statically allocated circular buffers ... clash with L1 buffers on core range
  [1-0 - 3-0]` abort does not recur, in four separate processes. Both attention
  decode projections now execute inside the three-column worker rectangle.
- **The decode graph reaches the LM head.** A whole Llama layer — distributed
  norm, QKV, production RoPE on real Q/K, SDPA, `wo`, the attention all-reduce,
  all three MLP ring matmuls and the axis-0 all-reduce — *and* the final
  distributed norm now execute on silicon with real layer-0 weights. Attempt 1's
  "final norm, LM head, logits: never reached" is superseded.
- **Four new defects, all in the decode LM head, all found on device, all fixed**
  (`D-B10`..`D-B13`): `_relocate` reaching `ttnn::prim::copy` for an interleaved
  non-DRAM target; the LM head having no program config at all and needing the
  24-core `gather_in0` ring rather than the dense three-column form; its output
  placed on `ring_receiver_cores()` when `gather_in0` with a DRAM-interleaved in1
  demands the *same* core set and order as in0; and `ttnn.linear` never being
  given a `sub_device_id`.
- **The generalisation that matters, and it is new.** Several ttnn ops do not
  default to "the whole compute grid" when given no `sub_device_id` — they
  default to **sub-device zero**, which on this mesh is the prefetch sender set,
  the one group of cores a compute program must never touch. The symptom is
  `TT_FATAL ... Expecting a non-empty CoreRangeSet!` from `CreateSemaphore`,
  because the op intersects its cores with the sub-device's and gets nothing.
  That is a worse default than the whole grid, which at least contains the right
  cores.
- **Two silent-failure traps found by reading, not running.**
  `galaxy_hardware.load_reference_tokens` returned a `(1, 1024)` tensor where
  every consumer treats the sequence as flat, so `len()` was 1 and **the
  Milestone B accuracy gate could only ever have _skipped_** — failing open, for
  both models. And every `*_weights_memcfgs` entry in `LMHead2DConfig` is inert,
  because `resolve_lazy_weight` fills only `None` fields and `_lazy` always sets
  a memory config.
- **Test coverage added**: KV-cache PCC against HF's own `past_key_values` after
  both prefill and decode, for all four prefilled user rows; a prefill-2048 case
  with its own recipe family; and the step-2 reference switched to
  `load_layer_subset_causal_lm`, which turns a ten-minute per-process setup into
  seconds and makes the three-runs rule affordable at all.
- **Two shared 2D modules changed, declared in the report**: `lm_head_2d.py`
  (gained the sub-device and mask-placement config surface it was missing — there
  was no value any model could have set) and `galaxy/collectives.py`
  (`GalaxyColumnAllReduce` gained an optional `subdevice_id`). No `*_1d.py` and no
  `llm_runtime` file changed; both greps empty. No test deleted, `xfail`ed or
  relaxed; the two `test_lm_head_2d.py` failures a first draft caused were fixed
  by changing the *code*, and 20/20 pass unmodified.
- **The mesh broke again mid-session, differently.** The routine post-run
  `tt-smi -glx_reset` after run 07 timed out inside `POST_RESET`, leaving a
  chip's ARC firmware wedged: `ARC startup error at core 0-10 over NOC0 ...
  Timed out after 300000 ms`. All 32 PCIe nodes stayed present, so this is not
  attempt 1's fault. Recovery and the resulting verdict are in the report.

## 2026-08-27 — `mb-llama` attempt 3: the step-2 gate is met, on silicon

Evidence: `tttv2_milestone_b_evidence/llama/REPORT.md` §"Attempt 3", run-by-run in
`ATTEMPT3.md`, logs in `logs3/`. Commits `361245c08eb..` on
`apbernal/tttv2_wh_glx_2d_modules_milestone_b`.

- **The first PCC numbers this model has ever produced.** One Llama block, real
  `meta-llama/Llama-3.3-70B-Instruct` layer-0 weights on WH Galaxy `(8, 4)`,
  against an independent Hugging Face reference, **three runs in three fresh
  processes with bit-identical results**: prefill 128 logits 0.99958, prefill 2048
  logits 0.99962, decode at batch 32 logits 0.99975, and the K/V cache after both
  prefill and decode at 0.99993 / 0.99975 — on all four column-local users. The
  step-2 gate of `job1_llama.md` is **MET**.
- **D-B19 closed, and it was a width.** Attempt 2's LM head all-reduce hang was
  the reduced logits being 501 tiles per device inside a 42-core x 12-tile spec:
  `all_reduce_async`'s reduction kernel does
  `cb_in.wait_front(ring_size * block_num_tiles)` on *every* output core, so the
  42nd core waited forever for a shard that was never full. 501 has no divisor
  between 4 and 50, so no core count could have fixed it. `galaxy_padded_vocab_size`
  now pads to a ring-exact width (Llama 129024, Qwen 153600), which is what
  production does by a different route.
- **Six more defects, two of which fail open.** The prefetcher's global CB made
  prefill unplaceable (D-B20); the prefill RoPE tables were row-major (D-B21) and
  its transformation matrix the wrong size (D-B22); the logits composed along the
  wrong mesh axis **silently**, and `GalaxyDirectRunner` did the same and then
  narrowed without raising, so every step-3 number would have been wrong with no
  symptom (D-B23); the KV reference was in the wrong RoPE convention (D-B24); the
  MLP read the attention's prefetched weights, again silently (D-B25a); and the
  non-fused decode RoPE wrote a K of `|max| = inf` into the cache (D-B25b).
- **Two of the brief's four ranked risks are discharged.** RoPE composed with
  `Attention2D` was indeed the first failure, and the fault was *which op*: on a
  prefetcher mesh production uses `rotary_embedding_llama_fused_qk` and the
  non-fused pair is the Blackhole fallback. And the fused decode norm is right —
  0.99999 on an exact input — so job 0's C1 fix holds on hardware.
- **Paged decode works on this partition**, measured by a one-layer runner smoke.
  That closes the step-3/step-7 dependency attempt 2 recorded: `from_pretrained`
  has no contiguous option, so every 80-layer path is paged whether it wants to be
  or not.
- **Four shared modules changed, all declared in the report** with the reduction
  the extension discipline asks for: `prefetcher_2d.py` (a new config value,
  defaulting to the old behaviour), `rope_2d.py` (two corrections where the module
  disagreed with the op), `lm_head_2d.py` and `sampling_2d.py` (a validation that
  forbade a legal geometry, loosened in the direction hardware requires). No
  `*_1d.py`, no `llm_runtime`; both greps empty. No test deleted, `xfail`ed or
  weakened — three host assertions were *corrected*, each with the device abort
  that refuted it quoted against it.
- **Step 3 is met too, and measured three times each.** The full 80-layer model
  prefills 128 real reference tokens and decodes one more, both predictions inside
  the reference model's top-5; the **Milestone B teacher-forced accuracy gate for
  Llama passes at top-1 501/511 = 98.04% (gate 91%) and top-5 511/511 = 100.00%
  (gate 99%)**, identical counts across three fresh processes; and the 80-layer
  demo produces fluent English at batch 1 and on all 32 slots of the physical batch
  ("A tensor is a multi-dimensional array of numerical values, similar to a
  matrix,"), with slot 0 character-identical served alone or alongside 31 others.
  Host regression gate 565 passed, exit 0.
- **One step-3 item is not met and is reported, not worked around.** Two runners in
  one process fails: the second prefills after the first decoded, and the
  prefetcher's global circular buffer is resident again — limitation L1's remaining
  half, which `defer_global_cb` narrowed rather than removed. Its obvious fix
  (release on prefill, recreate on decode) is implemented behind a default-off flag
  and **refuted on hardware**: the release runs and the L1 is not returned, because
  a `global_circular_buffer` has no `deallocate`. The better hypothesis — confine
  the prefill mode plan to the worker cores so a full-grid prefill program never
  needs the sender columns — is written down for the next session, along with the
  re-validation it implies.
- **Paged decode works on this partition**, which closes the step-3/step-7
  dependency attempt 2 recorded: `from_pretrained` has no contiguous option, so
  every 80-layer path is paged whether step 3 wants it or not.

## 2026-08-27 — `mb-qwen`, attempt 2 (device)

Mesh healthy, 32/32 in `/sys/class/tenstorrent`; `test_partition_wh_galaxy.py`
5 passed in 12.93 s. `HF_HOME` inherited from the shell pointed one directory too
deep (`.../hf_data/hub`, a cache holding only Mistral), under which every Qwen
test *skips*; every harness script here exports `/localdev/ctr-apbernal/hf_data`.

Ported Llama's six model-code fixes into the Qwen package (prefetch registration,
fused-QK rotary default, decode LM head on the ring, `_relocate`'s one-hop
interleaved target, the checkpoint loader seam, demo output printing), then found
and fixed two defects that are Qwen-only because Llama has no per-head Q/K norm
and a differently sized vocabulary:

* **D-B26** — the head-local Q/K decode norm was unplaceable. Interleaved
  `ttnn.rms_norm` spreads over the whole compute grid (`Kernel group cores do not
  match sub device cores`); the created heads are HEIGHT_SHARDED, which the op
  rejects; and naming any single sharded placement relocates Q and K onto the
  same cores, which the fused QK rotary rejects (`Q and K must not overlap`).
  Resolved by naming only the *cores* the kernel may use and returning each
  tensor to the placement it arrived in. Milestone A's D2, other half.
* **D-B27** — `lm_head_reduce_core_count` gave Qwen all 50 worker cores, leaving
  `all_reduce_async` none for its fabric links; it warns and then
  segmentation-faults. Now reserves four. Llama still resolves 42, Qwen 40.

Step-5 gate met, three fresh processes, bit-identical:

```text
prefill 128 logits                       0.999303669584255
prefill 128 cache K / V (users 0,8,16,24) 0.9998897994661545 / 0.9998944730661905
decode position 128 logits (u0,8,16,24)  0.999360219056066
decode 128 cache K / V  (users 0,8,16,24) 0.9998896420783983 / 0.9998939662639094
per-head Q/K norm, all 32 devices        prefill 0.99998 / decode 0.99999
```

Full account: `tttv2_milestone_b_evidence/qwen/REPORT.md`.
