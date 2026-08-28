# `mb-coverage` — Milestone B step 7: paged KV, concat-32, prefix cache, device sampling, long context

Written 2026-08-27, unattended, at commit `0c1ccd8557c7cb25cd1ca300d522eab1ed5db733`
on `apbernal/tttv2_wh_glx_2d_modules_milestone_b`.
Environment and mesh facts: `ENVIRONMENT.md`. Raw logs: `logs/`.

---

## Read this first

**The mesh is still down, in exactly the state `mb-qwen` left it.** Eleven of 32
boards are off the PCIe bus (`0 1 2 3 4 5 6 7 10 11 14`). `ttnn` cannot open a
cluster at all: every attempt dies at
`TTDevice::is_pcie_hung — Read 0xffffffff over PCIe ID 17`. **No recovery
attempt was spent**, because two jobs have now proved that neither
`tt-smi -glx_reset` nor `tt-smi -r` can bring back a board that is not on the
bus, and `mb-qwen`'s handoff says so explicitly.

So: **every device line of the Milestone B exit gate is `NOT REACHED`, and this
job measured none of them.** There is still no numerical result from silicon for
either model, of any kind. That has now been true for three consecutive jobs.

**Qwen is additionally blocked upstream.** Its weights are not on this machine —
`config.json` only, ~65 GB still to fetch. Per this job's brief, the Qwen half
of every area is recorded `BLOCKED (upstream)` and the scope was not quietly
halved: the Qwen device coverage was written, and it is marked never-executed
like the Llama half.

**What this job did instead.** Most of what makes step 7 *correct* is decided on
the host before a single TTNN call: which blocks a slot owns, which of the two
page-table layouts a call stages and how it is mapped, which tokens and source
rows a concatenated prefill plans, and what values `Sampling2D` writes into its
per-slot buffers. All five areas were attacked at that level, with **162 new
host tests that pass identically in three fresh processes**, plus 33 new device
tests that are written, collectible, and honestly labelled as never executed.

That produced **three defects/gaps and two corrections to the brief's own
premises** — including one, D-C1, that says a gate the brief asks for cannot be
met by the current contract at all.

---

## Summary by area

| # | Area | Host-decidable half | Device half | Findings |
| --- | --- | --- | --- | --- |
| 1 | Paged KV | **PASS** — 39 tests | `NOT REACHED` (paged-vs-contiguous PCC) | **D-C1** |
| 2 | Concat-32 physical prefill | **PASS** — 34 tests, lengths 128→2048 ascending | `NOT REACHED` | **G-C1**, **G-C2** |
| 3 | Prefix-cached / chunked prefill | **PASS** — 19 tests | `NOT REACHED` (the numerical gate) | **G-C3** |
| 4 | Device sampling | **PASS** — 26 tests | `NOT REACHED` | **D-C2**, **F-C1** |
| 5 | Long context 4K/32K/128K | **PASS** — 32 tests (capacity accounting) | `NOT REACHED` (functional smokes) | capacity table below |
| — | Repeat and cleanup | **PASS** — 12 tests | `NOT REACHED` (the L1 OOM itself) | L1 confirmed on host |
| — | Regression gates | **PASS**, boundaries clean | n/a | **F-C2** |

Per-area detail follows the exit-gate table.

---

## The Milestone B exit gate, measured at this tree

Every row carries the command that produced it. Nothing here is quoted from
`mb-llama` or `mb-qwen`.

| Gate line | Result | Measured value | Command / log |
| --- | --- | --- | --- |
| Llama teacher-forced, batch 1, prefill 512 / decode 511, top-1 ≥ 91%, top-5 ≥ 99% | **NOT REACHED** | none — cluster open fails | `pytest models/common/tests/models/llama33_70b_galaxy/test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_teacher_forced_accuracy_batch1` → `1 error in 7.72s`, `logs/12_device_attempt_*teacher_forced*.log` |
| Qwen teacher-forced, batch 1, sequence 512, top-1 ≥ 89%, top-5 ≥ 97% | **BLOCKED (upstream)** + NOT REACHED | none — weights absent *and* no mesh | `logs/12_device_attempt_*qwen*.log`; `ENVIRONMENT.md` §Checkpoints |
| Batch-32 direct demos valid, no cross-slot contamination | **PARTIAL** | mechanism proved on host; no device demo output | host: `test_step7_paged_kv.py::test_no_two_slots_can_address_the_same_block` (5 params) + `..._sink_never_lands_in_an_active_slots_run` — 8 passed ×3 processes. device: `models/common/models/llama33_70b_galaxy/demo.py` never executed |
| Batch-1 4K / 32K / 128K functional smokes | **NOT REACHED** | capacity accounting only (table below) | host: `test_step7_long_context.py` — 32 passed ×3. device: `test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_long_context_smoke` errors at cluster open |
| Prefix-cached output matches uncached execution | **NOT REACHED** | addressing proved on host; no PCC | host: `test_step7_prefix_cache.py` — 19 passed ×3. device: new `test_step7_coverage_wh_galaxy.py::test_llama_chunked_prefill_matches_a_single_uncached_prefill`, never executed |
| No dependency imports from an existing model-named implementation package | **PASS** | 0 matches | grep below |
| Zero changes to 1D module implementation files | **PASS** | 0 files | `git diff --name-only bc6ad03bfc2..HEAD \| grep '_1d\.py'` → empty |
| Zero changes to `llm_runtime` | **PASS** | 0 files | `git diff --name-only bc6ad03bfc2..HEAD \| grep 'llm_runtime'` → empty |
| Existing 1D model contract and demo-contract host tests green, expectations unchanged | **FAIL** (inherited, not caused here) | 5 failures | see below |

### The `re-measure, do not quote` instruction

The brief asks that the two accuracy numbers be re-measured at this tree rather
than quoted from `mb-llama` and `mb-qwen`. **Neither job ever measured one.**
Both recorded `BLOCKED (infra)` with no accuracy number at all. There is
therefore nothing to re-measure and nothing to disagree with; the honest
statement is that the two accuracy gates have never been measured, by anybody,
at any tree.

### The 1D demo-contract line

Re-measured here, and it is the same set `reconcile` recorded as its finding O2:

```text
FAILED models/common/tests/models/deepseek_r1_distill_qwen_14b/test_demo_contract.py::test_eval_prefill_signature_multiset_is_rotation_invariant_and_not_static_warmup_shaped
FAILED models/common/tests/models/qwen2_7b/test_demo_contract.py::test_eval_prefill_signature_multiset_is_rotation_invariant_and_not_static_warmup_shaped
FAILED models/common/tests/models/qwen25_7b/test_demo_contract.py::test_eval_prefill_signature_multiset_is_rotation_invariant_and_not_static_warmup_shaped
FAILED models/common/tests/models/llama33_70b/test_demo_contract.py::test_demo_resolves_central_trace_region_size_for_each_supported_sku
FAILED models/common/tests/models/llama32_3b/test_hf_adaptor.py::test_generator_downgrades_n150_all_trace_to_decode_only
```

Proved independent of Milestone B, mechanically rather than by assertion: the
complete set of files changed between the Milestone A tip `bc6ad03bfc2` and HEAD
is `models/common/models/{galaxy,llama33_70b_galaxy,qwen3_32b_galaxy}`, four 2D
module files, their tests, and markdown. Nothing under `models/common/llm_runtime`,
nothing in any 1D model package, and none of those five test files.
`_plan_prefill_requests` — the function three of the five failures land in — is
`llm_runtime` code and is byte-identical to Milestone A's.

```sh
git diff --name-only bc6ad03bfc2..HEAD | sed 's|/[^/]*$||' | sort | uniq -c
git diff --name-only bc6ad03bfc2..HEAD | grep -v '^models/common/\(models\|modules\|tests\)/' | grep -v '^tttv2'   # empty
```

So the gate line is **FAIL as measured** and **not Milestone B's to fix**. It
belongs to whoever owns those packages, and `mb-signoff` should record it that
way rather than as a Milestone B regression.

### Model-named import gate

```sh
grep -rnE 'from models\.(demos|common\.models\.(llama33_70b|qwen3_32b))[. ]' \
     --include='*.py' models/common/models/galaxy models/common/models/llama33_70b_galaxy \
     models/common/models/qwen3_32b_galaxy
```

`logs/15_import_boundary_20260827T030156Z.log` — 0 matches.

---

## Area 1 — Paged KV

`models/common/tests/models/galaxy/test_step7_paged_kv.py`, 39 tests.

### Proved on the host

| Claim | How |
| --- | --- |
| No two slots can address the same block, at active batch 1, 8, 16, 31, 32 | pairwise set intersection over all 32 rows of the real `_page_table_rows()` |
| No idle slot's sink block lands inside an active slot's run | same, split active/idle |
| Every addressed block is inside the allocated pool | `min >= 0`, `max < max_num_blocks` |
| Prefill replicates a **padded** table; decode shards an **unpadded** one | staged mapper and width captured at `ttnn.from_torch` |
| The decode table's device-local view is `[8, blocks]`; prefill's is `[32, blocks]` | shard arithmetic modelled from `TensorToMesh::Impl::create_tensor` |
| Late capacity resolution reaches every layer, for **both** model classes | `configure_paged_attention` on a detached model, then `local_cache_shape()` |
| Capacity cannot be re-resolved while a cache is bound | `RuntimeError("cannot be reconfigured")` |
| A bind that fails part-way leaves **no** layer bound | layer 2's cache given the wrong dtype; layers 0–1 already bound |
| A malformed layer entry unwinds every earlier layer | one tensor instead of two |
| Unbind is transactional, idempotent, and owner-only | `PermissionError` for a second owner |
| Rebinding replaces rather than stacks | binding identity compared |

Both model classes (`Llama33_70BGalaxyTransformer2D`, `Qwen3_32BGalaxyTransformer2D`)
are parametrized: their `set_kv_cache` and `configure_paged_attention` are
character-identical, and pinning both keeps them that way.

### `NOT REACHED`

Paged fill during prefill, then decode reading the same blocks, at PCC ≥ 0.99
against the contiguous path. Written as
`test_step7_coverage_wh_galaxy.py::test_{llama,qwen}_paged_and_contiguous_caches_agree`,
never executed. **Nothing in this tree has ever compared the two cache layouts.**

### D-C1 — a prefill-shaped page table fed to decode is accepted, not rejected

**Severity: correctness. This is the one gate in the brief that the current
contract cannot meet.**

The brief asks: *"feed decode a prefill-shaped table and assert it is rejected,
not silently accepted."* It is not rejected.

`Attention2D._validate_decode_page_table` discriminates on **row count alone**:

```python
per_column = self.config.users_per_column          # 8
if shape[0] < per_column or shape[0] % per_column:
    raise ValueError(...)
```

The modulo is deliberate — an L1-sharded decode table legitimately repeats the
device-local batch once per core. But the replicated prefill table's
device-local view is **32 rows**, and `32 == 4 * 8`, so it passes the row check.
The width check then passes too, because the prefill table is stick-aligned to
eight int32 entries and is therefore *wider* than the decode table, never
narrower. The dtype matches. The table reaches `paged_update_cache` and the
paged decode SDPA with the wrong layout.

**Why shape cannot fix it.** `ttnn` reports a distributed tensor's `.shape` as
the *shard* shape: `TensorToMesh::Impl::create_tensor` builds the output
`Tensor` from `compute_tensor_spec_for_shards`, for both the host-tensor and the
raw-buffer entry points. So the correct decode table presents `[8, W]` and the
prefill table presents `[32, W_padded]` — and `[32, W]` is *also* the legal
4-core L1-sharded form. Two different things with the same rank-2 shape.

**The discriminator that would work, and is never consulted:** placement. The
prefill table is DRAM-interleaved and replicated; a legitimate repeat is L1
height-sharded over exactly `rows / users_per_column` cores.
`_validate_decode_page_table` never calls `memory_config()`.

**Not fixed here, on purpose.** An existing 2D module test,
`test_attention_2d.py::test_decode_page_table_accepts_the_device_local_batch_and_its_core_repeats[32]`,
asserts that a 32-row table *is* accepted. Making decode reject it requires
changing that expectation, and the brief is explicit that changing an existing
expectation to accommodate this work is a boundary violation to report rather
than to commit. It also cannot be validated without a mesh.

**Proposed fix, for `mb-signoff` and Milestone C.** In
`_validate_decode_page_table`, require:

* `shape[0] == users_per_column` when the table's memory config is interleaved; or
* `shape[0] == users_per_column * n_cores` when it is L1 height-sharded, with
  `n_cores` read from the shard spec.

Then update the module test's `rows=32` case to supply an L1-sharded table, and
add the interleaved-32 case as a rejection. That is a coherent, testable change;
it is just not this job's to make unilaterally.

**Pinned, not papered over.** `test_step7_paged_kv.py::test_decode_cannot_tell_the_prefill_layout_from_a_four_core_l1_repeat`
records the behaviour that exists, and its docstring says in full why it is not
the behaviour the gate asks for. The reverse direction *does* fail closed and is
also pinned: a decode-shaped 8-row table handed to a prefill that fills user 8
raises `"page_table must have one row for every addressed user"`.

---

## Area 2 — Concat-32 physical prefill

`models/common/tests/models/galaxy/test_step7_concat32.py`, 34 tests.

The plan's risk is *padding inactive rows must not write KV or return logits for
inactive slots*. All three artefacts the brief names were inspected directly.

### The planned tokens

At lengths **128, 256, 512, 1024, 2048** — ascending, never jumping to 2048 —
the flat stream `prefill_batched` builds gives row *r* exactly
`[r * length, (r + 1) * length)` and nothing else. With active batches **16, 31
and 32**, every padded position in a row's span is token id `0`; no row's
padding ever carries a neighbour's token.

### The page table

The concatenated call passes the **replicated** prefill table, names
`user_ids == tuple(range(32))`, and passes no chunk table, no chunk start and no
prefix user. Verified at all five lengths.

### The source rows

With a deliberately non-identity user order (`reversed(range(32))`),
`_fill_prefill_cache` issues exactly one K and one V `paged_fill_cache` per row,
each with `batch_idx=0` against a **one-row slice** of the table, and the slices
come out in row order naming that row's user:
`[(slice(31, 32), …), (slice(30, 31), …), …]`. A row cannot address a user it was
not assigned.

### Logit isolation

`token_indices` addresses each row's last **real** token — `len(row) - 1`, not
`sequence_length - 1` — at active 16, 31 and 32. A padded row's logit is
computed from its own single real token, never from a zero.

### G-C1 — active batches 16 and 31 are not expressible as a smaller allocation

Recorded limitation, not a defect.

Two isolation mechanisms exist and **they do not compose**:

* `GalaxyDirectRunner(active_slots=k)` gives each idle slot its own sink block;
* `prefill_batched` refuses any runner with `active_slots != 32`
  (`"concatenated prefill needs exactly 32 active rows"`), and
  `Attention2D._recipe_identity` resolves only `SINGLE_ROW` or `CONCAT_32` —
  a 16- or 31-row prefill raises `"prefill recipes support exactly one row or
  concat-32 users"`.

So "active batch 16" through the concat path means *32 physical rows of which 16
carry real prompts*, which is what this suite measures. A 16-slot paged
allocation and a concatenated prefill cannot be used together. Both facts are
pinned by tests.

### G-C2 — an empty row is caught one call too late

`generate` refuses an empty prompt outright. `prefill_batched` called directly
does not: it plans `token_indices[r] == -1` and leaves the rejection to
`project_prefill_logits`. The rejection *does* happen, so no padded logit can be
returned — but only after the whole concatenated prefill graph has run. Minor;
worth an early check in the runner.

### `NOT REACHED`

Device KV and logit isolation at active 16/31/32, and concat-32 agreeing with
sequential prefill at each length. Written as
`test_step7_coverage_wh_galaxy.py::test_{llama,qwen}_concat32_*`, never executed.

---

## Area 3 — Prefix-cached and chunked prefill

`models/common/tests/models/galaxy/test_step7_prefix_cache.py`, 19 tests.

### Proved on the host

* A chunk table staged for chunk *c* starts at block `c * chunk / block_size` for
  every slot, is stick-aligned to eight entries, and pads with zeros only —
  checked against the real `_page_table_rows()` at chunks 1, 2 and 7.
* A chunk table never shares a block between two slots.
* An unaligned `chunk_start`, and a chunk past a slot's allocation, both fail
  closed.
* The chunked plan is right: chunk 0 is an ordinary prefill; every later chunk
  carries `chunk_start`, its own chunk table, and `prefix_user_id == slot`.
* Every chunk table is deallocated before the next chunk is staged — a chunk
  table that outlived its chunk would leak once per chunk across a long context.
* **Interaction, prefix-cached then normal**: after a chunked prefill on slot 0,
  a plain `prefill_row` on slot 1 plans with no prefix user, no chunk start, no
  chunk table, and the full replicated table.
* **Interaction, a mix across slots**: interleaving chunked and plain requests
  leaves every call addressing exactly one slot; no request widens another's
  batch.
* The single-row slice chunked SDPA needs is taken, and it follows
  `prefix_user_id` (`table[27:28, :]`), falling back to `user_ids[0]`
  (`table[4:5, :]`) when it is absent. An already-single-row table passes
  through; a concat-32 call keeps the full table because Q carries every row.

### A contract fact worth knowing

`_validate_prefill` requires `prefix_user_id in user_ids`. For a single-row
prefill that forces `prefix_user_id == user_ids[0]`, so the two branches in
`_sdpa_page_table` can only differ on a call that bypassed validation. The
branch is defensive, not load-bearing. Pinned both ways.

### G-C3 — the `chunk_page_table` guard is unreachable

`_recipe_identity` treats a non-`None` `chunk_page_table` as one of the four
signals that select `PREFIX_CHUNKED`. By the time `_validate_prefill` reaches

```python
if metadata.chunk_page_table is not None:
    raise ValueError("chunk_page_table requires a prefix/chunked recipe")
```

the recipe is *already* `PREFIX_CHUNKED`, so the branch can never fire. Passing a
chunk table with no chunk start silently runs the chunked recipe from token 0
instead of being refused. Dead code plus a missing check; pinned by
`test_a_chunk_page_table_alone_selects_the_prefix_chunked_recipe`.

### `NOT REACHED`

The gate itself — prefix-cached output matching uncached execution under the
model's numerical acceptance. Written for both models, never executed.

---

## Area 4 — Device sampling

`models/common/tests/models/galaxy/test_step7_sampling.py`, 26 tests.

### D-C2 — "moving a request to a different slot does not change its stream" is false

**Severity: contract conflict. Measured, and deliberately not "fixed".**

Both the device seed and the host seed are

```python
_seed_digest(seed, slot) = blake2b(f"sampling2d:{seed}:{slot}")
```

so the slot is part of the key. `_device_seed(1234, 3) != _device_seed(1234, 7)`,
and a request with one seed and one set of logits samples a different token in
slot 3 than in slot 7.

The brief's other clause **does** hold and is proved: the same seed in the same
slot gives the same token across runs, across three freshly constructed sampler
objects, and a request's *row position within a call* does not change its stream
(slot 25 at row 0 == slot 25 at row 2).

The slot mixing is not an accident — it is what stops 32 slots given one seed by
a serving front end from all emitting the same token, which is also proved here
(`test_one_seed_across_every_slot_does_not_collapse_the_batch`). The step-7
requirement and the module's design are in direct conflict. Resolving it is a
product decision — *is a seed per-request or per-(request, slot)?* — not a bug
fix, so this job measured it and left the module alone. **`mb-signoff` should
put this in front of whoever owns the serving contract.**

### F-C1 — Llama has no vocabulary padding, so its padded-vocab gate is vacuous

The brief says *"Llama's 128256 and Qwen's 151936 both pad"*. Llama does not.

```text
Galaxy alignment = 8 vocab shards * 32 = 256
128256 / 256 = 501 exactly  ->  padded_vocab_size == vocab_size == 128256
151936 / 256 = 593.5        ->  padded_vocab_size == 152064, 128 invalid ids
```

`build_invalid_vocab_mask(128256, 128256, 32)` returns `None` and
`Sampling2D(...).config.invalid_vocab_mask is None`. There is nothing to mask
for Llama and nothing that can be sampled. **A Llama pass on this gate would be
evidence of nothing**, which is why the device version of the case lives only in
the Qwen file. Asserted explicitly so the premise cannot quietly return.

For Qwen the gate is real and passes on host: the mask is `finfo(bfloat16).min`
on exactly the 128 padded ids, additively below every real logit, and no padded
id is sampled at temperature 0.0, 0.7, 1.0 or 2.0 even when every padded entry
carries a logit of `1e4`.

### Also proved on the host

| Claim | Detail |
| --- | --- |
| Greedy equals host argmax exactly | both vocabularies, 8 rows, `torch.equal` |
| `forced_argmax` and `temperature == 0` agree | same tokens with identical seeds |
| Per-slot heterogeneous top-k / top-p / temperature | slots 0, 5, 8, 17, 31 given five different triples; buffers read back per global slot; unnamed slots keep the greedy defaults `(1, 0.0, 1.0)` |
| **The temperature reciprocal pairing (defect D4)** | at T ∈ {0.25, 0.5, 0.8, 2.0, 4.0}: the runner hands the module the **raw** T, and the module writes **1/T** into the buffer. Never checked at T = 1.0, which is its own reciprocal and is what hid D4 |
| The runner's own host reference divides by T | low T concentrates on the largest logit; high T spreads across every candidate |
| The runner forces argmax for a greedy policy | `forced_argmax is True` reaches `sample_decode` |

`top_k > 32` was **not** tested and the contract was not extended, as the brief
requires.

### The composition property no single module can check

The sampler's slot→column map, the column selector's row gather, and the
runner's decode position sharding must all put global slot *s* on mesh column
`s // 8`. If any two disagree, a user samples from another user's logits — a
cross-slot contamination bug invisible to every per-module test.

Verified: `Sampling2D.slot_placement(s) == divmod(s, 8)`;
`GalaxyColumnUserSelector` stages `I(32)` sharded `(None, 2)` so column *c* owns
rows `8c..8c+7`, each an exact one-hot on its global slot;
`GalaxyDirectRunner._stage_positions` shards `[32]` with `(None, 0)`, shard width
8. All three agree.

### `NOT REACHED`

Every device half. Written for both models, never executed.

---

## Area 5 — Long context

`models/common/tests/models/galaxy/test_step7_long_context.py`, 32 tests.

The smokes are functional and the brief expects capacity, not numerics, to be
the limit — and asks for a record of where each one spends it. That record is
arithmetic over the resolved geometry, so it was produced and checked on the
host. Configuration mirrors
`test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_long_context_smoke`:
batch 1, one served slot, 2048-token chunks, one chunk of headroom, one sink
block per idle slot.

| Context | Served | Blocks/user | Pool | KV per device, Llama (80 layers) | KV per device, Qwen (64 layers) | RoPE tables per device | Chunks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4K | 6 144 | 192 | 223 | **0.14 GiB** | 0.12 GiB | 6 MiB | 2 |
| 32K | 34 816 | 1 088 | 1 119 | **0.73 GiB** | 0.58 GiB | 34 MiB | 16 |
| 128K | 133 120 | 4 160 | 4 191 | **2.72 GiB** | 2.17 GiB | 130 MiB | 64 |

Method, all of it checked by a test rather than asserted in prose:

* the paged pool is **replicated** — every device owns the whole block pool and
  writes only the users its column serves, which is what makes one page table
  valid on every device. So the KV figure is per device and **does not shrink
  with the mesh**;
* bfloat8_b is 1088 bytes per 32×32 tile, i.e. 1.0625 B/element;
* elements per device = `pool * (n_kv_heads / 8) * 32 * head_dim`, times 2 for
  K and V, times the layer count;
* RoPE cos/sin are replicated bf16 `[1, 1, table_len, 128]` with
  `table_len = max(2 * served, 8192)` rounded up to 128 → 266 240 at 128K;
* page tables stay sub-megabyte even at 128K (`32 * 4 168 * 4 B` ≈ 534 kB).

**Where the capacity goes at 128K, Llama:** ~2.3 GiB of weights per device
(70 B params at bfloat8_b over 32 devices) + **2.72 GiB** of KV + 0.13 GiB of
RoPE tables ≈ 5.2 GiB against a 12 GB device. It should fit; the risk is
fragmentation and the 64 sequential chunked-prefill graphs, not the total.

Also proved: each long-context geometry resolves and needs **one** prefill
recipe, not one per length; a pool one block short of the served context fails
closed with `"cannot hold max_seq_len"`; the chunked plan walks the context
without revisiting a block; and the headroom chunk really does leave the decode
after a full prefill a block to write into.

### `NOT REACHED`

The three functional smokes themselves.

---

## Repeat and cleanup

`models/common/tests/models/galaxy/test_step7_repeat_and_cleanup.py`, 12 tests.

### Repeated requests against one live model

Two identical `generate` calls produce identical tokens *and* identical plans:
same prefill and decode call counts, and the staged token rows compare equal
element for element. A repeated request rebinds nothing and restages nothing —
the KV binding happens once, at `open`.

### Runner teardown

`close` unbinds the cache, deallocates both page tables and every K/V tensor, and
is idempotent; a closed runner refuses further graph calls; reopening allocates a
genuinely fresh cache; `open` on an already-open runner is a no-op.

**A failed `open` leaves nothing bound.** Injecting a failure while staging the
*decode* table — after the cache has been allocated and bound — leaves
`bind_calls[-1] is None`, an empty `_kv_cache`, and both tables `None`. The
ordering comment in `direct_runner.open` ("Recorded before the page tables so a
staging failure still unbinds") is correct and is now pinned.

### L1, confirmed on the host

`Prefetcher2D.cleanup()` clears `self._global_cb` **without adding it to the
resources it deallocates** — ttnn exposes no free for a global circular buffer.
Measured with the module suite's injectable `create_global_cb`/`deallocate`:
after `cleanup()` the owner reports `owned_resources == ()` while the CB it
created was never handed to `deallocate`. Two owners in one process allocate two
CBs and neither is freed.

That gap is the whole of L1. The **OOM** it causes needs real L1 and was not
reproduced. The honest reading of a clean `cleanup()` is "nothing this object
still owns", not "nothing is left on the device", and the tests now say so.

**Is the ordering contract workable at model scale?** Unknown, and it should not
be guessed at. The 80-layer model has never been built, so no one has ever
observed a second Galaxy model construction in one process. `test_two_models_in_one_process`
exists in `llama33_70b_galaxy/test_bringup_wh_galaxy.py` and has never run. This
stays as Milestone C input, unchanged from `reconcile`'s O5.

---

## Regression gates

### The brief's command, before and after

```sh
python -m pytest -q models/common/tests/modules models/common/tests/models models/common/tests/llm_runtime
```

| | Before this job's changes | After |
| --- | --- | --- |
| Log | `logs/03_baseline_full_gate_BEFORE_20260827T020908Z.log.gz` | `logs/14_full_gate_AFTER_20260827T025719Z.log.gz` |
| Result | `18 failed, 1959 passed, 2059 skipped, 3276 deselected, 318 errors in 987.13s` | `18 failed, 2121 passed, 2059 skipped, 3276 deselected, 351 errors in 1048.36s` |
| Delta | — | **+162 passed** (exactly this job's host tests), **+33 errors** (exactly this job's two device files, both at cluster open), **failure set byte-identical** |

`logs/16_regression_delta_20260827T032039Z.log` holds the diff of the two
`FAILED` sets: **empty**. The three largest logs are stored gzipped.

The 18 failures are the same 18 before and after: 13 are F-C2 below, 5 are the
pre-existing 1D demo-contract set. **This job introduced no failure and fixed
none** — it changed no implementation file.

### F-C2 — `models/common/tests/models/galaxy/test_plans.py` is not a host-only suite

13 of the 18 baseline failures are in `test_plans.py`, and every one of them is
**device-induced**, not a real defect:

```text
galaxy_prefill_mode_plan -> ttnn.SubDevice([cores])
  -> SubDeviceImpl::SubDeviceImpl -> MetalContext::instance()
  -> Cluster::open_driver -> RuntimeError: Read 0xffffffff over PCIe ID 17
```

`ttnn.SubDevice` implicitly constructs the `MetalContext`, so a suite that looks
host-only — no `mesh_device` fixture, a `MagicMock` mesh, no `_wh_galaxy` in its
name — cannot run without a cluster. `mb-qwen`'s filtered host command missed
this because it does not include `models/common/tests/models/galaxy`. Worth
knowing for `mb-signoff`: on a healthy mesh these 13 should pass, and if they do
not, *that* is a finding.

The 318 errors are all `*_wh_galaxy*` device suites plus the three `moe` device
suites, all at cluster open.

### Boundaries

```sh
git diff --name-only bc6ad03bfc2..HEAD | grep '_1d\.py'      # empty
git diff --name-only bc6ad03bfc2..HEAD | grep 'llm_runtime'  # empty
```

Both empty, over all 190 changed paths. This job changed **no implementation
file at all** — only tests and evidence. That is deliberate: every defect it
found either needs a mesh to validate a fix, or needs a product decision first.

---

## Defects and gaps, collected

| ID | Severity | Where | One line |
| --- | --- | --- | --- |
| **D-C1** | correctness | `attention_2d.py::_validate_decode_page_table` | The replicated prefill page table is accepted by decode, because its device-local 32 rows are indistinguishable by shape from a 4-core L1-sharded repeat. The step-7 rejection gate cannot be met without consulting `memory_config()`. |
| **D-C2** | contract conflict | `sampling_2d.py::_seed_digest` | Moving a seeded request to another slot changes its stream, because the slot is part of the seed digest. Deliberate decorrelation; directly contradicts the step-7 slot-stability gate. Needs a product decision. |
| **G-C1** | limitation | `direct_runner.prefill_batched`, `attention_2d._recipe_identity` | Concat-32 requires all 32 slots active and exactly 1 or 32 prefill rows; the sink-block mechanism for `active_slots < 32` cannot be combined with it. |
| **G-C2** | minor | `direct_runner.prefill_batched` | An empty row plans `token_indices == -1`; rejection happens one call later, after the whole concatenated graph has run. |
| **G-C3** | dead code + missing check | `attention_2d._validate_prefill` | `"chunk_page_table requires a prefix/chunked recipe"` is unreachable, because a chunk table alone already selects `PREFIX_CHUNKED`. |
| **F-C1** | premise correction | `recipes.galaxy_padded_vocab_size` | Llama-3.3-70B has **no** vocabulary padding; its padded-vocab gate is vacuous. Only Qwen pads (128 ids). |
| **F-C2** | test-infra | `tests/models/galaxy/test_plans.py` | Looks host-only, needs a cluster: `ttnn.SubDevice` constructs the `MetalContext`. |

Inherited and untouched: **L1** (global-CB ownership — confirmed on host here),
**L3**, **D-B9**, and the five pre-existing 1D demo-contract failures.

---

## What was committed

**No implementation file changed.** New tests only:

```text
models/common/tests/models/galaxy/step7_harness.py                     (helper, not collected)
models/common/tests/models/galaxy/test_step7_paged_kv.py               39 tests
models/common/tests/models/galaxy/test_step7_concat32.py               34 tests
models/common/tests/models/galaxy/test_step7_prefix_cache.py           19 tests
models/common/tests/models/galaxy/test_step7_sampling.py               26 tests
models/common/tests/models/galaxy/test_step7_long_context.py           32 tests
models/common/tests/models/galaxy/test_step7_repeat_and_cleanup.py     12 tests
                                                                      --- 162 host tests, 3 fresh processes, identical

models/common/tests/models/llama33_70b_galaxy/test_step7_coverage_wh_galaxy.py   17 tests, NEVER EXECUTED
models/common/tests/models/qwen3_32b_galaxy/test_step7_coverage_wh_galaxy.py     16 tests, NEVER EXECUTED
```

### On committing device tests that have never run

`mb-qwen` deliberately wrote none, arguing that an unexecuted device test invites
you to trust it. That argument is right, and this job took the other side of it
for one reason: the step-7 device gaps are now *specific* — paged-vs-contiguous,
late capacity, concat-32 at 16/31/32 across five lengths, the two prefix-cache
interactions, four sampling claims — and leaving them as prose in a report means
the next person with a mesh has to re-derive them under time pressure.

The mitigation is to be loud rather than to abstain. Both files say **"This file
has never been executed"** in their module docstring, both name the date and the
reason, and both say to treat a first run as bringup rather than as a
regression. Both were verified to *collect* (17 and 16 node ids), which proves
the imports and fixtures resolve — and nothing more than that.

### The `step7_harness.py` shard-shape model

The harness reproduces one non-obvious `ttnn` fact: a distributed tensor's
`.shape` is the **shard** shape, not the global one. That was read out of
`ttnn/core/distributed/distributed_tensor.cpp` (`TensorToMesh::Impl::create_tensor`
builds the output `Tensor` from `compute_tensor_spec_for_shards`), not measured
on silicon. Every host conclusion that depends on a device-local shape — D-C1
most of all — rests on it. **First person with a mesh: check it.** One line:

```python
t = ttnn.from_torch(torch.zeros(32, 64, dtype=torch.int32), device=mesh,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 0), mesh_shape=(8, 4)),
                    dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
assert tuple(t.shape) == (8, 64)   # if this is (32, 64), D-C1 is worse than described
```

If `.shape` turns out to be the *global* shape, then the decode validator's
`shape[0] % users_per_column` check never sees 8 at all, the "device-local rows"
branch is unreachable for a correctly-mapped table, and D-C1 is not a loophole
but a total absence of validation.

---

## Two housekeeping notes

Commit produced: **`1cd451cd965`**.

**Pre-commit reformatted four test files.** `black` and `isort` rewrote
`test_step7_{long_context,paged_kv,sampling,concat32,prefix_cache,repeat_and_cleanup}.py`
and the Llama device file on the first commit attempt. The host suites were
re-run afterwards in **three more fresh processes** — `162 passed` each time,
`logs/17_step7_after_precommit_format_*.log` — and both device files re-collected
at 17 and 16 node ids. The committed content is the post-format content, and the
three-fresh-processes rule is satisfied against *it*, not only against the
pre-format text.

**Pre-commit's `trailing-whitespace` hook also rewrote eleven of the raw logs.**
It strips trailing spaces from line ends; no line was removed and no content
changed. Verified after the fact: the mesh-state log still has its 34 lines and
the device-attempt log still contains `Read 0xffffffff over PCIe ID 17`. Noted
because "never overwrite a log" is a house rule, and a hook did it rather than
this job — but the evidence is intact and this is what it looks like when it
happens.

---

# §A2 — attempt 2, on a live mesh

Written 2026-08-27/28 by `mb-coverage` **attempt 2**, unattended, at commit
`b1e824537a4` (`mb-qwen` attempt 2's tip) on
`apbernal/tttv2_wh_glx_2d_modules_milestone_b`.

Everything above this line is attempt 1's report and is left untouched. It was
written with the mesh down and is a host-only document; where it and this section
disagree about the machine, this section was measured on silicon and is later.

## The three premises of attempt 1 that are false at this tree

| Attempt 1 said | At this tree |
| --- | --- |
| "The mesh never came back … `ttnn` cannot open a cluster at all." | **Alive.** `ls /sys/class/tenstorrent \| wc -l` = 32, and `test_partition_wh_galaxy.py` opens a real 8×4 cluster: `5 passed in 12.32s` (`logs2/a2_00_mesh_health.log`). Established before planning anything. |
| "Three consecutive device jobs have produced zero numerical results from silicon, for either model." | **False.** `mb-qwen` attempt 2 (17:53–22:51 UTC, after attempt 1) qualified both models end to end: Llama 501/511 top-1, Qwen 498/511, PCC 0.999+ per block for both. Its handoff is `job2_completion_handoff_attempt2.md`. |
| "Qwen's weights are not on this machine." | **Present**, under `HF_HOME=/localdev/ctr-apbernal/hf_data` — *not* `/proj_sw/user_dev/hf_data`, which reaches Llama only. |

Attempt 1 was not wrong about what it saw at 03:00 UTC. It was superseded by a
mesh repair at ~17:00 and by a job that ran after it. This is the same failure
mode its own handoff warned about ("evidence collected at a tree that has since
moved is not evidence") applied to the *machine* rather than the tree.

## F-C1 is superseded: Llama does pad its vocabulary, by 768 ids

Attempt 1's finding F-C1 reads: *"**Llama has no vocabulary padding.** 128256 is
already a multiple of `8 * 32`. Its padded-vocab gate is vacuous; only Qwen pads
(128 ids)."* Both halves are false at this tree, and the tree already knew:

```python
>>> from models.common.models.galaxy.recipes import galaxy_padded_vocab_size
>>> galaxy_padded_vocab_size(128256), galaxy_padded_vocab_size(151936)
(129024, 153600)
```

The width is not rounded to `8 * 32`; it is rounded so that the **per-device**
width is a whole number of 24-core ring rows — `(padded // 8) % (24 * 32) == 0` —
which is the invariant D-B19 was named for. `128256 // 8 = 16032` is 501 tiles,
which no usable core count divides, so Llama pads to 129024 and carries **768**
invalid ids; Qwen pads to 153600 and carries **1664**, not 128.

`test_step7_sampling.py` was corrected for this in `60fdec0c09e` (after attempt
1's commit), so the host suite is right. What was left wrong was the *device*
coverage: `test_step7_coverage_wh_galaxy.py` for Llama said in its module
docstring that the padded-vocabulary case is "not applicable" and omitted it.
Attempt 2 added `test_llama_no_padded_vocabulary_id_is_ever_sampled` at three
policies (greedy, T=1.5, T=0.5 — never T=1.0, which is its own reciprocal and
hides D4) and corrected both files' docstrings.

**Why it matters beyond bookkeeping:** an invalid id winning is a correctness
bug, and for Llama the gate was recorded as vacuous — i.e. nobody would ever
measure it.

## The Milestone B exit gate, measured at this tree

Commit measured: `1451b192584` for runs 01/01b/02/03/g1 and `718997518ab` for every run from `g2` onward. `mb-coverage` attempt 3 established that `git diff 718997518ab..HEAD -- models/` is **empty**, so every `718997518ab` row below was produced against source identical to `af589dff4d5`; the two commits between `1451b192584` and `718997518ab` touched only the two `test_step7_coverage_wh_galaxy.py` files, which `test_full_model_wh_galaxy.py` does not import. See §A3 for the final table, which supersedes this one.

Every value below was produced by a command in this section — none is quoted from `mb-llama`,
`mb-qwen` or attempt 1. Where a number *does* agree with an earlier job's, that
agreement is stated as a result of re-measurement, which is what the brief asked
for.

| Gate line | Verdict | Measured |
| --- | --- | --- |
| Llama teacher-forced, batch 1, 512/511, top-1 ≥ 91% / top-5 ≥ 99% | **PASS**, 2 runs | top-1 **501/511 = 98.04%** (gate ≥ 91%), top-5 **511/511 = 100.00%** (gate ≥ 99%). `a2_01_llama_full_model_file.log` and `a2_g1_llama_tf.log`, character-identical |
| Qwen teacher-forced, batch 1, 512, top-1 ≥ 89% / top-5 ≥ 97% | **PASS**, 1 run | top-1 **498/511 = 97.46%** (gate ≥ 89%), top-5 **511/511 = 100.00%** (gate ≥ 97%). `a2_g12_qwen_tf.log` |
| Batch-32 direct demos valid, no cross-slot contamination | **PASS**, 1 run per model | Llama `a2_g9`, Qwen `a2_g21`: 32 slots, each answering its own prompt; Llama slot 0 character-identical to the batch-1 demo. The *test* `*_batch32_slots_are_isolated` is a different shape and FAILED for Llama on L1 (`a2_g7`), PASSED for Qwen 3/3 |
| Batch-1 4K / 32K / 128K functional smokes | **PASS**, 1 run per geometry per model | Llama 4K/32K/128K `a2_g3`/`a2_g4`/`a2_g5` (7/11/13 min); Qwen `a2_g14`/`a2_g15`/`a2_g16` (3/3/5 min). Qwen 128K exceeds its own `max_position_embeddings` (40960) and nothing enforces it: a capacity-and-plumbing result, not a quality one |
| Prefix-cached output matches uncached execution | **PASS**, 1 run per model | Llama `a2_g2`, Qwen `a2_g13`: two 128-token chunks against one 256-token prefill, same argmax and PCC ≥ 0.99 |
| No dependency imports from a model-named implementation package | **PASS** | 0 matches, over `models/common/{models/galaxy,modules,models/llama33_70b_galaxy,models/qwen3_32b_galaxy}` |
| Zero changes to 1D module implementation files | **PASS** | `git diff --name-only bc6ad03bfc2..HEAD \| grep '_1d\.py'` → 0 of 338 changed paths |
| Zero changes to `llm_runtime` | **PASS** | same diff, `grep llm_runtime` → 0 |
| Existing 1D model contract and demo-contract host tests green, expectations unchanged | **FAIL**, and not owned by Milestone B | **5 failed, 296 passed** (`a2_h1_1d_contract_gate.log`). The same five ids attempt 1 recorded. None of the five packages appears in `bc6ad03bfc2..HEAD` at all, so Milestone B cannot be their cause. Expectations unchanged — nothing was edited to accommodate this work |

## How attempt 2 ran, and what "recorded" means here

One pytest process on the mesh at a time, never piped, driven by
`cov_seq2.sh` over a manifest; each cycle reaps only the PID it started, refuses
to signal anything whose `comm` is not python, and runs `tt-smi -glx_reset`
after any non-clean exit. Logs are `logs2/a2_*.log`, one per cycle, never
overwritten. `RESULTS_A2.md` is the run-by-run index, written as each cycle
finished.

**The cost that shaped the night.** Every test in these files builds its own
model, and a Llama 80-layer build from the *warm* device weight cache is ~5.5
minutes; a *cold* recipe is far worse — `a2_01` spent 26 minutes staging 723
weights because the 512-token prefill recipe had never been resolved at this
commit. So a 17-node-id file is a three-hour run, and the house rule "three runs
in fresh processes before any device claim" cannot be applied to all 36 step-7
device cases in one night. What attempt 2 did instead, and states per row:

* the **exit-gate** lines and the **headline** step-7 mechanisms get three fresh
  processes;
* the remaining step-7 cases get **one** process, and are recorded as
  *observed, not qualified*. A single pass is bringup, and this project's own
  history says a case that passes once has proved nothing.

Nothing was recorded as evidence at a run count it did not get. Where a row says
`1 run` that is a statement about how much you may lean on it.

### One node id per process, and the 55 minutes it took to learn that

Attempt 2 began by running whole files in one process, on the reasoning that a
mesh open costs 25 s and 8 node ids in one process saves seven of them. That is
wrong on this stack, and expensively so — see **D-C3**: the device weight cache
fingerprint contains `MeshDevice.id()`, which increments per test, so test 2 of a
file re-stages all 965 weight tensors (138 GB, 26 min) and test 3 does it again.
The first cycle was stopped at 00:18 for that reason, its two completed tests
kept, and everything after it re-queued **one node id per process**. In that
shape every run is 100% cache hits.

The queue runner (`cov_queue.sh`) also grew a disk guard when this was found: it
prunes only the `.tensorbin` files this job wrote, and halts rather than
continue, if `/proj_sw` falls below 300 GB / 150 GB free.

## Which device case covers which of the brief's five areas

`L` = `models/common/tests/models/llama33_70b_galaxy/`,
`Q` = `models/common/tests/models/qwen3_32b_galaxy/`,
`G` = `models/common/tests/models/galaxy/`,
`step7` = `test_step7_coverage_wh_galaxy.py`,
`full` = `test_full_model_wh_galaxy.py`.

| Brief area | Claim it asks for | Device case |
| --- | --- | --- |
| 1 paged KV | paged fill then decode, PCC ≥ 0.99 vs contiguous | `{L,Q}/step7::*_paged_and_contiguous_caches_agree` |
| 1 | late capacity resolution | `{L,Q}/step7::*_paged_capacity_resolved_after_construction_serves_a_request` |
| 1 | transactional bind/unbind, failed bind leaves no partial state | host only (`G/test_step7_paged_kv.py`) — no device case needs one, the unwind is pure Python |
| 1 | no cross-slot contamination | `{L,Q}/step7::*_a_write_for_one_user_never_appears_in_another_users_blocks`, and `{L,Q}/demo.py::*_batch32_has_no_cross_slot_contamination` |
| 1 | a prefill-shaped table fed to decode is **rejected** | **not satisfiable at this contract** — D-C1. Pinned on the host and now on silicon: `G/test_step7_page_table_placement_wh_galaxy.py` |
| 2 concat-32 | concat-32 agrees with sequential prefill, 128 → 2048 ascending | `L/step7::*_concat32_matches_sequential_prefill_at_each_length[len128..len2048]`, `Q/…[len128..len512]` |
| 2 | padded rows change no active row's logits, active 16/31/32 | `{L,Q}/step7::*_concat32_padded_rows_change_no_active_rows_logits[active16,31,32]` |
| 3 prefix cache | prefix-cached output matches uncached | `{L,Q}/full::*_prefix_cached_prefill_matches_uncached` and `{L,Q}/step7::*_chunked_prefill_matches_a_single_uncached_prefill` (the second also decodes, so the cache the chunks *wrote* is read) |
| 3 | a prefix-cached request then a normal one | `{L,Q}/step7::*_a_prefix_cached_request_then_a_normal_one` |
| 3 | a mix of both in one batch | `L/step7::test_llama_prefix_cached_and_plain_requests_mixed_across_slots` (Llama only) |
| 4 sampling | greedy equals host argmax, every slot | `{L,Q}/step7::*_device_greedy_sampling_equals_host_argmax`, `{L,Q}/demo.py::*_device_sampling_matches_host_greedy` |
| 4 | seeded slot stability across runs | `{L,Q}/step7::*_a_seeded_slot_repeats_across_runs` |
| 4 | a padded id can never be sampled | `Q/step7::test_qwen_no_padded_vocabulary_id_is_ever_sampled`, and **new in attempt 2** `L/step7::test_llama_no_padded_vocabulary_id_is_ever_sampled` |
| 4 | per-slot heterogeneous top-k/top-p/temperature | `L/step7::test_llama_per_slot_heterogeneous_sampling_controls` (Llama only) |
| 5 long context | batch-1 4K / 32K / 128K functional smokes | `{L,Q}/full::*_long_context_smoke[4k,32k,128k]` |
| repeat/cleanup | repeated requests, deterministic | `{L,Q}/full::*_repeated_requests_and_deterministic_cleanup` |
| repeat/cleanup | two model constructions in one process | `G/test_step7_repeat_and_cleanup.py` on host; **no device case** — see L1 |

## Area by area, on silicon

Each row names the log. `runs` is how many fresh processes the claim got; a claim
with one run is *observed*, not qualified, and says so.

### Area 1 — paged KV

| Claim | Log(s) | Runs | Result |
| --- | --- | --- | --- |
| Prefill and decode page tables have the layouts D-C1 assumes | `a2_01b_page_table_placement`, `a2_s34_placement_run2`, `a2_s35_placement_run3` | **3** | **PASS.** decode global `(32, 64)` → device-local `(8, 64)`; prefill global `(32, 64)` → device-local `(32, 64)`; ratio 4; **both DRAM-interleaved**. Identical output all three runs |
| A cache bound after construction serves a request | `a2_02_llama_late_capacity` | 1 | **FAIL** on `assert all(spec.paged_attention_config is None …)`. Not a model defect — **D-C4**: `from_pretrained` substitutes the default pool for `None`. Test rewritten to the reachable claim and re-queued |
| Paged fill then decode, PCC ≥ 0.99 against the contiguous path | `a2_03_llama_paged_vs_contig` | 0 | **STOPPED at 4 min, deliberately** (`rc=143`). D-C4 makes both arms the same 2048-block pool, so the case was a tautology. Rewritten as `*_two_paged_pools_agree_and_a_contiguous_cache_is_unreachable` and re-queued. The gate line as written is **not expressible at this adaptor API** |
| No cross-slot contamination in the blocks | — | 0 | **NOT REACHED** |
| Transactional unbind, failed bind leaves no partial state | host suite only (attempt 1, 39 tests) | — | host PASS; no device case was reached |

### Area 2 — concat-32 physical prefill

| Claim | Log(s) | Runs | Result |
| --- | --- | --- | --- |
| Concat-32 prefill agrees with sequential prefill, Llama | `a2_g10_llama_demo_concat32` | 1 | **FAIL — L1 address clash, and a new detail.** `program 1552` clashes on `[0-0 - 6-9]` — the **whole 7×10 grid**, not the four sender cores of the other L1 failures. The test runs `run_direct_demo` twice, so the second prefill follows a decode |
| Concat-32 prefill agrees with sequential prefill, Qwen | `a2_g22_qwen_demo_concat32` | 1 | **FAIL, and not the Llama failure.** `Statically allocated circular buffers on core range [0-0 - 2-3] grow to 1669312 B which is beyond max L1 size of 1499136 B`, from `validate_circular_buffer_region` at `direct_runner.py:484` (`prefill_batched`). A **capacity** overflow, not an address clash. **Finding D-C6** |
| Active batches 16, 31, 32 write no KV and return no logits for inactive slots | — | 0 | **NOT REACHED** |
| Lengths 128 → 2048 in the padded lengths the policy supports | — | 0 | **NOT REACHED** on device. The host recipe suite covers all five Llama lengths |

### Area 3 — prefix-cached and chunked prefill

| Claim | Log(s) | Runs | Result |
| --- | --- | --- | --- |
| Prefix-cached prefill matches uncached, Llama | `a2_g2_llama_prefix` | 1 | **PASS** — two 128-token chunks vs one 256-token prefill, same argmax and PCC ≥ 0.99 |
| Prefix-cached prefill matches uncached, Qwen | `a2_g13_qwen_prefix` | 1 | **PASS** |
| Chunked prefill matches a single uncached prefill | — | 0 | **NOT REACHED** |
| A prefix-cached request then a normal one | — | 0 | **NOT REACHED** |
| A mix of both in one batch | — | 0 | **NOT REACHED** (and the Qwen test did not exist; attempt 3 wrote it) |

### Area 4 — device sampling

| Claim | Log(s) | Runs | Result |
| --- | --- | --- | --- |
| Device greedy sampling equals host argmax, Llama, through the demo | `a2_g11_llama_demo_sampling` | 1 | **FAIL — L1, `program 100`.** The demo runs twice (host policy, then device policy), so the second prefill follows a decode and never reaches the sampler. The claim itself is untested by this log |
| Device greedy sampling equals host argmax, Qwen, through the demo | `a2_g23_qwen_demo_sampling` | 1 | **FAIL, and not L1 at all.** `MatmulMultiCoreProgramConfig: Input B memory layout must be INTERLEAVED, got: TensorMemoryLayout::WIDTH_SHARDED` at `collectives.py:445`, `GalaxyColumnUserSelector.__call__`, reached from `model.sample_decode` → `select_decode_column_users`. The host-sampling half ran first and passed. **Finding D-C5** |
| Seeded slot stability, padded vocabulary, near-zero temperature (D4), per-slot heterogeneous controls | — | 0 | **NOT REACHED.** All four cases were written (the padded-vocabulary and temperature cases *by* attempt 2) and queued, and the host was withdrawn before they ran |

### Area 5 — long context

| Geometry | Llama | Qwen |
| --- | --- | --- |
| 4K | **PASS** (`a2_g3`, ~7 min, 2 chunks of 2048) | **PASS** (`a2_g14`, ~3 min) |
| 32K | **PASS** (`a2_g4`, ~11 min, 16 chunks) | **PASS** (`a2_g15`, ~3 min) |
| 128K | **PASS** (`a2_g5`, ~13 min, 64 chunks, then a decode at position 131072) | **PASS** (`a2_g16`, ~5 min) |

One run each. Attempt 1's accounting predicted ~5.2 GiB per device for Llama at
128K against 12 GB and named fragmentation as the risk; it fits. **Qwen3-32B's
`max_position_embeddings` is 40960**, so its 128K smoke runs three times past the
trained context and nothing in the stack refuses it — `max_context_len` is carried
on the runtime config and never checked against `max_seq_len`. Functional, as the
brief defines it; not a quality statement.

Attempt 1's capacity accounting for these three geometries (blocks per user,
pool size, KV bytes per device, RoPE table size, chunk count) is in area 5 above
this section and was not re-derived; what attempt 2 adds is whether each one
actually runs.

### Repeat and cleanup

| Shape | Llama | Qwen |
| --- | --- | --- |
| `*_repeated_requests_and_deterministic_cleanup` — the same request twice through two runners on one live model | **FAIL 2/2**, deterministic (`a2_g6`, `a2_L1_llama_repeat_run2`): `program 100` clashes on `[0-0 - 0-3]`, L1 buffer at 544832, static CB region ends at 630080 | **PASS 3/3** (`a2_g17`, `a2_L1_qwen_repeat_run2/3`) |
| `*_batch32_slots_are_isolated` — slot 0 alone, then slot 0 inside a full batch | **FAIL 1/1**, same signature (`a2_g7`) | **PASS 3/3** (`a2_g18`, `a2_L1_qwen_batch32_run2/3`) |
| Repeated model construction and teardown in one process (`test_two_models_in_one_process`) | **NOT REACHED** | not applicable — the bringup file is Llama-only |

The two Qwen run-3 logs (`a2_L1_qwen_repeat_run3`, `a2_L1_qwen_batch32_run3`,
both `exit=0`) landed after `RESULTS_A2.md`'s last row was written and are
recorded here for the first time; attempt 3 re-read them off disk to confirm it.
`a2_L1_llama_repeat_run3` was in flight when the host went away and has no
verdict.

## L1, and why four step-7 cases cannot be measured behind it

`mb-llama` attempt 3 named the shape of Milestone A limitation **L1** precisely:
`Prefetcher2D` allocates a `global_circular_buffer` on `activate("decode")`,
there is no `deallocate` for that type, and a *prefill* program afterwards cannot
place its circular buffers on the four sender cores the CB still occupies:

```text
TT_THROW ... Statically allocated circular buffers in program 100 clash with L1
             buffers on core range [0-0 - 0-3]
```

So **prefill-before-any-decode is fine; prefill-after-a-decode is not**, in one
process. Attempt 3 implemented the obvious fix
(`Prefetcher2DConfig.release_global_cb_on_prefill`) and *refuted it on hardware* —
the L1 base address is identical with the flag on, because dropping the last
Python reference does not return the L1.

Every step-7 case whose shape is *(prefill, decode) then (prefill, …)* in one
process inherits that, and there are five of them:

| Case | Why it has two phases |
| --- | --- |
| `{L,Q}/full::*_batch32_slots_are_isolated` | slot 0 alone, then all 32, and each `generate` decodes |
| `{L,Q}/full::*_repeated_requests_and_deterministic_cleanup` | the repeat *is* the second phase |
| `{L,Q}/step7::*_paged_and_contiguous_caches_agree` | two models, each prefilling and decoding |
| `{L,Q}/step7::*_a_write_for_one_user_never_appears_in_another_users_blocks` | two runners, decode after each |
| `{L,Q}/step7::*_a_seeded_slot_repeats_across_runs` | three runners, decode in each |
| `{L,Q}/step7::*_chunked_prefill_matches_a_single_uncached_prefill` | uncached then cached, decode after each |

**This is not a reason to restructure them.** The two-phase shape is the *claim*:
"a repeated identical request produces the same tokens" is not testable in one
phase, and neither is "slot 0's continuation does not depend on the other 31".
A single-phase rewrite would pass while proving nothing, which is the failure
mode this project distrusts most. They are recorded against L1 with their logs,
and they are the concrete cost of L1 to Milestone B's step-7 gate — which is
worth more to `mb-signoff` than a green tick would be.

The one open hypothesis, from attempt 3 and untried: confine the prefill mode
plan to the **worker** cores (`galaxy_prefill_mode_plan_cores` currently returns
the whole compute grid) so no prefill program can be placed on the sender
columns at all. Attempt 2 did not try it — it changes the grid of every prefill
program, so prefill 128, prefill 2048, the 80-layer prefill and both accuracy
gates all have to be re-taken behind it, and this job's own gate evidence would
have gone with it. One fact attempt 2 can add: the prefill matmuls are *already*
worker-confined (`dense_matmul_program_config` sets `allowed_worker_cores`), so
whatever program 100 is, it is not one of those two — narrowing the search to the
collectives and the MLP ring form.

## Findings, attempt 2

Attempt 1's seven (D-C1, D-C2, G-C1, G-C2, G-C3, F-C1, F-C2) plus what a live
mesh added. Only the changes are written out here; the unchanged ones keep
attempt 1's text above.

### F-C1 — **superseded, and it was the wrong way round**

See §A2's opening. Llama pads by 768 ids, Qwen by 1664. Attempt 1 recorded
Llama's padded-vocabulary gate as *vacuous*; it is live, and now has a device
case (`test_llama_no_padded_vocabulary_id_is_ever_sampled`, three policies).

### D-C1 — premise confirmed on silicon, verdict unchanged

Attempt 1 derived D-C1 from a host model of one `ttnn` fact and asked for one
line on a live mesh to settle it. That line is now a committed test,
`models/common/tests/models/galaxy/test_step7_page_table_placement_wh_galaxy.py`,
and it says attempt 1 read the fact correctly:

* a column-sharded decode table (`ShardTensor2dMesh(dims=(None, 0))`, mesh
  `(8, 4)`) has device-local shape **(8, 64)** — the shard shape, one mesh
  column's users;
* the replicated prefill table has device-local shape **(32, 64)**, and
  `32 % 8 == 0`.

So `_validate_decode_page_table`, which discriminates on the device-local row
count alone and accepts any positive multiple of `users_per_column`, cannot tell
the prefill layout from a legitimate four-core L1 repeat. **D-C1 stands exactly
as attempt 1 wrote it, and the worse variant it feared is ruled out.**

The test also records what attempt 1 could not check: **both tables are
DRAM-interleaved**, so `memory_config().is_sharded()` is false for both. A fix
therefore cannot be "reject unless sharded" applied to the 32-row case alone
without also deciding what a 32-row *interleaved* table means; the honest
discriminator is that a repeat is only legitimate when the tensor is L1
height-sharded over exactly `rows / users_per_column` cores, which makes the
existing 2D-module expectation
`test_decode_page_table_accepts_the_device_local_batch_and_its_core_repeats[16]`
and `[32]` — which pass a plain interleaved table — the thing that has to change.
That is the boundary attempt 1 declined to cross, and attempt 2 declines it for
the same reason: the brief says report it, do not edit the expectation.

### D-C2 — unchanged, and still a product decision

`_device_seed`/`_host_seed` are `blake2b("sampling2d:{seed}:{slot}")`, so a
request that migrates slots does not keep its stream. The step-7 gate asks for
the opposite. Attempt 2 measured only the half that holds — same seed, same slot,
same token across fresh runs — and did not assert the half that does not.

### D-C3 — the device weight cache is keyed by `MeshDevice.id()`, so every test after the first in a process re-stages every weight

**New, severity: test-infrastructure, and it costs hours and hundreds of GB.**

`LazyWeight._get_fingerprint` ends with

```python
device_id = self.device.id() if hasattr(self.device, "id") else "single"
parts.append(f"device_{device_id}")
```

`self.device` is the **`MeshDevice`**, and the `mesh_device` fixture builds a new
one per test, so its `.id()` is 0 for the first test in a pytest process, 1 for
the second, 2 for the third. The cache path therefore changes per test, and every
test after the first misses on **every** weight.

Measured, on this mesh, at this commit:

| | |
| --- | --- |
| whole-file run, `test_full_model_wh_galaxy.py` (8 node ids) | test 1: 240 cache hits, model built in ~6 min. Test 2: **965 misses**, 26 min of staging, **138 GB** written. Test 3: staging device_2's set again |
| the same test alone in its own process | **240 hits, 0 misses**, whole test 237 s |

A complete cache set is 138.5 GB for Llama-3.3-70B, so an 8-node-id file needs
**1.1 TB** of cache to run — on a filesystem that started this night with 1.0 TB
free and 95% used. This attempt paid 55 minutes and 277 GB of it before reading
the fingerprint, and then pruned the two duplicate sets.

**Consequence for anyone scheduling this hardware: one node id per pytest
process, always.** Every earlier job's harness happens to do that — `mb-qwen`'s
manifest format is one node id per line — but nothing in the tree says why, and
the cost of not knowing is a whole night.

The fix is a one-line change in shared 1D/2D code (`models/common/modules/lazy_weight.py`),
which is outside this job's mandate: a mesh of the same shape and mapper produces
the same tensor, so the fingerprint wants the mesh **shape**, not the instance id.
Reported, not changed.

### D-C4 — `from_pretrained` cannot build a contiguous KV cache, so area 1's headline gate is not expressible through the adaptor

**New, severity: contract gap. It also made one committed test a tautology.**

Both adaptors do

```python
paged = paged_attention_config or default_paged_attention_config(params)
```

so `paged_attention_config=None` does not mean "contiguous" - it means "give me
the default pool", `ceil(max_seq_len / 32) * max_batch_size` blocks. There is no
argument that yields `spec.paged_attention_config is None`, even though
`Attention2D`, `GalaxyPagedKVContract` and the model's own `kv_specs` all support
that state and the host suite exercises it.

Two consequences, both measured:

1. `test_*_paged_capacity_resolved_after_construction_serves_a_request` **failed**
   on `assert all(spec.paged_attention_config is None ...)` (`a2_02`). That is a
   true report of the gap, not a broken model.
2. `test_*_paged_and_contiguous_caches_agree` compared the default pool against
   an explicitly-constructed pool of **exactly the same geometry** - at
   `max_seq_len=2048`, batch 32, block 32, both are 2048 blocks. It would have
   passed at PCC 1.0 while proving nothing about paged addressing.

Attempt 2 rewrote both rather than leaving a green tautology:

* `test_*_two_paged_pools_agree_and_a_contiguous_cache_is_unreachable` runs the
  same 32 requests through a 2048-block and a 4096-block pool - which gives every
  slot a different run of block ids - and compares prefill and decode logits per
  slot at PCC ≥ 0.99. It asserts `resolved is not None` with a message telling a
  future reader to restore the original comparison once D-C4 is fixed;
* the late-capacity case now asserts the *reachable* claim: the geometry
  installed at construction can still be replaced before anything is bound, is
  refused while bound, and can be replaced again after unbind.

**The gate line "paged fill during prefill, then decode reading the same blocks,
PCC ≥ 0.99 against the contiguous path" therefore cannot be met at this API**, and
that is the honest verdict rather than a green tick from a tautology.

**Where the contiguous path does exist**, for whoever fixes D-C4:
`models/common/tests/models/llama33_70b_galaxy/test_bringup_wh_galaxy.py` builds
one with `_contiguous_kv_cache(...)` and `model.set_kv_cache(...)` directly, and
`GalaxyDirectRunner` has a contiguous branch (`self.paged = False`, which then
requires `active_slots == max_batch_size`). So the missing piece is only an
adaptor argument — something like `paged=False` alongside
`paged_attention_config` — not a new mechanism.

### D-C5 — the column user selector cannot accept Qwen's decode logits: its matmul requires an INTERLEAVED input B

**New, severity: correctness-blocking for device sampling on Qwen.** Added by
`mb-coverage` attempt 3 from attempt 2's `a2_g23_qwen_demo_sampling.log`; attempt
2 measured it and was cut off before writing it up.

`GalaxyColumnUserSelector.__call__` (`models/common/models/galaxy/collectives.py:445`)
is a single `ttnn.matmul(self.selector(), tensor, …)`: an identity-matrix selector
against the decode logits. `ttnn.matmul` with the default (multi-core) program
config requires **input B interleaved**
(`matmul_device_operation.cpp:1233`), and Qwen's decode logits arrive
**WIDTH_SHARDED**:

```
TT_FATAL: MatmulMultiCoreProgramConfig: Input B memory layout must be INTERLEAVED,
          got: TensorMemoryLayout::WIDTH_SHARDED
  models/common/models/qwen3_32b_galaxy/model.py:1810  in sample_decode
  models/common/models/qwen3_32b_galaxy/model.py:1793  in select_decode_column_users
  models/common/models/galaxy/collectives.py:445       in __call__
```

The selector's own `memory_config` default is `DRAM_MEMORY_CONFIG`, so the
constraint is on the *incoming* tensor, which the selector neither checks nor
converts. Its only guard is a shape check (`[1, 1, max_batch_size, W]`); memory
layout is unvalidated, so the failure surfaces as a `TT_FATAL` from inside
`ttnn` rather than as a contract error naming the caller.

Two things make this a *2D-module* finding rather than a Qwen one:

* the selector is shared Galaxy code (`collectives.py`), not model code, and its
  contract is silent about the layout it accepts;
* it is reached only through `model.sample_decode`, so **every** device-sampling
  claim for Qwen is behind it — greedy-vs-host-argmax, the padded vocabulary, the
  seeded slots and the heterogeneous controls alike.

The host-sampling half of the same test ran first and passed, which localises the
fault to the device path.

**What it needs**, for whoever owns `collectives.py`: either the selector accepts
a sharded input B (an `interleaved_to_sharded`/`sharded_to_interleaved` at the
boundary, or a matmul program config that takes it), or `sample_decode` states the
layout it requires and the model converts before the call. Both are runtime
changes, so attempt 3 reports rather than makes them.

### D-C6 — Qwen's concat-32 prefill program does not fit in L1 at all

**New, severity: limitation, and it is a capacity result rather than an
ownership one.** Added by attempt 3 from attempt 2's
`a2_g22_qwen_demo_concat32.log`.

```
TT_THROW: Statically allocated circular buffers on core range [0-0 - 2-3]
          grow to 1669312 B which is beyond max L1 size of 1499136 B
  tt::tt_metal::detail::ProgramImpl::validate_circular_buffer_region
  models/common/models/galaxy/direct_runner.py:484  in prefill_batched
  models/common/models/galaxy/direct_demo.py:69     in run_direct_demo
```

This is **not** the L1 limitation L1/G-C\* family. Those are address collisions —
"static circular buffers … *clash with* L1 buffers … L1 buffer allocated at
544832" — which depend on what a previous phase left allocated. This one is the
sum of the program's own static circular buffers exceeding the whole 1499136 B of
L1 on a 3×4 core range, **by 170176 B (11%)**, which is a property of the resolved
concat-32 prefill recipe alone and cannot be fixed by teardown ordering.

The distinction matters for scheduling Milestone C: L1's ownership redesign will
not make this case pass. Qwen's concat-32 prefill needs a smaller resolved recipe
(fewer or smaller CBs, or a narrower core range per stream) before it can run at
all, at any length.

**One thing this leaves open**, and attempt 3 queued it: the failure was observed
in the *second* `run_direct_demo` of the demo test, so it has not yet been
separated from "after a decode". If `test_qwen_concat32_matches_sequential_prefill_at_each_length`
fails the same way in a fresh model with no preceding decode, the finding is
unconditional; if it passes, the capacity overflow is history-dependent after all
and D-C6 collapses into the L1 family.

### L1's **address clash** is Llama-specific at this tree; Qwen fails the same two demo shapes for two unrelated reasons

**New, and it contradicts an inherited claim.** `mb-qwen` attempt 2's handoff
says of L1's remaining half — prefill after a decode — *"Untouched, inherited,
**identical for both models**."* Measured here, it is not:

| Test shape (two prefill phases with a decode between them) | Llama | Qwen |
| --- | --- | --- |
| `*_repeated_requests_and_deterministic_cleanup` | **FAIL**, `program 100` clashes on `[0-0 - 0-3]` (`a2_g6`) | **PASS**, no clash (`a2_g17`) |
| `*_batch32_slots_are_isolated` | **FAIL**, same signature (`a2_g7`) | **PASS**, no clash (`a2_g18`) |
| `demo.py::*_concat32_prefill_matches_sequential` | **FAIL**, `program 1552` clashes on `[0-0 - 6-9]` — the whole grid (`a2_g10`) | **FAIL**, but *not* an address clash: static CBs on `[0-0 - 2-3]` **grow to 1669312 B against a 1499136 B L1** (`a2_g22`) — a capacity overflow, **D-C6** |
| `demo.py::*_device_sampling_matches_host_greedy` | **FAIL**, `program 100` (`a2_g11`) | **FAIL**, and not L1 at all: the column user selector matmul refuses a WIDTH_SHARDED input B (`a2_g23`) — **D-C5** |

Both Qwen results were taken in fresh single-node-id processes and re-run to
three (`a2_L1_qwen_*_run2/3`); the Llama failures are four independent
reproductions in four different tests.

**Read the last two rows before the first two.** This section was written at
02:49 UTC against the first two rows only, when the heading said *"L1's remaining
half is Llama-specific"*. The Qwen cells then completed, and both are failures —
so the claim as first written is too strong and attempt 3 narrowed it, heading
included. What survives is precise and still useful:

* the **L1 address clash** — the `clash with L1 buffers on core range …, L1 buffer
  allocated at 544832` signature — is **Llama-only at this tree**: 4 reproductions
  in 4 Llama tests, 0 in 6 Qwen runs of the two shapes that reproduce it for
  Llama;
* but **Qwen is not clean on the two demo shapes**. It fails
  `*_concat32_prefill_matches_sequential` on an L1 **capacity** overflow (D-C6) and
  `*_device_sampling_matches_host_greedy` on a **matmul layout contract** (D-C5).
  Neither is an address collision, neither depends on a preceding decode as far as
  the evidence goes, and neither would be fixed by the teardown-ordering work L1
  points at.

So the honest one-line version is: *the address clash is a property of Llama's
resolved geometry, and the two-prefill-phase demo shapes are unreliable on both
models for three distinct reasons.* Qwen is still the differential reference L1
needs — it runs the two `*_repeated_requests*` / `*_batch32_slots*` shapes clean
3/3 — but it is not a clean bill of health for the concat-32 or sampling paths.

**Why this matters more than a green tick.** The clash is an address collision —
`L1 buffer allocated at 544832 and static circular buffer region ends at …` — and
Qwen's decode placements are narrower than Llama's (residual on 10 cores against
16, `local_dim` 1280 against 2048, and a 40-core LM-head reduction against 42).
So the failure is a function of *how much L1 the decode mode leaves below the
prefill program's static CB region*, not of the mechanism being absent. That
gives Milestone C something it did not have: **a working reference configuration
on the same silicon**, which turns "why does prefill-after-decode clash" from a
one-sided debugging problem into a differential one.

It also means the limitation cannot be stated as a property of the 2D modules. It
is a property of a *resolved geometry*, and the next model added to this stack may
land on either side of it with nothing in the contract to warn it.
### The command behind each exit-gate line

All of them under `HF_HOME=/localdev/ctr-apbernal/hf_data`, one pytest process at
a time, through `cov_run3.sh`:

```sh
L=models/common/tests/models/llama33_70b_galaxy
Q=models/common/tests/models/qwen3_32b_galaxy

# Llama teacher-forced, batch 1, prefill 512 / decode 511
$L/test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_teacher_forced_accuracy_batch1
# Qwen teacher-forced, batch 1, 512
$Q/test_full_model_wh_galaxy.py::test_qwen3_32b_galaxy_teacher_forced_accuracy_batch1
# batch-32 direct demos, no cross-slot contamination
models/common/models/llama33_70b_galaxy/demo.py::test_llama33_70b_galaxy_direct_demo_batch32_has_no_cross_slot_contamination
models/common/models/qwen3_32b_galaxy/demo.py::test_qwen3_32b_galaxy_direct_demo_batch32_has_no_cross_slot_contamination
# batch-1 4K / 32K / 128K functional smokes
$L/test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_long_context_smoke   # 4k, 32k, 128k
$Q/test_full_model_wh_galaxy.py::test_qwen3_32b_galaxy_long_context_smoke     # 4k, 32k, 128k
# prefix-cached output matches uncached execution
$L/test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_prefix_cached_prefill_matches_uncached
$Q/test_full_model_wh_galaxy.py::test_qwen3_32b_galaxy_prefix_cached_prefill_matches_uncached
$L/test_step7_coverage_wh_galaxy.py -k chunked_prefill_matches      # and the decode after it
$Q/test_step7_coverage_wh_galaxy.py -k chunked_prefill_matches
```

Host, device-free:

```sh
# no dependency imports from a model-named implementation package
grep -rnE '^\s*(from|import)\s+models\.(demos\.llama3_70b_galaxy|common\.models\.(llama33_70b|qwen3_32b)([^_]|$))' \
    models/common/models/galaxy models/common/modules models/common/models/*_galaxy
# zero changes to 1D module implementation files, and to llm_runtime
git diff --name-only bc6ad03bfc2..HEAD | grep '_1d\.py'
git diff --name-only bc6ad03bfc2..HEAD | grep 'llm_runtime'
# existing 1D model contract and demo-contract host tests
bash tttv2_milestone_b_evidence/coverage/cov_1d_contract_gate.sh <log>
```

## What attempt 2 committed

Tests, evidence and two docstring corrections. **No implementation file, in any
package.** Both boundary greps stay empty and the model-named import gate stays
at zero.

```text
models/common/tests/models/galaxy/test_step7_page_table_placement_wh_galaxy.py   new, 3 device cases
models/common/tests/models/llama33_70b_galaxy/test_step7_coverage_wh_galaxy.py   +1 case (x3 policies), docstring
models/common/tests/models/qwen3_32b_galaxy/test_step7_coverage_wh_galaxy.py     docstring, `_distinct_rows` fallback
tttv2_milestone_b_evidence/coverage/                                            logs2/, RESULTS_A2.md, this section
```

Three test-level changes, and the reason for each:

1. **`test_llama_no_padded_vocabulary_id_is_ever_sampled`** — the case F-C1 said
   was vacuous. It is not; Llama pads 768 ids.
2. **`test_step7_page_table_placement_wh_galaxy.py`** — the one host assumption
   attempt 1 flagged as needing a mesh, as a test rather than a one-off script,
   because D-C1's write-up depends on it.
3. **`_distinct_rows` cyclic fallback** — the reference file holds 1024 tokens, so
   the straight window walk *skipped* every concat-32 length ≥ 1024, which are
   exactly the lengths the brief asks for last. A skip is not a result. The
   exact-window path is untouched, so results taken before the change are
   comparable.

None of these relaxes a threshold, a tolerance or a parametrization; (3) widens
one.

## What Milestone C inherits from this job

* **L1's remaining half — prefill after a decode — is now costed.** Five step-7
  cases cannot be measured behind it, and the list is in §A2's L1 section with
  the one untried hypothesis (confine the prefill mode plan to worker cores) and
  the one new fact that narrows it (the prefill matmuls are already
  worker-confined, so the clashing program is a collective or the MLP ring form).
* **D-C1** — decode's page-table validator cannot separate the prefill layout
  from a legitimate L1 repeat, and the premise is now confirmed on silicon. The
  fix requires changing a 2D-module expectation, which two attempts have now
  declined as a boundary violation. It needs a decision, not a patch.
* **D-C2** — is a sampling seed per-request or per-(request, slot)? A product
  decision about the serving contract.
* **G-C1, G-C2, G-C3, F-C2** — unchanged from attempt 1.
* **The device weight cache is unbounded.** Staging Llama's full interleaved and
  ring weight sets at this commit wrote **138 GB** in 26 minutes, on a filesystem
  with 1.0 TB free and 95% used. A step-7 sweep that resolves many recipes is a
  disk-capacity question as much as a device-time one.
