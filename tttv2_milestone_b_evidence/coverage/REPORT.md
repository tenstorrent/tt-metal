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
