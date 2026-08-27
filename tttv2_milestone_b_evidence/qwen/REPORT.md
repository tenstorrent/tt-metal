# `mb-qwen` — Milestone B steps 4–6, Qwen3-32B on WH Galaxy `(8, 4)`

**Verdict: `BLOCKED (infra)`. The finish condition was not met and no device
result of any kind is claimed.**

Written 2026-08-27 by the `mb-qwen` job, unattended.
Environment and mesh facts: [`ENVIRONMENT.md`](ENVIRONMENT.md).
Raw logs: [`logs/`](logs/). Two of them — the full-gate runs `20_…` (2.4 MB) and
`21_…` (690 KB) — are committed **gzipped**, because the repo's pre-commit hook
rejects files over 500 KB. Nothing was truncated: `zcat` reproduces each byte for
byte, verified before committing. The uncompressed originals are left in place on
disk.

Commit produced: `768c5ca2771`.

---

## 1. Headline

Not one device test ran. Two independent blockers, either of which alone would
have stopped the device work:

1. **The mesh is down.** Eleven of the 32 boards (`0–7, 10, 11, 14`) are absent
   from the PCIe bus. Both permitted recovery attempts were spent and both
   failed. A `(8, 4)` Galaxy needs all 32.
2. **Qwen3-32B's weights are not on this machine.** The HF cache entry is
   **config-only** (12 KB, `config.json`); the shared cache
   `/proj_sw/user_dev/hf_data` has no Qwen3-32B at all.

So the block, the full model, the demo and the teacher-forced accuracy gate
could not be attempted. **No PCC number, no accuracy number, no demo output
exists for Qwen.** Sections 5 and 6 say exactly what remains.

What this job did instead was take the entire host half of the brief as far as
it goes — which turned out to be further than expected, because
`transformers` 5.12.1 ships the whole Qwen3 stack, so an independent HF
reference was available **without the checkpoint**. The 64-head geometry, which
the brief calls "the headline risk" and which had *no evidence of any kind*, is
now qualified on host and still unqualified on silicon.

---

## 2. Results table

| # | What | Result | Runs | Evidence |
| --- | --- | --- | --- | --- |
| 1 | Mesh health | **BLOCKED (infra)** — 21/32 boards enumerated | — | `logs/00`, `logs/03` |
| 2 | Recovery attempt 1 — `tt-smi -glx_reset` | **failed** (`Errno 6` on `/dev/tenstorrent/7`) | 1 | `logs/01` |
| 3 | Recovery attempt 2 — `tt-smi -r` | **failed** (`Read 0xffffffff` PCIe ID 17) | 1 | `logs/02` |
| 4 | Qwen3-32B weights present? | **NO** — config-only cache entry | — | §5.2 |
| 5 | Host adaptor qualification (13 tests) | **PASS** | **3 fresh** | `logs/11_…run{1,2,3}` |
| 6 | Qwen host suites combined (45 tests) | **PASS** | **3 fresh** | `logs/13_…run{1,2,3}` |
| 7 | Pure-host regression gate (410 tests) | **PASS**, 0 failed | 1 | `logs/21` |
| 8 | Brief's regression gate as written | 397 passed, **289 errors**, 0 failed | 1 | `logs/20` |
| 9 | Boundary gates (`_1d.py`, `llm_runtime`, no Llama import) | **PASS** (all empty) | 1 | §7 |
| 10 | One Qwen block, decode/prefill PCC ≥ 0.99 | **NOT RUN** (no mesh) | 0 | — |
| 11 | Full model + demo | **NOT RUN** (no mesh, no weights) | 0 | — |
| 12 | Teacher-forced accuracy gate | **NOT RUN** (no mesh, no weights) | 0 | — |

---

## 3. The 64-head geometry verdict — stated explicitly

The brief requires this verdict in plain terms, so here it is in two parts.

### 3.1 On host: QUALIFIED

The real decoupled geometry is correct in this tree and is now covered by tests
that fail if it regresses.

Derived from production code (`GalaxyDenseGeometry`), not restated:

```text
dim                        5120
attention_dim (64 x 128)   8192      decoupled: True
local_dim   (dim / 4 cols) 1280
local_attention_dim (/8)   1024      != local_dim
row_dim     (dim / 8 rows)  640
qkv_size                  10240
local_qkv_size             1280      == local_dim  (see the trap below)
local_heads / local_kv      8 / 1
local_hidden_dim           3200
```

Every item on the brief's risk-1 checklist:

| Check | Verdict |
| --- | --- |
| WO input width is `attention_dim / GALAXY_ROWS` (1024), not `dim / GALAXY_ROWS` | **correct** — asserted in `test_converted_attention_weights_reproduce_the_hf_attention_output` and `test_wo_weight_placement_is_paired_with_attention_dim_not_dim` |
| head-concat output width matches what WO expects | **correct** — the concat is asserted to be `attention_dim`-wide before `wo` |
| residual added after WO is `dim`-wide, not `attention_dim`-wide | **correct** — asserted on `wo`'s output width |
| `wo` gets `dram_sharded_weight_memory_config(mesh, local_attention_dim, local_dim)` (the D5 pairing) | **correct** — `model.py:483`, pinned by test |

And the end-to-end numerical claim: attention rebuilt **from the converted
tensors alone** — `wqkv`, `wo`, `q_norm`, `k_norm`, and the Meta RoPE tables —
reproduces the unmodified HF `Qwen3Attention` at **PCC ≥ 0.9999**, on a fixture
with the real decoupled shape character (`attention_dim / dim == 1.6`, exactly
the product's ratio) rather than Milestone A's square 40-head fixture.

### 3.2 On silicon: STILL UNQUALIFIED

Nothing changed here. Job 0's O4 stands. The brief was right that this needed to
be treated as unqualified rather than as a re-run — and it still does, because
the mesh never came up.

### 3.3 The trap worth carrying forward

`local_qkv_size` and `local_dim` are **both 1280** for Qwen3-32B. A confusion
between the fused-QKV width and the residual width is therefore
**shape-invisible** on this model — it produces exactly the right shape and the
wrong numbers. `local_attention_dim` (1024) is the one that differs and so the
one shape checks *can* catch. This is documented at the top of the new test file
because it is the failure mode most likely to survive a shape-only review.

---

## 4. The Q/K norm result

The brief asked for Q/K norm to be validated independently, with its own
geometry and its own PCC against the HF reference, *before* being enabled inside
the block. The host half of that is done; the device half was not reachable.

**Host: PASS.** Two separate claims, deliberately split:

1. **The relayout is the right permutation.**
   `test_qk_norm_relayout_is_the_same_permutation_as_the_projection_relayout`
   proves `reverse_permute_1d` (used for the `head_dim`-wide norm vectors) is
   *the same permutation* `reverse_permute` applies to the Q/K projection rows,
   and independently that it is the interleave of the two HF halves.
2. **The norm algebra composes.**
   `test_head_local_rms_norm_in_meta_layout_reproduces_hf_qk_norm` reproduces
   HF's `Qwen3RMSNorm` at PCC ≥ 0.9999 for both `q_norm` and `k_norm`.

A note on rigour, because it is easy to get this wrong: claim 2 alone is *not*
evidence about the adaptor. `norm(Px, Pw) == P norm(x, w)` holds for **any**
permutation `P`, so if the test derived `P` from the function under test it
would be a tautology. The first version of this test did exactly that. It was
rewritten to state the permutation from first principles (`_meta_permutation`),
so it now fails if the adaptor's relayout is wrong. The genuinely non-circular
evidence is the end-to-end attention test in §3.1, which consumes `q_norm` and
`k_norm` straight from the adaptor and compares against unmodified HF.

**Device: NOT RUN.** D2's own defect was that head-local decode aborted in op
validation before producing any number, so there is still no Qwen Q/K-norm
number from silicon anywhere. `test_model_host.py` already pins both sides of
the `HEAD_LOCAL` contract (job 0's `test_head_local_qk_norm_agrees_with_the_module_default_by_contract`);
Qwen's config passes interleaved DRAM explicitly and agrees with the post-D2
default. **The composed path has still never run.**

---

## 5. The ring-width finding

**The brief's account is correct, and the reason is exact divisibility.** Derived
rather than asserted, and now covered by
`test_decode_ring_widths_differ_from_llama_by_exact_divisibility`:

```text
RING_ALIGNMENT = TILE * RING_CORE_COUNT = 32 * 24 = 768

Llama-3.3-70B  local_hidden=3584  padded=3840  shard=160  3584 % 160 = 64
               -> scatters the PADDED width   -> resource key 960, placement 960
Qwen3-32B      local_hidden=3200  padded=3840  shard=160  3200 % 160 =  0
               -> scatters the LOGICAL width  -> resource key 800, placement 960
```

So Qwen's scattered W1/W3 *placement* is padded to 960 columns exactly as
Llama's is, while its resource *key* is 800. The divergence is an arithmetic
consequence of `decode_reduce_scatter_width`'s `scattered = padded if
local_hidden_dim % shard else local_hidden_dim`, not a defect. Llama's two
values coincide only because 3584 is not a multiple of 160.

**Still device-unverified.** If a Qwen decode all-gather cannot find its
resource, this pair is still the first thing to inspect — the brief's advice
stands, and it is now backed by the arithmetic rather than by a source reading.

---

## 6. Risk 4 — fused QKV bias: RESOLVED, no contract change needed

The brief said to stop and report if the resolved checkpoint carries QKV bias
tensors. It does not. Read from the real `config.json` in the local HF cache:

```json
"attention_bias": false,
"hidden_size": 5120, "num_attention_heads": 64, "head_dim": 128,
"num_key_value_heads": 8, "intermediate_size": 25600,
"rope_theta": 1000000, "rope_scaling": null, "vocab_size": 151936
```

`test_real_checkpoint_config_matches_the_contract_and_declares_no_qkv_bias`
asserts `attention_bias is False` and checks every field of
`QWEN3_32B_CHECKPOINT_CONTRACT` against the checkpoint. It **runs** (it does not
skip) because `config.json` is cached even though the weights are not.

If that test ever fails because `attention_bias` became true, the correct
response is still to stop and report — supporting a bias needs a bias placement
field on the module config, which is a contract change.

---

## 7. What was changed, and what it is worth

Commit `768c5ca2771`, on `apbernal/tttv2_wh_glx_2d_modules_milestone_b`.

### 7.1 Two placement defects ported from Llama — UNVERIFIED on device

Both were carried unchanged by the Qwen package; job 1 found and fixed both on
silicon for Llama. **Neither has been seen to run on a mesh.** They are ports of
qualified changes, not qualified changes.

1. **`_relocate`** (`model.py`) used `to_memory_config(t, memcfg, dtype)` — the
   three-argument form — which reaches `ttnn::prim::copy` and splits work over
   the full compute grid, aborting under the decode sub-device manager with
   `Kernel group cores do not match sub device cores`. Replaced with the
   worker-confined `sharded_to_interleaved` / `interleaved_to_sharded` pair.
   This is what job 1's handoff called "the single highest-value thing to do
   before your first device run".
2. **The embedding's decode output** was `ttnn.L1_MEMORY_CONFIG`, which is
   *interleaved*; `ttnn.embedding` takes its program grid from a sharded
   output's shard grid and only from there, so an interleaved output spread it
   over the whole grid and clashed with the prefetcher's L1 on the sender
   columns. It now names `decode.residual_memcfg`, which also makes the
   following relocation a no-op.

### 7.2 Three placement tests that pin them

Added to `test_model_host.py`. These exist because `test_model_host.py` **never
called `build_qwen3_32b_galaxy_transformer_2d_config`**, so the embedding
placement had no host coverage at all — a defect there would have cost a device
run to find. It turns out the whole transformer config *does* build against the
`MagicMock(spec=ttnn.MeshDevice)` the file already uses, and the decode
placements resolve to real `MemoryConfig` objects, so the wiring between modules
is checkable on host even though the partition is not.

The embedding test was **confirmed to fail against the unfixed code** (the fix
was temporarily reverted to check, then restored — `logs/` and §9).

### 7.3 The host adaptor qualification

`models/common/tests/models/qwen3_32b_galaxy/test_hf_conversion_host.py`, 13
tests, no device opened. This is the Qwen equivalent of what job 1 called the
most valuable thing it produced: it removes "is it the weights or the mesh?"
from every future silicon failure. Qwen's Q/K norm and its own RoPE theta make
it more valuable here than it was for Llama, not less.

It does not import the Llama package or the Llama test module. It does share
`models/common/tests/modules/_hf_reference`, which the house rules permit and
which has precedent.

---

## 8. Regression gates

### 8.1 Host

```sh
python -m pytest -q --ignore-glob="*_wh_galaxy*.py" \
    --ignore=models/common/tests/modules/moe/test_generalized_moe_gate.py \
    --ignore=models/common/tests/modules/moe/test_tt_moe_decode.py \
    --ignore=models/common/tests/modules/moe/test_tt_moe_gate.py \
    models/common/tests/modules \
    models/common/tests/models/llama33_70b_galaxy/test_model_host.py \
    models/common/tests/models/qwen3_32b_galaxy/test_model_host.py \
    models/common/tests/models/qwen3_32b_galaxy/test_hf_conversion_host.py
```

**410 passed, 1952 skipped, 0 failed** (`logs/21`). 397 of those are
pre-existing; 13 are new.

**The brief's gate as written is not host-only.** `models/common/tests/modules`
collects device suites, and with the mesh down they *error* rather than skip:
**397 passed, 289 errors, 0 failed** (`logs/20`). All 289 are device-open
failures (`Read 0xffffffff` at `conftest.py:452` /
`ttnn/distributed/distributed.py:631`) in `*_wh_galaxy*.py` and the three
`moe/` device suites. This is worth correcting in the brief for `mb-coverage`.

### 8.2 Boundaries — all clean

```sh
git diff --name-only 52def65194c..HEAD | grep '_1d\.py'      # empty  OK
git diff --name-only 52def65194c..HEAD | grep 'llm_runtime'  # empty  OK
git grep -n "models.common.models.llama33_70b_galaxy" \
    -- models/common/models/qwen3_32b_galaxy                 # empty  OK
```

Also checked from the Milestone A tip `bc6ad03bfc2` — both empty.

### 8.3 Llama's device gates were NOT re-run, and did not need to be

**No shared code under `models/common/models/galaxy/` was touched.** The only
files changed are inside the Qwen package and its tests:

```text
models/common/models/qwen3_32b_galaxy/model.py
models/common/tests/models/qwen3_32b_galaxy/test_model_host.py
models/common/tests/models/qwen3_32b_galaxy/test_hf_conversion_host.py   (new)
```

So Llama's evidence is not invalidated by this job. (It could not have been
re-run in any case — the mesh is down — which is precisely why not touching
shared code mattered tonight.)

---

## 9. Decisions made without being able to ask

Recorded as the brief instructs, with the conservative option taken.

1. **Spent both recovery attempts even though job 1's handoff said not to
   retry.** Justified because the failure signature had *changed* (eleven
   missing boards vs one; PCIe ID 17 vs board 7), so it was not the same state
   job 1 gave up on. The second attempt used `tt-smi -r`, a different code path
   that tt-smi itself suggests, rather than repeating `-glx_reset`. Both failed;
   no third attempt was made.
2. **Did not download the ~65 GB Qwen3-32B checkpoint.** With no mesh, the
   weights could not have been used tonight, and a large unattended network
   fetch is not something the brief authorises. The `config.json` that *is*
   cached was enough to settle the contract and the bias question.
3. **Did not write new device test files.** Deliverable 2 asks for them, and
   this is the one deliverable deliberately left undone. A large device test
   file that has never been executed — not even collected against real weights —
   would invite `mb-coverage` to trust it. Recording precisely what to run
   (§"Suggested order" in the handoff) is more useful and less misleading than
   shipping unrun code. This is a deliberate omission, not an oversight.
4. **Rewrote the Q/K-norm test after noticing it was circular.** See §4. The
   original would have passed against a broken adaptor.
5. **Left `lm_head`'s `decode_output_memcfg = ttnn.L1_MEMORY_CONFIG` alone.**
   Llama's is identical, so changing Qwen's would have introduced a divergence
   from the qualified model on no evidence. Job 1 flagged only the embedding.

---

## 10. Open items

| Item | State |
| --- | --- |
| Mesh: 11 boards off the PCIe bus | **BLOCKED (infra)** — needs IPMI tray power cycle or host reboot |
| Qwen3-32B weights absent from this host | **BLOCKED** — needs a ~65 GB fetch into `/proj_sw/user_dev/hf_data` |
| Qwen block decode/prefill PCC ≥ 0.99 | not run |
| Qwen full model + demo | not run |
| Qwen teacher-forced accuracy (top-1 ≥ 89%, top-5 ≥ 97%) | not run |
| 64-head geometry on silicon | still unqualified (host: qualified) |
| Q/K norm on silicon | still no number anywhere |
| Ring width 800 vs 960 on silicon | still unverified (arithmetic: confirmed) |
| `_relocate` and embedding ports | **UNVERIFIED on device** |
| D-B9 (attention decode matmul L1 clash) | still open, inherited from job 1 |
| `in0_block_w` gcd(k,4) hypothesis | still device-unverified, inherited from job 1 |
| L1 (global-CB ownership across two constructions) | still never measured |
| Brief's "host" regression gate includes device suites | worth correcting for `mb-coverage` |

---

## 11. Proposed status-page text

Not written to `MILESTONE_A_STATUS.md` or `tttv2_2d_modules_plan.md` — this
job's brief does not name them, so per the house rules the text goes here
instead.

> **O4 (Qwen 64-head decoupled geometry).** Partially closed. The real geometry
> — `dim 5120`, `n_heads 64`, `head_dim 128`, `attention_dim 8192`, `wo
> [8192, 5120]` — is now qualified **on host**: attention rebuilt from the
> converted tensors alone reproduces unmodified HF `Qwen3Attention` at
> PCC ≥ 0.9999, and the `wo` placement pairing, head-concat width and
> `dim`-wide post-`wo` residual are pinned by tests
> (`models/common/tests/models/qwen3_32b_galaxy/test_hf_conversion_host.py`,
> three fresh processes). It remains **unqualified on silicon**: `mb-qwen` had
> no working mesh. Note that `local_qkv_size == local_dim == 1280` for this
> model, so a fused-QKV-vs-residual width confusion is shape-invisible;
> `local_attention_dim` is 1024 and is the width that differs.
