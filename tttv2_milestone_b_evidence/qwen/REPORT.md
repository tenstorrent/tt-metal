> **SUPERSEDED IN ITS HEADLINE, 2026-08-27, by `mb-qwen` attempt 2.**
>
> Everything below this banner is attempt 1's record of a night on which the
> mesh had eleven boards off the PCIe bus. Its host work stands. Its two
> headline claims do not:
>
> 1. **The mesh is healthy.** 32/32 boards in `/sys/class/tenstorrent`, and
>    `test_partition_wh_galaxy.py` passes in 12.93 s.
> 2. **Qwen3-32B's weights are on this machine**, at
>    `/localdev/ctr-apbernal/hf_data/hub/models--Qwen--Qwen3-32B` (17/17 shards,
>    revision `9216db5781bf`). Attempt 1 searched two other caches and concluded
>    from them that the checkpoint did not exist. **`HF_HOME` must be
>    `/localdev/ctr-apbernal/hf_data`** — not `/proj_sw/user_dev/hf_data`, which
>    reaches Llama only, and not the `.../hf_data/hub` this job inherits from its
>    shell, which reaches neither.
>
> Attempt 2 put Qwen3-32B on silicon and met every gate in the brief. Its
> account is **§A2**, at the end of this file; the run-by-run narrative is
> [`ATTEMPT2.md`](ATTEMPT2.md). Where the two disagree, §A2 is later.

---

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
| `HF_HOME` unset makes Llama's real-checkpoint host tests **skip silently** | verified both ways; see §12 |

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


---

## 12. A late finding worth more than it looks

`HF_HOME` is **unset** in this job's environment. The consequence is not a
failure — it is a **silent skip**.

Llama's two real-checkpoint host tests resolve the checkpoint through
`snapshot_download`, which with no `HF_HOME` falls through to the network and
gets a 401 on a gated repo, so they skip:

```text
SKIPPED  checkpoint 'meta-llama/Llama-3.3-70B-Instruct' is unavailable:
         You are trying to access a gated repo. 401 Client Error.
```

Verified both ways:

```sh
# unset -> skipped
python -m pytest -q -rs models/common/tests/models/llama33_70b_galaxy/test_hf_conversion_host.py
# 7 passed, 2 skipped

# set -> the same test passes
HF_HOME=/proj_sw/user_dev/hf_data python -m pytest -q   "…/test_hf_conversion_host.py::test_real_checkpoint_rope_tables_match_an_independent_llama3_scaling"
# 1 passed
```

Llama-3.3-70B really is on this host — 31 safetensors shards, 368 GB, at
`/proj_sw/user_dev/hf_data/hub/models--meta-llama--Llama-3.3-70B-Instruct` —
so the skip is purely an environment artefact, not a missing checkpoint.

It matters because job 1's "the Llama adaptor is numerically correct on host"
result includes those two real-checkpoint tests. Whether they *ran* in that job
depends on whether `HF_HOME` was exported in its shell. Anyone auditing that
claim should check its logs for the skip lines rather than assume. `mb-coverage`
should export `HF_HOME=/proj_sw/user_dev/hf_data` before any host gate.


---

# §A2 — attempt 2 (2026-08-27): Qwen3-32B on silicon

The remainder of this file is `mb-qwen` attempt 2's report. It supersedes the
verdict above.


Written 2026-08-27, unattended, on branch
`apbernal/tttv2_wh_glx_2d_modules_milestone_b` from `690737450a8`.

Run-by-run narrative: `ATTEMPT2.md`. Environment and mesh facts:
`ENVIRONMENT.md`. Raw logs: `logs2/`.

**Attempt 1 of this job reported `BLOCKED (infra)` on a mesh with eleven boards
off the PCIe bus and did host work only. The mesh is healthy now, and this
attempt is the first to put Qwen3-32B on silicon.**

---

## 1. Verdict

| Milestone B item | Verdict | Evidence |
| --- | --- | --- |
| Step 4 — adaptor and one-layer model | **PASSED** | 74 host tests; `a2_01_geometry` |
| Q/K norm alone, both modes (brief risk 2) | **PASSED** | `a2_09/11/14_qknorm` |
| Step 5 — one block, decode + prefill, PCC >= 0.99 | **PASSED** | `a2_13/15/16_block`, bit-identical |
| Step 5 — prefill 2048 | **PASSED** | `a2_17/24/25_prefill2048` |
| Step 6 — full 64-layer model, prefill + first decode | **PASSED** | `a2_19/23/28_fullmodel` |
| Step 6 — coherent demo output, batch 1 and batch 32 | **PASSED** | `a2_21/29/30_demo_b1`, `a2_22/31/32_demo_b32` |
| **Milestone B accuracy gate for Qwen** (top-1 >= 89%, top-5 >= 97%) | **PASSED** | `a2_20/33/34_accuracy` |
| The 64-head decoupled geometry | **QUALIFIED ON SILICON** | §4 |
| Llama's gates after this job's shared changes | **GREEN** | §9 |

Two defects were found and fixed on hardware, both invisible to Llama by
construction, both named here for the first time: **D-B26** (the per-head Q/K
decode norm was unplaceable, three separate ways) and **D-B27** (the decode LM
head's all-reduce was left no worker cores and segmentation-faulted).

## 2. The numbers

Every figure below is from a log in `logs2/`, and every one was produced three
times in three fresh `python -m pytest` processes with a `tt-smi -glx_reset`
between them.

### Step 5 — one Qwen block

```text
prefill 128 logits                        0.999303669584255
prefill 128 cache K (users 0, 8, 16, 24)  0.9998897994661545
prefill 128 cache V (users 0, 8, 16, 24)  0.9998944730661905
decode position 128 logits (u 0,8,16,24)  0.999360219056066
decode 128 cache K (users 0, 8, 16, 24)   0.9998896420783983
decode 128 cache V (users 0, 8, 16, 24)   0.9998939662639094
prefill 2048 logits                       0.9990203192392576
prefill 2048 cache K (users 0,8,16,24)    0.9998918196733165
prefill 2048 cache V (users 0,8,16,24)    0.9998937907368274
```

All twenty-one `[pcc]` lines of the 128 gate are **bit-identical** across the
three runs (`md5sum` of the `[pcc]` lines: `7c751ada099943bbc51df1d4c1b3efc8`).
That is a property worth testing rather than assuming on this mesh: a bfloat16
cross-device logit sum is order-dependent on ETH ring arrival, and
`fp32_dest_acc=True` on the LM head all-reduce is what buys the determinism.

### Per-head Q/K normalization, alone

The brief asks for this to be validated in its own geometry before the block
runs it. Identical on all 32 devices, and identical across three processes:

```text
prefill q_norm  0.9999821268225385      decode q_norm  0.999988294981757
prefill k_norm  0.9999833417066442      decode k_norm  0.9999879678611943
```

This is the first Qwen per-head Q/K norm result on silicon in either mode.
Milestone A's D2 aborted in op validation before producing one, and its fix left
the decode side unrunnable rather than merely unrun - see §5.

### Step 6 — the full 64-layer model

```text
full model prefill 128 predicted 6049, reference top-5 [6049, 31728, 16073, 65129, 35965], target 6049
full model decode position 128 predicted 389, reference top-5 [389, 220, 6702, 3217, 6527], target 25
```

The decode prediction is the reference's own top-1 at that position, which is
what the gate asks; it differs from the *teacher-forced target* token because
the target is the next token of the stored reference text, not the reference
model's argmax. Both are printed so the distinction is on the record.

### The Milestone B accuracy gate for Qwen

Teacher-forced, batch 1, prompt 512, decode 511, against
`models/tt_transformers/tests/reference_outputs/Qwen3-32B.refpt`:

```text
[accuracy] reference=Qwen3-32B prompt=512 decode=511
[accuracy] top-1 498/511 = 0.9746 (gate >= 0.89)
[accuracy] top-5 511/511 = 1.0000 (gate >= 0.97)
```

Raw counts, not only percentages. Exact command:

```sh
export HF_HOME=/localdev/ctr-apbernal/hf_data
python -u -m pytest -v -rA --color=no -p no:cacheprovider --timeout=2300 \
  models/common/tests/models/qwen3_32b_galaxy/test_full_model_wh_galaxy.py::test_qwen3_32b_galaxy_teacher_forced_accuracy_batch1
```

### Demo output

`models/common/models/qwen3_32b_galaxy/demo.py`, greedy host sampling, 16 new
tokens. Batch 1:

```text
slot 0 prompt: 'Explain what a tensor is to a software engineer in two sentences.'
slot 0 text  : '<think>\nOkay, the user wants me to explain what a tensor is to a'
```

Batch 32, eight distinct prompts repeated four times:

```text
slot 0 text  : '<think>\nOkay, the user wants me to explain what a tensor is to a'
slot 1 text  : '<think>\nOkay, so I need to find three prime numbers that are greater than'
```

Slot 0's tokens are identical whether it is served alone or alongside 31 others,
slots holding the same prompt agree exactly, and slots holding different prompts
diverge - which is what separates "a slot read another slot's KV blocks" from
"every slot collapsed onto one user's cache". The `<think>` prefix is Qwen3's
own reasoning-mode template, not a defect.

## 3. What this job changed, and why

Fifteen files, six commits, all under `models/common/models/qwen3_32b_galaxy/`,
`models/common/tests/models/qwen3_32b_galaxy/` and two shared files declared in
§8.

### 3.1 Llama's six model-code fixes, ported (commit `77a04f233cd`)

`job1_llama_state_for_qwen.md` §3 lists six deltas that live in *model* code and
therefore do not come to Qwen by construction. All six were present in this tree
and all six are now applied:

1. **The prefetcher registered `wqkv` and `wo`** (D-B25a). A prefetched matmul
   reads its weight from the global circular buffer in registration order and
   only the 24 ring cores receive that buffer; `recipes.py` puts the MLP on the
   ring and the attention decode projections on a confined worker rectangle.
   Two unconsumed entries per layer shifted every later consumer by one.
   `QWEN3_32B_PREFETCHED_WEIGHT_NAMES` is now `("w1", "w3", "w2")` and attention
   takes `_UnprefetchedContext`, which keeps the worker sub-device id and drops
   the buffer.
2. **The rotary defaulted to the non-fused pair** (D-B25b), the Blackhole
   fallback, which wants a different cos/sin layout and writes an infinite K.
   Now `use_qk_fused_rotary: bool = True` in all three places that can set it.
3. **The decode LM head was interleaved L1** with no program config and no
   sub-device ids. Now the 24-core `gather_in0` ring with `bfloat8_b` output.
4. **`_relocate` staged sharded -> DRAM -> target** for a non-DRAM interleaved
   target, which is `ttnn::prim::copy` on the full compute grid (D-B10). One hop
   now.
5. **The checkpoint loader seam** (`load_hf_model`), which is what makes three
   fresh processes per gate affordable.
6. The two items §3.6 of the distilled file records as already discharged were
   re-verified present and not re-spent.

Two host assertions in the Qwen package were stale rather than wrong-headed and
are corrected with their reasons: the padded vocabulary is 153600 rather than
152064 (the shared `galaxy_padded_vocab_size` pads to
`GALAXY_ROWS * RING_ALIGNMENT` so the LM head's all-reduce is exact - D-B19),
and the prefetch-order test pinned the defect above. Neither was caught earlier
because the Qwen host suites are not in the Llama host gate.

### 3.2 The device test file, rewritten (same commit)

`test_model_wh_galaxy.py` was a static file that had never run. It is now a
faithful port of Llama's qualified one - `compose_galaxy_logits` rather than
`to_torch_auto_compose`, the KV cache compared against HF with
`reverse_permute_1d`, every column-local user reported rather than user 0, three
comparison windows on the cache, and a decode bisection - plus two tests that
exist only for Qwen: the decoupled geometry stated on the mesh, and the per-head
Q/K norm validated alone in both modes before the block runs it.

## 4. The 64-head geometry verdict, stated explicitly

**Qwen3-32B's real 64-head decoupled attention geometry is qualified on WH
Galaxy `(8, 4)` silicon**, in prefill and in decode, at PCC >= 0.999 on the
block's logits and >= 0.9998 on both KV caches, three times in three fresh
processes with bit-identical results, and at 97.46% top-1 through 64 layers and
511 teacher-forced decode steps.

Milestone A's recorded "Qwen3-32B attention qualified, PCC >= 0.99" was measured
against a **40-head** fixture (`test_attention_2d_wh_galaxy.py:86`,
`dim=5120, n_heads=40`) chosen so that `n_heads * head_dim` happened to equal
`dim`. That evidence does not transfer, and this is the first hardware evidence
for the square-free case. What the mesh reports:

```text
dim=5120  n_heads=64  head_dim=128  attention_dim=8192   (1.60 x dim)
local_dim=1280  local_attention_dim=1024  local_qkv_size=1280  local_hidden_dim=3200
wo is [8192, 5120]; per mesh row [1024, 1280]
wo DRAM shard (local_attention_dim) : 12 cores, shape [1024, 128]
wo DRAM shard if dim were used      : 12 cores, shape [1280, 128]
```

The four things the brief asks to be checked before a PCC is trusted:

* **the WO input width is `attention_dim / GALAXY_ROWS`, not `dim / GALAXY_ROWS`** —
  1024, not 1280. Pinned on host by
  `test_wo_weight_placement_is_paired_with_attention_dim_not_dim` and asserted on
  the mesh by `test_..._geometry_is_decoupled_8x4_qwen3_32b`, which also shows
  the two placements are distinguishable (`[1024, 128]` vs `[1280, 128]`);
* **the head-concat output width matches what WO expects** — exercised by the
  block gate; a mismatch could not produce logits at PCC 0.9994;
* **the residual added after WO is `dim`-wide** — the decode residual placement
  is `local_dim` 1280 over 10 cores, and the bisection reports the residual after
  attention against HF's own hidden state at every boundary;
* **`dram_sharded_weight_memory_config(mesh, local_attention_dim, local_dim)` is
  what `wo` gets** — pinned by source inspection on host and by the mesh test
  above.

**The trap worth carrying forward: `local_qkv_size == local_dim == 1280` for
this model.** A confusion between the fused-QKV width and the residual width is
shape-invisible here. `local_attention_dim` (1024) is the only one of the three
that differs.

## 5. D-B26 — the per-head Q/K decode norm was unplaceable, three ways

**Status: fixed, qualified on silicon.** This is the unresolved half of
Milestone A's D2, and it is Qwen-only: Llama has no per-head Q/K norm, so no
`HEAD_LOCAL` norm had ever executed a decode step on this mesh.

The brief's risk 2 asked for the Q/K norm to be validated alone before it was
enabled inside the block. That ordering is what turned this into three named
failures instead of one night of bisection.

**(a) Interleaved DRAM — the module's own post-D2 default — is unplaceable.**

```text
TT_FATAL: Kernel group cores do not match sub device cores for programmable
          core type TENSIX          program.cpp:2205: num_intersections == num_cores
```

`ttnn.rms_norm` on an interleaved input resolves `LayerNormDefaultProgramConfig`,
which splits its tile rows over `device->compute_with_storage_grid_size()`: the
whole compute grid, including the prefetch sender columns the loaded decode
sub-device manager does not own. **Prefill is correct on exactly this config, at
PCC 0.99998 on all 32 devices, in the same run** - the mode plan for prefill is a
single sub-device over the full grid. The host test that pinned this asserted
that the model's config *agreed with the module default* in both modes, as though
agreement made decode safe. Agreement is what carried the defect.

**(b) The created heads' own placement is rejected outright.**

```text
TT_FATAL: Height sharded inputs are not supported.
          layernorm_device_operation.cpp:166
```

A standing TODO in that file ("should be similar to interleaved"), not a property
of the arithmetic. So the tensor cannot be normalized where it lives.

**(c) Any single sharded placement destroys a property the rotary depends on.**

```text
TT_FATAL: Q and K must not overlap
          rotary_embedding_llama_fused_qk_device_operation.cpp:95
          is_overlap = q.shard_spec()->grid.intersects(k.shard_spec()->grid)
```

`Attention2D._apply_qk_norm` relocates the created heads to the norm's
`decode_input_memcfg` before calling it, and Q and K do not arrive on the same
cores: `nlp_create_qkv_heads_decode` cuts the head grid row-wise into consecutive
`batch`-core slices and gives Q the first and K the second. Naming any one
placement relocates both onto it.

**The fix**: decode names **no** placement, only the *cores* its kernel may run
on. `RMSNorm2D._decode_head_local` then block-shards the tensor onto those cores
- deriving the shard shape from the tensor, because the created-head padding is
the op's business and not the caller's - runs `ttnn.rms_norm` there, and puts the
result back into the placement the input arrived in. The relocations use
`sharded_to_interleaved` / `interleaved_to_sharded`, never `to_memory_config`,
which between two shard specs resolves to `reshard` and would abort like (a).

Cost: four extra ops per norm, eight per layer. That is a real decode-latency
cost and belongs on the performance follow-up list; it is not a correctness
argument. The obvious cheaper form - teaching `ttnn.rms_norm` to take a
`core_range_set` for interleaved inputs, which its *program factory* already
accepts (`layernorm_op_multi_core.cpp:193`) but no ttnn-level API exposes - is a
tt-metal change and out of scope here.

Two intermediate mistakes are worth recording because each cost a device run:
sizing the block-sharded rectangle from `GALAXY_PHYSICAL_BATCH` when the decode
Q and K carry `users_per_column` users of one *tile* of padded heads (256 rows,
not 1024), and `ttnn.Shape` not supporting slice indexing. The first got through
because the standalone test staged the shape the config assumed rather than the
shape attention produces; it now stages what attention produces, on the disjoint
core slices attention produces, and asserts the norm hands them back on the same
cores.

## 6. D-B27 — the decode LM head's all-reduce was left no worker cores

**Status: fixed, qualified on silicon.** Also Qwen-only, and by luck rather than
by design.

`a2_12_block` carried the entire Qwen decode graph through to the LM head's
column all-reduce and then segmentation-faulted the process:

```text
[ccl] lm_head in:     logical=(1,1,32,19200) tiles=600  shard=(32,800)  cores=24
[ccl] lm_head staged: logical=(1,1,32,19200) tiles=600  shard=(32,384)  cores=50
[ccl] lm_head buffer: logical=(1,1,32,76800) tiles=2400 shard=(32,1536) cores=50
AllGather is being launched on a subdevice with fewer worker cores available
than ideal. Ideally 4 cores (1 per link and 4 links) are made available but only
0 are available.          all_reduce_async_program_factory.cpp:61
Fatal Python error: Segmentation fault
  models/common/models/galaxy/collectives.py:292 in _persistent_all_reduce
```

D-B19's invariant held throughout — 50 x 384 = 19200 and 50 x 1536 = 76800, both
exact — so the reduction would not have hung. It had no cores to run its fabric
links on. `lm_head_reduce_core_count` returns the largest divisor of the tile
width that fits the worker envelope, and for Qwen's 600 tiles that is 50, the
whole envelope. Llama's 504 tiles have no divisor between 43 and 50, so its 42
leaves eight free — luck, not design.

The search now reserves `GALAXY_CCL_RESERVED_WORKER_CORES = 4` explicitly.
Llama still resolves 42 (bit-identical), Qwen resolves 40: 15 tiles, 480 columns
per core, and 40 x 480 = 19200 exactly, so D-B19's invariant is preserved.

**This is a shared-code change and it is the reason Llama's gates are re-run in
§9.**

## 7. The bisection's two low readings were the test's reference, not the model

Worth its own section because it is the sort of thing this project has learned to
distrust, and because the same mistake is live in the Llama package.

The decode bisection reported the residual stream at **0.9182** and the final
norm at **0.7657** while the logits computed from that same normed tensor read
0.999360 — three runs, identical. Rounding the reference to bfloat16 before
comparing changed neither number, so it was not quantization, and the magnitudes
were suspicious: the device's final norm peaked at 58.75 where the *reference's
residual* peaked at 58.

The cross-comparison settled it in one run:

```text
probe cross reference final norm    vs device residual : 0.4747339146632379
probe cross reference after layer 0 vs device normed   : 0.9995821444530748
```

and the host check settled why. `out.hidden_states[-1]` is the output of the
model's **final norm**, not of the last decoder layer:
`Qwen3Model.forward` in transformers 5.12.1 runs
`hidden_states = self.norm(hidden_states)` before the output-hidden-states hook
collects the last entry. On the real checkpoint, one layer, position 128:

```text
hidden_states[1]              |max| 79
layer 0's own forward hook    |max| 18     pcc vs hidden_states[1]  0.9178
norm(layer 0's output)        |max| 79     pcc vs hidden_states[1]  1.0
norm(hidden_states[1])        |max| 332
```

So the bisection compared the device's residual against a *normalized* reference
(0.9178 — the 0.9182 it reported, to the noise) and its final norm against the
norm applied *twice*. The device was right at both boundaries. The layer output
now comes from its own forward hook and the final norm from `hidden_states[-1]`;
the cross-comparison stays, because it is what would find the next one.

**The same mislabelling is in `models/common/tests/models/llama33_70b_galaxy/test_model_wh_galaxy.py`,**
and `tttv2_milestone_b_evidence/llama/REPORT.md` explains the resulting 0.979 and
0.990 as "the bfloat16 residual against an fp32 reference is the floor". That
explanation is wrong. The brief forbids this job from modifying the Llama
package, so the finding is recorded here and in the handoff rather than patched.

No gate moved: these readings were reported, never asserted, and the asserted
logits were 0.999360 in every run before and after.

## 8. Shared code touched, declared

Two files outside the Qwen package, both required by a defect this job found on
hardware, both with the reduction the plan's extension discipline asks for.

### `models/common/modules/rmsnorm/rmsnorm_2d.py`

Adds `RMSNorm2DConfig.decode_compute_cores` (optional, default `None`) and the
`HEAD_LOCAL` decode path that uses it, plus
`_place_without_leaving_subdevice` and `_head_local_compute_memory_config`.

*Reduction*: the module already owned "how a head-local norm runs"; what it did
not own was "on which cores", and on a partitioned decode mesh that is not
derivable from the tensor — only the caller knows which cores its sub-device
holds. Everything else stays derived: the shard shape comes from the tensor, and
the output placement is the input's. With the field unset the behaviour is
byte-for-byte what it was.

*Effect on Llama*: none, and not by inspection alone. Llama passes no
`q_norm_config`, so it constructs no `HEAD_LOCAL` norm at all and the branch is
unreachable from it. `_resolve_2d_config`'s only other change is guarded on
`decode_compute_cores is not None`.

### `models/common/models/galaxy/recipes.py`

`lm_head_reduce_core_count` reserves `GALAXY_CCL_RESERVED_WORKER_CORES = 4`.

*Reduction*: the function's job is "the largest core count that fits and divides
the width evenly"; "fits" had been read as "fits the worker envelope", and
hardware says it means "fits the worker envelope alongside the collective that
consumes it". That is a correction to the existing contract, not a new one.

*Effect on Llama*: the resolved count is **42 before and after**, pinned by a new
host test, and confirmed on silicon in §9.

### Boundaries

```text
git diff --name-only 690737450a8..HEAD | grep '_1d\.py'         (empty)
git diff --name-only 690737450a8..HEAD | grep 'llm_runtime'     (empty)
git grep -n "models.common.models.llama33_70b_galaxy" -- models/common/models/qwen3_32b_galaxy
                                                                (empty)
git grep -nE "models\.demos" -- models/common/models/qwen3_32b_galaxy
                                                                (empty)
```

`models/common/modules/MILESTONE_A_STATUS.md` and `tttv2_2d_modules_plan.md` are
untouched; proposed text for them is in §11.

## 9. Llama's gates, re-run because this job touched shared code

Every one of the six gates `job1_llama_state_for_qwen.md` §10 names, on this
commit, with `HF_HOME=/localdev/ctr-apbernal/hf_data`. Every number is
**bit-identical to the values that file records** for `mb-llama` attempt 3.

| gate | log | result |
| --- | --- | --- |
| step-2, prefill 128 + decode | `a2_40/41/42_llama_step2` | 1 passed x3, all 21 `[pcc]` lines identical (`db7664ee084920afb01f6b5835402511`) |
| single-row prefill 2048 | `a2_43_llama_prefill2048` | 1 passed |
| 80 layers, prefill + first decode | `a2_44_llama_fullmodel` | 1 passed, 13 m 44 s |
| Milestone B accuracy gate | `a2_45_llama_accuracy` | 1 passed, 21 m 33 s |
| demo batch 1 | `a2_46_llama_demo_b1` | 1 passed |
| demo batch 32, no cross-slot contamination | `a2_47_llama_demo_b32` | 1 passed |

```text
prefill 128 logits                        0.999584002863212
prefill 128 cache K / V                    0.9999347766610057 / 0.9997498179150203
decode position 128 logits (u 0,8,16,24)  0.9997463458407887
decode 128 cache K / V                     0.9999342257320987 / 0.999749334500399
prefill 2048 logits                       0.9996201066107949
accuracy top-1                            501/511 = 0.9804 (gate >= 0.91)
accuracy top-5                            511/511 = 1.0000 (gate >= 0.99)
demo slot 0  'A tensor is a multi-dimensional array of numerical values, similar to a matrix,'
```

The demo text is character-identical between batch 1 and batch 32, as
`mb-llama` recorded. **Llama is green, and the reduction-core count it resolves
is 42 before and after this job's change to `lm_head_reduce_core_count`.**

## 10. Still open

Nothing below was introduced by this job; all of it is inherited, and none of it
blocks a Milestone B gate.

* **L1's remaining half — prefill after a decode.** Untouched. Production has the
  same property, `mb-llama` attempt 3 implemented and then refuted the obvious
  fix on hardware, and the open hypothesis is still to confine the prefill mode
  plan to the worker cores. Not required by any Milestone B gate. Qwen inherits
  it unchanged: every gate here either prefills first or never prefills again.
* **L1's global-CB ownership across two constructions** (`test_two_models_in_one_process`).
  Still never run, for Qwen as for Llama.
* **D-B9** — the attention decode matmul CB clash. `job1`'s `in0_block_w`
  `gcd(k, 4)` change is in the tree and has now run for Qwen's geometry as well
  as Llama's without a clash: Qwen's `local_qkv_size` is 1280 against Llama's
  2048, and the decode block gate passes three times. The structural follow-up -
  moving the attention decode matmuls onto the 24-core ring, which would also
  let them be prefetched again - is unstarted and is a performance item.
* **L3** was closed by `mb-llama` and this job did not reopen it: Qwen's
  attention decode runs on the same confined worker rectangle and is numerically
  correct.
* **The Llama bisection's mislabelled reference stages** (§7). A defect in
  `models/common/tests/models/llama33_70b_galaxy/test_model_wh_galaxy.py` and in
  the narrative of `tttv2_milestone_b_evidence/llama/REPORT.md`. Out of scope for
  this job by the brief; the fix is four lines and is described in §7.
* **The head-local norm's four relocations per call.** Correctness first; this is
  a decode-latency item. The clean fix is a ttnn-level way to pass a
  `core_range_set` to `ttnn.rms_norm` for an interleaved input - the program
  factory already takes one (`layernorm_op_multi_core.cpp:193`) and only the
  low-level `create_descriptor` binding exposes it.
* **Plan step 7** - paged KV, prefix cache, concat-32, device sampling, long
  context - belongs to `mb-coverage` and was not started here. Qwen's
  `test_step7_coverage_wh_galaxy.py` exists and has still never been run.

## 11. Proposed text for the documents this job may not edit

For `models/common/modules/MILESTONE_A_STATUS.md`, under D2:

> **D2 (head-local RMSNorm) — the decode half is now closed, on silicon.** D2's
> fix made interleaved DRAM the default placement for `RMSNorm2DGeometry.HEAD_LOCAL`
> in both modes. That is correct for prefill and *unplaceable* for decode on a
> partitioned mesh: an interleaved `ttnn.rms_norm` resolves
> `LayerNormDefaultProgramConfig`, which spreads its tile rows over the whole
> compute grid. Milestone B's Qwen3-32B bring-up measured it (`D-B26`) and closed
> it with `RMSNorm2DConfig.decode_compute_cores`. Qwen3-32B's per-head Q/K norm is
> now qualified in both modes at PCC >= 0.99998 on all 32 devices.

For `tttv2_2d_modules_plan.md`, in the Milestone B exit-gate table:

> Qwen3-32B, WH Galaxy `(8, 4)`, real 64-head decoupled geometry: one block
> prefill 128 / decode 128 at PCC 0.9993 with both KV caches at 0.9999; prefill
> 2048 at 0.9990; 64-layer model prefill + first decode token; teacher-forced
> batch 1, 512/511: **top-1 498/511 = 97.46%, top-5 511/511 = 100.00%**. Three
> fresh processes per gate, bit-identical.

## 12. Index of the runs behind every number

| claim | run(s) |
| --- | --- |
| mesh healthy | `a2_00_partition` |
| 64-head geometry on the mesh | `a2_01_geometry` |
| D-B26, the three failures | `a2_02/03_qknorm`, `a2_04_block`, `a2_06/07_block`, `a2_10_block` |
| Q/K norm qualified, both modes | `a2_09/11/14_qknorm` |
| D-B27, the segfault | `a2_12_block` |
| step-5 gate x3 | `a2_13/15/16_block` |
| prefill 2048 x3 | `a2_17/24/25_prefill2048` |
| decode bisection | `a2_18/26/27_bisection` (old labels), `a2_35_bisection` (cross probe), `a2_50/51/52_bisection` (corrected) |
| full model x3 | `a2_19/23/28_fullmodel` |
| accuracy gate x3 | `a2_20/33/34_accuracy` |
| demo batch 1 x3 | `a2_21/29/30_demo_b1` |
| demo batch 32 x3 | `a2_22/31/32_demo_b32` |
| Llama regression | `a2_40..47_llama_*` |

## 13. Host regression gates

```sh
export HF_HOME=/localdev/ctr-apbernal/hf_data
bash tttv2_milestone_b_evidence/qwen/host_gate.sh <log>      # 570 passed, exit 0
python -m pytest -q -rA --color=no -p no:cacheprovider --ignore-glob="*_wh_galaxy*.py" \
  models/common/tests/models/qwen3_32b_galaxy/test_model_host.py \
  models/common/tests/models/qwen3_32b_galaxy/test_hf_conversion_host.py   # 50 passed, exit 0
```

`a2_60_host_gate.log`, `a2_61_qwen_host.log`. The Llama selection was 565 at the
start of this job and is 570 now: five tests added, four pinning D-B26 and D-B27
in the shared modules and one the reservation arithmetic for both models.

**Zero skips in either.** A skip is the failure mode to watch here: with the
wrong `HF_HOME`, `hf_config_or_skip` turns every real-checkpoint test into a
`SKIPPED` and a run looks green having measured nothing. This job saw that
happen once, by forgetting the export on a single host invocation, and it is why
every script under `tttv2_milestone_b_evidence/qwen/` sets it explicitly.
