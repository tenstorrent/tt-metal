# Job 1 (`mb-llama`) → `mb-qwen`: completion handoff

Written 2026-08-27 by `mb-llama`, unattended.
Full account: `tttv2_milestone_b_evidence/llama/REPORT.md`.
Environment and mesh facts: `tttv2_milestone_b_evidence/llama/ENVIRONMENT.md`.

## Read this paragraph first

**The mesh is down.** Board 7 dropped off the PCIe bus and `tt-smi -glx_reset`
cannot recover it, because the node it needs (`/dev/tenstorrent/7`) is the one
that is gone. Two recovery attempts were used and both failed, so this was
recorded as `BLOCKED (infra)`
(`tttv2_milestone_b_evidence/llama/logs/90_BLOCKED_infra_pcie_board7.log`).

**Check the mesh before you plan anything:**

```sh
ls /dev/tenstorrent | wc -l        # 32 nodes were still present, but board 7 was unreadable
tt-smi -ls                         # was aborting inside tt_umd and listing zero boards
```

If it is still broken, you cannot do device work at all. Say so and stop; do not
burn a night on retries. It needs an IPMI power cycle of the tray or a host
reboot, which is outside what an unattended job may do.

**Llama did not reach its gates.** No PCC number, no accuracy number, no demo
output exists. What exists is a decode graph that now executes most of the way
through one layer, and nine diagnosed defects, eight of them fixed. If you are
looking for a Llama baseline to compare Qwen against, there isn't one.

## The one thing that will save you the most time

Every Milestone B decode defect this job found — eight of nine — is the same
mistake in a different place:

> **A decode-mode program touched a core the loaded sub-device manager does not
> own.**

Measured on this mesh (not assumed — a mocked mesh cannot tell you this):

```text
compute grid      x=0..6, y=0..9                              70 cores
worker_cores()    {[1-0 - 3-9], [5-0 - 6-9]}                  50 cores
prefetch senders  x=0 and x=4                                 12 cores
in NO sub-device  {[0-1 - 0-3], [0-6 - 0-8], [4-3], [4-8]}     8 cores
```

Two consequences to keep in your head the whole night:

1. **The worker envelope is not contiguous.** The `x=4` sender column splits it.
   Its *bounding box* is `x=1..6`, which includes senders — and several ttnn ops
   use exactly the bounding box.
2. **Sender ∪ worker does not cover the grid.** Any program built over the full
   compute grid is illegal under the decode manager.

The symptom, always:

```text
TT_FATAL ... Kernel group cores do not match sub device cores
             for programmable core type TENSIX
```

and because the abort is inside a multi-sub-device program, **the mesh is left
un-drainable**: teardown blocks forever in
`FDMeshCommandQueue::~FDMeshCommandQueue`, the process keeps the per-chip UMD
locks, and the *next* run — or any host suite that opens a device — blocks on
`CHIP_IN_USE_<n>_PCIe`. Every one costs a kill and a `tt-smi -glx_reset`.

### Run this first, every time

```sh
python -u -m pytest -v -rA --color=no -p no:cacheprovider --timeout=900 \
  models/common/tests/models/galaxy/test_partition_wh_galaxy.py
```

New in this job. No checkpoint, no model, no weights — **13 seconds**. It prints
the real partition and fails if a dense matmul work grid would leave the worker
envelope. It is also the cheapest mesh health check you have: use it to decide
whether the mesh is worth a three-minute checkpoint-loading run.

### Ops that are NOT sub-device aware (verified in the source)

| Op | Site | What it uses |
| --- | --- | --- |
| `ttnn::prim::copy` (i.e. `to_memory_config(t, memcfg, dtype)`) | `copy_default_tilized_program_factory.cpp:44` | full compute grid (has a TODO) |
| `ttnn.typecast` fallback | `typecast_program_factory.cpp:109` | full grid (the *sharded* factory is fine) |
| generic reshard | `reshard_program_factory_generic.cpp:80` | full grid |
| `ttnn.reduce_scatter` | `reduce_scatter_program_factory.cpp:107` | sub-device **bounding box** (has a comment: "interaction with subdevice needs to be investigated") |

Ops that **are** safe, and what to use instead:

* `sharded_to_interleaved` — runs on its input's `shard_spec.grid`;
* `interleaved_to_sharded` — runs on its output shard's cores;
  both accept `output_dtype`, so a recast can ride along instead of needing
  `ttnn.typecast`;
* `ttnn.experimental.reduce_scatter_minimal_async` and `all_gather_async` —
  select cores via `choose_worker_cores`, which intersects the sub-device's real
  `CoreRangeSet`;
* `ttnn.all_gather` — safe *once given* `sub_core_grids`;
* `ttnn.embedding` — takes its program grid from a **sharded** output's shard
  grid, and only from there. With any interleaved output, L1 or DRAM, it spreads
  over the whole grid;
* matmul program configs — set **`allowed_worker_cores`**. ttnn added it for
  this, deprecated `compute_with_storage_grid_size`, and warns when a config that
  supports it leaves it unset.

## What you inherit in the tree

Three commits on `apbernal/tttv2_wh_glx_2d_modules_milestone_b`:

```text
3c1759ff20e  Make the Galaxy decode graph partition-safe: norm, matmul, all-reduce, relocation
1d4ded04592  Fix three Galaxy decode placement defects found on first silicon
e4c1adfa61f  Qualify the Llama Galaxy weight conversion and Llama 3 scaled RoPE on host
```

### Shared changes Qwen inherits automatically

You get these whether you want them or not, so know what they do:

1. **`modules/rope/rope_2d.py`** — the prefill cos/sin tables are written from
   the host source instead of with an on-device `ttnn.clone`. Identical numbers;
   the clone compiled over the full grid and aborted the first decode. This was
   the only lazy device-weight loader among the 2D modules that ran a compute op.
2. **`modules/rmsnorm/rmsnorm_2d.py`** — `_decode_distributed` used to
   **deallocate its own return value**. `to_memory_config` returns the *same*
   tt_metal tensor when the config already matches, and nanobind hands it back as
   a new Python wrapper, so `is not` could not tell "no copy" from "copy". Both
   sites now guard on the memory config first. **Latent since Milestone A; it
   only needed someone to run the decode norm.** Qwen's layer 0 will hit the same
   path.
3. **`models/galaxy/recipes.py`** — rope `batch_grid` now comes from
   `_subgrid_cores` (worker-confined); `dense_matmul_program_config` sets
   `allowed_worker_cores` and uses a smaller `in0_block_w`.
4. **`models/galaxy/collectives.py`** — `_all_reduce` is mode-split: decode uses
   `all_reduce_async` with the keyed resource's persistent buffer, prefill keeps
   the plain pair (which is exactly what `MLP2D` does).
5. **`models/galaxy/plans.py`** — `build_galaxy_decode_collectives` takes
   `residual_dtype` (default `ttnn.bfloat16`) and sizes the shared axis-0
   all-reduce buffer with it. **See the Qwen-specific warning below.**

### Test helpers you should reuse

* **`models/common/tests/models/galaxy/galaxy_checkpoint.py`** —
  `load_layer_subset_causal_lm(hf_model, layer_indices=(0,))` reads only the
  safetensors shards that hold the layers you ask for, plus the embedding, final
  norm and LM head. For Llama-3.3-70B layer 0 that is **3 shards of 30, ~12 GB,
  ~12 GB peak RSS** against 141 GB for `from_pretrained`-then-truncate. Its
  tensors were verified **bitwise equal** to the shards, and the rotary module is
  built from the checkpoint's own config. It is model-agnostic — it will work for
  Qwen3-32B unchanged. With the three-runs rule, this is the difference between a
  6-minute and a 40-second setup per process.
* **`models/common/tests/models/galaxy/test_partition_wh_galaxy.py`** — above.
* **`test_bringup_wh_galaxy.py`'s `_stage` context manager** — copy this pattern.
  It prints and flushes a stage name *before* entering each device call. When the
  mesh is left un-drainable the pytest session never reaches its failure summary
  and the traceback dies with the killed process; the last `[stage] enter` line
  in the log is then the only thing that tells you which call aborted. **It
  located every one of the nine defects with no debugger and no second run.**
* The scripts in `tttv2_milestone_b_evidence/llama/` — `cycle.sh`,
  `device_run.sh`, `ensure_mesh_free.sh`, `after_device_run.sh`. See the harness
  warnings below.

## Warnings specific to Qwen

1. **The shared all-reduce buffer dtype.** `plans.py` now sizes it from
   `residual_dtype`, defaulting to bfloat16 because both models set `MLP2D`'s
   `decode_ccl_dtype` to their `decode_residual_dtype` and both of those are
   bfloat16. **Check that Qwen's precision recipe agrees.** If Qwen's residual
   dtype differs, pass it through rather than letting the default stand — a
   mismatch shows up as:

   ```text
   TT_FATAL ... Cannot set circular buffer size to 65536. This is larger than
                the associated dynamically allocated L1 buffer bank size of 34816 B
   ```

   34816 B is a `[32, 1024]` bfloat8_b shard; 65536 B is the bfloat16 one. That
   defect (D-B8) took two device runs to find because both consumers of a
   *deliberately shared* resource disagreed with the resource.

2. **Qwen's `_relocate` still has the bug Llama's had.**
   `models/common/models/qwen3_32b_galaxy/model.py` has its own copy of
   `_relocate`, and **this job did not touch it** (the brief says do not touch
   Qwen). It still calls `to_memory_config(tensor, memory_config, dtype)` —
   the three-argument form — which reaches `ttnn::prim::copy` and the full grid.
   **Port the Llama version** (`llama33_70b_galaxy/model.py::_relocate` and
   `_place`): it routes through `sharded_to_interleaved` /
   `interleaved_to_sharded` with `output_dtype`. This is the single highest-value
   thing to do before your first device run.

3. **Qwen's embedding decode output.** Llama's was `ttnn.L1_MEMORY_CONFIG`, which
   is *interleaved* and so spread the embedding program over the whole grid,
   clashing with the prefetcher's L1 on the sender cores. Check Qwen's; if it is
   interleaved, name `decode.residual_memcfg` instead — that also makes the
   following relocation a no-op.

4. **The 64-head decoupled geometry has zero hardware evidence** (job 0's O4).
   Nothing in this job changed that, and nothing here exercised a geometry where
   `n_heads * head_dim != dim` — Llama's are equal. Every placement number in
   `ENVIRONMENT.md` is Llama's `local_dim 2048 / local_qkv_size 1280`. Re-derive
   yours; do not assume the shard widths carry over.

5. **L3 is still open, and it will block Qwen exactly as it blocked Llama.** The
   brief told this job that "the Milestone B recipes build the partition-compatible
   ring/`gather_in0` form instead" of Milestone A's terminal `(7,1)` grid. That is
   **true of the MLP and false of attention**: both attention decode matmuls were
   still on `dense_matmul_program_config`. It is now confined with
   `allowed_worker_cores`, which makes it *legal* but leaves only three worker
   columns — and its circular buffers then clash with the decode activations
   resident there (D-B9, open). See the next section.

## The one open defect, and the one unverified change

**D-B9 is open.** After the all-reduce buffer widened to bfloat16, the attention
dense matmul — confined to three columns, so `per_core_N` rose by the same factor
the grid narrowed — overflows L1 on `x=1..3` by about 20 kB:

```text
TT_THROW ... Statically allocated circular buffers in program 320 clash with
             L1 buffers on core range [1-0 - 3-0].
             L1 buffer allocated at 546432, static CB region ends at 566464
```

**In the tree is a candidate fix that hardware has never seen**: `in0_block_w`
from `gcd(k_tiles, 8)` to `gcd(k_tiles, 4)` in `dense_matmul_program_config`,
halving the in1 circular buffer. Host gate green; **device unverified**, because
the mesh died before it could run. Treat it as a hypothesis, not a fix. It is the
only change in this tree in that category.

**The structural answer is to move the attention decode matmuls to the 24-core
ring/`gather_in0` form the MLP already uses.** The recipes already contain
`attention_qkv_collective_input_memcfg` shaped for exactly those 24 ring cores,
so the design anticipated it and the matmuls were left behind. Spreading over 24
cores makes `per_core_N` and the circular buffers small, which removes both the
three-column penalty and the L1 clash. It was not attempted here: it changes the
QKV matmul's shard specs and how the fused create-QKV-heads collective consumes
them, and there was no working mesh left to qualify it.

## Harness bugs — inherit the fixes, not the bugs

Three ways this job's own tooling cost it runs. The scripts in
`tttv2_milestone_b_evidence/llama/` have all three fixed; if you write your own,
you will hit them.

1. **A reaper matching `pgrep -f pytest` killed the *next* run's pytest** a
   second after it started — empty log, exit 137. Reap **your own child by PID**.
2. **The same reaper killed a concurrent host-only gate.** Only reap a process
   that actually has `/dev/tenstorrent` open (`ls -l /proc/<pid>/fd`). Holding a
   device is the property that matters, not the command line.
3. **`tt-smi -glx_reset` fails with `[Errno 19] No such device`** if a holder is
   still alive. Kill the holder *first*, then reset.

And the one the brief already warned about, which is real: `pgrep -af pytest`
matches this job's **own wrapper shells and the `timeout` process**. Always check
`ps -o comm=` and refuse anything that is not `python`/`python3`/`pytest`. This
job verified that guard working — it refused to signal a `bash` and a `timeout`
(see `logs/` around the run-15 cleanup).

Also, three pytest facts worth knowing:

* **`pytest.ini` sets `timeout = 300` globally.** Any device test that loads a
  checkpoint blows through it and dies looking exactly like a hang. Pass
  `--timeout=900`.
* **Use `python -u`.** Without it a `TT_FATAL` traceback is lost when the process
  is killed, because Python block-buffers stdout to a file.
* **`export HF_HOME=/proj_sw/user_dev/hf_data`.** The real checkpoints are
  there; `~/.cache/huggingface` holds config-only entries and no Llama at all,
  and `$HOME` has a 9.4 GB quota. Qwen3-32B *is* in the default cache as a
  config-only entry — check for weights before you trust it.
* **Never run a host gate while a device cycle is live.**
  `models/common/tests/models/galaxy/test_plans.py` opens the UMD driver for all
  32 chips when run as a whole file, and `tests/models/galaxy` as a directory
  collects a `*_wh_galaxy*` device suite. Use
  `--ignore-glob="*_wh_galaxy*.py"`.

## What is actually proven, so you can compare against it

Two things, and no more:

1. **The Llama adaptor is numerically correct on host** — 9 tests, 3 fresh
   processes. `reverse_permute` composed with the device's interleaved rotation
   is exactly HF's layout composed with `rotate_half`, checked against the real
   Llama-3.3 scaled rotary at `head_dim 128`; converted attention, MLP and
   LM-head weights reproduce the unmodified HF modules at PCC ≥ 0.9999. **Write
   the equivalent for Qwen before your first device run** — it costs a minute and
   it removes the whole "is it the weights or the mesh?" question from every
   subsequent silicon failure. Qwen's Q/K-norm and its own RoPE theta make this
   *more* valuable for you than it was for Llama, not less.
2. **A one-layer Llama model constructs, seals its prefetcher, resolves both CCL
   contexts, binds and unbinds a KV cache, and tears down cleanly on the mesh in
   109 s** with real layer-0 weights. Run once, reported as one pass.

And what runs but is not qualified: the decode graph executes through the
distributed norm, QKV matmul, RoPE on real Q/K, SDPA, `wo`, the attention
all-reduce, and all three MLP ring matmuls. **No output was ever compared to a
reference**, so nothing about numerical correctness on the mesh is claimed.

**L1 (global-CB ownership across two constructions) was never measured** — the
80-layer model was never built, and `test_two_models_in_one_process` exists but
never ran. Job 0's O5 stands exactly as it was.

## Suggested order for your night

1. Check the mesh is alive at all (§"Read this paragraph first"). If not, report
   `BLOCKED (infra)` and stop.
2. `test_partition_wh_galaxy.py` — 13 s, confirms the mesh and the partition.
3. Port Llama's `_relocate`/`_place` into Qwen's model, and check Qwen's
   embedding decode placement and residual dtype (§"Warnings specific to Qwen"
   items 1–3). All host-side, all cheap, all things that will otherwise cost you
   a device run each.
4. Write Qwen's host conversion-equivalence test (§"What is actually proven" #1).
5. Only then: one-layer construction and teardown on the mesh.
6. Expect D-B9 to block the decode step. Decide up front whether you are
   verifying the `in0_block_w` hypothesis or doing the ring conversion — and if
   the latter, budget the whole night for it, because it is a shard-spec change
   across the QKV matmul and the fused create-QKV-heads collective.

## Do not

* Do not trust the `in0_block_w` change until you have seen it run.
* Do not edit `models/common/modules/MILESTONE_A_STATUS.md` — job 0 and job 4 own
  it. Proposed L3 text is in this job's `REPORT.md` §4.5.
* Do not touch `models/common/modules/**/*_1d.py` or `models/common/llm_runtime/**`.
  Both greps are empty across this job's commits and should stay that way.
* Do not read a passing device result into anything in this package that does not
  say "passed" with a log next to it. There are exactly two.
