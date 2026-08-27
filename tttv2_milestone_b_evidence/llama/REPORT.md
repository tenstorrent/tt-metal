# `mb-llama` — Milestone B steps 1–3, Llama-3.3-70B on WH Galaxy `(8, 4)`

Written 2026-08-27 by the `mb-llama` job, unattended.
Environment, exact commands and mesh facts: `ENVIRONMENT.md`.
Every log from every attempt: `logs/` (105 files, none overwritten).

## Verdict up front

| Plan step | Gate | Result |
| --- | --- | --- |
| 1 — adaptor, host | layout + Llama 3 scaled RoPE vs HF | **PASS**, 9 tests, 3 fresh processes |
| 1 — one-layer model on the mesh | construction, prefetcher sealing, CCL resolution, clean teardown | **PASS**, real layer-0 weights |
| 2 — one block, decode PCC ≥ 0.99 | PCC vs HF, KV-cache PCC | **NOT REACHED** — no PCC number was produced |
| 2 — one block, prefill 128 / 2048 | PCC vs HF | **NOT REACHED** |
| 3 — 80 layers, demo, teacher-forced accuracy | top-1 ≥ 91%, top-5 ≥ 99% | **NOT REACHED** |
| L3 verdict (attention decode on the prefetch partition) | — | **NOT CLOSED — still live.** See §3 |
| L1 verdict (global-CB ownership over two constructions) | — | **NOT MEASURED** |

**No PCC, accuracy or demo number exists in this package, because none was
produced.** The night was spent getting the decode graph to *execute* at all.
That was not the plan, and §2 explains exactly how far it got and why.

Two things ended the session:

* **nine defects** in the decode path, each blocking the next, all of the same
  family (§3). Eight are fixed and committed; the ninth has a candidate fix that
  is host-green and **never ran on hardware**;
* the mesh then **died** (§5): board 7 dropped off the PCIe bus and
  `tt-smi -glx_reset` cannot recover it because it needs that node. Recorded
  `BLOCKED (infra)` after the two recovery attempts the brief allows.

## 1. What is genuinely proven

### 1.1 Step 1, host — the adaptor is numerically correct

`models/common/tests/models/llama33_70b_galaxy/test_hf_conversion_host.py`, new,
9 tests, **passed in 3 fresh processes** (`logs/04`, `02b`, `03b`). No device.

The pre-existing host suite checked that every `weight_utils` conversion returns
the contracted *shape*. A Q/K permutation that pairs the wrong axes returns the
right shape and the wrong numbers, so shape was not the property that needed
proving. What is now proven:

* the Meta cos/sin tables are the HF rotary's **scaled** frequencies,
  pair-duplicated — and llama3 scaling really is applied (checked by showing the
  table moves away from the unscaled-theta table, so the check is not vacuous);
* **`reverse_permute` composed with the interleaved rotation the device kernel
  performs is the same operator as the HF weight layout composed with
  `rotate_half`**, at `head_dim = 128` against the real Llama-3.3 rotary. The
  rotation is not restated by hand: it is the `trans_mat` from
  `tensor_utils.get_rot_transformation_mat`, the matrix the kernel is handed, so
  the host reference cannot drift from the device one;
* `weight_utils.reverse_permute` is bit-identical to the qualified 1D suites'
  `_hf_reference.reverse_permute`;
* the fused row-major QKV packing is invertible per mesh row;
* converted attention, MLP and LM-head weights reproduce the **unmodified** HF
  modules at PCC ≥ 0.9999, with the LM-head padding columns proven inert;
* against the **real checkpoint**: the RoPE tables match Llama 3 scaling
  recomputed from its definition (`rope_theta`, `factor`, both frequency factors,
  original context length — no HF rope helper involved), and layer 0 converts to
  the contracted shapes.

**This closes the RoPE-convention half of the author's ranked risk #1.** The
other half — that the model wires those tables into `Attention2D` correctly on
the mesh — is device work, and §2 reports how far that got.

### 1.2 Step 1, device — the one-layer model builds and tears down

`test_bringup_wh_galaxy.py::test_one_layer_model_constructs_and_closes`
— **PASSED in 109 s** (`logs/10_construct_run1.log`), real layer-0 weights from
the real checkpoint. This is the first Milestone B code ever to run on hardware.

Proven: construction, prefetcher **sealing**, both operation-boundary CCL
contexts resolving, KV-cache bind/unbind, `close()`, and
`prefetcher.owned_resources == ()` afterwards. **The C1/D1 fused-norm statistics
placement holds at real scale** — it is a hard `ValueError` at construction if
re-introduced, and it did not fire.

It was run **once**. It is reported as a single pass, not as evidence under the
three-run rule.

### 1.3 The decode sub-device partition, measured

`models/common/tests/models/galaxy/test_partition_wh_galaxy.py`, new — **5
passed in 12.8 s**, no checkpoint (`logs/39_partition_probe_run4.log`):

```text
compute grid      x=0..6, y=0..9                              70 cores
worker_cores()    {[1-0 - 3-9], [5-0 - 6-9]}                  50 cores
prefetch senders  x=0 and x=4                                 12 cores
in NO sub-device  {[0-1 - 0-3], [0-6 - 0-8], [4-3], [4-8]}     8 cores
```

Two facts here caused most of §3, and neither is visible from a mocked mesh:

1. the worker envelope is **not contiguous** — the `x=4` sender column splits it
   — so its bounding box (`x=1..6`) is not a safe stand-in for it, and several
   ttnn ops use exactly that bounding box;
2. sender ∪ worker does **not** cover the compute grid, so any program built over
   the full grid touches cores owned by no sub-device.

This file is cheap and needs no checkpoint. **Run it first** when a decode
program aborts on placement.

## 2. How far the decode step got, and where it stopped

One decode step, batch 32, one real layer. 25 device processes opened the mesh
(`logs/` 06–89). Ordered by how far each got:

| Stage of `decode_forward` | Status | First proven in |
| --- | --- | --- |
| build / seal / CCL resolve / teardown | **runs** | `logs/10` |
| `activate("decode")`, persistent DRAM prefetch starts | **runs** | `logs/22` |
| `RotarySetup2D.decode_forward` (RoPE tables) | **runs** | `logs/22` |
| `Embedding2D.decode_forward` | **runs** | `logs/30` |
| layer-0 distributed RMS norm | **runs** | `logs/33` |
| QKV `ttnn.linear` | **runs** | `logs/40` |
| RoPE applied to real Q/K, SDPA, `wo` projection | **runs** | `logs/40` |
| attention output all-reduce | **runs** | `logs/80` |
| MLP ring `w1`/`w3`/`w2` matmuls | **runs** | `logs/80` |
| MLP shared axis-0 all-reduce | **blocked** → fixed (D-B8) | `logs/80` |
| back to the attention matmul: its CBs no longer fit L1 | **blocked** (D-B9) | `logs/82` |
| final norm, LM head, logits | **never reached** | — |

So: **RoPE composed with `Attention2D` runs on real silicon** — the pairing
Milestone A never qualified, and the author's predicted first failure. It failed
first, repeatedly, but for *placement* reasons, never for the numerical
convention, which §1.1 had already settled on host.

The last two rows are a coupled pair and are the honest stopping point:

* fixing **D-B8** meant widening the shared axis-0 all-reduce buffer from
  bfloat8_b to bfloat16 (34816 → 65536 B per core, on all 50 worker cores);
* that extra L1 is on `x=1..3`, which is precisely where **D-B5** had just
  confined the attention dense matmuls, and their in1 circular buffer then
  overflowed by about 20 kB.

The candidate fix for D-B9 — halving `in0_block_w` — is committed and host-green
but **was never executed on hardware**, because the mesh died first. It is
flagged in §3 and in the handoff.

**No case flipped across processes.** Every failure reproduced identically at the
same stage until it was fixed. The three-run rule was therefore never in
tension with anything: there is no intermittent result in this package, because
there is no *passing* device result beyond §1.2 and §1.3 to average.

## 3. Defects

All nine are first-silicon findings in Milestone B code or in shared 2D / Galaxy
plumbing. **None is in `**/*_1d.py` and none is in `llm_runtime/**`** (§6).

They share one root cause, worth stating once: **the Galaxy decode sub-device
manager owns only part of the compute grid, and several core ttnn ops choose
their cores from `device->compute_with_storage_grid_size()` or from a sub-device
*bounding box* — neither of which is the worker set.** tt-metal then rejects the
program:

```text
TT_FATAL ... Kernel group cores do not match sub device cores
             for programmable core type TENSIX
```

and, because the abort happens inside a multi-sub-device program, the mesh is
left un-drainable: teardown blocks forever in
`FDMeshCommandQueue::~FDMeshCommandQueue` and the process keeps the per-chip UMD
locks. Every one of these cost a kill and a `tt-smi -glx_reset`.

### D-B1 — `RotarySetup2D` built its prefill tables with an on-device clone

`models/common/modules/rope/rope_2d.py` (**shared 2D module**).
`load_device_weights` used `ttnn.clone` for the prefill cos/sin copies. It is
lazy, so on the decode path it first ran inside `decode_forward`, after the
partition was loaded, and `ttnn.clone` compiles over the full grid. Sealing the
prefetcher loads the decode partition *and leaves it loaded*, so there was no
later safe moment either — an eager call at construction aborted too
(`logs/18`).

Fix: write both copies from the same host source. Numerically identical — same
tensor, dtype, layout, DRAM placement — but a host-to-device write compiles no
program and is legal under any manager. This was the **only** lazy
device-weight loader among the 2D modules that ran a compute op; the other five
were audited and only write.

Evidence: `logs/17` (located it, `rope_2d.py:88`), `logs/20` (cleared it).

### D-B2 — the decode RoPE shards were placed on prefetch sender cores

`models/common/models/galaxy/recipes.py::rope_core_grids` (**shared Galaxy
plumbing**). It took the first `rows` cores of the *whole* compute grid, putting
the decode cos/sin shards on `(0,0)` and `(4,0)` — both prefetch senders — plus
one core outside every sub-device. `ttnn.embedding` then aborted.

Fix: delegate to `_subgrid_cores`, the already-qualified helper the attention KV,
SDPA and reduce-scatter placements use, which anchors at the first worker core
`(1,0)` and never leaves `worker_cores()`.

This is the **same defect shape as Milestone A D1/C1**: a grid named
independently of the partition that has to contain it. Every other placement
`resolve_galaxy_decode_placements` returns was audited — all worker-confined;
rope was the only exception.

Guarded on host: `test_recipes.py::test_rope_batch_grid_lies_inside_the_worker_sub_device`.
The old expression put **3 cores** outside; the new one puts 0.

### D-B3 — the Llama embedding decode output was L1-interleaved

`models/common/models/llama33_70b_galaxy/model.py`.
`decode_output_memcfg=ttnn.L1_MEMORY_CONFIG`. L1 *interleaved* round-robins over
the whole grid including the sender columns, where the ~55 MB global circular
buffer lives, so `ttnn.embedding` could not place its own static CBs:

```text
TT_THROW ... Statically allocated circular buffers in program 100 clash with
             L1 buffers on core range [0-0 - 0-0]
```

DRAM did **not** help (`logs/26`): `ttnn.embedding` takes its program grid from a
*sharded* output's shard grid and only from there. Fix: name
`decode.residual_memcfg`, which confines the program to the 16 worker cores that
placement already occupies and makes the following relocation a no-op. Tilized
width sharding is legal for this op — shard `[32, 128]` is tile-aligned and
divides the local hidden width.

### D-B4 — `RMSNorm2D` deallocated its own return value (latent, newly reachable)

`models/common/modules/rmsnorm/rmsnorm_2d.py` (**shared 2D module**).
`ttnn.to_memory_config` returns *the same* tt_metal tensor when the requested
config already matches, and nanobind hands that back as a **new Python
wrapper** — so `if tt_out is not unplaced_output: unplaced_output.deallocate(True)`
freed the buffer it was about to return:

```text
TT_THROW ... Tensor is not allocated     (in ShardedToInterleaved::validate_inputs)
```

`rms_norm_post_all_gather` already returns the tensor in `decode_output_memcfg`,
so the short-circuit always fired. Both call sites (input at line 273, output at
311) now guard on the memory config first — the idiom
`attention_2d.py::_place_qk` and the models' `_relocate` already use, and which a
docstring in `moe/tt_moe_decode.py` already warns about.

Pre-existing, not caused by this job; `_decode_distributed` had simply never run
on hardware. Evidence: `logs/30`.

Guarded on host:
`test_rmsnorm_2d.py::test_distributed_decode_does_not_deallocate_a_tensor_it_returns`.
It hands the norm tensors that already carry the requested configs - which is
what the hardware does, and where the short-circuit fires every time - and
asserts that neither placement is re-issued and nothing still in use is released.
**Verified to fail against the pre-fix code** (`logs/95`) and pass against the fix
(`logs/94`). The reason this stayed latent for a milestone is visible one test
above it: the sibling test mocks `to_memory_config` with
`side_effect=[distributed, output]`, which always returns a distinct object -
precisely the case where the identity test happens to be right.

### D-B5 — L3 is not closed: the attention decode matmuls still use the dense `(7,1)` grid

`models/common/models/galaxy/recipes.py::dense_matmul_program_config` (**shared
Galaxy plumbing**).

The brief states that "the Milestone B recipes build the partition-compatible
ring/`gather_in0` form instead" of the grid Milestone A recorded as terminal.
**That is true of the MLP and false of attention.** `attention_qkv_program_config`
and `attention_wo_program_config` are still `dense_matmul_program_config`, which
built a `(7, 1)` grid — *the exact grid L3 names* — anchored at `(0,0)`, spanning
both sender columns. On this build it aborts as:

```text
TT_FATAL ... Illegal kernel placement for bmm_large_block_zm_fused_bias_activation,
             Kernels cannot be placed on dispatch cores!
```

**So the L3 verdict is: still live, now with a precise diagnosis.**

Fix applied: populate `allowed_worker_cores`. ttnn grew that field for exactly
this, deprecating `compute_with_storage_grid_size`, and *warns* when a config
that supports it leaves it unset ("will become a hard error in a future
release"). The largest rectangle inside `worker_cores()` is searched for rather
than named, and is three columns wide.

**Cost: three worker columns instead of seven** for these two matmuls — and that
cost is what produced D-B9. The real fix is moving attention to the 24-core
ring/`gather_in0` form the MLP already uses; see §4.

Guarded on host and device:
`test_partition_wh_galaxy.py::test_dense_matmul_work_grid_stays_inside_the_worker_partition`,
four shapes, all confirming 0 non-worker cores (`logs/39`).

### D-B6 — the attention all-reduce used an op that cannot run on this partition

`models/common/models/galaxy/collectives.py::_all_reduce`.
It called `ttnn.reduce_scatter` + `ttnn.all_gather`, having first *validated*
that the keyed resource has a persistent buffer and then never passing it
anywhere.

`ttnn.reduce_scatter` cannot run here at all. Its program factory takes
`worker_cores(TENSIX, sub_device_id).bounding_box()` and lays its workers out
from that rectangle's origin — and our worker bounding box spans `x=1..6`,
straight across the `x=4` sender column. The file carries its own comment:
`// interaction with subdevice needs to be investigated`.

Fix: mode-split, mirroring the **qualified** `MLP2D._all_reduce_tg`.

* decode → `ttnn.experimental.all_reduce_async` with the resource's persistent
  buffer as `buffer_tensor`. That is the same call MLP2D makes for the same
  shared axis-0 resource — `build_galaxy_decode_collectives` says so in a comment
  — and the buffer's `(8, 4, TILE, W)` row-sharded shape *is* a `buffer_tensor`,
  not a scatter output, which is why the old code could validate it and use
  nothing. It rejects an interleaved input, so the interleaved `wo` output is
  width-sharded first with `interleaved_to_sharded`.
* prefill → keeps the plain pair, which is what MLP2D's prefill branch uses.

### D-B7 — `_relocate` reached a full-grid factory three different ways

`models/common/models/llama33_70b_galaxy/model.py`. All three obvious spellings
are unsafe under the partition:

| spelling | factory it reaches |
| --- | --- |
| `to_memory_config(t, memcfg, dtype)` | `ttnn::prim::copy` → `compute_with_storage_grid_size` (with a standing TODO to use worker cores) |
| `ttnn.typecast(t, dtype)` | same full-grid split |
| `to_memory_config(t, memcfg)`, shard specs differing in grid **and** width | `reshard_program_factory_generic` → full grid |

The residual grid (16 cores, 128 wide) → MLP ring grid (24 cores, 96 wide) hop
hits the third.

Fix: the explicit pair. `sharded_to_interleaved` runs on its input's
`shard_spec.grid`; `interleaved_to_sharded` runs on its output shard's cores;
both are worker-confined here, and both accept `output_dtype`, so the recast
rides along instead of needing an op of its own. **Cost: one DRAM round trip per
placement hop** — a real decode-latency cost, on the follow-up list in §4.

### D-B8 — the shared all-reduce buffer was allocated at the wrong dtype

`models/common/models/galaxy/plans.py`. The shared axis-0 all-reduce buffer took
`_spec`'s **bfloat8_b** default, while both Galaxy models set `MLP2D`'s
`decode_ccl_dtype` to their `decode_residual_dtype` — **bfloat16**, deliberately,
so an 80-layer residual sum is never re-quantized. `all_reduce_async` sizes its
circular buffer from the data and checks it against the buffer's L1 bank:

```text
TT_FATAL ... Cannot set circular buffer size to 65536. This is larger than the
             associated dynamically allocated L1 buffer bank size of 34816 B
```

34816 B is exactly a `[32, 1024]` bfloat8_b shard; 65536 B is the bfloat16 one.

Fix: `build_galaxy_decode_collectives` takes a `residual_dtype` parameter,
defaulting to `ttnn.bfloat16`, and sizes the buffer with it — a parameter rather
than a literal so a model with a different residual dtype can say so instead of
silently mismatching. Attention's reduction now runs at the same dtype, so both
consumers of the shared resource agree with it and with each other.

This one is worth reading twice: the two consumers of a *deliberately shared*
resource disagreed with the resource, and nothing on host could see it.

### D-B9 — OPEN: the confined attention matmul's circular buffers no longer fit

**Status: diagnosed, candidate fix committed, NEVER RUN ON HARDWARE.**

After D-B8 widened the shared buffer to bfloat16 (+30 kB of L1 on each of 50
worker cores), the attention dense matmul — confined by D-B5 to the three
columns `x=1..3`, which raised `per_core_N` by the same factor the grid narrowed
— overflows L1 on those cores by about 20 kB:

```text
TT_THROW ... Statically allocated circular buffers in program 320 clash with
             L1 buffers on core range [1-0 - 3-0].
             L1 buffer allocated at 546432, static CB region ends at 566464
```

Candidate fix in the tree: `in0_block_w` from `gcd(k_tiles, 8)` to
`gcd(k_tiles, 4)`, halving the in1 circular buffer (far more than the ~20 kB
shortfall) at the cost of more K iterations. **Host gate green
(`logs/92`, 390 passed). Device: unverified — the mesh died before it could
run.** Treat it as a hypothesis.

The structural answer is §4.1, not this.

## 4. Recommendations, in priority order

### 4.1 Move the attention decode matmuls to the ring/`gather_in0` form

This closes L3 properly and dissolves D-B9. The MLP already does it
(`ring_matmul_program_config`, 24 cores, `hop_cores`, `gather_in0=True`), and the
recipes already contain `attention_qkv_collective_input_memcfg` shaped for
exactly those 24 ring cores — so the design clearly anticipated it and the
attention matmuls were simply left behind. Spreading the work over 24 cores makes
`per_core_N` small and the circular buffers small, which removes both the
three-column penalty and the L1 clash.

Not attempted here: it changes the QKV matmul's input/output shard specs and how
the fused create-QKV-heads collective consumes them, and there was neither the
device time nor a working mesh to qualify it.

### 4.2 Populate `allowed_worker_cores` on the ring matmul config too

`logs/57` shows eight warnings per step:

```text
matmul_multi_core_reuse_mcast_1d_optimized_helper: program_config.allowed_worker_cores
not populated; auto-populating from compute_with_storage_grid_size. ...
This will become a hard error in a future release.
```

It currently works, so it was **not** changed — a qualified path should not be
altered on a night with no way to re-qualify it. But it is auto-populating from
the full grid, which is the exact hazard behind D-B2/D-B5, and it will stop being
a warning.

### 4.3 Reclaim the DRAM round trips D-B7 introduced

Every sharded→sharded relocation now goes through DRAM. Correct, and slower than
a reshard would be. Either the shard specs on either side of each hop should be
made compatible (so `reshard_program_factory_same_width` applies), or
`reshard_program_factory_generic` needs to respect the loaded sub-device.

### 4.4 Upstream: three ops are not sub-device aware

Worth filing against tt-metal, with these exact sites:

* `copy_default_tilized_program_factory.cpp:44` — uses
  `device->compute_with_storage_grid_size()`, with a TODO already in place;
* `reshard_program_factory_generic.cpp:80` — same;
* `reduce_scatter_program_factory.cpp:107` — uses the sub-device **bounding
  box**, which is wrong for any non-contiguous sub-device, and already carries
  `// interaction with subdevice needs to be investigated`;
* `typecast_program_factory.cpp:109` — full grid (the *sharded* typecast factory
  is fine; the fallback is not).

A non-contiguous worker sub-device is a normal Galaxy configuration, not an edge
case, so "bounding box" is a latent bug for every model that partitions its grid.

### 4.5 Proposed text for `MILESTONE_A_STATUS.md`

This job may not edit that file, so the proposal is here:

> **L3 — attention decode on the prefetch subdevice partition. Still open.**
> First tested on silicon by `mb-llama` (2026-08-26). The Milestone B recipes
> moved the *MLP* to the ring/`gather_in0` form but left both attention decode
> matmuls on `dense_matmul_program_config`, i.e. on the same `(7, 1)` grid this
> limitation names. On the current build it aborts as "Illegal kernel placement
> ... Kernels cannot be placed on dispatch cores" rather than as a sub-device
> mismatch. Confining it with `allowed_worker_cores` makes it legal but leaves
> only three worker columns, and its circular buffers then clash with the decode
> activations resident there. The remaining fix is the ring form.

## 5. `BLOCKED (infra)` — the mesh died

Full record: `logs/90_BLOCKED_infra_pcie_board7.log`.

```text
1. recurring:  Timed out waiting for ETH heartbeat on device
               ASIC ID 87032054158471220, ETH core e9-0 / e8-0
               - at mesh open, in TopologyDiscovery. Same ASIC every time.
               Cleared by tt-smi -glx_reset; returned after the next abort.
               Once it hung discovery outright instead of erroring (logs/70).
2. then:       Read 0xffffffff over PCIe ID 17: the board should be reset
               - in TTDevice::init_tt_device (logs/84)
3. then:       tt-smi -ls aborts inside tt_umd and lists zero boards (logs/87)
4. then:       tt-smi -glx_reset: Error in resetting galaxy 6u trays!
               [Errno 6] No such device or address: '/dev/tenstorrent/7'
```

The reset tool cannot recover the mesh because the node it needs is the one that
is gone. `/dev/tenstorrent/7` is still present as a character device but is
unreadable, and `dmesg` shows a kernel oops with `irqs disabled`.

**Recovery attempts used: 2 of 2** (`logs/85`, `logs/88`). Both failed. This
needs an IPMI power cycle of the tray or a host reboot, which is outside what an
unattended job may do. No device work was attempted afterwards.

The ETH instability correlated with a preceding `TT_FATAL` inside a
multi-sub-device program — which is what leaves the mesh un-drainable — so the
working theory is a fabric left dirty by aborted programs that the reset did not
always fully restore on that one board. Whether the ~23 resets this session
required *caused* the PCIe failure or merely preceded it is not something this
job can determine, and it is not claimed either way.

## 6. Regression gates and boundaries

Host gate, final code state, **driver-free selection** (`logs/96`):

```text
391 passed, 3 warnings in 87.18s     0 driver errors
```

Host gate, final code state, **standard selection** (`logs/91`):

```text
13 failed, 385 passed
```

All 13 failures are in `models/common/tests/models/galaxy/test_plans.py`, and all
13 are the dead driver: that file opens the UMD driver for all 32 chips when run
as a whole file (job 0 recorded this), so it cannot run at all now. 26 driver
error lines in that log. It was green at **398 passed** at 00:07, one code change
earlier, on a working mesh (`logs/81`). `test_plans.py` is therefore
**NOT RUN at the final code state**, by infrastructure failure, and is called out
rather than folded into a number.

Baseline at job start: **395 passed** (`logs/01`). Four tests were added to the
standard selection and ten to the driver-free one. **No test was deleted,
`xfail`ed, skipped, or had a threshold, tolerance or parametrization relaxed.**

Boundaries, `b350e51554470414d5a8b08f5ea9775c986145a4..HEAD`:

```text
$ git diff --name-only ... | grep '_1d\.py'       (empty - PASS)
$ git diff --name-only ... | grep 'llm_runtime'   (empty - PASS)
$ git diff --name-only ... | grep -i qwen         (empty - PASS)
```

Files changed (11): `galaxy/collectives.py`, `galaxy/recipes.py`,
`galaxy/plans.py`, `llama33_70b_galaxy/model.py`, `modules/rmsnorm/rmsnorm_2d.py`,
`modules/rope/rope_2d.py`, and five test files.

### Shared modules changed — declared, as the brief requires

| Module | Change | Why config alone could not express it |
| --- | --- | --- |
| `modules/rope/rope_2d.py` | prefill tables written from host instead of `ttnn.clone` | config has no say over whether materialization runs a device program |
| `modules/rmsnorm/rmsnorm_2d.py` | guard the two `to_memory_config` calls on the memory config before the identity test | a self-deallocation bug; no config value changes it |
| `models/galaxy/recipes.py` | rope `batch_grid` from `_subgrid_cores`; `allowed_worker_cores` on the dense matmul; smaller `in0_block_w` | the grids were computed, not configured; nothing upstream could pass a different one |
| `models/galaxy/collectives.py` | mode-split `_all_reduce`; decode uses `all_reduce_async` | the op itself was wrong for a non-contiguous sub-device |
| `models/galaxy/plans.py` | `residual_dtype` parameter for the shared all-reduce buffer | this *is* the config fix — the value was previously unreachable |

Extension-discipline order was followed: config first (D-B3, D-B8), frozen
config value second (D-B8's default), mechanical delegation to an existing
qualified helper third (D-B2, D-B5, D-B6). Nothing larger was attempted; §4.1,
which would be larger, is left as a recommendation and not half-done.

### A note on the logs, for `mb-signoff`

The 105 logs are committed, force-added past `.gitignore:7` (`*.log`). An
ordinary `git add` of the evidence directory silently drops them, which is why
job 0's 30 reconcile logs are present on disk in this checkout but untracked -
that is the ignore rule, not a missing deliverable. `trailing-whitespace` and
`end-of-file-fixer` were skipped for that commit: they rewrote 25 of the logs
(574 insertions, 574 deletions, all trailing spaces in pytest's output tables),
and evidence should be the bytes the tools emitted.

### Final mesh state, re-checked

At 2026-08-27T00:46Z, after the write-up: `/dev/tenstorrent/7` still unreadable,
`tt-smi -ls` still aborting with zero boards (`logs/93`). Still broken; no device
work was attempted after the `BLOCKED` call.

## 7. Commits

```text
3c1759ff20e  Make the Galaxy decode graph partition-safe: norm, matmul, all-reduce, relocation
1d4ded04592  Fix three Galaxy decode placement defects found on first silicon
e4c1adfa61f  Qualify the Llama Galaxy weight conversion and Llama 3 scaled RoPE on host
```

plus this evidence package and the work-log checkpoint. The `in0_block_w` change
of D-B9 is in the tree and is the one change here that hardware has never seen.

## 8. What this job did not do

Stated plainly so nothing is inferred from silence:

* **no PCC number** for a block, in either mode — decode never reached the LM head;
* **no KV-cache PCC**, no prefill 128 or 2048 result;
* **no 80-layer construction**, so **L1 (global-CB ownership) was not measured**;
  `test_two_models_in_one_process` exists but never ran;
* **no demo text**, no teacher-forced accuracy, no top-1/top-5 measurement —
  the accuracy gate this job exists to produce **was not measured**;
* `test_one_prefill_executes` was written but never ran: the decode step was
  ahead of it in the queue every time, and prefill shares the graph it was
  blocking on;
* the 1D device matrix was not run (Milestone A P4, separate hardware, not this
  job's).

Nothing in §8 is a judgement about whether those gates would pass. They were
not measured.

## 9. Out-of-scope items confirmed untouched

Paged KV, prefix-cached/chunked prefill, concat-32 physical batching, device
sampling and long-context smokes are plan step 7 and belong to `mb-coverage`.
Nothing was built for them. Every device test here ran with
`paged_attention_config=None` and `enable_device_sampling=False`.

**A dependency for `mb-coverage` to note:** step 3 was not blocked by anything in
step 7 — it was blocked by the decode graph not executing. No step-7 feature is
needed to reach the step-3 gates.

Qwen was not touched: no file under `models/common/models/qwen3_32b_galaxy/` or
its tests appears in the diff. Everything learned that applies to both models is
in `tttv2_milestone_b_briefs/job1_completion_handoff.md`, not in Qwen's code.

---
---

# Attempt 2 — 2026-08-27, on a recovered mesh

Attempt 1, everything above, ended `BLOCKED (infra)`: board 7 had dropped off the
PCIe bus and `tt-smi -glx_reset` could not recover it because the node it needed
was the missing one. **Between the two attempts the machine was power-cycled out
of band.** Attempt 2's first act was to re-check rather than trust the handoff,
and the mesh is healthy: `tt-smi -ls` exits 0 and enumerates all 32 Wormhole
boards including board 7 at `0000:08:0x`, and
`models/common/tests/models/galaxy/test_partition_wh_galaxy.py` passes 5/5 on
device with a clean open and close of all 32 chips
(`logs2/a2_00_partition.log`).

Nothing above this line is retracted. Everything above was measured, and the
partition numbers were re-measured on this attempt and are unchanged. What
changed is that device work became possible again.

The running account of attempt 2, run by run with every log named, is
`ATTEMPT2.md` in this directory. This section carries the verdict, the results
table, the defects and what is left open. Attempt 2's logs are in `logs2/`;
attempt 1's `logs/` were not touched.

## The single most useful sentence in this report

Attempt 1's ranked risk list was right about the *kind* of failure and wrong
about where it would stop. Every defect attempt 2 found is, again, the same
mistake:

> **A decode-mode program was placed on cores the loaded sub-device manager does
> not own, or was not told which sub-device it was running under at all.**

but attempt 2 found the second half of that sentence, which attempt 1 did not
reach: **several ttnn ops do not default to "the whole grid" when they are not
given a `sub_device_id` — they default to sub-device _zero_, which on this mesh
is the prefetch sender set, the one group of cores a compute program must never
use.** That is a strictly worse default than the whole grid, because the whole
grid at least contains the right cores. See D-B13.

## A2.1 What attempt 2 established on hardware

### D-B9 is fixed, and the fix was attempt 1's untested hypothesis

Attempt 1 left exactly one change in the tree that hardware had never seen:
`in0_block_w` from `gcd(k_tiles, 8)` to `gcd(k_tiles, 4)` in
`dense_matmul_program_config`, halving the in1 circular buffer of the
three-column attention matmuls. It **works.** The first decode run of attempt 2
(`logs2/a2_01_decode_step.log`) contains no `clash` at all — the

```text
TT_THROW ... Statically allocated circular buffers in program 320 clash with
             L1 buffers on core range [1-0 - 3-0]
```

of D-B9 does not occur — and execution proceeded through both attention
projections, the attention all-reduce, all three MLP ring matmuls, the MLP
all-reduce and the final distributed norm.

D-B9 is therefore **CLOSED**, and attempt 1's advice to "not trust the
`in0_block_w` change until you have seen it run" is discharged: it has run, four
times now, in four separate processes.

### The decode graph now executes a whole Llama layer and the final norm

From run 01's stage markers, which is the furthest any Milestone B decode had
reached at that point:

```text
build / allocate kv / bind kv                                   runs
activate("decode")  (persistent DRAM prefetch starts)           runs
RotarySetup2D.decode_forward                                    runs
Embedding2D.decode_forward                                      runs
layer 0: distributed norm, QKV, RoPE on real Q/K, SDPA, wo      runs
layer 0: attention all-reduce                                   runs
layer 0: MLP ring w1/w3/w2 and the axis-0 all-reduce            runs
final distributed norm                                          runs
LM head                                                         <- the frontier
```

So attempt 1's row "final norm, LM head, logits — never reached" is now "final
norm reached and executed; the LM head is where the remaining work is".

## A2.2 Defects found and fixed in attempt 2

Four, all in the LM head, all in the same family, each found by one device run
and each failing later than the last. Full evidence and quoted aborts in
`ATTEMPT2.md`.

| ID | Site | What was wrong | Status |
| --- | --- | --- | --- |
| D-B10 | `llama33_70b_galaxy/model.py::_relocate` | An **interleaved, non-DRAM** target fell through to `to_memory_config`, i.e. `ttnn::prim::copy` on the full grid. Latent for prefill too. | fixed |
| D-B11 | `llama33_70b_galaxy/model.py` LM head config | `decode_program_configs` resolved to `(None,)`, so ttnn auto-selected the full seven-column grid. Cannot use the dense config: decode has one row tile, so a 2D mcast matmul gets three cores and `per_core_N = 167`. | fixed — 24-core ring |
| D-B12 | `galaxy/recipes.py` LM head output placement | `ring_cores()` and `ring_receiver_cores()` are the *same 24 cores in a different order*, and `gather_in0` with a DRAM-interleaved in1 requires in0 and output on the same cores. Also: `decode_weights_memcfgs` was **dead config**. | fixed |
| D-B13 | `modules/lm_head/lm_head_2d.py` | `ttnn.linear` was never given a `sub_device_id`. The `gather_in0` factory intersects its ring with the named sub-device and **defaults to sub-device 0**, which here is the prefetch senders — disjoint from the ring, so the core set came out empty. | fixed |
| D-B14 | `galaxy/collectives.py`, `galaxy/plans.py` | `ttnn.all_reduce` forwards to the *buffer-less* `all_reduce_async` overload, which falls back to `composite_common::composite_all_gather` → `ttnn::concat` with no `sub_core_grids`, i.e. the full grid. `subdevice_id` is honoured by the fused path and ignored by the composite fallback. | fixed — keyed persistent buffer |
| D-B15 | `galaxy/plans.py` | That persistent buffer was allocated **resident in L1**. It is `GALAXY_COLUMNS` times the width of the logits — 129 kB per core at bfloat16 — and clashed with the decode activations' circular buffers. "Persistent" means the resource owns it across calls, not that it must sit in L1. | fixed — DRAM-resident, L1 view per call |
| ~~D-B16~~ | `modules/lm_head/lm_head_2d.py` | **Not a defect — struck.** I reasoned from the MLP's `decode_reduce_scatter_width` that a ring output reports its *padded* width, and widened the mask to match. Hardware says a matmul output keeps its **logical** width (16032); only a reduce-scatter output takes the padded one. Reverted, with the distinction recorded in the code. | reverted |
| D-B17 | `galaxy/recipes.py` | The reduce staging used the production's literal `num_cores_after_lm_head = 32`. A width-sharded L1 shard must be a whole number of tiles, and 32 does not divide Llama's 504. | fixed — core count derived |
| D-B18 | `llama33_70b_galaxy/model.py` precision recipe | The decode logits, and so the reduction buffer, were bfloat16 (`decode_activation_dtype`). The buffer is `GALAXY_COLUMNS` times the logits' width, ~96 kB/core, and clashed with the ring matmul's circular buffers on the cores they share. No core count fixes it: bfloat16 cannot go below ~82 kB. | fixed — `lm_head_output_dtype = bfloat8_b`, the production value |

D-B14's fix carries the one finding of the night that no amount of device time
would have produced: the production LM head all-reduce passes
**`fp32_dest_acc=True`**, with this comment against it —

> fp32 dest accumulation for the LM-head all_reduce only: its bf16 cross-device
> sum was order-dependent (ETH ring arrival order) -> per-row logit
> non-determinism -> greedy flips.

A bfloat16 cross-device sum of the logits is **not reproducible across runs**.
That is exactly the failure mode this project's three-runs-in-fresh-processes rule
exists to catch, and it would have presented as intermittent greedy-decode
disagreement rather than as a crash — the D1/D3 pattern again. It is set.

Also fixed, found by reading rather than by a run:

* **`galaxy_hardware.load_reference_tokens` returned a `(1, 1024)` tensor raw**
  while every consumer treats the sequence as flat, so `len()` was **1**. A
  caller asking for a 512-token prompt saw "reference sequence has 1 tokens" and
  **skipped**. The Milestone B accuracy gate could not have run, and it would
  have failed *open* — reported as a skip, not a failure. The 1D demo already
  squeezes at its own call site; this now happens once, in the loader.
* **`GalaxyColumnAllReduce` never passed `subdevice_id`** to `ttnn.all_reduce`,
  which forwards straight to `all_reduce_async`.

### The one precision value attempt 2 changed

D-B18 changed the decode LM head's output dtype from bfloat16 to **bfloat8_b**,
and that deserves to be stated plainly rather than left in a table, because
changing a precision to make something fit is normally exactly the wrong move.

It is defensible here for one reason: it moves *to* the qualified value, not away
from a gate.

* The production Galaxy LM head calls `ttnn.linear(..., dtype=ttnn.bfloat8_b)`
  for both modes and allocates `tt_lm_head_buffer` at bfloat8_b. The accuracy
  gates this milestone reuses — top-1 >= 91%, top-5 >= 99% from
  `tt_transformers` — were established against that.
* `Llama33_70BGalaxyPrecision.lm_head_dtype` already declared bfloat8_b for the
  LM head *weight*. The output now agrees, under its own named field
  (`lm_head_output_dtype`) rather than borrowing `decode_activation_dtype`.
* The **accumulation** is untouched: `fp32_dest_acc=True` on the all-reduce means
  the cross-device sum runs in fp32. Only the stored logits are bfloat8_b.
* Upstream's own unit test for this op reports PCC `0.99987` at bfloat8_b against
  its reference, comfortably above this job's `>= 0.99`.

The measurement is still the arbiter, and it has not been taken. Nothing here
claims otherwise.

## A2.3 The decode LM head, as it now stands

This is the design record, because it is the substantive engineering change of
attempt 2 and `mb-qwen` will need to make the same one.

**The decode LM head is a 24-core `gather_in0` ring matmul, on the same ring the
MLP uses.** It was never anything else in production: `_RING_CORE_COORDS` and
`_RING_RECEIVER_COORDS` in `models/common/models/galaxy/recipes.py` are,
coordinate for coordinate, `LM_HEAD_INPUT_GRID` and `LM_HEAD_OUTPUT_GRID` from
`models/demos/llama3_70b_galaxy/tt/model_config.py`, whose LM head is a 24-core
`gather_in0` ring at exactly this geometry (`LM_HEAD_RING_SIZE = 24`,
`LM_HEAD_TG_RING_PROGCFG`, `k = dim // 4`, `n = padded_vocab // 8`, and
`prefetch=False`). The Milestone B recipes copied the ring coordinates and wired
only the MLP to them.

Measured geometry, `dim 8192` / `vocab 128256` on `(8, 4)`:

```text
                 local_k  local_n  padded_n  in0_block_w  per_core_N  in1 CB
mlp w1/w3          2048     7168      7680        2           10       20 tiles
mlp w2             7168     2048      2304        9            3       27 tiles
lm_head            2048    16032     16128        2           21       42 tiles
lm_head (dense)    2048    16032        -         4          167      668 tiles   <- rejected
```

`local_k = dim / 4` because `LMHead2D`'s mapper is
`[PlacementShard(-1), PlacementShard(-2)]` over `(8, 4)`: mesh rows shard the
vocabulary, mesh columns shard the reduced hidden dimension. `local_n = 16032` is
`padded_vocab / 8`, and `pad_ring_width` takes it to 16128 so 24 cores divide it.

The four things that had to be true at once, none of which was:

1. **the program config** — `ring_matmul_program_config(local_dim, padded_local_vocab)`,
   because the dense config gives three cores and a 668-tile in1 buffer (D-B11);
2. **in0 and the output on the same cores, in the same order** — both
   `ring_cores()`. Not `ring_receiver_cores()`: same 24 cores, different order,
   and a `gather_in0` matmul with a DRAM-interleaved in1 compares those grids for
   equality (D-B12);
3. **`sub_device_id`** — the factory intersects the ring with the named
   sub-device and defaults to sub-device *zero*, the prefetch senders (D-B13);
4. **`num_global_cb_receivers = 1`** — the LM head is not prefetched, so it must
   not describe a global circular buffer that was never bound. The MLP keeps the
   qualified 2. This is now a parameter of `ring_matmul_program_config` rather
   than a literal.

Its in0 is the final-norm output at `local_dim`, which is the same width the MLP
feeds its ring, so the input placement is `mlp_input_memcfg`'s, unchanged and
already qualified on silicon. That is the one part of this that did not have to be
invented.

### Shared modules changed — declared, as the brief requires

Two, both outside the forbidden sets (no `*_1d.py`, no `llm_runtime/**`; both
greps are empty).

**`models/common/modules/lm_head/lm_head_2d.py`.** Three additions:
`decode_sub_device_id` / `prefill_sub_device_id` (callables, defaulting to a
`_no_sub_device` returning `None`), `decode_stage_mask` / `prefill_stage_mask`
(resolved bools), and the corresponding two lines in `_project`.

*Why config alone could not express it.* `LMHead2DConfig` had **no field** for a
sub-device id, and `_project` did not pass one to `ttnn.linear`. There was no
value any model could have set to fix D-B13 — the parameter did not exist. This
is the plan's first discipline step (config first) applied by *adding the config
surface the module was missing*, not by special-casing a model: `MLP2D` already
carries exactly this through `_prefetch_kwargs`, so the change makes the LM head
consistent with a sibling module rather than exceptional. The defaults preserve
present behaviour byte for byte for any caller that does not set them, which is
every caller other than Llama.

The mask staging is the same story one level down: the mask is one module-owned
*interleaved* DRAM tensor shared by both modes, and decode's output is now
sharded. The first attempt at this asked `output_memcfg.is_sharded()` inline and
broke two of the module's own tests, which drive `_project` with opaque sentinels
on purpose. The decision therefore moved to `_resolve_lm_head2d_config`, where
the real memory configs are, and arrives as a bool. **No `lm_head_2d` test was
modified; 20/20 pass** (`logs2/a2_08_lm_head_host.log`).

**`models/common/models/galaxy/collectives.py`.** `GalaxyColumnAllReduce` gained
an optional `subdevice_id`. Qwen constructs it without one and its behaviour is
unchanged.

`models/common/models/galaxy/recipes.py` also gained the four LM head decode
placement fields and the `global_cb_receivers` parameter; that file is Galaxy
topology, not a module, and Qwen picks the fields up without using them yet.

## A2.4 Test coverage added in attempt 2

All under `models/common/tests/models/llama33_70b_galaxy/`, all stating mesh,
checkpoint, mode, batch and sequence in their IDs or bodies.

**`test_model_wh_galaxy.py`** — the step-2 file. Three changes:

* **KV-cache PCC**, which the gate requires and the file did not have. The
  reference is HF's own cache: `hf(input_ids=..., use_cache=True)` and then
  `past_key_values.layers[0].keys / .values`, shaped
  `(1, n_kv_heads, sequence, head_dim)`. `K` is post-RoPE and `V` is the raw
  value projection, which is exactly what the device writes, so this is an
  independent reference rather than a hand-written re-implementation — the thing
  Milestone A found hides errors on both sides. Asserted after the 128 prefill
  **and** after the decode step, for all four prefilled user rows (0, 8, 16, 24),
  so a mesh column that silently wrote nothing is caught.
* **Prefill 2048**, as its own test rather than a parametrization, because the
  recipe family is keyed by sequence length: 2048 resolves a different attention
  program config, a different SDPA geometry and a different collective plan, and
  exercising those is the point.
* **`_one_layer_reference` now uses `load_layer_subset_causal_lm`** instead of
  `from_pretrained`-then-truncate. It reads 3 of the checkpoint's 30 shards,
  about 12 GB rather than 141 GB, so a fresh process costs seconds instead of
  ten minutes. Without this the three-runs-in-fresh-processes rule is
  unaffordable for the step-2 gate, which is precisely why attempt 1 wrote that
  loader.

Composing the KV cache needs care and the helper documents it:
`ConcatMesh2dToTensor(dims=(1, 0))`. The cache is allocated
`(32, n_local_kv_heads, max_seq, head_dim)` and mapped `dims=(None, 0)`, so mesh
*columns* hold disjoint users and mesh *rows* are allocated as replicas — but the
model writes a **different KV head** into each row. `to_torch_auto_compose` would
honour the declared replication and hand back one row's heads, silently dropping
seven eighths of the cache; the rows must be concatenated on the head axis.

**`test_bringup_wh_galaxy.py`** — one assertion strengthened. It asserted
`output.shape[-1] == padded_vocab_size` on the raw device tensor, whose shape is
the *per-device shard* width; the vocabulary is sharded over the eight mesh rows,
so the padded vocabulary is a property of the composed logits. It now composes
and asserts there, which fails if any row shard is missing or mis-sized — a check
a local-width assertion cannot make — and prints both shapes as evidence. This
was one of the file's declared hypotheses, written without a mesh.

## A2.5 Regression gates and boundaries

```sh
git diff --name-only 6a3e78a7227..HEAD | grep '_1d\.py'        # empty
git diff --name-only 6a3e78a7227..HEAD | grep 'llm_runtime'    # empty
```

Both empty. Also verified empty: any change to
`models/common/modules/MILESTONE_A_STATUS.md` or `tttv2_2d_modules_plan.md`
(neither is this job's to edit), and any added import of a model-named package
(`models.demos.*`, `models.common.models.llama33_70b`, `...qwen3_32b`).

**No Qwen file was touched.** `models/common/models/qwen3_32b_galaxy/**` is
absent from the changed-file list, as the brief requires. Qwen does pick up the
shared changes — the new `recipes.py` fields and the new `LMHead2DConfig` fields
— but every default is inert, so its behaviour is unchanged until `mb-qwen` wires
them.

Files changed against `6a3e78a7227`:

```text
models/common/models/galaxy/collectives.py
models/common/models/galaxy/recipes.py
models/common/models/llama33_70b_galaxy/model.py
models/common/modules/lm_head/lm_head_2d.py                    [SHARED MODULE]
models/common/tests/models/galaxy/galaxy_hardware.py
models/common/tests/models/llama33_70b_galaxy/test_bringup_wh_galaxy.py
models/common/tests/models/llama33_70b_galaxy/test_model_wh_galaxy.py
tttv2_2d_modules_milestone_b_work_log.md
tttv2_milestone_b_evidence/llama/{REPORT,ENVIRONMENT}.md, after_device_run.sh
```

### Host gate

`host_gate.sh`, the same selection attempt 1 used.

| Log | Result | Reading |
| --- | --- | --- |
| `logs2/a2_02_host_gate.log` | 2 failed, 559 passed | **Real**: both `test_lm_head_2d.py`, caused by a first draft of the mask staging that interrogated `output_memcfg` inline. Fixed by changing the code, not the tests. |
| `logs2/a2_03_lm_head_host.log` | 20 passed | `test_lm_head_2d.py` after that correction. |
| `logs2/a2_04_host_gate.log` | 13 failed, 548 passed | **Not code**: every failure is `test_plans.py` failing to open the cluster with `Timed out waiting for ETH heartbeat ... ETH core e2-0`, the dirty-fabric symptom. Cleared by `tt-smi -glx_reset` (`logs2/a2_05_reset.log`). |
| `logs2/a2_08_lm_head_host.log` | 20 passed | `test_lm_head_2d.py` after D-B13's fix. |

No test was deleted, `xfail`ed, skipped or relaxed, and no threshold was
touched. The one test *assertion* that changed —
`test_bringup_wh_galaxy.py`'s padded-vocabulary check — was made **stronger**, not
weaker: it now composes the mesh-sharded logits and checks the padded vocabulary
there, instead of asserting a global property of a per-device shard shape. Both
shapes are printed so the change is auditable from the log.

## A2.6 Upstream findings — ops that are not sub-device aware

Attempt 1 listed four. Attempt 2 adds two, both found on silicon, both with the
exact source site:

| Op | Site | What it does |
| --- | --- | --- |
| `ttnn.linear` with `gather_in0` | `matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp` | Intersects its ring with `device->worker_cores(TENSIX, sub_device_id)` and, given no `sub_device_id`, with `device->get_sub_device_ids().at(0)`. **The fallback is sub-device zero, not the whole grid.** |
| `ttnn.all_reduce` | `ccl/all_reduce/all_reduce.cpp` → `composite_common::composite_all_gather` → `ttnn::concat` | Forwards the `subdevice_id` to `all_reduce_async`, but the *buffer-less* overload falls back to a composite whose `ttnn::concat` receives no `sub_core_grids`. The sub-device is honoured on the fused path and dropped on the fallback. |

The first is the more dangerous of the two, and the reason is worth stating on its
own. Every other op in this family defaults to "the whole compute grid", which is
illegal on a partitioned mesh but at least *contains* the cores the program wants.
`gather_in0` defaults to **sub-device zero**, which on WH Galaxy decode is the
prefetch sender set — disjoint from any worker placement. The intersection is
empty and the failure surfaces as

```text
TT_FATAL ... Expecting a non-empty CoreRangeSet!   (program.cpp:1858)
  tt::tt_metal::CreateSemaphore(...)
```

which names neither sub-devices nor placement. Anyone debugging this from the
message alone will look in the wrong place. A default of "no sub-device means the
whole grid" would fail with the familiar `Kernel group cores do not match sub
device cores` and be diagnosed in minutes.

## A2.7 What the production reference was worth

Three of attempt 2's fixes could not have been derived from the failures alone,
and all three came from reading `models/demos/llama3_70b_galaxy/` — which the
house rules permit reading and forbid importing. Recorded because the next job
should read it *first*, not fifth:

1. **The LM head's ring is the recipes' ring.** `_RING_CORE_COORDS` and
   `_RING_RECEIVER_COORDS` are literally `LM_HEAD_INPUT_GRID` and
   `LM_HEAD_OUTPUT_GRID`. That turned "invent a placement" into "wire up the one
   that is already here".
2. **`buffer_shard_volume >= output_shard_volume * ring_size`.** This validation
   rule is written down nowhere except a comment in the reference's Qwen branch.
   It is why the LM head reduction stages onto 32 cores rather than the 24 the
   matmul used, and why `num_cores_after_lm_head = 32`.
3. **`fp32_dest_acc=True`, and why.** Quoted in full in §A2.2. A bfloat16
   cross-device sum of the logits is order-dependent on ETH ring arrival and
   produces per-row logit non-determinism — *greedy decode flips between runs*.
   No single passing run would have revealed it; three runs might have, at the
   cost of a night spent chasing an "intermittent" accuracy number.

Item 3 is the strongest argument in this report for the project's own
three-runs-in-fresh-processes rule, and also for reading the qualified
implementation before trusting a graph that merely executes.

## A2.8 The decode LM head as a checklist

For `mb-qwen`, and for anyone who has to do this again on another geometry. Seven
things must be simultaneously true. Getting six right still aborts, and each of
these cost one device run to learn:

```text
1  program config        ring_matmul_program_config(local_dim, pad_ring_width(local_padded_vocab))
                         NOT dense_matmul_program_config: decode has one row tile,
                         so a 2D mcast matmul gets 3 cores and per_core_N = 167
2  in0 placement         width-sharded on ring_cores() at pad_ring_width(local_dim)
                         (identical to mlp_input_memcfg -- already qualified)
3  out placement         width-sharded on ring_cores(), the SAME set and order as in0.
                         NOT ring_receiver_cores(): same 24 cores, different order,
                         and gather_in0 with a DRAM-interleaved in1 compares grids
4  global CB receivers   1, not the MLP's 2 -- the LM head is not prefetched
5  sub_device_id         passed to ttnn.linear. The default is sub-device ZERO,
                         which is the prefetch sender set
6  the reduction         all_reduce_async with a keyed persistent buffer, staged onto
                         the largest core count that divides the width in tiles,
                         with the buffer DRAM-resident and an L1 view per call,
                         and fp32_dest_acc=True
7  the resource key      the tensor's LOGICAL width (local_padded_vocab_size), not
                         the ring-padded physical width
```

Items 3, 5, 6 and 7 all default to something that is silently wrong on this mesh
rather than to something that fails loudly, which is why this took as many runs
as it did.

## A2.9 What is still open

**Attempt 1's open items, unchanged unless noted.**

* **L1 — global-CB ownership across two constructions.** Still never measured.
  `test_two_models_in_one_process` exists and has still not run: attempt 2 spent
  its device time on the single decode step, so the bringup file was never
  executed as a whole. Job 0's O5 stands.
* **L3 — attention decode on the prefetch partition.** Attempt 1 confined the two
  attention matmuls with `allowed_worker_cores` and left D-B9 open against it.
  **D-B9 is now closed and both matmuls execute**, so L3 is closed as an
  *execution* question at the cost attempt 1 named: three worker columns instead
  of seven. The performance answer — moving them to the 24-core ring, as the LM
  head now is — is unattempted and is now a much smaller job than it was, because
  the ring wiring exists and is exercised.
* **`allowed_worker_cores` unset on the ring matmul config** (attempt 1 §4.2).
  Deliberately untouched: it works, it warns, and a qualified path should not
  change on a night with no way to re-qualify it. Every log carries the warning.
* **The 64-head decoupled geometry still has no hardware evidence.** Llama's
  `n_heads * head_dim == dim`; nothing here changes that.

**New, and both are for whoever runs next.**

* **Step 3 cannot avoid paged KV.** `from_pretrained` computes
  `paged = paged_attention_config or default_paged_attention_config(params)`, so
  passing `None` selects the *default paged* geometry and there is no argument
  that selects a contiguous cache. Every 80-layer path — the full-model test, the
  accuracy gate, the demo — is therefore paged, while `job1_llama.md` assigns
  paged KV to step 7 and `test_model_wh_galaxy.py`'s docstring records paged
  decode as unqualified. `GalaxyDirectRunner` supports both; the loader does not
  expose the choice. **Recorded as a scope dependency rather than absorbed.**
* **`from_pretrained` loads the whole 141 GB checkpoint eagerly, once per
  process**, and each test in `test_full_model_wh_galaxy.py` calls `_load`
  separately — there is no shared-model fixture. For several 80-layer runs that
  cost dominates the session. `load_layer_subset_causal_lm` solved the equivalent
  problem for step 2; step 3 needs the same treatment or a module-scoped model.

## A2.10 The accuracy gate is now runnable — proved on host

Worth its own section, because it is the difference between a gate and a
placebo. `logs2/a2_24_accuracy_plumbing_host.log`:

```text
reference_tokens: (1024,) len = 1024   <- 1-D after the loader fix
top5_tokens     : (1024, 5)
prompt len = 512   targets = 511   aligned = (511, 5)
perfect prediction (reference argmax) -> (1.0, 1.0)
the reference's own target tokens     -> (0.9335, 0.9941)
```

Before the `load_reference_tokens` fix, `len(reference_tokens)` was **1** — the
loader returned the stored `(1, 1024)` tensor raw — so
`test_full_model_wh_galaxy.py::_reference_prompt(512)` raised

```text
pytest.skip("reference sequence has 1 tokens, need more than 512")
```

The Milestone B accuracy gate for **either** model could not have run, and it
would have reported a **skip**, not a failure. A gate that fails open is worse
than no gate, and this one was sitting in a file whose header says every threshold
is the plan's gate.

It now resolves 512/511 correctly, and a perfect prediction scores exactly
`(1.0, 1.0)` against `teacher_forcing_accuracy` — the self-consistency check that
says the scoring is measuring what it claims to. The second line is context, not
a ceiling: the reference sequence's own next tokens agree with the reference
model's top-1 93% of the time and its top-5 99% of the time, which is a property
of the text, while the gate scores the *model* against the *reference model's*
predictions.

No accuracy number is claimed for the Galaxy model. What is claimed is that the
apparatus for measuring one now works, which it demonstrably did not.

## A2.11 Method — what the shape of this night says

Ten device runs of the same one-step test, each aborting later than the last, each
at a different op, none reproducing a previously fixed defect. That is not a
struggle; it is what a correct diagnostic loop looks like when a graph has never
run. Three things made it work, and they are the transferable part:

**1. The `_stage` context manager.** Attempt 1 wrote it and attempt 2 relied on it
for every one of these. A `TT_FATAL` inside a multi-sub-device program leaves the
mesh un-drainable, so the pytest session never reaches its failure summary and the
traceback dies with the reaped process. The last `[stage] enter` line is then the
only thing that says which call aborted. In attempt 2 the Python traceback did
usually survive, but the stage line is what made the *log tail* readable at a
glance across ten runs.

**2. Reading the qualified implementation, not just the failures.** Three fixes
came from `models/demos/llama3_70b_galaxy` and could not have come from anywhere
else — §A2.7. One of them (`fp32_dest_acc=True`) prevents a defect that does not
crash at all.

**3. Refusing to batch speculative fixes.** Twice in this session a fix that
"obviously" also needed doing was wrong: the DRAM-width-sharded LM head weight
(silently discarded — D-B12) and the widened invalid-logits mask (the wrong
direction entirely — struck D-B16). Both were reasoned from a real precedent. A
device run costs about seven minutes and returns one exact fact; a speculative
batch costs the same run and returns an ambiguous one.

The cost is also worth recording honestly: **eight of the ten runs ended in a
`TT_FATAL` that left the mesh un-drainable**, so each needed a reap and a
`tt-smi -glx_reset`, and one of those resets timed out and wedged a chip's ARC
controller for ten minutes. About half the wall-clock of this session was mesh
recovery, not computation.

## A2.12 D-B19 — the one open defect

**Status: diagnosed to the layer, not to the op. Instrumentation committed; the
run that names the op has not happened.**

The decode LM head's axis-1 column all-reduce **hangs on device**. It does not
abort. The process stops producing output, is reaped at the deadline, and the mesh
needs a reset.

Diagnosed with `gdb` before spending a recovery attempt, per the house rules
(`logs2/a2_23_hang_gdb_dump1.log`):

```text
Thread 294:  SystemMemoryManager::completion_queue_wait_front
             FDMeshCommandQueue::read_completion_queue_event
             FDMeshCommandQueue::read_completion_queue          <- spinning, 151 s of CPU

Thread 1:    pthread_cond_wait
             FDMeshCommandQueue::wait_for_outstanding_reads
             FDMeshCommandQueue::finish_nolock
             distributed::Synchronize                           <- this collective's own finally
```

An enqueued device program never signalled completion. The blocked call is
`resources.synchronize("decode")` in
`GalaxyColumnAllReduce._persistent_all_reduce`, so the culprit is one of the three
ops that method enqueues: the DRAM→L1 buffer materialisation, the
`all_reduce_async`, or the placement of the result back onto the ring.

**A note on how nearly this was misdiagnosed.** The symptom is >100% CPU across
296 threads, which looks exactly like tt-metal JIT compilation — the benign
explanation, and the one I initially assumed. What ruled it out was cheap and
worth reusing: **no child processes, no new `.elf` artifacts, and no JIT cache
directory touched in three minutes.** Only after that did the per-thread CPU
accounting and the gdb dump become worth taking.

Ruled out by reading rather than by burning runs: the topology and link count
(6U Galaxy is `Ring`/4, and three other axis-1 decode collectives already use it
successfully), the semaphore count (the reference's decode
`gather_semaphore_handles` holds one global semaphore per slot, matching
`semaphores_per_slot = 1`), and the buffer sizing (equality in
`buffer_shard_volume >= output_shard_volume * ring_size`, exactly as the working
axis-0 buffer is sized).

What remains are two flag differences from the axis-0 all-reduce that *is* proven
on this mesh — it passes `use_optimal_ccl_for_llama=True` and this does not; this
passes `fp32_dest_acc=True` and it does not — and the possibility that a
relocation rather than the reduction is what hung.

**The next session should not guess between them.** `_ccl_trace` in
`collectives.py` prints and flushes a name before each device op the collective
enqueues, gated on `TTTV2_GALAXY_CCL_TRACE`, and `run_sequence.sh` exports it.
Because a hang leaves no traceback and no further log output, the last `[ccl]`
line is the only thing that can name the op. One run with it converts this from a
choice between three candidates into a fact.

## A2.13 Verdict

**The finish condition in `job1_llama.md` is NOT met.** Stated exactly:

| Required | Status |
| --- | --- |
| One Llama block qualified in decode and prefill at PCC >= 0.99 | **NOT MET** — no PCC number exists |
| KV-cache PCC >= 0.99 | **NOT MET** — the test now exists and has not run |
| 80-layer model producing coherent demo output | **NOT MET** — never attempted |
| Teacher-forced accuracy measured and recorded | **NOT MET** — the apparatus is now verified working (§A2.10); the number has not been taken |
| Handoff written | **MET** — `tttv2_milestone_b_briefs/job1_completion_handoff_attempt2.md` |

What *is* met, with a log for each, is the ground the two attempts have taken
together:

| Claim | Evidence |
| --- | --- |
| The mesh was recovered and is healthy | `logs2/a2_00_partition.log`, `logs2/a2_11_...` |
| The Llama adaptor is numerically correct on host (attempt 1) | attempt 1 §1.1, 9 tests, 3 processes |
| A one-layer model constructs, seals, resolves, binds and tears down (attempt 1) | attempt 1 §1.2 |
| **D-B9 is closed on silicon** | `logs2/a2_01`, and every run after it |
| **A whole Llama layer and the final norm execute at batch 32 on real weights** | `logs2/a2_01_decode_step.log` stage markers onward |
| **The 24-core `gather_in0` LM head matmul builds and runs** | `logs2/a2_14`, `a2_16`, `a2_21` |
| **The accuracy gate's apparatus works** (it previously could only skip) | `logs2/a2_24_accuracy_plumbing_host.log` |
| Host regression gate green on the changed modules | `logs2/a2_22_host_quick.log`, `a2_17`, `a2_18`, `a2_20` |

**One defect is open: D-B19**, the axis-1 LM head all-reduce hang, diagnosed to
three candidate ops with instrumentation in place to name it in one run (§A2.12).

### Why this is the honest verdict and not a pessimistic one

Attempt 1 stopped with the decode graph aborting at the *attention* matmul and a
dead mesh. Attempt 2 moved the frontier past attention, past the MLP, past the
final norm, through the LM head matmul and into its column reduction — seven
defects, each one a real placement or sub-device fault with a named site, plus one
that had to be struck when hardware refuted it. The remaining gap to the step-2
gate is a single hanging collective.

But "one defect from a PCC number" has been the shape of this milestone at every
stage, and the honest reading of ten runs is that each one revealed a constraint
nobody had written down. **Nothing in this report should be read as predicting
that D-B19 is the last one.**

---

# Attempt 3 — 2026-08-27

Attempt 2 ended with the decode graph reaching the LM head's column all-reduce and
hanging there (D-B19), with **no PCC number, no accuracy number and no demo text
in existence for this model**. Attempt 3's run-by-run account is `ATTEMPT3.md`;
its logs are in `logs3/`, and `logs/` and `logs2/` were not touched.

Nothing above this line is retracted. Two things in it are *superseded* and both
are named where they occur: attempt 2's `in0_block_w`-style reasoning about the LM
head width (§A2.8 item 1 was right about the placement and silent about the
tensor), and attempt 1's §4.2 note that `allowed_worker_cores` on the ring config
"currently works" - it does, and the prefetcher registration around it did not.

## Verdict up front

**The step-2 gate is MET.** One Llama block is qualified in decode and prefill
against an independent Hugging Face reference, with the KV cache checked on both
sides, at PCC >= 0.99, three times in three fresh processes with bit-identical
numbers.

| `job1_llama.md` finish condition | Status |
| --- | --- |
| One Llama block qualified in **prefill** at PCC >= 0.99 | **MET** — 0.99958 at 128 |
| One Llama block qualified in **decode** at PCC >= 0.99 | **MET** — 0.99975 at batch 32 |
| **KV-cache** PCC >= 0.99 | **MET** — K 0.99993, V 0.99975, after prefill *and* after decode, on all four column-local users |
| 80-layer model producing coherent demo output | *see §A3.6* |
| Teacher-forced accuracy measured and recorded | *see §A3.6* |
| Handoff written | **MET** — `job1_completion_handoff_attempt3.md` |

## A3.1 The step-2 gate, measured

Command, verbatim, run through `run3.sh` (which is `cycle.sh` plus the CCL trace,
a settable pytest deadline, and `logs3/` reset logs):

```sh
MB_DEADLINE=1200 MB_PYTEST_TIMEOUT=1080 \
  ./tttv2_milestone_b_evidence/llama/run3.sh <logname> \
  'models/common/tests/models/llama33_70b_galaxy/test_model_wh_galaxy.py::test_llama33_70b_galaxy_one_layer_prefill_and_decode' \
  -o faulthandler_timeout=600
```

Checkpoint: `meta-llama/Llama-3.3-70B-Instruct`, layer 0 only, read straight from
the safetensors shards by `load_layer_subset_causal_lm` so the same module
supplies both the TT weights and the reference logits. Mesh `(8, 4)`, 32 Wormhole
boards, firmware 18.12.1.

```text
[pcc] prefill 128:                          0.999584002863212
[pcc] prefill 128 cache K user 0/8/16/24:   0.9999347766610057   (each)
[pcc] prefill 128 cache V user 0/8/16/24:   0.9997498179150203   (each)
[pcc] decode position 128 user 0/8/16/24:   0.9997463458407887   (each)
[pcc] decode position 128 cache K u0/8/16/24: 0.9999342257320987 (each)
[pcc] decode position 128 cache V u0/8/16/24: 0.9997493345003990 (each)
```

All four column-local users report the *same* number to the last digit, which is
the property the four-user check exists to test: prefill filled local user 0 of
every mesh column, so a column that silently wrote nothing, or wrote something
else, cannot hide.

### Three runs in three fresh processes

The house rule exists because three of Milestone A's four defects presented as
intermittent *passes*. Here the three runs are not merely all green - they are
**bit-identical**, which is a stronger statement than "passed three times":

| Run | Log | Result | `prefill 128` | `decode 128 user 0` |
| --- | --- | --- | --- | --- |
| 1 | `logs3/a3_32_step2_gate_run1.log` | 1 passed, 147 s | 0.999584002863212 | 0.9997463458407887 |
| 2 | `logs3/a3_33_step2_gate_run2.log` | 1 passed, 149 s | 0.999584002863212 | 0.9997463458407887 |
| 3 | `logs3/a3_34_step2_gate_run3.log` | 1 passed, 151 s | 0.999584002863212 | 0.9997463458407887 |

Determinism to the last digit matters here for a specific reason. Attempt 2 found
that the production LM head all-reduce sets `fp32_dest_acc=True` against a comment
recording that a bf16 cross-device sum was *order-dependent* on ETH ring arrival
and produced per-row logit non-determinism. That flag is set in this tree, and
three identical runs are the evidence that it is doing its job. A drifting
low-order digit here would have been the D1/D3 pattern again.

## A3.2 Defects found and fixed in attempt 3

Seven, with a log behind each. Full evidence and quoted aborts in `ATTEMPT3.md`.

| ID | Site | What was wrong | Status |
| --- | --- | --- | --- |
| **D-B19** | `galaxy/recipes.py::galaxy_padded_vocab_size` | Attempt 2's open hang. The reduced logits were 501 tiles per device in a 42-core x 12-tile spec, so the 42nd core's shard was never full and `all_reduce_async`'s reduction kernel - `cb_in.wait_front(ring_size * block_num_tiles)` on *every* output core - waited for tiles the fabric would never send. No abort, no traceback, mesh reset. | **fixed** — vocab padded to a ring-exact width |
| **D-B20** | `modules/prefetcher/prefetcher_2d.py` | `seal()` allocated the global circular buffer at model build. 774 kB of unfreeable L1 per sender/receiver core made *every* prefill program that needs static CBs there unplaceable, starting with `ttnn.embedding`. Prefill never reads the buffer. | **fixed** — `defer_global_cb` |
| **D-B21** | `modules/rope/rope_2d.py` | The prefill RoPE table copy inherited decode's **row-major** layout. `rotary_embedding_llama` requires TILE; `ttnn.embedding` requires row-major. One legal layout per consumer and they differ. | **fixed** — the copy tilizes |
| **D-B22** | `modules/rope/rope_2d.py` | The prefill transformation matrix was `head_dim x head_dim`. The op applies it one tile at a time and validates `[-1] == TILE_WIDTH`; the helper's own docstring says "Must equal TILE_SIZE". A host assertion encoded the wrong shape. | **fixed** — `TILE_SIZE` |
| **D-B23** | `galaxy/direct_runner.py`, the step-2 test | The logits composed along the **wrong mesh axis**, and silently. A matmul output carries its *activation's* topology, not its weight's, so `to_torch_auto_compose` concatenated the four columns along the vocabulary axis: four copies of mesh row 0's slice. The runner then sliced `[:, :vocab_size]`, which narrows without raising. | **fixed** — `compose_galaxy_logits` |
| **D-B24** | the step-2 test | The KV reference was in the wrong RoPE convention. The device holds post-RoPE K in Meta interleaved order, HF in split order; the two cancel inside `Q.K^T`, so the logits agreed at 0.99958 while the caches scored 0.0386. | **fixed** — reference K permuted |
| **D-B25a** | `llama33_70b_galaxy/model.py` | `wqkv` and `wo` were registered with the prefetcher but their confined matmuls never read the global CB, so two entries per layer went unconsumed and the MLP's `w1` read the entry meant for `wqkv`. MLP PCC 0.096, and 0.085 as a function of its own input. | **fixed** — only the MLP's three are registered |
| **D-B25b** | `llama33_70b_galaxy/model.py` | The **non-fused** decode RoPE pair wrote a K of `|max| = inf` into the cache while V, which skips RoPE, was exact. Production selects the *fused* op whenever the prefetcher is active; the non-fused pair is the Blackhole fallback and wants a different cos/sin layout. | **fixed** — `use_qk_fused_rotary` defaults True |

Three of these - D-B23, D-B24 and D-B20 - would each have been enough on their own
to make the gate unmeasurable, and **two of the seven fail open**: D-B23 produced
wrong logits with no error anywhere, and D-B25a produced wrong numbers with no
error anywhere. Attempt 2 found the third of that family (`load_reference_tokens`
returning a length-1 sequence, so the accuracy gate *skipped*). That is now three
silent failures in this package in two nights, all in the measurement path rather
than the model, which is worth naming as a pattern: **on this mesh the apparatus
fails quietly more often than the graph does.**

## A3.3 Two claims attempt 3 discharged that were ranked as risks

`job1_llama.md` ranked four risks. Two are now closed as numerical questions, with
logs:

**Risk 1, "RoPE composed with `Attention2D` is the expected first failure".** It
was the first failure, and it was in the *pairing* exactly as predicted - but in a
way the prediction could not have named. The tables `RotarySetup2D` produces are
exact (PCC 1.0 against the adaptor's Meta-layout tables, on every column-local
user) and `rotary_embedding_llama` is a correct op. What was wrong was **which
op**: the composition called the variant meant for a mesh without a prefetcher.
See D-B25b.

**Risk 4, "fused decode norm at real scale; job 0 fixed the placement defect on
paper, this job runs it".** It runs, and it is right: the attention norm scores
0.9999956953474292 on an exact input and the FF norm 0.9311 on a 0.9435 input.
Job 0's C1 fix holds on silicon. This matters because C1 was described as making
every decode fail.

**Risk 3, L1/global-CB ownership**, is *half* closed and moved: the global CB is
no longer resident during prefill at all (D-B20), which removes the failure mode
attempt 1 predicted for a second construction in the same process during prefill.
Two constructions in one process is still unmeasured - `test_two_models_in_one_process`
has still never run - and the "prefill after a decode" case is a **new** instance
of the same limitation, stated in D-B20's own docstring rather than absorbed.

## A3.4 Shared modules changed — declared, as the brief requires

> "If you changed a shared 2D module to make the Llama model work, that is a
> significant event: name the module, the change, and why config alone could not
> express it... config first, frozen config value second, mechanical delegation
> third, and a written reduction before anything larger."

Three shared modules changed. Neither forbidden set was touched: `git diff
--name-only 45efb7c10e8..HEAD | grep '_1d\.py'` and `| grep 'llm_runtime'` are both
empty.

### 1. `models/common/modules/prefetcher/prefetcher_2d.py` — a new config value

**Change.** A `defer_global_cb: bool = False` field on `Prefetcher2DConfig`.
When set, `seal()` does not allocate the global circular buffer and the first
`activate("decode")` allocates it instead, before the prefetch program that reads
it is enqueued.

**Why config alone could not express it.** It *is* a config value, which is the
first rung of the extension discipline - but the value has to be able to change
*when* an allocation happens, and no existing field could. The module already
exposes a `create_global_cb` injection point, so the natural attempt is to inject a
lazy proxy; that cannot work, because `ttnn.dram_prefetcher(global_cb=...)` is a
nanobind boundary that needs the real object. The default is `False`, so the
Milestone A qualification of this module is bit-for-bit unchanged; only
`build_galaxy_prefetcher_config` opts in, and only because every Galaxy model here
prefills before it decodes.

**The one ugly part, named.** The buffer is bound onto the *sealed* decode context
with `object.__setattr__` on a frozen dataclass. That is deliberate and it is
documented at the call site: module configs capture the context *object* at
construction (`MLP2DConfig.decode_prefetch_context`, read as
`getattr(context, "global_cb", None)` at call time), so publishing a replacement
context would leave every already-built module holding `global_cb=None`. The field
is bound exactly once, `None` -> buffer, and never rebound. A host test covers all
three properties.

### 2. `models/common/modules/rope/rope_2d.py` — two corrections, not extensions

Neither is a configuration choice; both are cases where the module disagreed with
the op it feeds and the op is right.

* `_materialize_table_copy` now writes the prefill table **tilized**. Decode reads
  the table through `ttnn.embedding`, which requires row-major; prefill slices it
  and hands the slice to `rotary_embedding_llama`, which requires TILE. One legal
  layout per consumer, and they differ, so there is nothing to configure.
* the prefill transformation matrix is now `TILE_SIZE x TILE_SIZE`, not
  `head_dim x head_dim`. The op validates `trans_mat.logical_shape()[-1] ==
  TILE_WIDTH`; `get_rot_transformation_mat`'s docstring says "Must equal
  TILE_SIZE"; and the qualified 1D reference opens by discarding its argument
  (`dhead = 32  # ROPE op uses a single tile`). The module and its host test
  agreed with each other and both disagreed with the hardware.

### 3. `models/common/modules/lm_head/lm_head_2d.py` — a validation loosened

**Change.** `_resolve_lm_head2d_config` demanded the vocabulary be padded to
*exactly* the minimal multiple of `GALAXY_ROWS * TILE`. It now requires a multiple
of that, at least the minimum, and no more than one extra vocabulary shard per
mesh row.

**Why.** The old rule forbade the only width the decode chain can run.
`all_reduce_async`'s reduction kernel waits for a full shard on every output core,
so the reduced tensor's width must be an exact multiple of `cores * shard_width` -
and Llama's minimal padding leaves 501 tiles per device, which no usable core count
divides. This is a validation that rejected a legal geometry, not a threshold on a
result, and it is loosened in the direction hardware requires rather than relaxed
to make a failure green. The new upper bound is there so the check still fails
closed on a nonsense width.

### And one shared *test* helper, which the house rules explicitly permit

`models/common/tests/modules/_hf_reference.py::reverse_permute_1d` is now used by
the Llama Galaxy device test to put the reference K into the device's RoPE
convention. Sharing a test helper across the 1D and 2D suites "is fine and has
precedent" - that file *is* the precedent named in the README.

## A3.5 Model-level changes `mb-qwen` inherits

| Change | Why Qwen needs it |
| --- | --- |
| `galaxy_padded_vocab_size` pads to `GALAXY_ROWS * RING_ALIGNMENT` | Qwen's 151936 becomes 153600: 19200/device, 600 tiles, 50 reduce cores x 12. Without it Qwen hangs the way Llama did. |
| `use_qk_fused_rotary` defaults True | Qwen's decode RoPE will write an infinite K otherwise. Its 64-head geometry makes the head-row asymmetry *larger*, not smaller. |
| Only the MLP's projections are prefetched | Qwen's attention decode matmuls are confined for the same L3 reason, so it has the same unconsumed-entry problem. |
| `compose_galaxy_logits` | Qwen's logits compose along the same wrong axis, and just as silently. |
| `defer_global_cb=True` for Galaxy | Qwen prefills before it decodes too. |
| `from_pretrained(load_hf_model=...)` | Qwen's checkpoint is smaller but the three-runs rule still costs three loads. |
