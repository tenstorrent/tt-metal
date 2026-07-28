# constant_synthesis — read a constant from DRAM vs. invent it on-core

**Difficulty:** ⭐ T1  ·  **Concept(s):** a constant-valued output needs no source bytes — synthesize it in L1 and replicate, instead of streaming a DRAM-resident constant through the reader.
**First profiled on:** `bgd-lab-t3002-special-dnijemcevic-for-reservation-33181` · WH (wormhole_b0) · ~1000 MHz · 2026-07-22 · `b32a86c71da`

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
You must materialize a large output region that is entirely one constant value. The obvious kernel
reads a DRAM-resident constant tensor and writes it back out — a straight relocation. But the read
is unnecessary: a constant carries no information, so you can generate the bytes on-core instead of
fetching them. The question is whether removing the entire read half of the pipeline is worth it,
and by how much.

## What this isolates — and how
- **Concept:** where the output's source bytes come from — a real DRAM read per page, or a single
  on-core synthesized template replicated to every page (zero source bytes read).
- **Isolation setup:** the *DRAM-read* row of the isolation rule. There is no compute (pure
  dataflow). The output is DRAM-interleaved (one page per row); page size, dtype, constant value,
  core count and placement are held constant across variants. Both variants use the identical
  writer NoC pattern — one whole-page async write per output page, `block` writes in flight per
  barrier — so the write half is common. The ONLY difference is whether the source bytes are READ
  or INVENTED.
- **Why it's kernel-level:** the author chooses the NoC access pattern (fetch a DRAM constant vs.
  build a template locally and fan it out). It is not a model/dtype choice — shape, dtype, page
  size and cores are identical across variants.

## The methods being compared
| Variant | What it does | Why it should differ |
|---|---|---|
| `stream_from_dram` *(baseline)* | Reader streams every output page from a DRAM-resident constant tensor (`block` reads in flight per barrier, double-buffered); writer writes each page out. | Real DRAM read traffic: the read half of the roofline paid in full for bytes that are all identical. Moves read+write bytes through the DRAM controller. |
| `synthesize` | Reader builds ONE output page of the constant in L1, once (a handful of local word stores, zero DRAM reads); writer replicates that resident template to every output page. | Zero source bytes read. Moves only write bytes through DRAM — half the traffic of the baseline. |

## CLI — measure your own shapes/params
```bash
python -m ttnn.operations.examples.constant_synthesis [options]
```

**Common flags (every perf-lab example):**

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--variant` | `{all,stream_from_dram,synthesize}` | `all` | which method(s) to run/compare |
| `--trials` | int | `20` | measured trials; report shows the average ns/op |
| `--iters` | int | `1` | in-kernel loop count — **1 = per-launch latency; large = steady-state throughput** |
| `--report` | path | *(print only)* | pytest writes the committed report; the CLI prints the table |

**Example-specific flags:**

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--shape` | `ROWS,W` | `4096,1024` | rows × columns of the constant-valued output (bf16, one page per row) |
| `--value` | float | `1.0` | the constant that fills the whole output |
| `--cores` | comma list | `1,<full grid>` | core counts to sweep (1 = read hidden by overlap; full grid = DRAM-bandwidth-bound) |
| `--block` | int | `8` | async reads/writes in flight per NoC barrier (cb_data = 2·block deep) |

**Example invocations:**
```bash
# A/B both methods across 1 and full-grid cores, on your shape
python -m ttnn.operations.examples.constant_synthesis --shape 8192,1024 --value 2.5

# just the candidate at the full grid, steady state
python -m ttnn.operations.examples.constant_synthesis --variant synthesize --cores 64 --iters 50
```

## Measured result
*Illustrative — see the **First profiled on** stamp above; re-run the CLI for your box.*

```
constant_synthesis   box=bgd-lab-t3002...  arch=wormhole_b0   trials=20 (avg ns/op)   iters=1 (per-launch)
  rows=4096  W=1024  page_bytes=2048  write_bytes=8 MB
    cores=1   method=stream_from_dram (baseline)  ...  539155 ns  (31.1 GB/s r+w)  ✓
    cores=1   method=synthesize                   ...  514569 ns  (16.3 GB/s w)     ✓  → 1.05×
    cores=64  method=stream_from_dram (baseline)  ...   87984 ns  (190.7 GB/s r+w)  ✓
    cores=64  method=synthesize                   ...   62098 ns  (135.1 GB/s w)    ✓  → 1.42×
```

The win **grows with output size** (64 cores): 1.31× @ 4 MB → 1.42× @ 8 MB → 1.45× @ 16 MB.

**Reading of the result:** removing the DRAM read is confirmed as a real win — `synthesize` is
faster purely because it reads nothing — but the gap is **~1.45×, not 2×**, and that ceiling is a
DRAM-subsystem property, not an implementation limit. The baseline is DRAM-**combined**-bandwidth-
bound: reads on NoC0 and writes on NoC1 saturate the controller at ~195 GB/s of read+write, so its
writes get only ~half the bus. The candidate moves half the DRAM bytes, but a **pure-write** stream
tops out at ~142 GB/s (block-insensitive from 8→32 — it is write-bandwidth-bound), below the
combined ceiling. Net `2 × (142/195) ≈ 1.46×`, matching measurement. **At 1 core the win nearly
vanishes** (~1.05×): DRAM is not the bottleneck, so the baseline's reads overlap its writes on the
separate NoC for free — a free read costs nothing to remove. The win is a DRAM-bandwidth-contention
effect that only appears once enough cores make the DRAM bus the shared wall.

**Possible follow-on (not measured here):** the *store width* used to build the L1 template —
per-element 2-byte stores vs. 32-bit word-replicated stores with head/tail element fixups. It is
invisible in this example because the template is built **once**, off the write-bound critical path;
a variant that rebuilds the template per page would surface it.

## Run the predefined sweep (regenerates report.md)
```bash
scripts/run_safe_pytest.sh --run-all ttnn/ttnn/operations/examples/constant_synthesis/test_constant_synthesis.py
```
(Correctness: `test_constant_synthesis_correctness`; measurement: `test_constant_synthesis_device_perf`.)
The test lives at `tests/ttnn/unit_tests/operations/examples/test_constant_synthesis.py`.
