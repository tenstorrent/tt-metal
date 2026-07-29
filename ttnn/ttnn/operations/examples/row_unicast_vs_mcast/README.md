# Row Unicast vs. Multicast — one row-wide transfer instead of per-peer writes

**Difficulty:** ⭐⭐⭐ T3  ·  **Concept:** Tensix-to-Tensix NoC fan-out topology
**First profiled on:** `bgd-lab-09-special-astancov-for-reservation-53762` · Wormhole B0 · 1000 MHz · 2026-07-28 · `b92eaf7ae25`

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem

Every core in a hardware row owns a payload that every other participating core in that row needs.
The direct implementation issues one unicast write per peer; the alternative emits one row
multicast. This example measures how the two transports scale with the number of cores in each row,
the per-sender payload, and the number of NoC writes used to dispatch that payload.

## What this isolates — and how

- **Concept:** unicast fan-out versus hardware multicast fan-out.
- **Isolation setup:** Tensix ↔ Tensix NoC — state is height-sharded in L1, eight independent rows
  run concurrently, row width is swept over 2, 4, and 8 cores, and there is no compute kernel or
  DRAM traffic.
- **Why it is kernel-level:** the kernel author chooses whether a sender emits one write per peer or
  one multicast transaction.

An exchange has one ordered sender round per participating core. In a round, one core sends its
payload only to the other cores in the row; its own slot is already resident in local L1. After all
rounds, every core holds the row's payloads in sender order. Both methods therefore move the same
logical data and produce bit-exact outputs for every row width and dispatch count.

`num_writes = D` splits each sender's fixed-size payload into `D` equal chunks. It does **not**
increase the total bytes. For a row of width `W`, each sender round therefore issues:

- unicast: `D × (W - 1)` NoC write calls;
- multicast: `D` NoC multicast calls.

## The methods being compared

| Variant | What it does | Why it should differ |
|---|---|---|
| `unicast` *(baseline)* | One explicit NoC write to every other core for each payload chunk | NoC command issue and payload injection are repeated for every peer |
| `mcast` | One row multicast to all other cores for each payload chunk | The sender injects each chunk once and the NoC replicates it across the row |

## CLI — measure your own payload and row shape

```bash
python -m ttnn.operations.examples.row_unicast_vs_mcast [options]
```

**Common flags:**

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--variant` | `{all,unicast,mcast}` | `all` | methods to run |
| `--trials` | int | `5` | measured trials; report shows median and population standard deviation |
| `--kernel-iters` | int | `100` | exchanges inside one kernel launch; `1` measures per-launch latency, larger values steady-state transport |
| `--report` | path | example `report.md` | report destination |

**Example-specific flags:**

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--num-rows` | int | all rows | number of contiguous hardware rows starting at `(0,0)` |
| `--row-width` | int list | `2 4 8` | one or more contiguous core counts per row; the full grid width is also included when different |
| `--num-tiles` | int list | `1 4 16` | bf16 tile payloads per sender: 2 KiB, 8 KiB, and 32 KiB |
| `--num-writes` | int list | `1 2 4 8 16 32` | equal-sized NoC dispatches per fixed-size payload |

**Example invocations:**

```bash
# Sweep fan-out, payload size, and dispatch count for both transports
python -m ttnn.operations.examples.row_unicast_vs_mcast \
    --row-width 2 4 8 --num-tiles 1 4 16 --num-writes 1 2 4 8 16 32

# Eight cores per row, one 32 KiB payload split into 1, 2, 4, or 8 writes
python -m ttnn.operations.examples.row_unicast_vs_mcast \
    --row-width 8 --num-tiles 16 --num-writes 1 2 4 8
```

## Measured result

*Illustrative — see the first-profiled stamp above; re-run the CLI for your box.*

First hold the 32 KiB payload at one dispatch and vary only row width:

| Cores/row | Unicast calls/sender | Mcast calls/sender | Unicast ns | Mcast ns | Mcast speedup |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 1 | 2,808.8 | 2,643.1 | **1.06×** |
| 4 | 3 | 1 | 14,337.6 | 5,882.4 | **2.44×** |
| 8 | 7 | 1 | 64,491.4 | 12,168.8 | **5.30×** |

With two cores, both variants issue one NoC operation per sender, so they are nearly tied. As the
row widens, unicast grows from one to seven calls and reinjects the 32 KiB payload for each peer;
multicast remains one call and one injection. That is the fan-out effect the benchmark is intended
to expose.

Next hold row width at eight cores and the payload at 32 KiB, then split the same bytes into more
dispatches:

| Dispatches | Bytes/write | Unicast calls/sender | Mcast calls/sender | Unicast ns | Mcast ns | Mcast speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 KiB | 7 | 1 | 64,491.4 | 12,168.8 | **5.30×** |
| 2 | 16 KiB | 14 | 2 | 67,670.3 | 15,345.2 | **4.41×** |
| 4 | 8 KiB | 28 | 4 | 74,722.7 | 21,521.6 | **3.47×** |
| 8 | 4 KiB | 56 | 8 | 88,986.4 | 33,976.0 | **2.62×** |

The payload remains 32 KiB in every row of this table. More dispatches make both methods slower
because each chunk repeats command and semaphore overhead. Multicast still issues seven times fewer
NoC calls than unicast at this row width, but its fixed per-dispatch receiver synchronization becomes
a larger fraction of time, so its relative speedup falls. See [`report.md`](report.md) for the full
row-width × payload × dispatch matrix.

## The crossover: sometimes unicast really is better

With two cores per row there is only one peer, so multicast cannot save any payload injections:
both variants issue exactly one NoC operation per chunk. That exposes the multicast protocol's
extra fixed cost as chunks become small.

We query TT-Metal's empirical NoC estimator for a sender round as:

- unicast: `ONE_TO_ROW`, `UNICAST`, `D` transactions, one transaction per barrier;
- multicast: `ONE_TO_ROW`, `MULTICAST`, `D` transactions, one transaction per barrier;
- L1, NoC0, same-axis row, one subordinate, sender excluded.

The Wormhole estimator table currently has `ONE_TO_ROW` measurements only for the full hardware
row: seven unicast peers and eight multicast destinations. Its interpolation therefore clamps this
two-core query to the nearest measured row point. Treat the estimate as a transfer-size trend and
crossover-search proxy, not an exact prediction for one peer.

For a fixed 2 KiB payload, the estimator and device measurement give:

| Dispatches | Bytes/write | Est. unicast ns | Est. mcast ns | Model pick | Device unicast ns | Device mcast ns | Device pick |
|---:|---:|---:|---:|---|---:|---:|---|
| 1 | 2,048 | 728 | 659 | mcast | 730.1 | 572.7 | **mcast** |
| 2 | 1,024 | 970 | 1,240 | unicast | 1,195.5 | 1,135.5 | **mcast** |
| 4 | 512 | 1,736 | 2,376 | unicast | 2,118.1 | 2,216.5 | **unicast** |
| 8 | 256 | 3,384 | 4,688 | unicast | 3,946.0 | 4,418.4 | **unicast** |
| 16 | 128 | 6,672 | 9,216 | unicast | 7,490.0 | 8,824.3 | **unicast** |
| 32 | 64 | 13,952 | 18,464 | unicast | 14,823.8 | 17,826.9 | **unicast** |

The estimator proxy identifies the right small-chunk regime but predicts the crossover one step
early: between one and two dispatches instead of between two and four. The remaining difference is
not surprising: fan-out is clamped to the nearest measured row, and the transfer-only ceiling
excludes this kernel's rotating-sender and semaphore protocol. Use it to choose candidate regions,
then use the device profiler to select the implementation.

The practical rule from the measured sweep is:

| Case | Winner | Reason |
|---|---|---|
| 2 cores/row, few large writes | multicast | its pipe and handshake are efficient enough to win narrowly |
| 2 cores/row, many small writes | unicast | there is no replication benefit to amortize multicast setup |
| 4–8 cores/row | multicast | avoiding 3–7 explicit peer writes dominates, even for 64–128 B chunks |

Re-run the estimator model after building TT-Metal's `noc_estimate` target:

```bash
python -m ttnn.operations.examples.row_unicast_vs_mcast.noc_model \
    --row-width 2 --payload-bytes 2048 --num-writes 1 2 4 8 16 32
```

## Run the predefined sweep

This regenerates [`report.md`](report.md):

```bash
scripts/run_safe_pytest.sh --run-all \
    tests/ttnn/unit_tests/operations/examples/test_row_unicast_vs_mcast.py::test_row_unicast_vs_mcast_device_perf
```

Run the standalone correctness gate with:

```bash
scripts/run_safe_pytest.sh --run-all \
    tests/ttnn/unit_tests/operations/examples/test_row_unicast_vs_mcast.py::test_row_unicast_vs_mcast_correctness
```

## Code

[`program_descriptor_with_inline_kernels.py`](program_descriptor_with_inline_kernels.py) contains
both inline data-movement kernels and their pure-Python `ttnn.ProgramDescriptor` construction.
