# handshake_elision — drop the CB protocol where nothing is ever in flight

**Difficulty:** ⭐⭐ T2  ·  **Concept(s):** the reader → CB → compute → CB → writer credit protocol as a fixed
per-launch and per-round cost, isolated from data movement — and the one situation in which a kernel may
legally skip it.
**First profiled on:** `bh-50-special-dstoiljkovic-for-reservation-88042` · BH · 1350 MHz · 2026-09-14 · `6ba9ac2fabc`

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
Your op's input shard is already resident in the core's L1, and its output shard lands in the same L1 with
the same spec. Both circular buffers alias the shard buffers, so the reader kernel has nothing to fetch and
the writer kernel has nothing to send — each is a single `cb_reserve_back` + `cb_push_back` or
`cb_wait_front` + `cb_pop_front` that publishes bytes already sitting at the CB address. The compute kernel
still runs the full per-tile-row protocol against them: wait, reserve, tilize, push, pop. Every one of those
calls is an L1 counter written by one RISC-V and polled by another, guarding a transfer that never happens.
This example measures what the protocol costs on such a path, and shows the form of the kernel that omits it.

## What this isolates — and how
- **Concept:** the synchronization structure around a fixed compute payload. One compile-time constant in
  the compute kernel, `no_handshake`, selects it; the tilize call, the two aliased CBs, the shard, the cores
  and the tile order are identical in both arms.
- **Isolation setup:** *pure compute* row — no DRAM, no NoC (both CBs alias resident L1 shards), a small
  fixed compute per tile-row (`tilize_block`), one shard per core. Work per core is swept from 2 to 64 tiles
  so a fixed cost can be told from a per-tile cost. Cores in row 0 only; a 1-core table confirms the effect
  is per-core.
- **Why it's kernel-level:** which CB calls the compute kernel makes, and whether the program needs the two
  dataflow kernels at all, are decisions of the kernel author.

## The methods being compared
| Variant | `no_handshake` | Kernels | What it does | Why it should differ |
|---|---|---|---|---|
| `handshake` *(baseline)* | 0 | 3 | Reader (NCRISC) publishes the input CB, writer (BRISC) retires the output CB, compute runs `cb_wait_front` / `cb_reserve_back` / `cb_push_back` / `cb_pop_front` per tile-row. The standard skeleton. | — |
| `no_handshake` | 1 | 1 | No dataflow kernels and no CB calls. Tile-row `r` is addressed as tile index `r*Wt` on both CBs via `tilize_block(cb_in, Wt, cb_out, r*Wt, r*Wt)`; the CB base pointers never move. | Nothing ever waits and no credit is ever posted or polled. Legal only because both CBs are exactly one resident shard deep, so every input page is valid before launch and every output page is free until after it. |

## CLI — measure your own shapes/params
```bash
python -m ttnn.operations.examples.handshake_elision [options]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--variant` | `all` or comma list of `handshake,no_handshake` | `all` | which arm(s) to run (baseline first) |
| `--shards` | comma list of `HTxWT` (tiles) | `1x2,1x4,2x4,4x4,4x8,8x8` | per-core shard geometry; tiles/core = HT·WT, one `tilize_block` per tile-row |
| `--cores` | int | `4` | cores in row 0, one shard each (tensor is `[HT·32·cores, WT·32]`) |
| `--iters` | int | `1` | in-kernel repeat of the shard — **1 = per-launch latency; large = steady-state** |
| `--trials` | int | `10` | profiled launches averaged per cell |

```bash
# the predefined sweep, per-launch
python -m ttnn.operations.examples.handshake_elision

# steady state, to see what survives once the per-launch cost is amortized
python -m ttnn.operations.examples.handshake_elision --iters 20

# one core, three geometries
python -m ttnn.operations.examples.handshake_elision --cores 1 --shards 1x2,2x4,8x8
```

## Measured result
*Illustrative — see the **First profiled on** stamp above; re-run the CLI for your box. Full tables,
including steady state and 1 core, in [`report.md`](report.md).*

```
handshake_elision   box=bh-50-...-88042  arch=BH  clock=1350MHz   mean of 10 launches   kernel-iters=1
  cores=4  placement=row 0, x=0..3   bf16 resident-L1 same-spec tilize, no NoC
  shard  tiles/core  variant       no_handshake  kernels      ns/op   ns/tile   ratio
    1x2           2  handshake                0        3      495.7     247.8   1.00x
    1x2           2  no_handshake             1        1      420.4     210.2   1.18x

    2x4           8  handshake                0        3      806.9     100.9   1.00x
    2x4           8  no_handshake             1        1      726.5      90.8   1.11x

    8x8          64  handshake                0        3     3570.1      55.8   1.00x
    8x8          64  no_handshake             1        1     3444.0      53.8   1.04x

  kernel-iters=20 (steady state), 2 tiles/core:  handshake 85.4 ns/tile · no_handshake 57.5 (1.49x)
```

**Reading of the result.**

1. **The protocol is a fixed cost, so it matters where the work is small.** Dropping it saves **~70–90 ns per
   launch** at 1–4 tile-rows and ~125 ns at 8: a ~70 ns floor (the first wait's dependency on the reader's
   publish, the two dataflow kernels' own entry/exit) plus a few ns per tile-row of `wait/reserve/push/pop`.
   That is 1.18× at 2 tiles/core and 1.04× at 64 — the saving does not scale with work, and the ~50 ns/tile
   tilize LLK swallows it once the shard is large.
2. **In steady state the protocol shows up per round, not per launch.** With 20 in-kernel iterations the
   per-launch cost is amortized and `no_handshake` runs at the LLK's own ~49–50 ns/tile, while `handshake`
   still pays **~55–65 ns per iteration** regardless of shard size (1.49× at 2 tiles/core, 1.03× at 64). A CB
   exactly one shard deep makes every iteration a dependent round through the credits: the reader's
   re-publish waits on the compute's last pop, the compute's first reserve waits on the writer's retire.
   Inside one iteration the protocol hides under the tilize; between iterations it is the critical path.
3. **The trick is tile-index addressing, and it needs nothing else.** `tilize_block` takes an input and an
   output tile index; with both CBs one shard deep and their read/write pointers parked at the base, row `r`
   is simply index `r*Wt` on each side. No `cb_*` call is issued anywhere in the program, so the reader and
   writer kernels are not thinned out — they are absent.
4. **Where this is legal, and where it is not.** Two facts must hold together: both CBs are exactly one shard
   deep, and every byte of the input is present before launch while every byte of the output is consumed
   only after it. The moment either side has to be moved by a dataflow kernel — a DRAM or remote-L1 source,
   an output that is written out, a CB shallower than the data — the compute kernel has something to wait
   for and the protocol is load-bearing. Drop the handshake only where it provably guards nothing.

The 1-core table reproduces every saving within ~2%, so nothing here depends on the grid.

## Run the predefined sweep (regenerates the numbers in `report.md`)
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_handshake_elision.py
```
Correctness (`test_handshake_elision_correctness`, 12 cases: both arms × smallest/largest/non-square shard,
1–5 cores, multi-iteration) is a bitwise gate against the row-major input; `test_handshake_elision_device_perf`
only measures.

## Code
- `kernels/he_compute.cpp` — **the one file to read**: `constexpr bool no_handshake` and the two
  `if constexpr` arms around an identical `tilize_block`.
- `handshake_elision.py` — the two variants, the aliased-CB construction, `create_program_descriptor`
  (adds the reader/writer kernels only for `handshake`).
- `kernels/he_arm_reader.cpp`, `kernels/he_drain_writer.cpp` — the baseline's publish and retire; each is one
  CB call pair per iteration and no NoC.
