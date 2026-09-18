# Quasar-native optimized `binary_ng` — research base and decisions

> **Committed for the record.** This is an engineering record of the Quasar-native `binary_ng` effort, not
> user documentation. Two kinds of reference in here point outside the repository and are expected to
> dangle for a reader who was not on the original branch:
> - `debug/attrib/*` — the diagnostic drivers, sweeps and plotting scripts. `debug/` is deliberately
>   untracked; the numbers they produced are reproduced inline here.
> - `.link_to_claude/plans/*` — the implementation plan, the specialist review findings, and the
>   measurement-discipline notes, which stayed out of the repo.
>
> These state current conclusions directly. Where a measurement protocol exists because getting it wrong
> was expensive, the protocol is stated as a requirement rather than as an incident.

Scope: build a **Quasar-native** (multi-DM, multi-Tensix, deep-ring, implicit-sync) execution path for
`binary_ng`, as opposed to the existing functional bring-up, which is a backward-compatible
single-threaded CB→DFB mirror that also runs on WH/BH.


---

## 0. Decisions already taken (do not re-litigate)

1. **Demonstrator case** = *interleaved (DRAM-fed) tensor-tensor no-broadcast tiled binary add, bf16*.
   Rationale and rejected alternatives in §6.
2. **Code placement** = a **second Quasar-native factory behind the existing `program_factory_t` variant
   seam**. The current `ProgramFactoryMetalV2` stays live as the functional fallback/reference; the
   descriptor `ProgramFactory` remains the general fallback. Details in §7.
3. Broadcast is **phase 2**, not the first target (§6.3).

---

## 1. The machine: one Quasar cluster ("node")

### 1.1 Engine budget and the hard ceilings

| Resource | Count | Where enforced |
|---|---|---|
| DM RISC-V cores (SiFive Rocket, 64-bit, **no FP**) | 8 physical, **2 reserved**, **6 user** | `tt_metal/impl/metal2_host_api/program_spec.cpp:47-50` |
| — DM0 | DFB implicit-sync ISR | `dataflow_buffer_config.h:63-64`, `:101-137` |
| — DM1 | tile-counter remapper programming | same |
| Tensix engines ("Neo") | 4, each with 4 TRISCs | `program_spec.cpp:50` |
| L1 SRAM | 4 MB **shared by all 12 engines** | `soc_descriptors/quasar_32_arch.yaml` (`worker_l1_size: 4194304`), `quasar/dev_mem_map.h:33` |
| Cluster grid | 8×4 = 32 clusters | `quasar_32_arch.yaml` `functional_workers` (rows 2-5 × cols 2-9) |
| DRAM views in that descriptor | 2 × 1 GB | `quasar_32_arch.yaml` |

Validation rules (all `TT_FATAL`, `program_spec.cpp`):
- DM kernel `num_threads ≤ 6`; summed over **all** DM kernels in a `WorkUnitSpec` (`:780`, `:1752`).
- Compute `num_threads ∈ {1, 2, 4}` — **3 is explicitly rejected** (`:761`); summed ≤ 4 (`:1746`).
- **At most one compute kernel per `WorkUnitSpec`** (`:1780`).
- Gen1 (WH/BH) forces `num_threads == 1` for both kernel kinds → any multi-thread design is Quasar-only
  by construction, and degenerates cleanly at `num_threads = 1`.

### 1.2 L1 is a 64-bank machine, not a flat scratchpad

From *Tensix NEO High Level Specification* (Confluence TA/84508873, § "L1"):
- **64 logical banks × 64 B wide**, each physically 4 × 16 B independently-accessible sub-banks.
- A **partitioned crossbar**: a given client can reach only **¼ of the banks per clock**. Requests queue in
  per-port FIFOs; the L1 may issue to sub-banks **out of order**, with a per-port **response-reorder
  buffer** restoring client order. A request can wait ~3 cycles for its crossbar turn.
- **Address hashing** interleaves address sequences across banks/sub-banks; configurable via CSRs.
  Consequence for us: ring base addresses and per-thread strides should *spread*, not collide.
- Optional in-order CSR modes exist per address range: `ALL_IN_ORDER`, `WRITE_ORDER`, `RW_BARRIER_ORDER`.
- L1 supports **atomics** from any client (RISC-V AMO, THCON, float accumulate).

Port budget per cluster (49 total, 26R + 18W + 5RW):

| Client | Ports (64 B) | R/W |
|---|---|---|
| Per Tensix: TRISC0-3 (via L0) | 1 | RW |
| Per Tensix: Unpacker0,1,2 | 5 | R |
| Per Tensix: Packer0,1 | 3 | W |
| ×4 Tensix | 36 | 20R + 12W + 4RW |
| Overlay (DM) | 1 | RW |
| **Overlay DMA (IDMA)** | 4 | 2R + 2W |
| **NoC read** | 4 | R |
| **NoC write** | 4 | W |

The 4 NoC-read + 4 NoC-write ports are the structural reason multiple DM cores help: one DM thread
cannot keep 4 read ports busy.

### 1.3 Synchronization fabric

- **Tile counters**: 32 per Tensix; **16-bit** counters (15-bit exposed differences). Two coherent
  copies (one in the AI clock domain in the L1 partition, one in the DM partition) to hide the CDC
  round-trip. DM cores can access the tile counters of **all** Tensix engines.
  (*Overlay Tile Counter Interrupt Protocol*, TA/408289306; NEO HLS § "Tile Counters".)
- 16-bit width is a real cap, but **not** on `ring_trisc_units` (checked as `uint32_t`). The real caps, all
  in `tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp`:
  - `capacity = num_entries / max(R,C) <= 65535` (`:1155-1161`)
  - `threshold = num_entries / num_txn_ids <= 255` and `per_txn >= 1` — **both `uint8_t`, both unguarded**
    (`:1047-1071`); the only `TT_FATAL` there checks divisibility, which `0 % anything == 0` passes.
    **The cliff is at `num_entries > 255`, not ~510.** `num_txn_ids` is the smallest `n ∈ [2,4]` satisfying
    `num_entries % (n · prods_or_cons · tcs_per_risc) == 0` **and falls back to 1** when none does
    (`:1092-1106`) — so any assumption of `n >= 2` (halving the cliff to 510) is unsafe. n=3 is reachable;
    only n=4 is not.
  - a **stricter** divisibility rule than the familiar `% max(R,C)` one:
    `num_entries % (num_txn_ids · prods_or_cons · tcs_per_risc) == 0` (`:1040-1044`). This is the one that
    bites at `entries_per_thread == 1`.
  - `ring_bytes <= unreserved L1` (`:1223-1229`) — but `validate_ring_extent` early-returns for DFBs with no
    Tensix endpoint (`:1174-1177`), so DM-only DFBs are unchecked, and `ring_bytes` is the single-TC-slot
    extent, not the whole ring.
  - **Not** `stride_in_entries <= 255` (`:1203`): it is `max(num_producers, num_consumers)` of two `uint8_t`
    fields (`dataflow_buffer.hpp:41,44`), so that check is dead code. Do not budget against it.
- **Packer → DM interrupt**: the packer increments `buff_tile_rcv` and raises an interrupt to the overlay
  RISC; the ISR can program the NoC. This is the hardware under DFB implicit sync.
- **Buffer Descriptor Table**: 32 entries per Tensix, holding base address, formats, XYZ dims, and
  **hardware-maintained read/write pointers**; programmed at init. This is why Quasar requires an
  `*_init` before every op use when DFB ids change (see §8).
- **Sync Unit** per Tensix: 16 semaphores + 8 mutexes + `STALLWAIT`, for intra-Tensix thread and
  execution-engine synchronization.
- 32 cluster-wide **general-purpose registers** readable/writable by NoC, DM and compute — lower latency
  than L1, an escape hatch if tile counters do not fit a pattern.

### 1.4 Tile-counter remapper = the multi-endpoint DFB hardware

From *Tile Counter Remapping Block* (TA/1401028761):
- 4 tile-counter update buses each way; 8 bus IDs total (DM buses 0-3 covering DM counters 0-15/16-31/
  32-47/48-63; Tensix Neo 0-3).
- A mapping is `(src, dest0..dest3, divide, direction, grp_ptr)`: **fan-out is at most 4 destinations per
  mapping** (`dfb::MAX_CLIENT_RS = 4`). Reverse mappings are implied automatically.
- `divide=true` splits a push of N across C consumers → **N must be a multiple of C**.
- `direction = PUSH` gives one-producer→many-consumers; `POP` gives many-producers→one-consumer.
- **Arena vs grouped allocation**: with arena allocation the producer cannot safely reuse a slot until
  effectively all consumers drain; **grouped allocation** tracks per-consumer pending updates so the
  producer is credited earlier (bounded by `GROUP_MAX_DISPARITY = 16`). This is a latency/occupancy knob
  that matters for small rings.
- 64 mapping entries, 64 grouped allocations.

Mirrored software limits (`tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_config.h`):
`NUM_DFBS = 32`, `MAX_PRODUCERS_PER_DFB = MAX_NUM_TILE_COUNTERS_TO_RR = 6`,
`NUM_TILE_COUNTERS_PER_TENSIX = 32`, `NUM_TENSIX_TILE_COUNTERS_FOR_DM = 16`
(`TC_TENSIX_POOL_START = 16`; the remapper can expose the upper pool to DMs),
`NUM_TXN_IDS = 4` per side, user txn ids `[0,7]`, DFB pool `[8,31]`, `MAX_TCS_PER_TXN = 18`
(worst consumer case = 4 ALL DMs × 4 producer TCs).

**Simulator ceiling below the platform's, and it fails silently:** craq-sim rejects any DFB needing more than
**6** tile counters per RISC (`craq-sim/src/tile.cpp:938`, `QSR_DFB_MAX_TILE_COUNTERS 6`) — and the rejection
happens inside a config-blob parser whose `false` means "not recognised as a DFB config", so the DFB is simply
never configured rather than diagnosed. The platform permits up to 16 (4 `ALL` DM consumers × 4 producers).
Phase 1 needs 4, so it is clear; this bites when widening the **consumer** side, i.e. the broadcast phases.

### 1.5 IDMA: a fourth data mover nobody at the op layer uses

4 IDMA engines do **L1→L1** transfers driven by DM cores, with:
- a **3-level hardware address generator** (infinite *face* loop → *outer* loop → *inner* loop, address =
  `base + outer + inner`), i.e. strided/2D/tile-walking patterns without RISC address math;
- **format conversion in flight** through unpacker/packer gaskets (fp32↔fp16a/b, ↔fp8p/r, int32↔int8/uint8;
  MX formats not supported through the gaskets), 16 B/port granularity.

Worked, runnable examples: `tests/tt_metal/tt_metal/data_movement/quasar_examples/quasar_idma/`
(`idma_basic_example.cpp`, `idma_1d_strided_example.cpp`) and `.../quasar_addrgen/` (1D, 2D, face-loop,
interleaved, im2col), each with a README containing address tables and diagrams.

### 1.6 Compute-side facts that do not exist on WH/BH

- **FPU and SFPU are independent pipelines.** TRISC3 was added specifically to run SFPU work in parallel
  with FPU work on threads 0-2, and Unpack2 + Pack1 + SrcS were added so the SFPU can unpack and pack
  **without sharing resources with the FPU** (NEO HLS `RTL-Tensix-6`, `RTL-Tensix-28`, `RTL-TDMA-14`).
  Verified data flows include `Unpack2→SrcS→SFPU→SrcS→Pack1` and `Unpack-to-dest→SFPU→Dest→Pack1`.
- **`UnpackToDest` is free on Gen2**: "there is no performance penalty for unpacking directly to Dest, so
  UnpackMode=UnpackToDest is the preferred mode for any SFPU-consumed data"
  (`compute_hardware_config.hpp:155-157`).
- Dest double-buffering (`double_buffer_dest`) and 32-bit Dest are per-kernel knobs
  (`compute_hardware_config.hpp:129-157`).
- Formats: Quasar **drops Bfp8/Bfp4** in favour of MX formats; `DataFormat::UInt32` is absent; Float16
  produces Inf/NaN rather than saturating; overflow clamps where WH/BH wrap.
  (*Tensix Formats*, TA/237174853, not read in full; corroborated by the ResNet bringup issue list.)

---

## 2. What the Metal 2.0 host API already gives us

### 2.1 `num_threads` is SPMD, and the DFB hardware does the partitioning

`kernel_spec.hpp:94-100`: one kernel = N independent threads, each running the whole `kernel_main()`,
each with its own thread index, coordinating explicitly. Mapping of threads to physical cores and the
number of compiled binaries is hidden.

`kernel_spec.hpp:129-144` — per-binding access pattern:
- **producers are always STRIDED** (`ProducerOf` comment: "All DFB producers are STRIDED");
- consumers may be **STRIDED** (thread i takes every N-th entry) or **ALL** (every thread sees every
  entry — the broadcast primitive);
- **BLOCKED is declared but rejected at runtime**.

Crucially, **compute kernels need no thread-id code**: each thread's DFB interface is pre-initialized
with its own tile counter and base/stride, so plain `wait_front(1) / copy_tile(id,0,0) / pop_front(1)`
addresses that thread's own slot. See `tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_consumer_2_0.cpp`
(which also documents the HW rule that an unpack instruction must sit between `wait_front` and `pop_front`).

**DM kernels do** need thread identity, because choosing *which tensor page* to fetch is theirs:
`hw/inc/api/kernel_thread_globals.h` provides `get_my_thread_id()`, `get_num_threads()`,
`sync_threads()`, `wait_threads(participants, barrier_idx)` — with **two barrier slots** (`[0]` producer
side, `[1]` consumer side) because a DFB's producer and consumer kernels can co-reside on one node with
different thread counts and a shared barrier would deadlock.
`TensorAccessor` already ships thread-aware iterators: "the calling DM owns shards i, i+N, i+2N, …"
(`hw/inc/api/tensor/tensor_accessor.h:276-295`, `:499-501`).

Canonical multi-DM producer idiom (`test_kernels/dataflow/dfb_producer_with_id_2_0.cpp`):
```cpp
const uint32_t tid = get_my_thread_id(), n = get_num_threads();
for (uint32_t i = 0; i < per_thread; ++i) {
    const uint32_t page_id = chunk_offset + i * n + tid;   // STRIDED
    noc.async_read<NocOptions::TXN_ID>(accessor, dfb, {.page_id = page_id}, {});  // implicit sync
}
dfb.finish();
```

**There is no per-thread argument channel.** `KernelSpec::compile_time_args` is
`Table<std::string, uint32_t>` (`kernel_spec.hpp:192-193`) — one value per KernelSpec — and
`RuntimeArgValues` is per-**node**. Neither is per-thread. So a thread's tile count must be *derived in the
kernel* from `get_num_threads()`, never passed in; taking the count from the API rather than a CTA also makes
a host/kernel mismatch structurally impossible. Both APIs work on DM **and** TRISC
(`kernel_thread_globals.h:55-77`, outside the `#ifndef COMPILE_FOR_TRISC` block at `:79`), with
`my_thread_id` `thread_local` per-Neo (`trisck.cc:32-33,92-94`).

**STRIDED is not a choice.** It is the default at every `num_threads` (`kernel_spec.hpp:143`) and the only
legal producer access pattern (`dataflow_buffer.cpp:1754`) — so no binding changes are needed and
`StridedConsumerOf` is never required.

### 2.2 Implicit sync is the Quasar-native dataflow contract

Two styles exist side by side (same test kernels):
- **implicit** — `noc.async_read<NocOptions::TXN_ID>(...)` / `noc.async_write<NocOptions::TXN_ID>(...)`:
  **no `reserve_back`, no barrier, no `push_back`**. The DM0 ISR posts the credit when the tagged NoC
  transaction retires, so many transactions stay outstanding.
- **explicit** — `reserve_back → async_read → async_read_barrier → push_back` (the WH/BH shape).

Rules and hazards:
- Opt-out is **DM-only**: `DataMovementGen2Config::disable_dfb_implicit_sync_for[_all]`
  (`data_movement_hardware_config.hpp:108-119`). **There is no compute-side opt-out** — a compute
  producer+consumer self-loop DFB must be structured correctly, not flagged off.
- Doing **both** (explicit CB ops *and* implicit sync on the same DFB) double-counts the 16-bit counter →
  `TILE_COUNTERS` fault, `mtval 0x1`.
- A DFB filled by many **sub-tile** NoC ops auto-posts a credit *per NoC op*, so posts outrun acks → hang.
- Known gap in the test matrix: **DM→DM `ALL` with implicit sync auto-skips** (documented runtime gap).
- Gen2 DM config carries no NOC/processor choice at all: "Gen2 architectures have a unified NOC and fully
  automated DM kernel core selection."
- **`dfb.finish()` is mandatory on the implicit arm and carries a thread rendezvous.** Credits arrive only
  from the ISR at `threshold` granularity, so a tail batch shorter than `per_txn` is never posted → hang.
  `finish_impl` (`dataflow_buffer.inl:252-261`) calls `handle_final_credits`, which does an **unconditional**
  `sync_threads(is_producer ? 0 : 1)` (`:390`) — and it is called under `if (ptiles_read_ > 0)` /
  `if (ctiles_written_ > 0)` (`:255,258`). ⇒ **a thread that issues zero transactions skips the barrier while
  its siblings block: hard deadlock.**

  **Scope correction, 2026-09-04 (F1).** The deadlock above is real but **implicit-arm only**, and the
  sentence that used to end this bullet — "even divisibility is what makes the drain safe; uneven counts
  must clamp thread count" — generalised it to the whole design and was **wrong**. `ptiles_read_` /
  `ctiles_written_` are incremented *only* by `commit_implicit_read`/`commit_implicit_write` (`:538`,
  `:572`), reached only from the implicit-sync overloads. Under **explicit** sync — what the native
  factory hardcodes — both stay 0, `handle_final_credits` is never called by any thread, and the barrier
  is never reached. F1 therefore needed no thread clamping: uneven counts, including threads that draw
  **zero** tiles, are bit-exact across the full matrix (status 09-03 slide 3). The hazard remains a live
  trap for M2.6, when implicit sync is enabled.
- **Barrier slots are a budget of 2, keyed by *role*.** `NUM_KERNEL_BARRIERS = 2`
  (`kernel_thread_globals.h:40-41`), invariant at `:36-39`: at most one producer-role and one consumer-role
  multi-thread DM rendezvous per worker. A reader(R>1) + writer(W>1) pair fits (producer/consumer). **Two
  producer-role multi-thread DM groups on one node share slot 0 and deadlock** — which is what a
  tensor-scalar writer (it is an `in1` *producer*) or a "split `in0`/`in1` across two reader kernels" design
  would create. Note the platform's own validation is per *DFB* (`program_spec.cpp:1250-1262`), so nothing
  forbids the configuration; the constraint is only in this comment.
- Whether the **explicit** multi-thread path also needs `finish()` is unresolved — no production Quasar DFB
  kernel calls it today (all explicit, single-thread), and `finish_impl` still runs an unconditional
  `all_acked` drain spin (`:262-266`). Settle it by experiment, not by reading.

### 2.3 Ring depth: `num_entries`, `capacity`, and what depth actually buys

A DFB is a **ring of `num_entries` slots, one tile each**. `num_entries` is the API field
(`dataflow_buffer_spec.hpp`); it is *not* what the hardware credits against.

**With multiple threads the hardware partitions the ring**, and what reaches the credit register is
`capacity = num_entries / max(producers, consumers)` (`dataflow_buffer.cpp:1139`, cap 65535 at `:1155-1161`).
Each thread gets `capacity` slots of its own, strided through the ring by
`stride_in_entries = max(producers, consumers)` (`:1140`). Two consequences:

- `num_entries % max(producers, consumers) == 0` is enforced (`:1133-1138`) — so a ring size legal at one
  thread count can be **illegal** at another.
- The per-thread buffering is `capacity`, not `num_entries`. A 4-slot ring shared by 4 producers gives each
  producer **one** slot.

**What depth buys, semantically: how far a producer may run ahead of its consumer before `reserve_back`
blocks.** `capacity = 1` means lock-step — fill a slot, wait for it to drain, fill it again. `capacity = 2` is
double buffering: fill one slot while the consumer drains the other. Deeper rings let the producer absorb a
slow or bursty consumer, and let NoC transfers stay outstanding rather than being waited on one at a time
(§2.2: in-flight is `entries_per_thread / num_txn_ids`, so depth 2 yields **one** outstanding read — which is
why depth and implicit sync are the same lever from two angles and neither works without the other).

**This is why the design exposes `entries_per_thread` rather than a global `ring_depth`**
(design §3.3): a single global number means *different things* at different thread counts — global depth 2 is
illegal at 4 producers, and global depth 4 at 4 producers/4 consumers yields `capacity = 1`, i.e. no buffering
at all. Sweeping a global depth therefore varies two things at once and the sweep is uninterpretable. The
per-thread knob holds "buffering each thread gets" fixed while thread counts vary, so
`num_entries = entries_per_thread × max(producers, consumers)` is *derived* per DFB.

**Measured value, and mind the platform:** on craq-sim, depth 1→40 is worth **1.02×** and asymptotes by depth
4 (§9) — but craq-sim performs the NoC transfer as a host `memcpy` inside the issue instruction, so there is
no latency to hide and that figure is a **lower** bound. Depth is a latency-hiding lever, so it is one of the
two the sim structurally undervalues (§10.6).

**How ring depth relates to DFB call batching `n`.** `capacity` is the budget; `n` spends it. A producer
thread holding `capacity` slots reserves `n` of them per `reserve_back` call, so:

- **`capacity >= n` is required** — you cannot reserve 8 slots out of 4.
- **`capacity >= 2n` is required for *overlap*.** At `capacity == n` the producer reserves its entire
  allocation, fills it, pushes, and must then wait for the consumer to drain all of it before reserving
  again — lock-step at batch granularity. Raising `n` without raising depth therefore *destroys* double
  buffering, and a batch sweep that does not hold `capacity / n` constant measures two things at once.

**And batching has a second effect that makes all three knobs one lever.** A batched producer issues `n`
transfers and then **one** barrier, so `n` transfers are outstanding where a per-tile loop has 1 (measured:
barrier cost falls as `1/n` — 9.0 → 4.5 → 1.1 cyc/tile at n = 1, 2, 8, measured on `kernels_dfb/`;
the rewritten `kernels_qsr/` reader measures **8.00**, so re-measure rather than reusing the constant).
That is latency hiding, i.e. a
poor-man's implicit sync. So depth, `n`, and `implicit_sync` are three facets of **one** quantity — *how many
tile transfers are in flight at once*:

| knob | what it controls |
|---|---|
| `entries_per_thread` → `capacity` | how many slots exist to receive in-flight data |
| batch `n` | how many transfers are issued before waiting |
| `implicit_sync` | removes the wait entirely — bounded only by `capacity` and `num_txn_ids` |

This explains the measurements better than treating them separately: on craq-sim all three are worth almost
nothing (1.02×, 1.08×, ≤1.10× — §9) **for one shared reason**, namely that in-flight concurrency cannot pay
when a transfer costs zero cycles. §2.2's formula is the same statement algebraically:
`in_flight_per_thread = entries_per_thread / num_txn_ids`, and at depth 2 in-flight is **1**, which is exactly
why implicit sync is *equivalent* to an explicit barrier there. ⇒ **On the emulator, sweep in-flight
concurrency as ONE axis rather than three independent levers** — cheaper and far more interpretable.

### 2.4 Other levers already in the API

- **Dynamic ring sizing per execution**: `ProgramRunArgs::DFBRunOverrides{.dfb, .num_entries}`
  (`program_run_args.hpp:116-133`). Stateful across executions; borrowed-memory DFBs re-derive their L1
  base from the tensor arg.
- **Sizing convention** from the test helper: `num_entries` = 16 rounded up to a multiple of
  `lcm(num_producers, num_consumers)` (`dfb_test_common.hpp:187-190`).
- `ScratchpadSpec` — private, unsynchronized per-node L1 working memory, allocated from the same region as
  DFBs (`scratchpad_spec.hpp`). Explicitly cautions that in multi-threaded kernels each thread is a
  different core.
- DFB entry-format metadata (`data_format`, `tile_format`, `unpack_face_geometry`) and
  `borrowed_from` (build the DFB on a tensor's resident L1) — `dataflow_buffer_spec.hpp`.
- Cross-node DFB is **sketched but not implemented** (`CrossNodeDataflowBufferSpec`), so cross-cluster
  broadcast still has to be NoC multicast or per-cluster reads.

### 2.5 Legality matrix, as executable documentation

`tests/tt_metal/tt_metal/api/dataflow_buffer/test_dataflow_buffer_base.cpp` enumerates which
(producer type, consumer type, count, access pattern) combinations are supported, with the reasons for
each exclusion in comments. Highlights relevant to us:
Notation is `<producers><access pattern> × <consumers><access pattern>`, where **S = STRIDED** (thread
*t* takes every *N*-th entry — a work split) and **A = ALL** (every thread reads every entry — a
broadcast). The enum is `AccessPattern {STRIDED, ALL, BLOCKED}` (`kernel_spec.hpp:138`); BLOCKED is
rejected at runtime and has no coverage — the `…BConfig` tests in `test_dataflow_buffer_configs.cpp`
are named `B` but pass `ALL` (`:1751-1756`), and `BLOCKED` appears nowhere under
`dataflow_buffer/`. Producers are always STRIDED (`kernel_spec.hpp:246`), so only
the consumer letter varies. **We are STRIDED on both endpoints of all three DFBs, so the `nSxmS` rows
are ours**; the `A` rows describe replication, which would give every Neo the whole tile stream. Why
that is forced for an elementwise op, and the one place ALL would belong instead (tensor-scalar):
design §4.1.1.
- DM→DM: `num_p + num_c ≤ 6`; `4Sx4S` is impossible (8 > 6).
- DM→Tensix: producers ≤ 6, Tensix consumers ∈ {1,2,4}; passing combos include `6Sx4A`, `4Sx4S`, `1Sx4A`.
  `4Sx4S` (`:89`) and `4Sx4A` (`:128`) are distinct tests, both present.
- Tensix→DM: Tensix producers ∈ {1,2,4}, DM consumers ≤ 6; passing combos include `4Sx2S`, `2Sx6S`,
  `4Sx4S`, `1Sx6S`.
- `6Sx4S` does **not** exist although `6Sx4A` does (`:131`), and the file's stated filters (`:59-70`) do
  not explain it: 6 DM producers and a 4-thread Tensix consumer clear every one. The STRIDED ratio rule
  does — `max(6,4) % min(6,4) = 2 ≠ 0`, while ALL partitions nothing and is unconstrained. Inferred, not
  stated by the file, but it is the same rule the native gate mirrors.
- Documented divergence: Tensix→DM `2Sx3S` (asymmetric non-divisible ratio) hits an M2-vs-legacy
  ring-slot mapping difference and is omitted. **Prefer divisible producer/consumer ratios.**

**Read "passing" carefully — for the DM→Tensix family it means the program ran, not that the data was
right.** `dfb_test_common.hpp:539-540` and `test_kernels/compute/dfb_t6_consumer_2_0.cpp:21` both state that
DM→Tensix L1 verification is omitted; the consumer `copy_tile`s into dest and discards it. The mirror
direction (Tensix→DM) *does* verify data, but its producer only `reserve_back`/`push_back`s a **host-prefilled
ring** (`dfb_t6_producer_2_0.cpp`) and never writes per-thread data. ⇒ **no test in the tree data-verifies a
multi-thread STRIDED producer writing its own slots** — i.e. the matrix is *not* evidence for the
producer-pairing invariant, only for liveness. Treat a "passing combo" as a legality fact and nothing more.

Phase-1 combinations, for the record: `in0`/`in1` are **`4S×1S`** (`DMTensixTest1xDFB4Sx1S`,
`test_dataflow_buffer_base.cpp:53`) and `out` is **`1S×2S`** (`TensixDMTest1xDFB1Sx2S`, `:94`). The `4S×2S`
often cited is the C=4 *target*, not phase 1.

---

## 3. Prior-art file map (read before designing/implementing)

| What | Where |
|---|---|
| Multi-endpoint DFB legality matrix | `tests/tt_metal/tt_metal/api/dataflow_buffer/test_dataflow_buffer_base.cpp` |
| DFB config sweep (98 KB of cases) | `.../dataflow_buffer/test_dataflow_buffer_configs.cpp` |
| 6-DM / 4-Tensix runs + per-role cycle harness | `.../dataflow_buffer/dfb_init_timing_bench.cpp` (`TT_METAL_MEASURE_DFB_INIT_TIME=1`; region layout in `dataflow_buffer_config.h:368-427`) |
| Borrowed-memory / alias / intra-Tensix / multinode DFB tests | `.../dataflow_buffer/test_{borrowed_memory,alias,…}_dataflow_buffer.cpp`, `test_dataflow_buffer_intra.cpp`, `test_dataflow_buffer_multinode.cpp` |
| **Multi-DM add, Metal 2.0, runs on sim** (closest existing template to our case) | `tests/tt_metal/tt_metal/test_multi_dm_add_two_ints.cpp` (`QuasarMeshDeviceSingleCardFixture.MultiDmAddTwoInts`; parameterized `num_threads` via a `make_dm_kernel_spec` lambda; in `quasar_sim_regresion_tests.yaml`) |
| **Multi-thread compute kernel on sim** | `tests/tt_metal/tt_metal/test_quasar_compute_kernels.cpp::QuasarComputeKernelMultipleThreads` (in `quasar_sim_regresion_tests.yaml`) |
| Multi-thread DM producer/consumer kernels | `tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_{producer_with_id,consumer}_2_0.cpp` |
| Tensix-side DFB producer/consumer kernels | `tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_{producer,consumer}_2_0.cpp` |
| Thread barrier test | `tests/tt_metal/tt_metal/api/test_kernel_thread_sync.cpp` |
| IDMA + address-generator examples (+ READMEs) | `tests/tt_metal/tt_metal/data_movement/quasar_examples/quasar_{idma,addrgen,im2col}/` |
| DM cache hierarchy + flush/invalidate API + doc | `tests/tt_metal/tt_metal/data_movement/quasar_cache/` (incl. `quasar_dm_cache_management.md`), `quasar_cache_perf/` |
| Metal 2.0 headers (ground truth, self-documenting) | `tt_metal/api/tt-metalium/experimental/metal2_host_api/*.hpp` |
| Host-side spec validation / processor assignment | `tt_metal/impl/metal2_host_api/program_spec.cpp` |
| Device DFB API | `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h` (+ `internal/tt-2xx/dataflow_buffer.inl`) |
| Thread globals / barriers | `tt_metal/hw/inc/api/kernel_thread_globals.h` |
| **Canonical multi-thread STRIDED producer loop** (copy this, incl. `break` guard + `finish()`) | `tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer_2_0.cpp:27-32,47` |
| **Device-side cycle bracketing prior art** (`TT_METAL_MEASURE_DFB_INIT_TIME=1`) | `tests/tt_metal/tt_metal/api/dataflow_buffer/dfb_init_timing_bench.cpp` |
| **Bound-kernel-identity readout** (routing proof, no env var, written every run) | `generated/inspector/kernels.yaml` |
| Profiler's device-side wall-clock read (works on DM cores) | `tt_metal/tools/profiler/kernel_profiler.hpp:218-225` |

Confluence: NEO HLS `TA/84508873`; Errata `TA/1802436609`; Overlay Tile Counter Interrupt Protocol
`TA/408289306`; Tile Counter Remapping Block `TA/1401028761`; Quasar Programming Quirks `LLK/2316533761`;
ResNet OP bringup `LLK/2608463913`; Tensix Formats `TA/237174853`.

---

## 4. Baseline: what the current `binary_ng` Quasar path does

`ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/`

- Three kernels, all **`.num_threads = 1`** (`device/binary_ng_metal_v2_factory.cpp:759`, `:807`, `:872`)
  → 2 of 6 user DM cores, 1 of 4 Tensix engines, ≈3 of 24 RISC processors.
- **Interleaved operands get a 2-entry ring** (`:583-587`); borrowed (sharded) operands back the DFB with
  the resident L1 shard.
- Reader does, **per tile pair** (`device/kernels_dfb/dataflow/reader_no_bcast_dfb.cpp:109-126`):
  `reserve_back(1) ×2 → async_read ×2 → async_read_barrier() → push_back(1) ×2`.
  One outstanding read pair at a time; no pipelining; explicit sync throughout.
- `num_tiles_per_cycle` is pinned to 1 unless **all** operands are borrowed (`:509-511`); subtile
  broadcast additionally forces 1.
- Writer is a single-thread DFB consumer; on the scalar path it doubles as the `in1` producer (fills once).
- 4 MB of shared L1 is essentially unused: ~6 tile-sized ring entries plus intermediates.

Functional status (from `QUASAR_PARITY_GAPS.md` and prior sessions): no-broadcast, subtile broadcast and
tensor-scalar pass on craq-sim; int32 stays on the descriptor;
(The collection on this HEAD is **243 tests** — no_bcast 88 + bcast 130 + scalar 25 —
with **zero** skip/xfail markers — use these per-file counts as the no-regression reference.)
several sim/LLK carve-outs tracked separately.

**Consequence:** every Quasar-specific resource the project cares about — 6 DM cores, 4 compute engines,
multi-threading, the shared 4 MB pool, implicit sync, IDMA, FPU/SFPU overlap — is currently unexercised
by the op. That is the gap this work closes.

---

## 5. Measurement plan (how "gain" is demonstrated)

> For what the simulator can and cannot actually measure — instruments, mechanisms, and three traps that
> produce wrong conclusions rather than missing ones — see **§10**.

### 5.0.1 BASELINE MEASURED — 2026-08-19 (steps 0, 1, 2 done)

Environment: branch `dchen/binary_ng_quasar_native` @ `0cf20188874`; craq-sim `/workspaces/sim` →
`/workspaces/craq-sim/src/_out/release_qsr/libttsim.so` (built 2026-08-18 21:47, craq-sim `5ced8886`);
`.build/default` with `ENABLE_TRACY:BOOL=ON`; `_ttnn.so` in sync. **Sim wall-clock ≈ 12-15 s per run.**

Invocation (from the test's own docstring, plus the profiler):
```
TT_METAL_SIMULATOR=/workspaces/sim/libttsim.so TT_SIMULATOR_LOCALHOST=1 ARCH_NAME=quasar \
CHIP_ARCH=quasar TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DEVICE_PROFILER=1 \
python -m pytest "tests/ttnn/nightly/unit_tests/operations/experimental/quasar/\
test_binary_ng_no_bcast.py::test_no_bcast_interleaved[post_relu=False-dtype_tt=DataType.BFLOAT16-op_name=add]" \
  --timeout=0 -q
```
Artifacts: `generated/profiler/.logs/profile_log_device.csv` (header
`ARCH: quasar, CHIP_FREQ[MHz]: 0, Max Compute Cores: 32`; rows carry RiscType `QUASAR_DM*` /
`QUASAR_NEO*_TRISC*`, ZONE_START/ZONE_END, `time[cycles since reset]`).

**Three independent, deterministic metrics:**
1. **craq-sim global clock** — printed at exit as `[<cycles>] <wall>s (<rate>)`
   (`craq-sim/src/sim.cpp:502-513`, `g_clock`). Free with every run.
2. **Device profiler** — per-RISC kernel spans. Summarizer: `debug/prof_summary.py`.
3. **craq-sim perf trace** — `TTSIM_PERF_TRACE=1 TTSIM_PERF_TRACE_PER_DISPATCH=1
   TTSIM_PERF_TRACE_OUT=<dir>` → `ttsim_perf_trace.tsv`: per-engine instruction counts, DFB op counts,
   and **stall cycles** (`sim.cpp:143-150`).

**Determinism: confirmed.** A repeat profiled run was bit-identical (7781 / 8019 / 7492 / 8036 / 7531;
sim clock 17934 both times). (Per-cluster values are *nearly* uniform but not identical — see the caveat below.)

**Profiler perturbation: +11.3%** — 16115 sim cycles without the profiler vs 17934 with it, same test.
⇒ **always compare profiled-vs-profiled.**

**Baseline resource usage (measured, confirms §4 from hardware counters):**
- Active RISCs: **`QUASAR_DM2` (reader) and `QUASAR_DM3` (writer) only** — DM4-DM7 never appear ⇒ 2 of 6
  user DM cores. `QUASAR_DM0` shows only its firmware zone (the ISR core).
- **`QUASAR_NEO0` only** — NEO1/2/3 never appear ⇒ 1 of 4 Tensix engines. Within NEO0, TRISC0/1/2
  (unpack/math/pack) are busy; **TRISC3 runs 16 cycles** — the SFPU thread is idle, so Gen2
  FPU/SFPU independence is entirely unused.
- All 32 clusters participate (work splits across the full 8×4 grid).

**Shape ladder (DRAM-interleaved bf16 `add`, profiled, via `debug/bench_binary_ng_shapes.py`):**

| tiles/cluster | total tiles | reader DM2 | writer DM3 | math TRISC1 | kernel span | reader cyc/tile | sim clock |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2  | 64   | 675   | 913   | 142   | 1467  | 337.5 | 9606  |
| 8  | 256  | 1797  | 2035  | 2052  | 2589  | 224.6 | 10926 |
| 20 | 640  | 4041  | 4279  | 4296  | 4833  | 202.1 | 13554 |
| 40 | 1280 | 7781  | 8019  | 8036  | 8573  | 194.5 | 17934 |
| 80 | 2560 | 15283 | 15521 | 15538 | 16075 | 191.0 | 26717 |

**Exactly linear: ~187 cycles per output tile marginal, ~300 cycles fixed.** Successive differences are
187.0 / 187.0 / 187.0 / 187.6 at every rung.

**⇒ Benchmark shape: 32×40 tiles (1280 total, 40/cluster) as primary** — ~15 s a run and identical to the
existing functional test's shape, so that test doubles as the perf case. **64×40 (80/cluster) as the
confirmation point.**

> **Caveat added 2026-08-28 — "within 4% of the asymptote" is a property of THIS config, not of the op.**
> The 4% holds for the 2-DM-core `metal_v2` baseline above (raw@40 194.5 vs marginal 187.0). The Quasar-native path
> has a far larger prologue — 767 to 1491 cycles against this baseline's ~300 — so its ramp extends much
> further, and at 40 tiles/cluster `4,4,2` sits **64% above** its asymptote (72.42 vs 44.12), not 4%.
> **Its span curve is still bending at 40**, which is exactly where the successive-difference check above
> would have caught it: `1,1,1` gives 174.30 over 20→40 but 176.50 over 40→60, 60→80 and 80→100.
> **Linearity is per-config. Re-run this rung check for every configuration before quoting a marginal**;
> fit the native path over 60/120/180. Carrying this shape forward without re-checking is what put a
> two-point slope across a bend into a week of Milestone 1 numbers.

**Decomposition — the pipeline is lock-step and the Tensix is starved:**
- Reader, writer and compute converge on the **same** ~187-195 cycles/tile, and the whole-kernel span
  (8573 at 40/cluster) barely exceeds any single stage (7781-8036) ⇒ the stages *do* overlap, but all
  advance at one shared rate.
- Perf trace at 40 tiles/cluster: **46,752 instructions vs 701,616 stall cycles** (math_stall 246,648;
  sem_stall 224,410; other_stall 230,558) — the Tensix is overwhelmingly waiting. DFB counts corroborate
  the per-tile lock-step: `cb_waits` 2560 (2/tile), `cb_reserves` 1280, `cb_pushes` 1280, `cb_pops` 2560;
  `unpack_instr` 2560, `pack_instr` 1280.
- **187 cycles/tile with 2/6 DMs and 1/4 Neos is the number to beat.**

**Hypothesis vs evidence — keep separate.** The data *establishes* the uniform 187 cyc/tile rate, the
starved Tensix, and the 2-DM/1-Neo usage. It does **not** prove the per-tile `async_read_barrier` is the
cause; a competing explanation is a fixed per-transaction cost in the sim's model. The first native
experiment should isolate that (implicit-sync reads alone, nothing else changed).

**Caveats discovered:**
- The perf trace's **NoC counters read 0** on this path (`noc_reads`/`noc_writes`/`noc_bytes`/
  `dram_*_bytes`/`l1_*_bytes` all absent) — **not** because the tracer is unwired (it is called from the
  `TT_VERSION == 2` NoC branch) but because Quasar DM kernels move data via ROCC command buffers that never
  call it; see §10.4. Count NoC
  transactions analytically, or try `TT_METAL_DEVICE_PROFILER_NOC_EVENTS` (still unverified on Quasar).
- **Per-cluster values are not identical.** Real spreads: `QUASAR_DM3`
  8019→8031 (12 cycles), each `NEO0_TRISC0/1/2` 6, plus **24-cycle inter-core start skew**. The
  no-contention conclusion still holds but must be sourced, not inferred: craq-sim has no
  bandwidth/arbitration/queue model anywhere (only `eth_latency_cycles`, default 0; `libttsim.cpp:1961`
  prints `dram_in_flight=0` as a literal). Stronger still: it performs the NoC transfer as a host `memcpy`
  inside the issue instruction, pre-satisfies read barriers, and runs at IPC=1 — so **implicit sync and
  ring depth are unmeasurable here**, and cycles/tile ≈ per-core instruction count **on the DM cores**
  (strictly 1 instr/cycle). Not on Tensix: the default RTL-aware scheduler retires up to 3 backend
  instructions per cycle, so Tensix counts are compressed against the clock — see §10.3.
- Only *ethernet* latency is modelled (`eth_latency_cycles`, default 0); no DRAM in-flight model found
  (`libttsim.cpp:1961` prints `dram_in_flight=0` as a literal).
- **Do not put pytest files under `debug/`** — a stale `debug/conftest.py` re-registers `--tt-arch`, so
  collection dies with `ValueError: option names {'--tt-arch'} already added`. The ladder driver is a
  plain script using `ttnn.open_device` for that reason.

Harness (gitignored scratch): `debug/prof_summary.py`, `debug/bench_binary_ng_shapes.py`.

### 5.0.2 PER-ROLE OCCUPANCY at `4,4,2`, and which roof it lands on — 2026-09-04

Measured on anchor `8e3f13a177b` with craq-sim `ad401613`, interleaved bf16 `add`, 32 clusters, pinned
cycle model (`RTL_AWARE_SCHEDULER=1`, `DEFAULT_LINGER=1`, the six vars §10.3 says to record). Harness
`debug/perf/util_rcw.py`. **Two instrument checks passed before any number below was read:** its
direct-op runner reproduces the fitted pytest node *bit-identically* (2909 cyc at 40/cluster), and the
fitted marginal lands on the recorded value (below).

**The role map, read off the profiler rather than assumed.** §5.0.1 only ever named DM2/DM3 because the
baseline used two DM cores. At `4,4,2` the 6 user DM cores split by declaration order:

| RISC | role | note |
|---|---|---|
| `QUASAR_DM0` | DFB ISR | `DM0-FW` zone only, no kernel |
| `QUASAR_DM1` | remapper | no zones |
| `QUASAR_DM2-DM5` | the 4 **readers** | `R=4` |
| `QUASAR_DM6-DM7` | the 2 **writers** | `W=2` |
| `QUASAR_NEO{0-3}_TRISC0/1/2` | unpack / math / pack | all 4 Neos identical to within 45 cyc |
| `QUASAR_NEO{0-3}_TRISC3` | **idle — 18 cycles** | a spare RISC per Neo, in every config |

**Occupancy converges from below, so a single shape misreports it.** Occupancy is a role's `*-KERNEL`
span over the cluster's envelope across all its kernel zones:

| tiles/cluster | span | cyc/tile | reader DM | writer DM | unpack | math | pack |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | 2909 | 72.72 | 65.5% | 78.9% | 78.1% | 79.5% | 62.0% |
| 80 | 4635 | 57.94 | 78.5% | 86.8% | 87.0% | 87.1% | 76.1% |
| **160** | **8155** | **50.97** | 87.8% | **92.5%** | **92.6%** | **92.7%** | 86.4% |

This is the §5.0.1 caveat showing up in a second quantity: at 40/cluster the span is 64% above its
asymptote (72.72 here vs the 72.42 recorded there) **and** occupancy is 13 points below its. Quote
occupancy at ≥160/cluster or not at all.

```
4,4,2 at 160 tiles/cluster -- share of the 8155-cycle cluster span

  math    TRISC1   |############################################| 92.7%  <-- binds
  unpack  TRISC0   |############################################| 92.6%
  writer  DM6-DM7  |############################################| 92.5%
  reader  DM2-DM5  |##########################################  | 87.8%   4.9 pts slack
  pack    TRISC2   |#########################################   | 86.4%   6.3 pts slack
  idle    TRISC3   |                                            |  0.2%
```

**Read that chart carefully, because occupancy measured this way is close to content-free.** Each
role's *shortfall* from the envelope is *T-independent* — it is a fixed cycle count, not a growing one:

| shortfall = envelope − role span | T=40 | T=80 | T=160 | spread |
|---|---:|---:|---:|---:|
| reader `DM2` | 1004 | 970 | 980 | 34 |
| writer `DM6` | 613 | 613 | 613 | **0** |
| unpack `TRISC0` | 638 | 604 | 604 | 34 |
| math `TRISC1` | 595 | 599 | 599 | 4 |
| pack `TRISC2` | 1104 | 1108 | 1108 | 4 |

⇒ occupancy is `1 − const/envelope` and therefore **converges to 100% for every role**. The climb
79.6→87.1→92.7% is that constant amortising (fill, drain, in-zone setup), not resources filling up. A
blocked `cb_wait` sits *inside* the kernel zone, so a faster stage spends its surplus waiting inside its
own span and stretches to match the binder — §5.0.1 saw the same thing as "all stages advance at one
shared rate". **So this occupancy answers "is this stage inside the pipeline", not "is this stage the
bottleneck".**

The signal is the *difference between shortfalls*, not the distance from 100%: reader 980 vs math 599
= **~380 cycles of genuine reader idle**, against 7060 − 6600 = 460 predicted by the roofline terms.
Same quantity, two independent routes. Note also that `unpack`/`math`/`pack` are **serial within a
tile**, so their three near-identical bars are three overlapping windows on one chain, not three
parallel resources — which is why the model carries one `C` term rather than three.

**Which roof it lands on — and this adjudicates the two competing fits.** The design's model is
`marginal = max(165.0/R, 176.5/C, 83.5/W)` (§3.3.1, DESIGN §2.3); a rival 2-term fit dropping the `C`
term matched 12/13 configs and predicts `max(41.25, 41.75) = 41.75`. Successive differences over
40→80→160 give **43.15 then 44.00** cyc/tile — the first rung still inside the bend, the second clean:

| config | `165/R` | `176.5/C` | `83.5/W` | binds | 3-term | 2-term | measured |
|---|---:|---:|---:|:--:|---:|---:|---:|
| `1,1,1` | 165.00 | 176.50 | 83.50 | **C** | 176.50 | 165.00 | 176.50 (§5.0.1) |
| `2,2,1` | 82.50 | 88.25 | 83.50 | **C** | 88.25 | 83.50 | — |
| `4,4,2` | 41.25 | 44.125 | 41.75 | **C** | **44.12** | 41.75 | **44.00** |

⇒ **the 3-term model is right and the `C` term is real**: 44.00 is 0.3% from 44.12 and 5.4% from 41.75.
The occupancy table agrees on the ordering — math is the highest at 92.7%.

**But the `C` roof is not arithmetic.** Per tile a Neo retires 11 Tensix instructions — `unpack_instr`
2, `math_instr` 8, `pack_instr` 1, invariant across every config and shape measured. Against a 176.5
cyc/tile budget at `C=1` that is **≤4.5% arithmetic even charging one cycle per instruction**, and the
perf trace's own issue efficiency `instr/(instr+stall)` independently reads **4.4% at `1,1,1` and 4.3%
at `4,4,2`**, with `math_stall` covering 95.1% of the math TRISC's span. `unpack_stall` and
`pack_stall` are **exactly 0** — only math, `sem` and `other` stall.

```
what the 176.5 cyc/tile "compute" roof is actually made of (per Neo, per tile)

  arithmetic  8 math instr   |##                                  |  <=4.5%
  waiting     DFB + operands |####################################|  >=95.5%
```

⇒ the `C` roof is a **DFB-delivery roof wearing a compute label**. Consequences: more Neos cannot lower
it much **once it stops binding** (the recorded `C=2→4` = 0.3% is an `R=1` figure, where the reader
binds; where compute still binds it pays — `2,2,2`→`2,4,2` is 88.25→82.53, 6.5%), and the levers that
*can* lower the roof itself are fewer Tensix instructions
per tile or cheaper DFB handshakes — which is precisely §10.6's 56% call-batching lever, not more `C`.
This also generalises §5.0.1's "the Tensix is starved" from the 2-DM baseline to the tuned config: it
is not a baseline artifact, it survives 4x more DM cores and 4x more Neos.

**The two DM roofs are one constant.** `165.0 / 2 = 82.5` against `83.5` — 1.2% apart. One DM core costs
**~83 cycles per tile-transaction regardless of direction**; the reader's 165 is that constant times its
two transactions per tile, the writer's 83.5 times its one. That is also *why* the balanced frontier came
out at `R ≈ 2W` (`R = ceil(0.935·C)`, `W = ceil(0.473·C)`, ratio 1.977 — algebraically `165/83.5 =
1.976`, so this explains the frontier rather than confirming it independently).

**What must NOT be said about any of this.** Bytes moved per cycle. §10.4 is explicit that
`qsr_rocc_copy_bytes` is a host `memcpy` inside the issue instruction, so transfer size never becomes
cycles and there is no contention model anywhere. Dividing analytic bytes by these spans yields a
figure (~23 B/cyc/DM core) that is **invariant across `R`, `C`, `W` and shape by construction** — it is
the fixed 2048 B/transaction divided by the fixed ~83-cycle instruction cost, and it measures the tile
size, not a bandwidth. Per §10.6 report occupancy and per-core instruction cost; **the memory roof is
absent from this roofline entirely and only the emulator can place it.**

**Also re-verified, so §10.4's row is current, not stale:** the perf trace's NoC group
(`noc_reads`/`noc_writes`/`noc_bytes`/`dram_*`/`l1_*`) still reads **0** at craq-sim `ad401613`, with a
second independent witness — `TTSIM_L1_TRACE=1 TTSIM_L1_TRACE_TILE=0` emits 248 events and **zero**
`noc_read_return`, the tag the instrumented `noc_cmd_ctrl` path writes, and `trace_l1_enabled` has no
source filter. Consistent with §10.4's root cause (ROCC command buffers never call the tracer).

**All three arms re-fitted on ONE simulator (`ad401613`), 80→160 tiles/cluster — do not mix bases.**
The August marginals were taken on craq-sim `5ced8886`, so every ratio built from them was
cross-simulator. Re-measuring closes that, and the numbers reproduce almost exactly:

| arm | marginal | prologue | recorded (Aug, `5ced8886`) |
|---|---:|---:|---:|
| `metal_v2` (Milestone-0 **history record**) | **187.00** | 1118 | 187.0 |
| native `1,1,1` — **the baseline for every gain** | **176.00** | 773 | 176.50 |
| native `4,4,2` | **44.00** | 1115 | 44.12 |

⇒ **Perf gains divide by native `1,1,1` only.** Threads delivered `176.00/44.00` = **4.000×, i.e.
100.0% of the hard ceiling** — which is `C: 1→4`, since the `C` term binds at both endpoints and
`C <= 4`. The `metal_v2` row is kept as the Milestone-0 history record and as the input to the F13
price measurement below; **no gain is computed across the two factories** — a cross-arm ratio mixes a
factory change into a threading claim and lands above the 4.00× ceiling, which correctly reads as a
method error.

**Every ratio in this section is a THROUGHPUT gain** — all three arms are marginals, i.e. slopes, so the
prologues in the table above are excluded by construction. The **latency** gain (span ratio at a stated
tensor size) is strictly smaller and size-dependent, because `4,4,2` carries the larger prologue: 2.69×
at the 1280-tile benchmark shape, 3.59× at 5760 tiles, → 4.00× only asymptotically. Full curve and the
naming rule: status §3, design §2.1. Never quote either gain unlabelled.

**The `metal_v2` → native delta at identical threads (11.00 cyc/tile, ~6%) is the DELETED nD STRIDE
CASCADE — it is F13's price, measured at last.** It is a *cost record*, never a gain factor.
`native_span.py` expects zero here (*"A mechanical copy should be 0"*), and the copy is faithful
everywhere it was supposed to be: the Tensix side is **byte-identical** (2 unpack / 8 math / 1 pack /
21.10 other = 36.13 instr per tile, and `cb_waits` 2, `cb_reserves` 1, `cb_pushes` 1, `cb_pops` 2 —
every counter equal). Ring depth is equal too: `2u` on both (metal_v2 `a_entries`/`b_entries`/
`c_entries`, native `entries_per_thread × max(p,c)`), as is the sync mode
(`disable_dfb_implicit_sync_for_all=true` in both). The divergence is entirely in the **DM kernels**,
which is exactly where the perf trace is blind and the profiler is not:

| per tile, T=160 | metal_v2 | native `1,1,1` | delta |
|---|---:|---:|---:|
| reader `DM2` span | 188.9 | 175.6 | **−13.24** |
| writer `DM3` span | 190.7 | 177.6 | −13.05 |
| Tensix `other_stall` | 177.30 | 22.01 | −155.29 |
| Tensix instructions | 36.13 | 36.13 | **0** |

`kernels_dfb`'s reader takes **18 runtime args** and decomposes the global index per tile through the
nD/D/N/C/Ht/Wt/cND cascade; `kernels_qsr`'s takes **2** and computes `page = start_tile_id + k`.
Design §3.4.1 records the deletion as **F13, an unmandated divergence and a regression**, and DESIGN:686
priced it — *"4 div + 4 mod + ~10 mul-add per tile … at ~165 cyc/tile reader budget this is not free —
measure before adopting"*. **Measured: 11.00 cyc/tile of marginal plus 345 cycles of prologue** (the
args never read, the cascade never set up); those combine to the 13.16 cyc/tile of cluster span at
T=160. The collapse in `other_stall` is the same fact seen from the Tensix side — a shorter reader
chain means less time blocked in `cb_wait`, with the instruction mix untouched.

⇒ **The ~6% delta is not a win, it is a debt.** It is the performance of a *missing feature*, and F13
(Milestone 2.1) gives it back unless the cascade lands behind the compile-time flag §3.4.1 specifies so
dense shapes keep the linear form. **And craq-sim under-prices it**: the DM cores run at IPC=1
(§10.3), so a divide costs 1 cycle here against 10-30 on a real RISC-V, and the cascade is
divide-heavy. The silicon cost of restoring it unflagged is therefore **larger** than 11 cyc/tile.
⇒ **Report 4.000×, native basis. Folding the factory delta into a gain banks a regression as a win.**

Harness (gitignored scratch): `debug/perf/util_rcw.py` (`--fallback` runs the baseline arm), raw data
`debug/perf/util_442{,_T}.json`, `util_111_T.json`, `util_v2base.json`.

### 5.0.3 Synthesis: it is a rate-coupled pipeline of dependency chains

§5.0.2's three results — the roofline lands on `C`, every stage reads ~92% occupied, and the binding
stage is 95.7% stalled — are not three findings. They are one mechanism seen from three sides.

```
DRAM --> reader DM x4 --> in0/in1 DFB --> [unpack -> math -> pack] Neo x4 --> out DFB --> writer DM x2 --> DRAM
              165/R        2 entries               176.5/C                    2 entries       83.5/W
                           per thread                                         per thread
```

**1. Bounded buffers make it rate-coupled.** The rings hold `entries_per_thread = 2` per thread, so no
stage can run ahead. In steady state every stage passes the same tiles/cycle: there is **one rate for
the whole pipeline**, set by the slowest stage. That is exactly why the model is a `max()` and not a
sum — the stages overlap, so you do not add them; they are locked, so you take the worst.

**2. That is why utilization is ~92% everywhere, and why the number is nearly content-free** — see the
fixed-shortfall table in §5.0.2. Occupancy → 100% for every stage by construction.

**3. The three roofline terms are the three stages' service times.** At `4,4,2`, T=160 the terms
predict `165·T/R = 6600`, `83.5·T/W = 6680`, `176.5·T/C = 7060`; the measured spans order identically
(reader 7175 < writer 7542 < math 7556). The residual ~500 cycles each is in-zone fixed setup, which
these zones cannot separate from service — that needs the sub-kernel zones of §10.2's fourth row.

**4. The binder is waiting on its own dependency chain, not on an upstream stage.** The Neo is slower
than the reader (7060 vs 6600), so it is not starved by the DM side. Per tile it issues 11 Tensix
instructions and **6 DFB operations** (`cb_waits` 2, `cb_reserves` 1, `cb_pushes` 1, `cb_pops` 2 —
exactly, at every shape), strictly ordered: credit arrives → unpack A, unpack B → 8 math → pack → post
credit. `unpack_stall` and `pack_stall` are **exactly 0** while `math_stall` is ~everything, because
math waits on SrcA/SrcB valid — i.e. on the unpacker's turn, which is gated on the credit.

⇒ **176.5 is the length of a serialized chain, not the throughput limit of an engine.** The same holds
on the DM side, where ~83 cycles is one tile-transaction's chain: the reader's 165 is 2× it, the
writer's 83.5 is 1× it.

**So `165/R`, `176.5/C`, `83.5/W` are three chain lengths over how many copies run in parallel.** The
divisor is replication; the `max()` is rate-coupling. Two consequences fall straight out and both match
the record: thread counts never shorten a chain, so the gain is pure replication and **caps at 4.00×**
with the thread budget (6 user DM, 4 Neos); and the balanced frontier `0.935 : 1 : 0.473` is just
`165 : 176.5 : 83.5` normalised — **"replicate each chain in proportion to its length"**. `C` binds at
*every* frontier point because `ceil()` rounds `R` and `W` up, giving them slightly more capacity than
needed.

**This op is dependency-latency-bound, not throughput-bound on anything.** Which separates the two
levers cleanly:

| lever | effect | measurable here? |
|---|---|---|
| **Shorten a chain** — batched `reserve_back(n)`/`push_back(n)`, fewer handshakes per tile | lowers the roof itself, and it is the *same* lever for all three stages since all three chains are handshake-dominated | **yes** — §10.6 prices it at 56% of the baseline term; this gives that measurement its mechanism |
| **Overlap chains within a thread** — deeper rings, more tiles in flight per thread | hides latency inside a chain | **no** — barriers pre-satisfied, no modelled latency; measured 1.02% is a floor, not a ceiling |

Arithmetic never enters: 8 math instructions against a 176.5-cycle chain. Even a 20× shorter chain
leaves the FPU under half busy, and at 1 add per 3 bytes moved silicon's memory roof lands below its
arithmetic roof anyway.

### 5.0.5 RESOLVED: the compute chain is DFB-handshake-dominated, not intrinsic — 2026-09-09

§5.0.3 left one thing unmeasured, and it was the one that decided whether the batching ask is worth
making: of the compute chain's ~176 cyc/tile, how much is the DFB handshake and how much is intrinsic
Tensix dependency. **It is the handshake.** Credit operations cost **40-60 cycles each** and there are
six per tile, so batching amortises them and the DFB-side ask is justified.

**Method, and two traps that each cost a run.** Profiler SUM zones wrap each region of the compute
kernel's per-tile loop; they accumulate across iterations, so there is no zone-buffer pressure and the
full 160 tiles/cluster shape can be used. `SUM_COUNT == 2` and both macros declare the same local
names, so the six regions take three builds in separate scopes. Driver: `debug/perf/zone_split.py`.

1. **`TT_METAL_PROFILER_SUM=1` is required** — `profileScopeAccumulate`'s body is
   `if constexpr (DO_SUM)`, fed by `PROFILER_OPT_DO_SUM` (`jit_build/build.cpp:215`). Without it the
   zones compile to *nothing* and the run is byte-identical to the control, which reads as "no zones
   fired" rather than "misconfigured".
2. **The accumulated total lands in the CSV's `data` column, not `time[cycles since reset]`** — `time`
   carries a shared flush timestamp, so reading it makes every zone on a RISC report the same number.

Calibration: **`TRISC3` executes none of these regions and reads 3.0 cyc/tile on every one**, so that is
the instrumentation floor and is subtracted below. Enabling SUM mode also arms the *dataflow* kernels'
pre-existing zones, which inflates the cluster span 8155 → 9863 (**+20.9%**); all four builds share
that identical span, so the shares are internally consistent and `~clean` rescales by 0.858.

| sub-unit | region | raw | −floor | share of its span | ~clean |
|---|---|---:|---:|---:|---:|
| `TRISC0` unpack | `IN_WAIT` — 2× `wait_front` | 38.0 | 35.0 | 16.9% | 30.0 |
| `TRISC0` unpack | **`CREDITS` — 2× `pop_front`** | **121.0** | **118.0** | **57.1%** | **101.3** |
| `TRISC1` math | `MATH_OP` — 8 add instrs + commit | 198.4 | 195.4 | 88.8% | 167.7 |
| `TRISC2` pack | `PACK_SIDE` — `regs_wait`+pack+release | 113.1 | 110.1 | 52.4% | 94.5 |
| `TRISC2` pack | `CREDITS` — 1× `push_back` | 44.5 | 41.5 | 19.8% | 35.6 |

⇒ **the unpacker is the constraint, and 74% of its span is DFB credit work** (35 + 118 of 206.5). The
causal chain reads cleanly off the table: the unpacker is busy posting credits → the math pipe sits
inside `MATH_OP` at 88.8% waiting for operands it cannot get → the packer sits in `tile_regs_wait`
waiting for math. Nothing here is arithmetic and nothing is an unavoidable pipeline hazard.

**Priced per credit operation:** `pop_front` ≈ **59 cyc** (118 for two), `push_back` ≈ **42 cyc**, and on
the dataflow side `reserve_back` ≈ **68 cyc** and `wait_front` ≈ **20 cyc** — the reader's 68 being the
back-pressure §5.0.3 describes, which batching also relieves. The dataflow zones come free: `RD_RSV`
70.9, `RD_BAR` 8.0, `WR_WAIT` 23.3, `WR_BAR` 8.0 cyc/tile, and the two ~8s confirm §10.4's
pre-satisfied barriers.

**This corrects §5.0.2's reading.** That section put ~22 cyc/tile on the hand-off and ~146 on
"internal", from the perf trace's `other_stall`. The 22 was real but partial: `other_stall` saw only
the **math** TRISC's `cb_wait`, and was blind to the unpacker's 118 and the packer's 42. The trace
cannot attribute per sub-unit, which is exactly why the zones were needed. **The "internal" bucket was
mostly credit posting all along.**

⇒ **The substrate ask in §5.0.3 is justified — but not on the interleaved path, and the distinction
matters.** Multi-tile batching (`num_tiles_per_cycle` up to 8, the hardware limit) does amortise the
~195 cyc/tile of compute-side credit work: the Neo chain falls 176.5 → 110.9 → 61.6 at n = 1 → 2 → 8.
**That is not a speedup of the same size.** The `C` term stops binding at n = 2, after which
`max(165.0/R, 83.5/W) = 41.75` owns the marginal, so at `4,4,2` the whole compute-side lever is worth
`44.12 / 41.75` = **5.7%, and everything past n = 2 is free headroom nobody uses.** Same ceiling that
caps adding Neos, because both levers shrink the same term.

So there are **two batching levers and they attack different terms** — do not quote one's mechanism as
the other's payoff. **They are also two distinct values, not one.** A DFB ring permits producer and
consumer to transact at different granularities (`wait_front(N)` blocks until N entries exist, however
they arrived), so all four combinations of (reader batch, compute batch) are coherent. In the current
code the reader has **no batch parameter at all** — `constexpr uint32_t onetile = 1`
(`reader_no_bcast_dfb.cpp:32`) — while `num_tiles_per_cycle` is a compile-time arg to the *compute*
kernel only. What they share is two constraints, so they cannot be swept independently: ring capacity
must hold both parties' working sets, and both face the same non-contiguity at stride > 1.

| lever | shrinks | worth at `4,4,2` interleaved | where it pays | craq-sim verdict |
|---|---|---|---|---|
| compute-side `num_tiles_per_cycle` | the `C` term | **5.7%** | L1-resident/fused, where the DM terms vanish — but **unreachable on the native path today**, see below | a **ceiling** (instruction counts) |
| dataflow-side reader `n` + ring depth | `165.0/R`, `83.5/W` | the lever that actually matters here | the interleaved path, now | a **floor** — scores ~0, see §10.6 |

The blocker on the first is the STRIDED-ring restriction `max(R,C) == 1 && max(C,W) == 1` plus
`capacity >= 2n`; `entries_per_thread` is already a knob, so the 2-entry NoC ring is the binding reason
n = 1 on interleaved operands, not the layout as such. **Caveat:** these are craq-sim *instruction*
costs. A credit op's instruction cost is real on silicon too, but the round-trip latency is not modelled
here, so silicon's per-op cost is a floor, not a ceiling.

⇒ **For DRAM-interleaved operands, n = 2 is the step to take** — one notch above today's ship (n = 1 on
depth-2 rings; n = 2 needs depth 4). It collects the **entire** compute-side lever, which is solid: the
`C` term stops binding at n = 2. **On the read path, n = 2 is not a proven stopping point** — crossing
the 2.36 KB issue/transport knee is the first-order win, but how much further `Q` pays depends on
`T_fix` (§5.0.7 Q1), and the O2O study's per-core `B_eff` saturation (Q ≈ 3) and its whole-cluster
`f = Q·N/(k + Q·N)` curve do not obviously agree at our operating point — the per-core reading implies
~144 B/cyc/cluster at Q=1 against the cluster curve's ~25. **Reconciling those two, or deriving one from
the source study, is a prerequisite to quoting any read-path saturation figure.**

**`num_tiles_per_cycle` on the native path is 1 unconditionally, and three things stack to keep it there:**

1. **It is not a knob.** No env var sets it. It is derived:
   `n = 1; if (all_borrowed) n = min(is_sfpu ? 2 : 8, c_full_shard_tiles);`
   (`binary_ng_quasar_native_factory.cpp:576-581`). Trying n > 1 at all is a code edit.
2. **`all_borrowed` is unreachable.** It requires all three operands sharded, and the native gate rejects
   *any* sharded operand (`binary_ng_device_operation.cpp:718-724`). So the `if (all_borrowed)` branch is
   **dead code on this path**, and admitting sharded operands is F3 (M1.3).
3. **The real blocker: the DFB STRIDED rule makes threading and batching mutually exclusive.** The
   factory's own `TT_FATAL` permits n > 1 only when `max(R,C) == 1 && max(C,W) == 1`, i.e. **`R=C=W=1`** —
   and that mirrors the two directional STRIDED asserts in `dataflow_buffer.cpp`, so it is a substrate
   constraint, not our check. **`4,4,2` with n = 8 cannot be expressed today at all**; the choices are
   `4,4,2` with n = 1 or `1,1,1` with n = 8. That is what makes §5.0.3's item a *substrate ask* on the
   tt-metal DFB rather than a factory change — and it is also why craq-sim has no opinion to offer: the
   configuration cannot be built, so there is nothing to measure. (The 1.08x in the lever tables is
   *reader* batching from a hand-written probe, a different lever.)

   > **Corrected 2026-09-17 (§5.0.13).** `4,4,2` with n = 8 can be built and runs: 26.56 cyc/tile against
   > 44.00, output wrong only because of #56194. The `TT_FATAL` here guarded a bug, not a substrate
   > constraint, and the dataflow half needed a counter-major walk in our kernel, not a substrate change.

**Where n = 8 IS reachable: the `metal_v2` factory with all operands sharded** — identical
`min(8, shard_tiles)` logic (`binary_ng_metal_v2_factory.cpp:509-511`) and no sharded rejection. The
45.25 cyc/tile all-sharded roofline is a **metal_v2** measurement (design §2.2). Do not compare it to a
native interleaved marginal: that mixes both factory and layout in one ratio.

**Measured (2026-09-10), so the depth knob is not the lever:** at `4,4,2`, 1280 tiles,
`entries_per_thread` 2 / 4 / 8 / 16 all bit-exact, cluster span **2909 / 2897 / 2893 / 2889** — 0.7%
across an 8x depth increase. Depth alone cannot help: the reader calls `async_read_barrier()` **inside**
the per-tile loop with two reads outstanding (one per operand), so `Q` stays at 1 however deep the ring.
A deeper ring only lets the reader run ahead of *compute*. Raising `Q` means hoisting the barrier out of
the loop — reserve `n`, issue `n` reads, one barrier, push `n` — a kernel change, gated on depth >= 2n.

### 5.0.4 The missing roof: NoC/DRAM demand vs architectural peak

Since the sim charges no cycles for transfer, **achieved bandwidth cannot be measured here at all** —
but *demand* can be computed exactly and checked against architectural peaks, which is the useful
substitute. At `4,4,2`, T=160 (5120 tiles, bf16, `3 × 5120 × 2048 = 31.46 MB` over an 8155-cycle span):

| quantity | value | peak | ratio |
|---|---:|---:|---:|
| per-cluster NoC demand | 120.5 B/cyc | **256 B/cyc** — one NoC port, `NOC_PAYLOAD_WIDTH 2048`/8 (`tt-2xx/quasar/noc/noc_parameters.h:390`) | **47%** |
| device-wide DRAM demand | 3857 B/cyc | 2 channels in `quasar_32_arch.yaml` ⇒ **1929 B/cyc per channel** | **implausible** |

Two readings. The NoC itself is **not** the roof at this operating point — Quasar's word is 256 B
against Blackhole's 64 and Wormhole's 32, so one port covers our rate twice over. But the DRAM demand
is far beyond any real channel pair (~1.9 TB/s per channel at 1 GHz), which is the concrete form of
"craq-sim has no memory roof": **the simulated operating point is physically unreachable on silicon for
a DRAM-interleaved shape**, and the gap is the DRAM roof, not the NoC roof. L1-sharded operands remove
that term and put the 256 B/cyc port roof back in charge.

**Where a real denominator comes from.** `tt_metal/.../experimental/noc_estimator/` already returns
`bandwidth_bytes_per_cycle` per transaction pattern, but its `Architecture` enum is **WORMHOLE_B0 and
BLACKHOLE only** — no Quasar. It was fitted from the `tests/tt_metal/tt_metal/data_movement/`
microbenchmarks, so the path to a Quasar denominator is to run that suite on Quasar hardware and add
the arch. To measure *achieved* bytes rather than derive them, `TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1`
(+ `..._RPT_PATH`) is the per-transaction instrument — still unverified on Quasar. The craq-sim
one-call-site patch of §10.4 only validates the numerator we can already compute analytically; it
cannot produce a denominator.

### 5.0 Sequencing decision (2026-08-18): BASELINE FIRST, then native

Build and validate the measurement harness on the **existing** `ProgramFactoryMetalV2` path *before*
writing any native code. Reasons, in order of weight:
1. The gain is the deliverable (a functional path already exists), so "can we produce a credible number on
   craq-sim for a TTNN op" is the biggest unvalidated risk. Discovering a broken harness after
   implementation is the worst ordering.
2. "The per-tile `async_read_barrier` dominates" is a **code-reading hypothesis, not a measurement.** At
   sim-tractable tile counts, program setup + DFB init may dominate instead (`dfb_init_timing_bench.cpp`
   exists because DFB init cost warranted its own harness). Baseline decomposition may redirect the design.
3. The benchmark shape is empirical: the smallest per-cluster tile count where cycles/tile flattens into
   steady state while sim wall-clock stays tolerable.
4. Instrumentation is cheaper designed-in than retrofitted, so the measurement mechanism should shape the
   native kernels.
5. The A/B selector (both factories reachable in one run) belongs to the harness step anyway.

Baseline spike:
- **Step 0 — harness viability. ANSWERED 2026-08-18: the device profiler DOES run on craq-sim.** Evidence:
  - `tests/scripts/quasar/quasar_local_tests.yaml:48-66` runs `tests/tt_metal/tools/profiler/test_device_profiler.py`
    with `TT_METAL_DEVICE_PROFILER=1` — `test_custom_cycle_count_slow_dispatch` (1x3, + `TT_METAL_SLOW_DISPATCH_MODE=1`),
    `test_custom_cycle_count` (2x3_DISPATCH), `test_full_buffer` (2x3_DISPATCH).
  - `tests/scripts/quasar/run_quasar_regression.sh:138-140` resolves each `config` to
    `TT_METAL_SIMULATOR=$TT_METAL_SIMULATOR_BASE/emu-quasar-<config>` → those lists **are** the simulator.
  - `test_full_buffer` has an explicit Quasar arm: `QUASAR_RISC_COUNT = 6 + 4*4  # DM2-7 + Neo0-3 * TRISC0-3`
    → per-RISC data from **all 6 user DMs and all 16 TRISCs**, i.e. exactly the per-engine visibility this
    design needs. `tt_metal/impl/profiler/profiler_analysis.cpp:61-76` names those RiscTypes.
  - `test_custom_cycle_count*` asserts kernel cycle counts against a reference band → cycle numbers are
    extractable and meaningful.
  - Upstream is actively maintaining it: #51513 "Bringup DeviceTimestampedData + DeviceRecordEvent on
    quasar" (2026-07-30), #51425 "Add back device kernel time columns in perf report on quasar".
  - **No rebuild needed**: the profiler is ON by default (`--disable-profiler` is the opt-out,
    `build_metal.sh:29`) and `.build/default/CMakeCache.txt` has `ENABLE_TRACY:BOOL=ON`.

  Remaining step-0 work (small): run it on our own op, confirm repeat-run determinism, and check
  `TT_METAL_DEVICE_PROFILER_NOC_EVENTS` on Quasar (**unverified**; would directly show transaction
  concurrency, the quantity we intend to raise).
  **Constraint discovered:** the profiler L1 buffer saturates fast on Quasar — that test notes "Quasar runs
  only 1 OP to saturate the L1 buffer" (22 RISCs × 125 zones). **Use coarse zones, not per-tile zones.**
  **Fallback if needed:** `rdcycle` into an L1 scratch region read back by the host — the pattern
  `dfb_init_timing_bench.cpp` already proves works on Quasar (`TT_METAL_MEASURE_DFB_INIT_TIME=1`).
- **Step 0b — static cost model (free).** From the code: per tile pair = 2 `reserve_back` + 2 `async_read`
  + 1 `async_read_barrier` + 2 `push_back`; per output tile = 1 write + 1 barrier; ring depth 2. A
  falsifiable prediction to check the profiler against.
- **Step 1 — shape ladder.** One cluster; sweep per-cluster tiles (8/16/32/64/128/256); record cycles AND
  sim wall-clock. Output: the fixed benchmark shape(s).
- **Step 2 — baseline decomposition.** Reader vs compute vs writer; prologue (program + DFB init) vs steady
  state; engine idle time. This is the number the native path must beat.

Then implement the native factory and re-run the identical harness.

Known distortion to manage: instrumentation adds simulated cycles on craq-sim. Prefer coarse
program-level cycles plus one or two in-loop timestamps over dense instrumentation. Secondary payoff: the
harness becomes the regression gate for phase-2 broadcast work.


- **Primary metric: simulated cycles on craq-sim.** craq-sim applies every store synchronously and has
  produced *bit-identical simulated clocks across 13 repeat runs*, so cycle deltas are deterministic and
  comparable A/B. Quasar is pre-silicon; there are no real-HW numbers.
- **Secondary: the device profiler**, which has Quasar support (`tt_metal/tools/profiler/kernel_profiler.hpp`
  has `ARCH_QUASAR` paths at `:22,48,114,212,234,717,881`), for per-engine timestamps; and the per-role
  cycle harness pattern in `dfb_init_timing_bench.cpp` if finer instrumentation is needed.
- **A/B protocol**: identical shapes/dtypes/memory configs, `ProgramFactoryMetalV2` (baseline) vs the new
  native factory, selected by an explicit switch so both are reachable in one test run.
- **Report**: cycles per output tile at steady state, plus engine occupancy (how many DM/Tensix engines are
  non-idle) and outstanding-transaction depth if observable.
- **Honesty caveat to carry into any claim**: craq-sim is a functional simulator with a cycle model of the
  Tensix/NoC; it does not model DRAM bandwidth or contention faithfully (its descriptor has only 2 DRAM
  views). So the defensible claim is **"removes serialization and raises concurrency, measured as cycle
  deltas"**, not "achieves X GB/s". The ~350 ns figure for a DRAM read / L1 double-buffer turnaround in
  earlier notes is an estimate carried from the Aether study, not a measurement of this op.
- **Shape sizing tension**: craq-sim is slow, so per-cluster tile counts must be small enough to simulate
  yet large enough that steady state dominates the prologue. Start on 1 cluster with ~64-256 tiles per
  operand, then a small grid (e.g. 2×3) once single-cluster numbers are stable.

### 5.0.6 Scorecard — expectations vs measurements (moved here from the tutorial page, 2026-09-10)

Every pre-implementation expectation, scored against the Milestone-1 measurements. Result gains are
native-basis (§5.0.2's rule). Two rows' *expectations* were stated on the pre-implementation gate basis —
kept verbatim as history, per the same rule.

| expectation | source | result | verdict |
|---|---|---|---|
| Go/no-go threshold — stop if threads deliver <1.3x | design §2.4 | **4.000x** from threads | cleared 3x over |
| Capture >=50% of the 149.1 cyc/tile headroom | design §2.3.2 (gate metric — pre-implementation basis by definition) | 94.6% span / ~101% marginal | far exceeded; marginal >100% because native `4,4,2` (44.00) undercuts the 45.25 floor, which was measured on the old arm at `C=1` |
| Threads must deliver 2.83x for the ceiling | status, M0-entry plan table (history) | 4.000x, the ceiling exactly | exceeded |
| Combination law: additive was the default; `max()` needed justifying | design §2.3.2 | envelope 8155 vs additive 20,340 at T=160 | **`max()` confirmed — additive killed by 2.5x** |
| Issue-bound below a ~2.36 KB transaction | O2O study | 2048 B, measured issue-bound | predicted exactly |
| NoC peak 256 B/cyc | HAS + `noc_parameters.h` + O2O | three sources agree | confirmed |
| "More cores -> more sync -> more DM overhead" | Aether matmul notes | 6 DFB ops per 11 Tensix instructions | confirmed |
| Headroom is the DM loop; target DM threads, not machinery | design §2.2 | the `C` term binds at every frontier point | **inverted** |
| In-flight concurrency sufficient | implicit in the depth-2 default | Q=1, ~10% of achievable per-core NoC | **inverted — the large gap** |

Two rows deserve more than their checkmarks:

- **The combination law is settled.** The design flagged it as blocking falsifiability — the two laws are
  44% apart at the target config, and additive was the default assumption. The measurement kills additive
  by 2.5x (envelope 8155 vs additive 20,340). Every prediction that was hedged on this can now be stated
  plainly.
- **We are at the "irreducible machinery" floor — with a correction to the design doc.** 44.00 measured
  against the recorded 45.25 floor is not an anomaly: the floor was measured on the `metal_v2` arm at
  baseline knobs, so its Tensix machinery never divided by `C` — **45.25 is a floor only at `C=1`**, and
  design §2.2 should not be read as a bound on multi-Neo configs. Practically: craq-sim has nothing left
  to give on this op.

### 5.0.8 MEASURED 2026-09-10: batching works at stride 1, corrupts above it, and pays ~1%

Both steps of §5.0.7's experiment ran, via a new `TTNN_QSR_TILES_PER_CYCLE` override that reaches the
batched path on interleaved operands (the `all_borrowed` branch cannot). Depth held at 4 for both arms so
`N` is isolated from ring depth.

**Step 1 — `1,1,1`, stride 1: WORKS.** Bit-exact at 64, 96, 33, 97, 31 tiles — including the odd counts
that exercise the restored remainder branch. **The interleaved producer-1/consumer-N mechanism is sound**;
that is the part unbuilt on every architecture, and it is now built and validated.

| `1,1,1`, depth 4 | 60 t/c | 120 | 180 | marginal | prologue | span @ 40 t/c |
|---|---:|---:|---:|---:|---:|---:|
| N=1 | 11363 | 21953 | 32543 | **176.50** | 773 | **7833** |
| N=2 | 11039 | 20939 | 30839 | **165.00** | 1139 | **7739** |
| N=4 (depth 8) | 11203 | 21075 | 30943 | 164.50 | 1333 | — |

Both arms exactly linear (identical 60-cycle steps). **Three results:**

1. **The roofline predicted this and hit it to the digit.** `max(165.0/R, 176.5/C, 83.5/W)` says that once
   the `C` term falls, the marginal lands on `Rc = 165.0`. Measured **165.00**. The model is
   *mechanism-agnostic* about how `C` shrinks — batching and adding Neos are interchangeable to it. Note
   this bounds the batched compute chain only at `<= 165`; the reader floor hides its true value, so
   §5.0.5's `131/n + 45` decomposition is **consistent but still not pinned**.
2. **N=2 is the stopping point, now measured.** N=4 gives 164.50 — 0.3%, about what the depth change
   4 → 8 contributes by itself. Everything past N=2 is inaccessible headroom, as predicted.
3. **The prologue cost was badly underweighted, and it nearly cancels the gain.** Prologue rises
   **773 → 1139 (+366, +47%)**. So **7.0% throughput gain becomes 1.2% latency gain** at the 1280-tile
   benchmark shape (7833 → 7739, both measured, both exactly on the fitted line). This is the
   throughput-vs-latency rule of status §3 in its sharpest form yet.

**Step 2 — `4,4,2`, stride 4: CORRUPTS.** With the guard bypassed, output is wrong at
**(N-1)/N** — 49.9% at N=2 (128 and 1280 tiles), 69.9% at N=4 (predicted 75%; misplaced writes wrap
inside the thread's own region, so some land correctly by accident).

⇒ **The pack path is not stride-aware, and the `TT_FATAL` was protecting a real defect.** The unpack side
explicitly applies stride (`offset_address = stride_size * tile_index`, `cb_api.h:146`); the pack side
spaces tiles contiguously, so at stride 4 every tile but the first lands in a sibling's slot. **An earlier
draft of §5.0.7 suggested the guard was over-broad and should test the pairing ratio instead of `max()` —
that was wrong twice: `stride_in_entries = max(P,C)` is exactly what the guard tests, and the guard is
load-bearing.** The bypass has been removed; `4,4,2` + N=2 now throws as it should.

**`4,4,2` timing, taken with the guard bypassed.** Output is corrupt there, but a broken dataflow still
executes the full trip count, so the *timing* is real (§10's rule: measure the region your model cannot
see). Depth 4 both arms:

| `4,4,2`, depth 4 | 60 t/c | 120 | 180 | marginal | prologue |
|---|---:|---:|---:|---:|---:|
| N=1 | 3759 | 6405 | 9053 | **44.12** | 1112 |
| N=2 | 3911 | 6417 | 8923 | **41.77** | 1405 |

41.77 is the DM floor `max(165.0/4, 83.5/2) = 41.75` — the second exact out-of-sample hit for the
roofline. **Break-even** is `Δprologue / Δmarginal` = `293 / 2.35` = **125 tiles/cluster**, and the
measured spans straddle it: 60 t/c is **4.0% worse**, 120 t/c 0.2% worse, 180 t/c 1.4% better. At the
40 t/c benchmark shape the fit puts N=2 **6.9% worse**.

**Structural conclusion, platform-independent.** Batching buys marginal cost with fixed cost, so a
break-even size always exists at `Δprologue / Δmarginal` — and the scaling is perverse: **the more threads
already in play, the smaller `Δmarginal`, so the larger the tensor needed to repay the same fixed cost.**
At `1,1,1` the saving is 11.50 cyc/tile and break-even is 32 t/c; at `4,4,2` it is 2.35 cyc/tile and
break-even is 125. Same pattern as adding non-binding Neos (§8: `C 2→4` does not pay below ~440-740 t/c).

**These numbers do NOT transfer to silicon, and craq-sim is not a valid stop signal for this lever.**
Design §2.4 already classifies batching as **two-sided, net direction unknown**; the prologue term adds a
third unknown rather than resolving it:

| effect | direction on silicon |
|---|---|
| DM terms rise (we sustain ~10% of achievable NoC in-flight, §5.0.4) | **less** value — at `4,4,2` compute binds over the DM floor by only 5.6%, so any larger relative DM growth stops it binding and the lever is worth **zero** |
| credit ops cost real coherence traffic, not instruction counts | **more** value — the chain is 74% credit work, so it may inflate faster than the DM terms and bind harder |
| pipeline fill in the prologue is a real DRAM round trip, ~0 here | **less** value — the fixed-cost penalty is understated, though a bigger body could shrink its relative share |

⇒ **Do not use the negative craq-sim result to retire this lever.** What is established: the mechanism
works at stride 1, the cap mechanism is confirmed, the instruction-count half does not pay at the
benchmark shape, and **the pack path corrupts above stride 1** — the last being a real defect on every
platform. The substrate ask is specific and small: **make the pack path apply `stride_size` per tile
index, mirroring the unpack path.** Its value is clearest for the sharded/fused case, where the DM terms
vanish and the compute term keeps binding. The unexamined lever remains the dataflow side, where `RD_RSV`
is 70.9 of the reader's 165 cyc/tile — a body saving ~5x larger, so its break-even should be far more
favourable.

Harness: `TTNN_QSR_TILES_PER_CYCLE` (experimental, unstaged) + `debug/perf/util_rcw.py`.

---

### 5.0.9 MEASURED 2026-09-10: batch ALL THREE stages and it is worth 27%, not 7%

§5.0.8 batched compute only. This adds `TTNN_QSR_DM_BATCH` — a batched reader and writer, `n` tiles per
`async_read_barrier` / `async_write_barrier`, with each entry addressed at `i * get_stride_size()`. All
four arms at `1,1,1`, depth 4, bit-exact where they ran:

| `1,1,1` | depth | marginal | prologue | span @ 40 t/c | throughput | latency @ 40 t/c |
|---|---:|---:|---:|---:|---:|---:|
| dm=1, N=1 — baseline | 4 | 176.50 | 773 | 7833 | — | — |
| dm=1, N=2 — compute only | 4 | 165.00 | 1139 | 7739 | +7.0% | +1.2% |
| **dm=2, N=1 — dataflow only** | 4 | **176.50** | 831 | — | **0.0%** | — |
| dm=2, N=2 — all three | 4 | 139.00 | 1139 | 6699 | +27.0% | +16.9% |
| dm=4, N=4 — all three | 8 | 113.50 | 1246 | 5788 | +55.5% | +35.3% |
| **dm=8, N=8 — all three** | 16 | **97.39** | 1466 | **5363** | **+81.2%** | **+46.1%** |

**Bit-exact at every N**, across 64/96/33/97/31/1280 tiles. Batching pays all the way to the DST limit of
8, with diminishing but substantial increments: marginal falls 37.5, then 25.5, then 16.1. Fitting the
N=2 and N=8 points gives `marginal ~= 111/N + 83.5`, whose asymptote sits suspiciously close to
`Wc = 83.5` — suggestive, not established, and worth a look if this is ever pushed past 8.

**Tail-batch quantisation, and it briefly looked like curvature.** At N=8 the 60→120→180→240 segment
slopes are 99.83 / 94.95 / 99.83 — not monotonic. `Tc = 120, 240` are exact multiples of 8; `Tc = 60, 180`
leave a 4-tile tail that pays a full handshake for half a batch. Fitting the clean multiples gives 97.39,
matching the 3-point least-squares fit, so the marginal holds. **Consequence for shape choice: at large N,
per-cluster counts that are not multiples of N carry a tail penalty** — the 40 t/c benchmark shape happens
to divide by 8 exactly.

**1. Dataflow-only batching returns exactly zero** — the marginal does not move off 176.50 to the digit.
The reader was not the bottleneck, so making it cheaper changes nothing. Third independent confirmation of
the roofline, and the cleanest: a lever that should obviously help, returning precisely 0 because it
targets a non-binding stage.

**2. All three together at N=8 is +81% throughput, against 7.0% for compute-only** — and the reason is
the roofline again. Compute-only stops dead at the reader floor (165.00); batching the reader *moves the
floor*. Neither stage alone can do this. **This is the quantitative form of §10's "batch both sides".**

**3. And the latency gain tracks it: +46.1% at the benchmark shape** (7833 → 5363, measured, not
extrapolated). Break-even is `Δprologue / Δmarginal`, which for N=8 is `693 / 79.11` = **8.8
tiles/cluster** — far below the 40 t/c shape, and better than N=2's 9.8 despite a larger prologue,
because `Δmarginal` grows faster than the fixed cost does. Compute-only had 32; `4,4,2` compute-only had
125. **The prologue problem of §5.0.8 was never about batching — it was about batching too little.**

**Above `1,1,1` it fails, and the two failure modes classify differently** (§5.0.11):

| | mechanism | legal usage? | verdict |
|---|---|---|---|
| **compute batching** | `get_output_tile_index` adds +1 entry per tile into an index whose converter divides by `stride_size_tiles`, so at stride > 1 consecutive tiles collapse onto one byte offset | **yes** — public API, documented access pattern, no restriction stated | **BUG, filed** |
| **dataflow batching** | `push_back(n)` credits ONE tile counter and rotates `tc_idx` once, and a DM kernel can address only the active TC (`get_local_*` exposes `tc_slots[tc_idx]`) | **no** — placing n entries across n TCs is not expressible | **do not use; guard is permanent** |

⇒ **The two knobs are used together or not at all** — compute must not pop more than the reader pushes,
so compute-only batching is a diagnostic arm, never a shipping config. That makes the legal space for
batching the intersection of the two rules: compute needs `R <= C and W <= C`, dataflow needs
`C <= R and C <= W`, so **both together require `R == C == W`, i.e. `1,1,1` and `2,2,2` only.**
`4,4,2` and every other config is **N=1 permanently**, not pending a fix.

> **Corrected 2026-09-17 (§5.0.13).** The dataflow row above is wrong. A role owning K counters batches
> per counter with a `K*T` stride and rotates between them, so dataflow batching is legal at every clean
> `(R,C,W)` — 260 runs bit-exact. Only compute batching is constrained, by #56194. `4,4,2` N=8 runs at
> 26.56 cyc/tile, 1.66x over N=1.

⇒ **One substrate ask: make the pack path apply `stride_size_tiles` per tile index.** `cb_api.h:146`
already does exactly this on the unpack side, with the comment "per-tile spacing is `stride_size`, not
`entry_size`". That single fix delivers **`2,2,2` N=8** — the only configuration batching can reach, and
one that is **10% slower than the `4,4,2` N=1 default in absolute throughput**. So the fix is not on our
critical path; it converts batching from unusable into an engine-constrained option.

**Do not carry forward a `4,4,2` batching projection.** Earlier drafts of this section estimated 24-27
cyc/tile there by scaling the `1,1,1` cut, then a bound of [1.05x, 1.81x]. Both are moot: `4,4,2` N=8 is
not legal usage, so no fix makes it reachable. The batched-writer-constant probe at `4,4,1` that would
have narrowed that range is withdrawn with it.

Harness: `TTNN_QSR_DM_BATCH` + `debug/perf/batch_check.py` (both experimental, unstaged).

---

### 5.0.10 The whole picture at N=8 — batching IS the in-flight lever

Batching does not merely amortise credit calls. The batched reader issues all `N` reads on `in0` and all
`N` on `in1` **before** its `async_read_barrier`, so those transfers are concurrently outstanding. `Q`
goes from 1 to `N`, which is exactly the in-flight axis §5.0.4/§10.6 identified as the large unpriced
lever:

| config | reads outstanding | airborne per cluster | fraction of NoC peak (`f = Q·N/(k+Q·N)`, `k ≈ 112 KB`) | runs? |
|---|---:|---:|---:|---|
| `1,1,1`, N=1 | 2 | 4 KB | 3.4% | yes |
| **`1,1,1`, N=8** | 16 | 32 KB | **22.2%** | **yes — the only measured N=8** |
| `4,4,2`, N=1 | 8 | 16 KB | 12.5% | yes |
| `4,4,2`, N=8 | 64 | 128 KB | 53.3% | **NO — illegal combination, see §5.0.11** |

**Read the fourth row as arithmetic on a configuration that cannot execute.** `4,4,2` N=8 is rejected by
the factory guard and corrupts when the guard is bypassed, so its 128 KB / 53.3% is what the model says
*would* be airborne, not a state anything has reached. **The in-flight rise actually achieved is
3.4% → 22.2%, on one thread.** Do not quote 12.5% → 53.3% as the batching result: that attaches a blocked
config's projection to a `1,1,1` measurement.

§5.0.4 put the 50%-of-peak target at "~112 KB airborne, about 10 tiles per thread against the one we
have". **Nothing has reached that.** `1,1,1` N=8 delivers 16 tiles airborne per cluster against the ~56
that target needs; the config that would reach it is the blocked one. Consequence for how §5.0.9's number
should be read: **the measured +81% at `1,1,1` is the instruction-count half only, so on the dataflow side
it is a FLOOR, not a ceiling** — craq-sim charges nothing for transfers and cannot price the `Q` increase
at all. That floor argument stands on its own and does not need the `4,4,2` projection.

This also resolves why the dataflow-only arm returned exactly 0.0%: **two different zeros.** On the
instruction-count axis it is a genuine zero (the reader was not the binding stage, so `max()` ignores it).
On the in-flight axis it is the simulator declining to answer. On silicon that arm would return something
positive even though craq-sim says nothing moved.

**Two roofs arrive together at `4,4,2`, N=8** (marginal projected ~25 from §5.0.9's 44.8% cut):

| quantity | N=1 | N=8 |
|---|---:|---:|
| NoC demand per cluster vs the 256 B/cyc port | 139.6 B/cyc = 54% | **245.8 B/cyc = 96%** |
| device-wide DRAM **demand** | 4468 B/cyc | 7864 B/cyc |

**The NoC port saturates at 96%** — the first roof in this whole analysis to actually bind, and the one
figure here with a real denominator (`NOC_PAYLOAD_WIDTH`, `noc_parameters.h:390`).

**THERE IS NO DRAM DENOMINATOR — do not manufacture one.** An earlier version of this section compared
the 4468 B/cyc demand against "~3858 B/cyc available" and reported **1.16x over**, with a **"DRAM-feasible
floor of ~51 cyc/tile"**. Both are artifacts and are withdrawn:

- **3858 B/cyc is not a supply figure.** It is §5.0.4's *demand* at a different operating point
  (`4,4,2`, T=160: `3 × 5120 × 2048 = 31.46 MB` over an 8155-cycle span). So "1.16x over" compared two
  demands — `6144 × 32 / 44.00 = 4468` against `6144 × 32 / 50.97 = 3857` — and the ratio is just
  `50.97 / 44.00`.
- **~51 cyc/tile is not a DRAM roof.** `8155 / 160 = 50.97` is span/T, the *average* cost per tile, which
  is `marginal + prologue/T = 44.12 + 1106/160 = 51.03`. It differs from the marginal by prologue
  amortisation and nothing else. §5.0.6 states this identity directly.

What §5.0.4 actually established stands, and is a plausibility flag rather than a ceiling: a device-wide
demand of ~3857 B/cyc attributed to the descriptor's 2 DRAM views implies **1929 B/cyc per channel, about
1.9 TB/s at 1 GHz — implausible for real hardware.** So these operating points are very likely
unreachable on silicon for a DRAM-interleaved shape. **By how much is unknown**, and it stays unknown
until a denominator exists: `noc_estimator` has no Quasar arch, so the path is running
`tests/tt_metal/tt_metal/data_movement/` on Quasar hardware (§5.0.4, "where a real denominator comes
from"). The descriptor supplies **capacity only — 2 × 1 GB DRAM views — never bandwidth.**

**Compute does not change in character.** 8 math instructions per output tile, spread over `C` engines:

| config | per-engine arithmetic | issue efficiency | idle |
|---|---:|---:|---:|
| `1,1,1` N=1 | 8.0 instr/tile | 4.5% | 95.5% |
| `1,1,1` N=8 | 8.0 | 8.2% | **91.8%** |
| `4,4,2` N=1 | 2.0 | 4.5% | 95.5% |
| `4,4,2` N=8 | 2.0 | 8.0% | **92.0%** |

Batching doubles the arithmetic fraction and moves nothing structural: **the FPU stays ~92% idle at every
setting.** Arithmetic intensity is 12x below machine balance (§5.0.2), so no batching makes this
arithmetic-bound. The op is delivery-bound everywhere on the legal space.

⇒ **Production recommendation: threading is the bigger lever, but batching buys resource efficiency and
that is the more useful thing.** Two comparisons, and they say different things. **At equal N, threading
dominates** — `1,1,1` at N=8 reaches only 97.39, which loses even to `2,2,2` *unbatched* at 88.25, so
eight-way batching on one thread is beaten by two plain threads and N is never a substitute for thread
count. **But batching lets a cheaper config nearly match a bigger one:**

| config | marginal | engines | throughput per engine | basis |
|---|---:|---:|---:|---|
| `4,4,2` N=1 | 44.12 | 6 DM + 4 Neo = 10 | 1.00x | measured |
| `2,2,2` N=8 | 48.70 | 4 DM + 2 Neo = **6** | **1.51x** | projected (firm) |

`2,2,2` at N=8 delivers **90% of the throughput on 60% of the engines**. On a shared cluster — or any time
other work wants to be co-resident — that is the better trade, and it is the strongest argument batching
has. But state it as a trade, not an upgrade: **48.70 against 44.12 is 10% less throughput**, and
`4,4,2` N=1 remains the best config measured and the default. Do not frame N=8 as either a top-up on the
frontier config or a replacement for it.

**Zoom out: N>1 requires the two operand shapes to be EQUAL.** Any broadcast, of either kind, puts N
back to 1 — but by different routes, and the distinction matters because the flag name misleads:

| kind | why N is forced to 1 |
|---|---|
| **subtile** (ROW/COL/SCALAR) | the tile is not consumed 1:1; compute indexes *inside* it, so the batch cannot be formed. The Quasar factory states this directly (`factory:663-665`) |
| **outer-dim** (leading N/C/D/nD) | handled by the reader re-reading a tile through zeroed strides — but only the **interleaved** reader does that (`reader_interleaved_no_bcast.cpp`, `#else` branch, cascade bounded by `dst_num_tiles`). Under `#if SRC_SHARDED` the reader pushes `src_num_tiles`, its own shard, with no duplication — and N>1 is gated on sharding |

**Mind the two conditions and which way the implication runs.** The kernel named `no_bcast` is selected
on `SubtileBroadcastType::NONE`, and that is the *broader* condition:

```
shapes equal   ⊂   subtile broadcast == NONE
```

Equal shapes always imply subtile NONE, so the `no_bcast` kernel serves them — but **NONE does not imply
equal shapes**, because an outer-dim broadcast passes it. The kernel is named for the wider set it
covers, not for the narrower one N needs. Production's gate is
`subtile_broadcast_type == NONE && a_sharded && b_sharded && c_sharded`
(`binary_ng_program_factory.cpp:1016-1040`), and **the sharding half is what narrows NONE down to equal
shapes**: `is_native_L1_sharding` admits all-three-sharded only via the branch guarded by
`a.logical_shape() == b->logical_shape()` (`binary_ng_utils.cpp:806`, comment: *"no broadcast on any
dimension"*); the other sharded branch requires `!b_is_sharded`. Net:

| case | N | why |
|---|---:|---|
| any broadcast, either kind | **1** | see the table above |
| DRAM-interleaved, shapes equal | **1** | never implemented on WH/BH — the gate requires sharding |
| **L1-sharded, shapes equal** | **8** | the only paying case (2 for SFPU, reason undocumented) |

**On the Quasar path neither broadcast kind is supported yet**, so all of the above is forward-looking:

| kind | Quasar native today | roadmap |
|---|---|---|
| outer-dim | **not supported** — `a_dims_are_output_dims && b_dims_are_output_dims` (`factory:1044-1047`) rejects it, because `kernels_qsr/` collapsed the stride cascade to `page = start_tile_id + k` | **2.1 / F13** — a regression to restore, not a feature to add |
| subtile | **not supported** | **2.2 / F8**, gated on `#51291` |

Every measurement in §5.0.8-§5.0.11 is therefore on dense shapes. **Implication for F13:** restoring the
cascade brings back outer-dim broadcast on the *interleaved* path, where N is 1 regardless — so F13 and
the batch knob do not interact. They would only interact if a future sharded path wanted both, which
needs a compute-side index mapping that does not exist today.

In the paying row the reader has almost nothing to do, since the operand is already resident: it pushes
the tiles and stops, so the whole gain is compute-side. **One case of three — N=1 is the norm**, which is
the honest frame for §5.0.8-§5.0.11 as a whole.

Two consequences. **The experiment was instrumentation first** — varying N is how the per-tile credit
constant gets separated from the rest of the chain, and that is what pinned the `C` roof as a *delivery*
roof rather than an arithmetic one (§5.0.5). Read the 1.81x as a measurement of the cost model.

**But the missing row is an opportunity for Quasar, and it is cheap.** Because WH/BH gate multi-tile on
sharding, *DRAM-interleaved no-broadcast N>1 has never been implemented on any architecture* — and the
Quasar factory is being written now, so it can be the first. The gap is a single branch:

```cpp
uint32_t num_tiles_per_cycle = 1;                       // factory:578-581, today
if (all_borrowed) {
    num_tiles_per_cycle = std::min<uint32_t>(is_sfpu ? 2u : 8u, c_full_shard_tiles);
}
```

That is the production policy already ported — including the SFPU cap of 2 — and it covers the sharded
row only. The experimental knob exists precisely to "reach the batched path on interleaved operands,
which the `all_borrowed` branch above cannot". Making it permanent is: **extend that `if` with an
interleaved + no-broadcast + `R == C == W` branch, and derive `entries_per_thread = 2N` instead of
defaulting to 2.** The kernels are already written and measured bit-exact across
64/96/33/97/31/384/416/544/1280/1312 tiles.

**Reach, stated honestly, because it does not help the default config.** `1,1,1` works today; `2,2,2`
follows the pack fix; `4,4,2` follows it too — the "dataflow batching there is not expressible" this
sentence used to say was wrong, see §5.0.13. And `1,1,1` at N=8 is
*strictly worse* than `4,4,2` at N=1 — higher prologue **and** higher marginal — so nobody choosing on
speed alone would run it. The case for landing it is completeness and readiness: it closes a
cross-architecture gap, it is a correct generalisation of a policy already in the factory, it is gated so
the default is untouched, and the alternative is that a measured result evaporates with an unstaged env
knob.

**Why 48.70 is a firm projection.** `2,2,2` divides all three roofs by exactly 2, so
`marginal(2,2,2,N) = marginal(1,1,1,N)/2` holds under the fitted model whatever binds — validated at N=1
to 0.5% (88.00 predicted vs 88.43 measured). The 48.70 is that identity applied to a *measured* N=8
aggregate, and it already carries whatever the writer batching cost, since 97.39 was measured with all
three stages batched. Nothing else about it is extrapolated.

**`4,4,2` batching number — corrected 2026-09-17: 26.56 cyc/tile at N=8, 1.66x, see §5.0.13. The rest of
this paragraph is the superseded reasoning, kept for the record.** Dataflow batching there
needs the writer's two tile counters addressed from a DM kernel, which the DFB API cannot express
(§5.0.11) — not a defect awaiting a fix. And since the two knobs only ship together, the compute-only
figure is not a configuration either. For completeness, the reason it would have been small anyway:

| config | reader | compute | writer | binds | spread |
|---|---:|---:|---:|---:|---:|
| `1,1,1` | 165.00 | **176.50** | 83.50 | cmp | 2.11x |
| `2,2,2` | 82.50 | **88.25** | 41.75 | cmp | 2.11x |
| `4,4,2` | 41.25 | **44.12** | 41.75 | cmp | **1.07x** |

Batching compresses only roofs with slack above the next one down, and at `4,4,2` the three sit within 7%
of each other — threading already reached the balanced state batching exists to produce. **That is a
reason not to want it there, independent of legality.** `2,2,2` keeps the 2.11x spread, which is why the
same lever is worth 1.81x there and nothing at the frontier.

**So batching's value is NOT "a bit more throughput at `4,4,2`" — it is reaching a given throughput on
fewer engines.** The roof-balance argument above says only that batching adds little *on top of* the
frontier config; it says nothing against using batching to hit near-frontier throughput from `2,2,2`,
which is what the table shows. Batching is also worth most where DRAM is out of the picture — L1-sharded
operands and fused chains (F3) — since a DRAM-interleaved `4,4,2` is very likely transport-limited on
silicon (§5.0.10: demand is implausible, magnitude unknown), clipping
most of what batching could add there but not on a resident-L1 path.

**Open: `4,4,2` N=1 vs `2,2,2` N=8 is a 1.11x throughput gap, so the choice is a policy question, not a
perf question.** Pick `4,4,2` N=1 to minimise latency on a dedicated cluster; pick `2,2,2` N=8 to maximise
throughput per engine when the cluster is shared. `2,2,2` N=8 needs tt-metal#56194 first: today its
compute half silently corrupts 50% of each batch, and our `TT_FATAL` is what keeps that from escaping.

**Blocker, root-caused to one line — tt-metal#56194.** `get_output_tile_index` (`hw/ckernels/quasar/metal/llk_api/
llk_pack_tile_api.h:58-74`) computes the within-batch tile address as `wr_entry_idx + output_tile_index`
— **+1 entry per tile.** But that index is consumed by `dfb_slot_cursor_offset_units`, which converts with
`(delta_entries / stride_size_tiles) * stride_size`, i.e. it *divides* by stride. At stride 1 the two
agree. At stride > 1 the division truncates: tile `i` lands at `floor(i/S) * stride_size`, so a batch of
`N` covers only `ceil(N/S)` distinct slots and loses **`1 - ceil(N/S)/N`** of its tiles. That predicts
every measured rate — stride 4 at N=2 puts both tiles at offset 0 (50% predicted, 49.9% measured),
stride 2 at N=8 collides them in pairs (50%, 50.0%), stride 4 at N=8 in groups of four (75%, 74.9%).
The batch *base* is correct —
`dfb_advance_slot` already does `wr_entry_idx += num_tiles * stride_size_tiles`. Only the per-tile step
inside the batch was missed, and `cb_api.h:146` does it right on the unpack side with the comment
"per-tile spacing is `stride_size`, not `entry_size`". **Earlier drafts hedged this as "pack or unpack,
not decidable from any correct config" — reading the code settles it without needing a config.**

---

### 5.0.11 MEASURED 2026-09-10: localising the multi-thread batching failure

A matrix over (R,C,W) x which-stage-batches, each case in its own process with a 240 s cap and tile counts
chosen so every batching stage forms at least one FULL batch of 8. Guards bypassed; correctness is not
expected above stride 1 — the point is which failure MODE appears where.

| R,C,W | batched | stride in / out | result |
|---|---|---:|---|
| 1,1,1 | both | 1 / 1 | **OK** |
| 1,1,2 | dataflow | 1 / 2 | CORRUPT 99.9% — **void row, see below** |
| 1,1,2 | compute | 1 / 2 | CORRUPT 99.9% — **void row** |
| **2,2,2** | **dataflow** | **2 / 2** | **OK — bit-exact** |
| 2,2,2 | compute | 2 / 2 | CORRUPT 50.0% |
| 4,4,2 | dataflow | 4 / 4 | CORRUPT 87.4% |
| 4,4,2 | compute | 4 / 4 | CORRUPT 74.9% |

**Three earlier claims die here.**

1. **"It hangs" was shape-specific.** No case in this matrix deadlocks; all complete, either clean or
   corrupt. §5.0.9's hangs came from particular tile counts, not from batching per se.
2. **`1,1,2` is a void control.** With **no batching at all** it is already **CORRUPT 50.0%** — `C=1 <
   max(R,W)=2` makes it one of the 18 pre-existing corrupt configs (§4). Nothing it does under batching is
   attributable to batching.
3. **The wrap hypothesis is refuted.** Raising `entries_per_thread` 16 → 32 at `4,4,2` gave results
   **identical to the digit** (87.4 / 74.9 / 93.7%). The thread's region spans
   `(capacity-1)*stride*entry_size + entry_size`, which at depth 16 and stride 4 is 124928 B against a
   batch of 8 needing 59392 B — it never wrapped. So `entries_per_thread >= 2N` stands and the
   "wrap-aware per-entry address" ask of §5.0.9 is **WITHDRAWN**.

**Dataflow batching is bit-exact at `2,2,2` — a real multi-thread config at stride 2.** So stride > 1 is
not the barrier. The discriminator is **`num_tcs_to_rr`**, how many transaction counters a thread
round-robins through (`dataflow_buffer.cpp:1260-1286`); the kernel interface carries a base address **per
TC** (`base_addr[tc]`, `tc_slots[tc_idx]`), and our batched walk steps `i * stride` from a single base:

| config | reader `C>=R ? C/R : 1` | compute | writer `C>=W ? C/W : 1` | predicts |
|---|---:|---:|---:|---|
| 1,1,1 | 1 | 1 | 1 | OK ✓ |
| 2,2,2 | 1 | 1 | 1 | OK ✓ |
| 4,4,2 | 1 | 1 | **2** | dataflow fails ✓ |

⇒ **Two independent failures, and they are different KINDS, not two bugs:**

- **Compute batching — a BUG in the metal LLK pack path.** `2,2,2` isolates it cleanly: every role is
  single-TC, no wrap, stride 2, and compute batching still corrupts 50%. Root cause is
  `get_output_tile_index` adding +1 entry per tile into an index that `dfb_slot_cursor_offset_units`
  divides by `stride_size_tiles` (§5.0.10 has the derivation and the matching corruption rates). Public
  API, documented STRIDED pattern, no restriction stated, wrong data out — **filed as tt-metal#56194**, in
  `hw/ckernels/quasar/metal/llk_api/` — NOT tt-llk and NOT craq-sim.** It is integer arithmetic, so silicon
  behaves identically; craq-sim is only where it was observed.
- **Dataflow batching — NOT A BUG; the operation is not expressible.** *(Corrected 2026-09-17: it is
  expressible, with a counter-major walk — §5.0.13. The mechanism below is right; the conclusion drawn
  from it was not.)* `push_back(n)` credits the current
  tile counter by `n` and rotates `tc_idx` by exactly 1 (`dataflow_buffer.inl:203`), `reserve_back` does
  not rotate at all (`:140`), and a DM kernel can address only the **active** TC — `get_local_*` exposes
  `tc_slots[tc_idx]` and nothing else. So spreading a batch across `num_tcs_to_rr` counters cannot be
  written with today's API. The DFB *did* consider batched multi-TC: the `broadcast_tc` / BLOCKED branch
  posts `num_entries` to all N counters and deliberately skips the rotate — handled for broadcast only.
  **Our guard is correct and permanent**, not a placeholder. The one thing worth filing here is the
  **missing assert**: `push_back(n>1)` with `num_tcs_to_rr > 1` should fail loudly instead of
  silently mis-crediting.

> **Corrected 2026-09-17 (§5.0.13).** The consequence below rests on the "not expressible" bullet, which
> was wrong. Dataflow batching is legal at every clean config; the legal space for batching is set by
> #56194 alone, and `4,4,2` N=8 measures 1.66x.

**Consequence — and this is what makes the legal space so small.** The two knobs ship together or not at
all (compute must not pop more than the reader pushes), so batching needs compute's TCs at 1
(`R <= C and W <= C`) **and** the DM roles' TCs at 1 (`C <= R and C <= W`). Opposite constraints;
intersection `R == C == W`; with `C in {1,2,4}` and `R+W <= 6` that is **`1,1,1` and `2,2,2`, full stop.**
Every other config including `4,4,2` is **N=1 permanently.** Measured at `2,2,2`, depth 16:

| `2,2,2` | marginal | prologue |
|---|---:|---:|
| no batching | 88.43 | 864 |
| dataflow batched N=8 | **87.43** | 1458 |

**1.1% on the marginal, and a net loss at any real shape** once +594 prologue is paid — exactly as the
roofline requires: `Cc/C = 176.5/2 = 88.25` and compute is not batched, so **compute binds at 88.25 either
way**. The reader term falls to 48.70 and the writer to 28.05, both irrelevant because neither was binding.

⇒ **The 1.81x at `1,1,1` is the only measured batching win, and it exists solely because `1,1,1` is the
one config where the LLK bug cannot fire (stride 1 on both rings).** Compute is the binding stage
everywhere on the legal space (§8), so **the entire lever is gated on the single LLK fix** — which
delivers exactly one further configuration, **`2,2,2` N=8**, which is slower than the `4,4,2` N=1
default in absolute throughput and earns its place only on engines-per-tile (§5.0.10: 90% of the
throughput on 60% of the engines). **Batching is therefore not on the critical path**; `4,4,2` N=1 is the
default and no fix changes that. **The multi-TC walk fix we had
planned is cancelled**, not deferred: its only beneficiaries were `4,4,2`-shaped configs, which the
API cannot support at any batch size.

Harness: `debug/perf/hang_matrix.py` (experimental, unstaged).

---

### 5.0.12 2026-09-15: which ceiling binds — and why DRAM cannot be ranked yet

Three ceilings appear across §5.0.2-§5.0.10. Two have real denominators; the third does not, and that
asymmetry is the whole finding.

| ceiling | scope | denominator | status |
|---|---|---|---|
| **issue rate** | per DM core; `R + W` per cluster | ≤ 62 B/cyc, ~24 measured (O2O, BH-calibrated) | **binds today** at `4,4,2` N=1 |
| **NoC port** | **per cluster** — each of the 32 has its own | 256 B/cyc, `noc_parameters.h:390` | 54% used at N=1, 96% at N=8 |
| **DRAM** | **device-wide** — all 32 clusters share one pool | **none exists** | **unrankable** |

**The scope distinction is real and survives everything below.** The NoC scales with cluster count, DRAM
does not, so a per-cluster figure and a device-wide one are never directly comparable. This op moves
`3 × 2048 = 6144 B` per output tile, so `n` active clusters at `m` cyc/tile demand `n × 6144/m` B/cyc
from one pool.

**But the ranking cannot be completed, because there is no DRAM denominator.** `quasar_32_arch.yaml`
gives 2 × 1 GB DRAM **views** — capacity, not bandwidth — and `noc_estimator`'s `Architecture` enum has
no Quasar (§5.0.4). Any "% of DRAM" figure here would be invented. What can be said:

- **Demand is implausibly high.** ~3857 B/cyc device-wide at `4,4,2` T=160 implies ~1929 B/cyc per
  channel across the descriptor's two views, roughly 1.9 TB/s each. Real channels are far below that, so
  **these operating points are very likely unreachable on silicon for DRAM-interleaved shapes.**
- **The magnitude is unknown.** Nothing in this branch's measurements bounds it.

**A cluster-count escape does not exist, and this is worth stating because it looks like one.** For a
fixed tensor, using fewer clusters gives each more tiles: total bytes are unchanged and wall time rises.
With `n` clusters at `m` cyc/tile over `T_total` tiles, wall time is `T_total · m / n` while demand is
`n · 6144 / m`. If a device-wide supply `S` caps demand, then `n/m ≤ S/6144`, so:

```
wall time  >=  T_total * 6144 / S      — independent of cluster count AND of threading
```

**A device-wide bandwidth ceiling cannot be dodged by any (R,C,W) or any grid size.** Only two things
move it: moving fewer bytes (L1-sharded operands, fusion), or more bandwidth. A small tensor is not an
exception — it finishes sooner because there is less work, not because it found headroom.

### The model, with the unknown left as a parameter

Not knowing `S` is no reason to leave the roof unmodelled — it is a reason to carry it as a symbol and
invert the question. Four terms, three of them known:

```
  D(m)     = n · d · tile_bytes / m            device-wide DRAM demand, B/cyc
  m_dram   = n · d · tile_bytes / (S · η)      the DRAM-feasible marginal
  m_si     = max( m_roofline , m_dram )        what silicon would deliver
  wall     ≥ T_total · d · tile_bytes / (S · η)
```

| symbol | meaning | value |
|---|---|---|
| `n` | clusters concurrently active | ≤ 32 |
| `d` | how many of `{in0, in1, out}` live in DRAM | **3** interleaved, 2 with one operand sharded, **0** fully resident |
| `tile_bytes` | 2048 (bf16) | known |
| `m_roofline` | `max(165/R, 176.5/C, 83.5/W)` | measured, §5.0.5 |
| **`S`** | **device DRAM peak bandwidth** | **UNKNOWN for Quasar** |
| **`η`** | **achieved fraction of peak** | **unknown; rises with outstanding requests** |

**Note the wall-time floor contains neither `n` nor `R,C,W`.** That is the formal statement of why a
device-wide ceiling cannot be dodged by threading or by grid size: substituting `n/m ≤ S·η/(d·tile_bytes)`
into `wall = T_total·m/n` eliminates both. Only `d` and `S·η` move it.

### Inverting it: what each config REQUIRES

Rather than guess `S`, compute the supply each configuration needs. Full grid, `d = 3`, `η = 1` (so these
are floors — a real `η < 1` scales them up by `1/η`):

| config | marginal | required `S` | at 1 GHz |
|---|---:|---:|---:|
| `1,1,1` | 176.50 | 1114 B/cyc | 1.11 TB/s |
| `2,2,1` / `2,2,2` | 88.25 | 2228 | 2.23 TB/s |
| `2,4,2` | 82.53 | 2382 | 2.38 TB/s |
| **`4,4,2`** | **44.12** | **4456** | **4.46 TB/s** |
| `4,4,2` at N=8 (projected) | 24.35 | 8074 | 8.07 TB/s |

Read it the other way and it becomes a decision table — **when the real `S` is known, read off the row**:

| if `S·η` is… | DRAM-feasible marginal | configs that survive unclipped |
|---|---:|---|
| 1000 B/cyc | 196.6 | none — even `1,1,1` is transport-bound |
| 1500 | 131.1 | `1,1,1` only |
| 3000 | 65.5 | everything except `4,4,2` |
| **5000** | **39.3** | **all 13, including `4,4,2`** |

⇒ **The whole question reduces to one threshold: `S·η ≥ 4456 B/cyc` and `4,4,2` is unclipped; below it,
the frontier config is transport-bound and the 4.00x is not deliverable on a DRAM-interleaved shape.**

### Two levers the model exposes that a single number would hide

1. **`d` is a lever, and the biggest one.** Sharding one operand takes `d` from 3 to 2, dropping the
   requirement for `4,4,2` from 4456 to **2971 B/cyc**; full residency takes `d` to 0 and removes the
   roof entirely. **This is the same conclusion F3 keeps arriving at from other directions** — it is not
   only where batching pays, it is where the DRAM roof stops existing.
2. **`η` is where batching acts.** More outstanding reads means more banks in flight, so a higher
   achieved fraction of the same peak. Batching does not move `S`; it moves how much of `S` is realised.
   That is the precise form of §5.0.10's "reach the roof more efficiently".

### EVALUATED 2026-09-15: the spec exists, and it is decisive

`S` is not unknown after all — it is just not in this repo. **Quasar is GDDR7**, and *Grendel I Packages*
(Confluence, Packaging space) gives the per-SKU bandwidth:

| SKU | @32 Gbps | @28 Gbps | note |
|---|---:|---:|---|
| MK bring-up | 2 × 128 = **256 GB/s** | 224 GB/s | **2 channels — this is what craq-sim's descriptor models** |
| **QSR1.A1 (GDL-A01)** | 8 × 128 = **1.0 TB/s** (target) | 875 GB/s | the production part |
| QSR3.A2 | 12 × 128 = 1.5 TB/s | 1.3 TB/s (target) | |

So the descriptor's 2 DRAM views are not arbitrary — they model the **bring-up board**, not production.
Clocks come from the *Quasar HAS* Table 9 (preliminary): NoC **0.76 / 1.33 / 1.55 GHz** at 0.55/0.75/0.95 V.

**Evaluated at the nominal point** — 1.33 GHz, 1.0 TB/s, `η = 0.75` (the O2O study's "max effective is
75%") — the DRAM-feasible marginal is **349 cyc/tile**, against craq-sim's 176.50 at `1,1,1` and 44.12 at
`4,4,2`.

**Swept over all 18 corners** (3 clocks × 3 SKUs × 2 efficiencies), where "survives" means the config is
already slow enough that DRAM keeps up:

| config | survives in |
|---|---|
| `1,1,1` | **4 of 18** — and every one needs the 0.55 V clock or the 12-channel SKU, three of the four need η = 1.0 |
| `2,2,2` | **0 of 18** |
| `4,4,2` | **0 of 18** |

⇒ **For DRAM-interleaved bf16 operands this op is DRAM-bound at every legal configuration, and the 4.00x
threading gain does not reach the wall clock on silicon.** It remains a sound instruction-and-issue
result — that is what craq-sim measures and what §5.0.5 predicts — but it is not deliverable in this
memory configuration. Note also that "`1,1,1` survives" is not good news: it survives because it demands
so little, not because the memory supplies much.

The sharpest single figure: at nominal, each cluster's share of DRAM is **17.6 B/cyc — 6.9% of its own
256 B/cyc NoC port**, and a *single* DM core issuing at the measured ~24 B/cyc already exceeds its
cluster's entire DRAM allowance. Four readers are ~5x oversubscribed before any threading argument starts.

### This is a property of the OP, not of Quasar — WH and BH are worse

Same model, same `d = 3`, across architectures (DRAM and core counts from *Matmul Introduction*,
Confluence):

| arch | DRAM | clock | B/cyc | units | per unit | **per engine** | DRAM floor |
|---|---:|---:|---:|---:|---:|---:|---:|
| Wormhole N150 | 288 GB/s | 1.00 | 288 | 64 cores | 4.5 | 4.5 | **1365 cyc/tile** |
| Wormhole Galaxy | 336 | 1.00 | 336 | 64 | 5.2 | 5.2 | 1170 |
| Blackhole Galaxy | 512 | 1.35 | 379 | 110 | 3.4 | **3.4** | **1782** |
| Quasar QSR1.A1 | 1000 | 1.33 | 752 | 32 clusters | 23.5 | 5.9 | **261** |
| Quasar QSR3.A2 | 1500 | 1.33 | 1128 | 32 | 35.2 | 8.8 | 174 |

A bf16 add moves 6 B and does 1 FLOP per element. **Per compute engine, DRAM supplies 3-6 B/cyc — under
one element per cycle — while a Tensix FPU retires ~1024.** The op is DRAM-starved by three orders of
magnitude on every generation. Three consequences:

1. **Blackhole is *worse* per core than Wormhole** — 3.4 vs 4.5 B/cyc. BH added 1.7x the cores against
   1.8x the bandwidth at 1.35x the clock, so per core per cycle it went backwards. **For a DRAM-bound
   elementwise op, adding cores has never helped on any generation.** That is the cross-architecture form
   of §5.0.12's cluster-count invariant.
2. **Quasar's improvement is smaller than the per-cluster figure suggests.** 23.5 B/cyc per cluster is 5x
   a Wormhole core, but a cluster holds 4 Neos — per engine it is 5.9 vs 4.5, only **1.3x**. Bandwidth
   roughly tracked the compute increase; the starvation ratio barely moved.
3. **Which is why elementwise ops are fused in production.** A standalone add paying a full DRAM round
   trip is bandwidth-bound on any hardware. Our threading and batching work is not wasted — it applies
   exactly where the operands are *already resident*.

⇒ **Strongest argument yet for F3, and it is not the one we had.** It is not that sharding is faster, or
that batching pays there: **sharding (`d = 0`) is the only regime in which this op is not
bandwidth-bound at all.** Everything measured in §5.0.5-§5.0.11 is only observable on silicon there.

### Consequence: the NoC can never bind, and ~4 clusters is the right placement

Sweeping cluster count against both ceilings (QSR1.A1, 1.33 GHz, `η = 0.75`, `d = 3`):

| clusters | DRAM used | NoC port used | marginal | wall, cyc per *total* tile |
|---:|---:|---:|---:|---:|
| 1 | 25% | 54% | 44.12 | **44.12** |
| 2 | 49% | 54% | 44.12 | 22.06 |
| **4** | **99%** | 54% | **44.12** | **11.03** |
| 8 | 198% | 54% | 87.2 | 10.90 |
| 32 | 790% | 54% | 348.7 | **10.90** |

**1. The NoC port never binds for DRAM-sourced traffic — at any grid size.** Port utilisation is **54% in
every row**: per-cluster demand is set by the marginal, not by how many clusters exist, so `n` cancels.
A cluster's DRAM allowance at full grid is 23.5 B/cyc against a 256 B/cyc port — it cannot physically
pull enough from DRAM to stress its own port.

**But the NoC does not simply take over when DRAM drops out.** That depends on where the operand sits,
and there are three cases:

| operand placement | DRAM | NoC | what binds |
|---|---|---|---|
| DRAM-interleaved | **binds** | 54% — cannot bind | DRAM, at every thread count |
| **L1-sharded, matching grid** (all three co-resident — *borrowed*) | **zero** | **zero** | **the compute chain, `176.5/C`** |
| L1-sharded, non-matching grid | zero | **NoC read** | the 256 B/cyc port |

**In the borrowed case the reader and writer do no transfer work at all** — the factory says so directly:
`DataflowBufferSpec::borrowed_from (reader/writer do no NoC work)` (`factory:46`), with all three operands
co-resident L1 shards on one grid. Unpack from L1, compute, pack to L1. **Zero DRAM *and* zero NoC.**
(This sharpens §5.0.4's "L1-sharded operands remove that term and put the 256 B/cyc port roof back in
charge" — true for a *non-matching* grid, where the shard is still NoC-read, but not for the borrowed
case, where there is no transfer to price.)

⇒ **Which inverts this branch's standing caveat for exactly that configuration.** Everywhere else our
numbers are upper bounds because craq-sim has no transport model. In the borrowed case there *is* no
transport, so the missing model costs nothing: the `176.5/C` roof is the real governing limit and the
4.00x is a **prediction**, not a ceiling. **craq-sim is not a compromised instrument for the
fully-resident case — it is the right one.**

⇒ **The NoC roof arrives with broadcast, not before.** A broadcast operand must be fetched from wherever
it lives (multicast, or read across clusters), so the port becomes live at milestone 2 — outer-dim 2.1 /
F13, subtile 2.2 / F8. That is also the first point at which Quasar's 4x NoC width over BH has anything
to pay for itself on.

**2. One cluster makes the simulated marginal real, at 4x the wall time.** At `n = 1` DRAM falls to 25%
and the issue rate becomes the binder, so 44.12 cyc/tile is genuinely achievable — craq-sim is modelling
the right thing there. But wall time is 44.12 against the optimum's 11.03.

**3. ~4 clusters is the placement, and it is 99% of the full grid.** Four clusters saturate DRAM *while
still running at full simulated speed*: 11.03 cyc per total tile against 32 clusters' 10.90. **The other
28 clusters contribute about 1%.** That is the operational form of the wall-time invariant — past the
saturation point wall time is pinned at `total_bytes / (S·η)` and extra clusters merely run slower each.

⇒ **Recommendation for a DRAM-interleaved shape: place the op on ~4 clusters, not 32.** Same wall time,
28 clusters freed for concurrent work or fusion. **And threading matters *more* in that placement, not
less** — with 4 clusters you want maximum threads per cluster, which is exactly `4,4,2`. The
configuration we tuned is right; the **grid** is what is oversized. Caveat: "~4" scales with `S·η`, so
it is order-of-magnitude while the shape of the conclusion is not.

⇒ **Still open, and now narrower:** the numbers above are targets and preliminary clocks, not silicon.
Running `tests/tt_metal/tt_metal/data_movement/` on Quasar hardware and adding the arch to
`noc_estimator` would give measured `S·η` directly. But it can only move the magnitude — every corner of
the sweep already agrees on the direction.

---

### 5.0.13 CORRECTED 2026-09-17: dataflow batching works at every clean `(R,C,W)` — the "cannot be written" claim was wrong

**What §5.0.9–§5.0.11 claimed.** Dataflow batching needs a DM role to own exactly one tile counter
(`C <= R and C <= W`), because `push_back(n)` credits the active counter and rotates once, and a DM kernel
can address only the active counter. Together with compute batching's `R <= C and W <= C` that forced
`R == C == W`, so batching was "legal only at `1,1,1` and `2,2,2`" and **`4,4,2` was "N=1 permanently, not
a defect"**. That sentence also set the M2.6/F15 deliverable to `2,2,2` n=8 at 48.70 cyc/tile, slower than
the default.

**Why it was wrong.** The premise is true; the conclusion did not follow. "Can address only the active
counter" does not mean "can use only one counter": the rotation IS the addressing mechanism. The DFB header
documents a thread round-robining over `tc_slots[0..num_tcs_to_rr-1]`. A role that owns K counters batches
*within* each counter and rotates *between* them. What the batched kernel got wrong was not the number of
calls but the **tile-to-slot correspondence**. At N=1 every push rotates, so consecutive tiles land in
consecutive counters, and counter `c` holds tiles `thread_id + c*T, +K*T, +2K*T, ...` — stride `K*T`, not
`T`. A batch drawn from one counter must stride by `K*T`. The kernel strided by `T`.

**The fix: a two-level (counter-major) walk.** Outer rotation over the K counters, inner batch strided by
`K*T`. `num_tcs` is a compile-time arg (`C/R` for the reader, `C/W` for the writer, else 1). The factory's
`C <= R and C <= W` `TT_FATAL` is removed. The shipping path (`dm_batch == 1`) keeps the flat loop verbatim
behind `if constexpr` — see the regression below.

**Verification.**
- Model (`debug/perf/tc_rotate_walk_equiv.py`): the single-level walk permutes in 24 of 56
  (config, role, N) cases; the counter-major walk reproduces the unbatched per-counter sequence in all 56,
  over 20 tile totals including the `full_limit` boundaries.
- Device (`debug/qsrnative/dm_batch_full_sweep.py`): **13 clean configs x N in {1,2,4,8} x 5 shapes = 260
  runs, all bit-exact**, every `num_tcs` pairing from 1/1 to 4/4.
- In-tree: `test_native_dm_batch_above_one_counter_is_bit_exact` (two arms, `dm_batch=8`, and a ring wrap
  at 65 tiles/cluster that the sweep above never reached). The model and the sweep live under `debug/`,
  which is not tracked; the test is what the tree keeps.
- Known-bad control: forcing `num_tcs = 1` (the old walk) at `4,4,2` N=8 **hangs**. The writer asks
  counter 0 for 13 tiles when only 11 are ever credited (`wait_front(5)` with 3 left). Rule:
  **credit-count errors hang; address errors corrupt.** #56194 is the latter.

**What this makes reachable.** `4,4,2` with both knobs at N=8 is blocked by **#56194 alone** (compute
batching corrupts above stride 1; ~75% of elements at stride 4, matching `1 - ceil(N/S)/N`). Measured with
the pack bug present. The timing is assumed representative because the instruction and credit counts do not
change; only the destination offsets are wrong.

| `4,4,2`, 128/256 t/c | fit | marginal |
|---|---|---:|
| N=1 (HEAD kernels) | `1115 + 44.00*T` | 44.00 |
| N=8, both knobs | `1929 + 26.56*T` | **26.56** |

**1.66x**, break-even ~47 tiles/cluster; the writer binds (56.1/2 = 28.05 predicted). So #56194 gates the
largest speedup available to this op, not an engine-efficiency option. It was under-rated when filed; the
issue is left as filed by decision. The combination has never produced a correct result, so its correctness
— including the compute kernel's remainder path, live for the first time — is unverified until the pack fix
lands.

**Two constraints that do survive.**
- `entries_per_thread % dm_batch == 0` (and `% num_tiles_per_cycle` when overridden). A batch writes n
  slots from `wr_ptr` and the ring wraps only afterwards, so a batch that straddles the counter end writes
  past it. Found in review. The sweep used depth 16 with N in {1,2,4,8}, all divisors, so it could not have
  caught this. Guarded now.
- A full batch needs `total_tiles >= N * C * 32` (`my_tiles >= N` per compute thread). Below that the
  compute kernel runs its tiles as one short batch, silently — which still exercises the pack path, and
  still corrupts at stride > 1 whenever that batch holds more than one tile. That is 1024 tiles at
  `4,4,2` N=8.

**A regression found and fixed on the way.** Nesting the per-tile loop cost the shipping configuration
16.5%: `44.00 -> 51.25` at `4,4,2` N=1, via +19 cyc/tile on the writer (`(83.5 + 19)/2 = 51.25`). An
`if constexpr` to keep the batch count constant changed nothing (51.25 to the digit). The cost is the loop
shape on a DM core with no branch prediction, not constant propagation. Fixed by keeping HEAD's flat loop
verbatim at `dm_batch == 1`; verified identical to HEAD (`1115 + 44.00*T`) by A/B against HEAD's kernels on
the same basis (`git show HEAD:<kernel> > <kernel>`, measure, restore — JIT makes this free).

**The classification error underneath.** Four kinds of constraint were collapsed into "illegal":

| kind | example | permanent? |
|---|---|---|
| hardware | `C <= 4`, `R + W <= 6` | yes |
| DFB implementation | intra-tensix DFBs are always STRIDED (`dataflow_buffer.cpp:1775`); the ratio rule | deep in tt-metal |
| a filed bug | #56194 | no |
| the shape of our kernel | one bulk `push_back` ⇒ `C <= R, C <= W` | **no — fixed here** |

Only the first row is architecture. §5.0.9–§5.0.11 presented the last two as the first, and the factory
`TT_FATAL` that encoded the misclassification was then cited as evidence for it.

**Superseded statements**, kept in place with a pointer here: §5.0.8 item 3; the §5.0.9 dataflow row and
"N=1 permanently"; §5.0.10 "there is no `4,4,2` batching number to quote"; §5.0.11 "NOT A BUG; not
expressible" and "`1,1,1` and `2,2,2`, full stop".

---

### 5.0.7 Next action and open questions

> **SCOPE OF EVERY MEASUREMENT IN §5.0.1-§5.0.6: `num_tiles_per_cycle = 1`.** The roofline constants,
> the 4.00x/2.69x gains, the zone split, the occupancy figures — all of it is the unbatched compute
> chain. Nothing here measures batched compute, because the configuration cannot currently be built.

**NEXT ACTION — raise `num_tiles_per_cycle` from 1.** Two independent things pin it, and both are ours:

1. **Capacity — and `N > capacity` HANGS, it does not merely slow.** The compute kernel needs N available
   on `in0`, N on `in1` *and* N free on `out` (`wait_front(n)` ×2 + `reserve_back(n)`); if N exceeds any of
   those capacities the wait is never satisfiable. So `N <= capacity` is a **correctness** bound and
   `N <= capacity/2` is what buys overlap. `capacity = num_entries / max(P,C)`
   (`dataflow_buffer.cpp:1139`) and the factory sets `in_entries = entries_per_thread * max(R,C)`, so for
   the input DFBs **per-thread capacity == `entries_per_thread`** exactly. Double buffering a batch of N
   therefore needs **`entries_per_thread >= 2N`**: today's default of 2 permits only N=1, before any other
   rule is consulted. N=2 needs depth 4, N=8 needs depth 16 — all inside the knob's range (measured
   working to 16, ceiling 63 at `4,4,2`).

   **The reader's batch is NOT the reference.** A compute batch larger than the reader's push size is both
   legal and, wherever the `C` term binds, the objective: steady-state throughput is `min(stage rates)`,
   which batch granularity does not change, and shrinking the compute chain hands the bottleneck to the
   reader by design (at `4,4,2`, `C` 44.12 → reader 41.25). Compute idling in `wait_front` is then spare
   capacity, not lost work. The genuine cost of a large N is **pipeline fill** — nothing is produced until
   N tiles exist, which lengthens a prologue already worth 38% of span at the benchmark shape. So N trades
   steady-state cost/tile against small-tensor latency; it is not bounded by what the reader does.
2. **Stride, and the guard tests it correctly.** `stride_in_entries = max(num_producers, num_consumers)`
   (`dataflow_buffer.cpp:1140`) — **not** the pairing ratio; that is `num_tcs_to_rr`, a transaction-counter
   round-robin count and a different quantity. So our `TT_FATAL`'s `max(R,C)` and `max(C,W)` *are* the two
   rings' strides, and the guard is right. At `4,4,2` both rings have **stride 4**: each thread's entries
   are interleaved with its siblings', so a batch of N is non-contiguous by construction. **This is
   inherent to interleaved operands** — a real NoC-filled ring is shared by `R` producers, so multi-thread
   implies stride > 1.

**Why sharded is the easy case.** With `borrowed_from` set, the DFB is backed by the resident L1 shard and
"reader/writer do no NoC work" (factory :46, :648) — there is no ring for producers to share, so stride
collapses and a batch of 8 is just 8 tiles of an already-present shard. That is exactly what
`all_borrowed` gates. Note the `TT_FATAL` still applies: sharded + N=8 is legal only at `1,1,1`, since at
`4,4,2` the strides are 4 regardless of borrowing.

**What the WH/BH precedent does and does not cover.** Production `binary_ng` sets
`num_tiles_per_cycle = 8` (FPU, 16-bit) whenever the broadcast type is `NONE` **and** all three tensors
are sharded (`binary_ng_program_factory.cpp:1018-1040`). That establishes one thing: `llk_pack` can pack a
batch of 8 out of DST into a **contiguous** CB. It establishes **nothing about the interleaved
configuration**, which was never implemented or tested on WH/BH either — so nothing anywhere exercises a
producer pushing 1 while the consumer waits for N, input-ring sizing for that case, or the pipeline fill
it implies. **Interleaved + N > 1 is unbuilt on every architecture**; on Quasar we would be first.

Two unknowns, then, and they must not be moved together: (a) does interleaved producer-1/consumer-N work
at all, and (b) does the **strided** address walk survive the pack path — CBs are contiguous, so WH/BH
never probe it. Everything else on the Quasar side is verified stride-aware: the pointer/credit math
(`wr_ptr += n * stride_size`, `.inl:189,198,243`), the scoped-lock walk over n entries (`.inl:454-470`),
and explicitly the unpack-side tile addressing (`offset_address = stride_size * tile_index`,
`cb_api.h:146`, commented *"Per-tile spacing is stride_size, not entry_size"*).

**Two independent gates, and only one is a Quasar problem.** WH/BH require `NONE` broadcast AND all-sharded;
the sharding half is a *conservative gate nobody revisited for interleaved* (the code comments the `1` as
"Conservative default", and notes of SFPU that 4 "should handle... only 2 works, need further
investigation"). The broadcast half is real and independently asserted in both Quasar factories: **subtile
broadcast requires `num_tiles_per_cycle == 1` regardless of sharding**, because the broadcast operand is
reused across tiles, so "advance every DFB by N" is wrong for it. Our slice is no-broadcast, so only the
sharding-shaped gate and the stride question apply.

**Experiment, in two steps so the two unknowns stay separated** (not yet run):

| step | config | stride | isolates | change needed |
|---|---|---:|---|---|
| **1** | `1,1,1` interleaved, `entries_per_thread=4`, N=2 | **1** (contiguous) | interleaved producer-1/consumer-N, the part no arch has built | make the assignment reachable for interleaved + raise depth. **The `TT_FATAL` passes unmodified** — `max(1,1) == 1` |
| **2** | `4,4,2` interleaved, `entries_per_thread=4`, N=2 | 4 | the strided walk, given step 1 passed | additionally relax the `TT_FATAL` |

**Both steps test the MECHANISM, not the payoff, and that is deliberate.** They batch compute only, and
§10's lever table already measured why one-sided batching cannot pay: reader-only batching returned
**1.08x** even though `RD_RSV` falls as `1/n`, because the writer stayed per-tile and floored the span.
Whichever stage keeps per-tile granularity becomes the new floor — the `max()` law of §5.0.3 applied to
overheads instead of roofs. **A payoff measurement requires batching reader, compute and writer
together**, which is a larger change and a separate step 3. Sizing for it: `RD_RSV` is **70.9 of the
reader's 165 cyc/tile (43%)** against `RD_BAR`'s 8.0, so the reader-side prize is the `reserve_back`
calls, not the barrier — and it is far larger than the compute side's ~6% interleaved cap. Guideline when
that is built: `N_reader <= N_compute` (compute proceeds at `N_compute`; larger producer chunks only delay
its start), with equality the natural point. Measurement caveat: the SUM zones are per-*call* and so
**flatter** batching (1.21x instrumented vs 1.08x clean) — quote the clean number.

Step 1 is the smallest change that answers the harder question first, and it needs no guard relaxation.
**It also tests the withdrawn `131/45` chain decomposition:** at `1,1,1` the `C` term binds at 176.5
against the reader's 165, so the model predicts the chain falls to ~110.9, the reader takes over, and the
marginal lands near 165 — a **~7%** gain. Either outcome converts an unverifiable model into a data point.
Step 2's reading: corruption localised to the packed output confirms the strided pack path as the blocker;
bit-exact means the guard was inherited caution and the lever is available.

**Open questions.**

Q1-Q4 feed the emulator campaign (design §7.5); Q5-Q6 are platform/tooling asks. They live here rather
than on the tutorial page so answers can land — strike through and date them here — without the published
page going stale.

1. **What is `T_fix` for Quasar?** One number converts our in-flight footprint into an expected
   bandwidth fraction and decides whether depth or transaction size is the bigger lever; the O2O model
   has everything else. Highest-value question on this list.
2. **Should `entries_per_thread` default to >= 4?** We ship 2, giving Q=1 and ~39% of achievable
   per-core bandwidth. The footprint at depth 16 is still ~2.7% of L1. craq-sim reports ~0 gain by
   construction — that must not be read as a reason to leave it.
3. **Is 2-tile batching the single highest-value change?** It crosses the ~2.36 KB issue/transport knee
   AND raises Q. The `capacity >= 2n` coupling means it must be swept together with depth, never
   separately.
4. **Redo the lever-composition arithmetic on the native basis.** Threads and knobs are not independent
   levers — batching removes DM instructions that threads already divided by `R`, so per-lever gains do
   not multiply. Derive any composition from the native-basis measurements (4.00x throughput / 2.69x
   latency, knobs <= 1.17x), never by multiplying per-lever numbers.
5. **What is a "Neo"?** The Quasar HAS says 32 Neo instances per chiplet, each four Tensix cores plus a
   shared 4 MB L1 — so a Neo *is* the cluster. tt-metal's profiler emits four `QUASAR_NEO{0..3}` per
   CoreCoord. These docs follow the profiler; worth settling with the arch team before it spreads
   further.
6. **Can Quasar be added to `noc_estimator`?** Its `Architecture` enum is Wormhole and Blackhole only.
   §5.0.4 carries the mechanism (run the `data_movement/` microbenchmarks on Quasar and fit the arch);
   this entry is the ask.

---

## 6. The chosen demonstrator case

### 6.1 Pick

**Tensor-tensor, no broadcast (`SubtileBroadcastType::NONE`), TILE 32×32, bf16, `add`, all three operands
INTERLEAVED in DRAM** — i.e. the "normal path" of the current factory, on a large-enough tensor.

### 6.2 Why this makes the Quasar gain most visible

1. **Its entire runtime is the thing Quasar changes.** A no-broadcast eltwise add moves 3 tiles per output
   tile (read a, read b, write c) and does one FPU pass. Today those transfers are *serialized by an
   explicit barrier per tile pair* on *one* DM core with a *2-entry* ring — the pathological opposite of
   what the hardware offers (4 NoC read ports, 4 write ports, 6 user DMs, 4 MB of ring space).
2. **Every named Quasar feature is on the critical path**, so the win is attributable:
   6 DM cores (STRIDED multi-producer reads + multi-consumer writes), 4 compute engines (STRIDED consumer),
   multi-threading (SPMD on both sides), shared SRAM pool (deep rings), and implicit sync (many outstanding
   NoC transactions instead of one).
3. **There is a clean, already-green baseline**: this exact configuration is functionally passing today via
   `ProgramFactoryMetalV2`, so the comparison is a pure performance delta with no functional risk, and the
   fallback stays available if the native path regresses.
4. **It is the configuration that DRAM-resident model eltwise actually hits.** (The sharded residual-add
   config in ResNet does zero NoC work by design.)
5. **Simplest correctness surface of any case**: no subtile broadcast, no reader-side software fill (so no
   DM-cache-coherence/fence hazard), no scalar fill, no mixed dtypes, no row-major.

### 6.3 Explicitly rejected as the *first* target

- **All-borrowed sharded (resident-L1) path** — reader/writer do **zero NoC work**; there is no data
  movement to parallelize, so at most the 4-thread compute lever applies. It cannot demonstrate the DM
  architecture at all. (It remains interesting later for 4-Tensix compute scaling.)
- **Broadcast (row/col/scalar/mixed)** — this is where `ALL` access patterns and remapper fan-out become the
  headline, and it is the more *representative* long-run case, but as a first demonstration it moves less
  data (the broadcast operand is read once and reused), so the measured gain is smaller and harder to
  attribute; it also drags in the reader software-fill coherence/ordering hazard (`TODO(#51291)`), the
  ROW-LLK-vs-COL-reader-fill load-balancing design, and `num_tiles_per_cycle == 1`. Phase 2.
- **fp32 / int32** — fp32 add/sub are SFPU on Quasar (different pipeline), int32 is known-broken on the DFB
  compute path. Keep dtype constant while changing the dataflow architecture.

---

## 7. Where the code goes: a third variant alternative

### 7.1 The existing seam

`device/binary_ng_device_operation.hpp:110-138`:
```cpp
struct ProgramFactory { static ProgramDescriptor create_descriptor(...); };            // general fallback
struct ProgramFactoryMetalV2 { static ProgramArtifacts create_program_artifacts(...); }; // current DFB path
using program_factory_t = std::variant<ProgramFactory, ProgramFactoryMetalV2>;
static program_factory_t select_program_factory(...);   // impl at binary_ng_device_operation.cpp:646
static bool matches_metal_v2_slice(...);                // gate predicate, :469
```

### 7.2 The decided shape

Add a **third alternative**, `ProgramFactoryQuasarNative`, also satisfying
`ProgramSpecFactoryConcept`, in its own translation unit (e.g.
`device/binary_ng_quasar_native_factory.cpp`):

```cpp
using program_factory_t =
    std::variant<ProgramFactory, ProgramFactoryMetalV2, ProgramFactoryQuasarNative>;
```

Routing precedence in `select_program_factory`: **native gate → metal_v2 gate → descriptor**. The native
gate is a *narrow* predicate (`matches_quasar_native_slice`) over the §6.1 slice, and additionally
requires Quasar (`is_gen2_arch()`), since multi-threading is rejected on Gen1 by construction.

Why this is safe and cheap:
- The framework validates **each variant alternative independently** — `AllFactoriesValid` folds over all
  alternatives requiring each to satisfy exactly one factory concept
  (`ttnn/api/ttnn/operation_concepts.hpp:188`, used at `:208`). Variant arity is not fixed at 2.
- `ProgramFactoryMetalV2` and the descriptor path are untouched → zero regression risk to the current
  green test suite, and the fallback is a live reference for A/B measurement.
- Device-op contract (validate / output specs / program hash / skip_launch) is shared and unchanged.

### 7.3 Things to get right in the new factory (carry into design)

- **Both factories must be reachable in one build/run** for A/B. Options: an env var, an operation
  attribute, or a test-only selector. Decide during design; prefer something that does not leak into the
  public API.
- **Program cache**: the native factory must key on whatever it makes shape-dependent (thread counts, ring
  depth, per-thread tile counts). The op currently uses the framework default attribute-reflection hash,
  which **DOES include tensor shape for this op** (measured: 64 tiles → 1 cache entry, 256 tiles → 2, via `Tensor::attribute_names` → `tensor_spec` → `logical_shape`).
  The "excludes tensor volume" claim is true of **production** `eltwise/binary_ng`, which has a custom
  `tensor_args_t::to_hash()` hashing only dtypes + memory configs
  (`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.hpp:118-125`) — which is
  precisely why production needs `override_runtime_arguments`. The **quasar** op has no such override, so
  it gets full reflection. Do not carry production's constraint across.
  The real constraint is narrower: on a cache hit the adapter re-applies **only tensor bindings**
  (`UpdateTensorArgs`), so every non-tensor per-cluster arg must be a function of hashed inputs — and
  `worker_grid` is NOT hashed today, which is a hang risk once per-thread counts are baked in.
  `ProgramRunArgs` (and `DFBRunOverrides` if ring depth is dynamic), or folded into the hash.
- **Divisibility**: with R producer threads and C consumer threads on a DFB, keep `num_entries` a multiple
  of `lcm(R, C)`, prefer divisible R:C ratios, and make per-thread tile counts agree on both sides —
  mismatched counts are a credit mismatch, i.e. a hang, not a wrong answer.
- **One `WorkUnitSpec` budget**: DM threads across reader(s) + writer(s) ≤ 6, compute threads ≤ 4, one
  compute kernel. Candidate splits to evaluate: reader(4)+writer(2), reader_a(2)+reader_b(2)+writer(2),
  reader(3)+writer(3) — **ILLEGAL**: "larger divisible by smaller" is hard-`TT_FATAL`'d (as two
  *directional* checks, `:1267-1271` when C≥R and `:1278-1283` when R≥C, and **STRIDED-only** — `ALL` has no
  ratio constraint at all, `:1246-1259`),
  so 3 threads cannot pair with C=4. With C=4, R and W ∈ {1,2,4}, making R=4/W=2 the only 4-Tensix config
  that saturates the 6-DM budget.

---

## 8. Constraints and landmines checklist

Dataflow / DM:
1. Never mix explicit CB ops with implicit sync on the same DFB → 16-bit counter double-count →
   `TILE_COUNTERS` fault.
2. Sub-tile NoC ops auto-post one credit per op → posted outruns acked → stall. Keep transfers ≥ one entry.
3. DM→DM `ALL` + implicit sync: known runtime gap (the DFB matrix auto-skips it).
4. **DM core D$/L2 is incoherent with TL1 — the COHERENCE half is now handled by the platform API, the
   ORDERING half is still open.** Main commit `a00dd45324b` (#52769, "Have DFB get_read/write_ptr() APIs
   return the uncached address ranges on Quasar DM") makes `dfb.get_write_ptr()/get_read_ptr()` hand out
   the **uncached L1 alias** on Quasar DM, and `noc.h:103-108` maps such an address back to the cached
   range when it reaches a NOC API. The binary_ng kernels no longer hand-add `MEM_L1_UNCACHED_BASE` (that
   PR edited `reader_row_col_mixed_bcast_dfb.cpp` and `writer_scalar_dfb.cpp`). **Do not re-add it —
   double-aliasing.** Note `QUASAR_PARITY_GAPS.md:121` is now **stale** on this point.
   Still true: reading packer-written L1 needs `invalidate_l2_cache_range`; `invalidate_l1_cache()` is a
   **no-op** on Quasar; and **the release fence before `push_back` is still unmitigated** —
   `TODO(#51291)` with the full analysis survives at `reader_row_col_mixed_bcast_dfb.cpp:40-55`
   (bare `asm("fence")` = `iorw,iorw`; `__atomic_thread_fence(RELEASE)` emits only `fence rw,w`, which does
   **not** order the overlay register write). craq-sim is **blind to store ordering** — green on sim is not
   proof. (Only bites once we do reader/writer software fill, i.e. phase 2 and the scalar path.)
5. Tile counters are 16-bit → ring depth cap, but on `capacity` (= `num_entries / max(R,C)`), not on
   `ring_trisc_units`. Plus an **unguarded `uint8_t` cliff** on `threshold`/`num_entries_per_txn_id` above
   **255** entries — not ~510: `num_txn_ids` falls back to 1 rather than staying ≥2 (§1.3).
6. Two DM kernels historically collided on one NoC with a silent hang and no validator (Gen2 auto-assigns,
   so this should not recur, but watch for it).
6b. **`SubtileBroadcastType::NONE` does not mean "no broadcast".** `get_subtile_broadcast_type` takes four
   scalars — H and W only (`binary_ng_device_operation.cpp:198-200`) — so **leading-dim (N/C/D/nD) broadcast
   is `NONE`**. Any code that infers "operand shapes are equal" from it is wrong, which breaks both linear
   page addressing (`next_c_shift`/`next_n_shift` become nonzero) and any tile count read from `input_a`
   instead of the output. The quasar `no_bcast` suite cannot catch it — it passes one shape for *both*
   operands everywhere. Gate on full-rank `padded_shape` equality instead.
6c. **A copied program factory is a duplicate *symbol*, not just an ODR hazard.** Out-of-class member
   definitions like `create_program_artifacts` have external linkage, so a wholesale factory copy fails at
   **link** even with `TT_UNITY_BUILDS=OFF`; the class must be renamed. Separately, a bare `namespace {`
   collides under the unity build — measured at **32** redefinitions for `binary_ng_metal_v2_factory.cpp`,
   most of them `constexpr const char*` path literals rather than functions. Wrap the whole anonymous-
   namespace body in `CMAKE_UNIQUE_NAMESPACE` (`binary_ng_program_factory.cpp:19` is the idiom).
7. `qsr_async_read_page`-style direct L1→L1 copies exist as a workaround where sim drops NoC self/loopback
   reads (`tests/.../data_movement/common/kernels/common.hpp`).

Compute / LLK:
8. **Init before every op use when DFB ids change** — buffer descriptors (L1 addresses) are programmed in
   the init call. Alternating output DFBs per block requires re-init of the packer.
9. The real TEN-4746 rule is **same-DFB WAIT→retire**, not "no two counter ops
   back-to-back": three consecutive counter ops on three *different* DFBs are legal (which is why the
   shipped kernel passes). The stricter paraphrase below would drive unnecessary interposed dummy copies.
   `wait_tiles/pop_tiles/push_tiles/wait_for_free` on the SAME DFB need a TDMA (unpack/pack) LLK
   must sit between them (HW constraint, TEN-4746).
10. **No compute-side implicit-sync opt-out** → compute self-loop DFBs (our activation and `llk_post`
    intermediates) must be credit-balanced by construction.
11. Use semaphore-based Dest synchronization on Quasar (not the data-valid scheme) — this was the central
    ResNet conv blocker.
12. `compute_kernel_hw_startup` exactly once.
13. Quasar shape validators are stricter than WH/BH (e.g. `y_dim != 16` when `z_dim == 4` rejected;
    non-power-of-2 `face_r_dim` rejected).
14. Errata to keep in view when picking activation primitives: SFPU 2-cycle-op NOP insertion gaps
    (TEN-4581/4605), packer-RELU leaving one 16×16 face unclamped (found in ResNet conv), MOP double-loop
    with `loop1_len == 0`, `INC_SRC_TILE_FACE_ROW_IDX` bugs.

Environment:
15. Debug flags for bring-up: `TT_METAL_LLK_ASSERTS=1`, `TT_METAL_WATCHER=10`; keep
    `TT_METAL_WATCHER_DISABLE_ASSERT=0` / `..._NOC_SANITIZE=0` on the emulator. Watcher `0x19` (TRISC
    instruction-buffer interrupt) is frequently a *watchdog* symptom of compute idling on DM, not a fault.
16. Debug method that works: shrink to one cluster → one Tensix → minimum DM cores → minimum tile count;
    then comment out tile-counter APIs in pairs to separate LLK from DM sync.
17. Kernel `.cpp` edits are JIT-compiled (no `build_metal.sh`); host-side `.so` changes need the manual
    `cp` into `ttnn/ttnn/_ttnn.so`.

---

## 9. Design levers: what craq-sim measured, and what that does not tell us

**Re-derived from measurement on 2026-08-20; data and method in
`.link_to_claude/plans/quasar-native-binary-ng-review-findings.md` §K-MEASURED-1..4.** Numbers are craq-sim at T=40 tiles/cluster, so
they bound instruction-count effects and say nothing about contention — **and for the latency-hiding levers
they are floors, not ceilings.**

| lever | craq-sim result |
|---|---|
| DFB call batching (reader, n=2) | **1.08×** — and batching the *writer* is negative (serialization) |
| `implicit_sync` | **≤1.10×** — 22 of 228 cyc/tile, barriers pre-satisfied. **≫ sim, unmeasured** — a barrier is a real stall |
| ring depth | **1.02× on craq-sim** — 1→40, asymptotes by depth 4. A **lower** bound: depth hides transfer latency and the sim has none, so this is the one lever the sim undervalues |
| **multi-DM STRIDED producers/consumers (R, W)** | **unmeasured — the only untested major lever** |
| 4-thread compute | unmeasured; blocked on tt-llk #1678 |

1. **Multi-DM threads (R, W).** Untested, and now the whole question: everything else has been measured
   small. Faithfully modelled as instruction-count reduction, but **unpenalised by contention**, so craq-sim
   will give an upper bound — the 4 NoC-read / 4 write L1 ports and DRAM bank conflicts are what bound it on
   silicon. Cannot be measured without the native factory (`num_threads > 1` is a host-side path).
2. **4-thread compute.** Blocked. Note compute contributes only ~0.35 cyc/tile to the interleaved critical
   path today, so this buys headroom for (1) rather than time by itself.
3. **DFB call batching, reader side only, n=2.** Banked 1.08×. Do **not** batch the writer.
4. **Ring depth** — an *enabler* for (3), 1.02× alone. Do not plan on `DFBRunOverrides`: per-DFB overrides
   break the `in0`/`in1` pairing invariant (design §4.2).
5. **Implicit sync** — ≤1.10× here. May still matter on silicon where a barrier is a real stall, which is
   why it belongs in the emulator campaign rather than the sim sweep.
6. Secondary: IDMA staging/conversion; FPU/SFPU overlap (structurally blocked, design §8); grouped-allocation
   remapper mode.

**Two lessons worth carrying to the next op.** The a-priori ranking failed because it ranked by
*architectural narrative* — barriers look expensive, deep rings look important — when the cost was in the
per-tile overhead of the API expressing the dataflow. And the measured *attribution* then failed to predict
the *recoverable* gain: DFB calls are 56% of per-tile cost, cutting call count 8× cuts call cost ~8×, and
end-to-end gain still caps at 8% because the pipeline re-absorbs it. **Attribution locates cost; only an
experiment reveals what is recoverable.**

## 10. craq-sim: what it can and cannot measure

Standalone reference. Verified against `/workspaces/craq-sim` @ `5ced8886` (2026-08-20) by reading the
simulator source, plus our own runs. **Every "cannot" below is about the simulator's timing model, not its
functional fidelity** — craq-sim is functionally good enough that a whole op family was brought up on it.

### 10.1 What it IS, per its own documentation

`PERF_CALIBRATION.md` states the goal as adding "performance predictive awareness to craq-sim **without
changing its primary role as a functional simulator**", targeting "useful prediction and bottleneck
classification … **not cycle-accurate RTL replacement**", with explicit non-goals including "do not make
craq-sim cycle-accurate in the first implementation".

⇒ Treat it as a **functional simulator plus an offline calibration model**, not a performance model.
Docs: `PERF_CALIBRATION.md`, `PERF_AUDIT.md`, `docs/perf/{README,MODELING_WORKFLOW,MULTICHIP_PERF_MODELING,
TTNN_SILICON_HANDOFF}.md`, `docs/perf/calibration/`.

**Important update: the calibration has largely landed — but as an offline regression model, not as a change
to the simulator.** `PERF_CALIBRATION.md:31-33` states the architecture: *"Treat craq-sim as a deterministic
feature extractor and silicon profiler data as the target."* The git history on those paths shows a GBDT
fitted over a 31-shard silicon sweep (~11.7k targets). Nothing inside the simulator gained a cycle cost —
verdicts in §10.4 are **current, not stale** (the only latency in the tree is `eth_latency_cycles`, default 0,
plus a wall-clock read delay gated `#if TT_VERSION <= 1`).

**Why this matters to us:** §10.5.1 says "report the shape, not the multiplier" on the premise that nothing
can supply a multiplier. A fitted craq-sim-features → silicon-time model *is* a multiplier estimator. Before
committing to defer magnitude entirely to an emulator campaign, check whether eltwise/binary kernels fall
inside that model's calibrated envelope. Coverage and model quality are **unevaluated** — treat as a lead.

### 10.2 Three instruments that work today

| instrument | how | what you get |
|---|---|---|
| **Global cycle count** | free, printed at exit | `[<cycles>] <wall>s (<rate>)` from `g_clock` (`src/sim.cpp:502-513`) |
| **Device profiler** | `TT_METAL_DEVICE_PROFILER=1` (no rebuild — profiler is on by default) | per-RISC kernel spans in `generated/profiler/.logs/profile_log_device.csv`; RiscTypes `QUASAR_DM0-7`, `QUASAR_NEO0-3_TRISC0-3`; cycles-since-reset stamps. **The only cycle source, and the only per-core one** — §5.0.2 has the measured role map (readers/writers/pipes by RISC name) and the occupancy method |
| **craq-sim perf trace** | `TTSIM_PERF_TRACE=1 TTSIM_PERF_TRACE_PER_DISPATCH=1 TTSIM_PERF_TRACE_OUT=<dir>` | `ttsim_perf_trace.tsv`: per-engine instruction counts, DFB op counts (`cb_waits/reserves/pushes/pops`), `kernel_launches`, per-pipe **stall** cycles (`src/sim.cpp:143-150`) |
| **Profiler zones inside a kernel** (DM cores included) | wrap a region in a device-profiler zone | exact cycles for a **sub-kernel region on any core**. The profiler's device-side stamp is a direct read of `NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_WALL_CLOCK_0` (`tt_metal/tools/profiler/kernel_profiler.hpp:218-225`), which craq-sim answers with `g_clock` verbatim (`src/tile.cpp:1768`) and — unlike Gen1 — with **no read delay** (`src/riscv_impl.h:612` gates it on `TT_VERSION <= 1`). In-tree prior art: `tests/tt_metal/tt_metal/api/dataflow_buffer/dfb_init_timing_bench.cpp` (`TT_METAL_MEASURE_DFB_INIT_TIME=1`). |
| **DFB credit event log** | `TTSIM_QSR_DFB_TRACE=1`, `TTSIM_QSR_DFB_COUNTER_TRACE=1` | every credit post/ack with `posted→M acked=K` per `(tensix, counter)`, plus a distinct *blocked* event carrying capacity (`src/riscv_impl.h:1941-1948`, `:2235-2252`, `:2521-2530`). Post-process for the **ring-occupancy trajectory** — max occupancy, whether the ring ever fills, at what depth. Event-ordered, not clock-stamped; pair with a profiler zone for time. |

Both profiler and perf trace can run in the **same** process — do that, so numbers never get mixed across
runs. **The third row is the one that repairs the DM blind spot**: the perf trace cannot see the DM cores, but
a profiler zone can, so per-stage attribution does not need a ring-depth trick. Keep zones **per loop, not
per tile** — the buffer saturates near 22 RISCs × 125 zones.

### 10.3 What it models faithfully — and this is the part that matters for us

- **Instruction issue on the RISC-V (DM) cores.** Those step exactly once per simulated cycle
  (`src/libttsim.cpp:2272`, `:2080-2093`), so **cycles/tile ≈ per-core instruction count on the DM path** —
  which is the 76% of this op that matters. `g_clock` is the *max* over cores, never a sum: all cores step in
  lockstep and the clock advances once per global cycle.
- **NOT uniform on Tensix.** The RTL-aware scheduler is **on by default**
  (`TT_METAL_SIMULATOR_TENSIX_RTL_AWARE_SCHEDULER`, `src/libttsim.cpp:246-252`) and its issue-class loop
  (`:2194-2261`) retires up to `TENSIX_INST_PIPES = 3` (`src/sim.h:276`) backend instructions per cycle when
  the pipe heads fall in distinct classes. **Tensix instruction counts are compressed up to 3× against the
  clock**, so a compute-thread sweep is on a different scale than a DM-thread sweep.
- **Pin the cycle model.** `..._TENSIX_RTL_AWARE_SCHEDULER=0` restores a fast-drain scheduler where Tensix
  work is nearly free (`:2313`); `..._TENSIX_PIPE_ISSUE_BUDGET` (`:238-243`) and
  `..._PARALLEL_TENSIX_TILE_CLOCK` (`:125-132`) also change the model or the schedule. Record all three with
  every run; determinism below was verified for the default configuration only.
- **Thread parallelism.** Splitting work across DM cores or Tensix engines genuinely reduces per-core
  instruction count, and that shows up honestly.
- **Determinism.** Bit-identical across runs (verified: 7781/8019/7492/8036/7531 and sim clock 17934, twice).
  Two consequences: A/B deltas are exact, and **races are deterministic** — they fire on every run or never.
- **Speed.** ~12-15 s per run at 1280 tiles, so ladders and sweeps are cheap.
- **DM cache hierarchy** (D$/L2/TL1) — coherence *is* modelled, which is how the reader-fill coherence bug
  was caught.

### 10.4 What it does NOT model — with the mechanism, so the verdicts are checkable

| not modelled | mechanism |
|---|---|
| **NoC transfer cost** | `qsr_rocc_copy_bytes` is a host `memcpy` loop through a 256-byte stack buffer executed *inside* the issue instruction (`src/riscv_impl.h:1721-1738`). Transfer size never becomes cycles. |
| **Read/write barrier cost** | data lands and the response counter increments on adjacent lines (`src/tile.cpp:2399-2401`), so `async_read_barrier()` is free. |
| **DFB credit batching / the DM0 ISR** | `qsr_rocc_post_dfb_counter` increments `posted` by **1, per transaction, at issue** (`src/riscv_impl.h:2229-2243`, called from `:2827`). `PER_TR_ID_IP_*` reads hardwired 0 (`:3019-3028`); there is **no asynchronous interrupt delivery anywhere**. So implicit sync is *qualitatively* different from silicon: unbounded per-thread depth instead of batch-gated. |
| **NoC/DRAM contention or queueing** | `set_noc_outstanding` is `#if TT_VERSION <= 1` so the outstanding count is permanently 0 on Quasar (`src/tile.cpp:1049-1066`); `get_vc_space` returns `0xffffffff` (`src/riscv_impl.h:3070-3075`); DRAM is a flat `memcpy` (`src/tile.cpp:5601-5605`). Only *ethernet* latency exists (`eth_latency_cycles`, default 0). |
| **Store ordering** | every store is applied synchronously with no store-buffer state ⇒ the release-fence hazard (#51291) **cannot be reproduced or regressed here**. Coherence yes, ordering no. |
| **NoC event counters, on our path only** | the perf trace's `noc_reads`/`noc_writes`/`noc_bytes`/`dram_*_bytes`/`l1_*_bytes` read **0**, but *not* because the tracer is unwired — `ttsim_perf_trace_noc` is called from the `TT_VERSION == 2` branch of `noc_cmd_ctrl` (`src/tile.cpp:2390`, `:2402`, `:2680`). The reason is that Quasar DM kernels move data through ROCC command buffers in `riscv_impl.h`, which never calls the tracer (`grep -c g_perf_trace src/riscv_impl.h` → **0**). ⇒ byte/transaction counts are a **one-call-site craq-sim patch**, not an emulator errand. Re-verified still 0 at `ad401613` (2026-09-04) with a second witness — see §5.0.2. |
| **Cache and locality timing** | coherence is modelled but *cost* is not: a D$ or L2 miss is **zero cycles** — the QSR DM L1 read/write paths return unconditionally (`src/riscv_impl.h:836-845`, `:902-911`) with no replay. L2 is idealized to one slot per TL1 line so it never conflict-evicts (`src/sim.h:292-297`). ⇒ any lever that improves DM locality, or trades cached for uncached-alias access, shows **exactly zero** delta here. |
| **Anything DM-side in the *perf trace*** | `stall[engine]` is incremented per cycle per **Tensix** instruction returning `executed == false` (`src/tensix.cpp:18299-18302`); the DM RISC-V cores contribute nothing (`grep -c g_perf_trace src/riscv_impl.h` → 0). Since the DM path is ~76% of our measured cycles, **the trace** is blind to most of the op. **This is not true of the profiler** — see §10.2's third row: DM-side sub-kernel regions *are* measurable. |

### 10.5 Three traps that produce *wrong conclusions*, not missing ones

1. **The linear-scaling illusion — the one that bites the perf goal.** With no NoC, DRAM-bank or L1-port
   contention anywhere (§10.4), splitting a pure instruction-cost loop across R cores scales ~`1/R` **by
   construction**. On silicon the same sweep is bounded by DRAM bank conflicts and the 4-NoC-read-port
   budget per Tensix (§1.2). So a clean `1/R` curve here confirms *that the work was divided*, and says
   nothing about *how much silicon will gain*. Report the shape, not the multiplier.
2. **The ring-full false confirmation.** On a full ring the simulator replays the issue instruction
   (`p_hart->pc -= 4`, `src/riscv_impl.h:2771-2778`) at 1 cycle per retry. So a ring-depth sweep **will**
   show a knee — plausibly right around 4 — which looks exactly like a transaction-concurrency effect while
   being instruction replay. Do not read a depth knee here as evidence about latency hiding.
3. **Deterministic races look like correctness.** Because interleaving is fixed round-robin at
   instruction granularity, a race either fires identically every run or never. A green multi-thread run is
   therefore **evidence-free** about concurrency safety, and a bit-exact oracle can pass on the luck of one
   schedule (this is exactly the situation with tt-llk issue #1678).

### 10.6 Practical verdict per lever

| lever | measurable on craq-sim? | why | emulator |
|---|---|---|---|
| DM thread count (R, W) | **Whether: yes. How much: no** | instruction-count reduction is real, but the *magnitude* is unpenalised by contention (§10.5.1) | **≤ sim** — contention, NoC ports, txn-id rendezvous, DM0 ISR |
| **DFB call batching** (`reserve_back(n)`/`push_back(n)`) | **Yes — measured both ways** | the term is 56% of the baseline and `RD_RSV` falls as `1/n` when batched, but reader-only batching is **1.08×**: the span floors at ~196 because the writer is still per-tile. Batch both sides. Note the SUM zones are per-*call*, so they **flatter** batching (1.21× instrumented vs 1.08× clean) — quote the clean number | **≤ sim** — may vanish if DRAM-bound |
| Compute thread count (C) | Yes in principle | blocked by tt-llk #1678, a green run would be evidence-free (§10.5.3), and `TENSIX_DEFAULT_LINGER` becomes live here | ≤ sim |
| `entries_per_thread` (ring depth) | **Yes, but only a floor: 1.8% ⇒ 1.02×** | measured across depth 1→40, asymptoting by depth 4 — deterministic real signal, immaterial magnitude. Treat as an enabler for call batching. Attribution does **not** need a deep ring, and does **not** work with a wall-clock zone around the loop either (that includes blocking); use `DeviceZoneScopedSumN1/N2` | **≫ sim, unmeasured** — real latency to hide |
| `implicit_sync` | **Yes, but only a floor: ≤9.6% ⇒ ≤1.10×** | `RD_BAR + WR_BAR` = 22 of 228 cyc/tile, and barriers are pre-satisfied so even that is an instruction-count artifact | |
| `num_tiles_per_cycle` | Partly | fewer per-tile instructions is real; DST/bank timing is not modelled | unknown |

⇒ **State craq-sim results as "reduced per-core instruction count and raised engine occupancy", never as
"removed serialization" and never as a bandwidth figure.** *Latency-hiding* levers need the emulator — but note
that the dominant lever here is **not** one of them: DFB call batching removes executed instructions, which is
exactly what this simulator counts faithfully. The sim is a better instrument for this op than the earlier
"two of four levers register as zero" framing implied.
