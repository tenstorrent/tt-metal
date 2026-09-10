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
it much (matching the recorded `C=2→4` = 0.3%), and the levers that *can* are fewer Tensix instructions
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
the other's payoff:

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

**`num_tiles_per_cycle` on the native path is 1 unconditionally, and three things stack to keep it there
(corrected 2026-09-10 — an earlier draft of this section wrongly said n = 8 "already runs" here):**

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
| Kill criterion — stop if threads deliver <1.3x | design §2.4 | **4.000x** from threads | cleared 3x over |
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

### 5.0.7 Open questions (moved from the tutorial page, 2026-09-10)

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
