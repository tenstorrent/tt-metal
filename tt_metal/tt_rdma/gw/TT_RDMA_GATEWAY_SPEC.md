# TT-RDMA Gateway — Architecture B Spec (RoCEv2 ⇄ TT-RDMA on BlueField-3)

**Scope:** Architecture B is *the* scope of TT-RDMA: a BlueField-3 (BF-3) DPU that lets **unmodified RoCEv2
initiators** (host GPUs via GPUDirect/NCCL, or remote hosts on the wire) reach **Tenstorrent Blackhole (BH)**
memory, by terminating RoCEv2 in ConnectX HW and **re-heading** the payload into TT-RDMA (ethertype `0x1AF6`,
32B TT header, rkey→BH MR slot). The BH pool is **unchanged**; native-RoCE-on-BH is rejected (BH RX has no
landing engine + no RoCE reliability). The BF-3 simultaneously keeps its **traditional RoCEv2 NIC/switch**
role — a flow is gatewayed to TT-RDMA only when its **destination is a TT endpoint**; otherwise it is normal
RoCE.

This spec defines the two production deployment cases, the destination-based routing, the datapath mechanisms
(and the engine choice for line rate), what is already validated on silicon, and the remaining scope.

---

## 1. Roles the BF-3 plays (all concurrent)

1. **RoCE NIC / passthrough** — host/GPU RoCEv2 to a *traditional* RoCE target. Native ConnectX; the gateway
   is not involved. (GPUDirect RDMA = payload DMA'd straight from GPU memory over PCIe.)
2. **RoCEv2 → TT-RDMA gateway** — RoCEv2 whose *destination is a TT endpoint (BH)*: ConnectX terminates
   (PSN/ICRC/ACK in HW), the payload is **re-headed** to a TT-RDMA WRITE frame and egressed to the BH.
3. **RoCE ⇄ TT switch** — inline between a RoCEv2 physical link and a TT-RDMA physical link, applying role 1
   or 2 per flow by destination.

The distinguishing operation is **role 2 (the re-head)**. Roles 1 and 3 are ConnectX/eswitch native.

---

## 2. Two production cases

### Case 1 — PCIe-attached (NCCL / GPUDirect RDMA)

The initiator is a **host GPU/app on PCIe** using stock RoCE verbs (NCCL, UCX, GPUDirect). The BF-3 is the
host's NIC. Destination decides the path: a traditional RoCE peer → normal NIC; a **TT endpoint → gateway →
BH**.

```
            ┌──────────────────────── x86 host ────────────────────────┐
            │   GPU (HBM)  ──GPUDirect──┐        NCCL / verbs app        │
            └───────────────────────────┼──────────────────────────────┘
                                        │ PCIe (RoCEv2 WRITE/READ)
                                        ▼
        ┌──────────────────────────── BF-3 ────────────────────────────┐
        │  ConnectX RoCE engine (HW terminate: PSN/ICRC/ACK)            │
        │        │ dst = traditional RoCE            │ dst = TT endpoint │
        │        ▼                                   ▼                   │
        │   normal NIC TX  ───► wire/peer       RE-HEAD engine           │
        │   (passthrough, GPUDirect)            (RoCE payload → 46B TT   │
        │                                        hdr + payload)          │
        │                                            │                   │
        │                                            ▼  PF ETH SQ        │
        └────────────────────────────────────────── p0 ────────────────┘
                                                     │ TT-RDMA 0x1AF6
                                                     ▼
                                              Blackhole (drainer pool)
```

Key: the GPU app is **unmodified** — it just targets a TT GID/rkey. GPUDirect-to-TT means GPU HBM → (PCIe) →
BF-3 terminate → re-head → BH, with the Arm off the per-frame path (engine choice §5).

### Case 2 — Two-link switch (RoCEv2 link ⇄ TT-RDMA link)

One physical link carries **RoCEv2** (remote RoCE hosts), the other carries **TT-RDMA** (to the BH). The BF-3
is an inline gateway *and* switch, **plus** it still serves the PCIe host (Case 1 overlaid). Per-flow routing
by destination: traditional-RoCE dst → switch/NIC as RoCE; TT dst → gateway → TT link.

```
   remote RoCE host(s)                                              Blackhole
        │ RoCEv2 (wire)                                        TT-RDMA (wire)
        ▼                                                            ▲
   ┌── p1 (RoCEv2 link) ───────────── BF-3 ──────────── p0 (TT link) ──┐
   │                                                                   │
   │   ConnectX RoCE terminate (HW)          PF ETH SQ (TT egress)     │
   │        │                                        ▲                 │
   │        ├─ dst = TT endpoint ─► RE-HEAD ─────────┘                 │
   │        │                        (role 2)                          │
   │        ├─ dst = traditional RoCE ─► eswitch switch/route (role 3) │
   │        │                                                          │
   │   ┌────┴─────────── PCIe (host PF) ───────────────┐              │
   │   │  host/GPU RoCE (Case 1 overlaid): NIC + gateway│              │
   │   └─────────────────────────────────────────────  ┘              │
   └───────────────────────────────────────────────────────────────  ┘
```

Bench note: today both rails **p0 and p1 → BH**, host RoCE arrives via **PCIe → SF (mlx5_2, 10.99.0.1)**.
Case 2 production repurposes one rail (e.g. p1) as the RoCEv2-facing link; p0 stays the TT/BH link.

---

## 3. Destination-based routing (traditional RoCE vs TT gateway)

The gateway must decide, per flow, "traditional RoCE" (roles 1/3) vs "TT endpoint" (role 2). Options, in
preference order:

1. **By MR / rkey (recommended)** — a **TT rkey** (rkey → BH MR slot, e.g. `0x00CAFE42`) is advertised for
   TT-backed memory regions; a WRITE to a TT rkey ⇒ gateway/re-head, any other rkey ⇒ normal RoCE. This is
   the natural, per-region selector and folds into **MR federation (B2)**: a service maps rkey → {BH chip,
   MR slot, offset}. The initiator gets a TT rkey from a memory-registration handshake exactly like a normal
   RoCE MR — fully transparent.
2. **By destination GID/IP** — reserve a GID/subnet for TT endpoints; route those to the gateway. Coarser
   (per-peer, not per-region) but simplest for a pure two-link switch.
3. **By dedicated gateway QP/service** — the initiator opens a QP to a "TT gateway" service that always
   re-heads. Explicit, least transparent.

`(1)` is the target; `(2)` is a fine bring-up shortcut for Case 2.

---

## 4. Gateway datapath (role 2, the re-head)

```
RoCEv2 WRITE/WRITE_IMM ─► ConnectX RoCE terminate (HW: PSN, ICRC, ACK) ─► payload in memory
        │                                                                       │
        │  (reliable RC terminated here — cannot be "switched", it is now       │
        │   data-in-memory, not a packet)                                       │
        ▼                                                                       ▼
   recv completion (HW event) ───────────────────────────────► RE-HEAD engine
                                                                 build [14B L2][32B TT hdr][payload]
                                                                 (opcode WRITE, ver 1, len, seq,
                                                                  rkey→BH MR slot, roff, crc)
                                                                       │
                                                                       ▼
                                                                 PF ETH SQ ─► p0 ─► BH pool
```

Invariant discovered on silicon: **terminated RoCE lands in memory**, so re-emission always needs a **TX
engine** (the eswitch cannot forward terminated RC — it is no longer a packet). The only exception is an
**unreliable (UC/UD)** RoCE side, which need not be terminated and *could* be a pure HW header-rewrite forward
(see §5, option E).

---

## 5. Re-head engine options — **requirement: lowest latency AND highest bandwidth, all cases**

Both axes are hard requirements, so the engine must be **on-chip event-driven** (no host/Arm/PCIe round-trip
in the per-frame path → lowest latency) **and** use a **HW gather-TX engine with enough parallelism**
(→ highest bandwidth). That rules out the CPU/doorbell paths.

| Engine | Latency | Bandwidth | Arm on path? | Status |
|---|---|---|---|---|
| **B. DPA ETH-SQ, event-driven (recv-CQE→EU→TX)** | **Lowest** (on-chip HW event, no host) | ~0.95 Mpps/EU → ~146 G at N EUs | No | Egress leg validated (P1.4a, 1000 frames p0). |
| **A. HW-TX (`doca_eth_txq`)** — descriptor ring, NIC HW gathers+TXes | Low **iff** small batch + on-chip fill; higher if deep-batched | **~198 G** | No | 198 G measured (doca_ttblast). |
| **A+B hybrid (recommended)** — DPA EU **event-triggered** fills a HW-TX descriptor per frame | **Lowest** (event) + | **Highest** (HW gather; ~1 descriptor-write/frame lifts the per-EU wall) | No | **Target.** Combines B's latency + A's bandwidth. |
| C. Arm-CPU (ib_verbs + memcpy) | High (CPU wake+copy) | ~56 G | Yes | Proven (B1); interop/bring-up only. |
| D. Host-doorbell hybrid (A3) | High (host in loop) | ~40 G | host writes doorbell | Superseded. |
| E. eswitch cut-through (UC/UD only) | **Absolute floor** (no terminate, no mem round-trip) | line rate | No | Best on both — **but UC/UD only** (no RC reliability). NCCL/GPUDirect are RC → likely N/A. |

**Guidance (revised for latency+bandwidth):**
- **Reliable RC (NCCL/GPUDirect default):** the winner is the **on-chip event-driven re-head driving a HW
  gather-TX** (A+B hybrid): the recv CQE activates an on-chip DPA EU with **no host round-trip** (lowest
  latency), and the EU only writes a compact descriptor while the **NIC HW does the payload gather+TX**
  (highest bandwidth, and it sidesteps the ~1 µs/frame WQE-issue wall that caps pure-DPA single-EU at
  ~0.95 Mpps). Fan out to N EUs only if one can't fill the wire.
- **This forces the single-context topology** (see §7): to keep it one event-driven agent, terminate RoCE
  **and** run the TX engine on the **same function (the SF)**, triggered by the SF recv CQE, + **one eswitch
  flow to steer SF→p0**. Putting the TX on the PF (native p0 egress) re-introduces the cross-ctx split that
  breaks the single event-driven agent — so for latency, **SF-everything + steering beats PF-egress.**
- **E (eswitch cut-through)** is the theoretical best on *both* axes (no termination = no memory round-trip),
  but only for UC/UD. If any traffic class can be UC/UD, gateway it this way (zero re-head engine).
- **C** only for interop/functional bring-up, never the latency/bandwidth path.

### 5.1 Latency budget (reliable-RC gateway)
`RoCE arrive → ConnectX terminate+DMA (HW, sub-µs) → recv CQE → on-chip EU wake (event, sub-µs) → descriptor
write (few hundred ns) → NIC gather+TX (HW) → wire`. Added latency over native RoCE ≈ **single-digit µs** when
event-driven on-chip; **tens of µs** if a host/Arm poll or a doorbell window sits in the loop (why D's ~54.8 µs
READ RTT is not the target). Micro-tension: TX **batch depth** trades latency for bandwidth — the HW gather
engine lets small batches (low latency) while HW moves the bytes (bandwidth), so keep batch small and let
parallelism (N EUs / deep HW-TX ring) provide bandwidth.

---

## 6. What is validated on silicon (Phase 0 done)

- **RoCE terminate + recv-trigger**: x86 host WRITE_IMM → SF RC QP → recv CQE → DPA Thread A → produced flag.
  (P1.5 requester `dpa_rehead_verbs/tt_p15_requester.c`: 1000/1000 WRITE_IMM ok; Thread A ran per recv.)
- **TT egress to BH at line rate**: PF ETH SQ on p0 egresses TT-RDMA frames natively, delta-exact to 100000
  (P1.4a); HW-TX (doca_ttblast) hit **198.3 G** byte-exact, exactly-once (drainer pool acceptance).
- **Full E2E once (doorbell path)**: real RoCEv2 WRITE_IMM (x86) → ConnectX SF → DPA re-head → p0 → BH pool
  landed byte-exact (A3, engine D).
- **Arm-CPU gateway**: host RoCEv2 → DPU bridge → BH byte-exact (B1, engine C, ~56 G).

## 7. Key constraints discovered (design inputs)

- **PF has no RoCE GID** (switchdev uplink); RoCE terminates on an **SF**. p0/p1 are physical uplinks.
- **SF ETH SQ does not reach p0** without an eswitch steering flow (187/1000); the **PF ETH SQ reaches p0
  natively.** ⇒ egress belongs on the PF.
- **DPA context split (engine B):** the SF RC recv CQ is reachable only via the `device_extend`-ed ctx; the PF
  ETH SQ only via `pf_dpa_ctx`. **One thread cannot span both**, and **`doca_dpa_dev_thread_notify` does not
  cross contexts** (nor fire from an RPC). This is why engine B needs either a poll-coupled two-thread design
  or a single-context topology — and why **engine A (HW-TX) is the cleaner line-rate path.**

## 8. Scope / phasing (remaining) — optimized for lowest latency + highest bandwidth

- **P-GW1 — single-context event-driven re-head + HW gather-TX (A+B hybrid).** Terminate RoCE on the SF, run
  the TX engine on the **same SF context**, trigger on the **recv CQE** (on-chip, no host round-trip → lowest
  latency), have the on-chip EU write a compact **HW-TX (`doca_eth_txq`) descriptor** (TT hdr + payload
  gather; NIC HW moves the bytes → highest bandwidth), + **one eswitch flow steering SF→p0**. This dissolves
  the cross-ctx wall (§7). Sub-steps: (a) SF egress + SF→p0 steering flow (confirm 100% to p0 vs 187/1000
  unsteered); (b) one event-driven DPA thread on the SF recv CQ doing recv→descriptor→TX (DPA-heap header for
  per-frame seq); (c) measure added latency (target single-digit µs) + bandwidth (target line rate); (d) N-EU
  fan-out only if one EU can't fill the wire.
- **P-GW2 — destination routing (B2 MR federation)**: rkey→{BH chip, MR, offset} service; TT-rkey → gateway,
  else passthrough.
- **P-GW3 — Case 1 GPUDirect**: validate GPU-HBM source (GPUDirect) through the gateway to BH; measure GPU→BH
  latency + bandwidth.
- **P-GW4 — Case 2 two-link**: repurpose p1 as the RoCEv2 link; eswitch flows for role-3 switching + role-2
  gateway; verify passthrough + gateway coexist without adding latency to either.
- **P-GW5 — UC/UD HW cut-through probe (engine E)**: DOCA Flow feasibility of RoCE-strip + TT-encap + PSN→seq
  p1→p0 — the absolute-lowest-latency line-rate path for any UC/UD-tolerant class.
- **P-GW6 — READ / multi-QP / cross-chip**: gateway RDMA READ (B4), multi-QP (B5), TT-fabric bridge (4F).

> P-GW1 supersedes the earlier PF-ETH-SQ two-thread DPA build: PF egress is native line-rate but forces the
> SF↔PF cross-ctx split (§7) that adds a coupling hop and breaks the single event-driven agent. With the
> latency requirement, **SF-everything + one steering flow** is the right trade.

_Companions: `dpa_rehead_verbs/README.md` (engine B build + findings), `A3_rehead_plan.md` (engine D),
`A3_option3_research_plan.md` (native event-driven), `RUNBOOK.md` (bench). Memory:
[[tt-rdma-rocev2-gateway-arch-b]], [[tt-rdma-dpa-rehead-plan]], [[tt-rdma-production-path]]._

## 9. P-GW1 progress — step (a) SF→p0 steering: **GREEN (line rate, HW-offloaded)**

The single-context (SF) design's egress was the open risk (SF ETH SQ reached p0 only 187/1000 earlier). Now
resolved: an **eswitch/OVS steering flow** `in_port=en3f0pf0sf0(SF rep), dl_type=0x1af6 → output:p0`, once
**HW-offloaded**, egresses the SF ETH SQ to p0 at **100% (1000/1000)** in hardware.
- tc confirms offload: `eth_type 1af6, in_hw, action mirred (Egress Redirect to device p0)`.
- The earlier 16% was a **cold-flow artifact**: OVS HW-offloads a megaflow only after the first packets go
  slow-path; a one-shot 1000-frame burst mostly finishes in the slow-path window and drops. Warm runs
  (N=10, N=100 first) → the flow offloads → subsequent N=1000 = 100%.
- **Production requirement:** pin/pre-warm the offloaded flow before line-rate traffic (send a warm-up frame,
  or install a `skip_sw` tc flower filter to force HW-only from the start). Helper: `dpa_rehead_verbs/tt_gw_steer.sh`.

**Consequence:** the single-context SF design is line-rate-viable — RC recv CQ **and** SF ETH SQ on the same
(extended) ctx, egress to p0 via the HW-offloaded flow. This dissolves the cross-ctx wall (§7): one
event-driven thread can do recv→re-head→egress, **no two-thread, no polling, no PF split.** Code:
`dpa_rehead_verbs/dpa_verbs_initiator_target_sample.c` (ETH SQ flipped to SF verbs ctx + extended dpa_ctx;
buffers single-PD on the SF).

**Remaining P-GW1:** (b) single event-driven thread on the SF recv CQ doing recv → DPA-heap seq header →
SF ETH-SQ gather-egress → repost (the original single-thread kernel, now unblocked); (c) drive with the P1.5
requester + measure added latency (target single-digit µs) and bandwidth (target line rate); (d) N-EU fan-out
if one EU can't fill the wire; (e) BH-pool byte-exact landing.

## 10. P-GW1 step (b) — single-thread event-driven re-head: **FULL E2E GREEN on silicon**

The complete lowest-latency single-context gateway path runs end to end:
`x86 host RoCEv2 WRITE_IMM → SF RC QP terminate (HW) → recv CQE → ONE DPA thread (event-driven, no host
round-trip) re-heads (seq patched into a DPA-heap header + 2-SGE gather [hdr]+[landed payload]) → SF ETH SQ →
HW-offloaded eswitch flow → p0.`
- Result: requester **1000/1000 WRITE_IMM ok**; p0 tx delta **≈1000**; steering flow **n_packets=1000** (all
  re-headed frames reached p0). Target = `target_thread_kernel` (single thread, extended ctx) activated by the
  SF RC recv completion; no two-thread, no cross-ctx notify, no polling, no PF split.
- Constraint #4 resolved in-path: seq is patched into a **DPA-heap header** (extended ctx) mkey'd via
  `doca_mmap` MMAP_TYPE_DPA for the gather SGE0; the host header is just the template source.
- Code: `dpa_rehead_verbs/dpa_verbs_initiator_target_{sample,kernels_dev}.c`. Requester
  `tt_p15_requester.c`. Steering `tt_gw_steer.sh` (pre-warm the offloaded flow).

**Remaining P-GW1:** (c) measure added latency (single WRITE_IMM RTT vs native RoCE; target single-digit µs)
and bandwidth (timed run; single-EU DPA ETH-SQ ≈ 0.95 Mpps ≈ 7.8 G @1KB — bandwidth needs (d)); (d) raise
bandwidth to line rate via the A+B hybrid (DPA EU writes a compact HW-TX/`doca_eth_txq` descriptor, NIC HW
gathers) and/or N-EU fan-out; (e) BH-pool byte-exact landing (bring up the drainer pool, confirm write_ok=N).

## 11. Reconciled roadmap (from `BW_PLAN.md` + `E2E_TEST_PLAN.md`)

Two planning agents produced `gw/BW_PLAN.md` (bandwidth→line-rate) and `gw/E2E_TEST_PLAN.md` (correctness/validation).
Reconciled, with one important **correction to §5/§8**:

### ★ Engine correction: the literal "A+B hybrid" is likely NOT available
On this DOCA 3.4 there is **no DPA-side `doca_eth_txq` datapath** (no `doca_eth_txq_dpa_data_path.h`; DPA eth
egress is only via DPA-Verbs ETH-SQ or FlexIO). So "a DPA EU writes a compact `doca_eth_txq` descriptor" (§5
engine A / the A+B hybrid) is probably not buildable as written — **verify this first as a gate.** The realistic
on-chip line-rate path is **N-EU DPA-Verbs ETH-SQ fan-out** (§5 engine B scaled): N re-head EUs, each on its own
ETH SQ, fed by **multi-QP or SRQ** (one RC QP = one recv stream, so recv load must be spread across EUs).
Arm-driven HW-TX (`doca_eth_txq`, ~198 G) remains an option but is Arm-in-the-loop and only wins for a mandated
≤4 KB class at a latency cost — not the lowest-latency path.

### Two independent workstreams (run in parallel)
- **Correctness (gates "done"), E2E_TEST_PLAN Stage 1 — do first, cheap, on the current 0.32 Mpps path:**
  **byte-exact / exactly-once landing off the actual P-GW1 event-driven DPA gateway.** Byte-exact was only ever
  confirmed off the B1 Arm-CPU bridge and off the drainer pool *separately* — never off the P-GW1 path that
  reaches p0. This is the true open correctness gap (task #13). 6-step mismatch decision tree in the plan.
- **Bandwidth (BW_PLAN):**
  1. **d.1 async-ring fault fix** = a HOST-SETUP bisect (the 0x2 fault is in the trigger `post_recv`, which runs
     *before* the async loop → the async loop is ruled out first-order). Leading hypothesis: the **fixed 64 B DBR
     umem + the CQ umem don't scale with depth** (matches the earlier "256 bump broke recv-post"). Bisect the 4
     deltas (RQ 4→64, RC CQ 4→128, 1→1024 landing ring, 1→1024 header ring) against the *synchronous* kernel;
     ring-shrink fallback `TT_RING = 2×inflight`. Expected → ~0.95 Mpps single-EU.
  2. **d.2b N-EU DPA-Verbs ETH-SQ fan-out** (multi-QP/SRQ), landing in **DPA-private DDR** (host-MR gather caps
     ~42 G; DPA-heap scales ~134 G). Frame-size napkin: DPA-issue hits a ~4.4 Mpps aggregate wall (~145 G @4 KB),
     so **≥~5.7 KB is needed for 200 G → target 8 KB + ~4 EUs, Arm-off, latency preserved.**

### Ordering
Stage-1 byte-exact FIRST (validates the gateway is correct before scaling). Then d.1 async ring (→ ~0.95 Mpps),
then d.2b N-EU fan-out @8 KB (→ ~150–200 G). Destination routing (B2), Case-1 GPUDirect, Case-2 two-link, and
interop (perftest/NCCL) proceed per E2E_TEST_PLAN Stages 2–7. Latency (~3.4 µs) must not regress at any step.

---

## 12. Open-source / IP surface — what is public, what is ours, what stays closed

Verified by diffing **public upstream `tt-metal`** against our fork `tt-metal-external-eth` and the closed
`bh-erisc-fpga` FW repo. This answers "can this be open-sourced, and what must the FW expose?" — the short
answer is that **the eth-engine interface, the EDM fabric, and the TT-link wire format are already public**;
a fully-open BF-3 fabric edge is **not** blocked on opening the closed FW.

### 12.1 Tier 1 — already PUBLIC (upstream `tt-metal`, Apache-2.0)

The interface tt-metal *expects from the eth FW* is not just defined — it is **published to the world**:

| Public upstream file | What it contracts |
|---|---|
| `tt_metal/hw/inc/internal/ethernet/tt_eth_ss_regs.h` | eth-engine **register interface** (the eth SS reg map) |
| `.../ethernet/tt_eth_api.h`, `erisc.h`, `tunneling.h` | eth send/recv API + erisc primitives |
| `.../tt-1xx/{blackhole,wormhole}/eth_fw_api.h` | **the tt-metal ↔ eth-FW ABI** (`link_train_status`, `port_status`, `chip_info`, ctrl reg addrs) |
| `tt_metal/fabric/hw/inc/edm_fabric/*` (**33 files, byte-identical to our fork**) | the **entire EDM datamover / fabric router** |
| `tech_reports/TT-Fabric/TT-Fabric-Architecture.md` (TT-link section present upstream) | the **TT-link wire format**: 16-B TT-link header, `tx/rx seq`, **Go-Back-N ARQ**, DEST_ADDR-in-header forwarding vs compliance-mode RX-buffer (the pool path) |

**Consequence:** the eth register map, the FW ABI, the EDM fabric protocol, and the on-wire TT-link framing
are all public. tt-metal does not merely *expect* an eth-engine contract — it *ships* it. The closed FW is one
**implementation** behind an already-public interface.

### 12.2 Tier 2 — OURS, Apache-base, upstreamable (in `tt-metal-external-eth`, not yet upstream)

The TT-RDMA-to-**external** datapath — our work, on the Apache-2.0 base:

- `.../ethernet/tt_rdma_eth_tx.h` / `tt_rdma_eth_rx.h` / `tt_rdma_eth_icrc.h`
- `.../ethernet/tt_rdma_l1_layout.h`, `tt_rdma_hdr_build.h`, `tt_rdma_wire.h`
- `docs/tt-rdma-v1/*` (all design docs)

`tt_rdma_eth_tx.h` carries a **self-contained open copy** of the send sequence; it cites the closed
`eth_ss.cpp` only as the "golden reference," so the datapath does **not** require the closed source to build.

### 12.3 Tier 3 — CLOSED (`bh-erisc-fpga`), and whether it blocks an open edge

Only three things live *only* in the closed repo:

1. **ANLT/LT link bring-up** (`serdes_init.cpp`) — genuinely HW-coupled (AlphaCore N6 SerDes + Rianta MAC/PCS).
2. The golden **`eth_ss.cpp` send sequence** — already mirrored open in Tier 2 (`tt_rdma_eth_tx.h`).
3. The **PFC runtime driver** (`eth_runtime.cpp`).

None of these blocks a fully-open BF-3 edge: link bring-up is **per-side** (the BF-3 runs its own ConnectX/SerDes
link; the BH already runs its own with its shipping FW — they negotiate on the wire), and the send sequence is
already published in Tier 2.

### 12.4 What must be EXPOSED from the FW for a fully-open, wire-compatible BF-3 edge

**From the datapath / register / wire side: nothing** — Tier 1 (public) + Tier 2 (upstreamable) already cover it.
The single residual worth **formally publishing** is a small **AN/LT + mode-enable interop spec**:

- the Clause-73 AN / Clause-72/136/162 LT **negotiation parameters** the BH advertises, so a peer can bring up a
  compatible link (an interop doc, **not** the firmware source), and
- the eth-controller **init / TT-link-mode-enable register sequence** (compliance → TT-link so RX HW-forwards to
  the DEST_ADDR) — a short sequence expressed over the **already-public** `tt_eth_ss_regs.h` defs.

### 12.5 Licensing buckets (for a release plan)

- **tt-metal side (pool, fabric bridge, `tt_rdma_*` headers, docs):** Apache-2.0 → open / upstreamable.
- **BF-3 / DOCA gateway (`gw/dpa_rehead_verbs/*`, requester, steer):** BSD-3-Clause © NVIDIA, **DOCA-SDK-gated**
  (open source, but builds only against the DOCA SDK).
- **`bh-erisc-fpga` FW:** closed; stays closed, and is **not a build dependency** for the open edge.

*(Structure/IP-surface read, not legal advice — the actual release decision rests with the org. Confirm the
Tier-2 open send-sequence copy is treated as authoritative vs the closed `eth_ss.cpp` golden, and that the
AN/LT + mode-enable interop sequence is cleared for publication.)*
