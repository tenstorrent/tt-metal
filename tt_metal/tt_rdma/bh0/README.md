# BH.0 — active-eth heartbeat / coexistence gate

First milestone of the TT-RDMA Blackhole port (`docs/tt-rdma-v1/tt-rdma-bh-bf3-impl-plan.md`
§2.4 + the BH.0 deep-dive). **Nothing RDMA here** — this proves the *execution model* the whole
port depends on: a persistent kernel on **RISC1 (subordinate)** while **RISC0 (active_erisc)** keeps
yielding to the `bh-erisc` base FW so the trained link stays up.

## Files
- `kernels/bh_rdma_heartbeat.cpp` — the RISC1 kernel: a persistent loop that increments a heartbeat
  word at `TT_RDMA_RCB_ADDR` and paces itself. No NoC ops, no base-FW calls, never writes `0x70000+`.
- `bh0_heartbeat_host.cpp` — loads the kernel onto an active eth core's RISC1 and holds it resident.
- The addresses come from `tt_metal/hw/inc/internal/ethernet/tt_rdma_l1_layout.h` (the L1 contract).

## The gate (binary)
With the kernel resident for **10 minutes**:
1. `port_status` on the chosen eth core stays **UP** the whole time, and
2. the heartbeat word at `TT_RDMA_RCB_ADDR` **advances**.

If (1) fails → RISC0 isn't yielding enough (coexistence model wrong — the load-bearing risk).
If (2) fails → the kernel isn't actually running on RISC1.

## Build + run
The host is reconciled to this checkout's API (`distributed::MeshDevice` + `MeshWorkload`, pattern
from `programming_examples/hello_world_datamovement_kernel`).

**Builds IN-TREE only.** Active-eth kernel creation uses the *internal* `EthernetConfig` API, which
is NOT in the public installed TT-Metalium (verified: an out-of-tree `find_package(TT-Metalium)`
build compiles everything *except* `EthernetConfig`). So:

Two in-tree build products (both already wired):
- **Standalone tool `bh0_heartbeat`** (#1) — `add_subdirectory(... tt_metal/tt_rdma/bh0 ...)` is in
  `tests/tt_metal/tt_metal/CMakeLists.txt` (next to `deployment`); persistent kernel (num_beats=0),
  run + observe + reset. Full 10-min gate.
- **gtest `TtRdmaBH0CoexistenceHeartbeat`** (#4) — `tests/tt_metal/tt_metal/deployment/eth/test_bh0_coexistence.cpp`
  in `unit_tests_deployment` (added to `deployment/sources.cmake`); bounded kernel, CI-friendly
  ~seconds smoke, asserts `ensure_links()` before AND after. Guaranteed-compile (reuses the
  deployment fixture + helpers).

```bash
./build_metal.sh                                   # build_Release/ already exists on this box
# standalone tool (persistent):
TT_METAL_HOME=<repo-root> ./build_Release/.../bh0_heartbeat  [device_id] [eth_idx] [spin] [hold_s]
# gtest (bounded smoke):
./build_Release/test/tt_metal/unit_tests_deployment --gtest_filter='*TtRdmaBH0Coexistence*'
```
- **`eth_idx`** picks which active eth core (argv[2], default 0) — the "pin the core" knob. On start
  the tool **prints the chosen core's logical AND physical/NOC coords**, so you know exactly which
  `X-Y` to watch with `erisc_ports.sh` / `tt-exalens` (the logical eth coord ≠ the NOC coord).
- `spin` paces the heartbeat writes; `hold_s` is the resident window (default 600 = 10 min).

If a standalone `main()` can't link against `test_metal_common_libs`, fall back to the **gtest form**:
add `bh0_heartbeat_host.cpp` to `UNIT_TESTS_DEPLOYMENT_SRC` and drive it from a `TEST()` using the
`CommandQueueFixture` (that suite already uses `EthernetConfig` + `MeshDevice`).

## Observe (separate terminal, does NOT disturb the chip)
```bash
# link stays UP:
/home/alex/tenstorrent/bh-erisc-fpga/scripts/erisc_ports.sh <X-Y>
# heartbeat advances (TT_RDMA_RCB_ADDR — compute from tt_rdma_l1_layout.h; default base 0x42000):
TTX=/home/alex/tenstorrent/tt-exalens
$TTX/.venv/bin/python $TTX/tt-exalens.py --commands "brxy <X-Y> <TT_RDMA_RCB_ADDR> 1 -d 0; x" </dev/null
```
To stop the persistent kernel: `sudo reboot` (or `tt-smi -r --eth_train_skip`).

## Risks / what BH.0 actually tests (log the answers)
- **Does the link survive a resident RISC1 kernel?** — the whole point. This is the empirical
  yield-cadence tolerance the plan can't give a number for.
- **Running a user kernel on a core with a *live external link* is novel** — tt-metal normally owns
  eth cores for tunneling/dispatch. Starting on a live-link core IS the test; if it's too risky for
  the very first run, confirm the mechanics on a benign active core first, then repeat on the live one.
- **`TT_RDMA_L1_BASE` (0x42000) is a placeholder** — BH.0 must confirm it's clear of the real
  build's low FW/kernel-config watermark and the high fabric-router/barrier region (the
  `tt_rdma_l1_layout.h` static_asserts guard the top `0x70000` and the `0x40000` reset save).
- **NOC split:** RISC1 uses NOC1 (base FW owns NOC0 on RISC0). The heartbeat write is a local L1
  store, so it doesn't need the NoC — but the eventual RX/TX loop will, on NOC1.

## Next (BH.1)
Replace the heartbeat body with `eth_send_raw` of one 32 B `tt_rdma_wire.h` header + payload to the
BF3 "tt" MAC, and confirm it on the BF3 with `tcpdump -i <ttport> ether proto 0x1af6 -xx` (milestone
M-1a). Then the RX-classifier landing (M-1b).
