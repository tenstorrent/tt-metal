# BH.0-pre — confirm the tt-metal active-eth runtime + eth links on this bench

**Why this gate exists.** BH.0 (and the whole port) assumes the chip runs the **tt-metal active-eth
runtime** (`active_erisc` on RISC0 yielding to base FW, `subordinate_erisc` on RISC1) — *not* the
standalone `bh-erisc` base FW we flash from `bh-erisc-fpga`. BH.0-pre confirms that stack boots and
trains the eth links **on this bench** before we layer the heartbeat/RDMA kernel on top. If tt-metal
can't open the device or bring links up here, BH.0 can't run and that's the first thing to fix.

## The check
```bash
export TT_METAL_HOME=/home/alex/tenstorrent/tt-metal-external-eth
# guard with a timeout so a UMD eth-training wait can't hang the terminal:
timeout 180 $TT_METAL_HOME/build_Release/test/tt_metal/unit_tests_deployment \
    --gtest_filter='*TensixDeploymentEthernet00LinkUp*'
```
**PASS** = the tt-metal runtime opened the device, brought up the active-eth cores, and
`ensure_links()` reports links UP → the active-eth stack works here. Then proceed to run #4
(`*TtRdmaBH0CoexistenceHeartbeat*`) and #1 (`bh0_heartbeat`, 10-min hold).

## Prerequisites / caveats (this bench is non-standard)
1. **Known device state first.** The bench runs our *custom* `bh-erisc` topology-config FW
   (BF3 NIC on Cage A + P150↔P150 inter-chip). Reboot to the last-flashed stable config so UMD
   discovers cleanly — our fixes make that work: NIC cores report `PORT_UNUSED` (UMD skips them),
   inter-chip cores are `UP`, no core stuck at `PORT_UNKNOWN`. `tt-smi`/`system_health` already
   discover the 2-chip cluster on that FW.
2. **UMD hang = a non-terminal eth core.** If the run hangs (~900 s `wait_eth_core_training`), a core
   is stuck at `PORT_UNKNOWN`; reset (`sudo reboot`) and confirm the stable config with
   `bh-erisc-fpga/scripts/erisc_ports.sh` before retrying. (Same root cause we fixed this session.)
3. **tt-metal env:** `TT_METAL_HOME` set (above); hugepages configured for tt-metal; the device
   not held by another UMD client (kill stray `tt-smi`/`tt-mgmt`).
4. **Custom-FW integration is itself the risk.** tt-metal expects its own eth-FW behavior; running
   it on top of our custom `bh-erisc` FW + NIC topology is novel. A clean PASS means they coexist;
   a failure localizes the tt-metal-on-custom-FW gap (which core, which stage) — that becomes the
   real first task, not the heartbeat kernel.

## If it FAILS / can't run
Then the chip-side work must first establish a tt-metal-compatible bring-up on the bench — options:
flash a tt-metal-stock eth FW for a TT↔TT-only topology to prove the runtime, then reintroduce the
NIC rails; or run the tt-metal runtime against the two cross-linked P150s as a plain 2-chip cluster
(NIC rails parked). Capture *where* it fails (device open vs link-up vs a specific core).
