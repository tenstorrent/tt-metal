# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Force the Tensix clock (AICLK) and core voltage (VDD) of tt-metal devices through ARC messages, for DIDT V/F sweeps.

The force goes through the chip's ARC firmware (tt-system-firmware smc_msg.h FORCE_AICLK) and holds until it is
released (argument 0) or the board is reset. It overrides the firmware's own clock choice, including TDP/TDC/thermal
throttling. Only chips with a PCIe (MMIO) connection can be addressed.
"""

import time

import yaml
from loguru import logger

import ttnn

FORCE_AICLK = 0x33  # argument: MHz, 0 releases the force
FORCE_VDD = 0x39  # argument: mV, 0 releases the force; must stay inside the chip's VDD_LIMITS


def metal_id_to_pci_index():
    """Map tt-metal device id -> /dev/tenstorrent/<N> from the cluster descriptor's chips_with_mmio.

    ttnn.GetPCIeDeviceID does not give the /dev/tenstorrent index on a Galaxy (it returned 22 for a chip on
    /dev/tenstorrent/30), so read the descriptor instead.
    """
    path = ttnn.cluster.serialize_cluster_descriptor()
    with open(path) as f:
        # BaseLoader keeps every scalar a string; the default loader turns BDFs like 0000:01:00.0 into numbers.
        desc = yaml.load(f, Loader=yaml.BaseLoader)
    mapping = {}
    for entry in desc.get("chips_with_mmio", []):
        for chip, pci in entry.items():
            mapping[int(chip)] = int(pci)
    return mapping


class AiclkController:
    """Forces, verifies and releases AICLK and VDD on a set of tt-metal devices."""

    def __init__(self, device_ids, settle_timeout_s=10.0, tolerance_mhz=1, vdd_tolerance_mv=3):
        import tt_umd

        self._tags = tt_umd.TelemetryTag
        self.settle_timeout_s = settle_timeout_s
        self.tolerance_mhz = tolerance_mhz
        self.vdd_tolerance_mv = vdd_tolerance_mv

        id_to_pci = metal_id_to_pci_index()
        missing = [d for d in device_ids if d not in id_to_pci]
        if missing:
            raise RuntimeError(f"VF sweep: devices {missing} have no PCIe connection, so their clock cannot be forced")

        self.devices = {}
        self.vdd_limits = {}
        for device_id in device_ids:
            pci = id_to_pci[device_id]
            dev = tt_umd.TTDevice.create(pci)
            dev.init_tt_device()
            tel = dev.get_arc_telemetry_reader()
            self.devices[device_id] = (pci, dev, tel)
            limits = tel.read_entry(self._tags.VDD_LIMITS)  # vdd_max << 16 | vdd_min, mV
            self.vdd_limits[device_id] = (limits & 0xFFFF, limits >> 16)
        self.forced_mhz = None
        self.vdd_forced = False

    def read(self):
        """Current {device_id: (AICLK MHz, VCORE mV)} from ARC telemetry."""
        out = {}
        for device_id, (_, _, tel) in self.devices.items():
            out[device_id] = (tel.read_entry(self._tags.AICLK), tel.read_entry(self._tags.VCORE))
        return out

    def _msg(self, device_id, code, arg, what):
        pci, dev, _ = self.devices[device_id]
        reply = dev.arc_msg(code, True, [int(arg)], 1000)
        exit_code = reply[0] if isinstance(reply, (list, tuple)) else reply
        if exit_code != 0:
            raise RuntimeError(
                f"VF sweep: {what} {arg} failed on device {device_id} (/dev/tenstorrent/{pci}), ARC exit code {exit_code}"
            )

    def set_clock(self, mhz):
        """Force AICLK to mhz on every device and wait until telemetry reports it on all of them."""
        for device_id in self.devices:
            self._msg(device_id, FORCE_AICLK, mhz, "FORCE_AICLK MHz")
        self.forced_mhz = mhz
        deadline = time.time() + self.settle_timeout_s
        while True:
            readings = self.read()
            off = {d: r for d, r in readings.items() if abs(r[0] - mhz) > self.tolerance_mhz}
            if not off:
                return readings
            if time.time() > deadline:
                raise RuntimeError(
                    f"VF sweep: AICLK did not reach {mhz} MHz within {self.settle_timeout_s} s on devices "
                    + ", ".join(f"{d} (AICLK {r[0]} MHz)" for d, r in sorted(off.items()))
                )
            time.sleep(0.1)

    def _settled_vcore(self):
        """{device_id: VCORE} once the last 3 readings of every device agree within 2 mV (or the timeout passes)."""
        history = {d: [] for d in self.devices}
        deadline = time.time() + self.settle_timeout_s
        while True:
            for d, (_, vcore) in self.read().items():
                history[d] = (history[d] + [vcore])[-3:]
            if all(len(h) == 3 and max(h) - min(h) <= 2 for h in history.values()) or time.time() > deadline:
                return {d: h[-1] for d, h in history.items()}
            time.sleep(0.1)

    def set_vdd(self, targets_mv):
        """Force VDD so that telemetry VCORE reads targets_mv[device_id] on each device; returns the readback.

        Telemetry reads a little above the FORCE_VDD setpoint, so the setpoint is corrected once by the readback
        error. Refuses targets outside the chip's VDD_LIMITS.
        """
        for d, mv in targets_mv.items():
            lo, hi = self.vdd_limits[d]
            if not lo <= mv <= hi:
                raise RuntimeError(f"VF sweep: VDD target {mv} mV on device {d} is outside its VDD_LIMITS {lo}-{hi} mV")
        setpoints = dict(targets_mv)
        for d, mv in setpoints.items():
            self._msg(d, FORCE_VDD, mv, "FORCE_VDD mV")
        self.vdd_forced = True
        readback = self._settled_vcore()
        corrected = {}
        for d, mv in targets_mv.items():
            if abs(readback[d] - mv) > self.vdd_tolerance_mv:
                lo, hi = self.vdd_limits[d]
                corrected[d] = min(hi, max(lo, setpoints[d] + mv - readback[d]))
        if corrected:
            for d, sp in corrected.items():
                self._msg(d, FORCE_VDD, sp, "FORCE_VDD mV")
            readback = self._settled_vcore()
        off = {d: readback[d] for d, mv in targets_mv.items() if abs(readback[d] - mv) > self.vdd_tolerance_mv}
        if off:
            logger.warning(
                "VF sweep: VCORE readback off target after correction on devices "
                + ", ".join(f"{d} ({v} mV, target {targets_mv[d]} mV)" for d, v in sorted(off.items()))
            )
        return readback

    def release_vdd(self):
        """Hand VDD back to the firmware's V/F curve on every device and wait for VCORE to settle."""
        for device_id in self.devices:
            self._msg(device_id, FORCE_VDD, 0, "FORCE_VDD release")
        self.vdd_forced = False
        return self._settled_vcore()

    def release(self):
        """Release both forces on every device. Never raises; logs failures."""
        for device_id, (pci, dev, _) in self.devices.items():
            for code, what in ((FORCE_VDD, "VDD"), (FORCE_AICLK, "AICLK")):
                try:
                    reply = dev.arc_msg(code, True, [0], 1000)
                    exit_code = reply[0] if isinstance(reply, (list, tuple)) else reply
                    if exit_code != 0:
                        logger.error(
                            f"VF sweep: releasing {what} on device {device_id} returned ARC exit code {exit_code}"
                        )
                except Exception as e:  # keep releasing the others
                    logger.error(
                        f"VF sweep: releasing {what} on device {device_id} (/dev/tenstorrent/{pci}) failed: {e}"
                    )
        self.forced_mhz = None
        self.vdd_forced = False
