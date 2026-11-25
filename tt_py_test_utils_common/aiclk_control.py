# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Force the Tensix clock (AICLK) of tt-metal devices through ARC messages, for DIDT V/F sweeps.

The force goes through the chip's ARC firmware (tt-system-firmware smc_msg.h FORCE_AICLK) and holds until it is
released (argument 0) or the board is reset. It overrides the firmware's own clock choice, including TDP/TDC/thermal
throttling. Only chips with a PCIe (MMIO) connection can be addressed.
"""

import time

import yaml
from loguru import logger

import ttnn

FORCE_AICLK = 0x33  # argument: MHz, 0 releases the force


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
    """Forces, verifies and releases AICLK on a set of tt-metal devices."""

    def __init__(self, device_ids, settle_timeout_s=10.0, tolerance_mhz=1):
        import tt_umd

        self._tags = tt_umd.TelemetryTag
        self.settle_timeout_s = settle_timeout_s
        self.tolerance_mhz = tolerance_mhz

        id_to_pci = metal_id_to_pci_index()
        missing = [d for d in device_ids if d not in id_to_pci]
        if missing:
            raise RuntimeError(f"VF sweep: devices {missing} have no PCIe connection, so their clock cannot be forced")

        self.devices = {}
        for device_id in device_ids:
            pci = id_to_pci[device_id]
            dev = tt_umd.TTDevice.create(pci)
            dev.init_tt_device()
            self.devices[device_id] = (pci, dev, dev.get_arc_telemetry_reader())
        self.forced_mhz = None

    def read(self):
        """Current {device_id: (AICLK MHz, VCORE mV)} from ARC telemetry."""
        out = {}
        for device_id, (_, _, tel) in self.devices.items():
            out[device_id] = (tel.read_entry(self._tags.AICLK), tel.read_entry(self._tags.VCORE))
        return out

    def _send(self, mhz):
        for device_id, (pci, dev, _) in self.devices.items():
            reply = dev.arc_msg(FORCE_AICLK, True, [int(mhz)], 1000)
            exit_code = reply[0] if isinstance(reply, (list, tuple)) else reply
            if exit_code != 0:
                raise RuntimeError(
                    f"VF sweep: FORCE_AICLK {mhz} MHz failed on device {device_id} "
                    f"(/dev/tenstorrent/{pci}), ARC exit code {exit_code}"
                )

    def set_clock(self, mhz):
        """Force AICLK to mhz on every device and wait until telemetry reports it on all of them."""
        self._send(mhz)
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

    def release(self):
        """Hand the clock back to the firmware on every device. Never raises; logs failures."""
        for device_id, (pci, dev, _) in self.devices.items():
            try:
                reply = dev.arc_msg(FORCE_AICLK, True, [0], 1000)
                exit_code = reply[0] if isinstance(reply, (list, tuple)) else reply
                if exit_code != 0:
                    logger.error(f"VF sweep: releasing AICLK on device {device_id} returned ARC exit code {exit_code}")
            except Exception as e:  # keep releasing the others
                logger.error(f"VF sweep: releasing AICLK on device {device_id} (/dev/tenstorrent/{pci}) failed: {e}")
        self.forced_mhz = None
