# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Force the Tensix clock (AICLK) and core voltage (VDD) of tt-metal devices through ARC messages, for DIDT V/F sweeps.

The force goes through the chip's ARC firmware (tt-system-firmware smc_msg.h FORCE_AICLK) and holds until it is
released (argument 0) or the board is reset. It overrides the firmware's own clock choice, including TDP/TDC/thermal
throttling. Only chips with a PCIe (MMIO) connection can be addressed.

Telemetry is only as fresh as the firmware's last table update (100 ms by default), so every readback here waits for
TIMER_HEARTBEAT to advance first: a read right after a force can otherwise still show the old clock or voltage. For the
sweep the update interval is lowered (firmware >= 19.17, TT_SMC_MSG_CHARACTERISATION submessage
SET_TELEMETRY_UPDATE_INTERVAL) and restored to the default on release.
"""

import statistics
import threading
import time

import yaml
from loguru import logger

import ttnn

FORCE_AICLK = 0x33  # argument: MHz, 0 releases the force
FORCE_VDD = 0x39  # argument: mV, 0 releases the force; must stay inside the chip's VDD_LIMITS
# TT_SMC_MSG_CHARACTERISATION (0xC6) with submessage TT_SUB_MSG_SET_TELEMETRY_UPDATE_INTERVAL (0x5) in byte 1 of word 0;
# word 1 = interval in ms, 0 restores the default. tt_umd's arc_msg keeps only the low byte of word 0, so this goes
# through pyluwen's arc_msg_buf, which sends word 0 as given (on the tools queue, apart from tt_umd's queue).
SET_TELEMETRY_INTERVAL = 0xC6 | (0x5 << 8)
DEFAULT_TELEMETRY_INTERVAL_MS = 100
SWEEP_TELEMETRY_INTERVAL_MS = 10

# Telemetry recorded during each sweep step
STEP_TAGS = ("AICLK", "VCORE", "TDP", "TDC", "ASIC_TEMPERATURE", "AICLK_ARB_MAX")
# AICLK_ARB_MAX = allowed clock | (limiting arbiter << 16) (tt-system-firmware aiclk_ppm.h enum aiclk_arb_max)
ARB_MAX_NAMES = (
    "fmax",
    "tdp",
    "fast_tdc",
    "tdc",
    "thermal",
    "board_power",
    "voltage",
    "gddr_thermal",
    "doppler_slow",
    "doppler_critical",
    "host_fmax",
)


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


def decode_temperature(raw):
    """ASIC_TEMPERATURE is signed 16.16 fixed point, degrees C."""
    if raw & 0x80000000:
        raw -= 1 << 32
    return raw / 65536.0


def summarize(values):
    """(min, median, max) of a list, or None when it is empty."""
    return (min(values), statistics.median(values), max(values)) if values else None


class AiclkController:
    """Forces, verifies and releases AICLK and VDD on a set of tt-metal devices, and samples their telemetry."""

    def __init__(
        self,
        device_ids,
        settle_timeout_s=10.0,
        tolerance_mhz=1,
        vdd_tolerance_mv=3,
        telemetry_interval_ms=SWEEP_TELEMETRY_INTERVAL_MS,
    ):
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
        self.aiclk_limit_max = {}  # chip fmax, MHz: FORCE_AICLK bypasses every throttler, so never force above it
        for device_id in device_ids:
            pci = id_to_pci[device_id]
            dev = tt_umd.TTDevice.create(pci)
            dev.init_tt_device()
            tel = dev.get_arc_telemetry_reader()
            self.devices[device_id] = (pci, dev, tel)
            limits = tel.read_entry(self._tags.VDD_LIMITS)  # vdd_max << 16 | vdd_min, mV
            self.vdd_limits[device_id] = (limits & 0xFFFF, limits >> 16)
            if not tel.is_entry_available(self._tags.AICLK_LIMIT_MAX):
                raise RuntimeError(
                    f"VF sweep: device {device_id} does not report AICLK_LIMIT_MAX, so a forced clock cannot be checked "
                    "against its fmax; update the firmware"
                )
            self.aiclk_limit_max[device_id] = tel.read_entry(self._tags.AICLK_LIMIT_MAX)
        self.forced_mhz = None
        self.vdd_forced = False
        self._sampler = None
        # A thermal trip resets the chip through the board controller, which then counts it; the count survives the
        # chip reset, so a rise between here and release() means a forced point ended in a thermal reset.
        self.therm_trips_start = self._read_tag_all("THERM_TRIP_COUNT")
        # Tags every device publishes (AICLK_ARB_MAX is in firmware 19.5+); the limiting arbiter is packed into its
        # upper half only from 19.7
        self.step_tags = [n for n in STEP_TAGS if all(None not in [v] for v in self._read_tag_all(n).values())]
        bundles = [v for v in self._read_tag_all("FLASH_BUNDLE_VERSION").values() if v is not None]
        self.arbiter_named = bool(bundles) and min(((b >> 24) & 0xFF, (b >> 16) & 0xFF) for b in bundles) >= (19, 7)
        self.interval_changed = False
        try:
            self.telemetry_interval_ms = self._set_telemetry_interval(telemetry_interval_ms)
        except Exception:
            # Nothing is forced yet, but some chips may already publish every few ms: put them back before raising,
            # since the caller has no controller to release
            if self.interval_changed:
                try:
                    self._set_telemetry_interval(0)
                except Exception as e:
                    logger.error(f"VF sweep: restoring the telemetry interval failed: {e}")
            raise

    # --- telemetry -------------------------------------------------------------------------------------------

    def _read_tag_all(self, name):
        tag = getattr(self._tags, name)
        out = {}
        for device_id, (_, _, tel) in self.devices.items():
            out[device_id] = tel.read_entry(tag) if tel.is_entry_available(tag) else None
        return out

    def _heartbeats(self):
        return {d: tel.read_entry(self._tags.TIMER_HEARTBEAT) for d, (_, _, tel) in self.devices.items()}

    def wait_for_updates(self, n=1):
        """Block until the firmware has published n new telemetry tables on every device, so the next read reflects
        what happened before this call."""
        start = self._heartbeats()
        poll_s = max(self.telemetry_interval_ms / 4000.0, 0.0005)
        deadline = time.time() + self.settle_timeout_s
        while True:
            now = self._heartbeats()
            if all(now[d] - start[d] >= n for d in start):
                return
            if time.time() > deadline:
                raise RuntimeError(f"VF sweep: telemetry did not update within {self.settle_timeout_s} s: {now}")
            time.sleep(poll_s)

    def _set_telemetry_interval(self, interval_ms):
        """Set the firmware's telemetry update interval on every device; returns the interval read back on all of them
        (the default if it could not be changed)."""
        try:
            import pyluwen
        except ImportError:
            logger.warning(
                "VF sweep: pyluwen is not installed, telemetry stays at the firmware default "
                f"({DEFAULT_TELEMETRY_INTERVAL_MS} ms); readbacks are slower"
            )
            return DEFAULT_TELEMETRY_INTERVAL_MS
        for device_id, (pci, _, _) in self.devices.items():
            try:
                chip = pyluwen.PciChip(pci_interface=pci).as_bh()
                reply = chip.arc_msg_buf([SET_TELEMETRY_INTERVAL, interval_ms, 0, 0, 0, 0, 0, 0])
                # Response word 0: status in the low byte (0xFF = message not recognized), exit code in the high half
                status, exit_code = reply[0] & 0xFF, reply[0] >> 16
                if status == 0xFF or exit_code != 0:
                    logger.warning(
                        f"VF sweep: firmware on device {device_id} refused telemetry interval {interval_ms} ms "
                        f"(status {status:#x}, exit code {exit_code}); firmware older than 19.17 does not support it"
                    )
                elif interval_ms:
                    self.interval_changed = True
            except Exception as e:
                logger.warning(f"VF sweep: setting the telemetry interval on device {device_id} failed: {e}")
        # UMD's telemetry reader caches UPDATE_TELEM_SPEED when it is created, so measure the update rate instead
        measured = self._measured_interval_ms()
        stalled = sorted(d for d, m in measured.items() if m == float("inf"))
        if stalled:
            raise RuntimeError(f"VF sweep: telemetry heartbeat did not advance on devices {stalled}")
        target = interval_ms or DEFAULT_TELEMETRY_INTERVAL_MS
        if any(abs(m - target) > max(1.0, 0.2 * target) for m in measured.values()):
            logger.warning(f"VF sweep: telemetry updates every {measured} ms, asked for {target} ms")
        else:
            logger.info(f"VF sweep: telemetry updates every {target} ms on devices {sorted(self.devices)}")
        return max(measured.values())

    def _measured_interval_ms(self, window_s=0.5):
        """{device_id: ms between telemetry updates}, from how far TIMER_HEARTBEAT advances in window_s."""
        time.sleep(0.25)  # let a new interval take effect
        start, t0 = self._heartbeats(), time.time()
        time.sleep(window_s)
        end, dt = self._heartbeats(), time.time() - t0
        return {d: (dt * 1000.0 / (end[d] - start[d])) if end[d] > start[d] else float("inf") for d in start}

    def read(self):
        """Current {device_id: (AICLK MHz, VCORE mV)} from ARC telemetry."""
        out = {}
        for device_id, (_, _, tel) in self.devices.items():
            out[device_id] = (tel.read_entry(self._tags.AICLK), tel.read_entry(self._tags.VCORE))
        return out

    def _read_step_sample(self, tel):
        return {name: tel.read_entry(getattr(self._tags, name)) for name in self.step_tags}

    def start_step_sampling(self):
        """Record STEP_TAGS on every device at every telemetry update, in a background thread, until
        stop_step_sampling(). Call it only while no force is being changed."""
        assert self._sampler is None, "step sampler already running"
        stop = threading.Event()
        samples = {d: [] for d in self.devices}
        errors = []

        def run():
            last = {d: None for d in self.devices}
            # One pass reads 7 tags per device over PCIe from this Python thread, which shares the GIL with the
            # test's dispatch. Keep the read rate about constant (~2.8k reads/s) whatever the device count, so the
            # sampler does not change the load it measures: one device every telemetry update, a 32-chip Galaxy
            # every 80 ms.
            poll_s = max(self.telemetry_interval_ms / 1000.0, 0.0025 * len(self.devices))
            while not stop.is_set():
                for d, (_, _, tel) in self.devices.items():
                    try:
                        hb = tel.read_entry(self._tags.TIMER_HEARTBEAT)
                        if hb != last[d]:
                            last[d] = hb
                            samples[d].append(self._read_step_sample(tel))
                    except Exception as e:  # one failed read must not end the step's sampling
                        errors.append(f"device {d}: {e}")
                        if len(errors) > 100:
                            return
                stop.wait(poll_s)

        thread = threading.Thread(target=run, name="vf-sweep-telemetry", daemon=True)
        thread.start()
        self._sampler = (stop, thread, samples, errors)

    def stop_step_sampling(self):
        """Stop the step sampler; returns {device_id: summary} (see _summarize_step), or None if none was running."""
        if self._sampler is None:
            return None
        stop, thread, samples, errors = self._sampler
        stop.set()
        thread.join(timeout=5)
        self._sampler = None
        if errors:
            logger.warning(f"VF sweep: {len(errors)} telemetry reads failed during the step, first: {errors[0]}")
        return {d: self._summarize_step(s) for d, s in samples.items()}

    def _summarize_step(self, samples):
        """min/median/max of the step telemetry, how many samples held the forced clock, and how often the firmware's
        own arbiters would have allowed less than the forced clock (the step then ran past a power, current or
        thermal limit that the force overrode)."""
        out = {"samples": len(samples)}
        if not samples:
            return out
        for key, tag, conv in (
            ("aiclk", "AICLK", None),
            ("vcore_mv", "VCORE", None),
            ("tdp_w", "TDP", None),
            ("tdc_a", "TDC", None),
            ("temp_c", "ASIC_TEMPERATURE", decode_temperature),
        ):
            if tag in self.step_tags:
                out[key] = summarize([conv(s[tag]) if conv else s[tag] for s in samples])
        if self.forced_mhz and "AICLK" in self.step_tags:
            out["clock_held"] = sum(abs(s["AICLK"] - self.forced_mhz) <= self.tolerance_mhz for s in samples)
        if self.forced_mhz and "AICLK_ARB_MAX" in self.step_tags:
            capped = {}
            for s in samples:
                allowed, arb = s["AICLK_ARB_MAX"] & 0xFFFF, s["AICLK_ARB_MAX"] >> 16
                if allowed < self.forced_mhz:
                    if not self.arbiter_named:
                        name = "an arbiter"  # firmware before 19.7 does not say which
                    else:
                        name = ARB_MAX_NAMES[arb] if arb < len(ARB_MAX_NAMES) else str(arb)
                    capped[name] = capped.get(name, 0) + 1
            out["fw_would_cap"] = capped
        return out

    def therm_trips(self):
        """{device_id: (THERM_TRIP_COUNT at start, now)}."""
        now = self._read_tag_all("THERM_TRIP_COUNT")
        return {d: (self.therm_trips_start.get(d), now.get(d)) for d in self.devices}

    # --- forces --------------------------------------------------------------------------------------------

    def _msg(self, device_id, code, arg, what):
        pci, dev, _ = self.devices[device_id]
        reply = dev.arc_msg(code, True, [int(arg)], 1000)
        exit_code = reply[0] if isinstance(reply, (list, tuple)) else reply
        if exit_code != 0:
            raise RuntimeError(
                f"VF sweep: {what} {arg} failed on device {device_id} (/dev/tenstorrent/{pci}), ARC exit code {exit_code}"
            )

    def check_clocks(self, freqs_mhz):
        """Refuse any clock that is not positive or is above a device's fmax (AICLK_LIMIT_MAX), before forcing it."""
        for mhz in freqs_mhz:
            over = {d: f for d, f in self.aiclk_limit_max.items() if mhz > f}
            if mhz <= 0 or over:
                raise ValueError(
                    f"VF sweep: AICLK {mhz} MHz is not allowed"
                    + (f"; above fmax on devices {over}" if over else "")
                    + " (FORCE_AICLK bypasses the firmware's throttling)"
                )

    def set_clock(self, mhz):
        """Force AICLK to mhz on every device, wait until telemetry reports it on all of them and VCORE has settled at
        the firmware's voltage for that clock; returns {device_id: (AICLK, VCORE)}.

        Lowering the clock lowers the voltage after the clock (DVFS order), so VCORE is read only once it is steady.
        """
        self.check_clocks([mhz])
        for device_id in self.devices:
            self._msg(device_id, FORCE_AICLK, mhz, "FORCE_AICLK MHz")
        self.forced_mhz = mhz
        deadline = time.time() + self.settle_timeout_s
        while True:
            self.wait_for_updates(1)
            readings = self.read()
            off = {d: r for d, r in readings.items() if abs(r[0] - mhz) > self.tolerance_mhz}
            if not off:
                break
            if time.time() > deadline:
                raise RuntimeError(
                    f"VF sweep: AICLK did not reach {mhz} MHz within {self.settle_timeout_s} s on devices "
                    + ", ".join(f"{d} (AICLK {r[0]} MHz)" for d, r in sorted(off.items()))
                )
        vcore = self._settled_vcore()
        return {d: (readings[d][0], vcore[d]) for d in self.devices}

    SETTLE_WINDOW_S = 0.2  # a ramp slower than 2 mV per window would otherwise look settled at a fast interval

    def _settled_vcore(self):
        """{device_id: VCORE} once every device's readings from new telemetry tables agree within 2 mV over at least
        3 updates and SETTLE_WINDOW_S.

        Each reading comes from a new telemetry table, so readings from before the last force cannot make it look
        settled. On timeout the last readings are returned with a warning.
        """
        self.wait_for_updates(1)  # the first table after the force may predate it
        history = {d: [] for d in self.devices}  # (time, VCORE)
        deadline = time.time() + self.settle_timeout_s
        while True:
            self.wait_for_updates(1)
            now = time.time()
            for d, (_, vcore) in self.read().items():
                history[d].append((now, vcore))
                # keep the newest readings that cover the window, plus at least 3
                while len(history[d]) > 3 and now - history[d][1][0] >= self.SETTLE_WINDOW_S:
                    history[d].pop(0)
            if all(
                len(h) >= 3
                and h[-1][0] - h[0][0] >= self.SETTLE_WINDOW_S
                and max(v for _, v in h) - min(v for _, v in h) <= 2
                for h in history.values()
            ):
                return {d: h[-1][1] for d, h in history.items()}
            if time.time() > deadline:
                logger.warning(f"VF sweep: VCORE still moving after {self.settle_timeout_s} s: {history}")
                return {d: h[-1][1] for d, h in history.items()}

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
            logger.info(
                "VF sweep: VCORE setpoint corrected for the telemetry offset: "
                + ", ".join(
                    f"device {d} {setpoints[d]} -> {sp} mV (read {readback[d]} mV)"
                    for d, sp in sorted(corrected.items())
                )
            )
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
        """Stop sampling, release both forces and restore the telemetry interval on every device. Never raises; logs
        failures. Logs a THERM_TRIP_COUNT rise as an error."""
        try:
            self.stop_step_sampling()
        except Exception as e:
            logger.error(f"VF sweep: stopping the telemetry sampler failed: {e}")
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
        if self.interval_changed:
            try:
                self._set_telemetry_interval(0)
                self.interval_changed = False
            except Exception as e:
                logger.error(f"VF sweep: restoring the telemetry interval failed: {e}")
        try:
            for d, (start, now) in self.therm_trips().items():
                if start is not None and now is not None and now > start:
                    logger.error(
                        f"VF sweep: THERM_TRIP_COUNT on device {d} rose from {start} to {now}: a forced point ended in a "
                        "thermal trip and chip reset"
                    )
        except Exception as e:
            logger.error(f"VF sweep: reading THERM_TRIP_COUNT failed: {e}")
