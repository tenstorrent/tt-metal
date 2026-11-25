# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor

from loguru import logger
import torch

from tt_py_test_utils_common.utility_functions import comp_pcc
import ttnn


class OpParameter:
    def __init__(self, shape, dtype, layout, mem_config):
        self.shape = shape
        self.mem_config = mem_config
        self.dtype = dtype
        self.layout = layout


class OpTestBase:
    # VF sweep settings for tests that do not pass them to the constructor. tests/didt/conftest.py fills these from
    # --didt-vf-sweep / --sweep-iterations for each test, so every didt test gets the sweep without signature changes.
    vf_sweep_defaults = {"freqs": None, "sweep_iterations": None, "vdd_offsets": None}

    def __init__(
        self,
        mesh_device,
        activation: OpParameter,
        arguments: [OpParameter],
        out_mem_config,
        out_dtype,
        program_config,
        compute_config,
        loop_count=1000,
        determinism_check_enabled=False,
        determinism_check_interval=False,
        sync_every_iteration=True,
        vf_sweep_freqs=None,
        sweep_iterations=None,
        vdd_offsets=None,
    ):
        self.mesh_device = mesh_device
        # This will be removed once we rebase onto new main with the new open_mesh_device API
        # that will allow opening mesh_device with any specific device from the available ones
        if isinstance(mesh_device, ttnn.MeshDevice):
            self.from_torch_mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
            self.device_ids = self.mesh_device.get_device_ids()
        else:
            # ttnn.Device
            self.from_torch_mesh_mapper = None
            self.device_ids = [self.mesh_device.id()]

        self.activation = activation
        self.arguments = arguments
        self.out_dtype = out_dtype
        self.out_mem_config = out_mem_config
        self.program_config = program_config
        self.compute_config = compute_config
        self.loop_count = loop_count
        self.determinism_check_enabled = determinism_check_enabled
        self.determinism_check_interval = determinism_check_interval
        self.sync_every_iteration = sync_every_iteration

        # VF sweep: for each forced AICLK in vf_sweep_freqs (in that order), run sweep_iterations iterations at each
        # VDD offset in vdd_offsets (mV over the VCORE the firmware picks for that clock; 0 = firmware voltage).
        if vf_sweep_freqs is None:
            vf_sweep_freqs = OpTestBase.vf_sweep_defaults["freqs"]
        if sweep_iterations is None:
            sweep_iterations = OpTestBase.vf_sweep_defaults["sweep_iterations"]
        if vdd_offsets is None:
            vdd_offsets = OpTestBase.vf_sweep_defaults["vdd_offsets"]
        self.vf_sweep_freqs = list(vf_sweep_freqs) if vf_sweep_freqs else None
        self.sweep_iterations = sweep_iterations
        self.vdd_offsets = list(vdd_offsets) if vdd_offsets else [0]
        if self.vf_sweep_freqs:
            if not self.sweep_iterations or self.sweep_iterations <= 0:
                raise ValueError("VF sweep needs a positive sweep_iterations (--sweep-iterations)")
            if any(o < 0 or o > 100 for o in self.vdd_offsets):
                raise ValueError(f"VDD offsets must be 0-100 mV, got {self.vdd_offsets}")
            # One step per (clock, offset), offsets in the order given within each clock.
            self.vf_steps = [(mhz, off) for mhz in self.vf_sweep_freqs for off in self.vdd_offsets]
            self.loop_count = len(self.vf_steps) * self.sweep_iterations

        # Weights and activations tensors are needed for subclasses to run the operation
        self.activations = None
        self.inputs = []

    def get_device(self, device_idx):
        return self.mesh_device

    # Override if needed
    def set_seed(self):
        torch.manual_seed(1234)

    # Override if needed
    def generate_torch_activations(self, shape):
        return torch.randn(shape)

    # Override if needed
    def generate_torch_input(self, shape):
        return torch.randn(shape)

    def generate_tt_activations_from_torch(self, torch_tensor):
        return ttnn.from_torch(
            torch_tensor,
            dtype=self.activation.dtype,
            layout=self.activation.layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=self.mesh_device,
            mesh_mapper=self.from_torch_mesh_mapper,
        )

    def generate_tt_input_from_torch(self, torch_tensor, dtype, layout, mem_config, ind):
        return ttnn.from_torch(
            torch_tensor,
            dtype=dtype,
            layout=layout,
            memory_config=mem_config,
            device=self.mesh_device,
            mesh_mapper=self.from_torch_mesh_mapper,
        )

    def convert_activations_to_memory_config(self, activations):
        return ttnn.to_memory_config(activations, self.activation.mem_config)

    def run_device_operation(self):
        # Default Op is matmul. Override in derived class if needed.
        return ttnn.matmul(
            self.activations,
            self.inputs[0],
            program_config=self.program_config,
            memory_config=self.out_mem_config,
            dtype=self.out_dtype,
            compute_kernel_config=self.compute_config,
        )

    def deallocate_activations(self):
        self.activations.deallocate(True)

    def run_op_test(self):
        clock = None
        if self.vf_sweep_freqs:
            from tt_py_test_utils_common.aiclk_control import AiclkController

            clock = AiclkController(self.device_ids)
            logger.info(
                f"VF sweep: AICLK {self.vf_sweep_freqs} MHz x VDD offsets {self.vdd_offsets} mV, "
                f"{self.sweep_iterations} iterations each ({self.loop_count} in total), "
                f"determinism check every {self.determinism_check_interval}"
            )
        try:
            self._run_op_test(clock)
        finally:
            if clock is not None:
                clock.release()
                logger.info(f"VF sweep: released the forced AICLK and VDD on devices {list(self.device_ids)}")

    def _set_sweep_clock(self, clock, mhz, label):
        """Drain the devices, hand VDD back to the firmware, force AICLK, wait until every device reports it and
        return {device_id: VCORE mV} (the firmware's voltage for that clock).

        This readback is the only telemetry read of a sweep clock; nothing is read during the iterations.
        """
        for device_idx in self.device_ids:
            ttnn.device.synchronize_device(self.get_device(device_idx))
        if clock.vdd_forced:
            clock.release_vdd()  # lower or raise the clock on the firmware's own V/F curve
        readings = clock.set_clock(mhz)
        aiclk = sorted({r[0] for r in readings.values()})
        vcore = {device_idx: r[1] for device_idx, r in readings.items()}
        logger.info(
            f"VF sweep: {label}: AICLK forced to {mhz} MHz, telemetry AICLK {aiclk} MHz on all "
            f"{len(readings)} devices, VCORE {min(vcore.values())}-{max(vcore.values())} mV"
        )
        return vcore

    def _set_sweep_vdd(self, clock, base_vcore, offset_mv, label):
        """Drain the devices and force each one's VDD to its firmware VCORE + offset_mv; return the readback."""
        for device_idx in self.device_ids:
            ttnn.device.synchronize_device(self.get_device(device_idx))
        targets = {d: v + offset_mv for d, v in base_vcore.items()}
        readback = clock.set_vdd(targets)
        logger.info(
            f"VF sweep: {label}: VDD forced to firmware VCORE +{offset_mv} mV "
            f"(targets {min(targets.values())}-{max(targets.values())} mV), "
            f"VCORE readback {min(readback.values())}-{max(readback.values())} mV"
        )
        return readback

    def _run_op_test(self, clock):
        num_devices = len(self.device_ids)
        with ThreadPoolExecutor(num_devices) as pool:
            self.set_seed()

            logger.debug(f"Running on {num_devices} devices")

            a_shape = self.activation.shape

            num_activation_tensors = 1
            if self.determinism_check_enabled:
                # If we are running determinism checks, we want to switch activation tensors
                # every time we complete an iteration of a determinism check, to confirm that
                # device is producing new results, and not just reusing an already existing buffer
                num_activation_tensors = 10

            A = []
            for act in range(num_activation_tensors):
                A.append(self.generate_torch_activations(a_shape))
            B = [self.generate_torch_input(a.shape) for a in self.arguments]

            logger.debug("Pushing activations to devices...")
            a_t = []
            for act in range(num_activation_tensors):
                a_t.append(self.generate_tt_activations_from_torch(A[act]))

            logger.debug("Pushing inputs to devices...")
            self.inputs = [
                self.generate_tt_input_from_torch(b, a.dtype, a.layout, a.mem_config, i)
                for i, (b, a) in enumerate(zip(B, self.arguments))
            ]

            logger.debug("Activations and inputs pushed to devices!")

            reference_out = None
            if self.determinism_check_enabled:
                if clock is not None:
                    # A chip that is non-deterministic at its default clock can also corrupt the reference, which
                    # would then fail every sweep step. Take the reference at the lowest sweep clock instead.
                    self._set_sweep_clock(clock, min(self.vf_sweep_freqs), "reference outputs")
                # One reference run per activation; we then cycle through them and compare output to
                # reference_out[act]. Deallocate only for L1 due to memory constraints.
                # For DRAM we keep all a_t[act] live; we still switch self.activations each time, so
                # determinism checking is unchanged (different inputs, match reference per activation).
                num_nd_outputs = [0] * num_devices
                reference_out = [None for _ in range(num_activation_tensors)]
                for act in range(num_activation_tensors):
                    self.activations = self.convert_activations_to_memory_config(a_t[act])
                    output = self.run_device_operation()
                    reference_out[act] = list(pool.map(ttnn.to_torch, ttnn.get_device_tensors(output.cpu())))
                    output.deallocate(True)
                    if self.activation.mem_config.buffer_type == ttnn.BufferType.L1:
                        self.deallocate_activations()

            current_act_tensor = 0
            out = [None] * num_devices
            self.activations = self.convert_activations_to_memory_config(a_t[0])

            sweep_steps = []  # one record per VF sweep (clock, VDD offset)
            base_vcore = None  # firmware VCORE per device at the current clock
            logger.debug("Starting iterations")
            for i in range(self.loop_count):
                if clock is not None and i % self.sweep_iterations == 0:
                    step_idx = i // self.sweep_iterations
                    mhz, offset = self.vf_steps[step_idx]
                    label = (
                        f"step {step_idx + 1}/{len(self.vf_steps)} " f"(iterations {i}-{i + self.sweep_iterations - 1})"
                    )
                    if step_idx == 0 or self.vf_steps[step_idx - 1][0] != mhz:
                        base_vcore = self._set_sweep_clock(clock, mhz, label)
                    if offset:
                        vcore = self._set_sweep_vdd(clock, base_vcore, offset, label)
                    else:
                        if clock.vdd_forced:
                            for device_idx in self.device_ids:
                                ttnn.device.synchronize_device(self.get_device(device_idx))
                            clock.release_vdd()
                        vcore = base_vcore
                    step = {
                        "mhz": mhz,
                        "offset": offset,
                        "checks": 0,
                        "nd": [0] * num_devices,
                        "vcore": [vcore[device_idx] for device_idx in self.device_ids],
                    }
                    sweep_steps.append(step)

                out = self.run_device_operation()

                check_iteration = self.determinism_check_enabled and i % self.determinism_check_interval == 0

                if self.sync_every_iteration or check_iteration or i == self.loop_count - 1:
                    for device_idx in self.device_ids:
                        logger.debug(f"Start sync device id: {device_idx}")
                        ttnn.device.synchronize_device(self.get_device(device_idx))
                        logger.debug(f"End sync device id: {device_idx}")

                # Check if the output matches the first run output
                if check_iteration:
                    outputs = list(pool.map(ttnn.to_torch, ttnn.get_device_tensors(out.cpu())))
                    matches = list(pool.map(torch.equal, reference_out[current_act_tensor], outputs))
                    at_clock = (
                        f" (VF sweep AICLK {step['mhz']} MHz, VDD +{step['offset']} mV)" if clock is not None else ""
                    )
                    if clock is not None:
                        step["checks"] += 1

                    for output_id in range(num_devices):
                        device_idx = self.device_ids[output_id]
                        if matches[output_id]:
                            logger.debug(f"Device {device_idx} PCC: 1.0")
                        else:
                            # For determinism check, we avoid calling comp_pcc func as it is heavy and with too many operations,
                            # part of the code that replaces nans/infs with zeros starts leaking memory, even if deallocation is forced,
                            # so we call it only in case we see tensors are not equal
                            _, pcc = comp_pcc(reference_out[current_act_tensor][output_id], outputs[output_id])
                            logger.warning(f"Device {device_idx} PCC: {pcc}{at_clock}")
                            num_nd_outputs[output_id] += 1
                            if clock is not None:
                                step["nd"][output_id] += 1

                    current_act_tensor = (current_act_tensor + 1) % num_activation_tensors
                    logger.debug("Switching activation tensor for new determinism iterations...")
                    if self.activation.mem_config.buffer_type == ttnn.BufferType.L1:
                        self.deallocate_activations()
                    self.activations = self.convert_activations_to_memory_config(a_t[current_act_tensor])

                out.deallocate(True)

                logger.debug(f"Iteration = {i}, done!")

        if clock is not None:
            self._log_sweep_summary(sweep_steps)

        catch_nd_on_fail = True
        if self.determinism_check_enabled:
            for nd_output_id in range(num_devices):
                device_idx = self.device_ids[nd_output_id]
                if num_nd_outputs[nd_output_id] > 0:
                    logger.warning(
                        f"Number of non-deterministic outputs on device {device_idx} is {num_nd_outputs[nd_output_id]}"
                    )
                    catch_nd_on_fail = False
                else:
                    logger.debug(
                        f"Number of non-deterministic outputs on device {device_idx} is {num_nd_outputs[nd_output_id]}"
                    )
        assert catch_nd_on_fail

    def _log_sweep_summary(self, sweep_steps):
        """One line per (clock, VDD offset): checks, which devices failed and how often, VCORE readback."""
        lines = [
            "VF sweep summary (VCORE read back once per step, right after the clock/VDD was set, devices idle; "
            "+0 = firmware voltage):"
        ]
        for step in sweep_steps:
            failing = [
                f"dev {self.device_ids[o]}: {n}/{step['checks']} (VCORE {step['vcore'][o]} mV)"
                for o, n in enumerate(step["nd"])
                if n > 0
            ]
            vcore = sorted(step["vcore"])
            lines.append(
                f"  AICLK {step['mhz']:>5} MHz VDD +{step['offset']:>3} mV | checks {step['checks']:>4} | "
                + (f"ND {', '.join(failing) if failing else 'none'} | " if self.determinism_check_enabled else "")
                + f"VCORE min/median/max {vcore[0]}/{vcore[len(vcore) // 2]}/{vcore[-1]} mV"
            )
        if self.determinism_check_enabled:
            lines += self._sweep_result_lines(sweep_steps)
        logger.info("\n".join(lines))

    def _sweep_result_lines(self, sweep_steps):
        """Pass/fail grid (clock x VDD offset) over all devices, then the same grid per failing device with its VCORE.

        A point PASSES when every determinism check at it was bit-exact on every device.
        """
        clocks = list(dict.fromkeys(step["mhz"] for step in sweep_steps))
        offsets = list(dict.fromkeys(step["offset"] for step in sweep_steps))
        at = {(step["mhz"], step["offset"]): step for step in sweep_steps}

        def cell(step, o=None):
            nd = step["nd"] if o is None else [step["nd"][o]]
            failed = sum(nd)
            if o is not None:
                text = f"{'FAIL' if failed else 'PASS'} {failed}/{step['checks']} @{step['vcore'][o]}mV"
            elif failed:
                text = "FAIL " + ",".join(f"d{self.device_ids[i]}:{n}" for i, n in enumerate(step["nd"]) if n)
            else:
                text = "PASS"
            return text

        def grid(title, o=None):
            rows = [
                [f"{mhz} MHz"] + [cell(at[(mhz, off)], o) if (mhz, off) in at else "-" for off in offsets]
                for mhz in clocks
            ]
            header = ["AICLK \\ VDD"] + [f"+{off} mV" for off in offsets]
            widths = [max(len(r[c]) for r in rows + [header]) for c in range(len(header))]
            fmt = lambda r: "    " + " | ".join(v.ljust(w) for v, w in zip(r, widths))
            return [title, fmt(header)] + [fmt(r) for r in rows]

        failing = [o for o in range(len(self.device_ids)) if any(step["nd"][o] for step in sweep_steps)]
        lines = ["", *grid("VF sweep result, all devices (PASS = every check bit-exact on every device):")]
        if not failing:
            lines.append(f"  every clock and VDD combination PASSED on all {len(self.device_ids)} devices")
        for o in failing:
            lines += ["", *grid(f"Device {self.device_ids[o]} (failed checks/checks @ VCORE readback):", o)]
            for mhz in clocks:
                ran = [off for off in offsets if (mhz, off) in at]
                ok = [at[(mhz, off)]["nd"][o] == 0 for off in ran]
                # lowest offset from which every higher offset also passed
                k = next((k for k in range(len(ok)) if all(ok[k:])), None)
                if k is None:
                    verdict = f"fails at +{ran[-1]} mV" + (
                        " (passed at " + ", ".join(f"+{off}" for off, good in zip(ran, ok) if good) + " mV)"
                        if any(ok)
                        else ", every VDD offset failed"
                    )
                elif k == 0 and ran[0] == 0:
                    verdict = "passes at the firmware voltage"
                else:
                    verdict = f"passes from +{ran[k]} mV"
                lines.append(f"    {mhz} MHz: {verdict}")
        return lines


def get_mesh_grid_size(mesh_device):
    return mesh_device.compute_with_storage_grid_size()
