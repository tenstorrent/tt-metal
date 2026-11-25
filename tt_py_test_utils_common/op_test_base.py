# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor

from loguru import logger
import torch

from tt_py_test_utils_common import nd_core_map
from tt_py_test_utils_common.utility_functions import comp_pcc
import ttnn


# Physical location of each device id, e.g. "UBB3 ASIC2 /dev/tenstorrent/17 0000:c2:00.0 board=0x47331831011",
# read from the cluster descriptor metal is already using (same source as tt-triage's T<tray>:N<asic> labels)
def chip_locations():
    locations = {}
    try:
        import tt_umd

        with open(ttnn.cluster.serialize_cluster_descriptor()) as f:
            cd = tt_umd.ClusterDescriptor.create_from_yaml_content(f.read())
        pci, bdfs = cd.get_chips_with_mmio(), cd.get_chip_pci_bdfs()
        for chip in cd.get_all_chips():
            tray = cd.get_tray_id(chip)
            where = f"UBB{tray} ASIC{cd.get_asic_location(chip)}" if tray is not None else "non-UBB"
            if chip in pci:
                where += f" /dev/tenstorrent/{pci[chip]}"
            if chip in bdfs:
                where += f" {bdfs[chip]}"
            locations[chip] = f"{where} board=0x{cd.get_board_id_for_chip(chip):x}"
    except Exception as e:
        logger.warning(f"Chip location lookup failed, ND logs will show device ids only: {e}")
    return locations


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

        # ND core attribution. All core coordinates below are device logical coordinates.
        # device_idx -> {(x, y): failing checks}
        self.nd_core_hits = {}
        # device_idx -> {(x, y): [block_structure() per failing check]} (rectangular-block layouts only)
        self.nd_core_structures = {}
        self.nd_core_translated = {}
        self._chip_locations = None
        # Layout (output -> core map) of each check iteration, cached per signature, and every change of it:
        # [(iteration, signature)], so each attribution can cite the layout that check actually ran with
        self._nd_layout_cache = {}
        self.nd_layout_changes = []
        self.nd_reference_layouts = {}  # activation index -> layout of the run that made that reference output

    def get_device(self, device_idx):
        return self.mesh_device

    # --- Which core wrote which part of the output -------------------------------------------------------------
    #
    # A test declares how its output is laid out over cores in one of these ways (first match wins):
    #   nd_core_block = (rows, cols) [+ nd_core_linear_x, nd_core_transpose, nd_core_origin]: one rectangular block
    #       per core, the layout the program config fixes (matmul 2D / 1D)
    #   nd_core_capture = "conv2d" | "matmul_config": the op decides its layout at run time; each check iteration's
    #       op call is observed (graph capture / passive program-config listener, neither changes execution) and
    #       that iteration's layout is used. "conv2d" also needs nd_conv_out_hw = (output height, output width).
    #   nd_core_owner = callable(output_shape) -> (owner tile table, cores): a fixed per-tile table (nd_core_map.py
    #       owner_* builders) from the op's own work split
    #   nothing, and a sharded output: the output's shard spec read from that check's output tensor
    # Otherwise the mismatch is reported without a core, with the reason.

    def _run_captured(self):
        """Run the op once while observing the decisions it makes at run time; returns (output, observation)."""
        mode = getattr(self, "nd_core_capture", None)
        if mode == "conv2d":
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            try:
                out = self.run_device_operation()
            finally:
                graph = ttnn.graph.end_graph_capture()
            return out, ("conv2d", graph)
        if mode == "matmul_config":
            ttnn.graph.begin_program_config_capture()
            try:
                out = self.run_device_operation()
            finally:
                config = ttnn.graph.end_program_config_capture()
            return out, ("matmul_config", config)
        raise ValueError(f"unknown nd_core_capture {mode!r}")

    def _cached_layout(self, key, build):
        if key not in self._nd_layout_cache:
            try:
                self._nd_layout_cache[key] = build()
            except Exception as e:  # an unmappable layout is reported, never guessed
                self._nd_layout_cache[key] = {"kind": "none", "signature": str(key), "reason": f"{e}"}
        return self._nd_layout_cache[key]

    def _layout_for(self, out, observation=None):
        """The output -> core map of one op call: {"kind": "block" | "owner" | "none", "signature": ..., ...}."""
        shape = tuple(out.shape)
        block = getattr(self, "nd_core_block", None)
        if block is not None:
            origin = tuple(getattr(self, "nd_core_origin", (0, 0)))
            linear_x = getattr(self, "nd_core_linear_x", None)
            transpose = getattr(self, "nd_core_transpose", False)
            fold_batch = bool(getattr(self, "nd_core_fold_batch", False))
            sig = (
                f"declared blocks {tuple(block)} linear_x={linear_x} transpose={transpose} origin={origin}"
                f"{' fold_batch' if fold_batch else ''}"
            )
            return {
                "kind": "block",
                "block": tuple(block),
                "linear_x": linear_x,
                "transpose": transpose,
                "origin": origin,
                "fold_batch": fold_batch,
                "grid": getattr(self, "nd_core_grid_size", None),
                "signature": sig,
            }
        if observation is not None:
            kind, data = observation
            if kind == "conv2d":
                slices = nd_core_map.conv_slices_from_graph(data)
                return self._cached_layout(("conv2d", shape, repr(slices)), lambda: self._conv_layout(shape, slices))
            if kind == "matmul_config":
                return self._cached_layout(("matmul", shape, repr(data)), lambda: self._matmul_layout(data))
        owner_fn = getattr(self, "nd_core_owner", None)
        if owner_fn is not None:

            def build():
                owner, cores = owner_fn(shape)
                return {"kind": "owner", "owner": owner, "cores": cores, "signature": f"declared owner table {shape}"}

            return self._cached_layout(("owner", shape), build)
        mc = out.memory_config()
        if mc.is_sharded():
            return self._cached_layout(("shard", shape, str(mc)), lambda: self._shard_layout(shape, mc))
        return {
            "kind": "none",
            "signature": f"none {mc.memory_layout}",
            "reason": f"{mc.memory_layout} {mc.buffer_type} output and the test declares no core map",
        }

    @staticmethod
    def _shard_layout(shape, mc):
        spec = mc.shard_spec
        layout = {
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED: "HEIGHT",
            ttnn.TensorMemoryLayout.WIDTH_SHARDED: "WIDTH",
            ttnn.TensorMemoryLayout.BLOCK_SHARDED: "BLOCK",
        }[mc.memory_layout]
        grid = [((r.start.x, r.start.y), (r.end.x, r.end.y)) for r in spec.grid.ranges()]
        row_major = spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
        owner, cores = nd_core_map.owner_sharded(shape, layout, tuple(spec.shape), grid, row_major)
        sig = f"{layout} sharded {tuple(spec.shape)} on {grid} {'ROW' if row_major else 'COL'}_MAJOR"
        return {"kind": "owner", "owner": owner, "cores": cores, "signature": sig}

    def _conv_layout(self, shape, slices):
        out_h, out_w = self.nd_conv_out_hw
        if not slices:
            raise ValueError("graph capture shows no conv output slice (no SliceWrite / ShardedToInterleaved)")
        rows = 1
        for d in shape[:-1]:
            rows *= d
        batch = rows // (out_h * out_w)
        slices = [(0, batch, 0, out_h, 0, out_w) + s[6:] if s[0] is None else s for s in slices]
        owner, cores = nd_core_map.owner_conv_slices(shape, out_h, out_w, slices)
        sig = "conv2d " + "; ".join(
            f"n{n0}-{n1} h{h0}-{h1} w{w0}-{w1} {l} {tuple(sh)} on {g} {'ROW' if rm else 'COL'}_MAJOR"
            for n0, n1, h0, h1, w0, w1, l, sh, g, rm in slices
        )
        return {"kind": "owner", "owner": owner, "cores": cores, "signature": sig}

    @staticmethod
    def _matmul_layout(config):
        if config is None:
            raise ValueError("the program-config listener saw no matmul")
        from tt_py_test_utils_common.matmul_window import config_grid

        name = type(config).__name__
        block = (32 * config.per_core_M, 32 * config.per_core_N)
        # Both factories place and size the grid from the bounding box of allowed_worker_cores
        # (matmul_multicore_reuse_mcast_2d_program_factory.cpp: sub_device_start_core)
        origin, (gx, gy) = config_grid(config)
        # fuse_batch folds the batch into M (Mt = B * M), so the block rows run across batches: map the output with
        # its leading dims flattened into rows, not summed
        fold_batch = bool(getattr(config, "fuse_batch", False))
        if name == "MatmulMultiCoreReuseMultiCastProgramConfig":  # 2D: core (x, y) = block (by = y, bx = x)
            return {
                "kind": "block",
                "block": block,
                "linear_x": None,
                "transpose": bool(config.transpose_mcast),
                "origin": origin,
                "fold_batch": fold_batch,
                "grid": (gx, gy),
                "signature": f"{name} {config}",
            }
        if name == "MatmulMultiCoreReuseMultiCast1DProgramConfig":  # 1D: core i = block i, row-major over the grid
            # gather_in0 / hop cores place blocks in ring or shard order, not row-major: refuse rather than guess
            hop = getattr(config, "hop_cores", None)
            if (
                getattr(config, "gather_in0", False)
                or (hop is not None and hop.ranges())
                or getattr(config, "num_global_cb_receivers", 0) > 1
            ):
                raise ValueError(f"{name} with gather_in0, hop cores or global CB receivers has no core map")
            return {
                "kind": "block",
                "block": block,
                "linear_x": gx,
                "transpose": False,
                "origin": origin,
                "fold_batch": fold_batch,
                "grid": (gx, gy),
                "signature": f"{name} {config}",
            }
        raise ValueError(f"matmul program config {name} has no core map")

    def _note_layout(self, iteration, layout):
        if not self.nd_layout_changes or self.nd_layout_changes[-1][1] != layout["signature"]:
            self.nd_layout_changes.append((iteration, layout["signature"]))
            logger.info(f"ND layout from iteration {iteration}: {layout['signature']}")

    def _map_cores(self, ref, out, layout):
        """({(x, y): mismatching elements}, {(x, y): (by, bx) block} or {}, elements no core owns)."""
        if layout["kind"] == "block":
            ox, oy = layout["origin"]
            if layout.get("fold_batch"):
                ref, out = ref.reshape(-1, ref.shape[-1]), out.reshape(-1, out.shape[-1])
            mapped = nd_core_map.core_map(
                ref, out, layout["block"], linear_x=layout["linear_x"], transpose=layout["transpose"]
            )
            cores = {(x + ox, y + oy): n for (x, y), (n, _) in mapped.items()}
            blocks = {(x + ox, y + oy): b for (x, y), (_, b) in mapped.items()}
            return cores, blocks, 0
        if layout["kind"] == "owner":
            cores, unowned = nd_core_map.owner_core_mismatches(ref, out, layout["owner"], layout["cores"])
            return cores, {}, unowned
        return {}, {}, None

    def log_nd_cores(self, device_idx, ref, out, layout=None, reference_layout=None):
        """Log which cores produced the mismatching output elements of one failed check; returns {(x, y): elements}.

        layout is the check run's output -> core map, reference_layout the map of the run that produced the
        reference output. When they differ, a mismatch may come from either run, so both attributions are logged.
        """
        layout = layout or {"kind": "none", "signature": "none", "reason": "no layout"}
        if layout["kind"] == "none":
            logger.info(f"ND device {device_idx}: no core mapping ({layout.get('reason', '')})")
            return {}
        cores, blocks, unowned = self._map_cores(ref, out, layout)
        # Where the errors sit inside each failing core's block (row/column strips, whole tiles, a face lane,
        # scattered; bits), to tell which stage of the core corrupted them. Rectangular 2D blocks only; capped
        # at 8 cores per check; aggregated into the end-of-run report.
        structures = {}
        if layout["kind"] == "block" and not layout["linear_x"]:
            for core, _ in sorted(cores.items(), key=lambda c: -c[1])[:8]:
                by, bx = blocks[core]
                structures[core] = nd_core_map.block_structure(ref, out, by, bx, layout["block"])
        hits = self.nd_core_hits.setdefault(device_idx, {})
        for (x, y), count in sorted(cores.items(), key=lambda c: (c[0][1], c[0][0])):
            hits[(x, y)] = hits.get((x, y), 0) + 1
            if (x, y) in structures:
                self.nd_core_structures.setdefault(device_idx, {}).setdefault((x, y), []).append(structures[(x, y)])
            translated = self.nd_translated(device_idx, x, y)
            logger.info(
                f"ND device {device_idx} logical core ({x},{y})"
                f"{f' translated=({translated[0]},{translated[1]})' if translated else ''}: "
                f"{count} elements differ"
            )
        if unowned:
            logger.info(f"ND device {device_idx}: {unowned} differing elements lie in tiles no core writes")
        logger.info(
            f"ND device {device_idx} [{self.chip_locations().get(device_idx, 'unknown')}] pattern: "
            f"{self.nd_pattern(cores, layout)} | layout: {layout['signature']}"
        )
        if reference_layout is not None and reference_layout["signature"] != layout["signature"]:
            ref_cores, _, _ = self._map_cores(ref, out, reference_layout)
            logger.info(
                f"ND device {device_idx}: the reference output was made with a different layout "
                f"({reference_layout['signature']}); under it the same mismatches map to "
                + (" ".join(f"({x},{y})x{n}" for (x, y), n in sorted(ref_cores.items())) or "no core")
            )
        return cores

    def nd_pattern(self, cores, layout=None):
        """classify() of device-logical cores. For a block layout the edges are those of the op's grid, so the cores
        are shifted to grid-relative coordinates first, and the pattern says so when the grid is not at (0, 0)."""
        layout = layout or {}
        ox, oy = layout.get("origin", (0, 0))
        pattern = nd_core_map.classify({(x - ox, y - oy) for x, y in cores}, layout.get("grid"))
        return pattern if (ox, oy) == (0, 0) else f"{pattern} [grid-relative, grid origin ({ox},{oy})]"

    def nd_translated(self, device_idx, x, y):
        """Translated (NOC) coordinates of device logical core (x, y), or None."""
        key = (device_idx, x, y)
        if key not in self.nd_core_translated:
            try:
                core = self.get_device(device_idx).worker_core_from_logical_core(ttnn.CoreCoord(x, y))
                self.nd_core_translated[key] = (core.x, core.y)
            except Exception:
                self.nd_core_translated[key] = None
        return self.nd_core_translated[key]

    def chip_locations(self):
        if self._chip_locations is None:
            self._chip_locations = chip_locations()
        return self._chip_locations

    # End of run: location of every device in the test, then a short report per device with ND: where it is,
    # which cores failed in how many checks, the overall pattern, each core's error signature across checks, and the
    # layouts the checks ran with.
    def log_nd_core_summary(self, num_checks, nd_checks_per_device):
        locations = self.chip_locations()
        for device_idx in sorted(self.device_ids):
            logger.info(f"Chip map device {device_idx}: {locations.get(device_idx, 'unknown')}")
        failing = {d: n for d, n in nd_checks_per_device.items() if n}
        logger.info(
            f"ND report: {len(failing)} of {len(self.device_ids)} devices non-deterministic ({num_checks} checks)"
        )
        for it, sig in self.nd_layout_changes:
            logger.info(f"ND report: layout from iteration {it}: {sig}")
        for device_idx in sorted(failing):
            hits = self.nd_core_hits.get(device_idx, {})
            pattern = nd_core_map.classify(hits) if hits else "no core mapping for this test"
            by_count = sorted(hits.items(), key=lambda c: (-c[1], c[0][1], c[0][0]))
            per_core = " ".join(f"({x},{y})x{n}" for (x, y), n in by_count)
            where = locations.get(device_idx, "unknown")
            logger.info(
                f"ND summary device {device_idx} [{where}]: {failing[device_idx]}/{num_checks} checks; "
                f"pattern {pattern}; failing checks per core: {per_core}"
            )
            for (x, y), n in by_count[:8]:
                summary = nd_core_map.summarize_structures(self.nd_core_structures.get(device_idx, {}).get((x, y), []))
                translated = self.nd_translated(device_idx, x, y)
                logger.info(
                    f"ND summary device {device_idx} core ({x},{y})"
                    f"{f' translated=({translated[0]},{translated[1]})' if translated else ''}: "
                    f"failed {n}/{num_checks} checks; {nd_core_map.describe_summary(summary)}"
                )

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

        The caller drains the devices first. VCORE is read idle; the step sampler records telemetry during the
        iterations.
        """
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
        """Force each device's VDD to its firmware VCORE + offset_mv (devices drained by the caller); return the
        readback."""
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
                    for device_idx in self.device_ids:  # the input pushes above may still be in flight
                        ttnn.device.synchronize_device(self.get_device(device_idx))
                    self._set_sweep_clock(clock, min(self.vf_sweep_freqs), "reference outputs")
                # One reference run per activation; we then cycle through them and compare output to
                # reference_out[act]. Deallocate only for L1 due to memory constraints.
                # For DRAM we keep all a_t[act] live; we still switch self.activations each time, so
                # determinism checking is unchanged (different inputs, match reference per activation).
                num_nd_outputs = [0] * num_devices
                reference_out = [None for _ in range(num_activation_tensors)]
                for act in range(num_activation_tensors):
                    self.activations = self.convert_activations_to_memory_config(a_t[act])
                    observation = None
                    if getattr(self, "nd_core_capture", None):
                        output, observation = self._run_captured()
                    else:
                        output = self.run_device_operation()
                    try:
                        self.nd_reference_layouts[act] = self._layout_for(output, observation)
                    except Exception as e:
                        logger.warning(f"ND layout of reference run {act} unknown: {e!r}")
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
                    # Drain once, then close the previous step's telemetry before any force changes
                    for device_idx in self.device_ids:
                        ttnn.device.synchronize_device(self.get_device(device_idx))
                    if sweep_steps:
                        sweep_steps[-1]["tel"] = clock.stop_step_sampling()
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
                            clock.release_vdd()
                        vcore = base_vcore
                    step = {
                        "mhz": mhz,
                        "offset": offset,
                        "checks": 0,
                        "nd": [0] * num_devices,
                        "vcore": [vcore[device_idx] for device_idx in self.device_ids],
                        "cores": [{} for _ in range(num_devices)],  # failing checks per core, per device
                    }
                    sweep_steps.append(step)
                    clock.start_step_sampling()

                check_iteration = self.determinism_check_enabled and i % self.determinism_check_interval == 0
                observation = None
                if check_iteration and getattr(self, "nd_core_capture", None):
                    out, observation = self._run_captured()
                else:
                    out = self.run_device_operation()

                if self.sync_every_iteration or check_iteration or i == self.loop_count - 1:
                    for device_idx in self.device_ids:
                        logger.debug(f"Start sync device id: {device_idx}")
                        ttnn.device.synchronize_device(self.get_device(device_idx))
                        logger.debug(f"End sync device id: {device_idx}")

                # Check if the output matches the first run output
                if check_iteration:
                    try:  # this check's output -> core map, from this op call
                        check_layout = self._layout_for(out, observation)
                        self._note_layout(i, check_layout)
                    except Exception as e:
                        logger.warning(f"ND layout lookup failed at iteration {i}: {e!r}")
                        check_layout = {
                            "kind": "none",
                            "signature": "unknown",
                            "reason": f"layout lookup failed: {e!r}",
                        }
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
                            # Diagnostics only: a failure here must not replace the ND verdict below
                            try:
                                cores = self.log_nd_cores(
                                    device_idx,
                                    reference_out[current_act_tensor][output_id],
                                    outputs[output_id],
                                    check_layout,
                                    self.nd_reference_layouts.get(current_act_tensor),
                                )
                                if clock is not None:
                                    for core in cores:
                                        step["cores"][output_id][core] = step["cores"][output_id].get(core, 0) + 1
                            except Exception as e:
                                logger.warning(f"ND core attribution failed on device {device_idx}: {e!r}")
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
            try:
                if sweep_steps:
                    sweep_steps[-1]["tel"] = clock.stop_step_sampling()  # the last iteration synced the devices
                self._log_sweep_summary(sweep_steps, clock.therm_trips())
            except Exception as e:  # diagnostics only: the ND verdict below must still be reached
                logger.error(f"VF sweep summary failed: {e!r}")

        catch_nd_on_fail = True
        if self.determinism_check_enabled:
            # Checks run at i % interval == 0 for i in [0, loop_count)
            try:
                self.log_nd_core_summary(
                    -(-self.loop_count // self.determinism_check_interval),
                    {self.device_ids[i]: n for i, n in enumerate(num_nd_outputs)},
                )
            except Exception as e:
                logger.warning(f"ND core summary failed: {e!r}")
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

    def _log_sweep_summary(self, sweep_steps, therm_trips):
        """One line per (clock, VDD offset) and device: checks, failures with a 95 % interval, the idle VCORE readback,
        and the telemetry during the step's iterations; then the result grids and THERM_TRIP_COUNT."""
        lines = [
            "VF sweep summary. VCORE set = idle readback after the clock / VDD was set (+0 = firmware voltage); "
            "during step = min/median/max over every telemetry update of the step, workload and host-side checks "
            "alike (the minimum VCORE and maximum TDC / temperature are the loaded values):"
        ]
        per_device = len(self.device_ids) <= 4
        for step in sweep_steps:
            if not per_device:
                tels = [(step.get("tel") or {}).get(d) or {} for d in self.device_ids]
                worst = lambda key, i: [t[key][i] for t in tels if key in t]
                vmin, tmax, imax = worst("vcore_mv", 0), worst("temp_c", 2), worst("tdc_a", 2)
                failed = sum(1 for k in step["nd"] if k)
                lines.append(
                    f"  AICLK {step['mhz']:>5} MHz VDD +{step['offset']:>3} mV all {len(self.device_ids)} devices | "
                    f"{failed} failing | VCORE set {min(step['vcore'])}-{max(step['vcore'])} mV | during step: "
                    f"min VCORE {min(vmin) if vmin else '-'} mV, max temp {max(tmax) if tmax else '-'} C, "
                    f"max TDC {max(imax) if imax else '-'} A"
                )
            for o, device_idx in enumerate(self.device_ids):
                if not per_device and not step["nd"][o]:
                    continue
                k, n = step["nd"][o], step["checks"]
                tel = (step.get("tel") or {}).get(device_idx) or {}
                fail = f"ND {k}/{n} {self._ci_text(k, n)} | " if self.determinism_check_enabled else ""
                lines.append(
                    f"  AICLK {step['mhz']:>5} MHz VDD +{step['offset']:>3} mV dev {device_idx} | {fail}"
                    f"VCORE set {step['vcore'][o]} mV | {self._step_telemetry_text(tel)}"
                )
        if self.determinism_check_enabled:
            lines += self._sweep_result_lines(sweep_steps)
        trips = ", ".join(f"dev {d}: {a} -> {b}" for d, (a, b) in sorted(therm_trips.items()))
        rose = any(a is not None and b is not None and b > a for a, b in therm_trips.values())
        lines.append(
            f"THERM_TRIP_COUNT start -> end: {trips}{' (ROSE: a step ended in a thermal reset)' if rose else ''}"
        )
        logger.info("\n".join(lines))

    @staticmethod
    def _wilson(k, n, z=1.96):
        """95 % Wilson score interval for k failures in n checks."""
        if n == 0:
            return 0.0, 1.0
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
        return max(0.0, c - h), min(1.0, c + h)

    def _ci_text(self, k, n):
        lo, hi = self._wilson(k, n)
        return f"[{lo:.0%}-{hi:.0%}]"

    @staticmethod
    def _step_telemetry_text(tel):
        """Telemetry of one step on one device: VCORE, temperature, TDC, TDP over the step, how many updates held the
        forced clock, and whether the firmware would have throttled (forced clock above what its arbiters allow)."""
        if not tel.get("samples"):
            return "no telemetry samples"
        fmt = lambda t, unit, p=0: f"{t[0]:.{p}f}/{t[1]:.{p}f}/{t[2]:.{p}f}{unit}"
        parts = [
            f"{name} {fmt(tel[key], unit, p)}"
            for key, name, unit, p in (
                ("vcore_mv", "VCORE", " mV", 0),
                ("temp_c", "temp", " C", 1),
                ("tdc_a", "TDC", " A", 0),
                ("tdp_w", "TDP", " W", 0),
            )
            if key in tel
        ]
        text = f"during step {', '.join(parts)} | {tel['samples']} updates"
        if "clock_held" in tel:
            text += f", clock held in {tel['clock_held']}"
        if tel.get("fw_would_cap"):
            text += " | FW would throttle (" + ", ".join(f"{a} x{c}" for a, c in tel["fw_would_cap"].items()) + ")"
        return text

    def _sweep_result_lines(self, sweep_steps):
        """Result grid (clock x VDD offset) over all devices, then per failing device with its failing cores.

        A cell is k/n failed checks with the 95 % Wilson interval of the fail rate. A point PASSES when every check at
        it was bit-exact; with n checks that bounds the fail rate only to the interval's upper edge, so two points
        differ only when their intervals do not overlap.
        """
        clocks = list(dict.fromkeys(step["mhz"] for step in sweep_steps))
        offsets = list(dict.fromkeys(step["offset"] for step in sweep_steps))
        at = {(step["mhz"], step["offset"]): step for step in sweep_steps}

        def cell(step, o=None):
            n = step["checks"]
            if o is None:
                k = max(step["nd"], default=0)  # worst device
                failed = [f"d{self.device_ids[i]}" for i, c in enumerate(step["nd"]) if c]
                text = f"{'FAIL' if k else 'PASS'} {k}/{n} {self._ci_text(k, n)}"
                return text + (f" {','.join(failed)}" if failed else "")
            k = step["nd"][o]
            return f"{'FAIL' if k else 'PASS'} {k}/{n} {self._ci_text(k, n)} @{step['vcore'][o]}mV"

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
        lines = ["", *grid("VF sweep result, worst device per point (k/n failed checks [95 % interval]):")]
        if not failing:
            lines.append(f"  every clock and VDD combination PASSED on all {len(self.device_ids)} devices")
        for o in failing:
            lines += ["", *grid(f"Device {self.device_ids[o]} (failed checks/checks [95 %] @ VCORE set):", o)]
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
                cores = {}
                for off in ran:
                    for core, c in at[(mhz, off)]["cores"][o].items():
                        cores[core] = cores.get(core, 0) + c
                where = (
                    "; failing cores "
                    + " ".join(f"({x},{y})x{c}" for (x, y), c in sorted(cores.items(), key=lambda t: -t[1]))
                    if cores
                    else ""
                )
                lines.append(f"    {mhz} MHz: {verdict}{where}")
        return lines


def get_mesh_grid_size(mesh_device):
    return mesh_device.compute_with_storage_grid_size()
