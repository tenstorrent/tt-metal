# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attach a kernel_lib mcast family to a Metal 2.0 ProgramSpec.

The descriptor path makes the caller splice arg blocks and hand-thread the CT/RT base offsets that
the kernel's ``McastArgs<CT_BASE, RT_BASE>`` chain expects, keeping two orderings in agreement by
hand. Here the family owns that bookkeeping: ``attach`` writes its semaphores, bindings, named
compile-time args and per-node vararg values into the spec and run args, and ships its own vararg
base as the ``<prefix>_rt_base`` named arg so the kernel's ``MCAST_ARGS(prefix)`` chains nothing.

    # 1D: independent per-row / per-column mcasts over the grid.
    mc_a = McastFamily(device, grid, "a", shape=ttnn.Mcast1DShape.PerRow, sender_index=0)
    mc_b = McastFamily(device, grid, "b", shape=ttnn.Mcast1DShape.PerColumn, sender_index=0)
    mc_a.attach(spec, run_args, kernels=["reader"])
    mc_b.attach(spec, run_args, kernels=["reader"])

    # 2D: ONE mcast over a receiver rectangle, from a single sender core that may sit inside the
    # rectangle or outside it.
    mc_r = McastFamily(device, rect, "r", sender=ttnn.CoreCoord(0, 0))
    mc_r.attach(spec, run_args, kernels=["reader"])

Both topologies present the same face to the kernel -- four named CT words and a 4-word (or
rotating 4 + 2*rounds) vararg block -- so ``MCAST_ARGS(prefix)`` is unchanged and the kernel cannot
tell which topology it was handed. The coord math (logical->virtual, per-NoC rect corner order,
sender placement, rotating rounds) is ttnn.Mcast1D's / ttnn.Mcast2D's, unchanged.
"""

import ttnn


class McastFamily:
    """One mcast family, addressed by name instead of by CT/RT offset.

    ONE FAMILY == ONE PREFIX == ONE KERNEL-SIDE DECODER. The kernel decodes a family with
    ``MCAST_ARGS(prefix)``, which pastes the prefix at PREPROCESSOR time, so the number of
    INDEPENDENT mcast groups a kernel can address is fixed in kernel source, not by the host:
    ``Mcast2D`` expresses exactly ONE group (over an arbitrary rectangle), and ``Mcast1D``
    expresses one group PER GRID LINE (each forced to be a full line). Addressing N independent
    rectangles requires N prefixes baked into the kernel -- a design calling for one ``Mcast2D``
    per group over rectangles tiling the grid is not expressible with a single family.

    Topology is chosen by which sender argument you pass, because a 1D family and a 2D mcast do not
    name their sender the same way: 1D has a sender *index* per line (plus a placement rule that
    walks it across lines), 2D has one sender *core*. So ``sender=`` selects
    ``ttnn.Mcast2D``; ``shape=``/``sender_index=``/``sender_placement=`` (or nothing at all) selects
    ``ttnn.Mcast1D``. Mixing the two sets is an error rather than a precedence rule.

    Both semaphores are always created, even with ``handshake=False``: the ``flags`` word decides
    whether the ack path runs, and always emitting the pair keeps ``sem::<prefix>_consumer_ready``
    defined so a single MCAST_ARGS macro covers both cases. Costs one L1 word per core.
    """

    def __init__(
        self,
        device,
        grid: "ttnn.CoreRangeSet",
        prefix: str,
        *,
        shape=None,
        sender_index: int = None,
        config=None,
        sender_placement=None,
        sender=None,
        num_active: int = None,
        sender_grid=None,
    ):
        if not prefix.isidentifier():
            raise ValueError(f"mcast prefix {prefix!r} must be a valid C++ identifier")

        config = ttnn.McastConfig() if config is None else config

        # Semaphore ids come from the spec's SemaphoreBindings, so the helper must not assign any.
        # Adopting placeholders keeps owned_semaphores() empty and next_base_sem_id() unused.
        config.sem_ids = [0, 1]

        self.prefix = prefix
        self.grid = grid
        self._config = config
        self._mcast, self._nodes = self._build_mcast(
            device,
            grid,
            shape=shape,
            sender_index=sender_index,
            config=config,
            sender_placement=sender_placement,
            sender=sender,
            num_active=num_active,
            sender_grid=sender_grid,
        )
        self._added_semaphores = False

    def _build_mcast(
        self, device, grid, *, shape, sender_index, config, sender_placement, sender, num_active, sender_grid
    ):
        """Return (topology helper, the node set its semaphores and varargs must cover)."""
        # The node set is the receiver grid plus any core that only ever sends: a 2D sender outside
        # the rect, or a rotating sender_grid core outside it. Those cores run the same kernel and
        # wait on the same consumer_ready semaphore, so a semaphore placed on `grid` alone would
        # leave them without one. Mcast2D computes the same union for its own owned_semaphores(),
        # which we cannot use here because we adopt sem_ids and it therefore returns nothing.
        nodes = grid
        if sender_grid is not None:
            nodes = nodes.merge(sender_grid)

        if sender is not None:
            one_d_only = (
                ("shape", shape),
                ("sender_index", sender_index),
                ("sender_placement", sender_placement),
            )
            rejected = [name for name, value in one_d_only if value is not None]
            if rejected:
                raise ValueError(
                    f"McastFamily: {rejected} are 1D-only arguments but sender= selects the 2D "
                    "topology (one mcast over a receiver rectangle from one sender core). Drop them, "
                    "or drop sender= to get a 1D family."
                )
            sender = sender if isinstance(sender, ttnn.CoreCoord) else ttnn.CoreCoord(*sender)
            mcast = ttnn.Mcast2D(
                device,
                grid,
                sender,
                config,
                0 if num_active is None else num_active,
                sender_grid,
            )
            return mcast, nodes.merge(ttnn.CoreRangeSet([ttnn.CoreRange(sender, sender)]))

        if num_active is not None:
            raise ValueError(
                "McastFamily: num_active= is a 2D-only argument (the 2D ack wait-count); a 1D family "
                "derives its ack count per line. Pass sender= to select the 2D topology."
            )
        mcast = ttnn.Mcast1D(
            device,
            grid,
            ttnn.Mcast1DShape.PerRow if shape is None else shape,
            0 if sender_index is None else sender_index,
            ttnn.Mcast1DSenderPlacement.Uniform if sender_placement is None else sender_placement,
            config,
            sender_grid,
        )
        return mcast, nodes

    @property
    def data_ready_name(self) -> str:
        return f"{self.prefix}_data_ready"

    @property
    def consumer_ready_name(self) -> str:
        return f"{self.prefix}_consumer_ready"

    @property
    def nodes(self) -> "ttnn.CoreRangeSet":
        """Every core this family touches: the receiver grid plus any send-only core."""
        return self._nodes

    def num_senders(self) -> int:
        return self._mcast.num_senders()

    def is_sender(self, core) -> bool:
        return self._mcast.is_sender(core)

    def num_receivers(self, core) -> int:
        return self._mcast.num_receivers(core)

    def num_runtime_varargs(self) -> int:
        return 4 if not self._config.rotating_sender else 4 + 2 * self._mcast.num_senders()

    def semaphores(self) -> list:
        return [
            ttnn.SemaphoreSpec(unique_id=self.data_ready_name, target_nodes=self._nodes),
            ttnn.SemaphoreSpec(unique_id=self.consumer_ready_name, target_nodes=self._nodes),
        ]

    def semaphore_bindings(self) -> list:
        return [
            ttnn.SemaphoreBinding(self.data_ready_name, self.data_ready_name),
            ttnn.SemaphoreBinding(self.consumer_ready_name, self.consumer_ready_name),
        ]

    def compile_time_args(self, rt_base: int, *, pre_handshake: bool = None) -> dict:
        """The four named CT words. Semaphore ids are absent: they arrive via sem:: bindings."""
        # Reuse the topology helper's flags computation by reading the descriptor-path block:
        # [active, data_ready, consumer_ready, num_active, flags, rotating_span]. Only the leading
        # five are read here, and dropping the trailing span is deliberate rather than lossy: on the
        # spec path the span is a TEMPLATE argument the kernel passes itself via
        # MCAST_ARGS_ROTATING(prefix, span), because McastArgsSpec takes it as a non-type template
        # param. The descriptor path has no such channel, which is the only reason the helpers ship
        # it as a sixth CT word. Slice rather than unpack-all so a future seventh word cannot break
        # us. Both Mcast1D and Mcast2D emit this same six-word block.
        block = (
            self._mcast.compile_time_args() if pre_handshake is None else self._mcast.compile_time_args(pre_handshake)
        )
        active, _, _, num_active, flags = block[:5]
        return {
            f"{self.prefix}_active": active,
            f"{self.prefix}_num_active": num_active,
            f"{self.prefix}_flags": flags,
            f"{self.prefix}_rt_base": rt_base,
        }

    def runtime_varargs(self, core) -> list:
        return list(self._mcast.runtime_args(core))

    def attach(self, spec, run_args, kernels, *, cores=None, pre_handshake=None) -> None:
        """Write this family into `spec` and `run_args` for each kernel in `kernels`.

        Appends the semaphores once, then per kernel: the semaphore bindings, the named CT args, and
        the vararg count, plus the per-node vararg values on the matching KernelRunArgs. The vararg
        base is taken from whatever varargs that kernel already declares, so families attached in
        sequence pack one after another.
        """
        kernel_names = [kernels] if isinstance(kernels, str) else list(kernels)
        cores = self._grid_cores() if cores is None else list(cores)

        existing = {str(s.unique_id) for s in spec.semaphores}
        if not self._added_semaphores and existing & {self.data_ready_name, self.consumer_ready_name}:
            raise ValueError(
                f"attach: the spec already declares a semaphore named {self.data_ready_name!r} or "
                f"{self.consumer_ready_name!r}. Two families sharing a prefix would silently share "
                "semaphores; give this family a different prefix."
            )
        spec.semaphores = list(spec.semaphores) + [s for s in self.semaphores() if str(s.unique_id) not in existing]
        self._added_semaphores = True

        # Read the containers out, mutate, write back: correct whether the bindings hand back
        # aliases or copies.
        kernel_list = list(spec.kernels)
        run_list = list(run_args.kernel_run_args)
        by_name = {str(k.unique_id): k for k in kernel_list}
        run_by_name = {str(k.kernel): k for k in run_list}

        for name in kernel_names:
            kernel = by_name.get(name)
            if kernel is None:
                raise KeyError(f"attach: kernel {name!r} is not in the ProgramSpec")
            kernel_run_args = run_by_name.get(name)
            if kernel_run_args is None:
                raise KeyError(f"attach: kernel {name!r} has no KernelRunArgs in the ProgramRunArgs")

            # Per-kernel: compile_time_args live on the KernelSpec, so each kernel reads its own
            # <prefix>_rt_base and two kernels sharing a family may sit at different bases.
            rt_base = kernel.advanced_options.num_runtime_varargs
            new_ct = self.compile_time_args(rt_base, pre_handshake=pre_handshake)
            existing_ct = dict(kernel.compile_time_args)
            clashes = sorted(set(existing_ct) & set(new_ct))
            if clashes:
                raise ValueError(
                    f"attach: kernel {name!r} already declares compile-time args {clashes}; "
                    f"overwriting them would silently repoint the {self.prefix!r} family. Use a "
                    "different prefix."
                )

            kernel.semaphore_bindings = list(kernel.semaphore_bindings) + self.semaphore_bindings()
            kernel.compile_time_args = {**existing_ct, **new_ct}

            advanced = kernel.advanced_options
            advanced.num_runtime_varargs = rt_base + self.num_runtime_varargs()
            kernel.advanced_options = advanced

            varargs = dict(kernel_run_args.advanced_options.runtime_varargs)
            for core in cores:
                varargs[core] = list(varargs.get(core, [])) + self.runtime_varargs(core)
            run_advanced = kernel_run_args.advanced_options
            run_advanced.runtime_varargs = varargs
            kernel_run_args.advanced_options = run_advanced

        spec.kernels = kernel_list
        run_args.kernel_run_args = run_list

    def _grid_cores(self) -> list:
        # Enumerate the node set itself, not its bounding box: a 2D family whose sender sits outside
        # the rect is not rectangular, and a box would hand varargs to cores that never run.
        return list(ttnn.corerange_to_cores(self._nodes, None, True))
