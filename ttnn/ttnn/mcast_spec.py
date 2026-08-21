# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attach a kernel_lib mcast family to a Metal 2.0 ProgramSpec.

The descriptor path makes the caller splice arg blocks and hand-thread the CT/RT base offsets that
the kernel's ``McastArgs<CT_BASE, RT_BASE>`` chain expects, keeping two orderings in agreement by
hand. Here the family owns that bookkeeping: ``attach`` writes its semaphores, bindings, named
compile-time args and per-node vararg values into the spec and run args, and ships its own vararg
base as the ``<prefix>_rt_base`` named arg so the kernel's ``MCAST_ARGS(prefix)`` chains nothing.

    mc_a = McastFamily(device, grid, "a", shape=ttnn.Mcast1DShape.PerRow, sender_index=0)
    mc_b = McastFamily(device, grid, "b", shape=ttnn.Mcast1DShape.PerColumn, sender_index=0)
    mc_a.attach(spec, run_args, kernels=["reader"])
    mc_b.attach(spec, run_args, kernels=["reader"])

The coord math (logical->virtual, per-NoC rect corner order, sender placement, rotating rounds) is
ttnn.Mcast1D's, unchanged.
"""

import ttnn


class McastFamily:
    """One mcast family, addressed by name instead of by CT/RT offset.

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
        sender_index: int = 0,
        config=None,
        sender_placement=None,
    ):
        if not prefix.isidentifier():
            raise ValueError(f"mcast prefix {prefix!r} must be a valid C++ identifier")

        shape = ttnn.Mcast1DShape.PerRow if shape is None else shape
        config = ttnn.McastConfig() if config is None else config
        placement = ttnn.Mcast1DSenderPlacement.Uniform if sender_placement is None else sender_placement

        # Semaphore ids come from the spec's SemaphoreBindings, so the helper must not assign any.
        # Adopting placeholders keeps owned_semaphores() empty and next_base_sem_id() unused.
        config.sem_ids = [0, 1]

        self.prefix = prefix
        self.grid = grid
        self._config = config
        self._mcast = ttnn.Mcast1D(device, grid, shape, sender_index, placement, config)
        self._added_semaphores = False

    @property
    def data_ready_name(self) -> str:
        return f"{self.prefix}_data_ready"

    @property
    def consumer_ready_name(self) -> str:
        return f"{self.prefix}_consumer_ready"

    def num_senders(self) -> int:
        return self._mcast.num_senders()

    def is_sender(self, core) -> bool:
        return self._mcast.is_sender(core)

    def num_runtime_varargs(self) -> int:
        return 4 if not self._config.rotating_sender else 4 + 2 * self._mcast.num_senders()

    def semaphores(self) -> list:
        return [
            ttnn.SemaphoreSpec(unique_id=self.data_ready_name, target_nodes=self.grid),
            ttnn.SemaphoreSpec(unique_id=self.consumer_ready_name, target_nodes=self.grid),
        ]

    def semaphore_bindings(self) -> list:
        return [
            ttnn.SemaphoreBinding(self.data_ready_name, self.data_ready_name),
            ttnn.SemaphoreBinding(self.consumer_ready_name, self.consumer_ready_name),
        ]

    def compile_time_args(self, rt_base: int, *, pre_handshake: bool = None) -> dict:
        """The four named CT words. Semaphore ids are absent: they arrive via sem:: bindings."""
        # Reuse Mcast1D's flags computation by reading the descriptor-path block: [active,
        # data_ready, consumer_ready, num_active, flags, rotating_span]. Only the leading five are
        # read here, and dropping the trailing span is deliberate rather than lossy: on the spec
        # path the span is a TEMPLATE argument the kernel passes itself via
        # MCAST_ARGS_ROTATING(prefix, span), because McastArgsSpec takes it as a non-type template
        # param. The descriptor path has no such channel, which is the only reason Mcast1D ships it
        # as a sixth CT word. Slice rather than unpack-all so a future seventh word cannot break us.
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
        bbox = self.grid.bounding_box()
        return [ttnn.CoreCoord(x, y) for y in range(bbox.end.y + 1) for x in range(bbox.end.x + 1)]
