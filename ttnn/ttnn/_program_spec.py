# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The Metal 2.0 ProgramSpec surface of ttnn.experimental, with the names of upstream #63251.

ttnn.experimental.generic_op(io_tensors, spec, run_args, custom_program_hash=None) runs a ProgramSpec. The spec types
are the classes bound in ttnn._ttnn.operations.generic, which also stay at the top level of ttnn under their old
names (ttnn.ProgramSpec, ...), and still take their old constructor keywords with a DeprecationWarning.
"""

import ttnn

_generic = ttnn._ttnn.operations.generic

# Classes and functions that ttnn.experimental exposes under the same names as the C++ bindings.
_SPEC_NAMES = (
    "ComputeHardwareConfig",
    "DataMovementHardwareConfig",
    "DataflowBufferSpec",
    "DFBAdvancedOptions",
    "DFBAccessPattern",
    "DFBBinding",
    "DFBEndpointType",
    "DedicatedNocSharing",
    "KernelRunArgs",
    "KernelSpec",
    "Precision",
    "ProgramSpec",
    "ScratchpadBinding",
    "ScratchpadSpec",
    "SemaphoreBinding",
    "SemaphoreSpec",
    "TensorBinding",
    "TensorParameter",
    "UnpackMode",
    "WorkUnitSpec",
    "all_consumer_of",
    "blocked_consumer_of",
    "blocked_producer_of",
    "consumer_of",
    "create_reader_datamovement_config",
    "create_writer_datamovement_config",
    "producer_of",
    "strided_consumer_of",
)


class ProgramRunArgs:
    """The values of one program run: each kernel's runtime arguments and the tensor for each tensor parameter.

    Args:
        kernel_run_args (List[ProgramRunArgs.KernelRunArgs]): Runtime arg values, one entry per kernel that has
            runtime args.
        tensor_args (Dict[str, ttnn.Tensor]): The tensor bound to each TensorParameter. Every tensor must also be in
            the io_tensors given to generic_op.
        dfb_run_overrides: Not supported yet; must be empty.
    """

    KernelRunArgs = _generic.KernelRunArgs

    def __init__(self, *, kernel_run_args=(), tensor_args=None, dfb_run_overrides=()):
        self.kernel_run_args = list(kernel_run_args)
        self.tensor_args = dict(tensor_args or {})
        self.dfb_run_overrides = list(dfb_run_overrides)


def generic_op(io_tensors, spec, run_args, custom_program_hash=None):
    """Runs a Metal 2.0 ProgramSpec with its ProgramRunArgs on every device of the io tensors' mesh.

    Args:
        io_tensors (List[ttnn.Tensor]): Every tensor the program binds, with the preallocated output last.
        spec (ProgramSpec): The program.
        run_args (ProgramRunArgs): This run's argument values.
        custom_program_hash (int, optional): Program-cache key to use instead of the ProgramSpec's hash.

    Returns:
        ttnn.Tensor: The last element of io_tensors.
    """
    if run_args.dfb_run_overrides:
        raise NotImplementedError("ttnn.experimental.generic_op: dfb_run_overrides are not supported yet")
    legacy_run_args = _generic.ProgramSpecRunArgs(
        kernel_run_args=run_args.kernel_run_args,
        tensor_args=run_args.tensor_args,
        custom_program_hash=custom_program_hash,
    )
    return ttnn.generic_op(io_tensors, spec, legacy_run_args)


def install(experimental):
    """Adds the ProgramSpec surface to the ttnn.experimental module."""
    for name in _SPEC_NAMES:
        setattr(experimental, name, getattr(_generic, name))
    experimental.ProgramRunArgs = ProgramRunArgs
    experimental.generic_op = generic_op
