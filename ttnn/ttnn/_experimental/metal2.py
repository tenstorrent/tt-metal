# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program specs for ``ttnn.generic_op``.

Python view of the experimental Metal 2.0 host API (``tt::tt_metal::experimental::ProgramSpec``
and what it is assembled from). The host API is experimental and changes shape; this module
follows it, which is why it lives under ``ttnn.experimental`` rather than in the public
``ttnn`` namespace.

Typical use::

    from ttnn.experimental import metal2 as m2

    reader = m2.KernelSpec(
        unique_id="reader",
        source="/path/to/reader.cpp",
        hw_config=m2.create_reader_dm_config(),
        dfb_bindings=[m2.producer_of("in_a", "in_a")],
        tensor_bindings=[m2.TensorBinding("a", "a")],
        runtime_arg_schema=m2.RuntimeArgSchema(runtime_arg_names=["num_tiles", "start_id"]),
    )
    spec = m2.ProgramSpec(name="my_op", kernels=[reader, ...], ...)
    run_args = m2.ProgramRunArgs(kernel_run_args=[m2.KernelRunArgs(kernel="reader", runtime_arg_values={...})])

Every property returns a copy (``spec.kernels``, ``kernel.compile_time_args``,
``kernel.hw_config``, ...): mutating the returned list, dict or object changes nothing, so assign
a new value to change a field. ``copy.deepcopy`` works, so a spec can serve as a template.
"""

import ttnn

_program_spec = ttnn._ttnn.program_spec

# The kernel optimisation level is registered with the fabric mux bindings; re-exported here so a
# Metal 2.0 user does not have to import it from a fabric module.
KernelBuildOptLevel = ttnn._ttnn.operations.experimental.fabric_mux.KernelBuildOptLevel

# enums
UnpackMode = _program_spec.UnpackMode
Precision = _program_spec.Precision
DFBEndpointType = _program_spec.DFBEndpointType
DFBAccessPattern = _program_spec.DFBAccessPattern
DedicatedNocSharing = _program_spec.DedicatedNocSharing

# tensor parameters
TensorSpecRelaxations = _program_spec.TensorSpecRelaxations
TensorParameter = _program_spec.TensorParameter

# dataflow buffers, semaphores, scratchpads
DFBAdvancedOptions = _program_spec.DFBAdvancedOptions
DataflowBufferSpec = _program_spec.DataflowBufferSpec
SemaphoreAdvancedOptions = _program_spec.SemaphoreAdvancedOptions
SemaphoreSpec = _program_spec.SemaphoreSpec
ScratchpadSpec = _program_spec.ScratchpadSpec

# hardware configs
DataMovement1XXConfig = _program_spec.DataMovement1XXConfig
DataMovement2XXConfig = _program_spec.DataMovement2XXConfig
DataMovementHardwareConfig = _program_spec.DataMovementHardwareConfig
Compute1XXConfig = _program_spec.Compute1XXConfig
Compute2XXConfig = _program_spec.Compute2XXConfig
ComputeHardwareConfig = _program_spec.ComputeHardwareConfig
create_reader_dm_config = _program_spec.create_reader_dm_config
create_writer_dm_config = _program_spec.create_writer_dm_config

# kernels
SourceCode = _program_spec.SourceCode
CompilerOptions = _program_spec.CompilerOptions
DFBBinding = _program_spec.DFBBinding
producer_of = _program_spec.producer_of
consumer_of = _program_spec.consumer_of
strided_consumer_of = _program_spec.strided_consumer_of
all_consumer_of = _program_spec.all_consumer_of
blocked_producer_of = _program_spec.blocked_producer_of
blocked_consumer_of = _program_spec.blocked_consumer_of
SemaphoreBinding = _program_spec.SemaphoreBinding
ScratchpadBinding = _program_spec.ScratchpadBinding
TensorBinding = _program_spec.TensorBinding
RuntimeArgSchema = _program_spec.RuntimeArgSchema
TensorBindingSequence = _program_spec.TensorBindingSequence
PrefetcherPipeBinding = _program_spec.PrefetcherPipeBinding
KernelAdvancedOptions = _program_spec.KernelAdvancedOptions
KernelSpec = _program_spec.KernelSpec

# program spec
WorkUnitSpec = _program_spec.WorkUnitSpec
PrefetcherPipeParameter = _program_spec.PrefetcherPipeParameter
ProgramAdvancedOptions = _program_spec.ProgramAdvancedOptions
ProgramSpec = _program_spec.ProgramSpec

# run args
AdvancedKernelRunArgs = _program_spec.AdvancedKernelRunArgs
KernelRunArgs = _program_spec.KernelRunArgs
DFBRunOverrides = _program_spec.DFBRunOverrides
ProgramRunArgs = _program_spec.ProgramRunArgs

__all__ = [
    "KernelBuildOptLevel",
    "UnpackMode",
    "Precision",
    "DFBEndpointType",
    "DFBAccessPattern",
    "DedicatedNocSharing",
    "TensorSpecRelaxations",
    "TensorParameter",
    "DFBAdvancedOptions",
    "DataflowBufferSpec",
    "SemaphoreAdvancedOptions",
    "SemaphoreSpec",
    "ScratchpadSpec",
    "DataMovement1XXConfig",
    "DataMovement2XXConfig",
    "DataMovementHardwareConfig",
    "Compute1XXConfig",
    "Compute2XXConfig",
    "ComputeHardwareConfig",
    "create_reader_dm_config",
    "create_writer_dm_config",
    "SourceCode",
    "CompilerOptions",
    "DFBBinding",
    "producer_of",
    "consumer_of",
    "strided_consumer_of",
    "all_consumer_of",
    "blocked_producer_of",
    "blocked_consumer_of",
    "SemaphoreBinding",
    "ScratchpadBinding",
    "TensorBinding",
    "RuntimeArgSchema",
    "TensorBindingSequence",
    "PrefetcherPipeBinding",
    "KernelAdvancedOptions",
    "KernelSpec",
    "WorkUnitSpec",
    "PrefetcherPipeParameter",
    "ProgramAdvancedOptions",
    "ProgramSpec",
    "AdvancedKernelRunArgs",
    "KernelRunArgs",
    "DFBRunOverrides",
    "ProgramRunArgs",
]
