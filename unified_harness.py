# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host harness for running unified kernels through ``ttnn.generic_op``.

A unified kernel is ONE source file that describes a whole Tensix pipeline. The
trick that makes it work on today's API is that ``ttnn.generic_op`` takes a
*list* of ``KernelDescriptor``s, and nothing requires them to point at different
files. So we emit three descriptors from one source:

    reader   DataMovementConfigDescriptor(RISCV_1)   -> COMPILE_FOR_NCRISC
    writer   DataMovementConfigDescriptor(RISCV_0)   -> COMPILE_FOR_BRISC
    compute  ComputeConfigDescriptor()               -> UCK_CHLKC_{UNPACK,MATH,PACK}

No per-thread defines are passed from here. Metal already emits a thread
identity define for every kernel build, and ``unified_metal.hpp`` derives the
projection from it -- so the host side stays ignorant of the mechanism.

All three descriptors get *identical* compile-time and runtime args, because
they are the same source reading the same arg indices. That uniform arg schema
is a property of the model, not a limitation of the harness.
"""

import os

import ttnn

TT_METAL_HOME = os.environ.get("TT_METAL_HOME", os.path.dirname(os.path.abspath(__file__)))

# The unified headers live at the repo root; the kernel's quoted includes need
# that on the JIT include path.
UNIFIED_INCLUDE_PATHS = [TT_METAL_HOME]

TILE_HW = 32 * 32
DTYPE_TILE_BYTES = {ttnn.bfloat16: TILE_HW * 2, ttnn.float32: TILE_HW * 4}


def make_runtime_args(cores, values):
    """A RuntimeArgs with the same `values` on every core in `cores`."""
    args = ttnn.RuntimeArgs()
    for core in cores:
        args[core.x][core.y] = list(values)
    return args


def make_cb(cb_index, core_ranges, dtype=ttnn.bfloat16, num_pages=2):
    """A double-buffered CB by default: `num_pages` pages of one tile each."""
    page_size = DTYPE_TILE_BYTES[dtype]
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_index, data_format=dtype, page_size=page_size)],
    )


def unified_program(
    *,
    kernel_source,
    core_ranges,
    cores,
    cbs,
    compile_time_args,
    runtime_args,
    reader_processor=ttnn.DataMovementProcessor.RISCV_1,
    writer_processor=ttnn.DataMovementProcessor.RISCV_0,
    semaphores=None,
    defines=None,
):
    """Build a ProgramDescriptor that compiles ONE source for all five threads.

    Args:
        kernel_source: path to the unified kernel, relative to TT_METAL_HOME.
        core_ranges: ttnn.CoreRangeSet the program runs on.
        cores: the individual CoreCoords, for per-core runtime args.
        cbs: list of ttnn.CBDescriptor.
        compile_time_args: list of ints, shared by all three descriptors.
        runtime_args: list of ints, shared by all three descriptors.
        reader_processor / writer_processor: which RISC-V runs which DM role.
            Metal's convention is RISCV_1 for readers, RISCV_0 for writers.
        defines: optional extra (name, value) pairs, applied to all three.
    """
    shared = dict(
        kernel_source=kernel_source,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_ranges,
        compile_time_args=list(compile_time_args),
        defines=list(defines or []),
        compiler_include_paths=UNIFIED_INCLUDE_PATHS,
    )

    kernels = [
        # reader. `noc` must be set explicitly: DataMovementConfig defaults it to
        # RISCV_0_default (= NOC 0) regardless of processor, so leaving it out puts
        # BOTH data-movement kernels on NOC 0. That costs the second NOC's
        # bandwidth, and it makes the end-of-kernel NOC-idle asserts in brisck.cc
        # cross-talk: NOC_INDEX is emitted from this field (kernel.cpp:252), so
        # BRISC would check NOC 0's read counters against its own issued count and
        # trip on reads NCRISC issued.
        ttnn.KernelDescriptor(
            **shared,
            runtime_args=make_runtime_args(cores, runtime_args),
            config=ttnn.DataMovementConfigDescriptor(processor=reader_processor, noc=ttnn.NOC.RISCV_1_default),
        ),
        # writer
        ttnn.KernelDescriptor(
            **shared,
            runtime_args=make_runtime_args(cores, runtime_args),
            config=ttnn.DataMovementConfigDescriptor(processor=writer_processor, noc=ttnn.NOC.RISCV_0_default),
        ),
        # compute (fans out to TRISC 0/1/2)
        ttnn.KernelDescriptor(
            **shared,
            runtime_args=make_runtime_args(cores, runtime_args),
            config=ttnn.ComputeConfigDescriptor(),
        ),
    ]

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=list(semaphores or []), cbs=list(cbs))


def single_core():
    """The (0,0) core, as (core_ranges, cores)."""
    c = ttnn.CoreCoord(0, 0)
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c)]), [c]
