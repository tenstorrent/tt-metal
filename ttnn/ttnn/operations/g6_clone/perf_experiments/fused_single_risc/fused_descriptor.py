# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf tournament round 1, idea `fused_single_risc`: one copy kernel per core on one RISC.

Drop-in replacements for g6_clone_program_descriptor.create_program_descriptor / patch_addresses
that build the fused kernel kernels/g6_clone_copy.cpp from the op's unchanged plan (compute_plan):
same split, same staging CB (now only the slots' L1), same route knobs. `install(variant)`
monkeypatches them into the op's modules (the descriptor module and the entry module, which
imported both names); `install(None)` leaves the op as it is.

Variants (FUSED_VARIANTS): which RISC runs the kernel and which NoC each direction issues on.
Different read / write NoCs build the kernel in DM_DYNAMIC_NOC mode, as Refinement 3's noc_split
does; one NoC for both directions builds it in that NoC's dedicated mode. A bank-affine NoC split
(BANK_AFFINE_NOC_SPLIT != "none") always builds dynamic.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
FUSED_KERNEL = "g6_clone_copy.cpp"

# variant -> (processor, read_noc, write_noc, noc_mode, kernel_noc); noc_mode "dedicated"
# (read_noc == write_noc), "dynamic" (DM_DYNAMIC_NOC) or "own_other" (dedicated mode on kernel_noc,
# the kernel taking the other NoC's counters over at its start; see the kernel's head comment).
# kernel_noc is the config's NoC (noc_index): the wrapper syncs its counters before the go signal.
_R0 = ttnn.DataMovementProcessor.RISCV_0
_R1 = ttnn.DataMovementProcessor.RISCV_1
FUSED_VARIANTS = {
    # The reader/writer pair's traffic shape (reads NoC0, writes NoC1), on one RISC.
    "ncrisc_dyn": (_R1, 0, 1, "dynamic", 0),
    "brisc_dyn": (_R0, 0, 1, "dynamic", 1),
    "ncrisc_dual": (_R1, 0, 1, "own_other", 0),
    "brisc_dual": (_R0, 0, 1, "own_other", 0),
    "ncrisc_dualw": (_R1, 0, 1, "own_other", 1),
    "brisc_dualw": (_R0, 0, 1, "own_other", 1),
    # Both directions on the kernel's own (dedicated) NoC.
    "ncrisc_noc0": (_R1, 0, 0, "dedicated", 0),
    "brisc_noc1": (_R0, 1, 1, "dedicated", 1),
    "brisc_noc0": (_R0, 0, 0, "dedicated", 0),
    "ncrisc_noc1": (_R1, 1, 1, "dedicated", 1),
}

FUSED_KERNEL_IDX = 0

_NOCS = {0: ttnn.NOC.NOC_0, 1: ttnn.NOC.NOC_1}


def _descriptor_module():
    return importlib.import_module("ttnn.operations.g6_clone.g6_clone_program_descriptor")


def make_create_program_descriptor(variant: str):
    processor, read_noc, write_noc, noc_mode, kernel_noc = FUSED_VARIANTS[variant]
    dm = _descriptor_module()

    def create_program_descriptor(input_tensor, output_tensor, plan):
        all_cores = plan["core_ranges"]
        block_pages = plan["block_pages"]
        aligned_page_bytes = plan["aligned_page_bytes"]

        cb_page_staging = ttnn.CBDescriptor(
            total_size=plan["cb_depth_blocks"] * block_pages * aligned_page_bytes,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=dm.CB_PAGE_STAGING,
                    data_format=input_tensor.dtype,
                    page_size=aligned_page_bytes,
                )
            ],
        )

        input_accessor_args = ttnn.TensorAccessorArgs(input_tensor)
        output_accessor_args = ttnn.TensorAccessorArgs(output_tensor)
        if len(input_accessor_args.get_common_runtime_args()) or len(output_accessor_args.get_common_runtime_args()):
            raise RuntimeError("g6_clone: interleaved TensorAccessorArgs unexpectedly carry common runtime args")

        noc_split = plan["bank_noc_split"] != "none"
        own_other = noc_mode == "own_other"
        dynamic = noc_mode == "dynamic" or (noc_split and not own_other)
        walk = plan["partial_tile_walk"]
        ct_args = (
            [
                dm.CB_PAGE_STAGING,
                block_pages,
                int(walk is not None),
                plan["page_stride"],
                plan["bank_endpoint_noc"],
                int(plan["bank_stagger"]),
                int(noc_split),
                int("read" in plan["bank_vc_directions"]),
                int("write" in plan["bank_vc_directions"]),
                read_noc,
                write_noc,
                plan["cb_depth_blocks"],
                int(own_other),
            ]
            + dm._walk_ct_args(walk)
            + list(input_accessor_args.get_compile_time_args())
            + list(output_accessor_args.get_compile_time_args())
        )

        rt_args = ttnn.RuntimeArgs()
        for idx, ((core, _, start_page, num_pages), (read_vc, write_vc, other_noc_pages)) in enumerate(
            zip(plan["roles"], plan["route_args"])
        ):
            rotate = dm.stagger_rotate(idx, start_page, num_pages, plan["num_banks"]) if plan["bank_stagger"] else 0
            rt_args[core.x][core.y] = [start_page, num_pages, rotate, read_vc, write_vc, other_noc_pages]

        config = ttnn.DataMovementConfigDescriptor(
            processor=processor,
            noc=_NOCS[kernel_noc],
            noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC if dynamic else ttnn.NOC_MODE.DM_DEDICATED_NOC,
        )
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / FUSED_KERNEL),
            core_ranges=all_cores,
            compile_time_args=ct_args,
            defines=[(dm.ZONE_BUILD_DEFINE, "1")] if plan["zone_build"] else [],
            runtime_args=rt_args,
            common_runtime_args=[input_tensor.buffer_address(), output_tensor.buffer_address()],
            config=config,
        )
        return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb_page_staging])

    return create_program_descriptor


def patch_addresses(descriptor, input_tensor, output_tensor) -> None:
    """Write this call's buffer addresses into a cached fused descriptor, in place."""
    args = descriptor.kernels[FUSED_KERNEL_IDX].common_runtime_args
    args[0] = input_tensor.buffer_address()
    args[1] = output_tensor.buffer_address()


def install(variant):
    """Swap the op's descriptor builder for the fused one (variant None: no change)."""
    if not variant:
        return
    if variant not in FUSED_VARIANTS:
        raise ValueError(f"fused_single_risc: unknown variant {variant!r}; one of {sorted(FUSED_VARIANTS)}")
    dm = _descriptor_module()
    op = importlib.import_module("ttnn.operations.g6_clone.g6_clone")
    create = make_create_program_descriptor(variant)
    for module in (dm, op):
        module.create_program_descriptor = create
        module.patch_addresses = patch_addresses
    op._DESCRIPTOR_CACHE.clear()
