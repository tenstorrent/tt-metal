# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""perf experiment rm_issue_cost: the variants of the nhwc_rm_transpose reader/writer issue path.

A variant = kernel defines (on both kernels, copies in ./kernels) + host knob overrides of
g6_permute_nhwc_rm_transpose. An arm = extra ablation defines (ABL_NO_GATHER stubs the RISC-V transpose).
install() monkeypatches the regime's create_program_descriptor so the op dispatches the variant; the caller
clears g6_permute._PROGRAM_CACHE around it. "op" keeps the real op's kernel files (sanity reference).
"""

from pathlib import Path

KDIR = Path(__file__).resolve().parent / "kernels"

RC = "RIC_READ_CURSOR"
WC = "RIC_WRITE_CURSOR"
UP = "RIC_UNIT_PUSH"
TP = "RIC_TRID_PIPE"
R2 = "RIC_READ_2BUF"
W2 = "RIC_WRITE_2BUF"

# name -> (defines, nhwc knob overrides, use my kernel copies)
VARIANTS = {
    "op": ([], {}, False),
    "base": ([], {}, True),
    "bp64": ([], {"BLOCK_PAGES": 64}, True),
    "rcur": ([RC], {}, True),
    "wcur": ([WC], {}, True),
    "rwcur": ([RC, WC], {}, True),
    "rwcur_bp64": ([RC, WC], {"BLOCK_PAGES": 64}, True),
    "rwcur_bp64_d4": ([RC, WC], {"BLOCK_PAGES": 64, "INPUT_DEPTH": 4}, True),
    "rwcur_unit": ([RC, WC, UP], {}, True),
    "rwcur_unit_trid": ([RC, WC, UP, TP], {}, True),
    "unit": ([UP], {}, True),
    "rw2": ([RC, WC, R2, W2], {}, True),
    "rw2_bp64": ([RC, WC, R2, W2], {"BLOCK_PAGES": 64}, True),
    "rw2_unit": ([RC, WC, R2, W2, UP], {}, True),
    "rw2_unit_trid": ([RC, WC, R2, W2, UP, TP], {}, True),
    "rwcur_unit_trid1": ([RC, WC, UP, TP, "RIC_TRID_FIRST_WAIT"], {}, True),
    "rw2_unit_trid1": ([RC, WC, R2, W2, UP, TP, "RIC_TRID_FIRST_WAIT"], {}, True),
    "wcur_unit_trid": ([WC, UP, TP], {}, True),
    "rw2_unit_trid1_bp64": ([RC, WC, R2, W2, UP, TP, "RIC_TRID_FIRST_WAIT"], {"BLOCK_PAGES": 64}, True),
    "rw2_unit_bp64": ([RC, WC, R2, W2, UP], {"BLOCK_PAGES": 64}, True),
    # The recommended combination. L1 input: cursor + 2-buffer page reads, unit pushes on a transaction-id
    # pipeline, BLOCK_PAGES 256 (a ring of 8 units keeps the pipeline fed). DRAM input: BLOCK_PAGES 64 (a ring
    # of 2 blocks of <= 64 pages); pages <= 64 B take bank-coalesced reads + compact (BEST_DRAM_COMPACT_WORDS),
    # longer pages the page-wise bank-major path with unit pushes. Both placements: cursor writes.
    "best": ([RC, WC, R2, W2, UP, TP, "RIC_TRID_FIRST_WAIT"], {"BLOCK_PAGES": "per_placement"}, True),
}

# Variants whose reader takes the bank-coalesced + compact path (blocks_rm_bank_coalesced's) on nhwc:
# name -> (defines, knobs, compact batch words). Host-built: staging CB + reader_descriptor(staging=...).
COALESCED = {
    "coal": ([], {}, 8),
    "coal_w": ([WC], {}, 8),
    "coal_w_b32": ([WC], {}, 32),
    "coal_w2_b32": ([WC, W2], {}, 32),
    "coal_w2_b32_bp512": ([WC, W2], {"BLOCK_PAGES": 512}, 32),
    "coal_w2_b32_bp64": ([WC, W2], {"BLOCK_PAGES": 64}, 32),
    "coal_w2_b64_bp64": ([WC, W2], {"BLOCK_PAGES": 64}, 64),
    "coal_w2_b32_bp64_sd2": ([WC, W2], {"BLOCK_PAGES": 64}, 32, 2),
}
for _name, _spec in COALESCED.items():
    VARIANTS[_name] = (_spec[0], _spec[1], True)

ARMS = {
    "full": [],
    "nogather": ["ABL_NO_GATHER"],
}


def _coalesced_descriptor(nhwc_module, real, plan, input_tensor, output_tensor, batch_words, staging_depth=None):
    """The regime's descriptor with the shared reader on its bank-coalesced path: a staging CB of
    num_banks regions of staging_rounds slots, payloads compacted into cb_input_slots at slot_bytes."""
    import dataclasses

    import ttnn

    from ttnn.operations.g6_permute import g6_permute_blocks_rm_gather as blocks

    plan = dataclasses.replace(plan, bank_major=False)
    desc = real(plan, input_tensor, output_tensor)
    block_pages = plan.block_units * plan.block_w
    rounds = (plan.num_banks - 1 + block_pages - 1) // plan.num_banks + 1
    depth = min(staging_depth or blocks.STAGING_DEPTH, plan.max_blocks_per_core)
    stage_bytes = plan.num_banks * rounds * plan.slot_bytes
    cbs = list(desc.cbs)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=depth * stage_bytes,
            core_ranges=plan.all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=blocks.CB_BANK_STAGING, data_format=input_tensor.dtype, page_size=stage_bytes
                )
            ],
        )
    )
    starts = [(core, *nhwc_module._unit_start(plan, start), count) for core, start, count in plan.assignment]
    saved = blocks.COMPACT_BATCH_WORDS
    blocks.COMPACT_BATCH_WORDS = batch_words
    try:
        reader = blocks.reader_descriptor(
            plan,
            input_tensor,
            plan.block_units,
            plan.block_w,
            plan.units_per_row,
            plan.tail_w,
            [(core, first_page, count, cursor[0]) for core, first_page, cursor, count in starts],
            (rounds, depth, plan.slot_bytes),
        )
    finally:
        blocks.COMPACT_BATCH_WORDS = saved
    kernels = list(desc.kernels)
    kernels[nhwc_module.READER_KERNEL_INDEX] = reader
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


BLOCK_PAGES_PER_PLACEMENT = {"L1": 256, "DRAM": 64}
# best on DRAM input: the bank-coalesced + compact read path (8 banks: a 64-page unit is 8 reads of 8 slots),
# payload words copied per batch.
BEST_DRAM_COMPACT_WORDS = 32
# ... for pages up to one 64 B DRAM slot: the compact copies every payload word on the RISC, and a 128 B page
# (C = 64) costs more there than the DRAM request it saves (c64_dram: 52.2 us coalesced vs 35.7 us page-wise).
BEST_DRAM_COALESCED_MAX_PAGE_BYTES = 64


def _is_dram(input_tensor):
    import ttnn

    return input_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM


def install(monkeypatch, nhwc_module, variant, arm):
    defines, knobs, swap = VARIANTS[variant]
    for name, value in knobs.items():
        if value == "per_placement":
            real_build = nhwc_module.build_plan

            def build_plan(input_tensor, canonical_dims, _real=real_build, _name=name):
                import ttnn

                dram = input_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
                monkeypatch.setattr(nhwc_module, _name, BLOCK_PAGES_PER_PLACEMENT["DRAM" if dram else "L1"])
                return _real(input_tensor, canonical_dims)

            monkeypatch.setattr(nhwc_module, "build_plan", build_plan)
        else:
            monkeypatch.setattr(nhwc_module, name, value)
    all_defines = list(defines) + list(ARMS[arm])
    if not swap and not all_defines:
        return
    real = nhwc_module.create_program_descriptor

    def patched(plan, input_tensor, output_tensor):
        if variant in COALESCED:
            spec = COALESCED[variant]
            desc = _coalesced_descriptor(
                nhwc_module, real, plan, input_tensor, output_tensor, spec[2], spec[3] if len(spec) > 3 else None
            )
        elif variant == "best" and _is_dram(input_tensor) and plan.in_page_bytes <= BEST_DRAM_COALESCED_MAX_PAGE_BYTES:
            desc = _coalesced_descriptor(nhwc_module, real, plan, input_tensor, output_tensor, BEST_DRAM_COMPACT_WORDS)
        else:
            desc = real(plan, input_tensor, output_tensor)
        for kernel in desc.kernels:
            if swap:
                kernel.kernel_source = str(KDIR / Path(kernel.kernel_source).name)
            kernel.defines = [(d, "1") for d in all_defines]
        return desc

    monkeypatch.setattr(nhwc_module, "create_program_descriptor", patched)


def install_flux(monkeypatch, blocks_module, variant):
    """FLUX (blocks_rm_gather / blocks_rm_bank_coalesced): only the shared reader is swapped, with the
    variant's reader defines; the FLUX writer is the op's."""
    defines, _, swap = VARIANTS[variant]
    reader_defines = [d for d in defines if d in (RC, R2)]
    if not swap:
        return
    real = blocks_module.create_program_descriptor

    def patched(plan, input_tensor, output_tensor):
        desc = real(plan, input_tensor, output_tensor)
        for kernel in desc.kernels:
            if Path(kernel.kernel_source).name == "g6_permute_blocks_reader.cpp":
                kernel.kernel_source = str(KDIR / "g6_permute_blocks_reader.cpp")
                kernel.defines = [(d, "1") for d in reader_defines]
        return desc

    monkeypatch.setattr(blocks_module, "create_program_descriptor", patched)
