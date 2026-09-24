"""hop_aware_noc duo host patch (experiment only; the op's real descriptor is untouched).

Wraps tilize.py's create_program_descriptor: on a resident input with a streamed (non-resident)
output and the plain store_rows writer, NCRISC (the reader kernel) takes the NoC0 share of the
tile writes. Adds two program semaphores (ready / done flags), appends to the reader's CT args
[duo, cb_out, block_width, out_tile_bytes, depth_out, rows_per_quantum, ready_sem, done_sem] + the
output TensorAccessorArgs, to the writer's CT args [duo, ready_sem, done_sem], and the output
buffer address to the reader's RT args (index 10). Elsewhere duo = 0 and the kernels run as HEAD.
"""
import sys

import ttnn

DUO_READY_SEM, DUO_DONE_SEM = 1, 2  # CO_READ_SEM = 0 is the op's only other semaphore
READER_FIXED_CT, WRITER_FIXED_CT = 30, 20


def _cores(crs):
    for r in crs.ranges():
        for x in range(r.start.x, r.end.x + 1):
            for y in range(r.start.y, r.end.y + 1):
                yield x, y


def _duo(p):
    ks = list(p.kernels)
    reader = next(k for k in ks if k.kernel_source.endswith("tilize_reader.cpp"))
    writer = next(k for k in ks if k.kernel_source.endswith("tilize_writer.cpp"))
    rct, wct = list(reader.compile_time_args), list(writer.compile_time_args)
    n_in = len(rct) - READER_FIXED_CT
    out_args = wct[WRITER_FIXED_CT : len(wct) - n_in]
    duo = (
        rct[9] != 0  # input resident: NCRISC is idle
        and rct[12] == 0  # not retile
        and rct[20] == 0  # not padded (reader RT arg 10 is free)
        and wct[11] == 0  # output streamed
        and wct[3] == 0  # no split reader
        and wct[18] == 0  # no co-read
        and wct[16] == 1  # store_rows (write_ahead 1)
        and wct[14] == 0  # no parked write NoC split
    )
    reader.compile_time_args = (
        rct + [int(duo), wct[0], wct[1], wct[2], wct[15], wct[10], DUO_READY_SEM, DUO_DONE_SEM] + out_args
    )
    writer.compile_time_args = wct + [int(duo), DUO_READY_SEM, DUO_DONE_SEM]
    rrt, wrt = reader.runtime_args, writer.runtime_args
    for x, y in _cores(reader.core_ranges):
        args = list(rrt[x][y])
        if duo:
            assert len(args) == 10, len(args)
        rrt[x][y] = args + [list(wrt[x][y])[0]]
    reader.runtime_args = rrt
    sems = list(p.semaphores)
    if duo:
        sems += [
            ttnn.SemaphoreDescriptor(id=DUO_READY_SEM, core_ranges=reader.core_ranges, initial_value=0),
            ttnn.SemaphoreDescriptor(id=DUO_DONE_SEM, core_ranges=reader.core_ranges, initial_value=0),
        ]
    print(f"HOP duo={int(duo)}")
    return ttnn.ProgramDescriptor(kernels=ks, semaphores=sems, cbs=list(p.cbs))


def install(monkeypatch, pd):
    mod = sys.modules["ttnn.operations.tilize.tilize"]
    orig = pd.create_program_descriptor

    def patched(*a, **k):
        return _duo(orig(*a, **k))

    monkeypatch.setattr(mod, "create_program_descriptor", patched)


HOP_FLAG_SEM = 3  # dedf: BRISC's "NoC0 writes ACKed" flag for NCRISC's counter re-sync


def install_flag(monkeypatch, pd):
    mod = sys.modules["ttnn.operations.tilize.tilize"]
    orig = pd.create_program_descriptor

    def patched(*a, **k):
        p = orig(*a, **k)
        ks = list(p.kernels)
        crs = next(k_ for k_ in ks if k_.kernel_source.endswith("tilize_writer.cpp")).core_ranges
        sems = list(p.semaphores) + [ttnn.SemaphoreDescriptor(id=HOP_FLAG_SEM, core_ranges=crs, initial_value=0)]
        return ttnn.ProgramDescriptor(kernels=ks, semaphores=sems, cbs=list(p.cbs))

    monkeypatch.setattr(mod, "create_program_descriptor", patched)


# ---- dedg: the graduation candidate's host side ------------------------------------------------
HOP_WRITE_MIN_SAVING = 6  # hops; 28 % of (Tensix core, bank) pairs on NoC0 on WH n150 (sweep: T4 / T8 worse)
HOP_SEM = 3


def hop_gate(input_tensor, output_tensor, rct, wct):
    """Hop-aware writes where NoC0 carries no DRAM read traffic: input not in DRAM (resident shard or
    L1), output DRAM TensorMemoryLayout::INTERLEAVED and streamed by store_rows."""
    im, om = input_tensor.memory_config(), output_tensor.memory_config()
    return (
        im.buffer_type != ttnn.BufferType.DRAM
        and om.buffer_type == ttnn.BufferType.DRAM
        and om.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
        and wct[11] == 0  # output streamed (not resident)
        and wct[3] == 0  # no split reader
        and wct[16] == 1  # store_rows (write_ahead 1)
        and wct[14] == 0  # parked write NoC split off
        and rct[27] == 0  # parked loopback scatter WRITES off: NCRISC issues no NoC write
    )


def install_gated(monkeypatch, pd):
    mod = sys.modules["ttnn.operations.tilize.tilize"]
    orig = pd.create_program_descriptor

    def patched(input_tensor, output_tensor, **k):
        p = orig(input_tensor, output_tensor, **k)
        ks = list(p.kernels)
        reader = next(k_ for k_ in ks if k_.kernel_source.endswith("tilize_reader.cpp"))
        writer = next(k_ for k_ in ks if k_.kernel_source.endswith("tilize_writer.cpp"))
        rct, wct = list(reader.compile_time_args), list(writer.compile_time_args)
        hop_t = HOP_WRITE_MIN_SAVING if hop_gate(input_tensor, output_tensor, rct, wct) else 0
        reader.compile_time_args = rct + [hop_t, HOP_SEM]
        writer.compile_time_args = wct + [hop_t, HOP_SEM]
        sems = list(p.semaphores)
        if hop_t:
            sems.append(ttnn.SemaphoreDescriptor(id=HOP_SEM, core_ranges=writer.core_ranges, initial_value=0))
        print(f"HOP gated hop_t={hop_t}")
        return ttnn.ProgramDescriptor(kernels=ks, semaphores=sems, cbs=list(p.cbs))

    monkeypatch.setattr(mod, "create_program_descriptor", patched)


def install_dedr(monkeypatch, pd):
    """Reader twin gate: resident OUTPUT (BRISC idle), DRAM input read by the plain StickProducer."""
    mod = sys.modules["ttnn.operations.tilize.tilize"]
    orig = pd.create_program_descriptor

    def patched(input_tensor, output_tensor, **k):
        p = orig(input_tensor, output_tensor, **k)
        ks = list(p.kernels)
        reader = next(k_ for k_ in ks if k_.kernel_source.endswith("tilize_reader.cpp"))
        writer = next(k_ for k_ in ks if k_.kernel_source.endswith("tilize_writer.cpp"))
        rct, wct = list(reader.compile_time_args), list(writer.compile_time_args)
        on = (
            input_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
            and input_tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
            and wct[11] != 0  # output resident: BRISC issues nothing
            and rct[9] == 0
            and rct[12] == 0
            and rct[20] == 0  # stick reader, not retile, not padded
            and rct[5] == 0
            and rct[28] == 0
            and rct[26] == 0  # no split reader / co-read / bank_coalesced
            and rct[19] == 0
            and rct[11] == 1
            and rct[17] == 0  # no bank stride, one page per stick
        )
        if on:
            rct[17] = 1  # read_noc_split != 0: the StickProducer barriers both NoCs; the NoC is picked per bank
        reader.compile_time_args = rct + [int(on), HOP_SEM]
        writer.compile_time_args = wct + [int(on), HOP_SEM]
        sems = list(p.semaphores)
        if on:
            sems.append(ttnn.SemaphoreDescriptor(id=HOP_SEM, core_ranges=writer.core_ranges, initial_value=0))
        print(f"HOP dedr on={int(on)}")
        return ttnn.ProgramDescriptor(kernels=ks, semaphores=sems, cbs=list(p.cbs))

    monkeypatch.setattr(mod, "create_program_descriptor", patched)
