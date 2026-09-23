"""Generate ablation kernel dirs: payload stubbed, synchronization kept (/perf-measure).

R = bank-coalesced DRAM reads, S = loopback scatter, C = tilize compute (unpack/math/pack),
W = DRAM tile writes. Each variant is its own dir so each JIT-compiles fresh.
"""
import os, re, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "../../kernels")
VARIANTS = ["full", "S", "R", "C", "W", "RS", "RSC", "WC", "RSW", "RSCW", "WCS"]


def guard(s, needle, macro):
    assert s.count(needle) == 1, needle
    return s.replace(needle, f"\n#ifndef {macro}\n{needle}\n#endif\n")


def gen(v):
    d = os.path.join(HERE, "kernels_" + v)
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(SRC, d)
    defs = "".join(f"#define ABL_{c} 1\n" for c in v if v != "full")
    r = open(os.path.join(d, "tilize_reader.cpp")).read()
    r = guard(
        r,
        """                noc_async_read(
                    accessor.get_noc_addr(first_stick + j),
                    run_stage + run.offset(j) * stick_page_bytes,
                    run.sticks(j) * stick_page_bytes);""",
        "ABL_R",
    )
    r = guard(r, """                        noc_async_read_one_packet_with_state(src, dst);""", "ABL_S")
    h = open(os.path.join(d, "tilize_stick_reads.hpp")).read()
    h = guard(
        h,
        """                noc_async_write(
                    l1_row_addr + t * out_tile_bytes,
                    accessor.get_noc_addr(row_tile_idx + t, 0, noc),
                    out_tile_bytes,
                    noc);""",
        "ABL_W",
    )
    c = open(os.path.join(d, "tilize_compute.cpp")).read()
    call = """        compute_kernel_lib::tilize<
            block_width,
            cb_input_sticks,
            cb_output_tiles,
            InitUninitMode::InitAndUninit,
            WaitMode::WaitBlock,
            ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
            fp32_mode>(num_blocks);"""
    assert c.count(call) == 1
    c = c.replace(
        call,
        "#ifndef ABL_C\n"
        + call
        + """
#else
        for (uint32_t b = 0; b < num_blocks; ++b) {
            cb_wait_front(cb_input_sticks, block_width);
            cb_reserve_back(cb_output_tiles, block_width);
            cb_push_back(cb_output_tiles, block_width);
            cb_pop_front(cb_input_sticks, block_width);
        }
#endif""",
    )
    open(os.path.join(d, "tilize_reader.cpp"), "w").write(defs + r)
    open(os.path.join(d, "tilize_stick_reads.hpp"), "w").write(h)
    open(os.path.join(d, "tilize_compute.cpp"), "w").write(defs + c)
    w = open(os.path.join(d, "tilize_writer.cpp")).read()
    open(os.path.join(d, "tilize_writer.cpp"), "w").write(defs + w)


for v in VARIANTS:
    gen(v)
print("generated", VARIANTS)
