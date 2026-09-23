"""Step 1 (timing-only bound) kernel-dir generator for bank_paired_writes.

Variant name grammar: <base>_<writer>
  base   : W  = writes-only (reads R, scatter S, compute C stubbed; sync kept, as breakdown/RSC)
           F  = full op (only the writer's issue pattern changes)
  writer : base          = the op's store_rows (honest baseline)
           s<k>          = bpw MODE 0 (existing split), groups of up to k same-bank pages
           i<k>_<total>  = bpw MODE 1 (ideal bank-aligned re-split), total output tiles = <total>
Every dir is a copy of the real kernels (never modified in place) so each JIT-compiles fresh.
"""
import os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "../../kernels")


def guard(s, needle, macro):
    assert s.count(needle) == 1, needle
    return s.replace(needle, f"\n#ifndef {macro}\n{needle}\n#endif\n")


WRITER_LOOP = """        for (uint32_t done = 0; done < num_positions; done += rows_per_quantum) {
            const uint32_t remaining = num_positions - done;
            tilize_dataflow::store_rows<cb_output_tiles, block_width, out_tile_bytes, write_noc_split>(
                output_accessor,
                store_walk,
                tiles_per_row,
                remaining < rows_per_quantum ? remaining : rows_per_quantum,
                stick_rotation);
        }"""

BPW_LOOP = (
    """#ifdef BPW_K
        // TIMING-ONLY (data wrong): col_start == 0 and block_width == C assumed (focus shapes).
        bpw::store_all<cb_output_tiles, rows_per_quantum * block_width, out_tile_bytes, BPW_K, BPW_MODE, BPW_ORDER, block_width>(
            output_accessor, row_start * tiles_per_row, core_row_tiles * tiles_per_row, BPW_TOTAL, stick_rotation,
            row_rotation, core_row_tiles);
#else
"""
    + WRITER_LOOP
    + """
#endif"""
)


def gen(name):
    base, writer = name.split("_", 1)
    d = os.path.join(HERE, "kernels_" + name)
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(SRC, d)
    shutil.copy(os.path.join(HERE, "bpw_timing.hpp"), d)
    defs = ""
    if base == "W":
        defs += "#define ABL_R 1\n#define ABL_S 1\n#define ABL_C 1\n"
    wdefs = defs
    if writer != "base":
        split = writer[0] == "p"  # p<...> = same address sequence, one-page writes (packet-size control)
        if split:
            writer = writer[1:]
            wdefs += "#define BPW_SPLIT 1\n"
        order = 0 if writer[0] == "b" else 1  # b<...> = bank order, else walk order
        if order == 0:
            writer = writer[1:]
        mode = 0 if writer[0] == "s" else 1
        parts = writer[1:].split("_")
        k = int(parts[0])
        total = int(parts[1]) if mode == 1 else 0
        wdefs += f"#define BPW_K {k}\n#define BPW_MODE {mode}\n#define BPW_TOTAL {total}\n#define BPW_ORDER {order}\n"
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
    w = open(os.path.join(d, "tilize_writer.cpp")).read()
    assert w.count(WRITER_LOOP) == 1
    w = w.replace(WRITER_LOOP, BPW_LOOP)
    w = w.replace('#include "tilize_stick_reads.hpp"', '#include "tilize_stick_reads.hpp"\n#include "bpw_timing.hpp"')
    files = {"tilize_reader.cpp": defs + r, "tilize_compute.cpp": defs + c, "tilize_writer.cpp": wdefs + w}
    for f in ("tilize_stick_reads.hpp", "bpw_timing.hpp"):
        files[f] = open(os.path.join(d, f)).read()
    if not os.environ.get("BPW_ZONES"):
        # 16-bit zone-location hashes (name, file, line) collide across many variant dirs in one
        # profiled session (profiler.cpp TT_THROW): zones compiled out; *-KERNEL spans remain.
        inc = '#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"'
        for f in files:
            files[f] = files[f].replace(
                inc, inc + "\n#undef MaybeDeviceZoneScope\n#define MaybeDeviceZoneScope(name) (void(name))"
            )
    for f, txt in files.items():
        open(os.path.join(d, f), "w").write(txt)
    return
    open(os.path.join(d, "tilize_reader.cpp"), "w").write(defs + r)
    open(os.path.join(d, "tilize_compute.cpp"), "w").write(defs + c)
    open(os.path.join(d, "tilize_writer.cpp"), "w").write(wdefs + w)


if __name__ == "__main__":
    names = sys.argv[1:] or [
        "W_base",
        "W_s1",
        "W_s2",
        "W_s4",
        "W_i2_1024",
        "W_i4_1024",
        "F_base",
        "F_s1",
        "F_s2",
        "F_i2_1024",
        "F_i4_1024",
        "W_s3",
        "W_i2_2048",
        "W_i3_2048",
        "W_i4_2048",
        "F_s3",
        "F_i2_2048",
        "F_i4_2048",
    ]
    for n in names:
        gen(n)
    print("generated", names)
