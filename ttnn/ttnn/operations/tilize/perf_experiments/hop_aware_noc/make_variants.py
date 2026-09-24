"""Perf 2 / hop_aware_noc: generate kernel-dir variants from the op's CURRENT kernels.

Idea: choose the NoC of every DRAM tile write per (Tensix core, DRAM bank) by hop distance.
WH NoC0 routes east then south, NoC1 west then north (10 x 12 torus, physical NoC0 frame);
the write data rides the request, so the shorter core->bank path is the one that matters.

Stage 1 (timing bound, single RISC-V): BRISC issues every write, choosing the NoC per page's
bank, with both DM kernels in DM_DYNAMIC_NOC (the harness sets pd.WRITE_NOC_SPLIT = 1, which
flips the kernels' NOC_MODE and passes write_noc_split = 1 to store_rows; here that CT arg
means "use the hop mask"). HOP_MODE:
  0 own   every write on NoC1 (dynamic-mode cost control, same addresses)
  1 hop   bank on NoC0 iff hops(NoC0) < hops(NoC1)
  2 anti  the inverse (in case the routing model is backwards)
  3 rand  a fixed pseudo-random 50/50 per (core, bank) (two-NoC control, no hop awareness)
  4 xonly by horizontal hops only (east vs west)
  5 all0  every write on NoC0 (geometry control)
  6 hopT<k> hop-aware, but NoC0 only where it saves >= k hops (T6: 28 % of pairs on NoC0)
  7 rand28 pseudo-random 28 % of (core, bank) pairs on NoC0 (share control for T6)
  8 antiT6 the 28 % of pairs NoC0 serves worst (mirror of T6)
Ablations (payload stubbed, synchronization kept) reuse p2_breakdown/make_ablations.py's needles:
suffix _RSC = writes only (reads, scatter, compute stubbed), _C = compute stubbed.

Dirs: kernels_<name>/ (git-ignored; rerun this script to rebuild).
"""
import os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "../../kernels")

MODES = {"own": 0, "hop": 1, "anti": 2, "rand": 3, "xonly": 4, "all0": 5}
THRESHOLDS = (2, 4, 6, 8, 10)  # mode 6: NoC0 only when it saves >= T hops

HOP_HELPER = r"""
// ---- hop_aware_noc experiment: per-(Tensix core, DRAM bank) NoC choice -------------------------
// Bit b of the mask set = DRAM bank b's writes go on NoC0 (else NoC1). Coordinates are the
// physical NoC0 frame: this core's comes from NIU 0's NOC_NODE_ID register and
// dram_bank_to_noc_xy[0][b] is the bank's worker endpoint in NoC0 coordinates (DRAM is not
// virtualized on Wormhole). NoC0 goes east (+x) then south (+y), NoC1 west then north, both wrap.
#ifndef HOP_MODE
#define HOP_MODE 1
#endif
#ifndef HOP_T
#define HOP_T 1
#endif
namespace hop_noc {
constexpr uint32_t GX = 10, GY = 12;  // Wormhole NoC grid (soc descriptor)
inline uint32_t g_mask = 0;
inline uint32_t compute_mask() {
    // my_x / my_y hold the TRANSLATED id (NOC_ID_LOGICAL, 18.. on Wormhole); the physical one is
    // NIU 0's NOC_NODE_ID register.
    const uint32_t node_id = *reinterpret_cast<volatile uint32_t*>(NOC_NODE_ID);
    const uint32_t mx = node_id & NOC_NODE_ID_MASK, my = (node_id >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    uint32_t mask = 0;
    for (uint32_t b = 0; b < NUM_DRAM_BANKS; ++b) {
        const uint32_t xy = dram_bank_to_noc_xy[0][b] >> NOC_COORD_REG_OFFSET;
        const uint32_t x = xy & NOC_NODE_ID_MASK, y = (xy >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
        const uint32_t e = (x + GX - mx) % GX, s = (y + GY - my) % GY;  // NoC0 hops: east, south
        const uint32_t w = (GX - e) % GX, n = (GY - s) % GY;            // NoC1 hops: west, north
        bool use0;
        if constexpr (HOP_MODE == 0) {
            use0 = false;
        } else if constexpr (HOP_MODE == 1) {
            use0 = (e + s) < (w + n);
        } else if constexpr (HOP_MODE == 2) {
            use0 = (e + s) > (w + n);
        } else if constexpr (HOP_MODE == 3) {
            uint32_t h = (mx * 73856093u) ^ (my * 19349663u) ^ (b * 83492791u);
            h ^= h >> 13;
            h *= 0x5bd1e995u;
            h ^= h >> 15;
            use0 = (h & 1) != 0;
        } else if constexpr (HOP_MODE == 4) {
            use0 = e < w;
        } else if constexpr (HOP_MODE == 5) {
            use0 = true;
        } else if constexpr (HOP_MODE == 6) {
            use0 = (w + n) >= (e + s) + HOP_T;  // hop-aware with a minimum saving of HOP_T hops
        } else if constexpr (HOP_MODE == 7) {
            uint32_t h = (mx * 73856093u) ^ (my * 19349663u) ^ (b * 83492791u);
            h ^= h >> 13;
            h *= 0x5bd1e995u;
            h ^= h >> 15;
            use0 = (h & 0xff) < HOP_T;  // pseudo-random, NoC0 share HOP_T / 256 (share control)
        } else {
            use0 = (e + s) >= (w + n) + HOP_T;  // anti-threshold: the pairs NoC0 serves WORST, same share
        }
        mask |= (use0 ? 1u : 0u) << b;
    }
#ifdef HOP_DEBUG
    const uint32_t node1 = *reinterpret_cast<volatile uint32_t*>((1 << NOC_INSTANCE_OFFSET_BIT) + NOC_NODE_ID);
    DPRINT("HOP mx={} my={} tx={} ty={} n1x={} n1y={} mask={}\n", mx, my, (uint32_t)my_x[0], (uint32_t)my_y[0],
           node1 & 0x3f, (node1 >> 6) & 0x3f, mask);
    for (uint32_t b = 0; b < NUM_DRAM_BANKS; ++b) {
        const uint32_t xy = dram_bank_to_noc_xy[0][b] >> NOC_COORD_REG_OFFSET;
        const uint32_t xy1 = dram_bank_to_noc_xy[1][b] >> NOC_COORD_REG_OFFSET;
        DPRINT("HOP bank {} noc0 {},{} noc1 {},{}\n", b, (uint32_t)(xy & NOC_NODE_ID_MASK),
               (uint32_t)((xy >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK), (uint32_t)(xy1 & NOC_NODE_ID_MASK),
               (uint32_t)((xy1 >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK));
    }
#endif
    return mask;
}
FORCE_INLINE uint8_t pick(uint32_t page) { return ((g_mask >> (page % NUM_DRAM_BANKS)) & 1) ? 0 : 1; }
#ifdef RD_HOP_T
// Reader twin: the read DATA rides the response (bank -> core). NoC1's response path is west/north
// from the bank = e + s hops, NoC0's is w + n: bank b's reads go on NoC1 iff that saves >= RD_HOP_T.
inline uint32_t g_rmask = 0;
inline uint32_t compute_rmask() {
    const uint32_t node_id = *reinterpret_cast<volatile uint32_t*>(NOC_NODE_ID);
    const uint32_t mx = node_id & NOC_NODE_ID_MASK, my = (node_id >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    uint32_t mask = 0;
    for (uint32_t b = 0; b < NUM_DRAM_BANKS; ++b) {
        const uint32_t xy = dram_bank_to_noc_xy[0][b] >> NOC_COORD_REG_OFFSET;
        const uint32_t x = xy & NOC_NODE_ID_MASK, y = (xy >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
        const uint32_t e = (x + GX - mx) % GX, s = (y + GY - my) % GY;
        const uint32_t w = (GX - e) % GX, n = (GY - s) % GY;
        mask |= (((w + n) >= (e + s) + RD_HOP_T) ? 1u : 0u) << b;
    }
    return mask;
}
FORCE_INLINE uint8_t pick_read(uint32_t page) {
    return ((g_rmask >> (page % NUM_DRAM_BANKS)) & 1) ? 1 - noc_index : noc_index;
}
#endif
}  // namespace hop_noc
"""


def sub(s, old, new, count=1):
    assert s.count(old) == count, (old, s.count(old))
    return s.replace(old, new)


def guard(s, needle, macro):
    assert s.count(needle) == 1, needle
    return s.replace(needle, f"\n#ifndef {macro}\n{needle}\n#endif\n")


def apply_ablation(r, h, c, abl):
    if "R" in abl:
        r = guard(
            r,
            """                noc_async_read(
                    accessor.get_noc_addr(first_stick + j),
                    run_stage + run.offset(j) * stick_page_bytes,
                    run.sticks(j) * stick_page_bytes);""",
            "ABL_R",
        )
        h = guard(
            h,
            """            noc_async_read(accessor.get_noc_addr(first_stick + stick, segment_offset, noc), l1_dst, segment_bytes, noc);""",
            "ABL_R",
        )
    if "S" in abl:
        r = guard(r, """                        noc_async_read_one_packet_with_state(src, dst);""", "ABL_S")
    if "C" in abl:
        call = """        compute_kernel_lib::tilize<
            block_width,
            cb_input_sticks,
            cb_output_tiles,
            InitUninitMode::InitAndUninit,
            WaitMode::WaitBlock,
            ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
            fp32_mode>(num_blocks);"""
        c = sub(
            c,
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
    return r, h, c


def gen_dyn(name, mode, abl="", debug=False, thr=None):
    """Stage-1 single-RISC variant: BRISC picks the NoC per page's bank (DM_DYNAMIC_NOC)."""
    d = os.path.join(HERE, "kernels_" + name + (("_" + abl) if abl else ""))
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(SRC, d)
    defs = f"#define HOP_MODE {mode}\n" + ("#define HOP_DEBUG 1\n" if debug else "")
    defs += f"#define HOP_T {thr}\n" if thr is not None else ""
    defs += "".join(f"#define ABL_{x} 1\n" for x in abl)
    r = open(os.path.join(d, "tilize_reader.cpp")).read()
    h = open(os.path.join(d, "tilize_stick_reads.hpp")).read()
    c = open(os.path.join(d, "tilize_compute.cpp")).read()
    w = open(os.path.join(d, "tilize_writer.cpp")).read()
    if debug:
        h = sub(
            h,
            '#include "api/dataflow/dataflow_api.h"\n',
            '#include "api/dataflow/dataflow_api.h"\n#include "api/debug/dprint.h"\n',
        )
    h = sub(h, "namespace tilize_dataflow {\n", HOP_HELPER + "\nnamespace tilize_dataflow {\n")
    h = sub(
        h,
        "                const uint8_t noc = (noc_split != 0 && (n % noc_split) == noc_split - 1) ? 1 - noc_index : noc_index;",
        "                const uint8_t noc = (noc_split != 0) ? hop_noc::pick(row_tile_idx + t) : noc_index;",
    )
    w = sub(
        w,
        "    const auto output_accessor = TensorAccessor(output_args, dst_addr, out_tile_bytes);\n",
        "    const auto output_accessor = TensorAccessor(output_args, dst_addr, out_tile_bytes);\n"
        "    if constexpr (write_noc_split != 0) {\n        hop_noc::g_mask = hop_noc::compute_mask();\n    }\n",
    )
    r, h, c = apply_ablation(r, h, c, abl)
    if "W" in abl:
        h = guard(
            h,
            """                noc_async_write(
                    l1_row_addr + t * out_tile_bytes,
                    accessor.get_noc_addr(row_tile_idx + t, 0, noc),
                    out_tile_bytes,
                    noc);""",
            "ABL_W",
        )
    for fn, s in (
        ("tilize_reader.cpp", r),
        ("tilize_stick_reads.hpp", h),
        ("tilize_compute.cpp", c),
        ("tilize_writer.cpp", w),
    ):
        open(os.path.join(d, fn), "w").write(defs + s)
    return d


RD_ISSUE_OLD = """                noc_async_read(
                    accessor.get_noc_addr(first_stick + j),
                    run_stage + run.offset(j) * stick_page_bytes,
                    run.sticks(j) * stick_page_bytes);"""
RD_ISSUE_NEW = """                const uint8_t rnoc = hop_noc::pick_read(first_stick + j);
                noc_async_read(
                    accessor.get_noc_addr(first_stick + j, 0, rnoc),
                    run_stage + run.offset(j) * stick_page_bytes,
                    run.sticks(j) * stick_page_bytes,
                    rnoc);"""


def gen_twin(name, write_mode, write_t, read_t):
    """Reader twin (+ optional writer half), both DM kernels in DM_DYNAMIC_NOC.
    kernels_dyn_*: harness sets WRITE_NOC_SPLIT = 1 (bank_coalesced reader patched here);
    kernels_dynr_*: harness sets READ_NOC_SPLIT = 1 (StickProducer / read_tile_row_sticks path)."""
    d = gen_dyn(name, write_mode, "", thr=write_t)
    defs = f"#define RD_HOP_T {read_t}\n"
    r = open(os.path.join(d, "tilize_reader.cpp")).read()
    h = open(os.path.join(d, "tilize_stick_reads.hpp")).read()
    r = sub(r, RD_ISSUE_OLD, RD_ISSUE_NEW)
    r = sub(
        r,
        """\n        noc_async_read_set_trid(1 + (k % stage_depth));
""",
        """\n        noc_async_read_set_trid(1 + (k % stage_depth));
        noc_async_read_set_trid(1 + (k % stage_depth), 1 - noc_index);
""",
    )
    r = sub(
        r,
        """unit k's DRAM reads landing
            noc_async_read_barrier_with_trid(1 + (k % stage_depth));
""",
        """unit k's DRAM reads landing
            noc_async_read_barrier_with_trid(1 + (k % stage_depth));
            noc_async_read_barrier_with_trid(1 + (k % stage_depth), 1 - noc_index);
""",
    )
    r = sub(
        r,
        """    noc_async_read_set_trid(0);
}""",
        """    noc_async_read_set_trid(0);
    noc_async_read_set_trid(0, 1 - noc_index);
}""",
    )
    r = sub(r, "void kernel_main() {\n", "void kernel_main() {\n    hop_noc::g_rmask = hop_noc::compute_rmask();\n")
    h = sub(
        h,
        "            const uint8_t noc = (noc_split != 0 && (s % noc_split) == noc_split - 1) ? 1 - noc_index : noc_index;",
        "            const uint8_t noc = (noc_split != 0) ? hop_noc::pick_read(first_stick + stick) : noc_index;",
    )
    open(os.path.join(d, "tilize_reader.cpp"), "w").write(defs + r)
    open(os.path.join(d, "tilize_stick_reads.hpp"), "w").write(defs + h)
    return d


DUO_HELPER = r"""
// ---- hop_aware_noc duo: the tile writes of a quantum split by bank between two RISC-Vs, each on its
// own dedicated NoC. Bank b's pages go to NCRISC / NoC0 iff mask bit b (want0), else BRISC / NoC1.
namespace hop_noc {
template <uint32_t block_width, uint32_t out_tile_bytes, bool want0, typename Accessor, typename Walk>
FORCE_INLINE void issue_share(
    const Accessor& accessor,
    Walk& walk,
    uint32_t l1_row_addr,
    uint32_t tiles_per_row,
    uint32_t num_rows,
    uint32_t col_rotation,
    uint32_t mask) {
    for (uint32_t j = 0; j < num_rows; ++j, walk.advance()) {
        const uint32_t row_tile_idx = walk.row() * tiles_per_row + walk.first_col();
        const uint32_t valid_width = walk.valid_width();
        uint32_t t = col_rotation % valid_width;
        for (uint32_t n = 0; n < valid_width; ++n) {
            const uint32_t page = row_tile_idx + t;
            if ((((mask >> (page % NUM_DRAM_BANKS)) & 1) != 0) == want0) {
                noc_async_write(l1_row_addr + t * out_tile_bytes, accessor.get_noc_addr(page), out_tile_bytes);
            }
            if (++t == valid_width) {
                t = 0;
            }
        }
        l1_row_addr += block_width * out_tile_bytes;
    }
}
}  // namespace hop_noc
"""

DUO_WRITER_ARGS = """
    constexpr uint32_t duo_base = input_args.next_compile_time_args_offset();
    constexpr bool duo = get_compile_time_arg_val(duo_base) != 0;  // hop_aware_noc duo (host patch)
    constexpr uint32_t duo_ready_sem = get_compile_time_arg_val(duo_base + 1);
    constexpr uint32_t duo_done_sem = get_compile_time_arg_val(duo_base + 2);
"""

DUO_WRITER_LOOP = """    } else if constexpr (duo) {
        // BRISC stays cb_output_tiles' only consumer: per quantum it announces the front (ready = q + 1),
        // writes the NoC1 share, and pops only once NCRISC has flushed the NoC0 share (done >= q + 1).
        volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(duo_ready_sem));
        volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(duo_done_sem));
        const uint32_t mask = hop_noc::compute_mask();
        uint32_t q = 0;
        for (uint32_t pos = 0; pos < num_positions; pos += rows_per_quantum, ++q) {
            const uint32_t left = num_positions - pos;
            const uint32_t rows = left < rows_per_quantum ? left : rows_per_quantum;
            const uint32_t pages = rows * block_width;
            {
                MaybeDeviceZoneScope("writer_wait");
                cb_wait_front(cb_output_tiles, pages);
            }
            noc_semaphore_set(ready, q + 1);
            {
                MaybeDeviceZoneScope("writer_issue");
                hop_noc::issue_share<block_width, out_tile_bytes, false>(
                    output_accessor, store_walk, get_read_ptr(cb_output_tiles), tiles_per_row, rows, stick_rotation, mask);
            }
            {
                MaybeDeviceZoneScope("writer_flush");
                noc_async_writes_flushed();
                noc_semaphore_wait_min(done, q + 1);
            }
            cb_pop_front(cb_output_tiles, pages);
        }
        noc_semaphore_set(ready, 0);  // re-arm: NCRISC is past its last read of both flags
        noc_semaphore_set(done, 0);
    } else if constexpr (write_ahead > 1) {
"""

DUO_READER_ARGS = """
    constexpr uint32_t duo_base = input_args.next_compile_time_args_offset();
    constexpr bool duo = get_compile_time_arg_val(duo_base) != 0;  // hop_aware_noc duo (host patch)
    constexpr uint32_t duo_cb_out = get_compile_time_arg_val(duo_base + 1);
    constexpr uint32_t duo_block_width = get_compile_time_arg_val(duo_base + 2);
    constexpr uint32_t duo_out_tile_bytes = get_compile_time_arg_val(duo_base + 3);
    constexpr uint32_t duo_depth_out = get_compile_time_arg_val(duo_base + 4);
    constexpr uint32_t duo_rows_per_quantum = get_compile_time_arg_val(duo_base + 5);
    constexpr uint32_t duo_ready_sem = get_compile_time_arg_val(duo_base + 6);
    constexpr uint32_t duo_done_sem = get_compile_time_arg_val(duo_base + 7);
    constexpr auto duo_out_args = TensorAccessorArgs<duo_base + 8>();
"""

DUO_READER_RESIDENT = """        cb_push_back(cb_input_sticks, pages);
        if constexpr (duo) {
            // NCRISC is idle on a resident input: it writes the NoC0 share of every output quantum,
            // straight out of the slot BRISC announced (it never touches cb_output_tiles' counters).
            MaybeDeviceZoneScope("reader_duo_writes");
            const auto out_acc = TensorAccessor(duo_out_args, get_arg_val<uint32_t>(10), duo_out_tile_bytes);
            const uint32_t tiles_per_row = get_arg_val<uint32_t>(7);
            tilize_dataflow::Walker<duo_block_width> w(row_start, core_row_tiles, col_start, core_col_tiles, row_rotation);
            const uint32_t num_positions = w.num_positions();
            const uint32_t base = get_read_ptr(duo_cb_out);
            constexpr uint32_t slot_bytes = duo_rows_per_quantum * duo_block_width * duo_out_tile_bytes;
            volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(duo_ready_sem));
            volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(duo_done_sem));
            const uint32_t mask = hop_noc::compute_mask();
            uint32_t q = 0;
            for (uint32_t pos = 0; pos < num_positions; pos += duo_rows_per_quantum, ++q) {
                const uint32_t left = num_positions - pos;
                const uint32_t rows = left < duo_rows_per_quantum ? left : duo_rows_per_quantum;
                noc_semaphore_wait_min(ready, q + 1);
                hop_noc::issue_share<duo_block_width, duo_out_tile_bytes, true>(
                    out_acc, w, base + (q % duo_depth_out) * slot_bytes, tiles_per_row, rows, stick_rotation, mask);
                noc_async_writes_flushed();
                noc_semaphore_set(done, q + 1);
            }
            noc_async_write_barrier();
        }
        return;
"""


def gen_duo(name, mode, thr=None, abl=""):
    """Two-RISC-V writer (DM_DEDICATED_NOC): NCRISC writes the NoC0 share on a resident input."""
    d = os.path.join(HERE, "kernels_duo_" + name + (("_" + abl) if abl else ""))
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(SRC, d)
    defs = f"#define HOP_MODE {mode}\n" + (f"#define HOP_T {thr}\n" if thr is not None else "")
    defs += "".join(f"#define ABL_{x} 1\n" for x in abl)
    r = open(os.path.join(d, "tilize_reader.cpp")).read()
    h = open(os.path.join(d, "tilize_stick_reads.hpp")).read()
    c = open(os.path.join(d, "tilize_compute.cpp")).read()
    w = open(os.path.join(d, "tilize_writer.cpp")).read()
    h = sub(h, "namespace tilize_dataflow {\n", HOP_HELPER + DUO_HELPER + "\nnamespace tilize_dataflow {\n")
    w = sub(
        w,
        "    constexpr auto input_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();\n",
        "    constexpr auto input_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();\n"
        + DUO_WRITER_ARGS,
    )
    w = sub(w, "    } else if constexpr (write_ahead > 1) {\n", DUO_WRITER_LOOP)
    r = sub(
        r,
        "    constexpr auto input_args = TensorAccessorArgs<30>();\n",
        "    constexpr auto input_args = TensorAccessorArgs<30>();\n" + DUO_READER_ARGS,
    )
    r = sub(
        r,
        """        cb_push_back(cb_input_sticks, pages);
        return;
""",
        DUO_READER_RESIDENT,
    )
    r, h, c = apply_ablation(r, h, c, abl)
    if "W" in abl:
        h = sub(
            h,
            """                noc_async_write(l1_row_addr + t * out_tile_bytes, accessor.get_noc_addr(page), out_tile_bytes);""",
            "\n#ifndef ABL_W\n                noc_async_write(l1_row_addr + t * out_tile_bytes, accessor.get_noc_addr(page), out_tile_bytes);\n#endif\n",
        )
    for fn, s_ in (
        ("tilize_reader.cpp", r),
        ("tilize_stick_reads.hpp", h),
        ("tilize_compute.cpp", c),
        ("tilize_writer.cpp", w),
    ):
        open(os.path.join(d, fn), "w").write(defs + s_)
    return d


def gen_ded(name, mode, thr=None, abl=""):
    """BRISC writes on BOTH NoCs in DM_DEDICATED_NOC (no host change, no DM_DYNAMIC_NOC tax).

    Safe because the reader RISC-V (NCRISC) never issues a NoC WRITE on NoC0 in any enabled path
    (its reads use NoC0 command buffer 1, BRISC's writes command buffer 0 = NCRISC's unused write
    buffer), and the NIU's write counters (NONPOSTED_WR_REQ_SENT / WR_ACK_RECEIVED) are touched only
    by BRISC, which re-snapshots its NoC0 software write counters at kernel start
    (noc_local_state_init(0); firmware only initializes its own NoC's)."""
    d = os.path.join(HERE, "kernels_ded_" + name + (("_" + abl) if abl else ""))
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(SRC, d)
    defs = f"#define HOP_MODE {mode}\n" + (f"#define HOP_T {thr}\n" if thr is not None else "")
    defs += "".join(f"#define ABL_{x} 1\n" for x in abl)
    r = open(os.path.join(d, "tilize_reader.cpp")).read()
    h = open(os.path.join(d, "tilize_stick_reads.hpp")).read()
    c = open(os.path.join(d, "tilize_compute.cpp")).read()
    w = open(os.path.join(d, "tilize_writer.cpp")).read()
    h = sub(h, "namespace tilize_dataflow {\n", HOP_HELPER + "\nnamespace tilize_dataflow {\n")
    h = sub(
        h,
        "                const uint8_t noc = (noc_split != 0 && (n % noc_split) == noc_split - 1) ? 1 - noc_index : noc_index;",
        "                const uint8_t noc = hop_noc::pick(row_tile_idx + t);",
    )
    h = sub(
        h,
        """        noc_async_writes_flushed();
        if constexpr (noc_split != 0) {
            noc_async_writes_flushed(1 - noc_index);
        }""",
        """        noc_async_writes_flushed();
        noc_async_writes_flushed(1 - noc_index);""",
    )
    w = sub(
        w,
        "    const auto output_accessor = TensorAccessor(output_args, dst_addr, out_tile_bytes);\n",
        "    const auto output_accessor = TensorAccessor(output_args, dst_addr, out_tile_bytes);\n"
        "    noc_local_state_init(1 - noc_index);  // this RISC-V's NoC0 write counters <- NIU 0\n"
        "    hop_noc::g_mask = hop_noc::compute_mask();\n",
    )
    w = sub(
        w,
        """    noc_async_write_barrier();
    if constexpr (write_noc_split != 0) {
        noc_async_write_barrier(1 - noc_index);
    }""",
        """    noc_async_write_barrier();
    noc_async_write_barrier(1 - noc_index);""",
    )
    r, h, c = apply_ablation(r, h, c, abl)
    if "W" in abl:
        h = guard(
            h,
            """                noc_async_write(
                    l1_row_addr + t * out_tile_bytes,
                    accessor.get_noc_addr(row_tile_idx + t, 0, noc),
                    out_tile_bytes,
                    noc);""",
            "ABL_W",
        )
    for fn, s_ in (
        ("tilize_reader.cpp", r),
        ("tilize_stick_reads.hpp", h),
        ("tilize_compute.cpp", c),
        ("tilize_writer.cpp", w),
    ):
        open(os.path.join(d, fn), "w").write(defs + s_)
    return d


DEDF_WRITER_TAIL = """
// hop_aware_noc (dedf): every NoC0 write of this RISC-V has been ACKed; tell NCRISC, which then
// re-snapshots its NoC0 counters so the firmware's dedicated-NoC idle check holds for it too.
void kernel_main() {
    kernel_main_body();
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(HOP_FLAG_SEM)), 1);
}
"""
DEDF_READER_TAIL = """
// hop_aware_noc (dedf): BRISC also writes on this RISC-V's NoC (NIU 0). Wait until its NoC0 writes
// are all ACKed, then re-snapshot this RISC-V's NoC0 counters from the NIU (the kernel-end
// "no NoC transaction outstanding" check compares them with the NIU's).
void kernel_main() {
    kernel_main_body();
    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(HOP_FLAG_SEM));
    noc_semaphore_wait(flag, 1);
    noc_semaphore_set(flag, 0);
    noc_local_state_init(noc_index);
}
"""


def gen_dedf(name, mode, thr=None, abl=""):
    """ded + the NCRISC counter re-sync handshake (host patch adds semaphore HOP_FLAG_SEM = 3)."""
    d = gen_ded("tmpf_" + name, mode, thr, abl)  # a scratch dir: kernels_ded_<name> stays intact
    newd = os.path.join(HERE, "kernels_dedf_" + name + (("_" + abl) if abl else ""))
    shutil.rmtree(newd, ignore_errors=True)
    os.rename(d, newd)
    for fn, tail in (("tilize_writer.cpp", DEDF_WRITER_TAIL), ("tilize_reader.cpp", DEDF_READER_TAIL)):
        f = os.path.join(newd, fn)
        t = sub(open(f).read(), "void kernel_main() {\n", "void kernel_main_body() {\n")
        open(f, "w").write("#define HOP_FLAG_SEM 3\n" + t + tail)
    return newd


# ---- dedg: the GRADUATION candidate (host-gated CT arg; see host_patch.install_gated) ----------
DEDG_MASK = r"""
// ---------------------------------------------------------------------------
// Hop-aware write NoC (hop_aware_noc, Perf 2)
// ---------------------------------------------------------------------------
// The writer (BRISC) sends DRAM bank b's tile writes on NoC0 instead of its own NoC1 when NoC0's
// core -> bank path is at least `min_saving` hops shorter (the write data rides the request).
// Wormhole routes NoC0 east then south and NoC1 west then north on a 10 x 12 torus. Coordinates
// are physical NoC0 ones: this Tensix core's from NIU 0's NOC_NODE_ID register (my_x / my_y hold
// the translated id, 18.. on Wormhole) and the bank's from dram_bank_to_noc_xy[0] (DRAM is not
// translated). Bit b of the result set = bank b goes on NoC0.
template <uint32_t min_saving>
FORCE_INLINE uint32_t noc0_write_bank_mask() {
    constexpr uint32_t GX = 10, GY = 12;  // Wormhole NoC grid
    const uint32_t node_id = *reinterpret_cast<volatile uint32_t*>(NOC_NODE_ID);
    const uint32_t mx = node_id & NOC_NODE_ID_MASK, my = (node_id >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    uint32_t mask = 0;
    for (uint32_t b = 0; b < NUM_DRAM_BANKS; ++b) {
        const uint32_t xy = dram_bank_to_noc_xy[0][b] >> NOC_COORD_REG_OFFSET;
        const uint32_t x = xy & NOC_NODE_ID_MASK, y = (xy >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
        const uint32_t east = x >= mx ? x - mx : x + GX - mx, south = y >= my ? y - my : y + GY - my;
        const uint32_t west = east == 0 ? 0 : GX - east, north = south == 0 ? 0 : GY - south;
        if (west + north >= east + south + min_saving) {
            mask |= 1u << b;
        }
    }
    return mask;
}
"""


def gen_dedg(name="hop"):
    d = os.path.join(HERE, "kernels_dedg_" + name)
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(SRC, d)
    r = open(os.path.join(d, "tilize_reader.cpp")).read()
    h = open(os.path.join(d, "tilize_stick_reads.hpp")).read()
    w = open(os.path.join(d, "tilize_writer.cpp")).read()
    # header: the mask, and store_rows' per-bank NoC choice
    h = sub(
        h,
        "// One position of the per-core walk.\n",
        DEDG_MASK.lstrip("\n") + "\n// One position of the per-core walk.\n",
    )
    h = sub(
        h,
        """// noc_split-th write of a tile-row goes out on the other NoC (DM_DYNAMIC_NOC).
template <
    uint32_t cb_output_tiles,
    uint32_t block_width,
    uint32_t out_tile_bytes,
    uint32_t noc_split = 0,
    typename Accessor,
    typename Walk>
FORCE_INLINE void store_rows(
    const Accessor& accessor, Walk& walk, uint32_t tiles_per_row, uint32_t num_rows, uint32_t col_rotation) {""",
        """// noc_split-th write of a tile-row goes out on the other NoC (DM_DYNAMIC_NOC). `hop_t` (0 = off):
// page p's write goes on the other NoC iff bit (p mod NUM_DRAM_BANKS) of `other_noc_banks` is set
// (noc0_write_bank_mask; DM_DEDICATED_NOC, see the writer's kernel head).
template <
    uint32_t cb_output_tiles,
    uint32_t block_width,
    uint32_t out_tile_bytes,
    uint32_t noc_split = 0,
    uint32_t hop_t = 0,
    typename Accessor,
    typename Walk>
FORCE_INLINE void store_rows(
    const Accessor& accessor,
    Walk& walk,
    uint32_t tiles_per_row,
    uint32_t num_rows,
    uint32_t col_rotation,
    uint32_t other_noc_banks = 0) {""",
    )
    h = sub(
        h,
        "                const uint8_t noc = (noc_split != 0 && (n % noc_split) == noc_split - 1) ? 1 - noc_index : noc_index;",
        """                uint8_t noc = (noc_split != 0 && (n % noc_split) == noc_split - 1) ? 1 - noc_index : noc_index;
                if constexpr (hop_t != 0) {
                    noc = ((other_noc_banks >> ((row_tile_idx + t) % NUM_DRAM_BANKS)) & 1) ? 1 - noc_index : noc_index;
                }""",
    )
    h = sub(
        h,
        """        noc_async_writes_flushed();
        if constexpr (noc_split != 0) {
            noc_async_writes_flushed(1 - noc_index);""",
        """        noc_async_writes_flushed();
        if constexpr (noc_split != 0 || hop_t != 0) {
            noc_async_writes_flushed(1 - noc_index);""",
    )
    # writer
    w = sub(
        w,
        "    constexpr auto input_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();\n",
        """    constexpr auto input_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    // Hop-aware write NoC (0 = off, else the minimum hop saving that moves a bank to NoC0).
    constexpr uint32_t hop_t = get_compile_time_arg_val(input_args.next_compile_time_args_offset());
    constexpr uint32_t hop_sem = get_compile_time_arg_val(input_args.next_compile_time_args_offset() + 1);
    static_assert(hop_t == 0 || (!split_reader && !output_resident && write_noc_split == 0), "store_rows only");
""",
    )
    w = sub(
        w,
        "    const auto output_accessor = TensorAccessor(output_args, dst_addr, out_tile_bytes);\n",
        """    const auto output_accessor = TensorAccessor(output_args, dst_addr, out_tile_bytes);
    uint32_t noc0_banks = 0;
    if constexpr (hop_t != 0) {
        // DM_DEDICATED_NOC, BRISC writing on NoC0 too: NCRISC never issues a NoC write (its reads
        // use NIU 0's read command buffer, these writes the write buffer NCRISC leaves unused),
        // so NIU 0's write counters are BRISC's alone; firmware synced only NoC1's.
        noc_local_state_init(1 - noc_index);
        noc0_banks = tilize_dataflow::noc0_write_bank_mask<hop_t>();
    }
""",
    )
    w = sub(
        w,
        """            tilize_dataflow::store_rows<cb_output_tiles, block_width, out_tile_bytes, write_noc_split>(
                output_accessor,
                store_walk,
                tiles_per_row,
                remaining < rows_per_quantum ? remaining : rows_per_quantum,
                stick_rotation);""",
        """            tilize_dataflow::store_rows<cb_output_tiles, block_width, out_tile_bytes, write_noc_split, hop_t>(
                output_accessor,
                store_walk,
                tiles_per_row,
                remaining < rows_per_quantum ? remaining : rows_per_quantum,
                stick_rotation,
                noc0_banks);""",
    )
    w = sub(
        w,
        """    noc_async_write_barrier();
    if constexpr (write_noc_split != 0) {
        noc_async_write_barrier(1 - noc_index);
    }
}""",
        """    noc_async_write_barrier();
    if constexpr (write_noc_split != 0 || hop_t != 0) {
        noc_async_write_barrier(1 - noc_index);
    }
    if constexpr (hop_t != 0) {
        // Every NoC0 write is ACKed: NCRISC may re-sync its NoC0 counters (tilize_reader.cpp).
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(hop_sem)), 1);
    }
}""",
    )
    # reader: re-sync its NoC0 counters once BRISC's NoC0 writes are ACKed
    r = sub(r, "void kernel_main() {\n", "FORCE_INLINE void load_all() {\n")
    r += """
void kernel_main() {
    load_all();
    constexpr auto input_args = TensorAccessorArgs<30>();
    constexpr uint32_t hop_t = get_compile_time_arg_val(input_args.next_compile_time_args_offset());
    constexpr uint32_t hop_sem = get_compile_time_arg_val(input_args.next_compile_time_args_offset() + 1);
    if constexpr (hop_t != 0) {
        // Hop-aware write NoC: BRISC also writes on this RISC-V's NoC (NIU 0). Once those writes
        // are ACKed, re-snapshot this RISC-V's NoC0 counters from the NIU, so the firmware's
        // kernel-end dedicated-NoC idle check holds.
        volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(hop_sem));
        noc_semaphore_wait(flag, 1);
        noc_semaphore_set(flag, 0);
        noc_local_state_init(noc_index);
    }
}
"""
    for fn, s_ in (("tilize_reader.cpp", r), ("tilize_stick_reads.hpp", h), ("tilize_writer.cpp", w)):
        open(os.path.join(d, fn), "w").write(s_)
    return d


def gen_dedr(name, read_t):
    """Reader twin in DM_DEDICATED_NOC for a resident OUTPUT (BRISC issues nothing): the StickProducer's
    DRAM stick reads of bank b go on NoC1 when its bank -> core response path saves >= read_t hops.
    NCRISC uses NoC1's read command buffer (BRISC's, unused there) and NIU 1's per-trid counters;
    BRISC re-syncs its NoC1 counters after NCRISC's flag. Host (install_dedr) sets read_noc_split = 1
    (reader CT 17, the StickProducer's two-NoC barrier path) without DM_DYNAMIC_NOC, and appends
    [on, sem] to both kernels' CT args."""
    d = os.path.join(HERE, "kernels_dedr_" + name)
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(SRC, d)
    defs = f"#define RD_HOP_T {read_t}\n#define HOP_MODE 0\n"
    r = open(os.path.join(d, "tilize_reader.cpp")).read()
    h = open(os.path.join(d, "tilize_stick_reads.hpp")).read()
    w = open(os.path.join(d, "tilize_writer.cpp")).read()
    h = sub(h, "namespace tilize_dataflow {\n", HOP_HELPER + "\nnamespace tilize_dataflow {\n")
    h = sub(
        h,
        "            const uint8_t noc = (noc_split != 0 && (s % noc_split) == noc_split - 1) ? 1 - noc_index : noc_index;",
        "            const uint8_t noc = (noc_split != 0) ? hop_noc::pick_read(first_stick + stick) : noc_index;",
    )
    r = sub(r, "void kernel_main() {\n", "FORCE_INLINE void load_all() {\n")
    r += """
void kernel_main() {
    constexpr auto input_args = TensorAccessorArgs<30>();
    constexpr bool hop_r = get_compile_time_arg_val(input_args.next_compile_time_args_offset()) != 0;
    constexpr uint32_t hop_sem = get_compile_time_arg_val(input_args.next_compile_time_args_offset() + 1);
    if constexpr (hop_r) {
        noc_local_state_init(1 - noc_index);
        hop_noc::g_rmask = hop_noc::compute_rmask();
    }
    load_all();
    if constexpr (hop_r) {
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(hop_sem)), 1);
    }
}
"""
    w = sub(w, "void kernel_main() {\n", "FORCE_INLINE void store_all() {\n")
    w += """
void kernel_main() {
    store_all();
    constexpr auto output_args = TensorAccessorArgs<20>();
    constexpr auto input_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr bool hop_r = get_compile_time_arg_val(input_args.next_compile_time_args_offset()) != 0;
    constexpr uint32_t hop_sem = get_compile_time_arg_val(input_args.next_compile_time_args_offset() + 1);
    if constexpr (hop_r) {
        volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(hop_sem));
        noc_semaphore_wait(flag, 1);
        noc_semaphore_set(flag, 0);
        noc_local_state_init(noc_index);
    }
}
"""
    for fn, s_ in (("tilize_reader.cpp", r), ("tilize_stick_reads.hpp", h), ("tilize_writer.cpp", w)):
        open(os.path.join(d, fn), "w").write(defs + s_)
    return d


def gen_head_abl(abl):
    """Ablation of the unmodified kernels (dedicated-NoC baseline)."""
    d = os.path.join(HERE, "kernels_head_" + abl)
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(SRC, d)
    defs = "".join(f"#define ABL_{x} 1\n" for x in abl)
    r = open(os.path.join(d, "tilize_reader.cpp")).read()
    h = open(os.path.join(d, "tilize_stick_reads.hpp")).read()
    c = open(os.path.join(d, "tilize_compute.cpp")).read()
    w = open(os.path.join(d, "tilize_writer.cpp")).read()
    r, h, c = apply_ablation(r, h, c, abl)
    for fn, s in (
        ("tilize_reader.cpp", r),
        ("tilize_stick_reads.hpp", h),
        ("tilize_compute.cpp", c),
        ("tilize_writer.cpp", w),
    ):
        open(os.path.join(d, fn), "w").write(defs + s)
    return d


if __name__ == "__main__":
    made = []
    for abl in ("", "RSC", "C"):
        if abl:
            made.append(gen_head_abl(abl))
        for name, mode in MODES.items():
            made.append(gen_dyn("dyn_" + name, mode, abl))
        for t in THRESHOLDS:
            made.append(gen_dyn(f"dyn_hopT{t}", 6, abl, thr=t))
        made.append(gen_dyn("dyn_rand28", 7, abl, thr=72))  # 72 / 256 = 28 %, T6's NoC0 share
        made.append(gen_dyn("dyn_antiT6", 8, abl, thr=6))  # 28 %, the mirror pairs
    made.append(gen_dyn("dyn_hop_debug", 1, "", debug=True))
    for t in (4, 6, 8):
        made.append(gen_twin(f"dyn_rdT{t}", 0, None, t))  # reader twin only (coalesced path)
        made.append(gen_twin(f"dynr_rdT{t}", 0, None, t))  # reader twin only (StickProducer path)
        made.append(gen_twin(f"dyn_rwT{t}", 6, t, t))  # both twins, same pairs swapped
    # duo (two RISC-Vs, dedicated NoCs): own = BRISC writes everything (duo sync cost control),
    # all0 = NCRISC writes everything on NoC0, rand = two issuers without hop awareness.
    for abl in ("", "C"):
        for name, mode in (("own", 0), ("hop", 1), ("anti", 2), ("rand", 3), ("all0", 5)):
            made.append(gen_duo(name, mode, abl=abl))
        for t in (2, 4, 6, 8):
            made.append(gen_duo(f"hopT{t}", 6, t, abl=abl))
    # ded (BRISC on both NoCs, dedicated mode): own = mask 0 (control), hopT6 / hopT8, rand28 share control
    for abl in ("", "C", "RSC"):
        made.append(gen_ded("own", 0, abl=abl))
        made.append(gen_ded("rand28", 7, 72, abl=abl))
        for t in (4, 6, 8, 10):
            made.append(gen_ded(f"hopT{t}", 6, t, abl=abl))
    for t in (6, 8):
        made.append(gen_dedf(f"hopT{t}", 6, t))
    made.append(gen_dedf("own", 0))
    made.append(gen_dedg())
    for t in (2, 4, 6):
        made.append(gen_dedr(f"rdT{t}", t))
    for t in (2, 99):  # 99 = empty read mask: the DM_DYNAMIC_NOC cost control of the StickProducer path
        made.append(gen_twin(f"dynr_rdT{t}", 0, None, t))
    print("\n".join(os.path.basename(m) for m in made))
