"""hop_aware_coread: build the graduation candidate from the op's CURRENT files (never edits them).

Writes graduate/kernels/ (the op's kernels + the geometric co-read split) and
graduate/tilize_program_descriptor.py (the op's host + the stick lists + the new gate), then
graduate/kernels.patch and graduate/host.patch (unified diffs against the real op files, i.e. the
exact change that would graduate). Re-run after the op changes; the substitutions assert.
"""
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OP = os.path.normpath(os.path.join(HERE, "../.."))
OUT = os.path.join(HERE, "graduate")


def sub(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)


LISTED = """
// Co-read stick lists (Perf 2, hop_aware_coread). Which sticks of the tile-row each RISC-V reads is
// chosen on the host by DRAM-bank geometry (tilize_program_descriptor._co_read_lists): a read's DATA
// path is bank -> core, east/south on NoC0 but west/north on NoC1, so the same bank can be a few
// hops or most of the torus away depending on the NoC. The positional R8 cut sent half the sticks
// down the long NoC1 path (BRISC's 16 reads took 2x NCRISC's 32 at 1 KiB segments).
//
// The host cannot see physical Tensix coordinates (the bindings return translated ones, and WH
// row harvesting shifts the physical row), so it precomputes the lists for each physical row this
// Tensix core can be on and the kernel picks by its own NoC0 node id. RT layout at `arg`: one word
// of candidate physical NoC0 rows (row c in byte c, up to 3; unused bytes 0xFF), then one list block
// per candidate of 1 + ceil(tile_h / 4) words: n, then n stick indices (0..tile_h-1) packed 4 per
// word, low byte first, in issue order. Both RISC-Vs read the same NoC0 register, so they always
// pick complementary lists. Compiled in only under CO_READ_LISTED (the host keeps R8's positional
// cut, with HEAD's exact binaries, where its model predicts less than the list path's own overhead
// to gain: CO_READ_LIST_MIN_GAIN_CYCLES). Raw register read (NOC_CMD_BUF_READ_REG(NOC_NODE_ID)): no dataflow API returns
// the physical coordinate.
template <uint32_t tile_h>
FORCE_INLINE uint32_t select_co_read_list(uint32_t arg) {
    constexpr uint32_t block_words = 1 + (tile_h + 3) / 4;
    const uint32_t rows = get_arg_val<uint32_t>(arg);
    const uint32_t my_row = (NOC_CMD_BUF_READ_REG(0, 0, NOC_NODE_ID) >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    const uint32_t j = ((rows >> 8) & 0xFF) == my_row ? 1 : (((rows >> 16) & 0xFF) == my_row ? 2 : 0);
    return arg + 1 + j * block_words;
}

// The stick reads of ONE tile-row named by a co-read list (select_co_read_list). The packed words
// are loaded once per 4 sticks and unpacked in registers, so the per-read work matches
// read_tile_row_sticks' (measured: a per-read L1 byte load costs ~8 cycles per read, a bit-mask walk
// ~5.5 cycles per step -- both visible on the issue-bound one-position walks).
template <uint32_t tile_h, uint32_t block_stick_bytes, uint32_t page_bytes, uint32_t pages_per_stick, typename Accessor>
FORCE_INLINE void read_tile_row_sticks_listed(
    const Accessor& accessor,
    uint32_t first_stick,
    uint32_t l1_base,
    uint32_t segment_offset,
    uint32_t segment_bytes,
    uint32_t list_arg) {
    auto read_one = [&](uint32_t stick) {
        const uint32_t l1_dst = l1_base + stick * block_stick_bytes;
        if constexpr (pages_per_stick == 1) {
            noc_async_read(accessor.get_noc_addr(first_stick + stick, segment_offset), l1_dst, segment_bytes);
        } else {
            read_paged_segment<page_bytes, pages_per_stick>(
                accessor, first_stick + stick, segment_offset, l1_dst, segment_bytes);
        }
    };
    const uint32_t n = get_arg_val<uint32_t>(list_arg);
    const uint32_t full = n >> 2;
    for (uint32_t w = 0; w < full; ++w) {
        uint32_t word = get_arg_val<uint32_t>(list_arg + 1 + w);
#pragma GCC unroll 4
        for (uint32_t k = 0; k < 4; ++k, word >>= 8) {
            read_one(word & 0xFF);
        }
    }
    if ((n & 3) != 0) {
        uint32_t word = get_arg_val<uint32_t>(list_arg + 1 + full);
        for (uint32_t k = 0; k < (n & 3); ++k, word >>= 8) {
            read_one(word & 0xFF);
        }
    }
}

// Co-read (Refinement 8, CO_READ_SHARE)"""


def gen_kernels():
    d = os.path.join(OUT, "kernels")
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(os.path.join(OP, "kernels"), d)
    h = open(os.path.join(d, "tilize_stick_reads.hpp")).read()
    h = sub(h, "\n// Co-read (Refinement 8, CO_READ_SHARE)", LISTED)
    h = sub(
        h,
        "    uint32_t co_read = 0>\nstruct StickProducer {",
        "    uint32_t co_read = 0,\n    bool co_read_listed = false>\nstruct StickProducer {",
    )
    h = sub(
        h,
        "    uint32_t stick_stride_bytes = 0;\n",
        "    uint32_t stick_stride_bytes = 0;\n"
        "    uint32_t co_read_list_arg = 0;  // co_read_listed: RT arg index of NCRISC's stick list\n",
    )
    h = sub(
        h,
        """        } else {
            read_tile_row_sticks<tile_h, block_stick_bytes, page_bytes, pages_per_stick, noc_split>(
                accessor, first_stick, l1_base, segment_offset, segment_bytes, stick_rotation, 0, tile_h - co_read);
        }""",
        """        } else if constexpr (co_read_listed) {
            read_tile_row_sticks_listed<tile_h, block_stick_bytes, page_bytes, pages_per_stick>(
                accessor, first_stick, l1_base, segment_offset, segment_bytes, co_read_list_arg);
        } else {
            read_tile_row_sticks<tile_h, block_stick_bytes, page_bytes, pages_per_stick, noc_split>(
                accessor, first_stick, l1_base, segment_offset, segment_bytes, stick_rotation, 0, tile_h - co_read);
        }""",
    )
    open(os.path.join(d, "tilize_stick_reads.hpp"), "w").write(h)

    r = open(os.path.join(d, "tilize_reader.cpp")).read()
    r = sub(
        r,
        "    constexpr auto input_args = TensorAccessorArgs<30>();\n",
        "    constexpr auto input_args = TensorAccessorArgs<30>();\n"
        "#ifdef CO_READ_LISTED\n"
        "    constexpr bool co_read_listed = co_read != 0;  // Perf 2: the host's geometric stick lists\n"
        "#else\n"
        "    constexpr bool co_read_listed = false;\n"
        "#endif\n",
    )
    r = sub(
        r,
        "        co_read>\n        producer(stick_rotation);",
        "        co_read,\n        co_read_listed>\n        producer(stick_rotation);",
    )
    r = sub(
        r,
        "    producer.prime(input_accessor, row_start * tile_h, stick_page_bytes);\n",
        "    producer.prime(input_accessor, row_start * tile_h, stick_page_bytes);\n"
        "    if constexpr (co_read_listed) {\n"
        "        // RT arg 10..: the co-read stick lists (co-read excludes the padded path, the only other\n"
        "        // user of RT args >= 10)\n"
        "        producer.co_read_list_arg = tilize_dataflow::select_co_read_list<tile_h>(10);\n"
        "    }\n",
    )
    open(os.path.join(d, "tilize_reader.cpp"), "w").write(r)

    w = open(os.path.join(d, "tilize_writer.cpp")).read()
    orig_call = """        tilize_dataflow::read_tile_row_sticks<tile_h, block_width * tile_col_bytes, page_bytes, pages_per_stick>(
            TensorAccessor(input_args, src_addr, stick_page_bytes),
            load_walk.row() * tile_h,
            get_write_ptr(cb_input_sticks),
            load_walk.first_col() * tile_col_bytes,
            load_walk.valid_width() * tile_col_bytes,
            stick_rotation & (tile_h - 1),
            tile_h - co_read,
            tile_h);"""
    w = sub(
        w,
        orig_call,
        """#ifdef CO_READ_LISTED
        // Perf 2: the host's geometric stick list for this RISC-V (RT arg 9..)
        tilize_dataflow::read_tile_row_sticks_listed<tile_h, block_width * tile_col_bytes, page_bytes, pages_per_stick>(
            TensorAccessor(input_args, src_addr, stick_page_bytes),
            load_walk.row() * tile_h,
            get_write_ptr(cb_input_sticks),
            load_walk.first_col() * tile_col_bytes,
            load_walk.valid_width() * tile_col_bytes,
            tilize_dataflow::select_co_read_list<tile_h>(9));
#else
"""
        + orig_call
        + """
#endif""",
    )
    w = sub(
        w,
        "// co_read sticks of the tile-row into cb_input_sticks' slot (NCRISC reads the rest and stays\n"
        "// the CB's only producer), then raises the landed flag NCRISC waits on before its push.\n",
        "// co_read sticks of the tile-row into cb_input_sticks' slot (NCRISC reads the rest and stays\n"
        "// the CB's only producer), then raises the landed flag NCRISC waits on before its push. Which\n"
        "// sticks is the host's geometric split (Perf 2: tilize_stick_reads.hpp select_co_read_list).\n",
    )
    open(os.path.join(d, "tilize_writer.cpp"), "w").write(w)


HOST_KNOBS = '''CO_READ_SEGMENT_BYTES = {ttnn.BufferType.DRAM: (0, 256), ttnn.BufferType.L1: (0, None)}
# A resident output (compute packs into this Tensix core's own shard) issues no NoC writes, so the
# writer RISC-V's NoC1 carries only its co-read share: co-read then engages at ANY segment size
# (DRAM -> HEIGHT_SHARDED L1 [1,1,2048,W], 64 cores: 128 B -24 %, 256 B -17 %, 512 B -8 %,
# 1 KiB -11 % (LOOSE_CASES[8]), 2 KiB -12 %). With NoC-written output, past 256 B the op is at the
# DRAM roofline and the writer's own NoC1 tile writes collide with its reads: 512 B..2 KiB measured
# +0..6 % (DRAM out) and +6..14 % (L1-interleaved out), so the window stays 256 B there.
CO_READ_RESIDENT_OUTPUT_UNBOUNDED = True
# A 2-D split (several column groups per tile-row) makes those Tensix cores read segments of the SAME
# tile_h sticks, so the op's reads pile onto that tile-row's banks (tile_h = 32 sticks over 12 banks:
# 8 banks carry 3) and past 128 B it is bank-bound: co-read at 256 B measured +2..6 %
# ([1,1,32,8192] = LOOSE_CASES[4], [1,1,256,1024]) where a row split wins -9..-14 % ([1,1,2048,128]).
CO_READ_SHARED_STICK_MAX_BYTES = 128
# Without the geometric lists (not WH, an L1 / sharded / paged input, or a light-load walk the model
# keeps positional) R8's positional DRAM window applies unchanged.
CO_READ_POSITIONAL_DRAM_MAX_BYTES = 128

# ---- Perf 2 (hop_aware_coread): WHICH sticks the writer RISC-V co-reads, by DRAM-bank geometry.
# A read's data path is bank -> core: NoC0 routes east then south, NoC1 west then north, on a
# 10 x 12 torus (WH), and DRAM banks sit in physical columns x = 0 and x = 5. The request travels
# the other way round the same rings, so request + response is the full loop on either NoC; only
# the data path's length differs, and under load that is what costs. R8's positional cut (BRISC
# reads the last half of the rotated order) sends ~half its sticks the long way: at 1 KiB segments
# BRISC's 16 NoC1 reads took ~15.6 k cycles vs NCRISC's 32 NoC0 reads in ~8.3 k (zones,
# LOOSE_CASES[8]), which is why R8 had to gate DRAM co-read to <= 128 B. Inverting the preference at
# the same split sizes costs +24..80 % over the preferred split: the geometry is the lever.
# The split (_co_read_split): BRISC takes the k sticks whose NoC1 data path is most shorter, k
# minimizing  max(n_ncrisc, n_brisc) * CO_READ_ISSUE_CYCLES              (issue chain per RISC-V)
#           + w * (sum of data-path hops + CO_READ_NOC1_PENALTY_HOPS * n_brisc)  (shared link load)
# with w = CO_READ_HOP_WEIGHT * flits(segment) * active_cores / 64. Light load (few cores, 64-B
# segments) -> the balanced 16 / 16 cut (the issue chain is the op there); heavy load -> near-pure
# geometry. The NoC1 penalty is measured: sticks with EQUAL hops are cheaper on NoC0 (sending the
# ties to NoC1 cost +13 % on LOOSE_CASES[8]).
# Measured, this change vs R8 (WH B0 n150, bf16 unless noted, device-kernel ns, medians of 6
# same-session A/Bs; identical-program pairs spread -5.5..+4.8 %, the noise floor):
#   LOOSE_CASES[8] (DRAM -> HEIGHT_SHARDED L1, 1 KiB) -10 %   [1,1,2048,64] -5 %   [1,1,2048,128] -9 %
#   [1,1,2048,W] -> HEIGHT_SHARDED L1: W=64 -24 %, W=256 -6 %, W=1024 -12 %   fp32 [1,1,2048,32] -12 %
#   [1,1,2048,128] -> L1 -7 %   [1,1,32,4096] -6 %   LOOSE_CASES[3] / [4] / [5] / [0]: flat.
# CO_READ_SPLIT = "positional" restores R8's cut (same kernels, lists = the positional steps).
CO_READ_SPLIT = "geometry"
CO_READ_ISSUE_CYCLES = 45  # one stick-read issue on one RISC-V (NCRISC reader_issue zone: ~708 cycles / 16)
CO_READ_HOP_WEIGHT = 2.0  # cycles per 32-B flit-hop of data path at full-grid load
CO_READ_NOC1_PENALTY_HOPS = 2
_WH_NOC_GRID = (10, 12)  # NoC0 torus (x, y)
_WH_TRANSLATED_ORIGIN = 18  # first translated Tensix x / y (worker_core_from_logical_core)
_WH_TENSIX_X = (1, 2, 3, 4, 6, 7, 8, 9)  # translated x - 18 -> physical NoC0 x (WH harvests rows only)
_WH_TENSIX_ROWS = (1, 2, 3, 4, 5, 7, 8, 9, 10, 11)  # physical NoC0 rows a Tensix row can sit on
_CO_READ_ROW_CANDIDATES = 3  # up to 2 harvested rows above a Tensix row -> its row is one of 3
# The list path's own cost vs R8's contiguous loop (zones, [1,1,128,64]): ~4 cycles per listed read,
# ~65 cycles of physical-row select and ~90 cycles of launch (bigger binaries) per RISC-V. When the
# model's mean per-core saving is below this, the program keeps R8's positional cut and HEAD's
# exact binaries (no CO_READ_LISTED define, no list RT args): light-load walks, where any balanced
# cut is equivalent and only the issue chain counts ([1,1,128,64] / [1,1,256,64]: 8 / 16 cores).
CO_READ_LIST_MIN_GAIN_CYCLES = 250


def _co_read_bank_xy(device):
    """Physical NoC0 (x, y) of each DRAM bank (bank b = DRAM view b), or None when the geometric
    split does not apply (not Wormhole: its NoC grid / endpoint tables are the ones above)."""
    if str(device.arch()).lower().split(".")[-1] != "wormhole_b0":
        return None
    n = device.dram_grid_size().x
    return tuple((c.x, c.y) for c in (device.dram_core_from_logical_core(ttnn.CoreCoord(b, 0)) for b in range(n)))


@functools.lru_cache(maxsize=4096)
def _co_read_split(bank_xy, px, py, first_stick, rotation, co_read, tile_h, flits, num_cores):
    """(sequence steps (s -> stick (s + rotation) mod tile_h) the writer RISC-V reads, the model's
    predicted saving in cycles over R8's positional cut)."""
    gx, gy = _WH_NOC_GRID
    hops = []
    for s in range(tile_h):
        dx, dy = bank_xy[(first_stick + (s + rotation) % tile_h) % len(bank_xy)]
        hops.append(((px - dx) % gx + (py - dy) % gy, (dx - px) % gx + (dy - py) % gy))  # (NoC0, NoC1)
    w = CO_READ_HOP_WEIGHT * flits * num_cores / 64

    def cost(brisc):
        total = sum(hops[s][1] + CO_READ_NOC1_PENALTY_HOPS if s in brisc else hops[s][0] for s in range(tile_h))
        return max(len(brisc), tile_h - len(brisc)) * CO_READ_ISSUE_CYCLES + w * total

    order = sorted(range(tile_h), key=lambda s: (hops[s][1] - hops[s][0], s))  # most NoC1-favoured first
    total = sum(h0 for h0, _ in hops)
    best_cost, best_k = None, co_read
    for k in range(tile_h + 1):
        if k:
            s = order[k - 1]
            total += hops[s][1] + CO_READ_NOC1_PENALTY_HOPS - hops[s][0]
        c = (max(k, tile_h - k) * CO_READ_ISSUE_CYCLES + w * total, abs(k - co_read))
        if best_cost is None or c < best_cost:
            best_cost, best_k = c, k
    chosen = frozenset(order[:best_k])
    return chosen, cost(frozenset(range(tile_h - co_read, tile_h))) - cost(chosen)


def _co_read_lists(device, bank_xy, core, first_stick, rotation, co_read, tile_h, segment_bytes, num_cores):
    """(reader RT tail, writer RT tail, modelled saving in cycles over R8's positional cut): the
    stick-list blocks select_co_read_list decodes, one per candidate physical row; the saving is
    the worst candidate's. (None, None, 0) when this Tensix core has no geometry (not WH /
    untranslated coordinates)."""
    words = (tile_h + 3) // 4
    virt = device.worker_core_from_logical_core(core)
    xi, yi = virt.x - _WH_TRANSLATED_ORIGIN, virt.y - _WH_TRANSLATED_ORIGIN
    if bank_xy is None or not (0 <= xi < len(_WH_TENSIX_X) and 0 <= yi < len(_WH_TENSIX_ROWS)):
        return None, None, 0
    px, rows = _WH_TENSIX_X[xi], _WH_TENSIX_ROWS[yi : yi + _CO_READ_ROW_CANDIDATES]
    splits = [
        _co_read_split(bank_xy, px, py, first_stick, rotation, co_read, tile_h, max(1, segment_bytes // 32), num_cores)
        for py in rows
    ]
    packed_rows = 0
    for c in range(_CO_READ_ROW_CANDIDATES):
        packed_rows |= (rows[c] if c < len(rows) else 0xFF) << (8 * c)
    tails = ([packed_rows], [packed_rows])
    for brisc, _ in splits:
        for tail, mine in zip(tails, (False, True)):
            sticks = [(s + rotation) % tile_h for s in range(tile_h) if (s in brisc) == mine]
            packed = [0] * words
            for i, j in enumerate(sticks):
                packed[i // 4] |= j << (8 * (i % 4))
            tail.extend([len(sticks), *packed])
    return tails[0], tails[1], min(gain for _, gain in splits)
'''


def gen_host():
    src = os.path.join(OP, "tilize_program_descriptor.py")
    s = open(src).read()
    s = sub(s, "import os\nfrom pathlib import Path\n", "import functools\nimport os\nfrom pathlib import Path\n")
    s = sub(
        s,
        "# The writer's share travels on NoC1, which reads DRAM poorly: past ~128 B a DRAM read is no\n"
        "# longer issue-bound and the NoC1 half loses (192 B +4.6 %, 256 B +3..6 %, 512 B..2 KiB +26..31 %,\n"
        "# medians of 3). An L1 source",
        "# R8 measured the POSITIONAL split losing past 128 B on DRAM (192 B +4.6 %, 256 B +3..6 %,\n"
        "# 512 B..2 KiB +26..31 %): its NoC1 half took the long data path (Perf 2, below); with the\n"
        "# geometric split the DRAM window is 256 B (unbounded for a resident output). An L1 source",
    )
    s = sub(
        s,
        "CO_READ_SEGMENT_BYTES = {ttnn.BufferType.DRAM: (0, 128), ttnn.BufferType.L1: (0, None)}",
        HOST_KNOBS,
    )
    s = sub(
        s,
        "    co_read_min, co_read_max = CO_READ_SEGMENT_BYTES.get(in_mc.buffer_type, (0, -1))\n",
        "    co_read_min, co_read_max = CO_READ_SEGMENT_BYTES.get(in_mc.buffer_type, (0, -1))\n"
        "    # Perf 2: the geometric split applies to a DRAM-interleaved input with one page per stick\n"
        "    # (stick s lives in bank s mod num_banks) on a board whose geometry is known.\n"
        "    co_read_geometric = (\n"
        '        CO_READ_SPLIT == "geometry"\n'
        "        and in_mc.buffer_type == ttnn.BufferType.DRAM\n"
        "        and in_mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED\n"
        "        and pages_per_stick == 1\n"
        "    )\n"
        "    if in_mc.buffer_type == ttnn.BufferType.DRAM:\n"
        "        if output_resident and CO_READ_RESIDENT_OUTPUT_UNBOUNDED:\n"
        "            co_read_max = None  # no NoC writes: the writer RISC-V's NoC1 carries only its reads\n"
        "        elif not all(col_start == 0 and cols == C for _, _, _, col_start, cols in assignment):\n"
        "            co_read_max = min(co_read_max, CO_READ_SHARED_STICK_MAX_BYTES)\n",
    )
    s = sub(
        s,
        "        co_read = 0\n"
        "    coalesce_row_bytes = BANK_COALESCE_STAGE_DEPTH * tile_h * stick_page_bytes  # staging per tile-row\n",
        "        co_read = 0\n"
        "    # Perf 2: the per-core stick lists, sent (CO_READ_LISTED) only where the model's mean saving\n"
        "    # beats the list path's own cost. Without them a DRAM segment past R8's positional window\n"
        "    # (CO_READ_POSITIONAL_DRAM_MAX_BYTES) is not co-read at all.\n"
        "    co_read_lists = {}  # (x, y) -> (reader RT tail, writer RT tail)\n"
        "    co_read_banks = _co_read_bank_xy(device) if co_read and co_read_geometric else None\n"
        "    if co_read_banks is not None:\n"
        "        tails = []\n"
        "        for core_idx, (core, row_start, *_) in enumerate(assignment):\n"
        "            reader_tail, writer_tail, gain = _co_read_lists(\n"
        "                device,\n"
        "                co_read_banks,\n"
        "                core,\n"
        "                row_start * tile_h,  # one position per core: its tile-row is row_start\n"
        "                core_idx % tile_h,  # the RT loop's stick_rotation\n"
        "                co_read,\n"
        "                tile_h,\n"
        "                segment_bytes,\n"
        "                len(assignment),\n"
        "            )\n"
        "            tails.append((core, reader_tail, writer_tail, gain))\n"
        "        if all(t[1] is not None for t in tails) and (\n"
        "            sum(t[3] for t in tails) / len(tails) >= CO_READ_LIST_MIN_GAIN_CYCLES\n"
        "        ):\n"
        "            co_read_lists = {(core.x, core.y): (rt, wt) for core, rt, wt, _ in tails}\n"
        "    if (\n"
        "        co_read\n"
        "        and not co_read_lists\n"
        "        and in_mc.buffer_type == ttnn.BufferType.DRAM\n"
        "        and segment_bytes > CO_READ_POSITIONAL_DRAM_MAX_BYTES\n"
        "    ):\n"
        "        co_read = 0\n"
        "    coalesce_row_bytes = BANK_COALESCE_STAGE_DEPTH * tile_h * stick_page_bytes  # staging per tile-row\n",
    )
    s = sub(
        s,
        "        compute_rt_args[core.x][core.y] = [core_row_tiles, core_col_tiles]\n",
        "        if co_read_lists:\n"
        "            reader_tail, writer_tail = co_read_lists[(core.x, core.y)]\n"
        "            reader_rt_args[core.x][core.y].extend(reader_tail)\n"
        "            writer_rt_args[core.x][core.y].extend(writer_tail)\n"
        "        compute_rt_args[core.x][core.y] = [core_row_tiles, core_col_tiles]\n",
    )
    s = _host_after_loop(s)
    open(os.path.join(OUT, "tilize_program_descriptor.py"), "w").write(s)


def _host_after_loop(s):
    s = sub(
        s,
        "    reader_kernel = ttnn.KernelDescriptor(\n",
        '    co_read_defines = [("CO_READ_LISTED", "1")] if co_read_lists else []\n'
        "    reader_kernel = ttnn.KernelDescriptor(\n",
    )
    s = s.replace(
        """        kernel_source=str(KERNEL_DIR / "tilize_reader.cpp"),
        defines=_kernel_defines(),""",
        """        kernel_source=str(KERNEL_DIR / "tilize_reader.cpp"),
        defines=_kernel_defines() + co_read_defines,""",
    )
    s = s.replace(
        """        kernel_source=str(KERNEL_DIR / "tilize_writer.cpp"),
        defines=_kernel_defines(),""",
        """        kernel_source=str(KERNEL_DIR / "tilize_writer.cpp"),
        defines=_kernel_defines() + co_read_defines,""",
    )
    assert s.count("co_read_defines,") == 2
    return s


def patches():
    rel_op = os.path.relpath(OP, os.path.join(OP, "../../../.."))
    with open(os.path.join(OUT, "kernels.patch"), "w") as f:
        subprocess.run(["diff", "-ru", os.path.join(OP, "kernels"), os.path.join(OUT, "kernels")], stdout=f)
    with open(os.path.join(OUT, "host.patch"), "w") as f:
        subprocess.run(
            [
                "diff",
                "-u",
                os.path.join(OP, "tilize_program_descriptor.py"),
                os.path.join(OUT, "tilize_program_descriptor.py"),
            ],
            stdout=f,
        )
    root = os.path.normpath(os.path.join(OP, "../../../.."))
    for p in ("kernels.patch", "host.patch"):
        fp = os.path.join(OUT, p)
        t = open(fp).read().replace(root + "/", "")
        open(fp, "w").write(t)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    gen_kernels()
    gen_host()
    patches()
    print("wrote", OUT)
