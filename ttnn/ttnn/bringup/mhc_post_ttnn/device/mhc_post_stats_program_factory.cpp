// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.bringup.mhc_post(..., stats_weight=W): kernels/mhc_post_stats_{reader,writer,compute}.cpp.
//
// Work: column blocks of B = cols_per_block tiles (the plan, mhc_post_stats_device_operation.cpp); a unit is one block
// b of one token tile-row r, units block-major (u = b * RT + r) split contiguously over the grid, so a core keeps the
// same W block (n * B tiles) in L1 across its units. Per unit the core mixes the B column tiles of all n output
// streams (fp32 SFPU, the mhc_post arithmetic), writes them, and forms the unit's partial row: X'_block @ W_block on
// the FPU (fp32 DEST) with the block's sum of squares in column n (n + 2); the partial goes to a DRAM scratch page
// (r * NB + b) and increments the semaphore of row r's ROOT core (the owner of unit ((r mod NB) RT + r)). A root,
// after its own units, waits for NB partials per rooted row, sums them in block order (fp32 SFPU) and writes the row.

#include "mhc_post_stats_device_operation.hpp"
#include "../kernels/mhc_post_feed_layout.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::bringup::mhc_post_ttnn {

namespace {

using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;
using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::DataType;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::ProgramDescriptor;
using tt::tt_metal::SemaphoreDescriptor;

constexpr const char* KERNEL_DIR = "ttnn/ttnn/bringup/mhc_post_ttnn/kernels/";
constexpr uint32_t TILE_HW = 32;
constexpr size_t NUM_CIRCULAR_BUFFERS = 64;

constexpr uint8_t CB_IN = 0;       // F + X_0..X_{n-1} tiles of a unit's B columns      reader -> compute
constexpr uint8_t CB_W = 1;        // the W block (n x B tiles)                       reader -> compute
constexpr uint8_t CB_CF = 2;       // coefficient tiles of a unit (top + bottom half)  writer -> compute
constexpr uint8_t CB_RAW = 3;      // raw post / comb row tiles, writer scratch
constexpr uint8_t CB_SOA = 4;      // the unit's sum-of-squares tile, lane = token row writer -> compute
constexpr uint8_t CB_SOA_OUT = 5;  // its row totals (slot 0)                          compute -> writer
constexpr uint8_t CB_PART = 6;     // a row's partials from the scratch (root)         reader -> compute
constexpr uint8_t CB_F = 7;        // bf16 F: the unit's B F tiles (MHC_POST_STATS_F16)  reader -> compute
constexpr uint8_t CB_OUT = 16;     // X' tiles of a unit (n B)                          compute -> writer
constexpr uint8_t CB_XS = 17;      // X' tiles of the unit (matmul in0)                compute -> compute
constexpr uint8_t CB_SSC = 18;     // per-column sum of squares tiles                 compute -> compute
constexpr uint8_t CB_MFIN = 19;    // the unit's partial X' @ W tile                  compute -> writer
constexpr uint8_t CB_SFIN = 20;    // the unit's lane-wise sum-of-squares tile         compute -> writer
constexpr uint8_t CB_FIN = 21;     // a root's summed row                              compute -> writer
// collapse (P.34) only:
constexpr uint8_t CB_PB = 8;         // the unit's pre-block tile (collapse_pre)        writer -> compute
constexpr uint8_t CB_HSOA = 9;       // a root row, coefficient-major                   writer -> compute
constexpr uint8_t CB_HSTAGE = 10;    // the row-major hc stage                         writer scratch
constexpr uint8_t CB_Y = 22;         // the unit's y tiles                              compute -> writer
constexpr uint8_t CB_HSOA_OUT = 23;  // the row's hc, coefficient-major               compute -> writer
// norm (P.35) only:
constexpr uint8_t CB_NR = 11;      // the unit's row rstd tile (from the scratch)        reader -> compute
constexpr uint8_t CB_NG = 12;      // the unit's weight tile (slot 4 c + g)              reader -> compute
constexpr uint8_t CB_NSTAGE = 13;  // a root's rstd tile stage                          writer scratch
constexpr uint8_t CB_GSTAGE = 14;  // the unit's weight row (B x 32 fp32)               reader scratch
constexpr uint8_t CB_NSQ = 24;     // the unit's lane-wise sum of y^2 tile               compute -> writer
constexpr uint8_t CB_YN = 25;      // the unit's normalized y tiles                     compute -> writer
constexpr uint8_t CB_NRS = 15;     // a root row's rstd (slot n (n + 2) + 1)            compute -> writer
// feed (P.36) only, on the port cores:
constexpr uint8_t CB_FEED = 26;  // packet slots (up to P y tiles a slot)             port reader -> port sender
constexpr uint32_t FEED_SLOTS = 8;
constexpr uint32_t FEED_SLOTS_FAST = 16;  // P.83 (fast): the port reader keeps FEED_SLOTS_FAST - 4 packets in flight
constexpr uint32_t SEM_ROOT = 0;
constexpr uint32_t SEM_NORM = 1;
constexpr uint32_t SEM_FEED = 2;  // a port's count of finished units (one increment per (unit, target) it carries)
constexpr uint32_t SEM_FEED_TEARDOWN = 3, SEM_FEED_BUFIDX = 4;  // a port's fabric connection's worker semaphores
constexpr uint32_t SEM_W = 5;  // P.83 (fast, W shared): counts the W tiles the block's other cores forwarded here
constexpr uint32_t SEM_LAND = 6;  // PB.8 (v2, feed landing): a port's landing address (its reader writes it at start)
// PB.8 (v2): the feed landing's L1 never exceeds this (and never the workers' own CB total: the ports then fit where
// the workers' CBs fit)
constexpr uint32_t LAND_BUDGET = 1024 * 1024;

constexpr uint32_t READER_KERNEL = 0, WRITER_KERNEL = 1, COMPUTE_KERNEL = 2;
constexpr uint32_t FEED_READER_KERNEL = 3, FEED_SENDER_KERNEL = 4;

uint32_t f32_bits(double v) { return std::bit_cast<uint32_t>(static_cast<float>(v)); }

// The hc step's scalars in mhc_pre_xing's order (mhc_pre_xing_program_factory.cpp scalar_args; clamp unused: 0).
std::vector<uint32_t> hc_scalar_args(const MhcPostStatsParams& p, uint32_t n) {
    std::vector<uint32_t> a = {
        f32_bits(p.hc_scale[0]),
        f32_bits(p.hc_scale[1]),
        f32_bits(p.hc_scale[2]),
        f32_bits(p.hc_inv_nc),
        f32_bits(p.hc_norm_eps),
        f32_bits(p.hc_eps),
        0u,
        0u,
        p.hc_sinkhorn_iters};
    for (uint32_t k = 0; k < n * (n + 2); ++k) {
        a.push_back(k < p.hc_base.size() ? f32_bits(p.hc_base[k]) : 0u);
    }
    if (p.norm) {  // P.35: after the hc scalars (compute common args 9 + n (n + 2), + 1)
        a.push_back(f32_bits(p.rms_inv_w));
        a.push_back(f32_bits(p.rms_eps));
    }
    return a;
}

CBDescriptor make_cb(
    uint8_t index,
    uint32_t page_bytes,
    uint32_t num_pages,
    const CoreRangeSet& cores,
    DataType dt = DataType::FLOAT32) {
    CBDescriptor cb;
    cb.total_size = num_pages * page_bytes;
    cb.core_ranges = cores;
    cb.format_descriptors.push_back(CBFormatDescriptor{
        .buffer_index = index,
        .data_format = tt::tt_metal::datatype_to_dataformat_converter(dt),
        .page_size = page_bytes});
    return cb;
}

KernelDescriptor make_kernel(
    const char* file,
    const CoreRangeSet& cores,
    std::vector<uint32_t> ct,
    KernelDescriptor::RuntimeArgs rt,
    KernelDescriptor::ConfigDescriptor config) {
    KernelDescriptor k;
    k.kernel_source = std::string(KERNEL_DIR) + file;
    k.source_type = KernelDescriptor::SourceType::FILE_PATH;
    k.core_ranges = cores;
    k.compile_time_args = std::move(ct);
    k.runtime_args = std::move(rt);
    k.config = std::move(config);
    return k;
}

uint32_t addr(const Tensor& t) { return t.buffer()->address(); }

void append_accessor(std::vector<uint32_t>& ct, const Tensor& t) {
    const auto a = tt::tt_metal::TensorAccessorArgs(*t.buffer()).get_compile_time_args();
    ct.insert(ct.end(), a.begin(), a.end());
}

// P.36 (feed): the chip-independent feed table, the common args every feed kernel reads (mhc_post_feed_common.hpp):
// one program for every chip of the mesh; a kernel finds its own chip through the fabric routing table in L1
// (my_device_id), then its position, its group's chips, its ports' sides and fabric channels in these tables.
struct FeedTable {
    std::vector<uint32_t> head;  // FT_WORDS words (the table without the channels)
    std::vector<uint32_t> chan;  // FT_CHAN_WORDS words: the fabric channel (byte) of port slot s of chip pos, pos 8 + s
};

void put_byte(std::vector<uint32_t>& w, uint32_t base, uint32_t k, uint32_t v) {
    w[base + k / 4] = (w[base + k / 4] & ~(0xFFu << (8 * (k % 4)))) | ((v & 0xFFu) << (8 * (k % 4)));
}

// Port s = d L + l (side d: the eth direction of the fabric's first hop toward a target, one of the two on the
// group's axis - on a torus axis the far chips go round the wrap, DS41-224 - and link l) carries, to every target t
// on its side, the units u in [u_lo[t], u_hi[t]) with (u - u_lo[t]) mod L == l, the tiles of column block t.
FeedTable feed_table(
    tt::tt_metal::distributed::MeshDevice* mesh,
    const MhcPostStatsParams& params,
    const std::vector<CoreCoord>& ports,
    uint32_t cpt,
    uint32_t RT,
    uint32_t B,
    uint32_t col_tiles) {
    using namespace mhc_feed_layout;
    const auto& ms = mesh->shape();
    TT_FATAL(ms.dims() == 2, "mhc_post stats: feed needs a 2D mesh, got {} dims", ms.dims());
    const uint32_t R = ms[0], C = ms[1], axis = params.feed_axis, L = params.feed_links;
    TT_FATAL(axis < 2, "mhc_post stats: feed_axis {} of a 2D mesh", axis);
    const uint32_t G = axis == 1 ? C : R;
    TT_FATAL(G >= 2 && G <= FT_MAX_G, "mhc_post stats: feed needs a group of 2..{} chips, got {}", FT_MAX_G, G);
    TT_FATAL(R * C <= FT_MAX_CHIPS, "mhc_post stats: feed: {} chips > {}", R * C, FT_MAX_CHIPS);
    TT_FATAL(L >= 1 && 2 * L <= FT_MAX_PORTS, "mhc_post stats: feed: {} links (1..{})", L, FT_MAX_PORTS / 2);
    std::vector<tt::tt_fabric::FabricNodeId> node;
    for (uint32_t pos = 0; pos < R * C; ++pos) {
        node.push_back(mesh->get_fabric_node_id(ttnn::MeshCoordinate(pos / C, pos % C)));
        TT_FATAL(node.back().mesh_id == node.front().mesh_id, "mhc_post stats: feed: the mesh spans fabric meshes");
        TT_FATAL(node.back().chip_id < FT_MAX_CHIP_ID, "mhc_post stats: feed: fabric chip id {}", node.back().chip_id);
    }
    const auto at = [&](uint32_t pos, uint32_t t) {  // chip pos's group chip at position t
        const uint32_t r = pos / C, c = pos % C;
        return axis == 1 ? r * C + t : t * C + c;
    };
    // the sides: an eth direction per side, the same pattern in every group (checked)
    int32_t dir_of_side[2] = {-1, -1};
    std::vector<int32_t> side(G * G, -1);
    for (uint32_t pos = 0; pos < R * C; ++pos) {
        const uint32_t me = axis == 1 ? pos % C : pos / C;
        for (uint32_t t = 0; t < G; ++t) {
            if (t == me) {
                continue;
            }
            const auto d = tt::tt_fabric::get_eth_forwarding_direction(node[pos], node[at(pos, t)]);
            TT_FATAL(d.has_value(), "mhc_post stats: feed: no fabric route from chip {} to {}", pos, at(pos, t));
            const bool zero =
                *d == tt::tt_fabric::eth_chan_directions::EAST || *d == tt::tt_fabric::eth_chan_directions::NORTH;
            const int32_t sd = zero ? 0 : 1;
            TT_FATAL(
                dir_of_side[sd] < 0 || dir_of_side[sd] == static_cast<int32_t>(*d),
                "mhc_post stats: feed: the group's routes leave by more than two directions");
            dir_of_side[sd] = static_cast<int32_t>(*d);
            TT_FATAL(
                side[me * G + t] < 0 || side[me * G + t] == sd,
                "mhc_post stats: feed: the groups route {} -> {} differently",
                me,
                t);
            side[me * G + t] = sd;
        }
    }
    FeedTable ft;
    ft.head.assign(FT_WORDS, 0);
    auto& h = ft.head;
    h[FT_G] = G;
    h[FT_AXIS] = axis;
    h[FT_C] = C;
    h[FT_L] = L;
    h[FT_MESH] = static_cast<uint32_t>(*node.front().mesh_id);
    for (uint32_t k = 0; k < FT_MAX_CHIP_ID; ++k) {
        put_byte(h, FT_POS_OF_CHIP, k, 0xFF);
    }
    for (uint32_t pos = 0; pos < R * C; ++pos) {
        put_byte(h, FT_POS_OF_CHIP, node[pos].chip_id, pos);
        put_byte(h, FT_CHIP_OF_POS, pos, node[pos].chip_id);
    }
    for (uint32_t k = 0; k < G * G; ++k) {
        h[FT_SIDE + k / 32] |= static_cast<uint32_t>(std::max(side[k], 0)) << (k % 32);
    }
    h[FT_DIR] = static_cast<uint32_t>(std::max(dir_of_side[0], 0));
    h[FT_DIR + 1] = static_cast<uint32_t>(std::max(dir_of_side[1], 0));
    for (uint32_t sl = 0; sl < 2 * L; ++sl) {
        const CoreCoord v = mesh->worker_core_from_logical_core(ports[sl]);
        h[FT_PORTS + sl] = (static_cast<uint32_t>(v.x) << 16) | static_cast<uint32_t>(v.y);
    }
    for (uint32_t t = 0; t < G; ++t) {
        h[FT_ULO + t] = (t * cpt / B) * RT;
        h[FT_UHI + t] = std::min((((t + 1) * cpt + B - 1) / B), (col_tiles + B - 1) / B) * RT;
    }
    // the channels: the link-l channel of side d's direction on each chip (append_fabric_connection_rt_args' choice)
    ft.chan.assign(FT_CHAN_WORDS, 0xFFFFFFFFu);
    for (uint32_t pos = 0; pos < R * C; ++pos) {
        const uint32_t me = axis == 1 ? pos % C : pos / C;
        for (uint32_t d = 0; d < 2; ++d) {
            uint32_t tgt = G;
            for (uint32_t t = 0; t < G && tgt == G; ++t) {
                if (t != me && side[me * G + t] == static_cast<int32_t>(d)) {
                    tgt = t;
                }
            }
            if (tgt == G) {
                continue;
            }
            for (uint32_t l = 0; l < L; ++l) {
                tt::tt_metal::ProgramDescriptor scratch;
                std::vector<uint32_t> a;
                tt::tt_fabric::append_fabric_connection_rt_args<tt::tt_metal::ProgramDescriptor>(
                    *mesh, node[pos], node[at(pos, tgt)], l, scratch, ports[d * L + l], a);
                TT_FATAL(!a.empty() && a[0] < 0xFF, "mhc_post stats: feed: no fabric channel");
                const uint32_t k = pos * FT_MAX_PORTS + d * L + l;
                ft.chan[k / 4] &= ~(0xFFu << (8 * (k % 4)));
                ft.chan[k / 4] |= a[0] << (8 * (k % 4));
            }
        }
    }
    return ft;
}

// P.36 (feed): tiles a fabric packet carries (y's pages)
uint32_t feed_packet_tiles(uint32_t o_page) {
    const size_t max_payload = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    return std::max<uint32_t>(1, std::min<uint32_t>(4, static_cast<uint32_t>(max_payload / o_page)));
}

// PB.8 (v2): the packets of a stream of n tiles (mhc_post_feed_common.hpp packet_size: P a packet, the last tile alone)
uint32_t stream_packets(uint32_t n, uint32_t P) { return n <= 1 ? n : (n - 1 + P - 1) / P + 1; }

// PB.8 (v2): the most packets any port of any chip carries (the landing's slots): port (d, l) of group position me
// carries every target t != me on side d, of each the units u in [u_lo, u_hi) with (u - u_lo) mod L == l, their tiles
// of column block t (the feed table's sides and unit ranges)
uint32_t max_port_packets(const FeedTable& ft, uint32_t RT, uint32_t B, uint32_t col_tiles, uint32_t cpt, uint32_t P) {
    using namespace mhc_feed_layout;
    const auto& h = ft.head;
    const uint32_t G = h[FT_G], L = h[FT_L];
    auto side = [&](uint32_t a, uint32_t b) { return (h[FT_SIDE + (a * G + b) / 32] >> ((a * G + b) % 32)) & 1u; };
    uint32_t best = 0;
    for (uint32_t me = 0; me < G; ++me) {
        for (uint32_t d = 0; d < 2; ++d) {
            for (uint32_t l = 0; l < L; ++l) {
                uint32_t pk = 0;
                for (uint32_t t = 0; t < G; ++t) {
                    if (t == me || side(me, t) != d) {
                        continue;
                    }
                    const uint32_t k_lo = t * cpt, k_hi = k_lo + cpt;
                    uint32_t n = 0;
                    for (uint32_t u = h[FT_ULO + t] + l; u < h[FT_UHI + t]; u += L) {
                        const uint32_t col0 = (u / RT) * B, valid = std::min(B, col_tiles - col0);
                        const uint32_t c0 = col0 < k_lo ? k_lo - col0 : 0;
                        const uint32_t c1 = col0 + valid > k_hi ? k_hi - col0 : valid;
                        n += c1 > c0 ? c1 - c0 : 0;
                    }
                    pk += stream_packets(n, P);
                }
                best = std::max(best, pk);
            }
        }
    }
    return best;
}

// P.36 (feed): the port cores' kernels; every port core of every chip runs both (a side with no target: none); port 0
// also waits for the (G - 1) L arrivals into its chip's feed_output.
void add_feed_ports(
    ProgramDescriptor& desc,
    const MhcPostStatsParams& params,
    const MhcPostStatsInputs& in,
    std::vector<Tensor>& outputs,
    const std::vector<CoreCoord>& ports,
    const FeedTable& ft,
    uint32_t grp,
    uint32_t cpt,
    uint32_t RT,
    uint32_t B,
    uint32_t col_tiles,
    uint32_t land_slots) {
    const Tensor& y = outputs.at(4);
    const Tensor& fo = *in.feed_output;
    const uint32_t L = params.feed_links;
    const uint32_t o_page = y.buffer()->page_size();
    TT_FATAL(fo.buffer()->page_size() == o_page, "mhc_post stats: feed_output must be in y's tiles");
    TT_FATAL(fo.dtype() == y.dtype(), "mhc_post stats: feed_output must have y's dtype");
    TT_FATAL(
        fo.padded_shape()[-2] == grp * RT * TILE_HW && fo.padded_shape()[-1] == cpt * TILE_HW,
        "mhc_post stats: feed_output must be [{} rows, {} columns]",
        grp * RT * TILE_HW,
        cpt * TILE_HW);
    const uint32_t P = feed_packet_tiles(o_page);
    std::vector<CoreRange> rr;
    for (const auto& c : ports) {
        rr.emplace_back(c, c);
    }
    const CoreRangeSet port_set(rr);
    // PB.8 (v2, land_slots > 0): cb_feed is the landing, one slot per packet of the port's busiest stream set (the
    // workers write the tiles in place), then one page of per-unit flags (a unit's tiles have landed; the reader pushes
    // each packet once its units' flags are set)
    const uint32_t slots = land_slots ? land_slots + 1 : (params.fast ? FEED_SLOTS_FAST : FEED_SLOTS);
    desc.cbs.push_back(make_cb(CB_FEED, P * o_page, slots, port_set, y.dtype()));
    for (uint32_t id : {SEM_FEED, SEM_FEED_TEARDOWN, SEM_FEED_BUFIDX}) {  // (the last two: the fabric connection's)
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = id, .core_type = tt::CoreType::WORKER, .core_ranges = port_set, .initial_value = 0});
    }

    std::vector<uint32_t> common = {P, RT, B, col_tiles, cpt};
    std::vector<uint32_t> rd_ct = {CB_FEED, o_page};
    rd_ct.insert(rd_ct.end(), common.begin(), common.end());
    rd_ct.push_back(SEM_FEED);
    append_accessor(rd_ct, y);
    if (land_slots) {  // PB.8: after the accessor (the other programs keep their CT args)
        rd_ct.push_back(SEM_LAND);
        rd_ct.push_back(land_slots);
    }
    std::vector<uint32_t> sd_ct = {CB_FEED, o_page};
    sd_ct.insert(sd_ct.end(), common.begin(), common.end());
    append_accessor(sd_ct, fo);

    KernelDescriptor::RuntimeArgs rd_rt, sd_rt;
    for (uint32_t d = 0; d < 2; ++d) {
        for (uint32_t l = 0; l < L; ++l) {
            const CoreCoord core = ports[d * L + l];
            rd_rt.emplace_back(core, std::vector<uint32_t>{d, l});
            // (arg 2: the chip's fabric channel, written by the kernel from the table before build_from_args)
            sd_rt.emplace_back(core, std::vector<uint32_t>{d, l, 0u, SEM_FEED_TEARDOWN, SEM_FEED_BUFIDX});
        }
    }
    desc.kernels.push_back(
        make_kernel("mhc_post_feed_reader.cpp", port_set, rd_ct, rd_rt, tt::tt_metal::ReaderConfigDescriptor{}));
    desc.kernels.push_back(
        make_kernel("mhc_post_feed_sender.cpp", port_set, sd_ct, sd_rt, tt::tt_metal::WriterConfigDescriptor{}));
    TT_FATAL(desc.kernels.size() == FEED_SENDER_KERNEL + 1, "mhc_post stats: feed kernels out of place");
    if (land_slots) {  // PB.8: the reader only waits and pushes; the sender's slot count is the landing's
        desc.kernels[FEED_READER_KERNEL].defines.emplace_back("MHC_POST_LAND", "1");
        desc.kernels[FEED_SENDER_KERNEL].defines.emplace_back("MHC_POST_FEED_SLOTS", std::to_string(land_slots + 1));
        desc.kernels[FEED_SENDER_KERNEL].defines.emplace_back("MHC_POST_LAND", "1");
    }
    auto& rc = desc.kernels[FEED_READER_KERNEL].common_runtime_args;
    rc = {addr(y)};
    rc.insert(rc.end(), ft.head.begin(), ft.head.end());
    auto& sc = desc.kernels[FEED_SENDER_KERNEL].common_runtime_args;
    sc = {addr(fo), params.feed_sems[0], params.feed_sems[1]};
    sc.insert(sc.end(), ft.head.begin(), ft.head.end());
    sc.insert(sc.end(), ft.chan.begin(), ft.chan.end());
}

// The program (P.36 feed: plus the feed's ports; one program for every chip).
tt::tt_metal::ProgramDescriptor build_descriptor(
    const MhcPostStatsParams& params, const MhcPostStatsInputs& in, std::vector<Tensor>& outputs) {
    const Tensor& f = in.input;
    const Tensor& x = in.residual;
    const Tensor& out = outputs.at(0);
    const Tensor& stats = outputs.at(1);
    const Tensor& scratch = outputs.at(2);
    auto* device = f.device();
    const MhcPostStatsPlan p = mhc_post_stats_plan(in, params.max_grid);
    const uint32_t n = p.n, B = p.cols_per_block, NB = p.blocks, RT = p.row_tiles;
    const uint32_t terms = n * (n + 1);
    const uint32_t coef_tiles = (terms * 4 + TILE_HW - 1) / TILE_HW;  // per half (4 row blocks a half)
    const uint32_t page = TILE_HW * TILE_HW * 4;
    // P.19: a bf16 F goes through its own CB (unpacked to the fp32 DEST exactly); fp32 F: the program it always was
    const bool f16 = f.dtype() == DataType::BFLOAT16;
    const uint32_t f_page = f.buffer()->page_size();
    TT_FATAL(f_page == (f16 ? page / 2 : page), "mhc_post stats: F must be fp32 or bf16 tiles");
    // P.20: bf16 X (with bf16 F): X / X' tiles in bf16 pages (CB_IN, CB_OUT, CB_XS; define MHC_POST_STATS_X16), the
    // matmul reads the bf16 X'; fp32 X: the program it always was
    const bool x16 = x.dtype() == DataType::BFLOAT16;
    TT_FATAL(!x16 || f16, "mhc_post stats: a bf16 residual needs a bf16 F");
    const uint32_t x_page = x16 ? page / 2 : page;
    const DataType x_dt = x16 ? DataType::BFLOAT16 : DataType::FLOAT32;
    for (const Tensor* t : {&x, &out}) {
        TT_FATAL(t->buffer()->page_size() == x_page, "mhc_post stats: X / X' must be fp32 or bf16 tiles");
    }
    for (const Tensor* t : {&in.post, &in.comb, &in.weight, &stats, &scratch}) {
        TT_FATAL(t->buffer()->page_size() == page, "mhc_post stats: post / comb / W must be fp32 tiles");
    }
    TT_FATAL(coef_tiles + n + 1 <= 8, "mhc_post stats: n={} does not fit the 8 fp32 DEST tiles", n);
    const bool collapse = params.collapse;
    TT_FATAL(!collapse || (x16 && n <= 4), "mhc_post stats: collapse needs bf16 streams and n <= 4");
    const bool fast = params.fast;  // P.83
    TT_FATAL(!fast || collapse, "mhc_post stats: fast needs collapse");
    const bool v2 = params.v2;  // PB.8
    TT_FATAL(!v2 || fast, "mhc_post stats: v2 needs fast");
    const bool norm = params.norm;
    TT_FATAL(!norm || collapse, "mhc_post stats: norm needs collapse");
    TT_FATAL(!norm || n * (n + 2) + 1 < TILE_HW, "mhc_post stats: norm needs column n (n + 2) + 1 of the partial row");
    if (norm) {
        TT_FATAL(in.norm_weight->buffer()->page_size() == page, "mhc_post stats: norm_weight must be fp32 tiles");
        TT_FATAL(B <= 8, "mhc_post stats: norm needs B <= 8 (4 B weight slots in one tile)");
    }
    if (collapse) {
        TT_FATAL(in.collapse_pre->buffer()->page_size() == page, "mhc_post stats: collapse_pre must be fp32 tiles");
        TT_FATAL(outputs.at(3).buffer()->page_size() == page, "mhc_post stats: hc must be fp32 tiles");
        TT_FATAL(outputs.at(4).buffer()->page_size() == x_page, "mhc_post stats: y must be in X's tiles");
    }

    const CoreCoord grid = cut_grid(device->compute_with_storage_grid_size(), params.max_grid);
    // P.36 (feed): the ports take the grid's last 2 L cores (row-wise, port s = core G - 1 - s, the same logical cores
    // on every chip); the units split over the cores before them. Off: the split as it always was.
    const bool feed = params.feed;
    TT_FATAL(!feed || params.norm, "mhc_post stats: feed needs norm");
    std::vector<CoreCoord> ports;
    CoreRangeSet work_set;
    if (feed) {
        const uint32_t G = grid.x * grid.y, np = 2 * params.feed_links;
        TT_FATAL(params.feed_links >= 1 && G > np + 1, "mhc_post stats: feed: {} ports do not fit {} cores", np, G);
        for (uint32_t s2 = 0; s2 < np; ++s2) {
            if (v2) {  // PB.8: the last column, bottom up (the grid's other columns stay whole for the W segments)
                ports.push_back(CoreCoord{grid.x - 1 - s2 / grid.y, grid.y - 1 - s2 % grid.y});
            } else {
                const uint32_t i = G - 1 - s2;
                ports.push_back(CoreCoord{i % grid.x, i / grid.x});
            }
        }
        work_set = tt::tt_metal::num_cores_to_corerangeset(G - np, grid, true);
    }
    auto is_port = [&](const CoreCoord& c) { return std::find(ports.begin(), ports.end(), c) != ports.end(); };
    std::vector<CoreCoord> cores;
    std::vector<uint32_t> starts, counts;
    CoreRangeSet all_cores;
    // PB.8 (v2): with one unit a core, block b's RT units on a vertical segment of RT cores (column x, rows y0 ..
    // y0 + RT - 1; segments column-major, none holding a port), so each core's W share reaches the block's other cores
    // in one NoC multicast (and short hops); seg_of[b] = the segment's first core
    std::vector<CoreCoord> seg_of;
    if (v2 && RT >= 2 && RT <= 32 && RT <= grid.y && p.units <= grid.x * grid.y - ports.size()) {
        for (uint32_t xx = 0; xx < grid.x && seg_of.size() < NB; ++xx) {
            for (uint32_t y0 = 0; y0 + RT <= grid.y && seg_of.size() < NB; y0 += RT) {
                bool free = true;
                for (uint32_t r = 0; r < RT; ++r) {
                    free = free && !is_port(CoreCoord{xx, y0 + r});
                }
                if (free) {
                    seg_of.push_back(CoreCoord{xx, y0});
                }
            }
        }
        if (seg_of.size() < NB) {
            seg_of.clear();
        }
    }
    if (!seg_of.empty()) {
        std::vector<CoreRange> rr;
        for (uint32_t u = 0; u < p.units; ++u) {
            const CoreCoord c{seg_of[u / RT].x, seg_of[u / RT].y + u % RT};
            cores.push_back(c);
            starts.push_back(u);
            counts.push_back(1);
            rr.emplace_back(c, c);
        }
        all_cores = CoreRangeSet(rr);
    } else if (v2 && feed) {  // PB.8: the ports moved; the units split row-major over the other cores
        std::vector<CoreCoord> avail;
        for (uint32_t yy = 0; yy < grid.y; ++yy) {
            for (uint32_t xx = 0; xx < grid.x; ++xx) {
                if (!is_port(CoreCoord{xx, yy})) {
                    avail.push_back(CoreCoord{xx, yy});
                }
            }
        }
        const uint32_t nc = std::min<uint32_t>(avail.size(), p.units), q = p.units / nc, rem = p.units % nc;
        std::vector<CoreRange> rr;
        uint32_t start = 0;
        for (uint32_t i = 0; i < nc; ++i) {
            const uint32_t c = q + (i < rem ? 1 : 0);
            cores.push_back(avail[i]);
            starts.push_back(start);
            counts.push_back(c);
            start += c;
            rr.emplace_back(avail[i], avail[i]);
        }
        all_cores = CoreRangeSet(rr);
    } else {
        auto [num_cores, all_split, group_1, group_2, units_1, units_2] =
            feed ? tt::tt_metal::split_work_to_cores(work_set, p.units, true)
                 : tt::tt_metal::split_work_to_cores(grid, p.units, true);
        all_cores = all_split;
        uint32_t start = 0;
        for (const auto& [group, per_core] : {std::pair{group_1, units_1}, std::pair{group_2, units_2}}) {
            for (const auto& core : tt::tt_metal::corerange_to_cores(group, std::nullopt, true)) {
                cores.push_back(core);
                starts.push_back(start);
                counts.push_back(per_core);
                start += per_core;
            }
        }
    }
    {
        uint32_t covered = 0;
        for (const uint32_t c : counts) {
            covered += c;
        }
        TT_FATAL(covered == p.units, "mhc_post stats: work split covers {} of {} units", covered, p.units);
    }
    auto owner = [&](uint32_t u) {
        const auto it = std::upper_bound(starts.begin(), starts.end(), u);
        return static_cast<size_t>(std::distance(starts.begin(), it) - 1);
    };
    auto root_unit = [&](uint32_t r) { return (r % NB) * RT + r; };
    std::vector<size_t> root_core(RT);
    for (uint32_t r = 0; r < RT; ++r) {
        root_core[r] = owner(root_unit(r));
    }
    if (params.norm) {
        // P.35: when the grid has a spare core per row (units < cores), the rows' roots are those cores (no units of
        // their own): the partials' sum, the rstd and the Sinkhorn never wait behind a unit, and a unit's norm never
        // waits behind a Sinkhorn. Otherwise the roots stay where they are (the program as without the norm).
        std::vector<CoreCoord> spare;
        for (uint32_t yy = 0; yy < grid.y; ++yy) {
            for (uint32_t xx = 0; xx < grid.x; ++xx) {
                const CoreCoord c{xx, yy};
                if (!all_cores.contains(c) && !is_port(c)) {
                    spare.push_back(c);
                }
            }
        }
        if (spare.size() >= RT) {
            for (uint32_t r = 0; r < RT; ++r) {
                cores.push_back(spare[r]);
                starts.push_back(p.units);  // no units (after every unit core: owner() never picks it)
                counts.push_back(0);
                root_core[r] = cores.size() - 1;
                all_cores = all_cores.merge(CoreRangeSet(tt::tt_metal::CoreRange(spare[r], spare[r])));
            }
        }
    }
    std::vector<std::vector<uint32_t>> rooted(cores.size());
    for (uint32_t r = 0; r < RT; ++r) {
        rooted[root_core[r]].push_back(r);
    }
    std::vector<uint32_t> root_xy(RT);
    for (uint32_t r = 0; r < RT; ++r) {
        const CoreCoord v = device->worker_core_from_logical_core(cores[root_core[r]]);
        root_xy[r] = (static_cast<uint32_t>(v.x) << 16) | static_cast<uint32_t>(v.y);
    }
    auto core_xy = [&](size_t i) {
        const CoreCoord v = device->worker_core_from_logical_core(cores[i]);
        return (static_cast<uint32_t>(v.x) << 16) | static_cast<uint32_t>(v.y);
    };
    const uint32_t max_count = *std::max_element(counts.begin(), counts.end());
    TT_FATAL(!params.norm || max_count < 256, "mhc_post stats: norm takes at most 255 units a core");
    uint32_t max_rooted = 0;
    for (const auto& rr : rooted) {
        max_rooted = std::max<uint32_t>(max_rooted, rr.size());
    }
    // P.83 (fast): with one unit a core (units <= cores) the RT units of a block sit on RT cores; each reads W tiles
    // i (k valid + c order) with i mod RT == its row r and forwards them into the others' cb_w (the same address: the
    // first and only reservation), so a block's W leaves DRAM once instead of RT times. Otherwise each core reads its
    // blocks' W as before.
    const bool w_share = fast && max_count == 1 && RT >= 2 && RT <= 32;
    // PB.8 (v2): a segment whose RT cores are contiguous in NoC coordinates (one column, consecutive rows) takes the
    // W share by multicast; all must (one define), else the forwards stay unicast
    bool w_mcast = w_share && !seg_of.empty();
    for (uint32_t b = 0; w_mcast && b < NB; ++b) {
        const CoreCoord v0 = device->worker_core_from_logical_core(seg_of[b]);
        for (uint32_t r = 1; r < RT; ++r) {
            const CoreCoord v = device->worker_core_from_logical_core(CoreCoord{seg_of[b].x, seg_of[b].y + r});
            w_mcast = w_mcast && v.x == v0.x && v.y == v0.y + r;
        }
    }
    // PB.8 (v2, norm): a root's hc transposes on its SFPU (the writer no longer copies the row's words)
    const bool hct = v2 && params.norm;

    // P.36 (feed): the group along feed_axis, the feed table, and each unit's (target, link) pairs: chip-independent
    // (the writer skips its own block's pair and finds the port from the table)
    uint32_t grp = 1, cpt = 0;
    uint32_t land_slots = 0, land_bytes = 0;  // PB.8 (v2, feed): the landing (0 = off)
    std::vector<uint32_t> feed_word;  // per unit: up to 2 (t + 1) | l << 8 halves, 0 = none
    FeedTable ft;
    if (feed) {
        const auto& ms = device->shape();
        TT_FATAL(ms.dims() == 2 && params.feed_axis < 2, "mhc_post stats: feed needs a 2D mesh and axis 0 or 1");
        grp = ms[params.feed_axis];
        TT_FATAL(grp >= 2, "mhc_post stats: feed needs a group of >= 2 chips along axis {}", params.feed_axis);
        TT_FATAL(p.col_tiles % grp == 0, "mhc_post stats: feed: {} column tiles over {} chips", p.col_tiles, grp);
        cpt = p.col_tiles / grp;
        TT_FATAL(cpt >= B, "mhc_post stats: feed: a chip's {} column tiles < the block's {}", cpt, B);
        ft = feed_table(device, params, ports, cpt, RT, B, p.col_tiles);
        if (v2) {  // PB.8: the landing when it fits (else the y_output staging, as without v2)
            const uint32_t o_page = outputs.at(4).buffer()->page_size();
            const uint32_t P = feed_packet_tiles(o_page);
            land_slots = max_port_packets(ft, RT, B, p.col_tiles, cpt, P);
            land_bytes = (land_slots + 1) * P * o_page;  // (+ the flag page)
            TT_FATAL(p.units * 4 <= P * o_page, "mhc_post stats: v2: {} units' flags exceed a landing page", p.units);
        }
        const uint32_t L = params.feed_links;
        feed_word.assign(p.units, 0u);
        for (uint32_t t = 0; t < grp; ++t) {
            const uint32_t lo = ft.head[mhc_feed_layout::FT_ULO + t], hi = ft.head[mhc_feed_layout::FT_UHI + t];
            for (uint32_t u = lo; u < hi; ++u) {
                const uint32_t half = ((t + 1) | (((u - lo) % L) << 8));
                auto& w = feed_word[u];
                TT_FATAL((w >> 16) == 0, "mhc_post stats: feed: unit {} touches more than 2 chips' blocks", u);
                w = (w == 0) ? half : (w | (half << 16));
            }
        }
    }

    ProgramDescriptor desc;
    desc.cbs.push_back(make_cb(CB_IN, x_page, 2 * (n + 1) * B, all_cores, x_dt));
    desc.cbs.push_back(make_cb(CB_W, page, 2 * n * B, all_cores));
    desc.cbs.push_back(make_cb(CB_CF, page, 2 * 2 * coef_tiles, all_cores));
    // collapse: + the collapse_pre row; P.83 (fast): the writer pushes the raw rows to the compute, two units' worth
    desc.cbs.push_back(make_cb(CB_RAW, page, fast ? 6 : (collapse ? 3 : 2), all_cores));
    desc.cbs.push_back(make_cb(CB_SOA, page, 1, all_cores));
    desc.cbs.push_back(make_cb(CB_SOA_OUT, page, 1, all_cores));
    TT_FATAL(NB <= 64, "mhc_post stats: {} partials per row", NB);
    desc.cbs.push_back(make_cb(CB_PART, page, NB, all_cores));  // one row's partials
    desc.cbs.push_back(make_cb(CB_OUT, x_page, 2 * n * B, all_cores, x_dt));
    desc.cbs.push_back(make_cb(CB_XS, x_page, n * B, all_cores, x_dt));
    desc.cbs.push_back(make_cb(CB_SSC, page, B, all_cores));
    desc.cbs.push_back(make_cb(CB_MFIN, page, 2, all_cores));
    desc.cbs.push_back(make_cb(CB_SFIN, page, 2, all_cores));
    desc.cbs.push_back(make_cb(CB_FIN, page, 2, all_cores));
    if (f16) {
        desc.cbs.push_back(make_cb(CB_F, f_page, 2 * B, all_cores, DataType::BFLOAT16));
    }
    if (collapse) {
        desc.cbs.push_back(make_cb(CB_PB, page, 2, all_cores));
        desc.cbs.push_back(make_cb(CB_Y, x_page, 2 * B, all_cores, x_dt));
        desc.cbs.push_back(make_cb(CB_HSOA, page, 1, all_cores));
        desc.cbs.push_back(make_cb(CB_HSOA_OUT, page, 1, all_cores));
        desc.cbs.push_back(make_cb(CB_HSTAGE, page, 1, all_cores));
    }
    if (norm) {  // P.35: a core's y tiles stay in cb_y until its rows' rstd arrive (max_count units of B)
        for (auto& cb : desc.cbs) {
            const uint8_t id = cb.format_descriptors[0].buffer_index;
            if (id == CB_Y) {
                cb.total_size = max_count * B * x_page;
            } else if (id == CB_SOA || id == CB_SOA_OUT) {  // + the y^2 tile's
                cb.total_size = 2 * page;
            } else if (id == CB_HSOA) {  // every rooted row's copy waits for its hc until after the norm phase
                cb.total_size = std::max<uint32_t>(max_rooted, 1) * page;
            }
        }
        desc.cbs.push_back(make_cb(CB_NSQ, page, 2, all_cores));
        desc.cbs.push_back(make_cb(CB_YN, x_page, 2 * B, all_cores, x_dt));
        desc.cbs.push_back(make_cb(CB_NR, page, max_count, all_cores));  // the roots write unit j's at page j
        desc.cbs.push_back(make_cb(CB_NG, page, max_count, all_cores));
        desc.cbs.push_back(make_cb(CB_NSTAGE, page, 1, all_cores));
        desc.cbs.push_back(make_cb(CB_GSTAGE, page, 1, all_cores));
        desc.cbs.push_back(make_cb(CB_NRS, page, 2, all_cores));
    }

    if (land_slots) {  // PB.8: the landing only where the workers' own CBs would fit as much (same L1 limits)
        uint32_t worker_cb = 0;
        for (const auto& cb : desc.cbs) {
            worker_cb += cb.total_size;
        }
        if (land_bytes > std::min(LAND_BUDGET, worker_cb)) {
            land_slots = 0;
            land_bytes = 0;
        }
    }
    std::vector<uint32_t> common_ct = {n, p.col_tiles, RT, B, NB, page, coef_tiles};
    std::vector<uint32_t> reader_ct = common_ct;
    reader_ct.insert(reader_ct.end(), {CB_IN, CB_W, CB_PART, SEM_ROOT});
    for (const Tensor* t : {&f, &x, &in.weight, &scratch}) {
        append_accessor(reader_ct, *t);
    }
    if (f16) {  // appended only when on: the fp32-F program keeps its CT args
        reader_ct.insert(reader_ct.end(), {CB_F, f_page});
    }
    if (norm) {  // P.35 (collapse, so bf16 F: after CB_F, f_page)
        reader_ct.insert(reader_ct.end(), {CB_NG, CB_NR, CB_GSTAGE, SEM_NORM, CB_NRS, CB_NSTAGE});
        append_accessor(reader_ct, *in.norm_weight);
    }
    std::vector<uint32_t> writer_ct = common_ct;
    writer_ct.insert(
        writer_ct.end(),
        {CB_OUT,
         CB_CF,
         CB_RAW,
         CB_MFIN,
         CB_SFIN,
         CB_SOA,
         CB_SOA_OUT,
         CB_FIN,
         SEM_ROOT,
         n * (n + 2),
         static_cast<uint32_t>(params.comb_transposed)});
    for (const Tensor* t : {&out, &in.post, &in.comb, &scratch, &stats}) {
        append_accessor(writer_ct, *t);
    }
    if (collapse) {  // appended only when on (MHC_POST_COLLAPSE): the off program keeps its CT args
        writer_ct.insert(writer_ct.end(), {CB_PB, CB_Y, CB_HSOA, CB_HSOA_OUT, CB_HSTAGE});
        for (const Tensor* t :
             std::initializer_list<const Tensor*>{&*in.collapse_pre, &outputs.at(3), &outputs.at(4)}) {
            append_accessor(writer_ct, *t);
        }
    }
    if (norm) {  // P.35: after the collapse's
        writer_ct.insert(writer_ct.end(), {CB_NSQ, CB_YN});
    }
    if (feed) {  // P.36: after the norm's
        writer_ct.insert(writer_ct.end(), {cpt, SEM_FEED});
        append_accessor(writer_ct, *in.feed_output);
        if (land_slots) {  // PB.8: then the packet tiles and the landing-address semaphore
            writer_ct.insert(
                writer_ct.end(), {feed_packet_tiles(outputs.at(4).buffer()->page_size()), SEM_LAND, land_slots});
        }
    }
    std::vector<uint32_t> compute_ct = common_ct;
    compute_ct.insert(
        compute_ct.end(),
        {CB_IN, CB_W, CB_CF, CB_OUT, CB_XS, CB_SSC, CB_MFIN, CB_SFIN, CB_SOA, CB_SOA_OUT, CB_PART, CB_FIN});
    if (f16) {
        compute_ct.push_back(CB_F);
    }
    if (collapse) {  // (bf16 X needs bf16 F: CB_F is at 19, these from 20)
        compute_ct.insert(compute_ct.end(), {CB_PB, CB_Y, CB_HSOA, CB_HSOA_OUT});
    }
    if (norm) {
        compute_ct.insert(compute_ct.end(), {CB_NSQ, CB_NR, CB_NG, CB_YN, CB_NRS});
    }
    if (fast) {  // P.83: the raw rows at 29 (after the norm's 24..28, unused without the norm)
        compute_ct.resize(29, 0u);
        compute_ct.push_back(CB_RAW);
    }
    const auto hc_scalars = collapse ? hc_scalar_args(params, n) : std::vector<uint32_t>{};

    KernelDescriptor::RuntimeArgs reader_rt, writer_rt, compute_rt;
    for (size_t i = 0; i < cores.size(); ++i) {
        const uint32_t s = starts[i], c = counts[i];
        std::vector<uint32_t> r = {addr(f), addr(x), addr(in.weight), addr(scratch), s, c};
        r.push_back(static_cast<uint32_t>(rooted[i].size()));
        r.insert(r.end(), rooted[i].begin(), rooted[i].end());
        if (norm) {  // P.35: per rooted row, the owner of each of its NB units and the unit's slot there: j << 24 | xy
            for (const uint32_t row : rooted[i]) {
                for (uint32_t b = 0; b < NB; ++b) {
                    const size_t o = owner(b * RT + row);
                    const uint32_t j = b * RT + row - starts[o];
                    r.push_back((j << 24) | core_xy(o));
                }
            }
        }
        if (w_share && c == 1) {  // P.83: after every list: the unit's block's RT cores, row order
            const uint32_t b = s / RT;
            for (uint32_t rr = 0; rr < RT; ++rr) {
                r.push_back(core_xy(owner(b * RT + rr)));
            }
            if (w_mcast) {  // PB.8: then the segment's first and last core (NoC 0 multicast start / end)
                r.push_back(core_xy(owner(b * RT)));
                r.push_back(core_xy(owner(b * RT + RT - 1)));
            }
        }
        TT_FATAL(r.size() < 300, "mhc_post stats: {} reader runtime args on one core", r.size());
        reader_rt.emplace_back(cores[i], std::move(r));
        std::vector<uint32_t> w = {addr(out), addr(in.post), addr(in.comb), addr(scratch), addr(stats), s, c};
        w.push_back(static_cast<uint32_t>(rooted[i].size()));
        w.insert(w.end(), rooted[i].begin(), rooted[i].end());
        for (uint32_t u = s; u < s + c; ++u) {
            w.push_back(root_xy[u % RT]);  // unit u's row r = u mod RT
        }
        if (feed) {  // P.36: per unit, its (target, link) pairs
            for (uint32_t u = s; u < s + c; ++u) {
                w.push_back(feed_word[u]);
            }
        }
        TT_FATAL(w.size() < 300, "mhc_post stats: {} runtime args on one core", w.size());
        writer_rt.emplace_back(cores[i], std::move(w));
        compute_rt.emplace_back(cores[i], std::vector<uint32_t>{s, c, static_cast<uint32_t>(rooted[i].size())});
    }

    ComputeConfigDescriptor cc;
    cc.math_fidelity = params.compute_config.math_fidelity;
    cc.fp32_dest_acc_en = true;
    cc.math_approx_mode = false;
    cc.dst_full_sync_en = true;  // 8 fp32 DEST tiles: the coefficient tiles + F + X_0..X_{n-1}
    cc.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    for (uint8_t cb : {CB_IN, CB_CF, CB_SSC, CB_SOA, CB_PART}) {
        cc.unpack_to_dest_mode[cb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;  // exact copies
    }
    if (f16) {
        cc.unpack_to_dest_mode[CB_F] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;  // bf16 -> fp32 DEST, exact
    }
    if (collapse) {  // (CB_XS stays Default for the matmul: its bf16 X' reaches the fp32 DEST exactly through srcA)
        for (uint8_t cb : {CB_PB, CB_HSOA}) {
            cc.unpack_to_dest_mode[cb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        }
    }
    if (fast) {  // P.83: the raw post / comb / pre rows, copied exactly into the DEST
        cc.unpack_to_dest_mode[CB_RAW] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    if (norm) {  // (cb_y stays Default: its bf16 tiles reach the fp32 DEST exactly through srcA)
        for (uint8_t cb : {CB_NR, CB_NG}) {
            cc.unpack_to_dest_mode[cb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        }
    }
    // CB_XS / CB_W stay Default: the matmul's operands go through srcB / srcA like ttnn.matmul's fp32 inputs.

    desc.kernels.push_back(make_kernel(
        "mhc_post_stats_reader.cpp", all_cores, reader_ct, reader_rt, tt::tt_metal::ReaderConfigDescriptor{}));
    desc.kernels.push_back(make_kernel(
        "mhc_post_stats_writer.cpp", all_cores, writer_ct, writer_rt, tt::tt_metal::WriterConfigDescriptor{}));
    desc.kernels.push_back(make_kernel("mhc_post_stats_compute.cpp", all_cores, compute_ct, compute_rt, cc));
    if (f16) {  // absent for fp32 F: the reader / compute unchanged
        for (size_t k : {size_t{0}, size_t{2}}) {
            desc.kernels[k].defines.emplace_back("MHC_POST_STATS_F16", "1");
        }
    }
    if (x16) {  // absent for fp32 X: every kernel unchanged (P.20)
        for (auto& k : desc.kernels) {
            k.defines.emplace_back("MHC_POST_STATS_X16", "1");
        }
    }
    if (collapse) {  // absent when off: every kernel unchanged (P.34)
        for (size_t k : {size_t{WRITER_KERNEL}, size_t{COMPUTE_KERNEL}}) {
            desc.kernels[k].defines.emplace_back("MHC_POST_COLLAPSE", "1");
        }
        // the same on every core: common runtime args (no per-core host cost, DS41-145; fixed slots on a cache hit,
        // DS41-210): the writer's collapse_pre / hc / y addresses, the compute's hc scalars
        desc.kernels[WRITER_KERNEL].common_runtime_args = {
            addr(*in.collapse_pre), addr(outputs.at(3)), addr(outputs.at(4))};
        desc.kernels[COMPUTE_KERNEL].common_runtime_args = hc_scalars;
    }
    if (norm) {  // P.35: every kernel (the reader reads the weight; common arg 0 = its address)
        for (auto& k : desc.kernels) {
            k.defines.emplace_back("MHC_POST_NORM", "1");
        }
        desc.kernels[READER_KERNEL].common_runtime_args = {addr(*in.norm_weight)};
    }
    if (feed) {  // P.36: the writer's feed address (common arg 3), the feed table (from 4), then the ports
        desc.kernels[WRITER_KERNEL].defines.emplace_back("MHC_POST_FEED", "1");
        auto& wc = desc.kernels[WRITER_KERNEL].common_runtime_args;
        wc.push_back(addr(*in.feed_output));
        wc.insert(wc.end(), ft.head.begin(), ft.head.end());
        add_feed_ports(desc, params, in, outputs, ports, ft, grp, cpt, RT, B, p.col_tiles, land_slots);
        if (land_slots) {  // PB.8: the workers write the other chips' y tiles into the ports' landing
            desc.kernels[WRITER_KERNEL].defines.emplace_back("MHC_POST_LAND", "1");
            std::vector<CoreRange> pr;
            for (const auto& c : ports) {
                pr.emplace_back(c, c);
            }
            desc.semaphores.push_back(SemaphoreDescriptor{
                .id = SEM_LAND,
                .core_type = tt::CoreType::WORKER,
                .core_ranges = all_cores.merge(CoreRangeSet(pr)),
                .initial_value = 0});
        }
    }
    if (fast) {  // P.83: the stats kernels and the feed reader; the reader also MHC_POST_WSHARE when W is shared
        for (auto& k : desc.kernels) {
            if (k.kernel_source.find("mhc_post_stats_") != std::string::npos ||
                k.kernel_source.find("mhc_post_feed_reader") != std::string::npos) {
                k.defines.emplace_back("MHC_POST_FAST", "1");
            }
            if (k.kernel_source.find("mhc_post_feed_") != std::string::npos &&
                !(land_slots && k.kernel_source.find("mhc_post_feed_sender") != std::string::npos)) {
                k.defines.emplace_back("MHC_POST_FEED_SLOTS", std::to_string(FEED_SLOTS_FAST));
            }
        }
        if (w_share) {
            desc.kernels[READER_KERNEL].defines.emplace_back("MHC_POST_WSHARE", std::to_string(SEM_W));
        }
        if (w_mcast) {  // PB.8
            desc.kernels[READER_KERNEL].defines.emplace_back("MHC_POST_WMCAST", "1");
        }
        if (v2) {  // PB.8: the compute's bottom-half mix as the top half's code, shifted (kernel-config room)
            desc.kernels[COMPUTE_KERNEL].defines.emplace_back("MHC_POST_V2", "1");
        }
        if (hct) {  // PB.8
            desc.kernels[WRITER_KERNEL].defines.emplace_back("MHC_POST_HCT", "1");
            desc.kernels[COMPUTE_KERNEL].defines.emplace_back("MHC_POST_HCT", "1");
        }
        // the compute builds the coefficient tiles: it needs the columns of post / comb and the comb order
        auto& kd = desc.kernels[COMPUTE_KERNEL].defines;
        kd.emplace_back("MHC_POST_FAST_POST_COL", std::to_string(params.post_col));
        kd.emplace_back("MHC_POST_FAST_COMB_COL", std::to_string(params.comb_col));
        kd.emplace_back("MHC_POST_FAST_COMB_T", params.comb_transposed ? "1" : "0");
    }
    if (params.post_col != 0 || params.comb_col != 0) {  // absent when 0 / 0: the writer unchanged
        TT_FATAL(
            params.post_col + n <= TILE_HW && params.comb_col + n * n <= TILE_HW,
            "mhc_post stats: post / comb beyond one tile row");
        desc.kernels[WRITER_KERNEL].defines.emplace_back("MHC_POST_POST_COL", std::to_string(params.post_col));
        desc.kernels[WRITER_KERNEL].defines.emplace_back("MHC_POST_COMB_COL", std::to_string(params.comb_col));
    }
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = SEM_ROOT, .core_type = tt::CoreType::WORKER, .core_ranges = all_cores, .initial_value = 0});
    if (norm) {  // P.35: counts the rstd tiles of a core's units (one increment a unit, from its row's root)
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = SEM_NORM, .core_type = tt::CoreType::WORKER, .core_ranges = all_cores, .initial_value = 0});
    }
    if (w_share) {  // P.83
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = SEM_W, .core_type = tt::CoreType::WORKER, .core_ranges = all_cores, .initial_value = 0});
    }
    return desc;
}

}  // namespace

tt::tt_metal::ProgramDescriptor MhcPostStatsProgramFactory::create_descriptor(
    const MhcPostStatsParams& params, const MhcPostStatsInputs& in, std::vector<Tensor>& outputs) {
    return build_descriptor(params, in, outputs);
}

namespace {
void override_stats(
    tt::tt_metal::Program& program,
    const MhcPostStatsParams& params,
    const MhcPostStatsInputs& in,
    std::vector<Tensor>& outputs,
    bool feed) {
    const uint32_t scratch = addr(outputs.at(2));
    for (auto& col : tt::tt_metal::GetRuntimeArgs(program, READER_KERNEL)) {
        for (auto& args : col) {
            if (args.size() >= 7) {
                args[0] = addr(in.input);
                args[1] = addr(in.residual);
                args[2] = addr(in.weight);
                args[3] = scratch;
            }
        }
    }
    for (auto& col : tt::tt_metal::GetRuntimeArgs(program, WRITER_KERNEL)) {
        for (auto& args : col) {
            if (args.size() >= 8) {
                args[0] = addr(outputs.at(0));
                args[1] = addr(in.post);
                args[2] = addr(in.comb);
                args[3] = scratch;
                args[4] = addr(outputs.at(1));
            }
        }
    }
    if (params.collapse) {  // P.34: the common args (addresses; the hc scalars: one program for every layer)
        auto& w = tt::tt_metal::GetCommonRuntimeArgs(program, WRITER_KERNEL);
        w[0] = addr(*in.collapse_pre);
        w[1] = addr(outputs.at(3));
        w[2] = addr(outputs.at(4));
        const auto n = static_cast<uint32_t>(in.residual.logical_shape()[-1] / in.input.logical_shape()[-1]);
        const auto sc = hc_scalar_args(params, n);
        auto& k = tt::tt_metal::GetCommonRuntimeArgs(program, COMPUTE_KERNEL);
        for (size_t i = 0; i < sc.size(); ++i) {
            k[i] = sc[i];
        }
    }
    if (params.norm) {  // P.35: the weight's address (its scalars: in hc_scalar_args above)
        tt::tt_metal::GetCommonRuntimeArgs(program, READER_KERNEL)[0] = addr(*in.norm_weight);
    }
    if (feed) {  // P.36: the feed's addresses (y staging, feed_output) and the semaphores
        tt::tt_metal::GetCommonRuntimeArgs(program, WRITER_KERNEL)[3] = addr(*in.feed_output);
        tt::tt_metal::GetCommonRuntimeArgs(program, FEED_READER_KERNEL)[0] = addr(outputs.at(4));
        auto& sd = tt::tt_metal::GetCommonRuntimeArgs(program, FEED_SENDER_KERNEL);
        sd[0] = addr(*in.feed_output);
        sd[1] = params.feed_sems[0];
        sd[2] = params.feed_sems[1];
    }
}
}  // namespace

void MhcPostStatsProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const MhcPostStatsParams& params,
    const MhcPostStatsInputs& in,
    std::vector<Tensor>& outputs,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    override_stats(program, params, in, outputs, params.feed);
}

}  // namespace ttnn::operations::bringup::mhc_post_ttnn
