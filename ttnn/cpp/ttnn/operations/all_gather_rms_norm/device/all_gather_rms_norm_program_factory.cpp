// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_rms_norm_program_factory.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <map>
#include <string>
#include <functional>
#include <initializer_list>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt_stl/indestructible.hpp>

#include "kernels/all_gather_rms_norm_args.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::all_gather_rms_norm {

using namespace tt::tt_metal;
using namespace ::agrn_args;
using planner::Dir;
using tt::CoreType;

namespace {

constexpr const char* kAgrnKernelDir = "ttnn/cpp/ttnn/operations/all_gather_rms_norm/device/kernels/";

std::string agrn_kernel_path(const char* name) { return std::string(kAgrnKernelDir) + name; }

uint32_t agrn_f32_bits(double v) { return std::bit_cast<uint32_t>(static_cast<float>(v)); }

uint32_t agrn_pack_xy(const CoreCoord& v) {
    return (static_cast<uint32_t>(v.x) & 0xFFFFu) | (static_cast<uint32_t>(v.y) << 16);
}

uint32_t agrn_datum_bytes(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return 4;
        case DataType::BFLOAT16: return 2;
        case DataType::BFLOAT8_B: return 1;
        default: TT_THROW("all_gather_rms_norm: no datum size for dtype {}", dtype);
    }
}

CBDescriptor agrn_cb(
    uint32_t index, uint32_t num_pages, uint32_t page_size, DataType dtype, const CoreRangeSet& cores) {
    return CBDescriptor{
        .total_size = num_pages * page_size,
        .core_ranges = cores,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index),
            .data_format = datatype_to_dataformat_converter(dtype),
            .page_size = page_size,
        }},
    };
}

// Python _inc_sim: per entry the chunks sent and increments issued before it, and the total increments.
struct IncSim {
    std::vector<uint32_t> sent_before, incs_before;
    uint32_t total = 0;
};
IncSim agrn_inc_sim(const std::vector<uint32_t>& counts, uint32_t inc_every) {
    IncSim s;
    uint32_t sent = 0;
    for (const uint32_t n : counts) {
        s.sent_before.push_back(sent);
        s.incs_before.push_back(s.total);
        for (uint32_t i = 0; i < n; ++i) {
            ++sent;
            if (sent % inc_every == 0 || i + 1 == n) {
                ++s.total;
            }
        }
    }
    return s;
}

std::vector<uint32_t> agrn_slots(const planner::Schedule& s) {
    std::vector<uint32_t> out;
    out.reserve(s.size());
    for (const auto& [q, part] : s) {
        out.push_back(q);
    }
    return out;
}

const planner::Schedule& agrn_dir_sched(const std::pair<planner::Schedule, planner::Schedule>& s, Dir d) {
    return d == Dir::FWD ? s.first : s.second;
}

CoreRangeSet agrn_core_set(const std::vector<CoreCoord>& cores) {
    std::vector<CoreRange> ranges;
    ranges.reserve(cores.size());
    for (const auto& c : cores) {
        ranges.emplace_back(c, c);
    }
    return CoreRangeSet(ranges).merge_ranges();
}

}  // namespace

const PersistentSems& persistent_semaphores(
    tt::tt_metal::distributed::MeshDevice& mesh, const planner::MeshPlan& mesh_plan) {
    std::vector<CoreCoord> port_cores;
    for (const auto& [coord, ch] : mesh_plan.chips) {
        for (const auto& dir_ports : ch.ports) {
            port_cores.insert(port_cores.end(), dir_ports.begin(), dir_ports.end());
        }
    }
    std::sort(port_cores.begin(), port_cores.end());
    port_cores.erase(std::unique(port_cores.begin(), port_cores.end()), port_cores.end());
    std::vector<std::pair<uint32_t, uint32_t>> key_cores;
    key_cores.reserve(port_cores.size());
    for (const auto& c : port_cores) {
        key_cores.emplace_back(static_cast<uint32_t>(c.x), static_cast<uint32_t>(c.y));
    }
    using Key = std::tuple<uint32_t, std::vector<uint32_t>, std::vector<std::pair<uint32_t, uint32_t>>>;
    struct Entry {
        std::vector<GlobalSemaphore> sems;
        PersistentSems addrs;
    };
    static ttsl::Indestructible<std::map<Key, Entry>> cache;
    const auto& shape = mesh.shape();
    std::vector<uint32_t> dims(shape.cbegin(), shape.cend());
    Key key{static_cast<uint32_t>(mesh.id()), std::move(dims), std::move(key_cores)};
    auto it = cache.get().find(key);
    if (it == cache.get().end()) {
        Entry e;
        const CoreRangeSet cores = agrn_core_set(port_cores);
        for (uint32_t i = 0; i < planner::NUM_GLOBAL_SEMS; ++i) {
            e.sems.push_back(ttnn::global_semaphore::create_global_semaphore(&mesh, cores, 0));
            e.addrs.addresses[i] =
                static_cast<uint32_t>(ttnn::global_semaphore::get_global_semaphore_address(e.sems.back()));
        }
        // every chip's counters are 0 before any chip's first call
        tt::tt_metal::distributed::Synchronize(&mesh, std::nullopt);
        it = cache.get().emplace(std::move(key), std::move(e)).first;
    }
    return it->second.addrs;
}

ProgramDescriptor AllGatherRmsNormProgramFactory::create_descriptor(
    const AllGatherRmsNormParams& attrs,
    const AllGatherRmsNormInputs& tensors,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(
        cb::MAX_ID < hal::get_num_dataflow_buffers(),
        "all_gather_rms_norm: circular-buffer id {} exceeds the {} circular buffers of this architecture",
        cb::MAX_ID,
        hal::get_num_dataflow_buffers());
    TT_FATAL(mesh_dispatch_coordinate.has_value(), "all_gather_rms_norm: one program per mesh coordinate");
    const ttnn::MeshCoordinate& coord = *mesh_dispatch_coordinate;
    const Tensor& input = tensors.input;
    auto* mesh = input.device();
    TT_FATAL(mesh != nullptr, "all_gather_rms_norm: input must be on a mesh device");

    const uint32_t G = mesh->shape()[attrs.cluster_axis];
    const bool ring = attrs.topology == tt::tt_fabric::Topology::Ring && G > 2;
    const bool x_sharded = input.memory_config().is_sharded();
    std::vector<CoreCoord> excluded;
    if (x_sharded) {
        excluded = planner::ordered_shard_cores(input.memory_config().shard_spec()->grid, true);
    }
    const planner::MeshPlan& mp = planner::build_mesh_plan(*mesh, attrs.cluster_axis, ring, attrs.num_links, excluded);
    const PersistentSems& ps = persistent_semaphores(*mesh, mp);
    const planner::ProgramPlan& plan = planner::solve_program_plan(attrs, tensors, mp);
    const planner::Geometry& geo = plan.geo;
    const planner::Split& split = plan.split;
    const planner::Walk& walk = plan.walk;
    const planner::ChipPlan& ch = mp.chips.at(coord);
    const uint32_t p = ch.p;
    const uint32_t L = mp.num_links;
    const uint32_t n_ports = 2 * L;
    const uint32_t gx = mp.grid.x;
    const uint32_t gy = mp.grid.y;
    const uint32_t k = split.k;
    const uint32_t n_active = split.n_active();
    const bool is_ln = attrs.is_layer_norm;
    const bool fp32_dest = attrs.fp32_dest_acc_en;

    const DataType x_dtype = input.dtype();
    const DataType out_dtype = output.dtype();
    const DataType acc_dtype = fp32_dest ? DataType::FLOAT32 : DataType::BFLOAT16;
    const uint32_t T_acc = tile_size(acc_dtype);
    const uint32_t T_in = tile_size(x_dtype);
    const uint32_t T32 = tile_size(DataType::FLOAT32);
    const uint32_t out_page = plan.out_page;
    const bool has_gamma = tensors.gamma.has_value();
    const bool has_beta = tensors.beta.has_value();
    const DataType g_dtype = has_gamma ? tensors.gamma->dtype() : DataType::BFLOAT16;
    const uint32_t g_eb = agrn_datum_bytes(g_dtype);
    constexpr uint32_t kFace = tt::constants::FACE_HEIGHT * tt::constants::FACE_WIDTH;
    const uint32_t gamma_page = 2 * kFace * g_eb;  // faces 0/1 of a gamma tile
    const bool gamma_is_tile = has_gamma && tensors.gamma->layout() == Layout::TILE;
    const uint32_t gamma_src_page = gamma_is_tile ? tt::constants::TILE_HW * g_eb : tt::constants::TILE_WIDTH * g_eb;
    const bool x_is_f32 = x_dtype == DataType::FLOAT32;
    const bool pre_combine = plan.pre_combine;
    const bool is_ru = plan.is_ru;
    const bool mask_tail = plan.mask_tail;
    const uint32_t crt_max = plan.crt_max;
    const uint32_t cw_max = plan.cw_max;
    const uint32_t row_chunk = plan.row_chunk;
    const uint32_t nstats = plan.nstats;
    const uint32_t gt_max = plan.gt_max;
    const uint32_t own_bands = plan.own_bands;

    TT_FATAL(!is_ru || tensors.scratch.has_value(), "all_gather_rms_norm: the R_U regime needs its staging scratch");
    Buffer* dst_buffer = is_ru ? tensors.scratch->buffer() : output.buffer();
    Buffer* in_buffer = input.buffer();
    Buffer* gamma_buffer = has_gamma ? tensors.gamma->buffer() : nullptr;
    Buffer* beta_buffer = has_beta ? tensors.beta->buffer() : nullptr;

    const std::optional<planner::StagePlan>& sp = plan.stage_plans.at(coord);
    const uint32_t stage_cp = plan.run_pages;

    // ---- core sets ----
    const CoreRangeSet full(CoreRange(CoreCoord(0, 0), CoreCoord(gx - 1, gy - 1)));
    std::vector<CoreCoord> port_list;
    for (const Dir d : {Dir::FWD, Dir::BWD}) {
        const auto& v = ch.ports[static_cast<size_t>(d)];
        TT_FATAL(v.size() == L, "all_gather_rms_norm: {} port cores for {} links", v.size(), L);
        port_list.insert(port_list.end(), v.begin(), v.end());
    }
    const CoreRangeSet port_cores = agrn_core_set(port_list);
    const CoreRangeSet worker_set = full.subtract(port_cores);
    std::vector<CoreCoord> active;
    if (x_sharded) {
        active = plan.shard_order;
        for (const auto& c : active) {
            TT_FATAL(
                std::find(port_list.begin(), port_list.end(), c) == port_list.end(),
                "port cores must lie outside the input shard grid");
        }
    } else {
        for (uint32_t y = 0; y < gy && active.size() < n_active; ++y) {
            for (uint32_t x = 0; x < gx && active.size() < n_active; ++x) {
                const CoreCoord c(x, y);
                if (std::find(port_list.begin(), port_list.end(), c) == port_list.end()) {
                    active.push_back(c);
                }
            }
        }
    }
    auto virt = [&](const CoreCoord& c) { return mesh->worker_core_from_logical_core(c); };
    std::vector<uint32_t> active_noc;
    active_noc.reserve(active.size());
    for (const auto& c : active) {
        active_noc.push_back(agrn_pack_xy(virt(c)));
    }
    const uint32_t hub_fwd = agrn_pack_xy(virt(ch.ports[0].at(0)));
    const uint32_t hub_bwd = agrn_pack_xy(virt(ch.ports[1].at(0)));
    const uint32_t mc0 = agrn_pack_xy(virt(CoreCoord(0, 0)));
    const uint32_t mc1 = agrn_pack_xy(virt(CoreCoord(gx - 1, gy - 1)));

    // ---- schedules (own chip and its neighbours) ----
    auto stats_sched = [&](const ttnn::MeshCoordinate& c) {
        return planner::schedule(mp.chips.at(c).p, G, ring, false);
    };
    auto bulk_sched = [&](const ttnn::MeshCoordinate& c) {
        return planner::schedule(mp.chips.at(c).p, G, ring, plan.balance);
    };
    const std::array<std::optional<ttnn::MeshCoordinate>, 2> peer_of{ch.next, ch.prev};
    const std::array<std::optional<ttnn::MeshCoordinate>, 2> up_of{ch.prev, ch.next};
    std::array<std::vector<uint32_t>, 2> recv;
    for (const Dir d : {Dir::FWD, Dir::BWD}) {
        const auto di = static_cast<size_t>(d);
        if (up_of[di].has_value()) {
            recv[di] = agrn_slots(agrn_dir_sched(stats_sched(*up_of[di]), d));
        }
    }
    uint32_t hub_mask = 0;
    for (uint32_t q = 0; q < G; ++q) {
        if (q == p) {
            continue;
        }
        const bool in_f = std::find(recv[0].begin(), recv[0].end(), q) != recv[0].end();
        const bool in_b = std::find(recv[1].begin(), recv[1].end(), q) != recv[1].end();
        TT_FATAL(in_f || in_b, "slot {} never reaches chip {}", q, p);
        if (!in_f) {
            hub_mask |= 1u << q;
        }
    }

    ProgramDescriptor desc;
    // ---- program-local semaphores (first, on the full grid, ids 0..COUNT-1) ----
    for (uint32_t sid = 0; sid < sem::COUNT; ++sid) {
        const auto id = desc.find_available_semaphore_id(CoreCoord(0, 0), CoreType::WORKER);
        TT_FATAL(id.has_value() && *id == sid, "all_gather_rms_norm: semaphore id {} unavailable", sid);
        desc.semaphores.push_back(
            SemaphoreDescriptor{.id = *id, .core_type = CoreType::WORKER, .core_ranges = full, .initial_value = 0});
    }

    // ---- circular buffers (all on the full program range) ----
    if (x_sharded) {
        desc.cbs.push_back(cb_descriptor_from_sharded_tensor(static_cast<uint8_t>(cb::X), input, 0, 0, full));
    } else {
        desc.cbs.push_back(agrn_cb(cb::X, crt_max * cw_max, T_in, x_dtype, full));
    }
    desc.cbs.push_back(agrn_cb(cb::SCALER, 1, tile_size(DataType::BFLOAT16), DataType::BFLOAT16, full));
    desc.cbs.push_back(agrn_cb(cb::PARTIAL, row_chunk * nstats, T32, DataType::FLOAT32, full));
    desc.cbs.push_back(agrn_cb(cb::RECORD_STAGING, row_chunk * nstats, RECORD_BYTES, DataType::FLOAT32, full));
    desc.cbs.push_back(agrn_cb(cb::GATHER_ROWS, row_chunk * nstats * gt_max, T32, DataType::FLOAT32, full));
    desc.cbs.push_back(agrn_cb(cb::GATHERED, row_chunk * nstats * gt_max, T32, DataType::FLOAT32, full));
    desc.cbs.push_back(agrn_cb(cb::INV, row_chunk, T_acc, acc_dtype, full));
    if (plan.out_in_place) {
        desc.cbs.push_back(cb_descriptor_from_sharded_tensor(
            static_cast<uint8_t>(cb::OUT_BLOCK), output, p * geo.Wlt * out_page, geo.Wlt * out_page, full));
    } else {
        desc.cbs.push_back(agrn_cb(cb::OUT_BLOCK, crt_max * cw_max, out_page, out_dtype, full));
    }
    desc.cbs.push_back(agrn_cb(cb::PORT_CHUNKS, 2 * planner::BATCH_CHUNKS * plan.run_pages, out_page, out_dtype, full));
    desc.cbs.push_back(agrn_cb(cb::PORT_DESC, 2 * planner::BATCH_CHUNKS, PORT_DESC_BYTES, DataType::FLOAT32, full));
    if (plan.stage_max_pages > 0) {
        desc.cbs.push_back(agrn_cb(cb::STAGE, plan.stage_max_pages, out_page, out_dtype, full));
    }
    desc.cbs.push_back(
        agrn_cb(cb::STATS_TABLE, plan.table_bytes / RECORD_BYTES, RECORD_BYTES, DataType::FLOAT32, full));
    if (has_gamma) {
        desc.cbs.push_back(agrn_cb(cb::GAMMA, cw_max, gamma_page, g_dtype, full));
        desc.cbs.push_back(
            agrn_cb(cb::GAMMA_ROWS, 1, ((cw_max + 2) * RECORD_BYTES) + PORT_DESC_BYTES, DataType::FLOAT32, full));
    }
    if (has_beta) {
        desc.cbs.push_back(agrn_cb(cb::BETA, cw_max, gamma_page, g_dtype, full));
    }
    if (is_ln || mask_tail) {
        const DataType sq_dtype = (is_ln && x_sharded && !x_is_f32) ? DataType::BFLOAT16 : acc_dtype;
        desc.cbs.push_back(agrn_cb(cb::MASK, mask_tail ? cw_max : 1, T_acc, acc_dtype, full));
        desc.cbs.push_back(agrn_cb(cb::LN_SQ, row_chunk * cw_max, tile_size(sq_dtype), sq_dtype, full));
    }
    if (is_ln) {
        desc.cbs.push_back(agrn_cb(cb::LN_M, row_chunk, T32, DataType::FLOAT32, full));
        desc.cbs.push_back(agrn_cb(cb::LN_SHIFT, row_chunk, T32, DataType::FLOAT32, full));
        desc.cbs.push_back(
            agrn_cb(cb::STAT_CONSTS, pre_combine ? plan.gt_pc : plan.gather_tiles, T32, DataType::FLOAT32, full));
        desc.cbs.push_back(agrn_cb(cb::MEAN_COL, row_chunk, T32, DataType::FLOAT32, full));
        desc.cbs.push_back(agrn_cb(cb::MEAN_B, row_chunk, T32, DataType::FLOAT32, full));
        desc.cbs.push_back(agrn_cb(cb::RSTD_B, row_chunk, T_acc, acc_dtype, full));
        desc.cbs.push_back(agrn_cb(cb::LN_TMP, row_chunk * gt_max, T32, DataType::FLOAT32, full));
        if (pre_combine) {
            desc.cbs.push_back(agrn_cb(cb::STAT_CONSTS_X, plan.gather_tiles, T32, DataType::FLOAT32, full));
            desc.cbs.push_back(agrn_cb(cb::FINAL_MEAN, row_chunk, T32, DataType::FLOAT32, full));
        }
        if (!x_is_f32 && fp32_dest && !x_sharded) {
            desc.cbs.push_back(agrn_cb(cb::X32, crt_max * cw_max, T32, DataType::FLOAT32, full));
        }
    } else {
        desc.cbs.push_back(agrn_cb(cb::SUMSQ, row_chunk, T_acc, acc_dtype, full));
    }
    if (plan.xg_pre) {
        desc.cbs.push_back(agrn_cb(cb::XG, crt_max * cw_max, T_acc, acc_dtype, full));
    }
    if (is_ru) {
        desc.cbs.push_back(agrn_cb(cb::ASSEMBLE, plan.src_max + 1, out_page, out_dtype, full));
    }
    if (pre_combine) {
        desc.cbs.push_back(cb_descriptor_from_sharded_tensor(static_cast<uint8_t>(cb::X_STATS), input, 0, 0, full));
        desc.cbs.push_back(agrn_cb(cb::PRE_TABLE, plan.rpo * k * nstats, RECORD_BYTES, DataType::FLOAT32, full));
        desc.cbs.push_back(agrn_cb(cb::FINAL_TABLE, geo.R * plan.nfinal, RECORD_BYTES, DataType::FLOAT32, full));
        desc.cbs.push_back(agrn_cb(cb::FINAL_INV, row_chunk, T32, DataType::FLOAT32, full));
    }

    // ---- worker compile-time args ----
    std::vector<uint32_t> wct(worker_ct::COUNT, 0u);
    wct[worker_ct::R] = geo.R;
    wct[worker_ct::WLT] = geo.Wlt;
    wct[worker_ct::K] = k;
    wct[worker_ct::N_GROUPS] = split.n_groups;
    wct[worker_ct::ROWS_HI] = split.rows_hi;
    wct[worker_ct::BIG] = split.big;
    wct[worker_ct::CW_NOM] = split.cw_nom;
    wct[worker_ct::GRID_X] = gx;
    wct[worker_ct::G] = G;
    wct[worker_ct::NSTATS] = nstats;
    wct[worker_ct::GATHER_TILES] = plan.gather_tiles;
    wct[worker_ct::IS_LN] = is_ln ? 1u : 0u;
    wct[worker_ct::HAS_GAMMA] = has_gamma ? 1u : 0u;
    wct[worker_ct::HAS_BETA] = has_beta ? 1u : 0u;
    wct[worker_ct::GAMMA_IS_TILE] = gamma_is_tile ? 1u : 0u;
    wct[worker_ct::X_PAGE_BYTES] = T_in;
    wct[worker_ct::OUT_PAGE_BYTES] = out_page;
    wct[worker_ct::GAMMA_ELEM_BYTES] = g_eb;
    wct[worker_ct::W_VALID_LAST] = geo.w_valid_last;
    wct[worker_ct::W_LOCAL] = geo.W_local;
    wct[worker_ct::WT] = geo.Wt;
    wct[worker_ct::IS_RU] = is_ru ? 1u : 0u;
    wct[worker_ct::X_IS_F32] = x_is_f32 ? 1u : 0u;
    wct[worker_ct::OUT_ELEM_BYTES] = agrn_datum_bytes(out_dtype);
    wct[worker_ct::N_PORTS] = n_ports;
    wct[worker_ct::ASSEMBLE_SRC_MAX] = plan.src_max;
    wct[worker_ct::GAMMA_PAGE_BYTES] = gamma_page;
    wct[worker_ct::GAMMA_SRC_PAGE_BYTES] = gamma_src_page;
    wct[worker_ct::X_SHARDED] = x_sharded ? 1u : 0u;
    wct[worker_ct::LAST_CRT] = split.last_rows();
    wct[worker_ct::OUT_IN_PLACE] = plan.out_in_place ? 1u : 0u;
    wct[worker_ct::PRE_COMBINE] = pre_combine ? 1u : 0u;
    wct[worker_ct::ROW_CHUNK] = row_chunk;
    wct[worker_ct::ROWS_PER_OWNER] = plan.rpo;
    wct[worker_ct::OWN_BANDS] = own_bands;
    wct[worker_ct::XG_PRE] = plan.xg_pre ? 1u : 0u;
    wct[worker_ct::STAGE_ON] = sp.has_value() ? 1u : 0u;
    wct[worker_ct::ST_L] = L;
    wct[worker_ct::ST_NB] = walk.nb;
    wct[worker_ct::ST_COLS] = walk.cols_eff;
    wct[worker_ct::ST_ROW_STRIDE] = walk.row_stride;
    wct[worker_ct::ST_SLOT_STRIDE] = walk.slot_stride;
    wct[worker_ct::ST_BY_ROW] = walk.shard_w > 0 ? 1u : 0u;
    if (sp.has_value()) {
        TT_FATAL(sp->lanes.size() == L && L <= worker_ct::ST_MAX_LANES, "all_gather_rms_norm: stage lanes");
        for (uint32_t l = 0; l < L; ++l) {
            wct[worker_ct::ST_LANES + l] = sp->lanes[l].n_st;
        }
        for (uint32_t b = 0; b < worker_ct::ST_MAX_NB; ++b) {
            wct[worker_ct::ST_UNIT_OFF + b] = sp->unit_off[b];
        }
    }
    wct[worker_ct::ST_RP] = stage_cp;
    wct[worker_ct::ST_NCH] = sp.has_value() ? sp->nch_max : 1u;
    wct[worker_ct::ST_NACT] = n_active;
    wct[worker_ct::ST_CTR] = ps.stage_counter();

    // ---- worker common runtime args ----
    std::vector<uint32_t> inv_n;
    inv_n.reserve(k);
    for (uint32_t m = 0; m < k; ++m) {
        const uint32_t w_tiles = split.member_width(m, geo.Wlt);
        const uint32_t tail =
            (m * split.cw_nom + w_tiles == geo.Wlt) ? tt::constants::TILE_WIDTH - geo.w_valid_last : 0u;
        const uint32_t n = (w_tiles * tt::constants::TILE_WIDTH) - tail;
        inv_n.push_back(agrn_f32_bits(1.0 / static_cast<double>(std::max(1u, n))));
    }
    std::vector<uint32_t> tail_args;
    tail_args.reserve((2 * port_list.size()) + inv_n.size() + 1 + own_bands);
    for (const auto& c : port_list) {
        tail_args.push_back(static_cast<uint32_t>((c.y * gx) + c.x));
    }
    for (const auto& c : port_list) {
        tail_args.push_back(agrn_pack_xy(virt(c)));
    }
    tail_args.insert(tail_args.end(), inv_n.begin(), inv_n.end());
    tail_args.push_back(agrn_f32_bits(1.0 / static_cast<double>(geo.W_local)));
    for (uint32_t b = 0; b < own_bands; ++b) {
        tail_args.push_back(ps.band(b));
    }
    if (pre_combine) {
        tail_args.insert(tail_args.end(), active_noc.begin(), active_noc.end());
    }
    if (x_sharded) {
        std::vector<uint32_t> wmap(gx * gy, 0xFFu);
        for (uint32_t i = 0; i < plan.shard_order.size(); ++i) {
            const auto& c = plan.shard_order[i];
            wmap[(c.y * gx) + c.x] = i;
        }
        while (wmap.size() % 4 != 0) {
            wmap.push_back(0xFFu);
        }
        for (size_t i = 0; i < wmap.size(); i += 4) {
            tail_args.push_back(wmap[i] | (wmap[i + 1] << 8) | (wmap[i + 2] << 16) | (wmap[i + 3] << 24));
        }
    }
    if (sp.has_value()) {
        wct[worker_ct::ST_TBL] = worker_common::PORTS + static_cast<uint32_t>(tail_args.size());
        tail_args.insert(tail_args.end(), active_noc.begin(), active_noc.end());
    }
    KernelDescriptor::RTArgList worker_common_args;
    {
        std::vector<std::variant<uint32_t, Buffer*>> head(worker_common::PORTS, 0u);
        head[worker_common::X_ADDR] = in_buffer;
        head[worker_common::DST_ADDR] = dst_buffer;
        head[worker_common::OUT_ADDR] = output.buffer();
        head[worker_common::GAMMA_ADDR] = gamma_buffer;
        head[worker_common::BETA_ADDR] = beta_buffer;
        head[worker_common::EPS_BITS] = agrn_f32_bits(attrs.epsilon);
        head[worker_common::INV_W_BITS] = agrn_f32_bits(1.0 / static_cast<double>(geo.W));
        head[worker_common::P] = p;
        head[worker_common::HUB_MASK] = hub_mask;
        head[worker_common::FWD_HUB] = hub_fwd;
        head[worker_common::BWD_HUB] = hub_bwd;
        for (const auto& v : head) {
            if (std::holds_alternative<Buffer*>(v)) {
                worker_common_args.push_back(std::get<Buffer*>(v));
            } else {
                worker_common_args.push_back(std::get<uint32_t>(v));
            }
        }
        worker_common_args.append(tail_args);
    }

    // tensor accessors: x (sharded: placeholder), dst, out, gamma, beta
    std::vector<uint32_t> acc_ct;
    TensorAccessorArgs(x_sharded ? nullptr : in_buffer).append_to(acc_ct);
    TensorAccessorArgs(dst_buffer).append_to(acc_ct);
    TensorAccessorArgs(output.buffer()).append_to(acc_ct);
    TensorAccessorArgs(gamma_buffer).append_to(acc_ct);
    TensorAccessorArgs(beta_buffer).append_to(acc_ct);
    std::vector<uint32_t> wct_dm = wct;
    wct_dm.insert(wct_dm.end(), acc_ct.begin(), acc_ct.end());

    // ---- compute config ----
    std::vector<UnpackToDestMode> modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    std::vector<uint32_t> tagged;
    if (fp32_dest) {
        tagged.push_back(cb::GATHER_ROWS);
        if (is_ln) {
            tagged.insert(tagged.end(), {cb::GATHERED, cb::STAT_CONSTS, cb::LN_TMP, cb::MEAN_COL});
            if (pre_combine) {
                tagged.insert(tagged.end(), {cb::STAT_CONSTS_X, cb::FINAL_MEAN});
            }
            if (!mask_tail && (x_is_f32 || !x_sharded)) {
                tagged.push_back(x_is_f32 ? cb::X : cb::X32);
                if (pre_combine && x_is_f32) {
                    tagged.push_back(cb::X_STATS);
                }
            }
        } else {
            tagged.insert(tagged.end(), {cb::SUMSQ, cb::GATHERED});
        }
    }
    for (const uint32_t id : tagged) {
        modes[id] = UnpackToDestMode::UnpackToDestFp32;
    }

    KernelDescriptor reader;
    reader.kernel_source = agrn_kernel_path("all_gather_rms_norm_reader.cpp");
    reader.core_ranges = worker_set;
    reader.compile_time_args = wct_dm;
    reader.emplace_common_runtime_args(worker_common_args);
    reader.config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_0_default};

    KernelDescriptor writer;
    writer.kernel_source = agrn_kernel_path("all_gather_rms_norm_writer.cpp");
    writer.core_ranges = worker_set;
    writer.compile_time_args = wct_dm;
    writer.emplace_common_runtime_args(worker_common_args);
    writer.config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_1_default};

    KernelDescriptor compute;
    compute.kernel_source = agrn_kernel_path("all_gather_rms_norm_compute.cpp");
    compute.core_ranges = worker_set;
    compute.compile_time_args = wct;
    compute.emplace_common_runtime_args(worker_common_args);
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = attrs.math_fidelity,
        .fp32_dest_acc_en = fp32_dest,
        .unpack_to_dest_mode = modes,
        .math_approx_mode = attrs.math_approx_mode,
    };

    // ---- port kernels ----
    std::vector<uint32_t> pct(port_ct::COUNT, 0u);
    pct[port_ct::CB_CHUNKS] = cb::PORT_CHUNKS;
    pct[port_ct::PAGE_BYTES] = out_page;
    pct[port_ct::RUN_PAGES] = plan.run_pages;
    pct[port_ct::GROUP] = planner::BATCH_CHUNKS;
    pct[port_ct::INC_EVERY] = planner::INC_EVERY;
    pct[port_ct::NB] = walk.nb;
    pct[port_ct::ROWS_EFF] = walk.rows_eff;
    pct[port_ct::COLS_EFF] = walk.cols_eff;
    pct[port_ct::ROW_STRIDE] = walk.row_stride;
    pct[port_ct::SLOT_STRIDE] = walk.slot_stride;
    pct[port_ct::FLAT_SLOT] = is_ru ? 1u : 0u;
    pct[port_ct::WLT] = geo.Wlt;
    pct[port_ct::K] = k;
    pct[port_ct::N_GROUPS] = split.n_groups;
    pct[port_ct::ROWS_HI] = split.rows_hi;
    pct[port_ct::BIG] = split.big;
    pct[port_ct::CW_NOM] = split.cw_nom;
    pct[port_ct::PAYLOAD_BYTES] = plan.stats_payload;
    pct[port_ct::SCATTER_RUNS] = planner::SCATTER_RUNS;
    pct[port_ct::SHARD_W] = walk.shard_w;
    pct[port_ct::LAST_CRT] = split.last_rows();
    pct[port_ct::READ_DEPTH] = planner::READ_DEPTH;
    pct[port_ct::OWN_FROM_DST] = planner::OWN_SRC_DST;
    pct[port_ct::CB_DESC] = cb::PORT_DESC;
    pct[port_ct::SPLIT_RUNS] = planner::SPLIT_RUNS;
    pct[port_ct::OWN_BANDS] = own_bands;
    pct[port_ct::EAGER_PUSH] = planner::EAGER_PUSH;
    pct[port_ct::CB_STAGE] = cb::STAGE;
    pct[port_ct::STAGE_ON] = sp.has_value() ? 1u : 0u;
    TensorAccessorArgs(dst_buffer).append_to(pct);

    const uint32_t rec_writers = pre_combine ? plan.n_owners : n_active;
    KernelDescriptor port_reader;
    port_reader.kernel_source = agrn_kernel_path("all_gather_rms_norm_port_reader.cpp");
    port_reader.core_ranges = port_cores;
    port_reader.compile_time_args = pct;
    {
        std::vector<std::variant<uint32_t, Buffer*>> c(port_reader_common::COUNT, 0u);
        c[port_reader_common::DST_ADDR] = dst_buffer;
        c[port_reader_common::BANK_STRIDE] = L;
        c[port_reader_common::ARRIVAL_ADDR] = ps.bulk();
        c[port_reader_common::N_ACTIVE] = n_active;
        c[port_reader_common::NOTIFY_LANDED] = is_ru ? 1u : 0u;
        KernelDescriptor::RTArgList l;
        for (const auto& v : c) {
            if (std::holds_alternative<Buffer*>(v)) {
                l.push_back(std::get<Buffer*>(v));
            } else {
                l.push_back(std::get<uint32_t>(v));
            }
        }
        port_reader.emplace_common_runtime_args(l);
    }
    port_reader.config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_0_default};

    KernelDescriptor port_sender;
    port_sender.kernel_source = agrn_kernel_path("all_gather_rms_norm_port_sender.cpp");
    port_sender.core_ranges = port_cores;
    port_sender.compile_time_args = pct;
    {
        std::vector<std::variant<uint32_t, Buffer*>> c(port_sender_common::COUNT, 0u);
        c[port_sender_common::DST_ADDR] = dst_buffer;
        c[port_sender_common::BANK_STRIDE] = L;
        c[port_sender_common::ARRIVAL_ADDR] = ps.bulk();
        c[port_sender_common::READY_ADDR] = ps.ready();
        c[port_sender_common::SLOT_BYTES] = plan.slot_bytes;
        c[port_sender_common::STATS_ARRIVED_ADDR] = ps.stats();
        c[port_sender_common::N_ACTIVE] = rec_writers;
        KernelDescriptor::RTArgList l;
        for (const auto& v : c) {
            if (std::holds_alternative<Buffer*>(v)) {
                l.push_back(std::get<Buffer*>(v));
            } else {
                l.push_back(std::get<uint32_t>(v));
            }
        }
        port_sender.emplace_common_runtime_args(l);
    }
    port_sender.config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_1_default};

    const auto own_stats = stats_sched(coord);
    const auto own_bulk = bulk_sched(coord);
    const auto src_node = mesh->get_fabric_node_id(coord);
    for (const Dir d : {Dir::FWD, Dir::BWD}) {
        const auto di = static_cast<size_t>(d);
        const auto& peer = peer_of[di];
        const auto& up = up_of[di];
        const Dir opp = d == Dir::FWD ? Dir::BWD : Dir::FWD;
        const planner::Schedule empty;
        const planner::Schedule& sends = peer.has_value() ? agrn_dir_sched(own_bulk, d) : empty;
        std::pair<planner::Schedule, planner::Schedule> up_bulk;
        if (up.has_value()) {
            up_bulk = bulk_sched(*up);
        }
        const planner::Schedule& up_sends = up.has_value() ? agrn_dir_sched(up_bulk, d) : empty;
        for (uint32_t l = 0; l < L; ++l) {
            const CoreCoord core = ch.ports[di][l];
            std::vector<uint32_t> up_counts;
            up_counts.reserve(up_sends.size());
            for (const auto& [q, part] : up_sends) {
                up_counts.push_back(walk.chunks(q, part, l, L));
            }
            const IncSim sim = agrn_inc_sim(up_counts, planner::INC_EVERY);
            std::vector<uint32_t> r(port_reader_rt::ENTRIES, 0u);
            r[port_reader_rt::FIRST_BANK] = l;
            r[port_reader_rt::EXPECT_IN] = sim.total;
            r[port_reader_rt::N_ENTRIES] = static_cast<uint32_t>(sends.size());
            for (size_t kk = 0; kk < sends.size(); ++kk) {
                const auto [q, part] = sends[kk];
                if (kk == 0) {
                    r.insert(r.end(), {q | (part << 16), 0u, 0u});
                    continue;
                }
                TT_FATAL(kk - 1 < up_sends.size(), "all_gather_rms_norm: relay entry without an upstream send");
                const auto [uq, upart] = up_sends[kk - 1];
                TT_FATAL(uq == q && (upart == part || upart == 0), "all_gather_rms_norm: relay schedule mismatch");
                const uint32_t off = (part == 2 && upart == 0) ? walk.chunks(q, 1, l, L) : 0u;
                const size_t e = kk - 1;
                r.insert(
                    r.end(),
                    {q | (part << 16),
                     sim.incs_before[e] - (sim.sent_before[e] / planner::INC_EVERY) + 1,
                     sim.sent_before[e] + off});
            }
            r.insert(r.end(), active_noc.begin(), active_noc.end());
            const uint32_t hub_sem = l == 0 ? (d == Dir::FWD ? sem::STATS_READY : sem::STATS_READY_BWD) + 1 : 0u;
            r.insert(r.end(), {hub_sem, mc0, mc1, (gx * gy) - 1});
            const std::vector<uint32_t> recv_slots = l == 0 ? recv[di] : std::vector<uint32_t>{};
            r.insert(
                r.end(),
                {ps.stats(),
                 plan.slot_bytes,
                 (l == 0 && d == Dir::FWD) ? p : 0xFFFFu,
                 rec_writers,
                 static_cast<uint32_t>(recv_slots.size())});
            r.insert(r.end(), recv_slots.begin(), recv_slots.end());
            r.insert(r.end(), plan.band_need.begin(), plan.band_need.end());
            r.insert(r.end(), plan.band_row_end.begin(), plan.band_row_end.end());
            for (uint32_t b = 0; b < own_bands; ++b) {
                r.push_back(ps.band(b));
            }
            if (sp.has_value()) {
                r.insert(r.end(), {1u, sp->lanes[l].tot, ps.stage_counter(), n_active, sp->nch_max, l, L, stage_cp});
            } else {
                r.push_back(0u);
            }
            port_reader.runtime_args.emplace_back(core, std::move(r));

            const bool is_hub = l == 0;
            std::vector<uint32_t> st_sends;
            if (peer.has_value() && is_hub) {
                st_sends = agrn_slots(agrn_dir_sched(own_stats, d));
            }
            uint32_t st_expect = 0;
            if (up.has_value() && is_hub) {
                st_expect = static_cast<uint32_t>(agrn_dir_sched(stats_sched(*up), d).size());
            }
            std::vector<uint32_t> s(port_sender_rt::STATS, 0u);
            s[port_sender_rt::FIRST_BANK] = l;
            s[port_sender_rt::SEND_READY] = peer.has_value() ? 1u : 0u;
            s[port_sender_rt::IS_HUB] = is_hub ? 1u : 0u;
            s[port_sender_rt::STATS_EXPECT_IN] = st_expect;
            s[port_sender_rt::N_STATS] = static_cast<uint32_t>(st_sends.size());
            std::optional<tt::tt_fabric::FabricNodeId> pn;
            if (peer.has_value()) {
                const auto& peer_ch = mp.chips.at(*peer);
                s[port_sender_rt::READY_XY] = agrn_pack_xy(virt(peer_ch.ports[static_cast<size_t>(opp)][l]));
                s[port_sender_rt::PEER_XY] = agrn_pack_xy(virt(peer_ch.ports[di][l]));
                pn = mesh->get_fabric_node_id(*peer);
                s[port_sender_rt::DST_MESH_ID] = static_cast<uint32_t>(*pn->mesh_id);
                s[port_sender_rt::DST_CHIP_ID] = static_cast<uint32_t>(pn->chip_id);
                s[port_sender_rt::HAS_CONN] = 1u;
            }
            s.insert(s.end(), st_sends.begin(), st_sends.end());
            s.push_back(static_cast<uint32_t>(sends.size()));
            for (const auto& [q, part] : sends) {
                s.push_back(q | (part << 16));
                s.push_back(walk.chunks(q, part, l, L));
            }
            if (pn.has_value()) {
                tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                    src_node, *pn, ch.links[di].at(l), desc, core, s, CoreType::WORKER);
            }
            port_sender.runtime_args.emplace_back(core, std::move(s));
        }
    }

    desc.kernels.resize(kernel::COUNT);
    desc.kernels[kernel::READER] = std::move(reader);
    desc.kernels[kernel::WRITER] = std::move(writer);
    desc.kernels[kernel::COMPUTE] = std::move(compute);
    desc.kernels[kernel::PORT_READER] = std::move(port_reader);
    desc.kernels[kernel::PORT_SENDER] = std::move(port_sender);
    return desc;
}

AllGatherRmsNormMeshWorkloadFactory::cached_mesh_workload_t AllGatherRmsNormMeshWorkloadFactory::create_mesh_workload(
    const AllGatherRmsNormParams& attrs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const AllGatherRmsNormInputs& tensors,
    Tensor& output) {
    const uint32_t wlt = tt::div_up(tensors.input.logical_shape()[-1], tt::constants::TILE_WIDTH);
    const auto out_page = static_cast<uint32_t>(output.buffer()->aligned_page_size());
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared;
    for (const auto& coord : tensor_coords.coords()) {
        Program program{AllGatherRmsNormProgramFactory::create_descriptor(attrs, tensors, output, coord)};
        shared_variables_t sv{
            .worker_common =
                {std::ref(GetCommonRuntimeArgs(program, kernel::READER)),
                 std::ref(GetCommonRuntimeArgs(program, kernel::WRITER)),
                 std::ref(GetCommonRuntimeArgs(program, kernel::COMPUTE))},
            .port_reader_common = std::ref(GetCommonRuntimeArgs(program, kernel::PORT_READER)),
            .port_sender_common = std::ref(GetCommonRuntimeArgs(program, kernel::PORT_SENDER)),
            // the own slice of the output shard: slot p = this chip's position along the cluster axis
            .out_block_offset = coord[attrs.cluster_axis] * wlt * out_page,
        };
        for (const auto& circular_buffer : program.circular_buffers()) {
            if (!circular_buffer->globally_allocated()) {
                continue;
            }
            const auto& indices = circular_buffer->buffer_indices();
            if (indices.contains(cb::X)) {
                sv.x_cb = circular_buffer->id();
            } else if (indices.contains(cb::X_STATS)) {
                sv.x_stats_cb = circular_buffer->id();
            } else if (indices.contains(cb::OUT_BLOCK)) {
                sv.out_block_cb = circular_buffer->id();
            }
        }
        const ttnn::MeshCoordinateRange range(coord);
        workload.add_program(range, std::move(program));
        shared.emplace(range, sv);
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared)};
}

void AllGatherRmsNormMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AllGatherRmsNormParams& attrs,
    const AllGatherRmsNormInputs& tensors,
    Tensor& output) {
    // The key pins every compile-time arg, CB size and per-core arg; a hit rewrites the tensor addresses (worker
    // common X / DST / OUT / GAMMA / BETA, port common DST_ADDR), EPS_BITS (left out of the key) and the shard-backed
    // CB addresses, the call's values read once. Per-mesh global semaphore addresses stay as create_descriptor wrote
    // them. A program whose last completed write matches the call is left untouched.
    const Tensor& input = tensors.input;
    const bool is_ru = tensors.scratch.has_value() && input.logical_shape()[-1] % tt::constants::TILE_WIDTH != 0;
    const written_t call{
        .in_addr = input.buffer()->address(),
        .dst_addr = is_ru ? tensors.scratch->buffer()->address() : output.buffer()->address(),
        .out_addr = output.buffer()->address(),
        .gamma_addr = tensors.gamma.has_value() ? tensors.gamma->buffer()->address() : 0u,
        .beta_addr = tensors.beta.has_value() ? tensors.beta->buffer()->address() : 0u,
        .eps_bits = agrn_f32_bits(attrs.epsilon)};
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        auto& sv = cached_workload.shared_variables.at(range);
        if (sv.last_written.has_value() && *sv.last_written == call) {
            continue;
        }
        sv.last_written.reset();
        for (auto& common : sv.worker_common) {
            RuntimeArgsData& args = common.get();
            args[worker_common::X_ADDR] = call.in_addr;
            args[worker_common::DST_ADDR] = call.dst_addr;
            args[worker_common::OUT_ADDR] = call.out_addr;
            args[worker_common::GAMMA_ADDR] = call.gamma_addr;
            args[worker_common::BETA_ADDR] = call.beta_addr;
            args[worker_common::EPS_BITS] = call.eps_bits;
        }
        sv.port_reader_common.get()[port_reader_common::DST_ADDR] = call.dst_addr;
        sv.port_sender_common.get()[port_sender_common::DST_ADDR] = call.dst_addr;
        for (const auto& input_cb : {sv.x_cb, sv.x_stats_cb}) {
            if (input_cb.has_value()) {
                UpdateDynamicCircularBufferAddress(program, *input_cb, *input.buffer());
            }
        }
        if (sv.out_block_cb.has_value()) {
            UpdateDynamicCircularBufferAddress(program, *sv.out_block_cb, *output.buffer(), sv.out_block_offset);
        }
        sv.last_written = call;
    }
}

}  // namespace ttnn::operations::all_gather_rms_norm
