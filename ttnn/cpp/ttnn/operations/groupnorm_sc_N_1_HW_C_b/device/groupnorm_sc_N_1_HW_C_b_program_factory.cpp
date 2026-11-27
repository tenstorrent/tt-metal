// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_sc_N_1_HW_C_b_program_factory.hpp"

#include <array>
#include <bit>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "groupnorm_sc_N_1_HW_C_b_planner.hpp"
#include "kernels/groupnorm_sc_N_1_HW_C_b_args.hpp"
#include "ttnn/kernel_lib/mcast/host/mcast.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::groupnorm_sc_N_1_HW_C_b {

using namespace tt::tt_metal;
namespace gargs = ::gn_sc_args;

namespace {

constexpr const char* kGnKernelDir = "ttnn/cpp/ttnn/operations/groupnorm_sc_N_1_HW_C_b/device/kernels/";

// RM stick CB depth in chunks (Python D_RM); the reader's output share holds one chunk (OUT_READER_CHUNKS).
constexpr uint32_t kGnRmStickChunks = 2;
constexpr uint32_t kGnOutReaderChunks = 1;
// Per-chunk device zones are compiled in only when a core's chunk steps fit the marker budget (Python
// CHUNK_ZONE_MAX_STEPS).
constexpr uint32_t kGnChunkZoneMaxSteps = 16;

std::string gn_kernel_path(const char* name) { return std::string(kGnKernelDir) + name; }

struct GnCbFormat {
    uint32_t id;
    tt::DataFormat format;
    uint32_t page;
};

CBDescriptor gn_make_cb(uint32_t total, const CoreRangeSet& cores, std::initializer_list<GnCbFormat> formats) {
    CBDescriptor d{.total_size = total, .core_ranges = cores};
    for (const auto& f : formats) {
        d.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(f.id), .data_format = f.format, .page_size = f.page});
    }
    return d;
}

// Zero-copy: the CB is the core's own shard; its pages are input-dtype tiles over it (whole tile rows only).
CBDescriptor gn_shard_cb(
    const Tensor& tensor, uint32_t in_tile, uint32_t in_elem, std::initializer_list<GnCbFormat> formats) {
    const auto& spec = *tensor.memory_config().shard_spec();
    const uint32_t shard_bytes = spec.shape[0] * spec.shape[1] * in_elem;
    const uint32_t total = (shard_bytes / in_tile) * in_tile;
    CBDescriptor d =
        ttnn::cb_descriptor_from_sharded_tensor(static_cast<uint8_t>(formats.begin()->id), tensor, 0, total);
    d.format_descriptors.clear();
    for (const auto& f : formats) {
        d.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(f.id), .data_format = f.format, .page_size = f.page});
    }
    return d;
}

CoreRangeSet gn_core_set(const std::vector<CoreCoord>& cores) {
    std::vector<CoreRange> ranges;
    ranges.reserve(cores.size());
    for (const auto& c : cores) {
        ranges.emplace_back(c, c);
    }
    return CoreRangeSet(ranges).merge_ranges();
}

// The first num_cores cores of the grid in row-major order (Python _used_core_range_set).
CoreRangeSet gn_used_cores(uint32_t num_cores, uint32_t grid_x) {
    std::vector<CoreRange> ranges;
    const uint32_t full_rows = num_cores / grid_x;
    const uint32_t rem = num_cores % grid_x;
    if (full_rows > 0) {
        ranges.emplace_back(CoreCoord(0, 0), CoreCoord(grid_x - 1, full_rows - 1));
    }
    if (rem > 0) {
        ranges.emplace_back(CoreCoord(0, full_rows), CoreCoord(rem - 1, full_rows));
    }
    return CoreRangeSet(ranges);
}

// DRAM interleaved: odd grid columns of member cores take the swapped NoC pairing.
uint32_t gn_swapped(const planner::Plan& plan, const planner::CoreWork& wk) {
    return (!plan.sharded && wk.is_root == 0 && wk.core_x % 2 == 1) ? 1u : 0u;
}

Buffer* gn_opt_buffer(const std::optional<Tensor>& t) { return t.has_value() ? t->buffer() : nullptr; }

// The program of a cache miss, and the kernel layout its descriptor follows.
ProgramDescriptor gn_descriptor(
    const GroupnormScN1HwCbParams& attrs,
    const GroupnormScN1HwCbInputs& tensors,
    Tensor& output,
    KernelLayout& layout) {
    TT_FATAL(
        gargs::cb::MAX_ID < hal::get_num_dataflow_buffers(),
        "groupnorm_sc_N_1_HW_C_b: circular-buffer id {} exceeds the {} circular buffers of this architecture",
        gargs::cb::MAX_ID,
        hal::get_num_dataflow_buffers());
    const Tensor& input = tensors.input;
    const auto& gamma = tensors.gamma;
    const auto& beta = tensors.beta;
    auto* device = input.device();
    const planner::PlanInputs inputs = planner::plan_inputs(attrs, tensors);
    const planner::Plan& plan = planner::solve_plan(inputs);
    const std::vector<planner::CoreWork> work = planner::build_work(inputs, plan);

    const uint32_t W = plan.W;
    const uint32_t T = plan.T;
    const uint32_t R = plan.R;
    const uint32_t h = plan.h;
    const bool is_rm = plan.is_rm;
    const uint32_t hg = plan.has_gamma ? 1 : 0;
    const uint32_t hb = plan.has_beta ? 1 : 0;
    const bool sharded = plan.sharded;
    const auto& fmt = plan.fmt;
    TT_FATAL(
        fmt.in_dtype == input.dtype() && output.dtype() == input.dtype(),
        "groupnorm_sc_N_1_HW_C_b: the plan's formats must match the input and output dtypes");
    TT_FATAL(
        !plan.in_place || output.buffer() == input.buffer(),
        "groupnorm_sc_N_1_HW_C_b: in place, the output must be the input");

    const DataType in_dtype = fmt.in_dtype;
    const tt::DataFormat in_df = cb_dataformat_for(in_dtype);
    const uint32_t in_tile = tile_size(in_dtype);
    const uint32_t in_elem = in_tile / tt::constants::TILE_HW;  // sharded only (bf16): 2
    const bool in_bfp8 = in_dtype == DataType::BFLOAT8_B;
    const bool aff_bfp8 = fmt.aff_dtype == DataType::BFLOAT8_B;
    const DataType amap_dtype = aff_bfp8 ? DataType::BFLOAT16 : fmt.aff_dtype;
    const uint32_t amap_tile = tile_size(amap_dtype);
    const uint32_t aff_elem = tt::datum_size(datatype_to_dataformat_converter(amap_dtype));
    const tt::DataFormat bf16_df = tt::DataFormat::Float16_b;
    const tt::DataFormat f32_df = tt::DataFormat::Float32;
    const uint32_t bf16_tile = tile_size(DataType::BFLOAT16);
    const uint32_t f32_tile = tile_size(DataType::FLOAT32);
    const uint32_t map_period = plan.map_period;
    const uint32_t vc_last = plan.vc_last != 0 ? plan.vc_last : map_period;

    const CoreRangeSet cores = sharded ? input.memory_config().shard_spec()->grid
                                       : gn_used_cores(static_cast<uint32_t>(work.size()), plan.grid_x);

    ProgramDescriptor desc;

    // ---------------- Circular buffers (the Python order) ----------------
    namespace cb = gargs::cb;
    if (is_rm) {
        if (sharded) {
            desc.cbs.push_back(gn_shard_cb(input, in_tile, in_elem, {{cb::INPUT_RM, in_df, in_tile}}));
            desc.cbs.push_back(gn_shard_cb(output, in_tile, in_elem, {{cb::OUTPUT_RM, in_df, in_tile}}));
        } else {
            desc.cbs.push_back(gn_make_cb(kGnRmStickChunks * T * in_tile, cores, {{cb::INPUT_RM, in_df, in_tile}}));
        }
    }
    if (sharded && !is_rm) {
        desc.cbs.push_back(gn_shard_cb(
            input, in_tile, in_elem, {{cb::INPUT_TILES, in_df, in_tile}, {cb::INPUT_KEEP, in_df, in_tile}}));
        desc.cbs.push_back(gn_shard_cb(output, in_tile, in_elem, {{cb::OUTPUT_TILES, in_df, in_tile}}));
    } else {
        const uint32_t ring_pages =
            plan.ring_pages != 0 ? plan.ring_pages : planner::input_ring_pages(T, plan.K_alloc, plan.stream, is_rm);
        desc.cbs.push_back(gn_make_cb(
            ring_pages * in_tile, cores, {{cb::INPUT_TILES, in_df, in_tile}, {cb::INPUT_KEEP, in_df, in_tile}}));
        if (!sharded) {
            const uint32_t chunk_bytes = T * in_tile;
            const uint32_t seq_bytes = gargs::OUT_SEQ_PAGES * gargs::OUT_SEQ_PAGE_BYTES;
            desc.cbs.push_back(
                gn_make_cb(plan.out_writer_chunks * chunk_bytes, cores, {{cb::OUTPUT_TILES, in_df, in_tile}}));
            desc.cbs.push_back(gn_make_cb(kGnOutReaderChunks * chunk_bytes, cores, {{cb::OUTPUT_B, in_df, in_tile}}));
            desc.cbs.push_back(
                gn_make_cb(seq_bytes, cores, {{cb::OUT_SEQ_A, tt::DataFormat::UInt32, gargs::OUT_SEQ_PAGE_BYTES}}));
            desc.cbs.push_back(
                gn_make_cb(seq_bytes, cores, {{cb::OUT_SEQ_B, tt::DataFormat::UInt32, gargs::OUT_SEQ_PAGE_BYTES}}));
        }
    }
    desc.cbs.push_back(
        gn_make_cb(W * f32_tile, cores, {{cb::SUM_ACC, f32_df, f32_tile}, {cb::CHAN_SCALE, f32_df, f32_tile}}));
    desc.cbs.push_back(
        gn_make_cb(W * f32_tile, cores, {{cb::SUMSQ_ACC, f32_df, f32_tile}, {cb::CHAN_WORK, f32_df, f32_tile}}));
    desc.cbs.push_back(gn_make_cb(W * bf16_tile, cores, {{cb::GROUP_MAP, bf16_df, bf16_tile}}));
    desc.cbs.push_back(gn_make_cb(W * bf16_tile, cores, {{cb::PILOT_ROWS, bf16_df, bf16_tile}}));
    desc.cbs.push_back(gn_make_cb(bf16_tile, cores, {{cb::NEG_ONE, bf16_df, bf16_tile}}));
    desc.cbs.push_back(gn_make_cb(f32_tile, cores, {{cb::GROUP_SHIFT, f32_df, f32_tile}}));
    if (hg != 0) {
        desc.cbs.push_back(
            gn_make_cb((hg + hb) * W * amap_tile, cores, {{cb::GAMMA_ROWS, cb_dataformat_for(amap_dtype), amap_tile}}));
    }
    const uint32_t NU = plan.NU;
    const uint32_t NSEG = plan.NSEG;
    desc.cbs.push_back(gn_make_cb(2 * NSEG * f32_tile, cores, {{cb::LOCAL_PARTIALS, f32_df, f32_tile}}));
    desc.cbs.push_back(gn_make_cb(2 * R * NU * f32_tile, cores, {{cb::GATHERED_PARTIALS, f32_df, f32_tile}}));
    desc.cbs.push_back(gn_make_cb(2 * NU * f32_tile, cores, {{cb::STATS_OUT, f32_df, f32_tile}}));
    desc.cbs.push_back(gn_make_cb((2 * NU + 1) * f32_tile, cores, {{cb::GROUP_STATS, f32_df, f32_tile}}));
    const uint32_t stage = planner::staging_bytes(plan);
    desc.cbs.push_back(gn_make_cb(stage, cores, {{cb::AFFINE_STAGING, bf16_df, stage}}));
    const bool has_stage = plan.stage_pages > 0;
    if (has_stage) {
        desc.cbs.push_back(gn_make_cb(
            plan.stage_pages * in_tile,
            cores,
            {{cb::RM_STAGE_IN, in_df, in_tile}, {cb::RM_STAGE_OUT, in_df, in_tile}}));
    }

    // ---------------- Compile-time args ----------------
    const uint32_t in_page_bytes = static_cast<uint32_t>(input.buffer()->page_size());
    const uint32_t out_page_bytes = static_cast<uint32_t>(output.buffer()->page_size());
    const std::optional<Tensor>& aff_ref = gamma.has_value() ? gamma : beta;
    const uint32_t aff_page_bytes = aff_ref.has_value() ? static_cast<uint32_t>(aff_ref->buffer()->page_size()) : 0u;
    const uint32_t chunk_zones =
        (attrs.chunk_zones && plan.B_max + (plan.B_max - std::min(plan.K_alloc, plan.B_max)) <= kGnChunkZoneMaxSteps)
            ? 1u
            : 0u;

    std::vector<uint32_t> reader_ct(gargs::reader_ct::COUNT);
    {
        namespace rc = gargs::reader_ct;
        reader_ct[rc::W] = W;
        reader_ct[rc::CHUNK_HT] = h;
        reader_ct[rc::IS_RM] = is_rm ? 1 : 0;
        reader_ct[rc::HAS_GAMMA] = hg;
        reader_ct[rc::HAS_BETA] = hb;
        reader_ct[rc::R] = R;
        reader_ct[rc::CG] = plan.Cg;
        reader_ct[rc::WT_TOTAL] = plan.Wt;
        reader_ct[rc::HT] = plan.Ht;
        reader_ct[rc::GX] = plan.gx;
        reader_ct[rc::IN_PAGE_BYTES] = in_page_bytes;
        reader_ct[rc::MAP_PERIOD] = map_period;
        reader_ct[rc::SHARDED] = sharded ? 1 : 0;
        reader_ct[rc::CHUNK_ZONES] = chunk_zones;
        reader_ct[rc::HAS_STAGE] = has_stage ? 1 : 0;
        reader_ct[rc::OPS] = plan.ops_path ? 1 : 0;
        reader_ct[rc::IN_TILE_BYTES] = in_tile;
        reader_ct[rc::AFF_ELEM] = aff_elem;
        reader_ct[rc::AFF_TILE] = fmt.aff_tile ? 1 : 0;
        reader_ct[rc::AFF_PAGE_BYTES] = aff_page_bytes;
        reader_ct[rc::HW] = plan.HW;
        reader_ct[rc::VC_LAST] = vc_last;
        reader_ct[rc::IN_BFP8] = in_bfp8 ? 1 : 0;
        reader_ct[rc::AFF_BFP8] = aff_bfp8 ? 1 : 0;
        TensorAccessorArgs(input.buffer()).append_to(reader_ct);
        TensorAccessorArgs(gn_opt_buffer(gamma)).append_to(reader_ct);
        TensorAccessorArgs(gn_opt_buffer(beta)).append_to(reader_ct);
        reader_ct.push_back(out_page_bytes);
        TensorAccessorArgs(output.buffer()).append_to(reader_ct);
    }

    auto compute_ct = [&](uint32_t is_root) {
        namespace cc = gargs::compute_ct;
        std::vector<uint32_t> ct(cc::COUNT);
        ct[cc::W] = W;
        ct[cc::CHUNK_HT] = h;
        ct[cc::IS_RM] = is_rm ? 1 : 0;
        ct[cc::HAS_GAMMA] = hg;
        ct[cc::HAS_BETA] = hb;
        ct[cc::R] = R;
        ct[cc::SHARDED] = sharded ? 1 : 0;
        ct[cc::IS_ROOT] = is_root;
        ct[cc::CHUNK_ZONES] = chunk_zones;
        ct[cc::NU] = NU;
        ct[cc::MAP_PERIOD] = map_period;
        ct[cc::COL_MASK] = plan.col_mask ? 1 : 0;
        ct[cc::HAS_STAGE] = has_stage ? 1 : 0;
        ct[cc::OPS] = plan.ops_path ? 1 : 0;
        ct[cc::GX] = plan.gx;
        ct[cc::VC_LAST] = vc_last;
        return ct;
    };

    auto writer_ct = [&](uint32_t is_root) {
        namespace wc = gargs::writer_ct;
        std::vector<uint32_t> ct(wc::COUNT);
        ct[wc::W] = W;
        ct[wc::CHUNK_HT] = h;
        ct[wc::R] = R;
        ct[wc::WT_TOTAL] = plan.Wt;
        ct[wc::HT] = plan.Ht;
        ct[wc::GX] = plan.gx;
        ct[wc::OUT_PAGE_BYTES] = out_page_bytes;
        ct[wc::IS_ROOT] = is_root;
        ct[wc::IS_RM] = is_rm ? 1 : 0;
        ct[wc::SHARDED] = sharded ? 1 : 0;
        ct[wc::CHUNK_ZONES] = chunk_zones;
        ct[wc::NU] = NU;
        ct[wc::HAS_STAGE] = has_stage ? 1 : 0;
        ct[wc::OPS] = plan.ops_path ? 1 : 0;
        ct[wc::MAP_PERIOD] = map_period;
        ct[wc::VC_LAST] = vc_last;
        TensorAccessorArgs(output.buffer()).append_to(ct);
        return ct;
    };

    const uint32_t inv_n_bits =
        std::bit_cast<uint32_t>(static_cast<float>(1.0 / (static_cast<double>(plan.HW) * plan.Cg)));
    const uint32_t eps_bits = std::bit_cast<uint32_t>(attrs.eps);

    // ---------------- Kernels: readers by NoC pairing, writers by (role, pairing), computes by role ----------------
    using RtList = std::vector<std::variant<uint32_t, Buffer*>>;
    std::array<KernelDescriptor, 2> readers;
    std::array<std::vector<CoreCoord>, 2> reader_cores;
    std::array<std::array<KernelDescriptor, 2>, 2> writers;  // [is_root][sw]
    std::array<std::array<std::vector<CoreCoord>, 2>, 2> writer_cores;
    std::array<KernelDescriptor, 2> computes;  // [is_root]
    std::array<std::vector<CoreCoord>, 2> compute_cores;

    for (const auto& wk : work) {
        const CoreCoord core(wk.core_x, wk.core_y);
        const uint32_t sw = gn_swapped(plan, wk);
        const uint32_t b_core = tt::div_up(wk.rows_core, h);
        const uint32_t k_core = std::min(plan.K_alloc, b_core);
        const CoreCoord root_virt = device->worker_core_from_logical_core(CoreCoord(wk.root_x, wk.root_y));

        reader_cores[sw].push_back(core);
        {
            namespace rr = gargs::reader_rt;
            RtList args(rr::COLS_TAIL, 0u);
            args[rr::IN_ADDR] = input.buffer();
            args[rr::GAMMA_ADDR] = gn_opt_buffer(gamma);
            args[rr::BETA_ADDR] = gn_opt_buffer(beta);
            args[rr::UNIT_START] = wk.unit_start;
            args[rr::NUM_BLOCKS] = wk.num_blocks;
            args[rr::ROW_START] = wk.row_start;
            args[rr::ROWS_CORE] = wk.rows_core;
            args[rr::RESIDENT_CHUNKS] = k_core;
            args[rr::IS_ROOT] = wk.is_root;
            args[rr::ROOT_X] = static_cast<uint32_t>(root_virt.x);
            args[rr::ROOT_Y] = static_cast<uint32_t>(root_virt.y);
            args[rr::OUT_ADDR] = output.buffer();
            args[rr::XS] = wk.xs;
            args[rr::VC] = wk.vc;
            args[rr::L0] = wk.l0;
            args[rr::OFF0] = wk.off0;
            args[rr::G_UNIT] = wk.g_unit;
            args[rr::RM_CREDIT_PAGES] = wk.rm_credit_pages;
            args[rr::N_COLS] = static_cast<uint32_t>(wk.pilot_cols.size());
            for (uint32_t cx : wk.pilot_cols) {
                args.emplace_back(
                    static_cast<uint32_t>(device->worker_core_from_logical_core(CoreCoord(cx, wk.root_y)).x));
            }
            args.emplace_back(static_cast<uint32_t>(wk.staged.size()));
            for (const auto& trip : wk.staged) {
                for (uint32_t v : trip) {
                    args.emplace_back(v);
                }
            }
            readers[sw].emplace_runtime_args(core, args);
        }
        {
            namespace cr = gargs::compute_rt;
            std::vector<uint32_t> args(cr::OPS_TAIL);
            args[cr::NUM_BLOCKS] = wk.num_blocks;
            args[cr::ROWS_CORE] = wk.rows_core;
            args[cr::RESIDENT_CHUNKS] = k_core;
            args[cr::INV_N_BITS] = inv_n_bits;
            args[cr::EPS_BITS] = eps_bits;
            args[cr::N_IMG] = wk.n_img;
            args[cr::SEG_IMG0] = wk.seg_img0;
            args[cr::W_VALID] = wk.w_valid;
            args[cr::VC] = wk.vc;
            args[cr::UNIT_START] = wk.unit_start;
            args[cr::N_OPS] = static_cast<uint32_t>(wk.ops.size());
            args.insert(args.end(), wk.ops.begin(), wk.ops.end());
            computes[wk.is_root].runtime_args.emplace_back(core, std::move(args));
            compute_cores[wk.is_root].push_back(core);
        }
        {
            namespace wr = gargs::writer_rt;
            RtList args(wr::STAGED_TAIL, 0u);
            args[wr::OUT_ADDR] = output.buffer();
            args[wr::UNIT_START] = wk.unit_start;
            args[wr::NUM_BLOCKS] = wk.num_blocks;
            args[wr::ROW_START] = wk.row_start;
            args[wr::ROWS_CORE] = wk.rows_core;
            args[wr::RESIDENT_CHUNKS] = k_core;
            args[wr::SLOT] = wk.slot;
            args[wr::ROOT_X] = static_cast<uint32_t>(root_virt.x);
            args[wr::ROOT_Y] = static_cast<uint32_t>(root_virt.y);
            args[wr::CPI] = plan.cpi;  // gather arrivals: the cores of the unit
            args[wr::N_IMG] = wk.n_img;
            args[wr::N_SEG] = wk.n_seg;
            args[wr::SEG_IMG0] = wk.seg_img0;
            args[wr::RM_CREDIT_PAGES] = wk.rm_credit_pages;
            args[wr::N_STAGED] = static_cast<uint32_t>(wk.staged.size());
            for (const auto& trip : wk.staged) {
                for (uint32_t v : trip) {
                    args.emplace_back(v);
                }
            }
            writers[wk.is_root][sw].emplace_runtime_args(core, args);
            writer_cores[wk.is_root][sw].push_back(core);
        }
    }

    std::vector<KernelDescriptor> reader_kernels;
    for (uint32_t sw = 0; sw < 2; ++sw) {
        if (reader_cores[sw].empty()) {
            continue;
        }
        KernelDescriptor& k = readers[sw];
        k.kernel_source = gn_kernel_path("groupnorm_sc_N_1_HW_C_b_reader.cpp");
        k.core_ranges = gn_core_set(reader_cores[sw]);
        k.compile_time_args = reader_ct;
        k.config = DataMovementConfigDescriptor{
            .processor = DataMovementProcessor::RISCV_1, .noc = sw != 0 ? NOC::NOC_1 : NOC::NOC_0};
        reader_kernels.push_back(std::move(k));
    }
    std::vector<KernelDescriptor> writer_kernels;
    for (uint32_t role : {1u, 0u}) {
        for (uint32_t sw = 0; sw < 2; ++sw) {
            if (writer_cores[role][sw].empty()) {
                continue;
            }
            KernelDescriptor& k = writers[role][sw];
            k.kernel_source = gn_kernel_path("groupnorm_sc_N_1_HW_C_b_writer.cpp");
            k.core_ranges = gn_core_set(writer_cores[role][sw]);
            k.compile_time_args = writer_ct(role);
            k.config = DataMovementConfigDescriptor{
                .processor = DataMovementProcessor::RISCV_0, .noc = sw != 0 ? NOC::NOC_0 : NOC::NOC_1};
            writer_kernels.push_back(std::move(k));
        }
    }
    std::vector<UnpackToDestMode> unpack_modes(hal::get_num_dataflow_buffers(), UnpackToDestMode::Default);
    unpack_modes[cb::SUM_ACC] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[cb::SUMSQ_ACC] = UnpackToDestMode::UnpackToDestFp32;
    std::vector<KernelDescriptor> compute_kernels;
    for (uint32_t role : {1u, 0u}) {
        if (compute_cores[role].empty()) {
            continue;
        }
        KernelDescriptor& k = computes[role];
        k.kernel_source = gn_kernel_path("groupnorm_sc_N_1_HW_C_b_compute.cpp");
        k.core_ranges = gn_core_set(compute_cores[role]);
        k.compile_time_args = compute_ct(role);
        k.config = ComputeConfigDescriptor{
            .math_fidelity = MathFidelity::HiFi3,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = false,
            .unpack_to_dest_mode = unpack_modes,
            .math_approx_mode = false,
        };
        compute_kernels.push_back(std::move(k));
    }

    // ---------------- Semaphores and the stats multicast ----------------
    const CoreCoord first_core = cores.ranges().front().start_coord;
    const auto gather_id = desc.find_available_semaphore_id(first_core, tt::CoreType::WORKER);
    TT_FATAL(
        gather_id == gargs::GATHER_SEMAPHORE_ID,
        "groupnorm_sc_N_1_HW_C_b: the gather semaphore must take id {}",
        gargs::GATHER_SEMAPHORE_ID);
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = gargs::GATHER_SEMAPHORE_ID, .core_type = tt::CoreType::WORKER, .core_ranges = cores, .initial_value = 0});
    const auto mcast_first_id = desc.find_available_semaphore_id(first_core, tt::CoreType::WORKER);

    const ttnn::kernel_lib::host::Mcast mcast(
        *device,
        ttnn::kernel_lib::host::McastConfig{.noc = NOC::NOC_1, .handshake = true},
        cores,
        plan.cpi,
        ttnn::kernel_lib::host::McastFixedSenderConfig{.sender_index = 0},
        sharded && !plan.mcast_row_major ? ttnn::kernel_lib::host::McastCoreOrder::ColumnMajor
                                         : ttnn::kernel_lib::host::McastCoreOrder::RowMajor);
    std::vector<std::reference_wrapper<KernelDescriptor>> writer_refs;
    writer_refs.reserve(writer_kernels.size());
    for (auto& k : writer_kernels) {
        writer_refs.emplace_back(k);
    }
    TT_FATAL(mcast_first_id.has_value(), "groupnorm_sc_N_1_HW_C_b: no free semaphore id for the stats multicast");
    mcast.attach(desc, "mcast", writer_refs, *mcast_first_id);

    layout.num_readers = static_cast<uint32_t>(reader_kernels.size());
    layout.num_writers = static_cast<uint32_t>(writer_kernels.size());
    layout.num_computes = static_cast<uint32_t>(compute_kernels.size());
    for (auto& k : reader_kernels) {
        desc.kernels.push_back(std::move(k));
    }
    for (auto& k : writer_kernels) {
        desc.kernels.push_back(std::move(k));
    }
    for (auto& k : compute_kernels) {
        desc.kernels.push_back(std::move(k));
    }
    TT_FATAL(
        desc.kernels.size() == layout.num_readers + layout.num_writers + layout.num_computes,
        "groupnorm_sc_N_1_HW_C_b: kernel count out of sync with the kernel layout");
    return desc;
}

}  // namespace

ProgramDescriptor GroupnormScN1HwCbProgramFactory::create_descriptor(
    const GroupnormScN1HwCbParams& attrs, const GroupnormScN1HwCbInputs& tensors, Tensor& output) {
    KernelLayout layout;
    return gn_descriptor(attrs, tensors, output, layout);
}

GroupnormScN1HwCbMeshWorkloadFactory::cached_mesh_workload_t GroupnormScN1HwCbMeshWorkloadFactory::create_mesh_workload(
    const GroupnormScN1HwCbParams& attrs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const GroupnormScN1HwCbInputs& tensors,
    Tensor& output) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared;
    for (const auto& range : tensor_coords.ranges()) {
        shared_variables_t sv;
        const ProgramDescriptor desc = gn_descriptor(attrs, tensors, output, sv.layout);
        Program program{desc};
        // The shard-backed CBs, found once by CB index: a hit re-points them by handle.
        for (const auto& circular_buffer : program.circular_buffers()) {
            if (!circular_buffer->globally_allocated()) {
                continue;
            }
            const auto& indices = circular_buffer->buffer_indices();
            const bool backs_input = indices.contains(gargs::cb::INPUT_RM) || indices.contains(gargs::cb::INPUT_TILES);
            sv.shard_cbs.push_back(ShardCb{.handle = circular_buffer->id(), .backs_input = backs_input});
        }
        workload.add_program(range, std::move(program));
        shared.emplace(range, std::move(sv));
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared)};
}

void GroupnormScN1HwCbMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const GroupnormScN1HwCbParams& attrs,
    const GroupnormScN1HwCbInputs& tensors,
    Tensor& output) {
    // The key (compute_program_hash) pins every per-core arg except the tensor addresses (reader IN / GAMMA / BETA /
    // OUT, writer OUT) and compute EPS_BITS (eps is left out of the key); these, and the shard-backed CBs, are
    // rewritten here, in a program whose record differs from the call. The kernel indices come from the layout the
    // miss recorded.
    Buffer* const in_buffer = tensors.input.buffer();
    Buffer* const out_buffer = output.buffer();
    const GroupnormScN1HwCbHitValues call{
        .in_addr = in_buffer->address(),
        .gamma_addr = tensors.gamma.has_value() ? tensors.gamma->buffer()->address() : 0u,
        .beta_addr = tensors.beta.has_value() ? tensors.beta->buffer()->address() : 0u,
        .out_addr = out_buffer->address(),
        .eps_bits = std::bit_cast<uint32_t>(attrs.eps)};

    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        shared_variables_t& sv = cached_workload.shared_variables.at(range);
        if (sv.applied == call) {
            continue;
        }
        // Cleared before the first write and set after the last, so a hit that stops part-way leaves no record.
        const std::optional<GroupnormScN1HwCbHitValues> previous = std::exchange(sv.applied, std::nullopt);
        const KernelLayout& layout = sv.layout;

        for (uint32_t k = 0; k < layout.num_readers; ++k) {
            auto& grid = GetRuntimeArgs(program, k);
            for (auto& column : grid) {
                for (auto& args : column) {
                    if (args.size() == 0) {
                        continue;
                    }
                    args[gargs::reader_rt::IN_ADDR] = call.in_addr;
                    args[gargs::reader_rt::GAMMA_ADDR] = call.gamma_addr;
                    args[gargs::reader_rt::BETA_ADDR] = call.beta_addr;
                    args[gargs::reader_rt::OUT_ADDR] = call.out_addr;
                }
            }
        }
        for (uint32_t k = layout.first_writer(); k < layout.first_compute(); ++k) {
            auto& grid = GetRuntimeArgs(program, k);
            for (auto& column : grid) {
                for (auto& args : column) {
                    if (args.size() != 0) {
                        args[gargs::writer_rt::OUT_ADDR] = call.out_addr;
                    }
                }
            }
        }
        for (uint32_t k = layout.first_compute(); k < layout.first_compute() + layout.num_computes; ++k) {
            auto& grid = GetRuntimeArgs(program, k);
            for (auto& column : grid) {
                for (auto& args : column) {
                    if (args.size() != 0) {
                        args[gargs::compute_rt::EPS_BITS] = call.eps_bits;
                    }
                }
            }
        }

        // A shard-backed CB is re-pointed only when its tensor moved since the last completed hit (always after the
        // miss, which leaves no record).
        const bool in_moved = !previous.has_value() || previous->in_addr != call.in_addr;
        const bool out_moved = !previous.has_value() || previous->out_addr != call.out_addr;
        for (const ShardCb& shard_cb : sv.shard_cbs) {
            if (shard_cb.backs_input ? in_moved : out_moved) {
                UpdateDynamicCircularBufferAddress(
                    program, shard_cb.handle, shard_cb.backs_input ? *in_buffer : *out_buffer);
            }
        }
        sv.applied = call;
    }
}

}  // namespace ttnn::operations::groupnorm_sc_N_1_HW_C_b
