// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/sparse_sdpa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_common.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/recipe_ring_cbs.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/recipe_state_layout.hpp"
#include "ttnn/operations/transformer/sdpa/device/recipe/sdpa_recipe_program.hpp"
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_constants.h>  // NUM_CIRCULAR_BUFFERS
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>  // GetCommonRuntimeArgs (cache-hit in-place patch)
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>  // corerange_to_cores
#include <algorithm>
#include <array>
#include <bit>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ttnn::prim {

namespace {
namespace recipes = ttnn::operations::transformer::sdpa::detail;
namespace sparse_cb = ::sparse_sdpa::recipe_cb;
using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;

// Shapes and formats shared by the layout and create_descriptor.
struct OpDims {
    bool use_attention_sink;
    bool scaled_kv;
    // Fused FP8 dequant (one FPU pass for tilize x scale, see kernels/compute/sparse_sdpa_fused_dequant.hpp). It scales
    // V by 2^OFFSET, which compute undoes on the output; an attention sink adds an unscaled term to the denominator, so
    // the sink path keeps the two-pass dequant.
    bool fused_dequant;
    uint32_t H;      // head count, from the tensor (any multiple of TILE_HEIGHT)
    uint32_t S;      // query rows
    uint32_t topk;   // indices per query row
    uint32_t k_dim;  // head dim, from the tensor (e.g. 576)
    uint32_t v_dim;  // V width (op arg; e.g. 512)
    uint32_t k_chunk;
    uint32_t DHt;
    uint32_t vDHt;
    uint32_t Skt;                // tiles per chunk along keys
    uint32_t Sqt;                // query tile-rows (32 heads each)
    uint32_t packed_page_bytes;  // scaled FP8 packed row stride (0 otherwise)
    tt::DataFormat q_df;         // native (row-major) Q format
    tt::DataFormat kv_df;        // native (row-major) KV format
};

OpDims op_dims(const SparseSDPAParams& attrs, const SparseSDPAInputs& t) {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;
    const bool use_attention_sink = t.attention_sink.has_value();
    const bool scaled_kv = attrs.has_scaled_kv();
    const uint32_t H = t.q.logical_shape()[1];
    const uint32_t k_dim = t.q.logical_shape()[3];
    return OpDims{
        .use_attention_sink = use_attention_sink,
        .scaled_kv = scaled_kv,
        .fused_dequant = scaled_kv && !use_attention_sink,
        .H = H,
        .S = t.q.logical_shape()[2],
        .topk = t.indices.logical_shape()[3],
        .k_dim = k_dim,
        .v_dim = attrs.v_dim,
        .k_chunk = attrs.k_chunk_size,
        .DHt = k_dim / TILE_WIDTH,
        .vDHt = attrs.v_dim / TILE_WIDTH,
        .Skt = attrs.k_chunk_size / TILE_WIDTH,
        .Sqt = H / TILE_HEIGHT,
        .packed_page_bytes = scaled_kv ? static_cast<uint32_t>(t.kv.mesh_buffer().aligned_page_size()) : 0u,
        .q_df = tt::tt_metal::datatype_to_dataformat_converter(t.q.dtype()),
        .kv_df = tt::tt_metal::datatype_to_dataformat_converter(t.kv.dtype()),
    };
}

// The format compute tilizes K into (the recipe's K storage): FP8 rows into BFP8 (near-lossless, half the L1 and
// cheaper matmuls), BF16 rows and the scaled cache (dequantized latent beside BF16 RoPE in one CB) into BF16.
DataType sparse_k_tile_dtype(const OpDims& d) {
    return !d.scaled_kv && d.kv_df == tt::DataFormat::Fp8_e4m3 ? DataType::BFLOAT8_B : DataType::BFLOAT16;
}

recipes::PrecisionPolicy sparse_policy(const SparseSDPAParams& attrs, const OpDims& d) {
    return recipes::resolve_precision_policy(recipes::select_recipe(attrs.precision, sparse_k_tile_dtype(d)));
}

// Gathered chunks per recipe step. A step narrower than one full QK subblock (the recipes' widest, 4 tiles:
// recipe_subblock_width) pays the step's fixed cost for less work: on BH, Qwen 3.8 QSA decode (32 heads, 1 row,
// top-2048 of 4096, D 512) took 194 / 112 / 100 / 101 us at 1 / 2 / 4 / 8-tile steps. Compute tilizes `group` chunks
// into one step (the reader still gathers chunk by chunk); a token's last group may hold fewer chunks (masked tail).
uint32_t sparse_step_group(const OpDims& d) {
    constexpr uint32_t kWidestQkSubblockTiles = 4;
    if (d.Skt >= kWidestQkSubblockTiles) {
        return 1;
    }
    const uint32_t n_chunks = d.topk / d.k_chunk;
    return std::min((kWidestQkSubblockTiles + d.Skt - 1) / d.Skt, n_chunks);
}

// The layout of one plan: the recipe compute CBs (recipe_compute_program) resized for the sparse kernel, the sparse
// dataflow CBs (sparse_sdpa_common.hpp recipe_cb), and the recipe compute kernel (named arguments, defines, config).
struct SparseLayout {
    tt::tt_metal::ProgramDescriptor::CBDescriptors cbs;
    tt::tt_metal::KernelDescriptor compute;
    uint32_t q_tiles = 0;  // Q tile rows compute processes per pass (an odd pass padded where the recipe needs it)
    bool v_in_k = true;    // V read in place from the K^T chunk
};

SparseLayout sparse_make_layout(
    const SparseSDPAParams& attrs,
    const SparseSDPAInputs& t,
    tt::tt_metal::DataType out_dtype,
    bool split_k,
    uint32_t passes,
    bool fused,
    uint32_t group,
    uint32_t q_rm_rows,
    uint32_t k_rm_rows) {
    const OpDims d = op_dims(attrs, t);
    const auto policy = sparse_policy(attrs, d);
    const bool fp32 = policy.fp32_destination;
    const uint32_t pass_rows = (d.Sqt + passes - 1) / passes;
    const uint32_t step_kt = group * d.Skt;  // K tiles per recipe step
    SparseLayout layout;
    layout.q_tiles = recipes::recipe_compute_q_tiles(policy, pass_rows, step_kt);
    const uint32_t q_tiles = layout.q_tiles;
    // In-place V needs PV subblocks at least as wide as the BF16 recipes' two-row groups; a narrower V gets its own CB.
    layout.v_in_k = fp32 || recipes::recipe_subblock_width(d.vDHt) >= 2;
    const tt::DataFormat q_format =
        d.q_df == tt::DataFormat::Fp8_e4m3 ? tt::DataFormat::Bfp8_b : tt::DataFormat::Float16_b;
    const auto& grid = attrs.worker_core_grid;
    auto program = recipes::recipe_compute_program(
        policy, grid, 1, q_tiles, step_kt, d.DHt, attrs.scale, d.vDHt, std::nullopt, q_format);
    layout.compute = std::move(program.kernels.front());
    auto& cbs = program.cbs;
    // Compute tilizes Q and K itself, so one entry each; the output CB holds a pass's rows (compute untilizes them
    // after the step that normalizes them), BF16 whatever Q's type.
    for (auto& c : cbs) {
        auto& format = c.format_descriptors.front();
        if (format.buffer_index == recipe_ring_cb::q) {
            c.total_size = q_tiles * d.DHt * format.page_size;
        } else if (format.buffer_index == recipe_ring_cb::k) {
            c.total_size = step_kt * d.DHt * format.page_size;
        } else if (format.buffer_index == recipe_ring_cb::v) {
            c.total_size = step_kt * d.vDHt * format.page_size;
        } else if (format.buffer_index == recipe_ring_cb::out) {
            format.data_format = tt::DataFormat::Float16_b;
            format.page_size = tt::tile_size(tt::DataFormat::Float16_b);
            c.total_size = q_tiles * d.vDHt * format.page_size;
        }
    }
    if (!layout.v_in_k) {
        // Compute tilizes V beside K without a pack reconfigure (sparse_sdpa_compute.cpp produce_k_chunk).
        const auto format_of = [&](uint32_t id) {
            for (const auto& c : cbs) {
                if (c.format_descriptors.front().buffer_index == id) {
                    return c.format_descriptors.front().data_format;
                }
            }
            return tt::DataFormat::Invalid;
        };
        TT_FATAL(format_of(recipe_ring_cb::v) == format_of(recipe_ring_cb::k), "sparse_sdpa: V's CB needs K's format");
    }
    if (layout.v_in_k) {
        cbs.erase(
            std::remove_if(
                cbs.begin(),
                cbs.end(),
                [](const CBDescriptor& c) { return c.format_descriptors.front().buffer_index == recipe_ring_cb::v; }),
            cbs.end());
    }
    if (!fused) {
        recipes::recipe_drop_fused(cbs, layout.compute.named_compile_time_args, q_tiles);
    }
    const auto add_cb = [&](uint32_t id, uint32_t page_size, uint32_t num_pages, tt::DataFormat df) {
        cbs.push_back(CBDescriptor{
            .total_size = page_size * num_pages,
            .core_ranges = grid,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(id), .data_format = df, .page_size = page_size}}},
        });
    };
    constexpr tt::DataFormat bf = tt::DataFormat::Float16_b;
    constexpr uint32_t tile_bytes = 2048;
    // Q rows: two tile rows of heads, so the reader runs one ahead of compute's tilize (one, or none: through the K row
    // CB, at the L1 floor).
    if (q_rm_rows > 0) {
        add_cb(sparse_cb::q_rm, d.k_dim * t.q.element_size(), q_rm_rows * tt::constants::TILE_HEIGHT, d.q_df);
    }
    if (d.scaled_kv) {
        // One double-buffered allocation with an owning FP8 FIFO and a format-only BF16 RoPE alias. Both use
        // the physical packed-page stride. Only sparse_cb::k_rm advances FIFO state; compute derives the alias read
        // pointer from the owner before reading RoPE. This lets the reader gather slab N+1 while compute
        // reconstructs slab N without independent alias pointers drifting at the ring wrap.
        // k_rm_rows: the slabs the reader may run ahead (0: two; a step's rows or two steps' when L1 allows).
        const uint32_t slabs =
            k_rm_rows == 0 ? ::sparse_sdpa::CB_DOUBLE_BUFFER_DEPTH : k_rm_rows / tt::constants::TILE_HEIGHT;
        cbs.push_back(CBDescriptor{
            .total_size = d.packed_page_bytes * tt::constants::TILE_HEIGHT * slabs,
            .core_ranges = grid,
            .format_descriptors = {{
                CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(sparse_cb::k_rm),
                    .data_format = d.kv_df,
                    .page_size = d.packed_page_bytes},
                CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(sparse_cb::k_rope_rm),
                    .data_format = bf,
                    .page_size = d.packed_page_bytes},
            }},
        });
        const uint32_t scale_pages = ::sparse_sdpa::scale_block_count(d.v_dim) * slabs;
        if (d.fused_dequant) {
            // FP16 scale tiles. Declared UInt16: the JIT rejects an FP16 (A-exponent) CB next to the BF16 ones, and
            // compute reprograms srcB to Float16 for the fused multiply.
            add_cb(sparse_cb::k_scale, tt::tile_size(tt::DataFormat::UInt16), scale_pages, tt::DataFormat::UInt16);
        } else {
            add_cb(sparse_cb::k_scale, tt::tile_size(tt::DataFormat::Float32), scale_pages, tt::DataFormat::Float32);
            add_cb(sparse_cb::k_latent, tile_bytes, d.vDHt, bf);
        }
    } else {
        add_cb(sparse_cb::k_rm, d.k_dim * t.kv.element_size(), k_rm_rows == 0 ? d.k_chunk : k_rm_rows, d.kv_df);
    }
    const tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(out_dtype);
    add_cb(sparse_cb::out_rm, tt::tile_size(out_df), q_tiles * d.vDHt, out_df);
    add_cb(sparse_cb::idx, d.topk * t.indices.element_size(), 1, bf);
    add_cb(sparse_cb::ctrl, ::sparse_sdpa::control_message::PAGE_BYTES, ::sparse_sdpa::CB_DOUBLE_BUFFER_DEPTH, bf);
    add_cb(sparse_cb::kreq, ::sparse_sdpa::gather_request::PAGE_BYTES, ::sparse_sdpa::CB_DOUBLE_BUFFER_DEPTH, bf);
    add_cb(sparse_cb::kack, ::sparse_sdpa::ACK_PAGE_BYTES, ::sparse_sdpa::CB_DOUBLE_BUFFER_DEPTH, bf);
    if (d.use_attention_sink) {
        add_cb(sparse_cb::sink, tile_bytes, passes * q_tiles, bf);
        add_cb(sparse_cb::sink_scratch, d.H * sizeof(uint16_t), 1, bf);
    }
    const tt::DataFormat state_df = fp32 ? tt::DataFormat::Float32 : bf;
    if (split_k) {
        // The HEAD's checkpoint request / ack (StateTransfer words), the neighbour's state planes, c_b and the merge
        // control page.
        using Transfer = ::sdpa::streaming::StateTransfer;
        constexpr uint32_t request_bytes = ::sparse_sdpa::align_cb_page(Transfer::Words * sizeof(uint32_t));
        add_cb(sparse_cb::checkpoint_request, request_bytes, 1, tt::DataFormat::UInt32);
        add_cb(sparse_cb::checkpoint_ack, request_bytes, 1, tt::DataFormat::UInt32);
        add_cb(sparse_cb::recv_out, 4096, q_tiles * d.vDHt, tt::DataFormat::Float32);
        add_cb(sparse_cb::recv_max, tt::tile_size(state_df), q_tiles, state_df);
        add_cb(sparse_cb::recv_sum, 4096, q_tiles, tt::DataFormat::Float32);
        add_cb(sparse_cb::merge_corr, tt::tile_size(state_df), q_tiles, state_df);
        add_cb(sparse_cb::merge_ctrl, ::sparse_sdpa::control_message::PAGE_BYTES, 1, tt::DataFormat::UInt32);
    }
    // Compute: FP32 state CBs unpack to dest (the recipe sets its own); FP8 rows tilize through the FP32 dest.
    auto& config = std::get<tt::tt_metal::ComputeConfigDescriptor>(layout.compute.config);
    config.unpack_to_dest_mode.resize(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (fp32 && split_k) {
        for (uint32_t id : {sparse_cb::recv_out, sparse_cb::recv_max, sparse_cb::recv_sum, sparse_cb::merge_corr}) {
            config.unpack_to_dest_mode[id] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        }
    }
    if (d.kv_df == tt::DataFormat::Fp8_e4m3) {
        config.unpack_to_dest_mode[sparse_cb::k_rm] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    if (d.q_df == tt::DataFormat::Fp8_e4m3) {
        config.unpack_to_dest_mode[sparse_cb::q_rm] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    layout.cbs = std::move(cbs);
    return layout;
}

uint64_t sparse_layout_l1_bytes(const SparseLayout& layout, uint64_t cb_align) {
    uint64_t bytes = 0;
    for (const auto& c : layout.cbs) {
        bytes += (c.total_size + cb_align - 1) / cb_align * cb_align;
    }
    return bytes;
}

// Relative split-K gather speed (%) of Blackhole NoC0 rows y = FIRST_ROW_Y.. when every core gathers K at once: the
// top rows are the slowest and the bottom row the fastest, the same on every chip measured. Tuned on a p150 12x10
// grid for the GLM-5.3 (64 heads) and Kimi K3.1 (96 heads) shapes. Rows outside the table weigh 100.
constexpr std::array<uint32_t, 10> ROW_SPEED_PCT = {85, 84, 88, 97, 102, 107, 110, 112, 114, 115};
constexpr uint32_t FIRST_ROW_Y = 2;

// Split-K unit boundaries: core i owns units [bounds[i], bounds[i + 1]) of the token-major (token, chunk) order.
// Equal split, or (row_weighted) in proportion to ROW_SPEED_PCT so the slow top rows do not finish last. The
// weighted split falls back to the equal one when any core would get less than one token (the S > cores pairwise
// merge needs a whole token per core).
std::vector<uint64_t> split_k_unit_bounds(
    const tt::tt_metal::IDevice& device,
    const std::vector<CoreCoord>& cores,
    uint32_t num_used,
    uint64_t total_units,
    uint32_t n_chunks,
    bool row_weighted) {
    std::vector<uint64_t> cum_w(num_used + 1, 0);  // cum_w[i] = weight of cores [0, i)
    for (uint32_t i = 0; i < num_used; ++i) {
        uint64_t w = 1;
        if (row_weighted) {
            const uint32_t r = device.worker_core_from_logical_core(cores[i]).y - FIRST_ROW_Y;  // wraps for y < 2
            w = r < ROW_SPEED_PCT.size() ? ROW_SPEED_PCT[r] : 100u;
        }
        cum_w[i + 1] = cum_w[i] + w;
    }
    std::vector<uint64_t> bounds(num_used + 1);
    for (uint32_t i = 0; i <= num_used; ++i) {
        bounds[i] = total_units * cum_w[i] / cum_w[num_used];
    }
    if (row_weighted) {
        for (uint32_t i = 0; i < num_used; ++i) {
            if (bounds[i + 1] - bounds[i] < n_chunks) {
                return split_k_unit_bounds(device, cores, num_used, total_units, n_chunks, /*row_weighted=*/false);
            }
        }
    }
    return bounds;
}

SparseSDPAOperation::SparseSDPAProgramFactory::Plan sparse_resolve_plan(
    const SparseSDPAParams& attrs, const SparseSDPAInputs& t) {
    using Plan = SparseSDPAOperation::SparseSDPAProgramFactory::Plan;
    auto* device = t.q.device();
    const OpDims d = op_dims(attrs, t);
    const auto out_dtype = SparseSDPAOperation::compute_output_specs(attrs, t).data_type();
    // ProgramImpl::allocate_circular_buffers places each CB at the DRAM alignment, not the L1 one.
    const uint64_t cb_align = device->allocator()->get_alignment(tt::tt_metal::BufferType::DRAM);
    const uint64_t l1_available = recipes::recipe_free_l1(*device);
    const auto fits = [&](bool split_k, const Plan& p) {
        return sparse_layout_l1_bytes(
                   sparse_make_layout(
                       attrs, t, out_dtype, split_k, p.passes, p.fused, p.group, p.q_rm_rows, p.k_rm_rows),
                   cb_align) <= l1_available;
    };
    // The fewest head passes that fit L1 (each pass gathers the token's keys again), then grouped steps, fused chunks,
    // the reader's Q run-ahead and whole-chunk K gathers if they fit. None of these change the numerics (a step group
    // only adds masked keys to a step). Unscaled KV gathers 32 key rows at a time below a chunk (two, then one).
    Plan plan{.split_k = false, .num_used = attrs.worker_core_grid.num_cores()};
    bool found = false;
    const uint32_t group = sparse_step_group(d);
    // K row staging: one step's rows when the default holds less (compute tilizes a step's rows before its step, so the
    // reader then gathers the next step's while compute runs this one), the default (one chunk; scaled KV two slabs),
    // then (unscaled KV) 64 or 32 rows. With less than a step staged the gather serialized with the step (Qwen 3.8 QSA
    // decode, four 32-key chunks per step: 190 -> 148 us; GLM S40 scaled: 319 -> 294 us). Two steps' rows measured the
    // same or slower (Kimi K3.1 FP8 H64 S160: 870 vs 837 us).
    const auto k_rm_options = [&](uint32_t g) {
        std::vector<uint32_t> options;
        const uint32_t step_rows = g * d.k_chunk;
        const uint32_t default_rows =
            d.scaled_kv ? ::sparse_sdpa::CB_DOUBLE_BUFFER_DEPTH * tt::constants::TILE_HEIGHT : d.k_chunk;
        if (step_rows > default_rows) {
            options.push_back(step_rows);
        }
        options.push_back(0);
        for (const uint32_t rows : {2 * tt::constants::TILE_HEIGHT, tt::constants::TILE_HEIGHT}) {
            if (!d.scaled_kv && rows < d.k_chunk) {
                options.push_back(rows);
            }
        }
        return options;
    };
    // Q through the K row CB: unscaled KV in Q's row format (a K row stage holds at least one tile row of Q).
    std::vector<uint32_t> q_rm_options{2, 1};
    if (!d.scaled_kv && d.q_df == d.kv_df) {
        q_rm_options.push_back(0);
    }
    for (uint32_t passes = 1; passes <= d.Sqt && !found; ++passes) {
        for (const uint32_t g : {group, 1u}) {
            for (const bool fused : {true, false}) {
                for (const uint32_t q_rm_rows : q_rm_options) {
                    for (const uint32_t k_rm_rows : k_rm_options(g)) {
                        const Plan p{
                            .split_k = false,
                            .num_used = plan.num_used,
                            .passes = passes,
                            .fused = fused,
                            .group = g,
                            .q_rm_rows = q_rm_rows,
                            .k_rm_rows = k_rm_rows};
                        if (!found && fits(false, p)) {
                            plan = p;
                            found = true;
                        }
                    }
                }
            }
        }
    }
    TT_FATAL(found, "sparse_sdpa: one Q tile row of {} heads does not fit L1 ({} B free)", d.H, l1_available);
    // Split-K: deal (token, chunk) units contiguously over the cores when whole tokens do not divide evenly. GLM per
    // chip has 160 query rows: on 96 cores the whole-token split leaves 64 cores with 2 tokens and 32 with 1
    // (makespan 2 tokens); split-K gives every core 26-27 of the 2560 chunk units (1.67 tokens). Merges are pairwise
    // between neighbouring cores, at most one per core. One head pass per token.
    const uint32_t S = d.S;
    const uint32_t n_chunks = d.topk / d.k_chunk;
    const uint32_t num_cores = attrs.worker_core_grid.num_cores();
    // BF16 and plain FP8 KV: split-K only when every token gets two cores (2S <= cores); with more rows it was
    // slower in most measured shapes.
    const bool split_k_ok =
        plan.passes == 1 && !d.use_attention_sink && d.Sqt <= 4 && n_chunks > 1 && (d.scaled_kv || 2 * S <= num_cores);
    Plan split = plan;
    split.split_k = split_k_ok && S > num_cores && (S % num_cores) != 0;
    // Small S (fewer query rows than cores, e.g. GLM chunk 1024/2048: 32/64 rows per chip on 96 cores): the
    // whole-token split leaves most cores idle at one token-time (~330 us, compute-bound). Deal the (token, chunk)
    // units over `num_used` <= min(num_cores, 2S) cores instead, so every token straddles at most two neighbouring
    // cores and the pairwise HEAD -> TAIL merge applies: every core range must start at a token boundary or run to one
    // (no range strictly inside a token). num_used is the largest such count; S=32 -> 64 cores (half a token each),
    // S=64 on 96 cores -> 96 (2/3).
    if (split_k_ok && S < num_cores) {
        const uint64_t units = static_cast<uint64_t>(S) * n_chunks;
        const auto valid = [&](uint32_t m) {
            for (uint32_t i = 0; i < m; ++i) {
                const uint64_t b0 = units * i / m, b1 = units * (i + 1) / m;
                if (b1 == b0 || (b0 % n_chunks != 0 && b1 < (b0 / n_chunks + 1) * n_chunks)) {
                    return false;
                }
            }
            return true;
        };
        for (uint32_t m = std::min<uint32_t>(num_cores, 2 * S); m > S; --m) {
            if (valid(m)) {
                split.split_k = true;
                split.num_used = m;
                break;
            }
        }
    }
    // Keep the whole-token split when the merge CBs do not fit in the free L1.
    // Split-K takes the deepest K staging that fits beside its merge CBs (a whole-token split instead costs far more
    // than shallower staging: Kimi K3.1 H96 S160 989 vs 1225 us).
    if (split.split_k) {
        bool shallower = false;
        for (const uint32_t k_rm_rows : k_rm_options(plan.group)) {
            shallower = shallower || k_rm_rows == plan.k_rm_rows;
            split.k_rm_rows = k_rm_rows;
            if (shallower && fits(true, split)) {
                return split;
            }
        }
    }
    return plan;
}

}  // namespace

SparseSDPAOperation::SparseSDPAProgramFactory::Plan SparseSDPAOperation::SparseSDPAProgramFactory::select_plan(
    const SparseSDPAParams& attrs, const SparseSDPAInputs& t) {
    // Memoized: the plan builds candidate layouts, and a cache hit hashes it. The key holds everything the plan reads
    // (the device, its free L1, the op's scalars and the input specs it uses), as one flat array: hashing the tensor
    // shapes and the core range set per call cost more host time than the legacy op's hit path.
    const auto& qs = t.q.logical_shape();
    const std::array<uint32_t, 17> fields{
        attrs.v_dim,
        static_cast<uint32_t>(attrs.kv_format),
        attrs.k_chunk_size,
        static_cast<uint32_t>(attrs.precision),
        attrs.worker_core_grid.num_cores(),
        static_cast<uint32_t>(qs.rank()),
        qs[-3],
        qs[-2],
        qs[-1],
        static_cast<uint32_t>(t.q.dtype()),
        static_cast<uint32_t>(t.kv.dtype()),
        static_cast<uint32_t>(t.kv.mesh_buffer().aligned_page_size()),
        static_cast<uint32_t>(t.kv.element_size()),
        t.indices.logical_shape()[-1],
        static_cast<uint32_t>(t.indices.element_size()),
        static_cast<uint32_t>(t.attention_sink.has_value()),
        static_cast<uint32_t>(attrs.has_scaled_kv())};
    auto* device = t.q.device();
    const uint64_t free_l1 = recipes::recipe_free_l1(*device);
    // A repeated call (the steady state of a model's layers) compares against this thread's last call before hashing.
    struct LastCall {
        std::array<uint32_t, 17> fields{};
        int device_id = -1;
        uint64_t free_l1 = 0;
        Plan plan;
    };
    thread_local LastCall last;
    if (last.device_id == static_cast<int>(device->id()) && last.free_l1 == free_l1 && last.fields == fields) {
        return last.plan;
    }
    const auto key = ttsl::hash::hash_objects_with_default_seed(device->id(), free_l1, fields);
    static std::mutex mutex;
    static std::unordered_map<std::size_t, Plan> memo;
    Plan plan;
    bool hit = false;
    {
        const std::lock_guard lock(mutex);
        if (const auto it = memo.find(key); it != memo.end()) {
            plan = it->second;
            hit = true;
        }
    }
    if (!hit) {
        plan = sparse_resolve_plan(attrs, t);
        const std::lock_guard lock(mutex);
        if (memo.size() >= 4096) {
            memo.clear();
        }
        memo.emplace(key, plan);
    }
    last = LastCall{.fields = fields, .device_id = static_cast<int>(device->id()), .free_l1 = free_l1, .plan = plan};
    return plan;
}

tt::tt_metal::ProgramDescriptor SparseSDPAOperation::SparseSDPAProgramFactory::create_descriptor(
    const SparseSDPAParams& attrs, const SparseSDPAInputs& t, Tensor& output) {
    tt::tt_metal::ProgramDescriptor desc;

    const OpDims d = op_dims(attrs, t);
    const uint32_t H = d.H, S = d.S, topk = d.topk, k_dim = d.k_dim, v_dim = d.v_dim, k_chunk = d.k_chunk;
    const uint32_t DHt = d.DHt, vDHt = d.vDHt, Skt = d.Skt, packed_page_bytes = d.packed_page_bytes;
    const auto use_attention_sink = static_cast<uint32_t>(d.use_attention_sink);
    const auto scaled_kv = static_cast<uint32_t>(d.scaled_kv);
    const auto fused_dequant = static_cast<uint32_t>(d.fused_dequant);
    // Element sizes come from the tensors (no hardcoded byte counts); passed to the kernels as compile args.
    const uint32_t q_elem_bytes = t.q.element_size();
    const uint32_t kv_elem_bytes = t.kv.element_size();
    const uint32_t idx_elem_bytes = t.indices.element_size();  // uint32 -> 4
    const uint32_t out_elem_bytes = output.element_size();

    auto* device = t.q.device();
    const auto& core_grid = attrs.worker_core_grid;
    const auto cores = corerange_to_cores(core_grid, std::nullopt, /*row_wise=*/true);
    const uint32_t num_cores = cores.size();

    const uint32_t n_chunks = topk / k_chunk;
    const Plan plan = select_plan(attrs, t);
    const auto split_k = static_cast<uint32_t>(plan.split_k);
    const uint32_t num_used = plan.num_used;
    SparseLayout layout = sparse_make_layout(
        attrs, t, output.dtype(), plan.split_k, plan.passes, plan.fused, plan.group, plan.q_rm_rows, plan.k_rm_rows);
    const uint32_t q_tiles = layout.q_tiles;
    desc.cbs = std::move(layout.cbs);
    if (plan.split_k) {
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = 0, .core_type = tt::CoreType::WORKER, .core_ranges = core_grid, .initial_value = 0});
    }
    const auto policy = sparse_policy(attrs, d);

    // ---- compile-time args ----
    // Each block's order is the kernel's *_ct_arg enum in sparse_sdpa_common.hpp; the TensorAccessorArgs chain last.
    // The cache layout is compile-time so the remap is fully optimized out for natural-order KV.
    std::array<uint32_t, 5> block_cyclic_ct{0, 1, 1, 0, 0};
    if (attrs.has_block_cyclic()) {
        const auto& bc = attrs.block_cyclic.value();
        const uint32_t seq_len_local = t.kv.logical_shape()[2] / bc.sp;
        block_cyclic_ct = {1, bc.chunk_local, bc.sp, seq_len_local - bc.chunk_local, bc.chunk_local * (bc.sp - 1)};
    }

    std::vector<uint32_t> reader_ct = {H, S, topk, k_chunk, k_dim, q_elem_bytes, kv_elem_bytes, idx_elem_bytes};
    reader_ct.insert(reader_ct.end(), block_cyclic_ct.begin(), block_cyclic_ct.end());
    reader_ct.insert(reader_ct.end(), {scaled_kv, v_dim, packed_page_bytes, split_k, fused_dequant});
    reader_ct.insert(
        reader_ct.end(),
        {q_tiles,
         plan.passes,
         plan.k_rm_rows == 0 || plan.k_rm_rows >= k_chunk ? k_chunk : 32u,
         plan.q_rm_rows == 0 ? 1u : 0u});
    TT_FATAL(
        reader_ct.size() == ::sparse_sdpa::reader_ct_arg::END,
        "sparse_sdpa reader compile-time argument layout is out of sync");
    // kv (the K cache) uses a RUNTIME tensor shape: its T dimension (the cache length) is passed as common
    // runtime args, NOT compile-time args, so changing T reuses the same program (no recompile). q/indices
    // stay compile-time — their dims define the program. The accessor's runtime metadata is the same on
    // every core, so it rides on the kernel's common (not per-core) runtime args.
    std::vector<uint32_t> reader_crt;
    tt::tt_metal::TensorAccessorArgs(t.q.buffer()).append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(t.kv.buffer(), tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(t.indices.buffer()).append_to(reader_ct, reader_crt);

    // The writer is the lighter dataflow kernel, so it builds the persistent compute-input tiles.
    std::vector<uint32_t> writer_ct = {H, S, vDHt, out_elem_bytes};
    writer_ct.insert(writer_ct.end(), block_cyclic_ct.begin(), block_cyclic_ct.end());
    writer_ct.insert(writer_ct.end(), {scaled_kv, k_dim, kv_elem_bytes, packed_page_bytes, use_attention_sink});
    writer_ct.insert(
        writer_ct.end(),
        {split_k, fused_dequant, q_tiles, plan.passes, static_cast<uint32_t>(policy.fp32_destination)});
    TT_FATAL(
        writer_ct.size() == ::sparse_sdpa::writer_ct_arg::END,
        "sparse_sdpa writer compile-time argument layout is out of sync");
    std::vector<uint32_t> writer_crt;
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_ct, writer_crt);
    tt::tt_metal::TensorAccessorArgs(t.kv.buffer(), tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(writer_ct, writer_crt);
    tt::tt_metal::TensorAccessorArgs(d.use_attention_sink ? t.attention_sink->buffer() : nullptr)
        .append_to(writer_ct, writer_crt);

    // ---- kernels ----
    const auto make_kernel =
        [&](const char* source, std::vector<uint32_t> ct, tt::tt_metal::KernelDescriptor::ConfigDescriptor config) {
            tt::tt_metal::KernelDescriptor k;
            k.kernel_source = std::string("ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/") + source;
            k.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
            k.core_ranges = core_grid;
            k.compile_time_args = std::move(ct);
            k.config = std::move(config);
            return k;
        };
    auto reader_desc =
        make_kernel("dataflow/sparse_sdpa_reader.cpp", std::move(reader_ct), tt::tt_metal::ReaderConfigDescriptor{});
    auto writer_desc =
        make_kernel("dataflow/sparse_sdpa_writer.cpp", std::move(writer_ct), tt::tt_metal::WriterConfigDescriptor{});
    // The recipe compute: its named arguments (after the L1 choice above), compute config (fidelity, FP32 DEST and
    // unpack-to-dest modes) and defines, with the sparse kernel's own. Sparse masks only the sentinel tail of a
    // token's last chunk (recipe_ring_mask 0) and keeps each pass's state resident.
    auto compute_args = recipes::recipe_named_args(layout.compute.named_compile_time_args);
    recipes::set_named_arg(compute_args, "recipe_ring", 1);
    if (d.use_attention_sink) {
        recipes::set_named_arg(compute_args, "recipe_sink_cb", sparse_cb::sink);
        recipes::set_named_arg(compute_args, "recipe_sink_bf16", 1);
        recipes::set_named_arg(compute_args, "recipe_sink_rows", 1);
    }
    compute_args.emplace_back("q_tiles", q_tiles);
    compute_args.emplace_back("head_tiles", d.Sqt);
    compute_args.emplace_back("passes", plan.passes);
    compute_args.emplace_back("k_tiles", plan.group * Skt);  // one recipe step
    compute_args.emplace_back("gather_tiles", Skt);          // one gathered chunk
    compute_args.emplace_back("d_tiles", DHt);
    compute_args.emplace_back("vd_tiles", vDHt);
    compute_args.emplace_back("scale", std::bit_cast<uint32_t>(attrs.scale));
    compute_args.emplace_back("v_in_k", layout.v_in_k ? 1u : 0u);
    compute_args.emplace_back("scaled_kv", scaled_kv);
    compute_args.emplace_back("fused_dequant", fused_dequant);
    compute_args.emplace_back("packed_row_bytes", packed_page_bytes);
    compute_args.emplace_back("split_k", split_k);
    compute_args.emplace_back("q_via_k_rm", plan.q_rm_rows == 0 ? 1u : 0u);
    auto compute_desc = make_kernel("compute/sparse_sdpa_compute.cpp", {}, layout.compute.config);
    compute_desc.named_compile_time_args = std::move(compute_args);
    compute_desc.defines = layout.compute.defines;
    // Every thread builds at -O2: the tilizes and the dequant beside the recipe step put the FP32 recipes' unpack image
    // at -O3 4.6 KB over the kernel config buffer (BALANCED, scaled FP8, 64 heads).
    compute_desc.opt_level = recipes::recipe_opt_level(tt::tt_metal::KernelBuildOptLevel::O2);

    // Indexed KV cache: the gather page ids are offset by cache_batch_idx * T to select the cache's batch
    // slot (0 when not indexed). The cache-hit override re-derives this from the current attrs/tensor T.
    const uint32_t kv_batch_page_offset = attrs.cache_batch_idx.value_or(0) * t.kv.logical_shape()[2];
    reader_desc.emplace_common_runtime_args({t.q.buffer(), t.kv.buffer(), t.indices.buffer(), kv_batch_page_offset});
    writer_desc.emplace_common_runtime_args(
        {output.buffer(),
         t.kv.buffer(),
         kv_batch_page_offset,
         d.use_attention_sink ? t.attention_sink->buffer() : nullptr});
    TT_FATAL(
        reader_desc.common_runtime_args.size() == ::sparse_sdpa::reader_common_arg::END &&
            writer_desc.common_runtime_args.size() == ::sparse_sdpa::writer_common_arg::END,
        "sparse_sdpa common runtime argument layout is out of sync");
    reader_desc.common_runtime_args.insert(reader_desc.common_runtime_args.end(), reader_crt.begin(), reader_crt.end());
    writer_desc.common_runtime_args.insert(writer_desc.common_runtime_args.end(), writer_crt.begin(), writer_crt.end());

    // ---- per-core work: whole tokens, or split-K (token, chunk) unit ranges ----
    const uint32_t base = S / num_cores;
    const uint32_t extra = S % num_cores;
    std::vector<uint64_t> unit_bounds;
    if (plan.split_k) {
        const bool row_weighted = S > num_cores && device->arch() == tt::ARCH::BLACKHOLE;
        unit_bounds =
            split_k_unit_bounds(*device, cores, num_used, static_cast<uint64_t>(S) * n_chunks, n_chunks, row_weighted);
    }
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores[i];
        uint32_t tok_start = i * base + std::min(i, extra);
        uint32_t tok_count = base + (i < extra ? 1u : 0u);
        uint32_t head_c0 = 0, tail_c1 = 0, prev_x = 0, prev_y = 0;
        if (plan.split_k && i >= num_used) {
            tok_start = 0;  // small-S split-K: cores past num_used stay idle (no segments)
            tok_count = 0;
        } else if (plan.split_k) {
            // Core i owns units [b0, b1) of the token-major (token, chunk) order.
            const uint64_t b0 = unit_bounds[i];
            const uint64_t b1 = unit_bounds[i + 1];
            const uint32_t t0 = b0 / n_chunks, c0 = b0 % n_chunks, t1 = b1 / n_chunks, c1 = b1 % n_chunks;
            // S > cores: >= one token of units per core. S < cores: a range may be shorter than a token but must
            // touch a token boundary, so every token straddles at most two cores (pairwise merge).
            TT_FATAL(
                S < num_cores ? (c0 == 0 || b1 >= (static_cast<uint64_t>(t0) + 1) * n_chunks) : b1 - b0 >= n_chunks,
                "sparse_sdpa split-K: a core range lies strictly inside one token (would need a multi-way merge)");
            head_c0 = c0;                       // HEAD = token t0, chunks [c0, n_chunks)
            tok_start = c0 != 0 ? t0 + 1 : t0;  // FULL tokens [tok_start, t1)
            tok_count = t1 - tok_start;
            tail_c1 = c1;  // TAIL = token t1, chunks [0, c1)
            if (head_c0 != 0) {
                TT_FATAL(i > 0, "sparse_sdpa split-K: core 0 cannot have a HEAD segment");
                const auto prev = device->worker_core_from_logical_core(cores[i - 1]);
                prev_x = prev.x;
                prev_y = prev.y;
            }
        }
        for (auto* kernel : {&reader_desc, &writer_desc, &compute_desc}) {
            kernel->emplace_runtime_args(core, {tok_start, tok_count, head_c0, tail_c1, prev_x, prev_y});
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

void SparseSDPAOperation::SparseSDPAProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const SparseSDPAParams& operation_attributes,
    const SparseSDPAInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Token assignments are fixed by the hashed q shape. Refresh only the uniform dispatch values;
    // tensor-accessor metadata follows these slots and remains unchanged under the cache key.
    const auto& t = tensor_args;
    const uint32_t offset = operation_attributes.cache_batch_idx.value_or(0) * t.kv.logical_shape()[2];
    const uint32_t kv_addr = t.kv.mesh_buffer().address();
    auto& reader_args = tt::tt_metal::GetCommonRuntimeArgs(program, 0);
    auto& writer_args = tt::tt_metal::GetCommonRuntimeArgs(program, 1);
    reader_args[::sparse_sdpa::reader_common_arg::Q_ADDRESS] = t.q.mesh_buffer().address();
    reader_args[::sparse_sdpa::reader_common_arg::KV_ADDRESS] = kv_addr;
    reader_args[::sparse_sdpa::reader_common_arg::INDICES_ADDRESS] = t.indices.mesh_buffer().address();
    reader_args[::sparse_sdpa::reader_common_arg::KV_BATCH_PAGE_OFFSET] = offset;
    writer_args[::sparse_sdpa::writer_common_arg::OUTPUT_ADDRESS] = tensor_return_value.mesh_buffer().address();
    writer_args[::sparse_sdpa::writer_common_arg::KV_ADDRESS] = kv_addr;
    writer_args[::sparse_sdpa::writer_common_arg::KV_BATCH_PAGE_OFFSET] = offset;
    writer_args[::sparse_sdpa::writer_common_arg::ATTENTION_SINK_ADDRESS] =
        t.attention_sink.has_value() ? t.attention_sink->mesh_buffer().address() : 0u;
}

}  // namespace ttnn::prim
