// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deltanet_prefill_program_factory.hpp"

#include "deltanet_prefill_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::deltanet {

namespace prefill_factory {

constexpr uint32_t kTileSize = tt::constants::TILE_WIDTH;

constexpr auto kReaderPath =
    "ttnn/cpp/ttnn/operations/experimental/deltanet/device/kernels/dataflow/reader_deltanet_prefill.cpp";
constexpr auto kComputePath =
    "ttnn/cpp/ttnn/operations/experimental/deltanet/device/kernels/compute/deltanet_prefill_compute.cpp";
constexpr auto kWriterPath =
    "ttnn/cpp/ttnn/operations/experimental/deltanet/device/kernels/dataflow/writer_deltanet_prefill.cpp";

// CB indices — same as S=1 full fused kernel plus additions
constexpr auto kCbStateA      = tt::CBIndex::c_0;   // ping-pong state A
constexpr auto kCbQ           = tt::CBIndex::c_1;
constexpr auto kCbK           = tt::CBIndex::c_2;
constexpr auto kCbV           = tt::CBIndex::c_3;
constexpr auto kCbG           = tt::CBIndex::c_4;
constexpr auto kCbBeta        = tt::CBIndex::c_5;
constexpr auto kCbOutput      = tt::CBIndex::c_6;
constexpr auto kCbStateB      = tt::CBIndex::c_7;   // ping-pong state B
constexpr auto kCbZ           = tt::CBIndex::c_8;
constexpr auto kCbNormW       = tt::CBIndex::c_9;
constexpr auto kCbStateMid    = tt::CBIndex::c_16;
constexpr auto kCbKT          = tt::CBIndex::c_17;
constexpr auto kCbConvScratch = tt::CBIndex::c_18;
constexpr auto kCbConvStateOut= tt::CBIndex::c_19;
constexpr auto kCbEps         = tt::CBIndex::c_10;
constexpr auto kCbScaler      = tt::CBIndex::c_20;
constexpr auto kCbRawOut      = tt::CBIndex::c_21;
constexpr auto kCbOutputAccum = tt::CBIndex::c_22;   // writer output accumulation
constexpr auto kCbStateInit   = tt::CBIndex::c_23;   // reader-produced initial state (avoids producer conflict with compute on A/B)
constexpr auto kCbTmp0        = tt::CBIndex::c_24;
constexpr auto kCbTmp1        = tt::CBIndex::c_25;
constexpr auto kCbAcc         = tt::CBIndex::c_26;

tt::tt_metal::CBHandle make_cb(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& cores,
    uint32_t cb_index,
    tt::DataFormat fmt,
    uint32_t num_tiles) {
    uint32_t tile_size = tt::tile_size(fmt);
    auto config = tt::tt_metal::CircularBufferConfig(num_tiles * tile_size, {{cb_index, fmt}})
                      .set_page_size(cb_index, tile_size);
    return tt::tt_metal::CreateCircularBuffer(program, cores, config);
}

}  // namespace prefill_factory

DeltaNetPrefillFullProgramFactory::cached_program_t DeltaNetPrefillFullProgramFactory::create(
    const DeltaNetPrefillFullProgramFactory::operation_attributes_t& attrs,
    const DeltaNetPrefillFullProgramFactory::tensor_args_t& inputs,
    DeltaNetPrefillFullProgramFactory::tensor_return_value_t& outputs) {
    using namespace tt::tt_metal;
    namespace pf = prefill_factory;

    const uint32_t H = attrs.num_heads;
    const uint32_t Hk = attrs.num_k_heads;
    const uint32_t Dk = attrs.k_head_dim;
    const uint32_t Dv = attrs.v_head_dim;
    const uint32_t Dk_tiles = Dk / pf::kTileSize;
    const uint32_t Dv_tiles = Dv / pf::kTileSize;
    const uint32_t state_tiles = Dk_tiles * Dv_tiles;
    const uint32_t conv_dim = attrs.conv_dim;
    const uint32_t conv_k = attrs.conv_kernel_size;
    const uint32_t S = attrs.seq_len;

    auto* device = inputs.recurrent_state.device();
    Program program{};

    tt::DataFormat data_fmt = datatype_to_dataformat_converter(inputs.qkv_proj.dtype());

    auto grid = device->compute_with_storage_grid_size();
    TT_FATAL(H <= grid.x * grid.y, "Need {} cores for {} heads, grid is {}x{}", H, H, grid.x, grid.y);

    std::vector<CoreRange> core_ranges;
    for (uint32_t h = 0; h < H; h++) {
        CoreCoord core = {h % grid.x, h / grid.x};
        core_ranges.emplace_back(core, core);
    }
    CoreRangeSet all_cores(core_ranges);

    // CB allocation — all bf16 uniform format
    // Per-token CBs are double-buffered (2x) so the reader can push token s+1
    // while compute still processes token s.
    pf::make_cb(program, all_cores, pf::kCbStateInit, data_fmt, state_tiles);
    pf::make_cb(program, all_cores, pf::kCbStateA, data_fmt, state_tiles);
    pf::make_cb(program, all_cores, pf::kCbStateB, data_fmt, state_tiles);
    pf::make_cb(program, all_cores, pf::kCbQ, data_fmt, 2 * Dk_tiles);
    pf::make_cb(program, all_cores, pf::kCbK, data_fmt, 2 * Dk_tiles);
    pf::make_cb(program, all_cores, pf::kCbV, data_fmt, 2 * Dv_tiles);
    pf::make_cb(program, all_cores, pf::kCbG, data_fmt, 2);
    pf::make_cb(program, all_cores, pf::kCbBeta, data_fmt, 2);
    pf::make_cb(program, all_cores, pf::kCbOutput, data_fmt, 2 * Dv_tiles);
    pf::make_cb(program, all_cores, pf::kCbZ, data_fmt, 2 * Dv_tiles);
    pf::make_cb(program, all_cores, pf::kCbNormW, data_fmt, Dv_tiles);
    pf::make_cb(program, all_cores, pf::kCbStateMid, data_fmt, state_tiles);
    pf::make_cb(program, all_cores, pf::kCbKT, data_fmt, 2 * Dk_tiles);
    pf::make_cb(program, all_cores, pf::kCbConvScratch, data_fmt, 24);  // 12 state + 12 weight
    pf::make_cb(program, all_cores, pf::kCbConvStateOut, data_fmt, 12);
    pf::make_cb(program, all_cores, pf::kCbScaler, data_fmt, 1);
    pf::make_cb(program, all_cores, pf::kCbEps, data_fmt, 1);
    pf::make_cb(program, all_cores, pf::kCbRawOut, data_fmt, Dv_tiles);
    pf::make_cb(program, all_cores, pf::kCbOutputAccum, data_fmt, Dv_tiles);
    pf::make_cb(program, all_cores, pf::kCbTmp0, data_fmt, Dv_tiles);
    // cb_tmp1 per-iteration throughput = Dv(4) + Dk*Dv(16) + 1 = 21 tiles; size must divide 21
    pf::make_cb(program, all_cores, pf::kCbTmp1, data_fmt, 7);
    // cb_acc per-iteration throughput = Dv(4) + 1 + Dv(4) = 9 tiles; size must divide 9
    pf::make_cb(program, all_cores, pf::kCbAcc, data_fmt, 9);

    auto* state_buf      = inputs.recurrent_state.buffer();
    auto* qkv_buf        = inputs.qkv_proj.buffer();
    auto* z_buf          = inputs.z_proj.buffer();
    auto* b_buf          = inputs.b_proj.buffer();
    auto* a_buf          = inputs.a_proj.buffer();
    auto* conv_state_buf = inputs.conv_state.buffer();
    auto* conv_w_buf     = inputs.conv1d_weight.buffer();
    auto* a_log_buf      = inputs.a_log.buffer();
    auto* dt_bias_buf    = inputs.dt_bias.buffer();
    auto* norm_w_buf     = inputs.norm_weight.buffer();

    uint32_t qkv_col_tiles = conv_dim / pf::kTileSize;
    uint32_t z_col_tiles = (H * Dv) / pf::kTileSize;
    uint32_t ba_col_tiles = (H + pf::kTileSize - 1) / pf::kTileSize;

    std::vector<uint32_t> reader_ct_args = {
        static_cast<uint32_t>(pf::kCbStateInit),
        static_cast<uint32_t>(pf::kCbQ),
        static_cast<uint32_t>(pf::kCbK),
        static_cast<uint32_t>(pf::kCbV),
        static_cast<uint32_t>(pf::kCbG),
        static_cast<uint32_t>(pf::kCbBeta),
        static_cast<uint32_t>(pf::kCbZ),
        static_cast<uint32_t>(pf::kCbNormW),
        static_cast<uint32_t>(pf::kCbKT),
        Dk_tiles,
        Dv_tiles,
        H,
        Hk,
        Dk,
        Dv,
        conv_dim,
        conv_k,
        attrs.head_expand_ratio,
        static_cast<uint32_t>(pf::kCbConvScratch),
        static_cast<uint32_t>(pf::kCbConvStateOut),
        static_cast<uint32_t>(pf::kCbScaler),
        static_cast<uint32_t>(pf::kCbEps),
        S,
        qkv_col_tiles,
        z_col_tiles,
        ba_col_tiles,
    };
    TensorAccessorArgs(state_buf).append_to(reader_ct_args);

    auto reader_kernel = CreateKernel(
        program, pf::kReaderPath, all_cores, ReaderDataMovementConfig(reader_ct_args));

    std::vector<uint32_t> compute_ct_args = {
        static_cast<uint32_t>(pf::kCbStateA),
        static_cast<uint32_t>(pf::kCbQ),
        static_cast<uint32_t>(pf::kCbK),
        static_cast<uint32_t>(pf::kCbV),
        static_cast<uint32_t>(pf::kCbG),
        static_cast<uint32_t>(pf::kCbBeta),
        static_cast<uint32_t>(pf::kCbOutput),
        static_cast<uint32_t>(pf::kCbStateB),
        static_cast<uint32_t>(pf::kCbTmp0),
        static_cast<uint32_t>(pf::kCbTmp1),
        static_cast<uint32_t>(pf::kCbAcc),
        Dk_tiles,
        Dv_tiles,
        static_cast<uint32_t>(pf::kCbStateMid),
        static_cast<uint32_t>(pf::kCbKT),
        static_cast<uint32_t>(pf::kCbZ),
        static_cast<uint32_t>(pf::kCbNormW),
        static_cast<uint32_t>(pf::kCbScaler),
        static_cast<uint32_t>(pf::kCbRawOut),
        static_cast<uint32_t>(pf::kCbEps),
        S,
        static_cast<uint32_t>(pf::kCbStateInit),
    };

    auto compute_kernel = CreateKernel(
        program, pf::kComputePath, all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .fp32_dest_acc_en = true,
            .math_approx_mode = true,
            .compile_args = compute_ct_args,
        });

    auto* state_out_buf = outputs[1].buffer();
    auto* conv_state_out_buf = outputs[2].buffer();
    auto* out_buf = outputs[0].buffer();

    std::vector<uint32_t> writer_ct_args = {
        static_cast<uint32_t>(pf::kCbStateA),
        static_cast<uint32_t>(pf::kCbStateB),
        static_cast<uint32_t>(pf::kCbOutput),
        Dk_tiles,
        Dv_tiles,
        static_cast<uint32_t>(pf::kCbConvStateOut),
        attrs.head_expand_ratio,
        S,
        H,
    };
    TensorAccessorArgs(state_out_buf).append_to(writer_ct_args);

    auto writer_kernel = CreateKernel(
        program, pf::kWriterPath, all_cores, WriterDataMovementConfig(writer_ct_args));

    uint32_t key_dim = Dk * Hk;
    uint32_t key_dim_tiles = key_dim / pf::kTileSize;

    for (uint32_t h = 0; h < H; h++) {
        CoreCoord core = {h % grid.x, h / grid.x};

        uint32_t k_head_idx = h / attrs.head_expand_ratio;

        uint32_t conv_q_tile = k_head_idx * Dk_tiles;
        uint32_t conv_k_tile = key_dim_tiles + k_head_idx * Dk_tiles;
        uint32_t conv_v_tile = 2 * key_dim_tiles + h * Dv_tiles;
        uint32_t qkv_q_tile = k_head_idx * Dk_tiles;
        uint32_t qkv_k_tile = key_dim_tiles + k_head_idx * Dk_tiles;
        uint32_t qkv_v_tile = 2 * key_dim_tiles + h * Dv_tiles;

        SetRuntimeArgs(program, reader_kernel, core, {
            state_buf->address(),
            qkv_buf->address(),
            z_buf->address(),
            b_buf->address(),
            a_buf->address(),
            conv_state_buf->address(),
            conv_w_buf->address(),
            a_log_buf->address(),
            dt_bias_buf->address(),
            norm_w_buf->address(),
            h * state_tiles,          // state_start_tile
            h,                        // head_idx
            conv_q_tile,
            conv_k_tile,
            conv_v_tile,
            qkv_q_tile,
            qkv_k_tile,
            qkv_v_tile,
            h * Dv_tiles,             // z head tile offset
        });

        SetRuntimeArgs(program, writer_kernel, core, {
            state_out_buf->address(),
            out_buf->address(),
            h * state_tiles,           // state_out_start_tile
            conv_state_out_buf->address(),
            h,                         // head_idx
            conv_q_tile,
            conv_k_tile,
            conv_v_tile,
        });
    }

    return cached_program_t{
        std::move(program),
        {
            .reader_kernel_id = reader_kernel,
            .compute_kernel_id = compute_kernel,
            .writer_kernel_id = writer_kernel,
            .all_cores = all_cores,
        }};
}

void DeltaNetPrefillFullProgramFactory::override_runtime_arguments(
    DeltaNetPrefillFullProgramFactory::cached_program_t& cached_program,
    const DeltaNetPrefillFullProgramFactory::operation_attributes_t& attrs,
    const DeltaNetPrefillFullProgramFactory::tensor_args_t& inputs,
    DeltaNetPrefillFullProgramFactory::tensor_return_value_t& outputs) {
    using namespace tt::tt_metal;

    auto& program = cached_program.program;
    auto& shared = cached_program.shared_variables;

    const uint32_t H = attrs.num_heads;
    auto* device = inputs.recurrent_state.device();
    auto grid = device->compute_with_storage_grid_size();

    auto& reader_rt = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_rt = GetRuntimeArgs(program, shared.writer_kernel_id);

    for (uint32_t h = 0; h < H; h++) {
        CoreCoord core = {h % grid.x, h / grid.x};

        auto& r_args = reader_rt[core.x][core.y];
        r_args[0] = inputs.recurrent_state.buffer()->address();
        r_args[1] = inputs.qkv_proj.buffer()->address();
        r_args[2] = inputs.z_proj.buffer()->address();
        r_args[3] = inputs.b_proj.buffer()->address();
        r_args[4] = inputs.a_proj.buffer()->address();
        r_args[5] = inputs.conv_state.buffer()->address();
        r_args[6] = inputs.conv1d_weight.buffer()->address();
        r_args[7] = inputs.a_log.buffer()->address();
        r_args[8] = inputs.dt_bias.buffer()->address();
        r_args[9] = inputs.norm_weight.buffer()->address();

        auto& w_args = writer_rt[core.x][core.y];
        w_args[0] = outputs[1].buffer()->address();
        w_args[1] = outputs[0].buffer()->address();
        w_args[3] = outputs[2].buffer()->address();
    }
}

}  // namespace ttnn::operations::experimental::deltanet
