// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deltanet_full_program_factory.hpp"

#include "deltanet_full_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::deltanet {

namespace full_factory {

constexpr uint32_t kTileSize = tt::constants::TILE_WIDTH;

constexpr auto kReaderPath =
    "ttnn/cpp/ttnn/operations/experimental/deltanet/device/kernels/dataflow/reader_deltanet_full.cpp";
constexpr auto kComputePath =
    "ttnn/cpp/ttnn/operations/experimental/deltanet/device/kernels/compute/deltanet_full_compute.cpp";
constexpr auto kWriterPath =
    "ttnn/cpp/ttnn/operations/experimental/deltanet/device/kernels/dataflow/writer_deltanet_full.cpp";

constexpr auto kCbStateIn   = tt::CBIndex::c_0;
constexpr auto kCbQ         = tt::CBIndex::c_1;
constexpr auto kCbK         = tt::CBIndex::c_2;
constexpr auto kCbV         = tt::CBIndex::c_3;
constexpr auto kCbG         = tt::CBIndex::c_4;
constexpr auto kCbBeta      = tt::CBIndex::c_5;
constexpr auto kCbOutput    = tt::CBIndex::c_6;
constexpr auto kCbStateOut  = tt::CBIndex::c_7;
constexpr auto kCbZ         = tt::CBIndex::c_8;
constexpr auto kCbNormW     = tt::CBIndex::c_9;
constexpr auto kCbStateMid  = tt::CBIndex::c_16;
constexpr auto kCbKT        = tt::CBIndex::c_17;
constexpr auto kCbConvScratch  = tt::CBIndex::c_18;
constexpr auto kCbConvStateOut = tt::CBIndex::c_19;
constexpr auto kCbEps       = tt::CBIndex::c_10;
constexpr auto kCbScaler    = tt::CBIndex::c_20;
constexpr auto kCbRawOut    = tt::CBIndex::c_21;
constexpr auto kCbTmp0      = tt::CBIndex::c_24;
constexpr auto kCbTmp1      = tt::CBIndex::c_25;
constexpr auto kCbAcc       = tt::CBIndex::c_26;

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

}  // namespace full_factory

DeltaNetDecodeFullProgramFactory::cached_program_t DeltaNetDecodeFullProgramFactory::create(
    const DeltaNetDecodeFullProgramFactory::operation_attributes_t& attrs,
    const DeltaNetDecodeFullProgramFactory::tensor_args_t& inputs,
    DeltaNetDecodeFullProgramFactory::tensor_return_value_t& outputs) {
    using namespace tt::tt_metal;
    namespace ff = full_factory;

    const uint32_t H = attrs.num_heads;
    const uint32_t Hk = attrs.num_k_heads;
    const uint32_t Dk = attrs.k_head_dim;
    const uint32_t Dv = attrs.v_head_dim;
    const uint32_t Dk_tiles = Dk / ff::kTileSize;
    const uint32_t Dv_tiles = Dv / ff::kTileSize;
    const uint32_t state_tiles = Dk_tiles * Dv_tiles;
    const uint32_t conv_dim = attrs.conv_dim;
    const uint32_t conv_k = attrs.conv_kernel_size;

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

    // All compute CBs use data_fmt (bf16) for uniform format.
    // State stored in DRAM as f32 is converted by reader/writer.
    ff::make_cb(program, all_cores, ff::kCbStateIn, data_fmt, state_tiles);
    ff::make_cb(program, all_cores, ff::kCbQ, data_fmt, Dk_tiles);
    ff::make_cb(program, all_cores, ff::kCbK, data_fmt, Dk_tiles);
    ff::make_cb(program, all_cores, ff::kCbV, data_fmt, Dv_tiles);
    ff::make_cb(program, all_cores, ff::kCbG, data_fmt, 1);
    ff::make_cb(program, all_cores, ff::kCbBeta, data_fmt, 1);
    ff::make_cb(program, all_cores, ff::kCbOutput, data_fmt, Dv_tiles);
    ff::make_cb(program, all_cores, ff::kCbStateOut, data_fmt, state_tiles);
    ff::make_cb(program, all_cores, ff::kCbZ, data_fmt, Dv_tiles);
    ff::make_cb(program, all_cores, ff::kCbNormW, data_fmt, Dv_tiles);
    ff::make_cb(program, all_cores, ff::kCbStateMid, data_fmt, state_tiles);
    ff::make_cb(program, all_cores, ff::kCbKT, data_fmt, Dk_tiles);
    ff::make_cb(program, all_cores, ff::kCbConvScratch, data_fmt, 12);
    ff::make_cb(program, all_cores, ff::kCbConvStateOut, data_fmt, 12);
    ff::make_cb(program, all_cores, ff::kCbScaler, data_fmt, 1);
    ff::make_cb(program, all_cores, ff::kCbEps, data_fmt, 1);
    ff::make_cb(program, all_cores, ff::kCbRawOut, data_fmt, Dv_tiles);
    ff::make_cb(program, all_cores, ff::kCbTmp0, data_fmt, Dv_tiles);
    ff::make_cb(program, all_cores, ff::kCbTmp1, data_fmt, std::max(Dk_tiles, Dv_tiles));
    ff::make_cb(program, all_cores, ff::kCbAcc, data_fmt, Dv_tiles);

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

    std::vector<uint32_t> reader_ct_args = {
        static_cast<uint32_t>(ff::kCbStateIn),
        static_cast<uint32_t>(ff::kCbQ),
        static_cast<uint32_t>(ff::kCbK),
        static_cast<uint32_t>(ff::kCbV),
        static_cast<uint32_t>(ff::kCbG),
        static_cast<uint32_t>(ff::kCbBeta),
        static_cast<uint32_t>(ff::kCbZ),
        static_cast<uint32_t>(ff::kCbNormW),
        static_cast<uint32_t>(ff::kCbKT),
        Dk_tiles,
        Dv_tiles,
        H,
        Hk,
        Dk,
        Dv,
        conv_dim,
        conv_k,
        attrs.head_expand_ratio,
        static_cast<uint32_t>(ff::kCbConvScratch),
        static_cast<uint32_t>(ff::kCbConvStateOut),
        static_cast<uint32_t>(ff::kCbScaler),
        static_cast<uint32_t>(ff::kCbEps),
    };
    TensorAccessorArgs(state_buf).append_to(reader_ct_args);

    auto reader_kernel = CreateKernel(
        program, ff::kReaderPath, all_cores, ReaderDataMovementConfig(reader_ct_args));

    std::vector<uint32_t> compute_ct_args = {
        static_cast<uint32_t>(ff::kCbStateIn),
        static_cast<uint32_t>(ff::kCbQ),
        static_cast<uint32_t>(ff::kCbK),
        static_cast<uint32_t>(ff::kCbV),
        static_cast<uint32_t>(ff::kCbG),
        static_cast<uint32_t>(ff::kCbBeta),
        static_cast<uint32_t>(ff::kCbOutput),
        static_cast<uint32_t>(ff::kCbStateOut),
        static_cast<uint32_t>(ff::kCbTmp0),
        static_cast<uint32_t>(ff::kCbTmp1),
        static_cast<uint32_t>(ff::kCbAcc),
        Dk_tiles,
        Dv_tiles,
        static_cast<uint32_t>(ff::kCbStateMid),
        static_cast<uint32_t>(ff::kCbKT),
        static_cast<uint32_t>(ff::kCbZ),
        static_cast<uint32_t>(ff::kCbNormW),
        static_cast<uint32_t>(ff::kCbScaler),
        static_cast<uint32_t>(ff::kCbRawOut),
        static_cast<uint32_t>(ff::kCbEps),
    };

    auto compute_kernel = CreateKernel(
        program, ff::kComputePath, all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .fp32_dest_acc_en = true,
            .math_approx_mode = true,
            .compile_args = compute_ct_args,
        });

    auto* state_out_buf  = outputs[1].buffer();
    auto* conv_state_out_buf = outputs[2].buffer();
    std::vector<uint32_t> writer_ct_args = {
        static_cast<uint32_t>(ff::kCbStateOut),
        static_cast<uint32_t>(ff::kCbOutput),
        Dk_tiles,
        Dv_tiles,
        static_cast<uint32_t>(ff::kCbConvStateOut),
        attrs.head_expand_ratio,
    };
    TensorAccessorArgs(state_out_buf).append_to(writer_ct_args);

    auto writer_kernel = CreateKernel(
        program, ff::kWriterPath, all_cores, WriterDataMovementConfig(writer_ct_args));

    auto* out_buf = outputs[0].buffer();

    uint32_t key_dim = Dk * Hk;
    uint32_t key_dim_tiles = key_dim / ff::kTileSize;

    for (uint32_t h = 0; h < H; h++) {
        CoreCoord core = {h % grid.x, h / grid.x};

        uint32_t k_head_idx = h / attrs.head_expand_ratio;

        uint32_t q_byte_offset = k_head_idx * Dk * 2;
        uint32_t k_byte_offset = (key_dim + k_head_idx * Dk) * 2;
        uint32_t v_byte_offset = (key_dim * 2 + h * Dv) * 2;
        uint32_t z_byte_offset = h * Dv * 2;

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
            h * state_tiles,
            h,
            q_byte_offset,
            k_byte_offset,
            v_byte_offset,
            z_byte_offset,
            conv_q_tile,
            conv_k_tile,
            conv_v_tile,
            qkv_q_tile,
            qkv_k_tile,
            qkv_v_tile,
        });

        SetRuntimeArgs(program, writer_kernel, core, {
            state_out_buf->address(),
            out_buf->address(),
            h * state_tiles,
            h * Dv_tiles,
            conv_state_out_buf->address(),
            h,
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

void DeltaNetDecodeFullProgramFactory::override_runtime_arguments(
    DeltaNetDecodeFullProgramFactory::cached_program_t& cached_program,
    const DeltaNetDecodeFullProgramFactory::operation_attributes_t& attrs,
    const DeltaNetDecodeFullProgramFactory::tensor_args_t& inputs,
    DeltaNetDecodeFullProgramFactory::tensor_return_value_t& outputs) {
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
        w_args[4] = outputs[2].buffer()->address();
    }
}

}  // namespace ttnn::operations::experimental::deltanet
