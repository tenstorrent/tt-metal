// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deltanet_decode_program_factory.hpp"

#include "deltanet_decode_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::deltanet {

namespace {

constexpr uint32_t TILE_SIZE = tt::constants::TILE_WIDTH;  // 32

constexpr auto kReaderKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deltanet/device/kernels/dataflow/reader_deltanet_decode.cpp";
constexpr auto kComputeKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deltanet/device/kernels/compute/deltanet_decode_compute.cpp";
constexpr auto kWriterKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deltanet/device/kernels/dataflow/writer_deltanet_decode.cpp";

constexpr auto cb_state_in  = tt::CBIndex::c_0;
constexpr auto cb_q         = tt::CBIndex::c_1;
constexpr auto cb_k         = tt::CBIndex::c_2;
constexpr auto cb_v         = tt::CBIndex::c_3;
constexpr auto cb_g         = tt::CBIndex::c_4;
constexpr auto cb_beta      = tt::CBIndex::c_5;
constexpr auto cb_output    = tt::CBIndex::c_6;
constexpr auto cb_state_out = tt::CBIndex::c_7;
constexpr auto cb_state_mid = tt::CBIndex::c_16;
constexpr auto cb_k_T       = tt::CBIndex::c_17;
constexpr auto cb_tmp0      = tt::CBIndex::c_24;
constexpr auto cb_tmp1      = tt::CBIndex::c_25;
constexpr auto cb_acc       = tt::CBIndex::c_26;

tt::tt_metal::CBHandle create_cb(
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

}  // namespace

DeltaNetDecodeProgramFactory::cached_program_t DeltaNetDecodeProgramFactory::create(
    const DeltaNetDecodeProgramFactory::operation_attributes_t& attrs,
    const DeltaNetDecodeProgramFactory::tensor_args_t& inputs,
    DeltaNetDecodeProgramFactory::tensor_return_value_t& outputs) {
    using namespace tt::tt_metal;

    const uint32_t num_heads = attrs.num_heads;
    const uint32_t Dk_tiles = attrs.k_head_dim / TILE_SIZE;
    const uint32_t Dv_tiles = attrs.v_head_dim / TILE_SIZE;
    const uint32_t state_tiles = Dk_tiles * Dv_tiles;

    auto* device = inputs.state.device();
    Program program{};

    tt::DataFormat data_fmt = datatype_to_dataformat_converter(inputs.state.dtype());

    // Map heads to cores: one head per core, column-major within compute grid
    auto grid = device->compute_with_storage_grid_size();
    TT_FATAL(num_heads <= grid.x * grid.y, "Need {} cores for {} heads, grid is {}x{}", num_heads, num_heads, grid.x, grid.y);

    std::vector<CoreRange> core_ranges;
    for (uint32_t h = 0; h < num_heads; h++) {
        CoreCoord core = {h % grid.x, h / grid.x};
        core_ranges.emplace_back(core, core);
    }
    CoreRangeSet all_cores(core_ranges);

    // Create circular buffers
    create_cb(program, all_cores, cb_state_in, data_fmt, state_tiles);
    create_cb(program, all_cores, cb_q, data_fmt, Dk_tiles);
    create_cb(program, all_cores, cb_k, data_fmt, Dk_tiles);
    create_cb(program, all_cores, cb_v, data_fmt, Dv_tiles);
    create_cb(program, all_cores, cb_g, data_fmt, 1);
    create_cb(program, all_cores, cb_beta, data_fmt, 1);
    create_cb(program, all_cores, cb_output, data_fmt, Dv_tiles);
    create_cb(program, all_cores, cb_state_out, data_fmt, state_tiles);
    create_cb(program, all_cores, cb_state_mid, data_fmt, state_tiles);
    create_cb(program, all_cores, cb_tmp0, data_fmt, Dv_tiles);
    create_cb(program, all_cores, cb_tmp1, data_fmt, std::max(Dk_tiles, Dv_tiles));
    create_cb(program, all_cores, cb_acc, data_fmt, Dv_tiles);
    create_cb(program, all_cores, cb_k_T, data_fmt, Dk_tiles);

    // Reader kernel compile-time args
    auto* state_buf = inputs.state.buffer();
    std::vector<uint32_t> reader_ct_args = {
        static_cast<uint32_t>(cb_state_in),
        static_cast<uint32_t>(cb_q),
        static_cast<uint32_t>(cb_k),
        static_cast<uint32_t>(cb_v),
        static_cast<uint32_t>(cb_g),
        static_cast<uint32_t>(cb_beta),
        Dk_tiles,
        Dv_tiles,
        static_cast<uint32_t>(cb_k_T),
    };
    TensorAccessorArgs(state_buf).append_to(reader_ct_args);

    auto reader_kernel = CreateKernel(
        program, kReaderKernelPath, all_cores, ReaderDataMovementConfig(reader_ct_args));

    // Compute kernel compile-time args
    std::vector<uint32_t> compute_ct_args = {
        static_cast<uint32_t>(cb_state_in),
        static_cast<uint32_t>(cb_q),
        static_cast<uint32_t>(cb_k),
        static_cast<uint32_t>(cb_v),
        static_cast<uint32_t>(cb_g),
        static_cast<uint32_t>(cb_beta),
        static_cast<uint32_t>(cb_output),
        static_cast<uint32_t>(cb_state_out),
        static_cast<uint32_t>(cb_tmp0),
        static_cast<uint32_t>(cb_tmp1),
        static_cast<uint32_t>(cb_acc),
        Dk_tiles,
        Dv_tiles,
        static_cast<uint32_t>(cb_state_mid),
        static_cast<uint32_t>(cb_k_T),
    };

    auto compute_kernel = CreateKernel(
        program, kComputeKernelPath, all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_ct_args,
        });

    // Writer kernel compile-time args
    auto* state_out_buf = outputs[1].buffer();  // new_state
    std::vector<uint32_t> writer_ct_args = {
        static_cast<uint32_t>(cb_state_out),
        static_cast<uint32_t>(cb_output),
        Dk_tiles,
        Dv_tiles,
    };
    TensorAccessorArgs(state_out_buf).append_to(writer_ct_args);

    auto writer_kernel = CreateKernel(
        program, kWriterKernelPath, all_cores, WriterDataMovementConfig(writer_ct_args));

    // Set per-core runtime args
    auto* q_buf = inputs.query.buffer();
    auto* k_buf = inputs.key.buffer();
    auto* v_buf = inputs.value.buffer();
    auto* g_buf = inputs.decay.buffer();
    auto* beta_buf = inputs.beta.buffer();
    auto* out_buf = outputs[0].buffer();  // output vector

    for (uint32_t h = 0; h < num_heads; h++) {
        CoreCoord core = {h % grid.x, h / grid.x};

        // Reader runtime args: base addresses + per-head start tile indices
        SetRuntimeArgs(program, reader_kernel, core, {
            state_buf->address(),
            q_buf->address(),
            k_buf->address(),
            v_buf->address(),
            g_buf->address(),
            beta_buf->address(),
            h * state_tiles,    // state_start_tile
            h * Dk_tiles,       // q_start_tile
            h * Dk_tiles,       // k_start_tile
            h * Dv_tiles,       // v_start_tile
            h,                  // g_start_tile (1 tile per head)
            h,                  // beta_start_tile
        });

        // Writer runtime args
        SetRuntimeArgs(program, writer_kernel, core, {
            state_out_buf->address(),
            out_buf->address(),
            h * state_tiles,    // state_out_start_tile
            h * Dv_tiles,       // output_start_tile
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

void DeltaNetDecodeProgramFactory::override_runtime_arguments(
    DeltaNetDecodeProgramFactory::cached_program_t& cached_program,
    const DeltaNetDecodeProgramFactory::operation_attributes_t& attrs,
    const DeltaNetDecodeProgramFactory::tensor_args_t& inputs,
    DeltaNetDecodeProgramFactory::tensor_return_value_t& outputs) {
    using namespace tt::tt_metal;

    auto& program = cached_program.program;
    auto& shared = cached_program.shared_variables;

    const uint32_t num_heads = attrs.num_heads;

    auto* device = inputs.state.device();
    auto grid = device->compute_with_storage_grid_size();

    auto& reader_rt = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_rt = GetRuntimeArgs(program, shared.writer_kernel_id);

    for (uint32_t h = 0; h < num_heads; h++) {
        CoreCoord core = {h % grid.x, h / grid.x};

        // Update reader addresses (tile offsets don't change across calls)
        auto& r_args = reader_rt[core.x][core.y];
        r_args[0] = inputs.state.buffer()->address();
        r_args[1] = inputs.query.buffer()->address();
        r_args[2] = inputs.key.buffer()->address();
        r_args[3] = inputs.value.buffer()->address();
        r_args[4] = inputs.decay.buffer()->address();
        r_args[5] = inputs.beta.buffer()->address();

        // Update writer addresses
        auto& w_args = writer_rt[core.x][core.y];
        w_args[0] = outputs[1].buffer()->address();
        w_args[1] = outputs[0].buffer()->address();
    }
}

}  // namespace ttnn::operations::experimental::deltanet
