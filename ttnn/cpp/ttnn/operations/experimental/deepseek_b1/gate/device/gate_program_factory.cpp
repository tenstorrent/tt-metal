// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gate_program_factory.hpp"

#include <algorithm>
#include <utility>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

using namespace tt;
using namespace tt::constants;

namespace ttnn {

namespace operations {

namespace experimental {

namespace deepseek_b1 {

namespace gate {

gate_common_override_variables_t deepseek_b1_gate_(
    tt_metal::Program& program,
    const Tensor& a,
    const Tensor& b,
    const Tensor& expert_bias,
    const Tensor& output_tensor,
    CoreCoord compute_with_storage_grid_size,
    DeviceComputeKernelConfig compute_kernel_config) {
    const auto& output = output_tensor;

    // Get tensor shapes and tiles
    const auto& ashape = a.padded_shape();
    // const auto& bshape = b.padded_shape();  // Not used yet
    auto in0_tile = a.tensor_spec().tile();
    auto in1_tile = b.tensor_spec().tile();
    auto in0_tile_shape = in0_tile.get_tile_shape();
    auto in1_tile_shape = in1_tile.get_tile_shape();
    auto output_tile = tt::tt_metal::Tile({in0_tile_shape[0], in1_tile_shape[1]});

    // CB dataformats
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    tt::DataFormat interm0_data_format = output_data_format;

    tt_metal::IDevice* device = a.device();

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    ////////////////////////////////////////////////////////////////////////////
    //                      Gate Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // Calculate dimensions in tiles
    uint32_t B = get_batch_size(ashape);
    // uint32_t Mt = ashape[-2] / in0_tile_shape[0];  // M dimension in tiles (not used yet)
    uint32_t Kt = ashape[-1] / in0_tile_shape[1];  // K dimension in tiles (inner dim)
    // uint32_t Nt = bshape[-1] / in1_tile_shape[1];  // N dimension in tiles (not used yet)

    // Fuse batch dimension (not used yet, but keeping structure consistent with matmul_1d)
    // bool fuse_batch = true;
    // if (fuse_batch) {
    //     Mt = B * Mt;
    //     B = 1;
    // }
    B = 1;  // Simplified: assume batch is fused

    // Gate operation parameters (similar to matmul_1d)
    uint32_t in0_block_w = Kt;  // Process full K dimension
    uint32_t out_subblock_h = 1;
    uint32_t out_subblock_w = 1;
    // uint32_t per_core_M = 1;  // Not used yet
    // uint32_t per_core_N = 1;  // Not used yet
    uint32_t out_block_h = 1;
    uint32_t out_block_w = 1;

    TT_FATAL(Kt % in0_block_w == 0, "Kt ({}) must be divisible by in0_block_w ({})", Kt, in0_block_w);

    uint32_t in0_block_h = out_block_h;
    uint32_t in1_block_w = out_block_w;

    uint32_t in0_block_tiles = in0_block_h * in0_block_w;
    uint32_t in0_block_num_tiles = in0_block_tiles;
    uint32_t in1_block_tiles = out_block_w * in0_block_w;
    uint32_t in1_block_num_tiles = in1_block_tiles;
    uint32_t num_blocks = Kt / in0_block_w;
    uint32_t num_blocks_inner_dim = num_blocks;
    uint32_t batch = B;

    uint32_t in0_CB_size = in0_block_num_tiles * in0_single_tile_size;
    uint32_t in1_CB_size = in1_block_num_tiles * in1_single_tile_size;
    uint32_t out_CB_size = out_block_h * out_block_w * output_single_tile_size;

    CoreCoord start_core = {0, 0};
    uint32_t num_cores = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;
    uint32_t num_cores_with_work = num_cores;
    constexpr bool row_major = true;

    using tt::tt_metal::num_cores_to_corerangeset;
    CoreRangeSet all_cores =
        num_cores_to_corerangeset(start_core, num_cores, compute_with_storage_grid_size, row_major);

    const auto& cores = corerange_to_cores(all_cores, std::nullopt, row_major);

    // Get compute kernel config parameters
    auto [math_fidelity, math_approx_mode, _, __, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                      Kernel Creation
    ////////////////////////////////////////////////////////////////////////////

    // Create semaphores for multicast synchronization
    auto in0_mcast_sender_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in0_mcast_receiver_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);

    // Prepare NOC coordinates
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    std::vector<uint32_t> in0_mcast_noc_x;
    std::vector<uint32_t> in0_mcast_noc_y;
    in0_mcast_noc_x.reserve(num_cores_x);
    in0_mcast_noc_y.reserve(num_cores_y);
    for (uint32_t core_idx_x = 0; core_idx_x < num_cores_x; ++core_idx_x) {
        in0_mcast_noc_x.push_back(device->worker_core_from_logical_core({core_idx_x, 0}).x);
    }
    for (uint32_t core_idx_y = 0; core_idx_y < num_cores_y; ++core_idx_y) {
        in0_mcast_noc_y.push_back(device->worker_core_from_logical_core({0, core_idx_y}).y);
    }

    CoreCoord top_left_core = all_cores.bounding_box().start_coord;
    CoreCoord bottom_right_core = all_cores.bounding_box().end_coord;
    auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    // Reader kernel compile-time args
    uint32_t in0_block_size_bytes = in0_block_num_tiles * in0_single_tile_size;
    uint32_t shard_width_in_tiles = 1;  // Simple non-sharded case

    std::vector<uint32_t> reader_compile_time_args = {
        in0_block_num_tiles,              // 0: in0_block_num_tiles
        in0_block_size_bytes,             // 1: in0_block_size_bytes
        num_blocks_inner_dim,             // 2: num_blocks_inner_dim
        in0_mcast_sender_semaphore_id,    // 3: in0_mcast_sender_semaphore_id
        in0_mcast_receiver_semaphore_id,  // 4: in0_mcast_receiver_semaphore_id
        num_cores,                        // 5: in0_mcast_num_dests
        num_cores,                        // 6: in0_mcast_num_cores
        num_cores_x,                      // 7: num_x
        num_cores_y,                      // 8: num_y
        (uint32_t)false,                  // 9: transpose_mcast
        shard_width_in_tiles,             // 10: shard_width_in_tiles
        in0_block_w,                      // 11: in0_block_w
        batch                             // 12: batch
    };

    // Writer kernel compile-time args
    std::vector<uint32_t> writer_compile_time_args = {
        in1_block_num_tiles,   // 0: in1_block_num_tiles
        num_blocks_inner_dim,  // 1: num_blocks_inner_dim
        batch,                 // 2: batch
        out_subblock_w,        // 3: out_subblock_w
        out_subblock_h         // 4: out_subblock_h
    };

    // Compute kernel compile-time args (must match gate.cpp expectations)
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;
    bool untilize_out = false;

    std::vector<uint32_t> compute_kernel_args = {
        in0_block_w,             // 0: in0_block_w
        in0_block_num_tiles,     // 1: in0_block_num_tiles
        in1_block_num_tiles,     // 2: in1_block_num_tiles
        in1_block_w,             // 3: in1_block_w
        out_subblock_h,          // 4: out_subblock_h
        out_subblock_w,          // 5: out_subblock_w
        out_subblock_num_tiles,  // 6: out_subblock_num_tiles
        untilize_out             // 7: untilize_out
    };

    // Kernel defines for compute throttling
    std::map<std::string, std::string> mm_kernel_defines;
    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), num_cores, mm_kernel_defines, ttnn::get_throttle_level(compute_kernel_config));

    // NOC selection
    tt_metal::NOC reader_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt_metal::NOC writer_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    // Create reader kernel (RISCV_1)
    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_b1/gate/device/kernels/reader.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = reader_noc,
            .compile_args = reader_compile_time_args});

    // Create writer kernel (RISCV_0)
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_b1/gate/device/kernels/writer.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = writer_noc,
            .compile_args = writer_compile_time_args});

    // Create compute kernel
    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_b1/gate/device/kernels/gate.cpp",
        all_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = false,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args,
            .defines = mm_kernel_defines});

    ////////////////////////////////////////////////////////////////////////////
    //                      Circular Buffer Creation
    ////////////////////////////////////////////////////////////////////////////

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_single_tile_size)
            .set_tile_dims(src0_cb_index, in0_tile);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt_metal::CircularBufferConfig src1_cb_config =
        tt_metal::CircularBufferConfig(in1_CB_size, {{src1_cb_index, in1_data_format}})
            .set_page_size(src1_cb_index, in1_single_tile_size)
            .set_tile_dims(src1_cb_index, in1_tile);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, src1_cb_config);

    uint32_t output_cb_index = tt::CBIndex::c_4;
    uint32_t interm0_cb_index = tt::CBIndex::c_5;
    std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
        {output_cb_index, output_data_format}, {interm0_cb_index, interm0_data_format}};
    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
            .set_page_size(output_cb_index, output_single_tile_size)
            .set_page_size(interm0_cb_index, interm0_single_tile_size)
            .set_tile_dims(output_cb_index, output_tile)
            .set_tile_dims(interm0_cb_index, output_tile);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);

    ////////////////////////////////////////////////////////////////////////////
    //                      Runtime Args Setup
    ////////////////////////////////////////////////////////////////////////////

    CoreCoord start_core_noc = top_left_core_physical;
    CoreCoord end_core_noc = bottom_right_core_physical;

    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores[i];

        // Reader runtime args
        std::vector<uint32_t> reader_runtime_args;
        reader_runtime_args.reserve(5 + in0_mcast_noc_x.size() + in0_mcast_noc_y.size());
        reader_runtime_args.push_back(i);  // sender_id
        reader_runtime_args.push_back(start_core_noc.x);
        reader_runtime_args.push_back(start_core_noc.y);
        reader_runtime_args.push_back(end_core_noc.x);
        reader_runtime_args.push_back(end_core_noc.y);
        reader_runtime_args.insert(reader_runtime_args.end(), in0_mcast_noc_x.begin(), in0_mcast_noc_x.end());
        reader_runtime_args.insert(reader_runtime_args.end(), in0_mcast_noc_y.begin(), in0_mcast_noc_y.end());

        tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);

        // Writer runtime args
        std::vector<uint32_t> writer_runtime_args = {
            out_subblock_h,  // out_num_nonzero_subblocks_h
            out_subblock_w   // out_num_nonzero_subblocks_w
        };

        tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
    }

    return gate_common_override_variables_t{
        {reader_kernel_id, writer_kernel_id, compute_kernel_id},
        {cb_src0, cb_src1, cb_output},
        false,
        start_core,
        cores,
        num_cores_with_work,
    };
}

tt::tt_metal::operation::ProgramWithCallbacks deepseek_b1_gate(
    const Tensor& a,
    const Tensor& b,
    const Tensor& expert_bias,
    const Tensor& output_tensor,
    CoreCoord compute_with_storage_grid_size,
    DeviceComputeKernelConfig compute_kernel_config) {
    tt_metal::Program program{};

    gate_common_override_variables_t shared_vars = deepseek_b1_gate_(
        program, a, b, expert_bias, output_tensor, compute_with_storage_grid_size, compute_kernel_config);
    auto override_runtime_arguments_callback =
        [shared_vars](
            const void* operation,
            tt_metal::Program& program,
            const std::vector<tt::tt_metal::Tensor>& input_tensors,
            const std::vector<std::optional<const tt::tt_metal::Tensor>>& optional_input_tensors,
            const std::vector<tt::tt_metal::Tensor>& output_tensors) {
            auto src_buffer_a = input_tensors.at(0).buffer();
            auto src_buffer_b = input_tensors.at(1).buffer();
            // auto expert_bias_buffer = input_tensors.at(2).buffer();
            auto dst_buffer = output_tensors.at(0).buffer();

            // bool src0_sharded = input_tensors[0].is_sharded();
            // bool src1_sharded = input_tensors[1].is_sharded();
            // bool out_sharded = output_tensors[0].is_sharded();

            UpdateDynamicCircularBufferAddress(program, shared_vars.cbs.at(0), *src_buffer_a);
            UpdateDynamicCircularBufferAddress(program, shared_vars.cbs.at(1), *src_buffer_b);
            UpdateDynamicCircularBufferAddress(program, shared_vars.cbs.at(2), *dst_buffer);
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace gate

}  // namespace deepseek_b1

}  // namespace experimental

}  // namespace operations

}  // namespace ttnn
