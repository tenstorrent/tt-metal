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
    // const auto& ashape = a.padded_shape();  // TODO: Use when implementing kernel logic
    // const auto& bshape = b.padded_shape();  // TODO: Use when implementing kernel logic
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

    // tt_metal::IDevice* device = a.device();  // TODO: Use when implementing kernel logic

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    // tt_metal::Buffer* in0_buffer = a.buffer();  // TODO: Use when implementing kernel logic
    // tt_metal::Buffer* in1_buffer = b.buffer();  // TODO: Use when implementing kernel logic
    // tt_metal::Buffer* out_buffer = output.buffer();  // TODO: Use when implementing kernel logic

    // Placeholder CB sizes - these should be calculated based on your gate operation requirements
    uint32_t in0_CB_size = in0_single_tile_size;
    uint32_t in1_CB_size = in1_single_tile_size;
    uint32_t out_CB_size = output_single_tile_size;

    CoreCoord start_core = {0, 0};
    uint32_t num_cores = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;
    uint32_t num_cores_with_work = num_cores;
    constexpr bool row_major = true;

    using tt::tt_metal::num_cores_to_corerangeset;
    CoreRangeSet all_cores =
        num_cores_to_corerangeset(start_core, num_cores, compute_with_storage_grid_size, row_major);

    const auto& cores = corerange_to_cores(all_cores, std::nullopt, row_major);

    std::vector<uint32_t> compute_kernel_args = {};

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t src1_cb_index = tt::CBIndex::c_1;
    uint32_t output_cb_index = tt::CBIndex::c_2;
    uint32_t interm0_cb_index = tt::CBIndex::c_5;

    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_single_tile_size)
            .set_tile_dims(src0_cb_index, in0_tile);

    tt_metal::CircularBufferConfig src1_cb_config =
        tt_metal::CircularBufferConfig(in1_CB_size, {{src1_cb_index, in1_data_format}})
            .set_page_size(src1_cb_index, in1_single_tile_size)
            .set_tile_dims(src1_cb_index, in1_tile);

    std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
        {output_cb_index, output_data_format}, {interm0_cb_index, interm0_data_format}};
    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
            .set_page_size(output_cb_index, output_single_tile_size)
            .set_page_size(interm0_cb_index, interm0_single_tile_size)
            .set_tile_dims(output_cb_index, output_tile)
            .set_tile_dims(interm0_cb_index, output_tile);

    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, src1_cb_config);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);

    // Placeholder kernel handles - these need to be created with actual kernel creation calls
    tt::tt_metal::KernelHandle mm_kernel_in0_mcast_cores_with_work_and_in_receiver_grid_id = 0;
    tt::tt_metal::KernelHandle mm_kernel_in1_sender_writer_id = 0;

    return gate_common_override_variables_t{
        {mm_kernel_in0_mcast_cores_with_work_and_in_receiver_grid_id, mm_kernel_in1_sender_writer_id},
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
