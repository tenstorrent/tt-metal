// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/deepseek/moe/deepseek_moe_gate/deepseek_moe_gate.hpp"

#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "ttnn/operations/generic/generic_op.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::experimental::deepseek::moe {

namespace {

constexpr const char* kDeepseekMoeGateKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/moe/deepseek_moe_gate/device/kernels/deepseek_moe_gate_kernel.cpp";

uint32_t float_bits_u32(float value) {
    uint32_t bits = 0;
    static_assert(sizeof(float) == sizeof(uint32_t));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void set_cb_page_size_for_tile(tt::tt_metal::CBDescriptor& cb_desc, const tt::tt_metal::Tensor& tensor) {
    const auto& spec = tensor.tensor_spec();
    const auto& tile = spec.tile();
    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(spec.data_type());
    uint32_t tile_size = tile.get_tile_size(data_format);
    auto& fd = cb_desc.format_descriptors[0];
    fd.tile = tt::tt_metal::TileDescriptor(tile);
    fd.page_size = (cb_desc.total_size % tile_size == 0) ? tile_size : cb_desc.total_size;
}

}  // namespace

std::tuple<tt::tt_metal::Tensor, tt::tt_metal::Tensor> deepseek_moe_gate(
    const tt::tt_metal::Tensor& input_tensor,
    const tt::tt_metal::Tensor& bias_tensor,
    const tt::tt_metal::Tensor& input_indices_tensor,
    tt::tt_metal::Tensor& output_tensor,
    tt::tt_metal::Tensor& output_indices_tensor,
    float eps,
    float scaling_factor,
    bool enable_sigmoid) {
    using tt::tt_metal::CoreRangeSet;
    using tt::tt_metal::DataMovementConfigDescriptor;
    using tt::tt_metal::DataMovementProcessor;
    using tt::tt_metal::DataType;
    using tt::tt_metal::KernelDescriptor;
    using tt::tt_metal::NOC;
    using tt::tt_metal::NOC_MODE;

    TT_FATAL(input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "input_tensor must be on device");
    TT_FATAL(bias_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "bias_tensor must be on device");
    TT_FATAL(
        input_indices_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
        "input_indices_tensor must be on device");
    TT_FATAL(output_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "output_tensor must be on device");
    TT_FATAL(
        output_indices_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
        "output_indices_tensor must be on device");

    TT_FATAL(input_tensor.device() == bias_tensor.device(), "All tensors must be on the same device");
    TT_FATAL(input_tensor.device() == input_indices_tensor.device(), "All tensors must be on the same device");
    TT_FATAL(input_tensor.device() == output_tensor.device(), "All tensors must be on the same device");
    TT_FATAL(input_tensor.device() == output_indices_tensor.device(), "All tensors must be on the same device");

    TT_FATAL(input_tensor.dtype() == DataType::BFLOAT16, "input_tensor must be BFLOAT16");
    TT_FATAL(bias_tensor.dtype() == DataType::BFLOAT16, "bias_tensor must be BFLOAT16");
    TT_FATAL(output_tensor.dtype() == DataType::BFLOAT16, "output_tensor must be BFLOAT16");

    TT_FATAL(input_tensor.is_sharded(), "input_tensor must be sharded");
    TT_FATAL(bias_tensor.is_sharded(), "bias_tensor must be sharded");
    TT_FATAL(input_indices_tensor.is_sharded(), "input_indices_tensor must be sharded");
    TT_FATAL(output_tensor.is_sharded(), "output_tensor must be sharded");
    TT_FATAL(output_indices_tensor.is_sharded(), "output_indices_tensor must be sharded");

    const auto& in_shape = input_tensor.logical_shape();
    const auto& bias_shape = bias_tensor.logical_shape();
    const auto& out_shape = output_tensor.logical_shape();
    const auto& in_idx_shape = input_indices_tensor.logical_shape();
    const auto& out_idx_shape = output_indices_tensor.logical_shape();

    TT_FATAL(bias_shape == in_shape, "Bias and input tensors must have the same shape");
    TT_FATAL(in_idx_shape == in_shape, "Input indices and input tensors must have the same shape");
    TT_FATAL(out_idx_shape == out_shape, "Output indices and output tensors must have the same shape");

    TT_FATAL(in_shape.size() >= 2, "input_tensor must have rank >= 2");
    uint32_t h = in_shape[in_shape.size() - 2];
    uint32_t w = in_shape[in_shape.size() - 1];
    TT_FATAL(h * w == 256, "Input tensor must have 256 elements per shard (last two dims)");

    const auto& input_shard = input_tensor.shard_spec().value();
    const auto& output_shard = output_tensor.shard_spec().value();
    const auto& bias_shard = bias_tensor.memory_config().shard_spec().value();
    const auto& in_indices_shard = input_indices_tensor.memory_config().shard_spec().value();
    const auto& out_indices_shard = output_indices_tensor.memory_config().shard_spec().value();

    CoreRangeSet all_cores = input_shard.grid;

    TT_FATAL(input_shard.shape == bias_shard.shape, "Input and bias shard shapes must match");
    TT_FATAL(input_shard.orientation == bias_shard.orientation, "Input and bias shard orientations must match");
    TT_FATAL(bias_shard.grid.contains(all_cores), "Bias shard grid must contain input shard grid");

    TT_FATAL(input_shard.shape == in_indices_shard.shape, "Input and input-indices shard shapes must match");
    TT_FATAL(
        input_shard.orientation == in_indices_shard.orientation, "Input and input-indices orientations must match");
    TT_FATAL(in_indices_shard.grid.contains(all_cores), "Input-indices shard grid must contain input shard grid");

    TT_FATAL(output_shard.grid == out_indices_shard.grid, "Output and output-indices shard grids must match");
    TT_FATAL(output_shard.shape == out_indices_shard.shape, "Output and output-indices shard shapes must match");
    TT_FATAL(output_shard.orientation == out_indices_shard.orientation, "Output orientations must match");
    TT_FATAL(output_shard.grid.contains(all_cores), "Output shard grid must contain input compute grid");

    const auto& in_tile = input_tensor.tensor_spec().tile();
    const auto& out_tile = output_tensor.tensor_spec().tile();
    TT_FATAL(in_tile == bias_tensor.tensor_spec().tile(), "Input and bias tiles must match");
    TT_FATAL(in_tile == input_indices_tensor.tensor_spec().tile(), "Input and input-indices tiles must match");
    TT_FATAL(out_tile == output_indices_tensor.tensor_spec().tile(), "Output tiles must match");

    TT_FATAL(in_tile.get_height() == 32 && in_tile.get_width() == 32, "Input tile must be 32x32");
    TT_FATAL(out_tile.get_height() == 32 && out_tile.get_width() == 32, "Output tile must be 32x32");
    TT_FATAL(input_shard.shape[0] == 32 && input_shard.shape[1] == 32, "Input shard shape must be 32x32");
    TT_FATAL(output_shard.shape[0] == 32 && output_shard.shape[1] == 32, "Output shard shape must be 32x32");

    constexpr uint8_t input_cb = 0;
    constexpr uint8_t bias_cb = 1;
    constexpr uint8_t output_cb = 2;
    constexpr uint8_t input_indices_cb = 3;
    constexpr uint8_t output_indices_cb = 4;

    auto in_cb_desc = tt::tt_metal::cb_descriptor_from_sharded_tensor(input_cb, input_tensor);
    auto bias_cb_desc = tt::tt_metal::cb_descriptor_from_sharded_tensor(bias_cb, bias_tensor);
    auto out_cb_desc = tt::tt_metal::cb_descriptor_from_sharded_tensor(output_cb, output_tensor);
    auto in_indices_cb_desc = tt::tt_metal::cb_descriptor_from_sharded_tensor(input_indices_cb, input_indices_tensor);
    auto out_indices_cb_desc =
        tt::tt_metal::cb_descriptor_from_sharded_tensor(output_indices_cb, output_indices_tensor);

    set_cb_page_size_for_tile(in_cb_desc, input_tensor);
    set_cb_page_size_for_tile(bias_cb_desc, bias_tensor);
    set_cb_page_size_for_tile(out_cb_desc, output_tensor);
    set_cb_page_size_for_tile(in_indices_cb_desc, input_indices_tensor);
    set_cb_page_size_for_tile(out_indices_cb_desc, output_indices_tensor);

    KernelDescriptor::NamedCompileTimeArgs ncrisc_named = {
        {"moe_gate_input_cb", input_cb},
        {"moe_gate_bias_cb", bias_cb},
        {"moe_gate_input_indices_cb", input_indices_cb},
        {"moe_gate_is_active_core", 1},
    };
    KernelDescriptor::NamedCompileTimeArgs brisc_named = {
        {"moe_gate_output_cb", output_cb},
        {"moe_gate_output_indices_cb", output_indices_cb},
        {"moe_gate_is_active_core", 1},
    };
    KernelDescriptor::NamedCompileTimeArgs trisc_named = {
        {"moe_gate_input_cb", input_cb},
        {"moe_gate_bias_cb", bias_cb},
        {"moe_gate_input_indices_cb", input_indices_cb},
        {"moe_gate_output_cb", output_cb},
        {"moe_gate_output_indices_cb", output_indices_cb},
        {"moe_gate_eps", float_bits_u32(eps)},
        {"moe_gate_scaling_factor", float_bits_u32(scaling_factor)},
        {"moe_gate_enable_sigmoid", enable_sigmoid ? 1u : 0u},
        {"moe_gate_is_active_core", 1},
    };

    tt::tt_metal::ComputeConfigDescriptor compute_config{};
    compute_config.math_fidelity = MathFidelity::HiFi4;
    compute_config.math_approx_mode = false;
    compute_config.fp32_dest_acc_en = false;
    compute_config.dst_full_sync_en = false;

    KernelDescriptor reader{
        .kernel_source = std::string(kDeepseekMoeGateKernelPath),
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = all_cores,
        .named_compile_time_args = std::move(ncrisc_named),
        .config =
            DataMovementConfigDescriptor{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_0_default,
                .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
            },
    };

    KernelDescriptor writer{
        .kernel_source = std::string(kDeepseekMoeGateKernelPath),
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = all_cores,
        .named_compile_time_args = std::move(brisc_named),
        .config =
            DataMovementConfigDescriptor{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_1_default,
                .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
            },
    };

    KernelDescriptor compute{
        .kernel_source = std::string(kDeepseekMoeGateKernelPath),
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = all_cores,
        .named_compile_time_args = std::move(trisc_named),
        .config = compute_config,
    };

    tt::tt_metal::ProgramDescriptor program_desc;
    program_desc.kernels.reserve(3);
    program_desc.kernels.push_back(std::move(reader));
    program_desc.kernels.push_back(std::move(writer));
    program_desc.kernels.push_back(std::move(compute));

    program_desc.cbs.reserve(5);
    program_desc.cbs.push_back(std::move(in_cb_desc));
    program_desc.cbs.push_back(std::move(bias_cb_desc));
    program_desc.cbs.push_back(std::move(out_cb_desc));
    program_desc.cbs.push_back(std::move(in_indices_cb_desc));
    program_desc.cbs.push_back(std::move(out_indices_cb_desc));

    std::vector<tt::tt_metal::Tensor> io_tensors = {
        input_tensor, bias_tensor, input_indices_tensor, output_tensor, output_indices_tensor};
    ttnn::generic_op(io_tensors, program_desc);

    return {output_tensor, output_indices_tensor};
}

}  // namespace ttnn::experimental::deepseek::moe
