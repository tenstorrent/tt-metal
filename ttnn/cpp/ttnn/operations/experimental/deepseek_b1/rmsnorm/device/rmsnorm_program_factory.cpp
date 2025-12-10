// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_device_operation.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

#define DEB(x) x

void print_map(std::unordered_map<std::string, uint32_t>& m, const std::string& header) {
    std::cout << header << std::endl;
    for (const auto& pair : m) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }
    std::cout << std::endl;
}

using namespace tt::constants;

namespace ttnn::operations::experimental::deepseek_b1::rmsnorm {

RmsnormDeviceOperation::RmsnormProgramFactory::cached_program_t RmsnormDeviceOperation::RmsnormProgramFactory::create(
    const RmsnormDeviceOperation::operation_attributes_t& operation_attributes,
    const RmsnormDeviceOperation::tensor_args_t& tensor_args,
    RmsnormDeviceOperation::tensor_return_value_t& tensor_return_value) {
    const Tensor& input = tensor_args.input_tensor;
    const Tensor& gamma = tensor_args.gamma_tensor;
    Tensor& output_tensor = tensor_return_value;

    /*
    DEB(std::cout << "Input " << std::endl;)
    DEB(std::cout << input;)
    DEB(std::cout << "Gamma " << std::endl;)
    DEB(std::cout << gamma;)
    */

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // Get shard specs - all tensors should be sharded on the same cores
    const auto& input_shard_spec = input.memory_config().shard_spec().value();
    const auto& cores = input_shard_spec.grid;

    tt::tt_metal::IDevice* device = input.device();
    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());

    DEB(std::cout << "Data forma " << data_format << std::endl;)

    bool tiny_tile = (input.logical_shape()[1] / tt::constants::TILE_WIDTH) % tt::constants::TILE_HEIGHT != 0;
    uint32_t tile_height = tiny_tile ? tt::constants::TILE_HEIGHT / 2 : tt::constants::TILE_HEIGHT;
    tt::tt_metal::Tile tile({tile_height, tt::constants::TILE_WIDTH});

    // Calculate number of tiles and elements per shard
    uint32_t single_tile_size = tiny_tile ? tt::tile_size(data_format) / 2 : tt::tile_size(data_format);
    uint32_t shard_size_bytes = input.buffer()->aligned_size() / cores.num_cores();
    uint32_t num_tiles = shard_size_bytes / single_tile_size;

    DEB(std::cout << "Single tile size " << single_tile_size << " shard_size_bytes " << shard_size_bytes
                  << " num_tiles " << num_tiles << std::endl;)

    // Create circular buffers (all globally allocated, backed by L1 shards)
    uint32_t cb_idx = 0;
    // CB 0: Input (backed by input tensor L1 shard)
    uint32_t input_cb = cb_idx++;
    tt::tt_metal::CircularBufferConfig input_cb_config =
        tt::tt_metal::CircularBufferConfig(num_tiles * single_tile_size, {{input_cb, data_format}})
            .set_page_size(input_cb, single_tile_size)
            .set_tile_dims(input_cb, tile)
            .set_globally_allocated_address(*input.buffer());
    tt::tt_metal::CreateCircularBuffer(program, cores, input_cb_config);

    // CB 1: Scalars (epsilon and reduction scalar 1/sqrt(num_elements))
    uint32_t scalars_cb = cb_idx++;
    tt::tt_metal::CircularBufferConfig scalars_cb_config =
        tt::tt_metal::CircularBufferConfig(2 * single_tile_size, {{scalars_cb, data_format}})
            .set_page_size(scalars_cb, single_tile_size)
            .set_tile_dims(scalars_cb, tile);
    tt::tt_metal::CreateCircularBuffer(program, cores, scalars_cb_config);

    // CB 2: Intermediate buffer for squared values and RMS
    uint32_t interm_cb = cb_idx++;
    tt::tt_metal::CircularBufferConfig interm_cb_config =
        tt::tt_metal::CircularBufferConfig((num_tiles + 1) * single_tile_size, {{interm_cb, data_format}})
            .set_page_size(interm_cb, single_tile_size)
            .set_tile_dims(interm_cb, tile);
    tt::tt_metal::CreateCircularBuffer(program, cores, interm_cb_config);

    // CB 3: Gamma (backed by gamma tensor L1 shard)
    uint32_t gamma_cb = cb_idx++;
    tt::tt_metal::CircularBufferConfig gamma_cb_config =
        tt::tt_metal::CircularBufferConfig(num_tiles * single_tile_size, {{gamma_cb, data_format}})
            .set_page_size(gamma_cb, single_tile_size)
            .set_tile_dims(gamma_cb, tile)
            .set_globally_allocated_address(*gamma.buffer());
    tt::tt_metal::CreateCircularBuffer(program, cores, gamma_cb_config);

    // CB 4: Output (backed by output tensor L1 shard)
    uint32_t output_cb = cb_idx++;
    tt::tt_metal::CircularBufferConfig output_cb_config =
        tt::tt_metal::CircularBufferConfig(num_tiles * single_tile_size, {{output_cb, data_format}})
            .set_page_size(output_cb, single_tile_size)
            .set_tile_dims(output_cb, tile)
            .set_globally_allocated_address(*output_tensor.buffer());
    tt::tt_metal::CreateCircularBuffer(program, cores, output_cb_config);

    // Create reader kernel
    const std::string reader_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_b1/rmsnorm/device/kernels/rmsnorm_reader.cpp";

    std::unordered_map<std::string, uint32_t> reader_named_compile_args = {
        {"input_cb", input_cb},
        {"scalars_cb", scalars_cb},
        {"gamma_cb", gamma_cb},
        {"num_tiles", num_tiles},
        {"tiny_tile", tiny_tile ? 1u : 0u},
    };

    DEB(print_map(reader_named_compile_args, "reader_named_compile_args");)

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        reader_kernel_path,
        cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .named_compile_args = reader_named_compile_args});

    // Create compute kernel
    const std::string compute_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_b1/rmsnorm/device/kernels/rmsnorm_compute.cpp";

    // Extract compute kernel config using the helper
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    std::unordered_map<std::string, uint32_t> compute_named_compile_args = {
        {"rms_input_cb", input_cb},
        {"rms_scalars_cb", scalars_cb},
        {"rms_interm_cb", interm_cb},
        {"rms_gamma_cb", gamma_cb},
        {"rms_output_cb", output_cb},
        {"rms_enforce_fp32_accumulation", fp32_dest_acc_en ? 1u : 0u},
        {"rms_num_tiles", num_tiles},
        {"rms_epsilon_index", 0},  // First tile in scalars_cb
        {"rms_scalar_index", 1},   // Second tile in scalars_cb
    };

    DEB(print_map(compute_named_compile_args, "compute_named_compile_args");)

    tt::tt_metal::CreateKernel(
        program,
        compute_kernel_path,
        cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .named_compile_args = compute_named_compile_args});

    // Create writer kernel
    const std::string writer_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_b1/rmsnorm/device/kernels/rmsnorm_writer.cpp";

    std::unordered_map<std::string, uint32_t> writer_named_compile_args = {
        {"output_cb", output_cb},
        {"num_tiles", num_tiles},
    };

    DEB(print_map(writer_named_compile_args, "writer_named_compile_args");)

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        writer_kernel_path,
        cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .named_compile_args = writer_named_compile_args});

    // Set runtime arguments for reader kernel
    // Pack epsilon as bfloat16
    bfloat16 bfloat_epsilon = bfloat16::truncate(operation_attributes.epsilon);
    uint32_t packed_epsilon = pack_two_bfloat16_into_uint32({bfloat_epsilon, bfloat_epsilon});

    // Compute 1/sqrt(num_elements) for RMS reduction and pack as bfloat16
    float inv_sqrt_num_elements = 1.0f / std::sqrt(static_cast<float>(operation_attributes.numel));
    bfloat16 bfloat_scalar = bfloat16::truncate(inv_sqrt_num_elements);
    uint32_t packed_scalar = pack_two_bfloat16_into_uint32({bfloat_scalar, bfloat_scalar});

    DEB(std::cout << "numel " << operation_attributes.numel << std::endl;)
    DEB(std::cout << "epsilon " << operation_attributes.epsilon << " scalar " << inv_sqrt_num_elements << std::endl;)
    DEB(std::cout << "bfloat_epsilon " << bfloat_epsilon << " bfloat_scalar " << bfloat_scalar << std::endl;)

    for (const auto& core_range : cores.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                CoreCoord core(x, y);
                std::vector<uint32_t> reader_runtime_args = {packed_epsilon, packed_scalar};
                tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
            }
        }
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .cores = cores,
         .input_cb = input_cb,
         .gamma_cb = gamma_cb,
         .output_cb = output_cb}};
}

void RmsnormDeviceOperation::RmsnormProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const RmsnormDeviceOperation::operation_attributes_t& operation_attributes,
    const RmsnormDeviceOperation::tensor_args_t& tensor_args,
    RmsnormDeviceOperation::tensor_return_value_t& tensor_return_value) {
    const auto& input = tensor_args.input_tensor;
    const auto& gamma = tensor_args.gamma_tensor;
    const auto& output = tensor_return_value;

    auto& program = cached_program.program;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto& cores = cached_program.shared_variables.cores;
    const auto& input_cb = cached_program.shared_variables.input_cb;
    const auto& gamma_cb = cached_program.shared_variables.gamma_cb;
    const auto& output_cb = cached_program.shared_variables.output_cb;

    // Update circular buffer addresses
    UpdateDynamicCircularBufferAddress(program, input_cb, *input.buffer());
    UpdateDynamicCircularBufferAddress(program, gamma_cb, *gamma.buffer());
    UpdateDynamicCircularBufferAddress(program, output_cb, *output.buffer());

    // Update epsilon if it changed
    bfloat16 bfloat_epsilon = bfloat16::truncate(operation_attributes.epsilon);
    uint32_t packed_epsilon = pack_two_bfloat16_into_uint32({bfloat_epsilon, bfloat_epsilon});

    // Update 1/sqrt(num_elements) if it changed
    float inv_sqrt_num_elements = 1.0f / std::sqrt(static_cast<float>(operation_attributes.numel));
    bfloat16 bfloat_scalar = bfloat16::truncate(inv_sqrt_num_elements);
    uint32_t packed_scalar = pack_two_bfloat16_into_uint32({bfloat_scalar, bfloat_scalar});

    DEB(std::cout << "numel " << operation_attributes.numel << std::endl;)
    DEB(std::cout << "epsilon " << operation_attributes.epsilon << " scalar " << inv_sqrt_num_elements << std::endl;)
    DEB(std::cout << "bfloat_epsilon " << bfloat_epsilon << " bfloat_scalar " << bfloat_scalar << std::endl;)

    for (const auto& core_range : cores.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                CoreCoord core(x, y);
                auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                reader_runtime_args[0] = packed_epsilon;
                reader_runtime_args[1] = packed_scalar;
            }
        }
    }
}

}  // namespace ttnn::operations::experimental::deepseek_b1::rmsnorm
