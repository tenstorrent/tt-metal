// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sys/types.h>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"

#include "tt_metal/common/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

namespace ttnn::operations::mul_add {

struct MulAddDeviceOperation {
    struct operation_attributes_t {
        bool attribute;
    };

    struct tensor_args_t {
        const Tensor& input_tensor_a;
        const Tensor& input_tensor_b;
        const Tensor& input_tensor_c;
    };

    using tensor_return_value_t = Tensor;
    using spec_return_value_t = TensorSpec;

    struct MulAddProgramFactorySingleCore {
        struct shared_variables_t {
            KernelHandle reader_kernel_id;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            tt::tt_metal::Program program{};

            const auto& input_tensor_a = tensor_args.input_tensor_a;
            const auto& input_tensor_b = tensor_args.input_tensor_b;
            auto& output = tensor_return_value;
            auto src_buffer_a = input_tensor_a.buffer();
            auto src_buffer_b = input_tensor_a.buffer();
            auto dst_buffer = output.buffer();

            tt::tt_metal::Device* device = input_tensor_a.device();
            auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
            uint32_t num_cores_x = compute_with_storage_grid_size.x;
            uint32_t num_cores_y = compute_with_storage_grid_size.y;
            auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

            auto compute_kernel_id = tt::tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/eltwise/mul_add/device/kernels/compute/muladd_compute.cpp",
                all_device_cores,
                tt::tt_metal::ComputeConfig{.fp32_dest_acc_en = false});

            uint32_t output_cb_index = tt::CBIndex::c_2;
            uint32_t num_output_tiles = 2;
            tt::DataFormat dst_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
            uint32_t dst_single_tile_size = tt::tt_metal::detail::TileSize(dst_cb_data_format);

            tt::tt_metal::CircularBufferConfig cb_output_config =
                tt::tt_metal::CircularBufferConfig(
                    num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
                    .set_page_size(output_cb_index, dst_single_tile_size);

            auto cb_output = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_output_config);

            KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/eltwise/mul_add/device/kernels/dataflow/writer.cpp",
                all_device_cores,
                tt::tt_metal::WriterDataMovementConfig());

            KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/eltwise/mul_add/device/kernels/dataflow/reader.cpp",
                all_device_cores,
                tt::tt_metal::ReaderDataMovementConfig());

            std::vector<std::vector<uint32_t>> reader_args;
            std::vector<std::vector<uint32_t>> compute_args;
            std::vector<std::vector<uint32_t>> writer_args;

            uint32_t num_cores_total = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;
            std::vector<CoreCoord> cores = grid_to_cores(
                num_cores_total,
                compute_with_storage_grid_size.x,
                compute_with_storage_grid_size.y,
                /*row_major*/ true);

            reader_args = {cores.size(), std::vector<uint32_t>(4)};
            compute_args = {cores.size(), std::vector<uint32_t>(2)};
            writer_args = {cores.size(), std::vector<uint32_t>(3)};

            uint32_t num_tiles = input_tensor_a.volume() / tt::constants::TILE_HW;
            CoreRangeSet all_cores, core_group_1, core_group_2;
            uint32_t num_cores, num_tiles_per_core_group_1, num_tiles_per_core_group_2;
            std::tie(
                num_cores,
                all_cores,
                core_group_1,
                core_group_2,
                num_tiles_per_core_group_1,
                num_tiles_per_core_group_2) =
                tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles, /*row_major*/ true);

            uint32_t block_size_per_core_group_1 = 1, block_size_per_core_group_2 = 1, max_block_size = 1;
            uint32_t block_cnt_per_core_group_1 = num_tiles_per_core_group_1;
            uint32_t block_cnt_per_core_group_2 = num_tiles_per_core_group_2;

            uint32_t g1_numcores = core_group_1.num_cores();
            uint32_t g2_numcores = core_group_2.num_cores();

            for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; ++i) {
                uint32_t start_id = num_tiles_read;
                uint32_t num_tiles_per_core;
                uint32_t block_cnt_per_core;
                uint32_t block_size_per_core;

                if (i < g1_numcores) {
                    num_tiles_per_core = num_tiles_per_core_group_1;
                    block_cnt_per_core = block_cnt_per_core_group_1;
                    block_size_per_core = block_size_per_core_group_1;
                } else {
                    num_tiles_per_core = num_tiles_per_core_group_2;
                    block_cnt_per_core = block_cnt_per_core_group_2;
                    block_size_per_core = block_size_per_core_group_2;
                }

                reader_args[i] = {src_buffer_a->address(), src_buffer_b->address(), num_tiles_per_core, start_id};
                compute_args[i] = {block_cnt_per_core, block_size_per_core};
                writer_args[i] = {dst_buffer->address(), num_tiles_per_core, num_tiles_read};

                num_tiles_read += num_tiles_per_core;
            }

            SetRuntimeArgs(program, reader_kernel_id, cores, reader_args);
            SetRuntimeArgs(program, compute_kernel_id, cores, compute_args);
            SetRuntimeArgs(program, writer_kernel_id, cores, writer_args);

            return {std::move(program), {.reader_kernel_id = 1}};
        }

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            Tensor& tensor_return_value) {}
    };

    struct MulAddProgramFactoryMultiCore {
        struct shared_variables_t {
            KernelHandle reader_kernel_id;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            tt::tt_metal::Program program{};
            return {std::move(program), {.reader_kernel_id = 1}};
        }

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {}
    };

    using program_factory_t = std::variant<MulAddProgramFactorySingleCore, MulAddProgramFactoryMultiCore>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor_a, const Tensor& input_tensor_b, const Tensor& input_tensor_c);
};

}  // namespace ttnn::operations::mul_add

namespace ttnn::prim {

constexpr auto muladd =
    ttnn::register_operation<"ttnn::prim::muladd", ttnn::operations::mul_add::MulAddDeviceOperation>();
}  // namespace ttnn::prim
