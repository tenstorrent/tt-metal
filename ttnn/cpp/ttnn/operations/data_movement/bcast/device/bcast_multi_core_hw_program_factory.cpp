// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/bcast/bcast_types.hpp"
#include "bcast_multi_core_hw_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::data_movement::bcast::program {

using namespace tt::tt_metal;
using namespace tt::constants;

BcastMultiCoreHWProgramFactory::cached_program_t BcastMultiCoreHWProgramFactory::create(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& tensor_return_value) {
    const Tensor& a = tensor_args.input_a;
    const Tensor& b = tensor_args.input_b;
    Tensor& output = tensor_return_value;

    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();
    const uint32_t N = ashape.rank() >= 4 ? ashape[-4] : 1;
    const uint32_t C = ashape.rank() >= 3 ? ashape[-3] : 1;
    const uint32_t H = ashape[-2];
    const uint32_t W = ashape[-1];
    const uint32_t bN = bshape.rank() >= 4 ? bshape[-4] : 1;
    const uint32_t bC = bshape.rank() >= 3 ? bshape[-3] : 1;
    const uint32_t NC = N * C;

    const uint32_t Wt = W / TILE_WIDTH;
    const uint32_t Ht = H / TILE_HEIGHT;
    const uint32_t HtWt = Ht * Wt;

    const uint32_t num_tensor_tiles = NC * Ht * Wt;

    const uint32_t bnc1 = (bN * bC == 1);

    Program program = CreateProgram();

    IDevice* device = a.device();

    std::optional<ShardSpec> shard_spec = std::nullopt;
    const bool src0_sharded = a.memory_config().is_sharded();
    const bool output_sharded = output.memory_config().is_sharded();
    if (src0_sharded) {
        shard_spec = a.shard_spec().value();
    } else if (output_sharded) {
        shard_spec = output.shard_spec().value();
    }

    const tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    const tt::DataFormat src1_cb_data_format = datatype_to_dataformat_converter(b.dtype());
    const tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output.dtype());

    const uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    const uint32_t src1_single_tile_size = tt::tile_size(src1_cb_data_format);
    const uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles);

    Buffer* src0_buffer = a.buffer();
    Buffer* src1_buffer = b.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const uint32_t src0_cb_index = 0;
    const uint32_t num_input_tiles = 2;
    uint32_t num_tiles_per_shard = 0;
    if (shard_spec.has_value()) {
        num_tiles_per_shard = shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW;
        num_tiles_per_core_group_1 = num_tiles_per_shard;
        num_tiles_per_core_group_2 = 0;
        all_cores = shard_spec.value().grid;
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
    }

    const uint32_t num_input_tiles_cb0 = src0_sharded ? num_tiles_per_shard : num_input_tiles;

    CircularBufferConfig src0_cb_config =
        CircularBufferConfig(num_input_tiles_cb0 * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
            .set_page_size(src0_cb_index, src0_single_tile_size);
    if (src0_sharded) {
        src0_cb_config = src0_cb_config.set_globally_allocated_address(*a.buffer());
    }
    const auto cb_src0 = CreateCircularBuffer(program, all_device_cores, src0_cb_config);

    const uint32_t src1_cb_index = 1;
    CircularBufferConfig src1_cb_config =
        CircularBufferConfig(num_input_tiles * src1_single_tile_size, {{src1_cb_index, src1_cb_data_format}})
            .set_page_size(src1_cb_index, src1_single_tile_size);
    CreateCircularBuffer(program, all_device_cores, src1_cb_config);

    const uint32_t output_cb_index = tt::CBIndex::c_16;
    const uint32_t num_output_tiles = output_sharded ? num_tiles_per_shard : 2;
    CircularBufferConfig output_cb_config =
        CircularBufferConfig(num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, dst_single_tile_size);
    if (output_sharded) {
        output_cb_config = output_cb_config.set_globally_allocated_address(*output.buffer());
    }
    const auto cb_output = CreateCircularBuffer(program, all_device_cores, output_cb_config);

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    std::map<std::string, std::string> reader_defines;
    std::vector<uint32_t> reader_compile_time_args;
    std::map<std::string, std::string> bcast_compute_defines =
        bcast_op_utils::get_defines(BcastOpDim::HW, operation_attributes.math_op);
    if (bnc1) {
        reader_defines["BCAST_SCALAR"] = "1";
        bcast_compute_defines["BCAST_SCALAR"] = "1";
    }
    if (src0_sharded) {
        reader_defines["IN0_SHARDED"] = "1";
    } else {
        TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    }
    TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args);
    const KernelHandle binary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/dataflow/"
        "reader_bcast_hw_interleaved_partitioned.cpp",
        all_device_cores,
        ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    std::map<std::string, std::string> writer_defines;
    if (output_sharded) {
        writer_defines["OUT_SHARDED"] = "1";
    }
    const KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_device_cores,
        WriterDataMovementConfig(writer_compile_time_args, writer_defines));

    const auto bcast_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/compute/bcast_hw.cpp",
        all_device_cores,
        ComputeConfig{.compile_args = {}, .defines = bcast_compute_defines});

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_y * num_cores_x; i++) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tensor_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tensor_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tensor_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            constexpr std::array<uint32_t, 7> binary_reader_kernel_args{0};
            constexpr std::array<uint32_t, 3> bcast_kernel_args{1, 1, 0};
            constexpr std::array<uint32_t, 3> unary_writer_kernel_args{0};

            SetRuntimeArgs(program, binary_reader_kernel_id, core, binary_reader_kernel_args);
            SetRuntimeArgs(program, bcast_kernel_id, core, bcast_kernel_args);
            SetRuntimeArgs(program, unary_writer_kernel_id, core, unary_writer_kernel_args);
            continue;
        }

        SetRuntimeArgs(
            program,
            binary_reader_kernel_id,
            core,
            {a.buffer()->address(),  // 0
             b.buffer()->address(),
             num_tensor_tiles_per_core,
             HtWt,
             num_tiles_read / HtWt * HtWt,
             num_tiles_read % HtWt,
             bnc1 ? 0 : num_tiles_read / HtWt});

        SetRuntimeArgs(
            program,
            bcast_kernel_id,
            core,
            {
                1,                         // B
                1,                         // Ht
                num_tensor_tiles_per_core  // Wt
            });

        SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            {
                output.buffer()->address(),
                num_tensor_tiles_per_core,
                num_tiles_read,
            });
        num_tiles_read += num_tensor_tiles_per_core;
    }

    return cached_program_t{
        std::move(program),
        {binary_reader_kernel_id,
         unary_writer_kernel_id,
         bcast_kernel_id,
         compute_with_storage_grid_size,
         (src0_sharded ? std::make_optional(cb_src0) : std::nullopt),
         (output_sharded ? std::make_optional(cb_output) : std::nullopt),
         src0_single_tile_size,
         dst_single_tile_size,
         operation_attributes.in_place}};
}

void BcastMultiCoreHWProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    const uint32_t num_cores_x = cached_program.shared_variables.compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = cached_program.shared_variables.compute_with_storage_grid_size.y;
    const auto& output_tensor = cached_program.shared_variables.inplace ? tensor_args.input_a : tensor_return_value;

    Buffer* src_buffer_a = tensor_args.input_a.buffer();
    Buffer* src_dram_buffer_b = tensor_args.input_b.buffer();
    std::optional<ShardSpec> shard_spec = std::nullopt;
    const bool src0_sharded = tensor_args.input_a.memory_config().is_sharded();
    const bool out_sharded = output_tensor.memory_config().is_sharded();

    if (src0_sharded) {
        shard_spec = tensor_args.input_a.shard_spec().value();
    } else if (out_sharded) {
        shard_spec = output_tensor.shard_spec().value();
    }

    Buffer* dst_buffer = output_tensor.buffer();

    const auto ashape = tensor_args.input_a.padded_shape();
    const auto bshape = tensor_args.input_b.padded_shape();
    const uint32_t N = ashape.rank() >= 4 ? ashape[-4] : 1;
    const uint32_t C = ashape.rank() >= 3 ? ashape[-3] : 1;
    const uint32_t H = ashape[-2];
    const uint32_t W = ashape[-1];
    const uint32_t bN = bshape.rank() >= 4 ? bshape[-4] : 1;
    const uint32_t bC = bshape.rank() >= 3 ? bshape[-3] : 1;
    const uint32_t NC = N * C;

    const uint32_t Wt = W / TILE_WIDTH;
    const uint32_t Ht = H / TILE_HEIGHT;
    const uint32_t HtWt = Ht * Wt;

    const uint32_t num_tensor_tiles = NC * Ht * Wt;
    const uint32_t bnc1 = (bN * bC == 1);

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(cached_program.shared_variables.compute_with_storage_grid_size, num_tensor_tiles);

    if (shard_spec.has_value()) {
        const uint32_t num_tiles_per_shard = shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW;
        num_tiles_per_core_group_1 = num_tiles_per_shard;
        num_tiles_per_core_group_2 = 0;
        all_cores = shard_spec.value().grid;
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
    }

    auto& cached_reader_args =
        GetRuntimeArgs(cached_program.program, cached_program.shared_variables.binary_reader_kernel_id);
    auto& cached_eltwise_args = GetRuntimeArgs(cached_program.program, cached_program.shared_variables.bcast_kernel_id);
    auto& cached_writer_args =
        GetRuntimeArgs(cached_program.program, cached_program.shared_variables.unary_writer_kernel_id);

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_y * num_cores_x; i++) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tensor_tiles_per_core;

        auto& binary_reader_args = cached_reader_args.at(core.x).at(core.y);
        auto& bcast_kernel_args = cached_eltwise_args.at(core.x).at(core.y);
        auto& unary_writer_args = cached_writer_args.at(core.x).at(core.y);

        if (core_group_1.contains(core)) {
            num_tensor_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tensor_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            binary_reader_args[2] = 0;
            bcast_kernel_args[2] = 0;
            unary_writer_args[1] = 0;
            continue;
        }

        binary_reader_args[0] = src_buffer_a->address();
        binary_reader_args[1] = src_dram_buffer_b->address();
        binary_reader_args[2] = num_tensor_tiles_per_core;
        binary_reader_args[3] = HtWt;
        binary_reader_args[4] = num_tiles_read / HtWt * HtWt;
        binary_reader_args[5] = num_tiles_read % HtWt;
        binary_reader_args[6] = bnc1 ? 0 : num_tiles_read / HtWt;

        bcast_kernel_args[2] = num_tensor_tiles_per_core;

        unary_writer_args[0] = dst_buffer->address();
        unary_writer_args[1] = num_tensor_tiles_per_core;
        unary_writer_args[2] = num_tiles_read;

        num_tiles_read += num_tensor_tiles_per_core;
    }

    if (src0_sharded && cached_program.shared_variables.cb_src0.has_value()) {
        UpdateDynamicCircularBufferAddressAndTotalSize(
            cached_program.program,
            cached_program.shared_variables.cb_src0.value(),
            *src_buffer_a,
            num_tiles_per_core_group_1 * cached_program.shared_variables.src0_single_tile_size);
    }

    if (out_sharded && cached_program.shared_variables.cb_output.has_value()) {
        UpdateDynamicCircularBufferAddressAndTotalSize(
            cached_program.program,
            cached_program.shared_variables.cb_output.value(),
            *dst_buffer,
            num_tiles_per_core_group_1 * cached_program.shared_variables.dst_single_tile_size);
    }
}

}  // namespace ttnn::operations::data_movement::bcast::program
