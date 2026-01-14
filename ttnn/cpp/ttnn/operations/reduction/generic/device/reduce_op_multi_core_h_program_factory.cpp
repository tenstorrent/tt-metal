// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_op_multi_core_h_program_factory.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <cmath>

using namespace tt::constants;

namespace ttnn::operations::reduction::generic::program {

ReduceMultiCoreHProgramFactory::cached_program_t ReduceMultiCoreHProgramFactory::create(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    const auto& a = tensor_args.input_tensor;
    auto& output = tensor_return_value;
    const auto& shape = a.padded_shape();
    uint32_t W = shape[3], H = shape[2], NC = shape[1] * shape[0];

    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ht = H / TILE_HEIGHT;
    uint32_t HtWt = Ht * Wt;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), operation_attributes.compute_kernel_config);

    tt_metal::Program program = tt_metal::CreateProgram();

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    tt::DataFormat scaler_cb_data_format = DataFormat::Float16_b;
    uint32_t scaler_single_tile_size = tt::tile_size(scaler_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    tt_metal::IDevice* device = a.device();

    bool use_width_sharding = a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
                              output.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto num_cols = NC * Wt;
    uint32_t num_cores;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_cols_per_core_group_1, num_cols_per_core_group_2;
    if (operation_attributes.sub_core_grids.has_value()) {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(*operation_attributes.sub_core_grids, num_cols);
    } else {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_cols);
    }

    // Current sharding only supports width, and that input and output are sharded
    if (use_width_sharding) {
        all_cores = a.shard_spec().value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
        num_cols_per_core_group_1 = NC * (a.shard_spec().value().shape[1] / TILE_WIDTH);
        num_cols_per_core_group_2 = 0;
    }

    uint32_t src0_cb_index = CBIndex::c_0;
    uint32_t src1_cb_index = CBIndex::c_1;
    CBHandle cb_src1 = 0;
    if (use_width_sharding) {
        uint32_t num_shard_tiles = a.shard_spec().value().numel() / TILE_HW;
        uint32_t num_input_tiles = 2;
        tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(
                num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
                .set_page_size(src0_cb_index, src0_single_tile_size);
        tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

        tt_metal::CircularBufferConfig cb_src1_config =
            tt_metal::CircularBufferConfig(
                num_shard_tiles * src0_single_tile_size, {{src1_cb_index, src0_cb_data_format}})
                .set_page_size(src1_cb_index, src0_single_tile_size)
                .set_globally_allocated_address(*a.buffer());
        cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);
    } else {
        uint32_t num_input_tiles = 2;
        tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(
                num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
                .set_page_size(src0_cb_index, src0_single_tile_size);
        tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);
    }

    uint32_t scaler_cb_index = CBIndex::c_2;
    tt_metal::CircularBufferConfig cb_scaler_config =
        tt_metal::CircularBufferConfig(1 * scaler_single_tile_size, {{scaler_cb_index, scaler_cb_data_format}})
            .set_page_size(scaler_cb_index, scaler_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_scaler_config);

    uint32_t output_cb_index = CBIndex::c_3;
    CBHandle cb_output = 0;
    if (use_width_sharding) {
        uint32_t num_output_tiles = output.shard_spec().value().numel() / TILE_HW;
        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
                .set_page_size(output_cb_index, dst_single_tile_size)
                .set_globally_allocated_address(*output.buffer());
        cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);
    } else {
        uint32_t num_output_tiles = 2;
        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
                .set_page_size(output_cb_index, dst_single_tile_size);
        cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);
    }
    tt_metal::Buffer* src0_buffer = a.buffer();
    tt_metal::KernelHandle reader_kernel_id;
    bfloat16 bfloat_scaler_value = bfloat16::truncate(operation_attributes.scaler);
    uint32_t packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});

    uint32_t chunk_size = use_width_sharding ? 1 : ttnn::get_dest_reg_count(operation_attributes.compute_kernel_config);

    if (use_width_sharding) {
        std::vector<uint32_t> reader_compile_time_args = {src0_cb_index, src1_cb_index, scaler_cb_index};
        std::map<std::string, std::string> reader_defines;
        reader_defines["REDUCE_SCALER"] = "1";
        reader_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp",
            all_cores,
            tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));
    } else {
        std::vector<uint32_t> reader_compile_time_args = {Ht, Wt, HtWt, chunk_size, packed_scaler_value};
        TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);

        reader_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_transpose_wh_universal_input_cols_partitioned.cpp",
            all_cores,
            tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    }

    tt_metal::Buffer* dst_buffer = output.buffer();
    tt_metal::KernelHandle writer_kernel_id;

    if (use_width_sharding) {
        std::vector<uint32_t> writer_ct_args = {
            output_cb_index,
        };
        writer_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp",
            all_cores,
            WriterDataMovementConfig(writer_ct_args));
    } else {
        std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index};
        TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

        writer_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
            all_cores,
            tt_metal::WriterDataMovementConfig(writer_compile_time_args));
    }
    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils::get_defines(operation_attributes.math_op, tt::tt_metal::ReduceOpDim::H);
    std::vector<uint32_t> compute_kernel_args_group_1 = {
        Ht,                         // Ht
        num_cols_per_core_group_1,  // Wt
        1,                          // NC
        chunk_size,                 // Column Chunk Size
    };

    tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_h.cpp",
        core_group_1,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = compute_kernel_args_group_1,
            .defines = reduce_defines});

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_kernel_args_group_2 = {
            Ht,                         // Ht
            num_cols_per_core_group_2,  // Wt
            1,                          // NC
            chunk_size,                 // Column Chunk Size
        };

        tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_h.cpp",
            core_group_2,
            tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .compile_args = compute_kernel_args_group_2,
                .defines = reduce_defines});
    }

    std::vector<CoreCoord> cores;
    if (operation_attributes.sub_core_grids.has_value()) {
        for (const auto& range : all_cores.ranges()) {
            for (int y = range.start_coord.y; y <= range.end_coord.y; ++y) {
                for (int x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                    cores.emplace_back(x, y);
                }
            }
        }
    } else {
        cores = grid_to_cores(num_cores, compute_with_storage_grid_size.x, compute_with_storage_grid_size.y, false);
    }
    if (use_width_sharding) {
        uint32_t shard_Wt = num_cols_per_core_group_1 / NC;
        uint32_t shard_row_size = shard_Wt * src0_single_tile_size;
        uint32_t shard_batch_size = shard_row_size * Ht;
        std::vector<uint32_t> reader_rt_args = {
            num_cols_per_core_group_1 * Ht, shard_Wt, Ht, NC, shard_row_size, shard_batch_size, packed_scaler_value};
        tt_metal::SetRuntimeArgs(program, reader_kernel_id, all_cores, reader_rt_args);

        std::vector<uint32_t> writer_rt_args = {num_cols_per_core_group_1};
        tt_metal::SetRuntimeArgs(program, writer_kernel_id, all_cores, writer_rt_args);
    } else {
        for (uint32_t i = 0, num_cols_read = 0; i < num_cores; i++) {
            const CoreCoord& core = cores[i];
            uint32_t num_cols_per_core = 0;
            if (core_group_1.contains(core)) {
                num_cols_per_core = num_cols_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_cols_per_core = num_cols_per_core_group_2;
            } else {
                TT_THROW("Core not in specified core ranges");
            }
            tt_metal::SetRuntimeArgs(
                program,
                reader_kernel_id,
                core,
                {a.buffer()->address(),
                 (num_cols_read / Wt * HtWt) + (num_cols_read % Wt),
                 num_cols_read % Wt,
                 num_cols_per_core});

            tt_metal::SetRuntimeArgs(
                program,
                writer_kernel_id,
                core,
                {
                    output.buffer()->address(),
                    num_cols_per_core,  // number of tiles to write
                    num_cols_read       // output tile start index
                });
            num_cols_read += num_cols_per_core;
        }
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, cb_src1, cb_output, cores}};
}

void ReduceMultiCoreHProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    auto* src_buffer = tensor_args.input_tensor.buffer();
    auto* dst_buffer = tensor_return_value.buffer();

    bool use_width_sharding =
        tensor_args.input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
        tensor_return_value.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;

    if (use_width_sharding) {
        UpdateDynamicCircularBufferAddress(
            cached_program.program, cached_program.shared_variables.cb_src1, *src_buffer);
        UpdateDynamicCircularBufferAddress(
            cached_program.program, cached_program.shared_variables.cb_output, *dst_buffer);
    } else {
        auto& reader_runtime_args_by_core =
            GetRuntimeArgs(cached_program.program, cached_program.shared_variables.reader_kernel_id);
        auto& writer_runtime_args_by_core =
            GetRuntimeArgs(cached_program.program, cached_program.shared_variables.writer_kernel_id);
        for (const auto& core : cached_program.shared_variables.cores) {
            {
                auto& runtime_args = reader_runtime_args_by_core[core.x][core.y];
                runtime_args[0] = src_buffer->address();
            }

            {
                auto& runtime_args = writer_runtime_args_by_core[core.x][core.y];
                runtime_args[0] = dst_buffer->address();
            }
        }
    }
}

}  // namespace ttnn::operations::reduction::generic::program
