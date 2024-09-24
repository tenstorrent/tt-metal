// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

using namespace tt::constants;
using uint32_t = std::uint32_t;
namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks reduce_multi_core_h(
    const Tensor &a,
    Tensor &output,
    ReduceOpMath reduce_op,
    const ttnn::DeviceComputeKernelConfig &compute_kernel_config,
    float scaler) {
    const auto shape = a.get_legacy_shape();
    uint32_t W = shape[3], H = shape[2], NC = shape[1] * shape[0];

    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ht = H / TILE_HEIGHT;
    uint32_t HtWt = Ht * Wt;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] =
        get_compute_kernel_config_args(a.device()->arch(), compute_kernel_config);

    tt_metal::Program program = tt_metal::CreateProgram();

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t src0_single_tile_size = tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat scaler_cb_data_format = DataFormat::Float16_b;
    uint32_t scaler_single_tile_size = tt_metal::detail::TileSize(scaler_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t dst_single_tile_size = tt_metal::detail::TileSize(dst_cb_data_format);

    uint32_t num_tiles = a.volume() / TILE_HW;

    tt_metal::Device *device = a.device();

    bool in_sharded = a.is_sharded();
    bool out_sharded = output.is_sharded();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto num_cols = NC * Wt;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_cols);

    // Current sharding only supports width, and that input and output are sharded
    if (in_sharded) {
        all_cores = a.shard_spec().value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet({});
        num_cols_per_core_group_1 = NC * (a.shard_spec().value().shape[1] / TILE_WIDTH);
        num_cols_per_core_group_2 = 0;
    }

    uint32_t src0_cb_index = CB::c_in0;
    CBHandle cb_src0;
    uint32_t src1_cb_index = CB::c_in1;
    CBHandle cb_src1 = 0;
    if (in_sharded) {
        uint32_t num_shard_tiles = a.shard_spec().value().numel() / TILE_HW;
        uint32_t num_input_tiles = 2;
        tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(
                num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
                .set_page_size(src0_cb_index, src0_single_tile_size);
        cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

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
        cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);
    }

    uint32_t scaler_cb_index = CB::c_in2;
    tt_metal::CircularBufferConfig cb_scaler_config =
        tt_metal::CircularBufferConfig(1 * scaler_single_tile_size, {{scaler_cb_index, scaler_cb_data_format}})
            .set_page_size(scaler_cb_index, scaler_single_tile_size);
    auto cb_scaler = tt_metal::CreateCircularBuffer(program, all_cores, cb_scaler_config);

    uint32_t output_cb_index = CB::c_out0;  // output operands start at index 16
    CBHandle cb_output;
    if (out_sharded) {
        uint32_t num_output_tiles = output.shard_spec().value().numel() / TILE_HW;
        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
                .set_page_size(output_cb_index, dst_single_tile_size)
                .set_globally_allocated_address(*output.buffer());
        ;
        cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);
    } else {
        uint32_t num_output_tiles = 2;
        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
                .set_page_size(output_cb_index, dst_single_tile_size);
        cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);
    }
    tt_metal::Buffer *src0_buffer = a.buffer();
    tt_metal::KernelHandle reader_kernel_id;
    bfloat16 bfloat_scaler_value = bfloat16(scaler);
    uint32_t packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});
    if (in_sharded) {
        std::vector<uint32_t> reader_compile_time_args = {src0_cb_index, src1_cb_index, scaler_cb_index};
        std::map<string, string> reader_defines;
        reader_defines["REDUCE_SCALER"] = "1";
        reader_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp",
            all_cores,
            tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));
    } else {
        bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t)src0_is_dram, Ht, Wt, HtWt, packed_scaler_value};

        std::map<string, string> reader_defines;
        reader_defines["REDUCE_SCALER"] = "1";

        reader_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_transpose_wh_interleaved_input_cols_partitioned.cpp",
            all_cores,
            tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));
    }

    tt_metal::Buffer *dst_buffer = output.buffer();
    tt_metal::KernelHandle writer_kernel_id;

    if (out_sharded) {
        vector<uint32_t> writer_ct_args = {
            output_cb_index,
        };
        writer_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp",
            all_cores,
            WriterDataMovementConfig(writer_ct_args));
    } else {
        bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
        std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)dst_is_dram};

        writer_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
            all_cores,
            tt_metal::WriterDataMovementConfig(writer_compile_time_args));
    }
    std::map<string, string> reduce_defines = reduce_op_utils::get_defines(reduce_op, ReduceOpDim::H);
    vector<uint32_t> compute_kernel_args_group_1 = {
        Ht,                         // Ht
        num_cols_per_core_group_1,  // Wt
        1,                          // NC
    };

    auto reduce_compute_kernel_group_1_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_h.cpp",
        core_group_1,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = compute_kernel_args_group_1,
            .defines = reduce_defines});

    if (!core_group_2.ranges().empty()) {
        vector<uint32_t> compute_kernel_args_group_2 = {
            Ht,                         // Ht
            num_cols_per_core_group_2,  // Wt
            1,                          // NC
        };

        auto reduce_compute_kernel_group_2_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_h.cpp",
            core_group_2,
            tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .compile_args = compute_kernel_args_group_2,
                .defines = reduce_defines});
    }

    const auto &cores =
        grid_to_cores(num_cores, compute_with_storage_grid_size.x, compute_with_storage_grid_size.y, false);
    if (in_sharded && out_sharded) {
        uint32_t shard_Wt = num_cols_per_core_group_1 / NC;
        uint32_t shard_row_size = shard_Wt * src0_single_tile_size;
        uint32_t shard_batch_size = shard_row_size * Ht;
        vector<uint32_t> reader_rt_args = {
            num_cols_per_core_group_1 * Ht, shard_Wt, Ht, NC, shard_row_size, shard_batch_size, packed_scaler_value};
        tt_metal::SetRuntimeArgs(program, reader_kernel_id, all_cores, reader_rt_args);

        vector<uint32_t> writer_rt_args = {num_cols_per_core_group_1};
        tt_metal::SetRuntimeArgs(program, writer_kernel_id, all_cores, writer_rt_args);
    } else {
        for (uint32_t i = 0, num_cols_read = 0; i < num_cores; i++) {
            const CoreCoord &core = cores[i];
            uint32_t num_cols_per_core = 0;
            if (core_group_1.core_coord_in_core_ranges(core)) {
                num_cols_per_core = num_cols_per_core_group_1;
            } else if (core_group_2.core_coord_in_core_ranges(core)) {
                num_cols_per_core = num_cols_per_core_group_2;
            } else {
                TT_ASSERT(false, "Core not in specified core ranges");
            }
            tt_metal::SetRuntimeArgs(
                program,
                reader_kernel_id,
                core,
                {a.buffer()->address(),
                 num_cols_read / Wt * HtWt + num_cols_read % Wt,
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

    auto override_runtime_arguments_callback = [reader_kernel_id = reader_kernel_id,
                                                writer_kernel_id = writer_kernel_id,
                                                cb_src1 = cb_src1,
                                                cb_output = cb_output,
                                                cores = cores](
                                                   const void *operation,
                                                   Program &program,
                                                   const std::vector<Tensor> &input_tensors,
                                                   const std::vector<std::optional<const Tensor>> &,
                                                   const std::vector<Tensor> &output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        bool src_sharded = input_tensors.at(0).memory_config().is_sharded();
        bool out_sharded = output_tensors.at(0).memory_config().is_sharded();

        if (src_sharded && out_sharded) {
            UpdateDynamicCircularBufferAddress(program, cb_src1, *src_buffer);
            UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
        } else {
            auto &reader_runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
            auto &writer_runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);
            for (const auto &core : cores) {
                {
                    auto &runtime_args = reader_runtime_args_by_core[core.x][core.y];
                    runtime_args[0] = src_buffer->address();
                }

                {
                    auto &runtime_args = writer_runtime_args_by_core[core.x][core.y];
                    runtime_args[0] = dst_buffer->address();
                }
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace tt_metal

}  // namespace tt
