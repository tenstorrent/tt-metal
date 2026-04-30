// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_op_multi_core_w_rm_program_factory.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <bit>
#include <cmath>

namespace ttnn::prim {

ReduceMultiCoreWRmProgramFactory::cached_program_t ReduceMultiCoreWRmProgramFactory::create(
    const ReduceParams& operation_attributes, const Tensor& tensor_args, Tensor& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    const auto& a = tensor_args;
    auto& output = tensor_return_value;
    const auto& shape = a.logical_shape();
    uint32_t W = shape[3], H = shape[2], NC = shape[1] * shape[0];
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();

    uint32_t Wt = (W + tile_width - 1) / tile_width;
    uint32_t Ht = (H + tile_height - 1) / tile_height;
    (void)Ht;  // each reader page is a full (N,C,h) row; compute uses a 1×W face per page (Ht_tilize=1 per row)

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), operation_attributes.compute_kernel_config);

    tt_metal::Program program = tt_metal::CreateProgram();

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    uint32_t datum_size = tt::datum_size(src0_cb_data_format);

    tt::DataFormat scaler_cb_data_format =
        src0_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t scaler_single_tile_size = tt::tile_size(scaler_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    tt_metal::IDevice* device = a.device();
    const uint32_t rm_page_size = a.buffer()->page_size();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_rows = NC * H;
    uint32_t num_cores;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_rows_per_core_group_1, num_rows_per_core_group_2;
    if (operation_attributes.sub_core_grids.has_value()) {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(*operation_attributes.sub_core_grids, num_rows);
    } else {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows);
    }

    // CB 24: row-major input (one page = one tensor row in W)
    constexpr uint32_t cb_rm = tt::CBIndex::c_24;
    uint32_t num_rm_pages = 2;
    tt_metal::CircularBufferConfig cb_rm_config =
        tt_metal::CircularBufferConfig(num_rm_pages * rm_page_size, {{cb_rm, src0_cb_data_format}})
            .set_page_size(cb_rm, rm_page_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_rm_config);

    // CB 0,2,3: tile path (tilize output + reduce)
    uint32_t num_input_tiles = std::max(2U, Wt);
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(num_input_tiles * src0_single_tile_size, {{0, src0_cb_data_format}})
            .set_page_size(0, src0_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    tt_metal::CircularBufferConfig cb_scaler_config =
        tt_metal::CircularBufferConfig(scaler_single_tile_size, {{CBIndex::c_2, scaler_cb_data_format}})
            .set_page_size(CBIndex::c_2, scaler_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_scaler_config);

    uint32_t num_output_tiles = 2;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(num_output_tiles * dst_single_tile_size, {{3, dst_cb_data_format}})
            .set_page_size(3, dst_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    const bool use_post_mul = operation_attributes.post_mul_scaler != 1.0f;
    uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);

    tt_metal::Buffer* src_buffer = a.buffer();
    std::vector<uint32_t> reader_compile_time_args = {std::bit_cast<uint32_t>(operation_attributes.scaler)};
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    tt_metal::Buffer* dst_buffer = output.buffer();
    std::vector<uint32_t> writer_compile_time_args = {datum_size};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils::get_defines(operation_attributes.math_op, ReduceOpDim::W);
    if (use_post_mul) {
        reduce_defines["REDUCE_POST_MUL"] = "1";
    }

    tt_metal::KernelHandle reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_reduce_rm_w.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reduce_defines));

    tt_metal::KernelHandle writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/writer_reduce_w_rm_scalar.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args, reduce_defines));

    std::vector<uint32_t> compute_args_g1 = {
        num_rows_per_core_group_1,
        Wt,
        1,  // NC (unused; matches classic reduce signature)
        post_mul_scaler_bits,
    };

    const std::string compute_kernel =
        "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_rm_w.cpp";

    tt_metal::CreateKernel(
        program,
        compute_kernel,
        core_group_1,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = compute_args_g1,
            .defines = reduce_defines});

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_args_g2 = {
            num_rows_per_core_group_2,
            Wt,
            1,
            post_mul_scaler_bits,
        };
        tt_metal::CreateKernel(
            program,
            compute_kernel,
            core_group_2,
            tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .compile_args = compute_args_g2,
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

    for (uint32_t i = 0, num_rows_read = 0; i < num_cores; i++) {
        const CoreCoord& core = cores[i];
        uint32_t num_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }
        tt_metal::SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {
                a.buffer()->address(),
                num_rows_per_core,
                num_rows_read,
            });

        uint32_t out_dim_divider = 1;  // one output page (scalar) per input row
        tt_metal::SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {
                output.buffer()->address(),
                num_rows_per_core / out_dim_divider,
                num_rows_read / out_dim_divider,
            });
        num_rows_read += num_rows_per_core;
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, std::move(cores)}};
}

void ReduceMultiCoreWRmProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ReduceParams& /*operation_attributes*/,
    const Tensor& tensor_args,
    Tensor& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    auto* src_dram_buffer = tensor_args.buffer();
    auto* dst_dram_buffer = tensor_return_value.buffer();

    auto& reader_runtime_args_by_core =
        GetRuntimeArgs(cached_program.program, cached_program.shared_variables.reader_kernel_id);
    auto& writer_runtime_args_by_core =
        GetRuntimeArgs(cached_program.program, cached_program.shared_variables.writer_kernel_id);
    for (const auto& core : cached_program.shared_variables.cores) {
        {
            auto& runtime_args = reader_runtime_args_by_core[core.x][core.y];
            runtime_args[0] = src_dram_buffer->address();
        }

        {
            auto& runtime_args = writer_runtime_args_by_core[core.x][core.y];
            runtime_args[0] = dst_dram_buffer->address();
        }
    }
}

}  // namespace ttnn::prim
