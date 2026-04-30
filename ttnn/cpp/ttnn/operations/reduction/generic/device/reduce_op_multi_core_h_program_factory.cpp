// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_op_multi_core_h_program_factory.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <bit>
#include <cmath>
#include <numeric>

namespace ttnn::prim {

ReduceMultiCoreHProgramFactory::cached_program_t ReduceMultiCoreHProgramFactory::create(
    const ReduceParams& operation_attributes, const Tensor& tensor_args, Tensor& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    const auto& a = tensor_args;
    auto& output = tensor_return_value;
    const auto& shape = a.padded_shape();
    uint32_t W = shape[3], H = shape[2], NC = shape[1] * shape[0];
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();
    const uint32_t tile_hw = a.tensor_spec().tile().get_tile_hw();

    uint32_t Wt = W / tile_width;
    uint32_t Ht = H / tile_height;
    uint32_t HtWt = Ht * Wt;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), operation_attributes.compute_kernel_config);

    tt_metal::Program program = tt_metal::CreateProgram();

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    tt::DataFormat scaler_cb_data_format =
        src0_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t scaler_single_tile_size = tt::tile_size(scaler_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    tt_metal::IDevice* device = a.device();

    bool use_width_sharding = a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
                              output.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;

    uint32_t chunk_size = use_width_sharding ? 1 : ttnn::get_dest_reg_count(operation_attributes.compute_kernel_config);

    // For min/max with non-unity scalar, the GMPOOL hardware path only respects the scaler's
    // exponent, so the device reduces with scaler=1.0 and the user scalar is applied after the
    // reduction via SFPU mul_unary_tile inside the compute kernel.
    const bool use_post_mul = operation_attributes.post_mul_scaler != 1.0f;

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
        num_cols_per_core_group_1 = NC * (a.shard_spec().value().shape[1] / tile_width);
        num_cols_per_core_group_2 = 0;
    }

    uint32_t src0_cb_index = CBIndex::c_0;
    uint32_t src1_cb_index = CBIndex::c_1;
    CBHandle cb_src1 = 0;
    if (use_width_sharding) {
        uint32_t num_shard_tiles = a.shard_spec().value().numel() / tile_hw;
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
        uint32_t num_input_tiles = operation_attributes.negate ? chunk_size : 2;
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
        uint32_t num_output_tiles = output.shard_spec().value().numel() / tile_hw;
        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
                .set_page_size(output_cb_index, dst_single_tile_size)
                .set_globally_allocated_address(*output.buffer());
        cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);
    } else {
        uint32_t num_output_tiles = operation_attributes.negate ? chunk_size : 2;
        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
                .set_page_size(output_cb_index, dst_single_tile_size);
        cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);
    }
    tt_metal::Buffer* src0_buffer = a.buffer();
    tt_metal::KernelHandle reader_kernel_id;
    uint32_t scaler_bits = std::bit_cast<uint32_t>(operation_attributes.scaler);
    // Packed fp32 scalar passed to the compute kernel for mul_unary_tile post-reduction scaling.
    uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);

    if (operation_attributes.negate) {
        // The reduce_h_neg kernel pushes ntiles tiles per inner-loop iteration
        // via push_back(ntiles).  The CB FIFO write pointer only wraps when it
        // exactly reaches fifo_limit, so the CB size must be a multiple of
        // every push size that occurs.
        //
        // For a core with Wt_per_core columns and row_chunk == chunk_size:
        //   - Wt_per_core >= chunk_size: push sizes are chunk_size (full
        //     chunks) and Wt_per_core % chunk_size (partial last chunk).
        //   - Wt_per_core < chunk_size:  the only push size is Wt_per_core
        //     (no full-sized chunk ever occurs).
        //
        // Compute the LCM of only the push sizes that actually occur across
        // both core groups to avoid unnecessarily inflating the CB allocation.
        uint32_t negate_cb_tiles = 1;
        auto include_push_sizes = [&](uint32_t cols_per_core) {
            if (cols_per_core == 0) {
                return;
            }
            if (cols_per_core >= chunk_size) {
                negate_cb_tiles = std::lcm(negate_cb_tiles, chunk_size);
                uint32_t partial = cols_per_core % chunk_size;
                if (partial > 0) {
                    negate_cb_tiles = std::lcm(negate_cb_tiles, partial);
                }
            } else {
                negate_cb_tiles = std::lcm(negate_cb_tiles, cols_per_core);
            }
        };
        include_push_sizes(num_cols_per_core_group_1);
        if (num_cols_per_core_group_2 > 0) {
            include_push_sizes(num_cols_per_core_group_2);
        }

        uint32_t acc_cb_index = CBIndex::c_4;
        tt_metal::CircularBufferConfig cb_acc_config =
            tt_metal::CircularBufferConfig(negate_cb_tiles * dst_single_tile_size, {{acc_cb_index, dst_cb_data_format}})
                .set_page_size(acc_cb_index, dst_single_tile_size);
        tt_metal::CreateCircularBuffer(program, all_cores, cb_acc_config);

        uint32_t ineg_cb_index = CBIndex::c_5;
        tt_metal::CircularBufferConfig cb_ineg_config =
            tt_metal::CircularBufferConfig(
                negate_cb_tiles * dst_single_tile_size, {{ineg_cb_index, dst_cb_data_format}})
                .set_page_size(ineg_cb_index, dst_single_tile_size);
        tt_metal::CreateCircularBuffer(program, all_cores, cb_ineg_config);
    }

    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils::get_defines(operation_attributes.math_op, tt::tt_metal::ReduceOpDim::H);
    if (use_post_mul) {
        reduce_defines["REDUCE_POST_MUL"] = "1";
    }

    if (use_width_sharding) {
        std::vector<uint32_t> reader_compile_time_args = {src0_cb_index, src1_cb_index, scaler_cb_index, scaler_bits};
        std::map<std::string, std::string> reader_defines;
        reader_defines["REDUCE_SCALER"] = "1";
        // Pass DEST config so reader can compute DEST_AUTO_LIMIT
        reader_defines["ENABLE_FP32_DEST_ACC"] = fp32_dest_acc_en ? "1" : "0";
        reader_defines["DST_SYNC_FULL"] = dst_full_sync_en ? "1" : "0";
        reader_defines.insert(reduce_defines.begin(), reduce_defines.end());
        reader_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp",
            all_cores,
            tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));
    } else {
        std::vector<uint32_t> reader_compile_time_args = {Ht, Wt, HtWt, scaler_bits, /*use_welford=*/0};
        TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);

        // Pass DEST config so reader can compute DEST_AUTO_LIMIT
        std::map<std::string, std::string> reader_defines;
        reader_defines["ENABLE_FP32_DEST_ACC"] = fp32_dest_acc_en ? "1" : "0";
        reader_defines["DST_SYNC_FULL"] = dst_full_sync_en ? "1" : "0";
        reader_defines.insert(reduce_defines.begin(), reduce_defines.end());

        reader_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_transpose_wh_universal_input_cols_partitioned.cpp",
            all_cores,
            tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));
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
    // For width-sharding, num_cols_per_core_group_1 == NC * shard_Wt. Expose (shard_Wt, NC)
    // to the compute kernel so its (nc, wt_chunk, ht, wt_in_chunk) iteration matches the
    // reader's per-batch tile layout.
    uint32_t compute_Wt = use_width_sharding ? (num_cols_per_core_group_1 / NC) : num_cols_per_core_group_1;
    uint32_t compute_NC = use_width_sharding ? NC : 1;
    std::vector<uint32_t> compute_kernel_args_group_1 = {
        Ht,                    // Ht
        compute_Wt,            // Wt
        compute_NC,            // NC
        post_mul_scaler_bits,  // packed fp32 user scalar (only used if REDUCE_POST_MUL is set)
    };

    const std::string compute_kernel =
        std::string("ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce") +
        (operation_attributes.negate ? "_h_neg" : "") + ".cpp";

    tt_metal::CreateKernel(
        program,
        compute_kernel,
        core_group_1,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .compile_args = compute_kernel_args_group_1,
            .defines = reduce_defines});

    if (!core_group_2.ranges().empty()) {
        uint32_t compute_Wt_group_2 = use_width_sharding ? (num_cols_per_core_group_2 / NC) : num_cols_per_core_group_2;
        uint32_t compute_NC_group_2 = use_width_sharding ? NC : 1;
        std::vector<uint32_t> compute_kernel_args_group_2 = {
            Ht,                    // Ht
            compute_Wt_group_2,    // Wt
            compute_NC_group_2,    // NC
            post_mul_scaler_bits,  // packed fp32 user scalar (only used if REDUCE_POST_MUL is set)
        };

        tt_metal::CreateKernel(
            program,
            compute_kernel,
            core_group_2,
            tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .dst_full_sync_en = dst_full_sync_en,
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
        TT_FATAL(NC != 0, "Batch size NC must be non-zero (shape[0]={}, shape[1]={})", shape[0], shape[1]);
        uint32_t shard_Wt = num_cols_per_core_group_1 / NC;
        uint32_t shard_row_size = shard_Wt * src0_single_tile_size;
        uint32_t shard_batch_size = shard_row_size * Ht;
        std::vector<uint32_t> reader_rt_args = {
            num_cols_per_core_group_1 * Ht, shard_Wt, Ht, NC, shard_row_size, shard_batch_size};
        tt_metal::SetRuntimeArgs(program, reader_kernel_id, all_cores, reader_rt_args);

        std::vector<uint32_t> writer_rt_args = {num_cols_per_core_group_1};
        tt_metal::SetRuntimeArgs(program, writer_kernel_id, all_cores, writer_rt_args);
    } else {
        TT_FATAL(Wt != 0, "Width in tiles (Wt) must be non-zero (W={}, tile_width={})", W, tile_width);
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
    const ReduceParams& /*operation_attributes*/,
    const Tensor& tensor_args,
    Tensor& tensor_return_value) {
    auto* src_buffer = tensor_args.buffer();
    auto* dst_buffer = tensor_return_value.buffer();

    bool use_width_sharding = tensor_args.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
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

}  // namespace ttnn::prim
