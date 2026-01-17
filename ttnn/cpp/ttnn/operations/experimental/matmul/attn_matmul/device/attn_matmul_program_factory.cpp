// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "attn_matmul_program_factory.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

AttnMatmulProgramFactory::cached_program_t AttnMatmulProgramFactory::create(
    const AttnMatmulParams& operation_attributes, const AttnMatmulInputs& tensor_args, Tensor& tensor_return_value) {
    tt::tt_metal::Program program{};

    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;
    auto& output = tensor_return_value;

    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::IDevice* device = a.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    tt::DataFormat in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat interm_data_format = fp32_dest_acc_en and in0_data_format == tt::DataFormat::Float32
                                            ? tt::DataFormat::Float32
                                            : tt::DataFormat::Float16_b;
    tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t in0_single_tile_size = tt::tile_size(in0_data_format);
    uint32_t in1_single_tile_size = tt::tile_size(in1_data_format);
    uint32_t interm_single_tile_size = tt::tile_size(interm_data_format);
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    if (in0_data_format == tt::DataFormat::Float32 or in1_data_format == tt::DataFormat::Float32 or
        output_data_format == tt::DataFormat::Float32) {
        TT_ASSERT(fp32_dest_acc_en == true, "when inputs/output are in fp32 format, fp32_dest_acc_en must be set");
    }

    log_debug(tt::LogOp, "math_fidelity: {}", math_fidelity);
    log_debug(tt::LogOp, "math_approx_mode: {}", math_approx_mode);
    log_debug(tt::LogOp, "fp32_dest_acc_en: {}", fp32_dest_acc_en);
    log_debug(tt::LogOp, "packer_l1_acc: {}", packer_l1_acc);
    log_debug(tt::LogOp, "in0_data_format: {}", in0_data_format);
    log_debug(tt::LogOp, "in1_data_format: {}", in1_data_format);
    log_debug(tt::LogOp, "interm_data_format: {}", interm_data_format);
    log_debug(tt::LogOp, "output_data_format: {}", output_data_format);

    tt::tt_metal::Buffer* src0_buffer = a.buffer();
    tt::tt_metal::Buffer* src1_buffer = b.buffer();

    // A block of work is one MtNt
    uint32_t num_cores_y = operation_attributes.compute_with_storage_grid_size.y;
    auto num_output_blocks_total = ashape[1];  // ashape[1] is Q num_heads; only parallelize on this
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_output_blocks_per_core_group_1,
         num_output_blocks_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(
                operation_attributes.compute_with_storage_grid_size, num_output_blocks_total);

    auto all_device_cores = CoreRange(
        {0, 0},
        {a.device()->compute_with_storage_grid_size().x - 1, a.device()->compute_with_storage_grid_size().y - 1});
    auto total_num_cores =
        a.device()->compute_with_storage_grid_size().x * a.device()->compute_with_storage_grid_size().y;

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // C = torch.matmul(A.transpose(0, 2) * B).transpose(0, 2)
    // MN = MK*KN
    // Note, in1 K may not be the same as in0 K. We will read up to in0 K from in1 K for matmul.
    const bool transpose_hw_bool = operation_attributes.transpose_hw.value_or(false);
    const uint32_t num_tokens_val =
        operation_attributes.num_tokens.value_or(0);  // should not be nullopt if transpose_hw=true
    constexpr uint32_t num_rows_in_one_tile = 32;

    uint32_t Mt = ashape[2] / TILE_HEIGHT;
    uint32_t Kt = ashape[3] / TILE_WIDTH;
    // For transpose_hw=true, in1_Kt is same as in0_Kt but on bshape[3]
    // For transpose_hw=false, in1_Kt is on bshape[2] but represents the max cache length to read from (ie. may not
    // equal in0_Kt)
    uint32_t in1_Kt = transpose_hw_bool ? Kt : bshape[2] / TILE_HEIGHT;
    uint32_t Nt = transpose_hw_bool ? num_tokens_val / TILE_HEIGHT : bshape[3] / TILE_WIDTH;
    uint32_t MtKt = Mt * Kt;
    uint32_t MtNt = Mt * Nt;
    // For transpose_hw=true, in1_Kt is max cache length
    // For transpose_hw=false, bshape[2] is max cache length
    uint32_t in1_KtNt_stride = transpose_hw_bool ? bshape[2] / TILE_HEIGHT * in1_Kt : in1_Kt * Nt;
    uint32_t in1_KtNt_skip = transpose_hw_bool ? (bshape[2] / TILE_HEIGHT - 1) * in1_Kt : (in1_Kt - Kt) * Nt;

    uint32_t src0_addr = src0_buffer->address();
    uint32_t src1_addr = src1_buffer->address();
    uint32_t dst_addr = dst_buffer->address();

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t cb0_num_input_tiles = Kt * 2;
    tt::tt_metal::CircularBufferConfig src0_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb0_num_input_tiles * in0_single_tile_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, src0_cb_config);

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    uint32_t cb1_num_input_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(
            cb1_num_input_tiles * in1_single_tile_size, {{src1_cb_index, in1_data_format}})
            .set_page_size(src1_cb_index, in1_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src1_config);

    uint32_t cb_intermed0_index = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig cb_interm0_config =
        tt::tt_metal::CircularBufferConfig(1 * interm_single_tile_size, {{cb_intermed0_index, interm_data_format}})
            .set_page_size(cb_intermed0_index, interm_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_interm0_config);

    uint32_t cb_intermed1_index = tt::CBIndex::c_3;
    tt::tt_metal::CircularBufferConfig cb_interm1_config =
        tt::tt_metal::CircularBufferConfig(1 * interm_single_tile_size, {{cb_intermed1_index, interm_data_format}})
            .set_page_size(cb_intermed1_index, interm_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_interm1_config);

    uint32_t cb_intermed2_index = tt::CBIndex::c_4;
    tt::tt_metal::CircularBufferConfig cb_interm2_config =
        tt::tt_metal::CircularBufferConfig(1 * interm_single_tile_size, {{cb_intermed2_index, interm_data_format}})
            .set_page_size(cb_intermed2_index, interm_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_interm2_config);

    uint32_t output_cb_index = tt::CBIndex::c_5;
    uint32_t num_output_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * output_single_tile_size, {{output_cb_index, output_data_format}})
            .set_page_size(output_cb_index, output_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_output_config);

    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t)transpose_hw_bool, (uint32_t)(fp32_dest_acc_en and in0_data_format == tt::DataFormat::Float32)};
    tt::tt_metal::TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    auto reader_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul/device/kernels/dataflow/"
        "reader_transformer_attn_matmul.cpp",
        all_device_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_device_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_args = {
        (uint32_t)transpose_hw_bool,  // transpose_hw for matmul_init
    };  // bmm compute kernel the B, Mt, Nt are just 3 for loops that technically act as 1 large loop, so only set Nt
        // for simplicity

    auto eltwise_binary_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul/device/kernels/compute/transformer_attn_matmul.cpp",
        all_device_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_args});

    uint32_t num_output_blocks_per_core;
    for (uint32_t i = 0, num_blocks_written = 0; i < total_num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        if (core_group_1.contains(core)) {
            num_output_blocks_per_core = num_output_blocks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_output_blocks_per_core = num_output_blocks_per_core_group_2;
        } else {
            tt::tt_metal::SetRuntimeArgs(program, reader_id, core, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
            tt::tt_metal::SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {0, 0, 0, 0});
            tt::tt_metal::SetRuntimeArgs(program, writer_id, core, {0, 0, 0});
            continue;
        }

        tt::tt_metal::SetRuntimeArgs(
            program,
            reader_id,
            core,
            {
                src0_addr,
                src1_addr,
                Mt,
                Kt,
                Nt,
                MtKt,
                in1_KtNt_skip,  // Skip to get next batch for in1 after reading in0 Kt
                in1_KtNt_stride *
                    num_rows_in_one_tile,  // itileB stride; skips 32 * KtNt in bshape[0] for one block of MtNt
                num_output_blocks_per_core,
                num_blocks_written * MtKt,  // itileA_start
                0,                          // itileB_start; always read in same in1 per core TODO: multi-cast
            });

        tt::tt_metal::SetRuntimeArgs(
            program,
            eltwise_binary_kernel_id,
            core,
            {
                1,                                  // B
                1,                                  // Mt
                Kt,                                 // Kt
                num_output_blocks_per_core * MtNt,  // Nt
            });

        tt::tt_metal::SetRuntimeArgs(
            program,
            writer_id,
            core,
            {
                dst_addr,
                num_output_blocks_per_core * MtNt,
                num_blocks_written * MtNt,
            });
        num_blocks_written += num_output_blocks_per_core;
    }

    return cached_program_t(
        std::move(program),
        shared_variables_t{
            .reader_id = reader_id,
            .writer_id = writer_id,
            .eltwise_binary_kernel_id = eltwise_binary_kernel_id,
            .total_num_cores = total_num_cores,
            .in0_single_tile_size = in0_single_tile_size,
            .cb_src0 = cb_src0,
            .src0_cb_index = src0_cb_index,
            .num_cores_y = num_cores_y});
}

void AttnMatmulProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const AttnMatmulParams& operation_attributes,
    const AttnMatmulInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    const auto& shared_vars = cached_program.shared_variables;

    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;
    auto& output = tensor_return_value;

    auto* src_dram_buffer_a = a.buffer();
    auto* src_dram_buffer_b = b.buffer();
    auto* dst_dram_buffer = output.buffer();

    auto ashape = a.padded_shape();
    auto bshape = b.padded_shape();

    // C = torch.matmul(A.transpose(0, 2) * B).transpose(0, 2)
    // MN = MK*KN
    // Note, in1 K may not be the same as in0 K. We will read up to in0 K from in1 K for matmul.
    const bool transpose_hw_bool = operation_attributes.transpose_hw.value_or(false);
    const uint32_t num_tokens_val =
        operation_attributes.num_tokens.value_or(0);  // should not be nullopt if transpose_hw=true
    constexpr uint32_t num_rows_in_one_tile = 32;

    uint32_t Mt = ashape[2] / TILE_HEIGHT;
    uint32_t Kt = ashape[3] / TILE_WIDTH;
    // For transpose_hw=true, in1_Kt is same as in0_Kt but on bshape[3]
    // For transpose_hw=false, in1_Kt is on bshape[2] but represents the max cache length to read from (ie. may
    // not equal in0_Kt)
    uint32_t in1_Kt = transpose_hw_bool ? Kt : bshape[2] / TILE_HEIGHT;
    uint32_t Nt = transpose_hw_bool ? num_tokens_val / TILE_HEIGHT : bshape[3] / TILE_WIDTH;
    uint32_t MtKt = Mt * Kt;
    uint32_t MtNt = Mt * Nt;
    // For transpose_hw=true, in1_Kt is max cache length
    // For transpose_hw=false, bshape[2] is max cache length
    uint32_t in1_KtNt_stride = transpose_hw_bool ? bshape[2] / TILE_HEIGHT * in1_Kt : in1_Kt * Nt;
    uint32_t in1_KtNt_skip = transpose_hw_bool ? (bshape[2] / TILE_HEIGHT - 1) * in1_Kt : (in1_Kt - Kt) * Nt;

    UpdateCircularBufferTotalSize(program, shared_vars.cb_src0, Kt * shared_vars.in0_single_tile_size);
    UpdateCircularBufferPageSize(
        program, shared_vars.cb_src0, shared_vars.src0_cb_index, shared_vars.in0_single_tile_size);

    auto num_output_blocks_total = ashape[1];  // ashape[1] is Q num_heads; only parallelize on this
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_output_blocks_per_core_group_1,
         num_output_blocks_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(
                operation_attributes.compute_with_storage_grid_size, num_output_blocks_total);

    uint32_t num_output_blocks_per_core;
    for (uint32_t i = 0, num_blocks_written = 0; i < shared_vars.total_num_cores; i++) {
        CoreCoord core = {i / shared_vars.num_cores_y, i % shared_vars.num_cores_y};

        if (core_group_1.contains(core)) {
            num_output_blocks_per_core = num_output_blocks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_output_blocks_per_core = num_output_blocks_per_core_group_2;
        } else {
            tt::tt_metal::SetRuntimeArgs(program, shared_vars.reader_id, core, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
            tt::tt_metal::SetRuntimeArgs(program, shared_vars.eltwise_binary_kernel_id, core, {0, 0, 0, 0});
            tt::tt_metal::SetRuntimeArgs(program, shared_vars.writer_id, core, {0, 0, 0});
            continue;
        }

        tt::tt_metal::SetRuntimeArgs(
            program,
            shared_vars.reader_id,
            core,
            {
                src_dram_buffer_a->address(),
                src_dram_buffer_b->address(),
                Mt,
                Kt,
                Nt,
                MtKt,
                in1_KtNt_skip,  // Skip to get next batch for in1 after reading in0 Kt
                in1_KtNt_stride *
                    num_rows_in_one_tile,  // itileB stride; skips 32 * KtNt in bshape[0] for one block of MtNt
                num_output_blocks_per_core,
                num_blocks_written * MtKt,  // itileA_start
                0,                          // itileB_start; always read in same in1 per core TODO: multi-cast
            });

        tt::tt_metal::SetRuntimeArgs(
            program,
            shared_vars.eltwise_binary_kernel_id,
            core,
            {
                1,                                  // B
                1,                                  // Mt
                Kt,                                 // Kt
                num_output_blocks_per_core * MtNt,  // Nt
            });

        tt::tt_metal::SetRuntimeArgs(
            program,
            shared_vars.writer_id,
            core,
            {
                dst_dram_buffer->address(),
                num_output_blocks_per_core * MtNt,
                num_blocks_written * MtNt,
            });
        num_blocks_written += num_output_blocks_per_core;
    }
}

}  // namespace ttnn::experimental::prim
