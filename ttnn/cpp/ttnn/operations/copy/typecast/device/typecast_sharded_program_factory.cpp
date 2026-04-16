// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast_sharded_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>

namespace ttnn::prim {

using namespace tt::constants;
using namespace tt::tt_metal;

tt::tt_metal::ProgramDescriptor TypecastShardedProgramFactory::create_descriptor(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;
    const auto& input_dtype = args.input_dtype;
    const auto& output_dtype = args.output_dtype;

    const auto shard_spec = input.shard_spec().value();
    const auto all_cores = shard_spec.grid;

    auto out_shard_spec = output.shard_spec().value();
    TT_FATAL(
        out_shard_spec.num_cores() == shard_spec.num_cores(),
        "Output tensor should have same number of cores {} as input tensor {}",
        out_shard_spec.num_cores(),
        shard_spec.num_cores());

    const tt::DataFormat act_df = datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat out_df = datatype_to_dataformat_converter(output.dtype());

    const uint32_t input_tile_size = tt::tile_size(act_df);
    const uint32_t output_tile_size = tt::tile_size(out_df);
    TT_FATAL(input_tile_size == output_tile_size, "Input and output tile size should be same");

    uint32_t num_tile_per_core = 0;
    if (input.dtype() == DataType::BFLOAT8_B || input.dtype() == DataType::BFLOAT4_B) {
        const uint32_t ntiles_along_width = std::ceil(shard_spec.shape[1] / (float)TILE_WIDTH);
        const uint32_t ntiles_along_height = std::ceil(shard_spec.shape[0] / (float)TILE_HEIGHT);
        num_tile_per_core = ntiles_along_width * ntiles_along_height;
    } else {
        TT_FATAL(
            (shard_spec.shape[1] * tt::datum_size(act_df)) % hal::get_l1_alignment() == 0,
            "Shard width should be multiple of {} to satisfy L1 alignment",
            hal::get_l1_alignment());
        const size_t shard_size_in_bytes = shard_spec.shape[0] * shard_spec.shape[1] * tt::datum_size(act_df);
        TT_FATAL(shard_size_in_bytes % input_tile_size == 0, "Shard Size must be multiple of input_tile_size");
        num_tile_per_core = (shard_size_in_bytes + input_tile_size - 1) / input_tile_size;
    }

    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();
    TT_FATAL(src_buffer->buffer_type() != BufferType::DRAM, "Input buffer should be in L1");
    TT_FATAL(dst_buffer->buffer_type() != BufferType::DRAM, "Output buffer should be in L1");

    const uint32_t aligned_input_tile_nbytes = round_up_to_mul32(input_tile_size);
    const uint32_t aligned_output_tile_nbytes = round_up_to_mul32(output_tile_size);

    ProgramDescriptor program_descriptor;

    // ── Circular Buffers (sharded — globally allocated) ──────────────────
    constexpr uint32_t in_cb_id = tt::CBIndex::c_0;
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = aligned_input_tile_nbytes * num_tile_per_core;
        cb_desc.core_ranges = all_cores;
        cb_desc.buffer = src_buffer;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(in_cb_id),
            .data_format = act_df,
            .page_size = aligned_input_tile_nbytes});
        program_descriptor.cbs.push_back(std::move(cb_desc));
    }

    constexpr uint32_t out_cb_id = tt::CBIndex::c_2;
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = aligned_output_tile_nbytes * num_tile_per_core;
        cb_desc.core_ranges = all_cores;
        cb_desc.buffer = dst_buffer;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(out_cb_id),
            .data_format = out_df,
            .page_size = aligned_output_tile_nbytes});
        program_descriptor.cbs.push_back(std::move(cb_desc));
    }

    // ── Reader kernel (sharded) ──────────────────────────────────────────
    {
        KernelDescriptor kernel_desc;
        kernel_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp";
        kernel_desc.core_ranges = all_cores;
        kernel_desc.compile_time_args = {in_cb_id};
        kernel_desc.common_runtime_args = {num_tile_per_core};
        kernel_desc.config = ReaderConfigDescriptor{};
        program_descriptor.kernels.push_back(std::move(kernel_desc));
    }

    // ── Compute kernel ───────────────────────────────────────────────────
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (args.preserve_fp32_precision) {
        unpack_to_dest_mode[in_cb_id] = UnpackToDestMode::UnpackToDestFp32;
    }

    {
        KernelDescriptor kernel_desc;
        kernel_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp";
        kernel_desc.core_ranges = all_cores;
        kernel_desc.compile_time_args = {1, num_tile_per_core, in_cb_id, out_cb_id};
        kernel_desc.defines = {
            {"TYPECAST_LLK_INIT",
             fmt::format(
                 "typecast_tile_init<{0}u, {1}u>",
                 static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
                 static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype)))},
            {"TYPECAST_LLK",
             fmt::format(
                 "typecast_tile<{0}u, {1}u>",
                 static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
                 static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype)))}};
        kernel_desc.config = ComputeConfigDescriptor{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = args.fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .bfp8_pack_precise = args.bfp8_pack_precise,
            .math_approx_mode = false};
        program_descriptor.kernels.push_back(std::move(kernel_desc));
    }

    return program_descriptor;
}

TypecastShardedProgramFactory::cached_program_t TypecastShardedProgramFactory::create(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    auto descriptor = create_descriptor(args, tensor_args, output);
    Program program{descriptor};

    // CB handles are assigned sequentially: in_cb=0, out_cb=1
    constexpr CBHandle cb_src0 = 0;
    constexpr CBHandle out_cb = 1;

    return cached_program_t{std::move(program), {cb_src0, out_cb}};
}

void TypecastShardedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TypecastParams& /*operation_attributes*/,
    const TypecastInputs& tensor_args,
    Tensor& output) {
    auto& program = cached_program.program;
    const auto& cb_src0 = cached_program.shared_variables.cb_src0;
    const auto& out_cb = cached_program.shared_variables.out_cb;

    auto* src_buffer = tensor_args.input.buffer();
    auto* dst_buffer = output.buffer();
    UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
    UpdateDynamicCircularBufferAddress(program, out_cb, *dst_buffer);
}

}  // namespace ttnn::prim
