// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast_sharded_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>

namespace ttnn::prim {

using namespace tt::constants;
using namespace tt::tt_metal;

TypecastShardedProgramFactory::cached_program_t TypecastShardedProgramFactory::create(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto& input_dtype = args.input_dtype;
    const auto& output_dtype = args.output_dtype;

    tt::tt_metal::Program program = CreateProgram();

    auto shard_spec = input.shard_spec().value();
    auto all_cores = shard_spec.grid;
    uint32_t ncores = shard_spec.num_cores();

    auto out_shard_spec = output.shard_spec().value();
    TT_FATAL(
        out_shard_spec.num_cores() == ncores,
        "Output tensor should have same number of cores {} as input tensor {}",
        out_shard_spec.num_cores(),
        ncores);

    tt::DataFormat act_df = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    uint32_t input_tile_size = tt::tile_size(act_df);
    uint32_t output_tile_size = tt::tile_size(out_df);

    TT_FATAL(input_tile_size == output_tile_size, "Input and output tile size should be same");

    uint32_t num_tile_per_core = 0;

    if (input.dtype() == DataType::BFLOAT8_B || input.dtype() == DataType::BFLOAT4_B) {
        uint32_t ntiles_along_width = std::ceil(shard_spec.shape[1] / (float)tt::constants::TILE_WIDTH);
        uint32_t ntiles_along_height = std::ceil(shard_spec.shape[0] / (float)tt::constants::TILE_HEIGHT);
        num_tile_per_core = ntiles_along_width * ntiles_along_height;
    } else {
        TT_FATAL(
            (shard_spec.shape[1] * datum_size(act_df)) % hal::get_l1_alignment() == 0,
            "Shard width should be multiple of {} to satisfy L1 alignment",
            hal::get_l1_alignment());
        size_t shard_height = shard_spec.shape[0];
        size_t shard_width = shard_spec.shape[1];
        size_t shard_size_in_bytes = shard_height * shard_width * datum_size(act_df);
        TT_FATAL(shard_size_in_bytes % input_tile_size == 0, "Shard Size must be multiple of input_tile_size");
        num_tile_per_core = (shard_size_in_bytes + input_tile_size - 1) / input_tile_size;  // ceil value
    }

    uint32_t in_cb_id = tt::CBIndex::c_0;
    uint32_t buffering_factor = 1;  // data is already fully buffered in the CBs since its sharded
    uint32_t aligned_input_tile_nbytes =
        round_up_to_mul32(input_tile_size);  // will have issue if the page is not multiple of 32
    uint32_t in_cb_pagesize = aligned_input_tile_nbytes;
    uint32_t in_cb_npages = num_tile_per_core * buffering_factor;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(in_cb_pagesize * in_cb_npages, {{in_cb_id, act_df}})
            .set_page_size(in_cb_id, in_cb_pagesize)
            .set_globally_allocated_address(*input.buffer());
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    // output sharded CB
    uint32_t out_cb_id = tt::CBIndex::c_2;
    uint32_t aligned_output_tile_nbytes =
        round_up_to_mul32(output_tile_size);  // will have issue if the page is not multiple of 32
    uint32_t out_cb_pagesize = aligned_output_tile_nbytes;
    uint32_t out_cb_npages = num_tile_per_core * buffering_factor;
    tt::tt_metal::CircularBufferConfig out_cb_config =
        tt::tt_metal::CircularBufferConfig(out_cb_pagesize * out_cb_npages, {{out_cb_id, out_df}})
            .set_page_size(out_cb_id, out_cb_pagesize)
            .set_globally_allocated_address(*output.buffer());
    auto out_cb = tt::tt_metal::CreateCircularBuffer(program, all_cores, out_cb_config);

    log_debug(tt::LogOp, "input_cb: {}, npages: {}, pagesize: {}", in_cb_id, in_cb_npages, in_cb_pagesize);
    log_debug(tt::LogOp, "out_cb_id: {}, npages: {}, pagesize: {}", out_cb_id, out_cb_npages, out_cb_pagesize);
    log_debug(tt::LogOp, "input_tile_size: {}", input_tile_size);
    log_debug(tt::LogOp, "output_tile_size: {}", output_tile_size);

    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    TT_FATAL(src_is_dram == 0, "Input buffer should be in L1");
    std::vector<uint32_t> reader_compile_time_args = {
        in_cb_id,
    };

    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    TT_FATAL(dst_is_dram == 0, "Output buffer should be in L1");

    std::map<std::string, std::string> kernel_defines;
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, kernel_defines));

    std::vector<uint32_t> compute_kernel_args_group_1 = {
        1,                  // per_core_block_cnt
        num_tile_per_core,  // per_core_block_size
        in_cb_id,
        out_cb_id};

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (args.preserve_fp32_precision) {
        unpack_to_dest_mode[in_cb_id] = UnpackToDestMode::UnpackToDestFp32;
    }

    bool math_approx_mode = false;

    std::map<std::string, std::string> unary_defines;
    unary_defines["TYPECAST_LLK_INIT"] = fmt::format(
        "typecast_tile_init<{0}u, {1}u>",
        static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
        static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype)));
    unary_defines["TYPECAST_LLK"] = fmt::format(
        "typecast_tile<{0}u, {1}u>",
        static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
        static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype)));

    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = args.fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .bfp8_pack_precise = args.bfp8_pack_precise,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args_group_1,
            .defines = unary_defines});

    tt::tt_metal::SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        all_cores,
        {
            static_cast<uint32_t>(num_tile_per_core),
        });

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
    tt::tt_metal::UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
    tt::tt_metal::UpdateDynamicCircularBufferAddress(program, out_cb, *dst_buffer);
}

}  // namespace ttnn::prim
