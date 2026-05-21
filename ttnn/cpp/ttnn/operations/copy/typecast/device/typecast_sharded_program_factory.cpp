// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast_sharded_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

using namespace tt::constants;
using namespace tt::tt_metal;

ProgramDescriptor TypecastShardedProgramFactory::create_descriptor(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto& input_dtype = args.input_dtype;
    const auto& output_dtype = args.output_dtype;

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

    // For TILE layout, input_tile_size != output_tile_size is supported (e.g., BFLOAT8_B <-> BFLOAT16).
    // The number of tiles stays the same; only the bytes per tile changes.
    if (input_tile_size != output_tile_size) {
        TT_FATAL(
            (input.layout() == Layout::TILE && output.layout() == Layout::TILE),
            "TypecastShardedProgramFactory requires TILE layout when input and output tile sizes differ "
            "(input_tile_size={}, output_tile_size={}).",
            input_tile_size,
            output_tile_size);
    }

    uint32_t num_tile_per_core = 0;

    // Use dimension-based tile count if either input or output is block format
    bool is_block_format =
        (input.dtype() == DataType::BFLOAT8_B || input.dtype() == DataType::BFLOAT4_B ||
         output.dtype() == DataType::BFLOAT8_B || output.dtype() == DataType::BFLOAT4_B);

    if (is_block_format) {
        // For block formats, calculate tile count based on element dimensions
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

    // output sharded CB
    uint32_t out_cb_id = tt::CBIndex::c_2;
    uint32_t aligned_output_tile_nbytes =
        round_up_to_mul32(output_tile_size);  // will have issue if the page is not multiple of 32
    uint32_t out_cb_pagesize = aligned_output_tile_nbytes;
    uint32_t out_cb_npages = num_tile_per_core * buffering_factor;

    log_debug(tt::LogOp, "input_cb: {}, npages: {}, pagesize: {}", in_cb_id, in_cb_npages, in_cb_pagesize);
    log_debug(tt::LogOp, "out_cb_id: {}, npages: {}, pagesize: {}", out_cb_id, out_cb_npages, out_cb_pagesize);
    log_debug(tt::LogOp, "input_tile_size: {}, output_tile_size: {}", input_tile_size, output_tile_size);
    log_debug(
        tt::LogOp,
        "input_dtype: {}, output_dtype: {}",
        static_cast<uint32_t>(input_dtype),
        static_cast<uint32_t>(output_dtype));
    log_debug(tt::LogOp, "act_df: {}, out_df: {}", static_cast<uint32_t>(act_df), static_cast<uint32_t>(out_df));
    log_debug(
        tt::LogOp,
        "num_tile_per_core: {}, shard_shape: [{}, {}]",
        num_tile_per_core,
        shard_spec.shape[0],
        shard_spec.shape[1]);
    log_debug(
        tt::LogOp,
        "preserve_fp32_precision: {}, fp32_dest_acc_en: {}",
        args.preserve_fp32_precision,
        args.fp32_dest_acc_en);

    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    TT_FATAL(src_is_dram == 0, "Input buffer should be in L1");

    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    TT_FATAL(dst_is_dram == 0, "Output buffer should be in L1");

    ProgramDescriptor desc;

    // Dynamic (globally-allocated) CBs: setting `.buffer` makes the framework patch
    // the CB base address from the live tensor buffer on every dispatch — this is
    // the descriptor equivalent of the old UpdateDynamicCircularBufferAddress
    // calls that lived in override_runtime_arguments().
    desc.cbs.push_back(CBDescriptor{
        .total_size = in_cb_pagesize * in_cb_npages,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(in_cb_id),
            .data_format = act_df,
            .page_size = in_cb_pagesize,
        }}},
        .buffer = src_buffer,
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = out_cb_pagesize * out_cb_npages,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(out_cb_id),
            .data_format = out_df,
            .page_size = out_cb_pagesize,
        }}},
        .buffer = dst_buffer,
    });

    std::vector<uint32_t> reader_compile_time_args = {
        in_cb_id,
    };

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> compute_kernel_args_group_1 = {
        1,                  // per_core_block_cnt
        num_tile_per_core,  // per_core_block_size
        in_cb_id,
        out_cb_id};

    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (args.preserve_fp32_precision) {
        unpack_to_dest_mode[in_cb_id] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }

    bool math_approx_mode = false;

    // KernelDescriptor::Defines is a vector<pair> rather than a std::map; key
    // order does not matter for `#define` substitution.
    KernelDescriptor::Defines unary_defines;
    unary_defines.emplace_back(
        "TYPECAST_LLK_INIT",
        fmt::format(
            "typecast_tile_init<{0}u, {1}u>",
            static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
            static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype))));
    unary_defines.emplace_back(
        "TYPECAST_LLK",
        fmt::format(
            "typecast_tile<{0}u, {1}u>",
            static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
            static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype))));

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = std::move(compute_kernel_args_group_1);
    compute_desc.defines = std::move(unary_defines);
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .fp32_dest_acc_en = args.fp32_dest_acc_en,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
        .bfp8_pack_precise = args.bfp8_pack_precise,
        .math_approx_mode = math_approx_mode,
    };

    // The legacy SetRuntimeArgs(program, kernel, all_cores, {num_tile_per_core})
    // call broadcasts a single uint32 value to every core in the CoreRangeSet —
    // the kernel reads it via get_arg_val<uint32_t>(0). The descriptor pattern
    // does not have a CoreRangeSet broadcast form, so enumerate the cores
    // (the same way the framework would internally) and push the args per-core.
    const auto cores = corerange_to_cores(all_cores);
    for (const auto& core : cores) {
        reader_desc.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{static_cast<uint32_t>(num_tile_per_core)});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
