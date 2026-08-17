// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast_sharded_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tilize_utils.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim {

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts TypecastShardedProgramFactory::create_program_artifacts(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto& input_dtype = args.input_dtype;
    const auto& output_dtype = args.output_dtype;

    auto shard_spec = input.shard_spec().value();
    auto all_cores = shard_spec.grid;
    uint32_t ncores = shard_spec.num_cores();

    const auto* device = input.device();

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

    // ---- Resource names ----
    const DFBSpecName IN_DFB{"in"};    // legacy CBIndex::c_0
    const DFBSpecName OUT_DFB{"out"};  // legacy CBIndex::c_2
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName COMPUTE{"compute"};

    // Input DFB: built on the input shard's own L1 memory (borrowed), so the data is already resident.
    uint32_t buffering_factor = 1;  // data is already fully buffered in the DFBs since its sharded
    uint32_t aligned_input_tile_nbytes =
        round_up_to_mul32(input_tile_size);  // will have issue if the page is not multiple of 32
    uint32_t in_cb_pagesize = aligned_input_tile_nbytes;
    uint32_t in_cb_npages = num_tile_per_core * buffering_factor;
    const DataflowBufferSpec in_dfb{
        .unique_id = IN_DFB,
        .entry_size = in_cb_pagesize,
        .num_entries = in_cb_npages,
        .data_format_metadata = act_df,
        .borrowed_from = INPUT,
    };

    // output sharded DFB: borrowed from the output shard buffer — the result is left resident there
    uint32_t aligned_output_tile_nbytes =
        round_up_to_mul32(output_tile_size);  // will have issue if the page is not multiple of 32
    uint32_t out_cb_pagesize = aligned_output_tile_nbytes;
    uint32_t out_cb_npages = num_tile_per_core * buffering_factor;
    const DataflowBufferSpec out_dfb{
        .unique_id = OUT_DFB,
        .entry_size = out_cb_pagesize,
        .num_entries = out_cb_npages,
        .data_format_metadata = out_df,
        .borrowed_from = OUTPUT,
    };

    log_debug(tt::LogOp, "input_dfb: {}, npages: {}, pagesize: {}", IN_DFB.get(), in_cb_npages, in_cb_pagesize);
    log_debug(tt::LogOp, "out_dfb: {}, npages: {}, pagesize: {}", OUT_DFB.get(), out_cb_npages, out_cb_pagesize);
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

    const TensorParameter input_param{.unique_id = INPUT, .spec = input.tensor_spec()};
    const TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    const KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/reader_unary_sharded_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles_per_core"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    KernelSpec::CompilerOptions::Defines unary_defines;
    unary_defines.emplace(
        "TYPECAST_LLK_INIT",
        fmt::format(
            "typecast_tile_init<{0}u, {1}u>",
            static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
            static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype))));
    unary_defines.emplace(
        "TYPECAST_LLK",
        fmt::format(
            "typecast_tile<{0}u, {1}u>",
            static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
            static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype))));

    // Legacy set unpack_to_dest_mode[in_cb] = UnpackToDestFp32 when preserve_fp32_precision and left
    // every other CB at Default; the named equivalent is an UnpackToDest entry for the input DFB.
    ComputeUnpackModes unpack_modes;
    if (args.preserve_fp32_precision) {
        unpack_modes.emplace(IN_DFB, tt::tt_metal::UnpackMode::UnpackToDest);
    } else if (args.fp32_dest_acc_en && act_df == tt::DataFormat::Float32) {
        // Metal 2.0 requires an explicit entry for a consumed Float32 DFB under a 32-bit Dest, where
        // legacy silently defaulted. UnpackToSrc is that legacy default.
        unpack_modes.emplace(IN_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
    }
    if (args.fp32_dest_acc_en && out_df == tt::DataFormat::Float32) {
        // The output DFB is self-looped on this kernel (compute is its only toucher), which makes
        // compute a *consumer* of it too — so the same required-entry rule applies. Legacy left this
        // CB's unpack mode at Default, i.e. UnpackToSrc.
        unpack_modes.emplace(OUT_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
    }

    // Field values carried over from the legacy ComputeConfigDescriptor: math_fidelity=HiFi4,
    // fp32_dest_acc_en, bfp8_pack_precise, math_approx_mode=false. dst_full_sync_en was left at its
    // legacy default (false) = double_buffer_dest true, which is also the Metal 2.0 default.
    const KernelSpec compute{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp",
        .compiler_options = {.defines = std::move(unary_defines)},
        // The output DFB has no other toucher — no writer kernel drains it, the borrowed output
        // buffer *is* the result — so compute binds it as both PRODUCER and CONSUMER (self-loop).
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .compile_time_args = {{"per_core_block_cnt", 1u}, {"per_core_block_dim", num_tile_per_core}},
        .hw_config = ComputeHardwareConfig{ComputeGen1Config{
            .fpu_math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
            .sfpu_precision_mode = tt::tt_metal::Precision::Precise,  // legacy math_approx_mode = false
            .bfp_pack_precision_mode =
                args.bfp8_pack_precise ? tt::tt_metal::Precision::Precise : tt::tt_metal::Precision::Approximate,
            .enable_32_bit_dest = args.fp32_dest_acc_en,
            .unpack_modes = std::move(unpack_modes),
        }},
    };

    KernelRunArgs reader_run_args{.kernel = READER};
    for (const CoreCoord& core : corerange_to_cores(all_cores)) {
        AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"num_tiles_per_core", num_tile_per_core}});
    }

    ProgramSpec spec{
        .name = "typecast_sharded",
        .kernels = {reader, compute},
        .dataflow_buffers = {in_dfb, out_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = {WorkUnitSpec{
            .name = "typecast_sharded", .kernels = {READER, COMPUTE}, .target_nodes = all_cores}},
    };

    // The compute kernel has no runtime args, so it needs no KernelRunArgs entry.
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args)};
    run_args.tensor_args = {{INPUT, input.mesh_tensor()}, {OUTPUT, output.mesh_tensor()}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
