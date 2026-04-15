// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_sharded_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

namespace ttnn::prim {

using ttnn::operations::unary::UnaryOpType;
namespace utils = ttnn::operations::unary::utils;

static const std::string compute_root_sharded = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/";

using namespace tt::constants;
using namespace tt::tt_metal;

tt::tt_metal::ProgramDescriptor UnaryShardedProgramFactory::create_descriptor(
    const UnaryParams& args, const UnaryInputs& tensor_args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto& ops_chain = args.op_chain;
    uint32_t packed_scalar1 = 0u;
    uint32_t packed_scalar2 = 0u;

    TT_FATAL(args.sub_core_grids == std::nullopt, "Sub core grids are not supported for sharded input tensors");
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
    // For bitcast, use output format for input CB to avoid unpacker conversion
    // This ensures raw bit copying without conversion
    tt::DataFormat cb_data_format_for_input = (ops_chain[0].type() == UnaryOpType::BITCAST) ? out_df : act_df;

    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    TT_FATAL(src_is_dram == 0, "Input buffer should be in L1");
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    TT_FATAL(dst_is_dram == 0, "Output buffer should be in L1");

    ProgramDescriptor desc;

    // ---- Circular buffers ----

    // src0 CB - sharded (globally allocated)
    desc.cbs.push_back(CBDescriptor{
        .total_size = in_cb_pagesize * in_cb_npages,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(in_cb_id),
            .data_format = cb_data_format_for_input,
            .page_size = in_cb_pagesize,
        }}},
        .buffer = src_buffer,
    });

    // tmp CB for intermediate results (e.g., HARDSHRINK)
    uint32_t tmp_cb_id = tt::CBIndex::c_1;
    if (ops_chain[0].type() == UnaryOpType::HARDSHRINK) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = in_cb_pagesize * in_cb_npages,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tmp_cb_id),
                .data_format = act_df,
                .page_size = in_cb_pagesize,
            }}},
        });
    }

    // output CB - sharded (globally allocated)
    uint32_t out_cb_id = tt::CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = in_cb_pagesize * in_cb_npages,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(out_cb_id),
            .data_format = out_df,
            .page_size = in_cb_pagesize,
        }}},
        .buffer = dst_buffer,
    });

    log_debug(tt::LogOp, "input_cb: {}, npages: {}, pagesize: {}", in_cb_id, in_cb_npages, in_cb_pagesize);
    log_debug(tt::LogOp, "input_tile_size: {}", input_tile_size);

    // ---- Reader kernel ----
    std::vector<uint32_t> reader_compile_time_args = {in_cb_id};

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    // Set runtime args for all cores (broadcast to all cores in the shard grid)
    for (const auto& range : all_cores.ranges()) {
        for (uint32_t y = range.start_coord.y; y <= range.end_coord.y; ++y) {
            for (uint32_t x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                reader_desc.runtime_args.emplace_back(
                    CoreCoord{x, y}, KernelDescriptor::CoreRuntimeArgs{num_tile_per_core});
            }
        }
    }

    // ---- Compute kernel ----
    std::vector<uint32_t> compute_kernel_args = {
        1,                 // per_core_block_cnt
        num_tile_per_core  // per_core_block_size
    };

    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (args.preserve_fp32_precision) {
        unpack_to_dest_mode[in_cb_id] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[tmp_cb_id] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }

    bool math_approx_mode = std::all_of(
        args.op_chain.begin(), args.op_chain.end(), [](const auto& u) { return utils::get_op_approx_mode(u.type()); });
    std::map<std::string, std::string> unary_defines = utils::get_block_defines(args.op_chain, "0", "0", input.dtype());

    if (input.dtype() == DataType::FLOAT32) {
        unary_defines["INP_FLOAT32"] = "1";
    } else if (input.dtype() == DataType::INT32) {
        unary_defines["INP_INT32"] = "1";
    } else if (input.dtype() == DataType::UINT32) {
        unary_defines["INP_UINT32"] = "1";
    } else {
        unary_defines["INP_FLOAT"] = "1";
    }

    if (!ops_chain[0].empty()) {
        switch (ops_chain[0].type()) {
            case UnaryOpType::HARDSHRINK:
            case UnaryOpType::MISH:
                packed_scalar1 = utils::pack_scalar_runtime_arg(ops_chain[0], 0, input.dtype());
                break;
            case UnaryOpType::WHERE_TSS:
                packed_scalar1 = utils::pack_scalar_runtime_arg(ops_chain[0], 0, input.dtype());
                packed_scalar2 = utils::pack_scalar_runtime_arg(ops_chain[0], 1, input.dtype());
                break;
            default: break;
        }
    }

    auto path =
        fmt::format("{}/{}", compute_root_sharded, utils::get_compute_kernel_path(ops_chain[0].type(), input.dtype()));

    // Due to hardware bug (#38306), HiFi4 + fp32_dest_acc_en can sometime produce incorrect results on Wormhole.
    // Use HiFi3 when fp32_dest_acc_en is True on Wormhole (less likely to give bad results).
    const auto default_fp32_acc_math_fidelity =
        (args.fp32_dest_acc_en && input.device()->arch() == tt::ARCH::WORMHOLE_B0) ? tt::tt_metal::MathFidelity::HiFi3
                                                                                   : tt::tt_metal::MathFidelity::HiFi4;

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = path;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = compute_kernel_args;
    compute_desc.defines = {unary_defines.begin(), unary_defines.end()};
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = default_fp32_acc_math_fidelity,
        .fp32_dest_acc_en = args.fp32_dest_acc_en,
        .unpack_to_dest_mode = {unpack_to_dest_mode.begin(), unpack_to_dest_mode.end()},
        .bfp8_pack_precise = args.bfp8_pack_precise,
        .math_approx_mode = math_approx_mode,
    };

    // Set compute runtime args for all cores
    for (const auto& range : all_cores.ranges()) {
        for (uint32_t y = range.start_coord.y; y <= range.end_coord.y; ++y) {
            for (uint32_t x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                compute_desc.runtime_args.emplace_back(
                    CoreCoord{x, y}, KernelDescriptor::CoreRuntimeArgs{packed_scalar1, packed_scalar2});
            }
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
