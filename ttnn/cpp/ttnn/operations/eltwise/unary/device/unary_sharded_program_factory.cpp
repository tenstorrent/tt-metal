// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_sharded_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

namespace ttnn::operations::unary::program {

static const std::string compute_root_sharded = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/";

using namespace tt::constants;
using namespace tt::tt_metal;

tt::tt_metal::ProgramDescriptor UnaryShardedProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto& ops_chain = args.op_chain;

    tt::tt_metal::ProgramDescriptor program;
    tt::tt_metal::IDevice* device = input.device();

    auto shard_spec = input.shard_spec().value();
    auto all_cores = shard_spec.grid;
    uint32_t ncores = shard_spec.num_cores();

    auto out_shard_spec = output.shard_spec().value();
    TT_FATAL(
        out_shard_spec.num_cores() == ncores,
        "Output tensor should have same number of cores {} as input tensor {}",
        out_shard_spec.num_cores(),
        ncores);

    tt::DataFormat act_df = tt::tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    uint32_t input_tile_size = tt::tt_metal::detail::TileSize(act_df);
    uint32_t output_tile_size = tt::tt_metal::detail::TileSize(out_df);

    TT_FATAL(input_tile_size == output_tile_size, "Input and output tile size should be same");

    uint32_t num_tile_per_core = 0;

    if (input.get_dtype() == DataType::BFLOAT8_B || input.get_dtype() == DataType::BFLOAT4_B) {
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
    program.cbs.push_back(CBDescriptor{
        .total_size = in_cb_pagesize * in_cb_npages,
        .core_ranges = all_cores.ranges(),
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = in_cb_id,
            .data_format = act_df,
            .page_size = in_cb_pagesize,
        }},
        .buffer = input.buffer(),
    });

    // output sharded CB
    uint32_t out_cb_id = tt::CBIndex::c_2;
    program.cbs.push_back(CBDescriptor{
        .total_size = in_cb_pagesize * in_cb_npages,
        .core_ranges = all_cores.ranges(),
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = out_cb_id,
            .data_format = out_df,
            .page_size = in_cb_pagesize,
        }},
        .buffer = output.buffer(),
    });

    log_debug(tt::LogOp, "input_cb: {}, npages: {}, pagesize: {}", in_cb_id, in_cb_npages, in_cb_pagesize);
    log_debug(tt::LogOp, "input_tile_size: {}", input_tile_size);

    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    TT_FATAL(src_is_dram == 0, "Input buffer should be in L1");

    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    TT_FATAL(dst_is_dram == 0, "Output buffer should be in L1");

    constexpr size_t num_kernels = 2;
    program.kernels.resize(num_kernels);

    auto& unary_reader_kernel = program.kernels[0];
    unary_reader_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp";
    unary_reader_kernel.core_ranges = all_cores.ranges();
    unary_reader_kernel.compile_time_args = {in_cb_id};
    unary_reader_kernel.config = ReaderConfigDescriptor{};
    unary_reader_kernel.reserve_runtime_args();

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (args.preserve_fp32_precision) {
        unpack_to_dest_mode[in_cb_id] = UnpackToDestMode::UnpackToDestFp32;
    }

    bool math_approx_mode = std::all_of(
        args.op_chain.begin(), args.op_chain.end(), [](const auto& u) { return utils::get_op_approx_mode(u.op_type); });
    KernelDescriptor::Defines unary_defines;
    utils::append_block_defines(unary_defines, args.op_chain, "0", "0", input.get_dtype());
    auto path = utils::get_compute_kernel_path(ops_chain[0].op_type, compute_root_sharded);

    auto& eltwise_unary_kernel_group_1 = program.kernels[1];
    eltwise_unary_kernel_group_1.kernel_source = path;
    eltwise_unary_kernel_group_1.core_ranges = all_cores.ranges();
    eltwise_unary_kernel_group_1.compile_time_args = {
        1,                 // per_core_block_cnt
        num_tile_per_core  // per_core_block_size
    };
    eltwise_unary_kernel_group_1.defines = std::move(unary_defines);
    eltwise_unary_kernel_group_1.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = args.fp32_dest_acc_en,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
        .bfp8_pack_precise = args.bfp8_pack_precise,
        .math_approx_mode = math_approx_mode,
    };

    for (const auto& core_range : all_cores.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                unary_reader_kernel.runtime_args[x][y] = {
                    (uint32_t)(num_tile_per_core),
                };
            }
        }
    }

    return program;
}

}  // namespace ttnn::operations::unary::program
