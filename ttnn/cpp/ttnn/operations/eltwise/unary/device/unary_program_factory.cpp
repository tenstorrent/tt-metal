// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "unary_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

namespace ttnn::operations::unary::program {

static const std::string compute_root = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/";

using namespace tt::constants;

tt::tt_metal::ProgramDescriptor UnaryProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto& ops_chain = args.op_chain;

    tt::tt_metal::ProgramDescriptor program;

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);
    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t single_tile_size_output = tt::tt_metal::detail::TileSize(cb_data_format_output);

    uint32_t num_tiles = input.volume() / tt::constants::TILE_HW;

    tt::tt_metal::IDevice* device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 2;
    program.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * single_tile_size,
        .core_ranges = all_cores.ranges(),
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }},
    });

    uint32_t output_cb_index = tt::CBIndex::c_2;
    uint32_t num_output_tiles = 2;
    program.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * single_tile_size_output,
        .core_ranges = all_cores.ranges(),
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = cb_data_format_output,
            .page_size = single_tile_size_output,
        }},
    });

    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    constexpr size_t max_num_kernels = 4;
    program.kernels.resize(max_num_kernels);
    size_t num_kernels = 0;

    auto& unary_reader_kernel = program.kernels[num_kernels++];
    unary_reader_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp";
    unary_reader_kernel.core_ranges = all_cores.ranges();
    unary_reader_kernel.common_runtime_args = {(uint32_t)src_is_dram};
    unary_reader_kernel.config = ReaderConfigDescriptor{};
    unary_reader_kernel.reserve_runtime_args();

    auto& unary_writer_kernel = program.kernels[num_kernels++];
    unary_writer_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    unary_writer_kernel.core_ranges = all_cores.ranges();
    unary_writer_kernel.common_runtime_args = {(std::uint32_t)output_cb_index, (std::uint32_t)dst_is_dram};
    unary_writer_kernel.config = WriterConfigDescriptor{};
    unary_writer_kernel.reserve_runtime_args();

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (args.preserve_fp32_precision) {
        unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    bool math_approx_mode = std::all_of(
        args.op_chain.begin(), args.op_chain.end(), [](const auto& u) { return utils::get_op_approx_mode(u.op_type); });
    KernelDescriptor::Defines unary_defines;
    utils::append_block_defines(unary_defines, args.op_chain, "0", "0", input.get_dtype());
    auto path = utils::get_compute_kernel_path(ops_chain[0].op_type, compute_root);

    auto& eltwise_unary_kernel_group_1 = program.kernels[num_kernels++];
    eltwise_unary_kernel_group_1.kernel_source = path;
    eltwise_unary_kernel_group_1.core_ranges = core_group_1.ranges();
    eltwise_unary_kernel_group_1.compile_time_args = {
        num_tiles_per_core_group_1,  // per_core_block_cnt
        1                            // per_core_block_size
    };
    eltwise_unary_kernel_group_1.defines = unary_defines;
    eltwise_unary_kernel_group_1.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = args.fp32_dest_acc_en,
        .unpack_to_dest_mode = unpack_to_dest_mode,
        .bfp8_pack_precise = args.bfp8_pack_precise,
        .math_approx_mode = math_approx_mode,
    };

    if (!core_group_2.ranges().empty()) {
        auto& eltwise_unary_kernel_group_2 = program.kernels[num_kernels++];
        eltwise_unary_kernel_group_2.kernel_source = path;
        eltwise_unary_kernel_group_2.core_ranges = core_group_2.ranges();
        eltwise_unary_kernel_group_2.compile_time_args = {
            num_tiles_per_core_group_2,  // per_core_block_cnt
            1                            // per_core_block_size
        };
        eltwise_unary_kernel_group_2.defines = std::move(unary_defines);
        eltwise_unary_kernel_group_2.config = ComputeConfigDescriptor{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = args.fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .bfp8_pack_precise = args.bfp8_pack_precise,
            .math_approx_mode = math_approx_mode,
        };
    }

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        auto& unary_reader_args = unary_reader_kernel.runtime_args[core.x][core.y];
        auto& unary_writer_args = unary_writer_kernel.runtime_args[core.x][core.y];

        unary_reader_args = {src_buffer->address(), num_tiles_per_core, num_tiles_written};
        unary_writer_args = {dst_buffer->address(), num_tiles_per_core, num_tiles_written};

        num_tiles_written += num_tiles_per_core;
    }

    program.kernels.resize(num_kernels);
    return program;
}

}  // namespace ttnn::operations::unary::program
