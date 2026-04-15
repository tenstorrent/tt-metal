// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "unary_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

namespace ttnn::prim {

using ttnn::operations::unary::UnaryOpType;
namespace utils = ttnn::operations::unary::utils;

static const std::string compute_root = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/";

using namespace tt::constants;

tt::tt_metal::ProgramDescriptor UnaryProgramFactory::create_descriptor(
    const UnaryParams& args, const UnaryInputs& tensor_args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto& ops_chain = args.op_chain;
    uint32_t packed_scalar1 = 0u;
    uint32_t packed_scalar2 = 0u;

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t single_tile_size_output = tt::tile_size(cb_data_format_output);

    // Get number of pages (tiles for TILE layout, rows for ROW_MAJOR layout)
    const uint32_t num_pages = input.buffer()->num_pages();
    const bool is_row_major = input.layout() == Layout::ROW_MAJOR;

    tt::tt_metal::IDevice* device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_pages);

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 2;
    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    // Set CB page size correctly based on layout
    const uint32_t input_cb_page_size = is_row_major ? src_buffer->page_size() : single_tile_size;
    const uint32_t output_cb_page_size = is_row_major ? dst_buffer->page_size() : single_tile_size_output;

    // For bitcast, use output format for input CB to avoid unpacker conversion
    tt::DataFormat cb_data_format_for_input =
        (ops_chain[0].type() == UnaryOpType::BITCAST) ? cb_data_format_output : cb_data_format;

    ProgramDescriptor desc;

    // ---- Circular buffers ----

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * input_cb_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = cb_data_format_for_input,
            .page_size = input_cb_page_size,
        }}},
    });

    uint32_t output_cb_index = tt::CBIndex::c_2;
    uint32_t num_output_tiles = 2;

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * output_cb_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = cb_data_format_output,
            .page_size = output_cb_page_size,
        }}},
    });

    // ---- Kernel compile-time args and defines ----

    std::vector<uint32_t> reader_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(output_cb_index)};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    std::vector<uint32_t> compute_kernel_args_group_1 = {
        num_pages_per_core_group_1,  // per_core_block_cnt
        1,                           // per_core_block_size
    };

    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (args.preserve_fp32_precision) {
        unpack_to_dest_mode[src0_cb_index] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
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

    auto path = fmt::format("{}/{}", compute_root, utils::get_compute_kernel_path(ops_chain[0].type(), input.dtype()));

    // Due to hardware bug (#38306), HiFi4 + fp32_dest_acc_en can sometime produce incorrect results on Wormhole.
    // Use HiFi3 when fp32_dest_acc_en is True on Wormhole (less likely to give bad results).
    const auto default_fp32_acc_math_fidelity =
        (args.fp32_dest_acc_en && device->arch() == tt::ARCH::WORMHOLE_B0)
            ? tt::tt_metal::MathFidelity::HiFi3
            : tt::tt_metal::MathFidelity::HiFi4;

    // Convert map defines to vector of pairs for KernelDescriptor
    KernelDescriptor::Defines kernel_defines;
    for (const auto& [key, value] : unary_defines) {
        kernel_defines.emplace_back(key, value);
    }

    // ---- Reader kernel ----

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    // ---- Writer kernel ----

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.config = WriterConfigDescriptor{};

    // ---- Compute kernel (core_group_1) ----

    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source = path;
    compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = compute_kernel_args_group_1;
    compute_desc_1.defines = kernel_defines;
    compute_desc_1.config = ComputeConfigDescriptor{
        .math_fidelity = default_fp32_acc_math_fidelity,
        .fp32_dest_acc_en = args.fp32_dest_acc_en,
        .unpack_to_dest_mode =
            {unpack_to_dest_mode.begin(), unpack_to_dest_mode.end()},
        .bfp8_pack_precise = args.bfp8_pack_precise,
        .math_approx_mode = math_approx_mode,
    };

    // ---- Compute kernel (core_group_2, if non-empty) ----

    KernelDescriptor compute_desc_2;
    bool has_core_group_2 = !core_group_2.ranges().empty();
    if (has_core_group_2) {
        std::vector<uint32_t> compute_kernel_args_group_2 = {
            num_pages_per_core_group_2,  // per_core_block_cnt
            1,                           // per_core_block_size
        };

        compute_desc_2.kernel_source = path;
        compute_desc_2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_2.core_ranges = core_group_2;
        compute_desc_2.compile_time_args = compute_kernel_args_group_2;
        compute_desc_2.defines = kernel_defines;
        compute_desc_2.config = ComputeConfigDescriptor{
            .math_fidelity = default_fp32_acc_math_fidelity,
            .fp32_dest_acc_en = args.fp32_dest_acc_en,
            .unpack_to_dest_mode =
                {unpack_to_dest_mode.begin(), unpack_to_dest_mode.end()},
            .bfp8_pack_precise = args.bfp8_pack_precise,
            .math_approx_mode = math_approx_mode,
        };
    }

    // ---- Per-core runtime args ----

    for (uint32_t i = 0, num_pages_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_pages_per_core = 0;
        if (core_group_1.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        reader_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{src_buffer->address(), num_pages_per_core, num_pages_written});

        writer_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{dst_buffer->address(), num_pages_per_core, num_pages_written});

        // Compute runtime args go to the appropriate core group's descriptor
        if (core_group_1.contains(core)) {
            compute_desc_1.runtime_args.emplace_back(
                core, KernelDescriptor::CoreRuntimeArgs{packed_scalar1, packed_scalar2});
        } else if (has_core_group_2) {
            compute_desc_2.runtime_args.emplace_back(
                core, KernelDescriptor::CoreRuntimeArgs{packed_scalar1, packed_scalar2});
        }

        num_pages_written += num_pages_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_1));
    if (has_core_group_2) {
        desc.kernels.push_back(std::move(compute_desc_2));
    }

    return desc;
}

// Sub Core Grids : should be fused later with UnaryProgramFactory directing all_device_cores to use cores from
// sub_core_grids and update the override args accordingly after adding cores as rtargs
tt::tt_metal::ProgramDescriptor UnarySubCoreGridProgramFactory::create_descriptor(
    const UnaryParams& args, const UnaryInputs& tensor_args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto& ops_chain = args.op_chain;
    uint32_t packed_scalar1 = 0u;
    uint32_t packed_scalar2 = 0u;
    const auto& sub_core_grids = args.sub_core_grids;

    TT_FATAL(sub_core_grids.has_value(), "sub_core_grids cannot be null");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t single_tile_size_output = tt::tile_size(cb_data_format_output);

    uint32_t num_tiles = input.physical_volume() / tt::constants::TILE_HW;

    uint32_t ncores = sub_core_grids->num_cores();

    TT_FATAL(ncores != 0, "number of cores cannot be 0");

    for (uint32_t core_id = ncores; core_id >= 1; core_id--) {
        if (num_tiles % ncores == 0) {
            break;
        }
        ncores--;
    }
    TT_FATAL(
        (num_tiles % (ncores) == 0),
        "{} num of tiles are not split uniformly across {} num of cores",
        num_tiles,
        ncores);

    auto cores = corerange_to_cores(sub_core_grids.value(), ncores, true);
    auto all_cores = num_cores_to_corerangeset_in_subcoregrids(cores[0], ncores, sub_core_grids.value(), true);
    if (ncores == 1) {
        all_cores = ttnn::CoreRangeSet(ttnn::CoreRange(cores[0]));
    }

    uint32_t ntiles_per_block = num_tiles / ncores;
    uint32_t nblocks = (num_tiles / ntiles_per_block);
    uint32_t nblocks_per_core = nblocks / ncores;

    uint32_t num_input_tiles = ntiles_per_block * 2;

    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    ProgramDescriptor desc;

    // ---- Circular buffers ----

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    uint32_t output_cb_index = tt::CBIndex::c_2;
    uint32_t num_output_tiles = ntiles_per_block * 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * single_tile_size_output,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = cb_data_format_output,
            .page_size = single_tile_size_output,
        }}},
    });

    // ---- Reader kernel ----

    std::vector<uint32_t> reader_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    // ---- Writer kernel ----

    std::vector<uint32_t> writer_compile_time_args = {output_cb_index};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.config = WriterConfigDescriptor{};

    // ---- Compute kernel ----

    std::vector<uint32_t> compute_kernel_args = {
        (uint32_t)nblocks_per_core,  // per_core_block_cnt
        (uint32_t)ntiles_per_block,  // per_block_num_tiles // per_core_block_size
    };

    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (args.preserve_fp32_precision) {
        unpack_to_dest_mode[src0_cb_index] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
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

    auto path = fmt::format("{}/{}", compute_root, utils::get_compute_kernel_path(ops_chain[0].type(), input.dtype()));

    // Due to hardware bug (#38306), HiFi4 + fp32_dest_acc_en can sometime produce incorrect results on Wormhole.
    // Use HiFi3 when fp32_dest_acc_en is True on Wormhole (less likely to give bad results).
    const auto default_fp32_acc_math_fidelity_sub =
        (args.fp32_dest_acc_en && input.device()->arch() == tt::ARCH::WORMHOLE_B0)
            ? tt::tt_metal::MathFidelity::HiFi3
            : tt::tt_metal::MathFidelity::HiFi4;

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = path;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = compute_kernel_args;
    compute_desc.defines = {unary_defines.begin(), unary_defines.end()};
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = default_fp32_acc_math_fidelity_sub,
        .fp32_dest_acc_en = args.fp32_dest_acc_en,
        .unpack_to_dest_mode = {unpack_to_dest_mode.begin(), unpack_to_dest_mode.end()},
        .bfp8_pack_precise = args.bfp8_pack_precise,
        .math_approx_mode = math_approx_mode,
    };

    // ---- Per-core runtime args ----

    uint32_t tile_start_id = 0;
    auto ntiles_per_core = ntiles_per_block * nblocks_per_core;

    for (auto core : cores) {
        reader_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{src_buffer->address(), ntiles_per_core, tile_start_id});

        writer_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{dst_buffer->address(), ntiles_per_core, tile_start_id});

        compute_desc.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{packed_scalar1, packed_scalar2});

        tile_start_id += ntiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
