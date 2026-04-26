// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "binary_device_operation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::binary {

tt::tt_metal::ProgramDescriptor BinaryDeviceOperation::ElementWiseMultiCore::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using ttnn::operations::unary::EltwiseFusedActivations;
    using namespace tt::constants;

    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;
    auto& output = tensor_return_value;
    const auto& op_type = operation_attributes.binary_op_type;

    auto fused_activations = operation_attributes.activations.value_or(EltwiseFusedActivations{});

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    tt::DataFormat src1_cb_data_format = tt_metal::datatype_to_dataformat_converter(b->dtype());
    uint32_t src1_single_tile_size = tt::tile_size(src1_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    tt::DataFormat interim_cb0_format = src0_cb_data_format;
    tt::DataFormat interim_cb1_format = src1_cb_data_format;

    tt_metal::Buffer* src0_buffer = a.buffer();
    tt_metal::Buffer* src1_buffer = b->buffer();

    std::optional<ShardSpec> shard_spec = std::nullopt;
    bool src0_sharded = a.memory_config().is_sharded();
    bool src1_sharded = b->memory_config().is_sharded();
    bool out_sharded = output.memory_config().is_sharded();

    bool block_or_width_sharded = false;
    std::optional<TensorMemoryLayout> sharded_layout = std::nullopt;

    if (src0_sharded) {
        shard_spec = a.shard_spec().value();
        block_or_width_sharded = a.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED;
        sharded_layout = a.memory_config().memory_layout();
    } else if (src1_sharded) {
        shard_spec = b->shard_spec().value();
        block_or_width_sharded = b->memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED;
        sharded_layout = b->memory_config().memory_layout();
    } else if (out_sharded) {
        shard_spec = output.shard_spec().value();
        block_or_width_sharded = output.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED;
        sharded_layout = output.memory_config().memory_layout();
    }

    uint32_t max_block_size = 1, num_tiles_per_shard = 0;
    if (shard_spec.has_value()) {
        num_tiles_per_shard = shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW;
        max_block_size = find_max_block_size(num_tiles_per_shard);
    }

    tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const auto& all_device_cores = operation_attributes.worker_grid;

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t src1_cb_index = tt::CBIndex::c_1;
    uint32_t output_cb_index = tt::CBIndex::c_2;

    uint32_t num_src0_tiles = src0_sharded ? num_tiles_per_shard : 2 * max_block_size;
    uint32_t num_src1_tiles = src1_sharded ? num_tiles_per_shard : 2 * max_block_size;
    uint32_t num_output_tiles = (out_sharded || block_or_width_sharded) ? num_tiles_per_shard : 2 * max_block_size;

    std::map<std::string, std::string> eltwise_defines = utils::get_defines(
        op_type, a.dtype(), output.dtype(), fused_activations, operation_attributes.input_tensor_a_activation);

    ProgramDescriptor desc;

    // ---- Circular buffers ----

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_src0_tiles * src0_single_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = src0_cb_data_format,
            .page_size = src0_single_tile_size,
        }}},
        .buffer = src0_sharded ? src0_buffer : nullptr,
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_src1_tiles * src1_single_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src1_cb_index),
            .data_format = src1_cb_data_format,
            .page_size = src1_single_tile_size,
        }}},
        .buffer = src1_sharded ? src1_buffer : nullptr,
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * dst_single_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = dst_cb_data_format,
            .page_size = dst_single_tile_size,
        }}},
        .buffer = out_sharded ? dst_buffer : nullptr,
    });

    if (eltwise_defines.contains("SFPU_OP_INIT_PRE_IN0_0")) {
        if (op_type == BinaryOpType::LOGADDEXP || op_type == BinaryOpType::LDEXP ||
            op_type == BinaryOpType::LOGADDEXP2) {
            interim_cb0_format = tt::DataFormat::Float16_b;
        }
        uint32_t interim0_single_tile_size = tt::tile_size(interim_cb0_format);
        desc.cbs.push_back(CBDescriptor{
            .total_size = max_block_size * interim0_single_tile_size,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_3),
                .data_format = interim_cb0_format,
                .page_size = interim0_single_tile_size,
            }}},
        });
    }
    if (eltwise_defines.contains("SFPU_OP_INIT_PRE_IN1_0")) {
        if (op_type == BinaryOpType::LOGADDEXP || op_type == BinaryOpType::LDEXP ||
            op_type == BinaryOpType::LOGADDEXP2) {
            interim_cb1_format = tt::DataFormat::Float16_b;
        }
        uint32_t interim1_single_tile_size = tt::tile_size(interim_cb1_format);
        desc.cbs.push_back(CBDescriptor{
            .total_size = max_block_size * interim1_single_tile_size,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_4),
                .data_format = interim_cb1_format,
                .page_size = interim1_single_tile_size,
            }}},
        });
    }

    // ---- Kernel compile-time args and defines ----

    std::map<std::string, std::string> reader_defines;
    KernelDescriptor::CompileTimeArgs reader_compile_time_args = {(std::uint32_t)block_or_width_sharded};
    if (src0_sharded) {
        reader_defines["IN0_SHARDED"] = "1";
    } else {
        TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    }
    if (src1_sharded) {
        reader_defines["IN1_SHARDED"] = "1";
    } else {
        TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args);
    }

    std::map<std::string, std::string> writer_defines;
    if (out_sharded) {
        writer_defines["OUT_SHARDED"] = "1";
    }

    KernelDescriptor::CompileTimeArgs writer_compile_time_args = {(std::uint32_t)output_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    bool fp32_dest_acc_en = dst_cb_data_format == tt::DataFormat::UInt32 ||
                            dst_cb_data_format == tt::DataFormat::Int32 ||
                            dst_cb_data_format == tt::DataFormat::Float32;

    // ---- Reader kernel ----

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_device_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.defines = {reader_defines.begin(), reader_defines.end()};
    reader_desc.config = ReaderConfigDescriptor{};

    // ---- Writer kernel ----

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        (block_or_width_sharded and not out_sharded)
            ? "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
              "writer_unary_sharded_blocks_interleaved_start_id.cpp"
            : "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_device_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.defines = {writer_defines.begin(), writer_defines.end()};
    writer_desc.config = WriterConfigDescriptor{};

    // ---- Compute kernel ----

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_kernel.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_device_cores;
    compute_desc.defines = {eltwise_defines.begin(), eltwise_defines.end()};
    compute_desc.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };

    // ---- Per-core runtime args (inlined from set_eltwise_binary_runtime_args) ----

    bool zero_start_grid = false;
    CoreCoord compute_with_storage_grid_size;
    if (all_device_cores.size() == 1) {
        const auto& cr = *all_device_cores.ranges().begin();
        if (cr.start_coord.x == 0 && cr.start_coord.y == 0) {
            if (shard_spec.has_value()) {
                const auto& shard_start_coord = shard_spec->grid.ranges()[0].start_coord;
                if (shard_start_coord.x == 0 && shard_start_coord.y == 0) {
                    zero_start_grid = true;
                    compute_with_storage_grid_size = CoreCoord(cr.end_coord.x + 1, cr.end_coord.y + 1);
                }
            } else {
                zero_start_grid = true;
                compute_with_storage_grid_size = CoreCoord(cr.end_coord.x + 1, cr.end_coord.y + 1);
            }
        }
    }

    uint32_t num_tiles = a.physical_volume() / TILE_HW;

    uint32_t num_cores_total;
    if (zero_start_grid) {
        num_cores_total = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;
    } else {
        num_cores_total = all_device_cores.num_cores();
    }

    CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_cores, num_tiles_per_core_group_1, num_tiles_per_core_group_2;
    uint32_t block_size_per_core_group_1 = 1, block_size_per_core_group_2 = 1;
    uint32_t block_cnt_per_core_group_1, block_cnt_per_core_group_2;

    bool row_major;
    uint32_t block_height = 0, block_width = 0, block_size = 0, output_width = 0;
    uint32_t last_unpadded_block_height = 0, last_unpadded_block_width = 0;
    CoreCoord end_core;
    std::vector<CoreCoord> cores;

    if (shard_spec.has_value()) {
        all_cores = shard_spec.value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
        num_tiles_per_core_group_1 = shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW;
        num_tiles_per_core_group_2 = 0;
        block_size_per_core_group_1 = find_max_block_size(num_tiles_per_core_group_1);

        block_cnt_per_core_group_1 = num_tiles_per_core_group_1 / block_size_per_core_group_1;
        block_cnt_per_core_group_2 = num_tiles_per_core_group_2 / block_size_per_core_group_2;
        row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
        block_height = shard_spec.value().shape[0] / TILE_HEIGHT;
        block_width = shard_spec.value().shape[1] / TILE_WIDTH;
        if (block_or_width_sharded) {
            block_size = block_width * block_height;
            end_core = (*shard_spec.value().grid.ranges().begin()).end_coord;
            output_width = output.padded_shape()[-1] / TILE_WIDTH;
            uint32_t output_height = output.physical_volume() / output.padded_shape()[-1] / TILE_HEIGHT;
            last_unpadded_block_height = block_height - (round_up(output_height, block_height) - output_height);
            last_unpadded_block_width = block_width - (round_up(output_width, block_width) - output_width);
        }
        if (zero_start_grid) {
            auto bbox = core_group_1.bounding_box();
            cores = grid_to_cores_with_noop(
                bbox.end_coord.x,
                bbox.end_coord.y,
                compute_with_storage_grid_size.x,
                compute_with_storage_grid_size.y,
                row_major);
        } else {
            cores = grid_to_cores_with_noop(all_cores, all_device_cores, row_major);
        }
    } else {
        row_major = true;
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2) =
            zero_start_grid ? tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles, row_major)
                            : tt::tt_metal::split_work_to_cores(all_device_cores, num_tiles, row_major);
        block_cnt_per_core_group_1 = num_tiles_per_core_group_1;
        block_cnt_per_core_group_2 = num_tiles_per_core_group_2;
        if (zero_start_grid) {
            cores = grid_to_cores(
                num_cores_total, compute_with_storage_grid_size.x, compute_with_storage_grid_size.y, row_major);
        } else {
            cores = corerange_to_cores(all_device_cores, {}, row_major);
        }
    }

    uint32_t g1_numcores = core_group_1.num_cores();

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; ++i) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_tiles_per_core = 0;
        uint32_t block_cnt_per_core = 0;
        uint32_t block_size_per_core = 0;
        uint32_t num_shards_per_width = 0;
        uint32_t start_id = 0;

        if (shard_spec.has_value()) {
            if (sharded_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
                num_shards_per_width = 1;
            } else if (sharded_layout == TensorMemoryLayout::WIDTH_SHARDED) {
                num_shards_per_width = num_cores;
            } else {  // block sharded
                auto bbox = core_group_1.bounding_box();
                if (shard_spec.value().orientation == ShardOrientation::ROW_MAJOR) {
                    num_shards_per_width = bbox.end_coord.x - bbox.start_coord.x + 1;
                } else {
                    num_shards_per_width = bbox.end_coord.y - bbox.start_coord.y + 1;
                }
            }
            start_id = (i / num_shards_per_width) * (block_height * block_width * num_shards_per_width) +
                       (i % num_shards_per_width) * block_width;
        } else {
            start_id = num_tiles_read;
        }

        if (i < g1_numcores) {
            num_tiles_per_core = num_tiles_per_core_group_1;
            block_cnt_per_core = block_cnt_per_core_group_1;
            block_size_per_core = block_size_per_core_group_1;
        } else if (i < num_cores) {
            num_tiles_per_core = num_tiles_per_core_group_2;
            block_cnt_per_core = block_cnt_per_core_group_2;
            block_size_per_core = block_size_per_core_group_2;
        } else {
            // Noop core
            reader_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs(7, 0));
            compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs(2, 0));
            if (block_or_width_sharded and not out_sharded) {
                writer_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs(7, 0));
            } else {
                writer_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs(3, 0));
            }
            continue;
        }

        reader_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                src0_buffer->address(),
                src1_buffer->address(),
                num_tiles_per_core,
                start_id,
                block_height,
                block_width,
                num_shards_per_width,
                num_shards_per_width});

        compute_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{block_cnt_per_core, block_size_per_core});

        if (block_or_width_sharded and not out_sharded) {
            uint32_t unpadded_block_height = block_height;
            uint32_t unpadded_block_width = block_width;
            if (row_major) {
                if (core.x == end_core.x) {
                    unpadded_block_width = last_unpadded_block_width;
                }
                if (core.y == end_core.y) {
                    unpadded_block_height = last_unpadded_block_height;
                }
            } else {
                if (core.y == end_core.y) {
                    unpadded_block_width = last_unpadded_block_width;
                }
                if (core.x == end_core.x) {
                    unpadded_block_height = last_unpadded_block_height;
                }
            }
            writer_desc.runtime_args.emplace_back(
                core,
                KernelDescriptor::CoreRuntimeArgs{
                    dst_buffer->address(),
                    block_height,
                    block_width,
                    unpadded_block_height,
                    unpadded_block_width,
                    output_width,
                    block_size,
                    ((i / num_shards_per_width) * (block_height * block_width * num_shards_per_width)) +
                        ((i % num_shards_per_width) * block_width),
                    0});
        } else {
            writer_desc.runtime_args.emplace_back(
                core,
                KernelDescriptor::CoreRuntimeArgs{dst_buffer->address(), num_tiles_per_core, num_tiles_read});
        }
        num_tiles_read += num_tiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::operations::binary
