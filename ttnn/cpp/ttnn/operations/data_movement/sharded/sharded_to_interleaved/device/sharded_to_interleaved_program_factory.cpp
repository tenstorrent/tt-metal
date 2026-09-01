// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts ShardedToInterleavedProgramFactory::create_program_artifacts(
    const ShardedToInterleavedParams& operation_attributes,
    const ShardedToInterleavedInputs& tensor_args,
    Tensor& output_tensor) {
    const auto& input = tensor_args.input_tensor;
    const auto& output = output_tensor;
    const uint32_t num_slices = operation_attributes.num_slices;
    const uint32_t slice_index = operation_attributes.slice_index;
    const bool is_l1_aligned = true;

    tt::DataFormat input_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());

    auto shard_spec = input.shard_spec().value();
    auto shard_strategy = input.memory_config().memory_layout();
    const bool is_tile = input.layout() == Layout::TILE;
    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    auto& all_cores = shard_spec.grid;

    // Extents in destination units: tiles for tile layout, sticks x bytes for row-major. A row-major
    // page is one logical row, so its extent comes from the logical shape, not the padded one.
    uint32_t input_unit_size = 0;
    uint32_t output_unit_size = 0;
    uint32_t tensor_h = 0;
    uint32_t tensor_w = 0;
    uint32_t shard_h = 0;
    uint32_t shard_w = 0;
    uint32_t num_units_per_shard = 0;
    if (is_tile) {
        input_unit_size = tt::tile_size(input_data_format);
        output_unit_size = tt::tile_size(output_data_format);
        tensor_h = (input.physical_volume() / input.padded_shape()[-1]) / TILE_HEIGHT;
        tensor_w = input.padded_shape()[-1] / TILE_WIDTH;
        shard_h = shard_spec.shape[0] / TILE_HEIGHT;
        shard_w = shard_spec.shape[1] / TILE_WIDTH;
        num_units_per_shard = shard_h * shard_w;
    } else {
        input_unit_size = static_cast<uint32_t>(shard_spec.shape[1] * input.element_size());
        output_unit_size = static_cast<uint32_t>(shard_spec.shape[1] * output.element_size());
        tensor_h = static_cast<uint32_t>(input.logical_volume() / input.logical_shape()[-1]);
        tensor_w = static_cast<uint32_t>(input.logical_shape()[-1] * input.element_size());
        shard_h = shard_spec.shape[0];
        shard_w = output_unit_size;
        num_units_per_shard = shard_h;
    }

    const uint32_t height_shards = div_up(tensor_h, shard_h);
    const uint32_t width_shards = div_up(tensor_w, shard_w);
    const uint32_t num_active_cores = height_shards * width_shards;

    // A grid provisioned wider than the data leaves cores holding only padding; leave them out.
    const CoreCoord grid_origin = all_cores.bounding_box().start_coord;
    CoreRangeSet used_cores;
    if (shard_strategy == TensorMemoryLayout::BLOCK_SHARDED) {
        const uint32_t grid_x = rm_orientation ? width_shards : height_shards;
        const uint32_t grid_y = rm_orientation ? height_shards : width_shards;
        used_cores =
            CoreRangeSet(CoreRange(grid_origin, CoreCoord{grid_origin.x + grid_x - 1, grid_origin.y + grid_y - 1}));
    } else {
        used_cores = num_active_cores < all_cores.num_cores()
                         ? select_from_corerangeset(all_cores, 0, num_active_cores - 1, rm_orientation)
                         : all_cores;
    }
    const auto cores = corerange_to_cores(all_cores, std::nullopt, rm_orientation);

    bool convert_df = input_data_format != output_data_format;

    uint32_t num_input_units = num_units_per_shard;
    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();
    uint32_t input_page_size = tt::align(input_unit_size, src_buffer->alignment());
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool is_blackhole = (input.device()->arch() == tt::ARCH::BLACKHOLE);

    // ---- Resource names ----
    const DFBSpecName IN_DFB{"in"};    // the resident input shard
    const DFBSpecName OUT_DFB{"out"};  // converted tiles; exists only when converting formats
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    Group<DataflowBufferSpec> dataflow_buffers;

    // Sharded input DFB, built on the input shard's own L1 buffer (borrowed memory) so the pages are
    // already resident and no NoC read is needed. The backing address re-resolves from the INPUT tensor
    // argument on every enqueue, so a program-cache hit picks up a reallocated input shard.
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN_DFB,
        .entry_size = input_page_size,
        .num_entries = num_input_units,
        .data_format_metadata = input_data_format,
        .borrowed_from = INPUT,
    });

    if (convert_df) {
        // Separate output DFB, allocated in L1 for the compute kernel to pack converted tiles into.
        // Unlike the input DFB it is not borrowed — nothing outside the program backs it.
        uint32_t output_page_size = tt::align(output_unit_size, dst_buffer->alignment());
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = OUT_DFB,
            .entry_size = output_page_size,
            .num_entries = num_input_units,
            .data_format_metadata = output_data_format,
        });
    }

    // Reader kernel (sharded input handed over through the borrowed input DFB). This binds the shared
    // Metal 2.0 reader fork that already lives in typecast's tree, so its accessor and argument names
    // are that kernel's interface, not this op's choice.
    const KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/reader_unary_sharded_metal2.cpp",
        .dfb_bindings =
            {DFBBinding{
                .dfb_spec_name = IN_DFB,
                .accessor_name = "in",
                .endpoint_type = DFBEndpointType::PRODUCER,
            }},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles_per_core"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    // Writer kernel (writes interleaved output to DRAM). Both layout variants present the same binding
    // interface — dfb::out for the resident block, tensor::dst for the interleaved destination — so only
    // the source path and the runtime-arg schema differ between them.
    KernelSpec writer{
        .unique_id = WRITER,
        .dfb_bindings =
            {DFBBinding{
                // With no format conversion there is nothing to convert into, so the writer drains
                // the very DFB the reader fills rather than a separate output buffer.
                .dfb_spec_name = convert_df ? OUT_DFB : IN_DFB,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::CONSUMER,
            }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };
    if (is_tile) {
        writer.source =
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
            "writer_unary_sharded_blocks_interleaved_start_id_metal2.cpp";
        writer.runtime_arg_schema.runtime_arg_names = {
            "block_height_tiles",
            "block_width_tiles",
            "unpadded_block_height_tiles",
            "unpadded_block_width_tiles",
            "output_width_tiles",
            "block_num_tiles",
            "start_id_offset",
            "start_id_base"};
    } else {
        writer.source =
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
            "writer_unary_stick_layout_sharded_blocks_interleaved_start_id_metal2.cpp";
        writer.runtime_arg_schema.runtime_arg_names = {
            "block_height", "block_width_bytes", "padded_block_width_bytes", "input_width_offset_bytes", "start_id"};
    }

    Group<KernelSpec> kernels;
    kernels.push_back(reader);
    kernels.push_back(std::move(writer));

    Group<KernelSpecName> work_unit_kernels{READER, WRITER};

    // Optional compute kernel for data-format conversion.
    if (convert_df) {
        kernels.push_back(KernelSpec{
            .unique_id = COMPUTE,
            .source = "ttnn/cpp/ttnn/kernel/compute/eltwise_copy_metal2.cpp",
            // The legacy ComputeConfigDescriptor set no opt_level, which resolves to O3 for a compute
            // kernel; Metal 2.0's CompilerOptions defaults to O2, so O3 has to be stated here.
            .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings =
                {DFBBinding{
                     .dfb_spec_name = IN_DFB,
                     .accessor_name = "in",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 },
                 DFBBinding{
                     .dfb_spec_name = OUT_DFB,
                     .accessor_name = "out",
                     .endpoint_type = DFBEndpointType::PRODUCER,
                 }},
            .compile_time_args = {{"per_core_tile_cnt", num_units_per_shard}},
            // Every field of the legacy ComputeConfigDescriptor{} was left at its default, and the
            // Metal 2.0 Gen1 compute defaults match those field for field (HiFi4; math_approx_mode
            // false = Precise SFPU; bfp8_pack_precise false = Approximate pack; fp32_dest_acc_en
            // false; dst_full_sync_en false = double_buffer_dest true), so an all-default Gen1 config
            // reproduces the legacy settings exactly.
            .hw_config = ComputeHardwareConfig{},
        });
        work_unit_kernels.push_back(COMPUTE);
    }

    // Reader runtime args: identical on every used core.
    KernelRunArgs reader_run_args{.kernel = READER};
    for (const auto& core_range : used_cores.ranges()) {
        for (const auto& core : core_range) {
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values, core, {{"num_tiles_per_core", num_units_per_shard}});
        }
    }

    uint32_t starting_idx_h =
        operations::data_movement::detail::calculate_starting_idx_h(output, num_slices, slice_index);

    // Source stride: the shard's row pitch in L1, which stays the full shard width.
    uint32_t padded_shard_width = tt::align(output_unit_size, dst_buffer->alignment());
    if (is_blackhole or is_l1_aligned) {
        if (!dst_is_dram or is_l1_aligned) {
            padded_shard_width = tt::align(output_unit_size, hal::get_l1_alignment());
        }
    }

    // One clipped shard->destination mapping drives the writer. The destination base address is no
    // longer a runtime arg — it arrives with the OUTPUT tensor binding.
    KernelRunArgs writer_run_args{.kernel = WRITER};
    for (uint32_t sh = 0; sh < height_shards; sh++) {
        for (uint32_t sw = 0; sw < width_shards; sw++) {
            // Height sharding has a single column and width sharding a single row, so one index is 0.
            const CoreCoord core =
                shard_strategy == TensorMemoryLayout::BLOCK_SHARDED
                    ? CoreCoord{grid_origin.x + (rm_orientation ? sw : sh), grid_origin.y + (rm_orientation ? sh : sw)}
                    : cores[sh + sw];
            const uint32_t h0 = sh * shard_h;
            const uint32_t w0 = sw * shard_w;
            // Clipping to the tensor is what keeps the write inside the destination page.
            const uint32_t shard_height = std::min(shard_h, tensor_h - h0);
            const uint32_t shard_width = std::min(shard_w, tensor_w - w0);

            if (is_tile) {
                AddRuntimeArgsForNode(
                    writer_run_args.runtime_arg_values,
                    core,
                    {{"block_height_tiles", shard_h},
                     {"block_width_tiles", shard_w},
                     {"unpadded_block_height_tiles", shard_height},
                     {"unpadded_block_width_tiles", shard_width},
                     {"output_width_tiles", tensor_w},
                     {"block_num_tiles", num_units_per_shard},
                     {"start_id_offset", h0 * tensor_w + w0},
                     {"start_id_base", starting_idx_h}});
            } else {
                // The row-major kernel reads no equivalent of the legacy arg 1 (num_units_per_row),
                // so the named schema has no slot for it.
                AddRuntimeArgsForNode(
                    writer_run_args.runtime_arg_values,
                    core,
                    {{"block_height", shard_height},
                     {"block_width_bytes", shard_width},
                     {"padded_block_width_bytes", padded_shard_width},
                     {"input_width_offset_bytes", w0},
                     {"start_id", h0}});
            }
        }
    }

    ProgramSpec spec{
        .name = "sharded_to_interleaved",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units = {WorkUnitSpec{
            .name = "sharded_to_interleaved",
            .kernels = std::move(work_unit_kernels),
            .target_nodes = used_cores,
        }},
    };

    // The compute kernel has no runtime args, so it needs no KernelRunArgs entry.
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {{INPUT, input.mesh_tensor()}, {OUTPUT, output.mesh_tensor()}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
