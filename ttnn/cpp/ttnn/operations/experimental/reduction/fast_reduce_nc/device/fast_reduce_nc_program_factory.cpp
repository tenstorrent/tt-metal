// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reduction/fast_reduce_nc/device/fast_reduce_nc_program_factory.hpp"

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

bool is_tensor_divisible_by_shard(const ttnn::Shape& tensor_shape, const ttnn::Shape& shard_shape) {
    // Only compare common (end) dimensions. Any extra front dimensions would be
    // divisible by the implied 1 in the non-existent dimensions in shard_shape.
    // Use negative dimensions to compare the end of both shapes.
    for (int i = 1; i <= shard_shape.size(); i++) {
        if (shard_shape[-i] == 0 || tensor_shape[-i] % shard_shape[-i] != 0) {
            return false;
        }
    }
    return true;
}
std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> extract_and_scale_spatial_dims(
    const ttnn::Shape& shape, uint32_t dim) {
    const auto rank = shape.rank();

    TT_FATAL(rank >= 2, "Shape must have at least two dims.");
    uint32_t Wt = shape[-1] / TILE_WIDTH;
    uint32_t Ht = shape[-2] / TILE_HEIGHT;

    uint32_t reduce_dim = shape[dim];
    uint32_t inner_dims_product = 1;
    for (auto i = dim + 1; i < rank - 2; ++i) {
        inner_dims_product *= shape[i];
    }

    uint32_t inner_tile_size = inner_dims_product * Ht * Wt;
    uint32_t reduce_tile_size = reduce_dim * inner_tile_size;

    return {Wt, Ht, inner_tile_size, reduce_tile_size};
}

}  // namespace

tt::tt_metal::ProgramDescriptor FastReduceNCProgramFactory::create_descriptor(
    const FastReduceNCParams& operation_attributes,
    const FastReduceNCInputs& tensor_args,
    Tensor& tensor_return_value) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto* device = tensor_args.input.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // Input and output CBs may differ when the Sum precision chain requests FP32 packing.
    const auto input_data_format = datatype_to_dataformat_converter(tensor_args.input.dtype());
    const auto input_tile_size = tt::tile_size(input_data_format);
    const auto output_data_format = datatype_to_dataformat_converter(tensor_return_value.dtype());
    const auto output_tile_size = tt::tile_size(output_data_format);
    const auto cb_1_data_format = datatype_to_dataformat_converter(DataType::BFLOAT16);
    const auto cb_1_tile_size = tt::tile_size(cb_1_data_format);

    const auto& input_shape = tensor_args.input.padded_shape();
    const auto [Wt, Ht, inner_tile_size, reduce_tile_size] =
        extract_and_scale_spatial_dims(input_shape, static_cast<uint32_t>(operation_attributes.dim));
    const auto num_reduce_input_tile = input_shape[operation_attributes.dim];
    const auto num_output_tiles = tensor_return_value.physical_volume() / TILE_HW;
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(tensor_args.input.device()->arch(), operation_attributes.compute_kernel_config);
    // Choose granularity as the largest factor of num_reduce_input_tile that is less than or equal to 8.
    // Helps with locality and increases work unit for better performance.
    uint32_t input_granularity;
    for (input_granularity = 8; input_granularity > 1; --input_granularity) {
        if (num_reduce_input_tile % input_granularity == 0) {
            break;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_x = grid.x;
    const auto num_cores_y = grid.y;

    const uint32_t in0_t = input_granularity * 2;  // input
    const uint32_t in1_t = 1;                      // zero
    const uint32_t intermed0_t = 1;                // accumulated sum
    const uint32_t out0_t = 2;                     // output
    uint32_t shard_factor = 1;

    // When dim=0, nd sharded, tile sizes are the same, the shards are compatible
    // with the kernel accesses, tensor shape is divisible by shard shape, and
    // number of shards is larger than core count, divide the work by shards.
    uint32_t output_shard_size = 1;
    auto input_tile = tensor_args.input.tensor_spec().tile().get_tile_shape();
    auto output_tile = tensor_return_value.tensor_spec().tile().get_tile_shape();
    bool nd_sharded = tensor_args.input.nd_shard_spec().has_value() && tensor_return_value.nd_shard_spec().has_value();
    bool same_tiles = input_tile[0] == output_tile[0] && input_tile[1] == output_tile[1];
    bool divide_by_shards = false;
    const auto& dspec = *tensor_return_value.buffer()->buffer_distribution_spec();
    if (nd_sharded && same_tiles && operation_attributes.dim == 0) {
        const NdShardSpec& input_nd_shard_spec = tensor_args.input.nd_shard_spec().value();
        const NdShardSpec& output_nd_shard_spec = tensor_return_value.nd_shard_spec().value();
        const Shape& input_shard_shape = input_nd_shard_spec.shard_shape;
        bool compatible_shards =
            input_nd_shard_spec.orientation == ShardOrientation::ROW_MAJOR &&
            input_nd_shard_spec.shard_distribution_strategy == ShardDistributionStrategy::ROUND_ROBIN_1D &&
            output_nd_shard_spec.orientation == ShardOrientation::ROW_MAJOR &&
            output_nd_shard_spec.shard_distribution_strategy == ShardDistributionStrategy::ROUND_ROBIN_1D;
        if (compatible_shards && is_tensor_divisible_by_shard(input_shape, input_shard_shape)) {
            uint32_t num_output_shards = dspec.num_shards();
            output_shard_size = dspec.shard_shape_in_pages().volume();
            bool more_shards_than_cores = num_output_shards > (num_cores_x * num_cores_y);
            if (more_shards_than_cores) {
                divide_by_shards = true;
                shard_factor = output_shard_size;
            }
        }
    }
    bool use_sub_core_grids = operation_attributes.sub_core_grids.has_value() && !divide_by_shards;
    auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_cols_per_core_group_1,
         num_cols_per_core_group_2] =
            divide_by_shards
                ? dspec.core_groups_tuple()
                : (use_sub_core_grids
                       ? tt::tt_metal::split_work_to_cores(
                             *operation_attributes.sub_core_grids, num_output_tiles, /*row_wise=*/true)
                       : tt::tt_metal::split_work_to_cores(grid, num_output_tiles, /*row_wise=*/true));
    num_cols_per_core_group_1 *= shard_factor;
    num_cols_per_core_group_2 *= shard_factor;

    const auto intermed_cb_data_format = (fp32_dest_acc_en) ? tt::DataFormat::Float32 : output_data_format;
    const auto intermed_cb_single_tile_size = tt::tile_size(intermed_cb_data_format);

    ProgramDescriptor desc;

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    desc.cbs.push_back(CBDescriptor{
        .total_size = in0_t * input_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_0),
            .data_format = input_data_format,
            .page_size = input_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = in1_t * cb_1_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_1),
            .data_format = cb_1_data_format,
            .page_size = cb_1_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = intermed0_t * intermed_cb_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_24),
            .data_format = intermed_cb_data_format,
            .page_size = intermed_cb_single_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = out0_t * output_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_16),
            .data_format = output_data_format,
            .page_size = output_tile_size,
        }}},
    });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> reader_compile_time_args = {input_granularity, shard_factor, num_cores_to_be_used};
    TensorAccessorArgs(*tensor_args.input.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {shard_factor, num_cores_to_be_used};
    TensorAccessorArgs(*tensor_return_value.buffer()).append_to(writer_compile_time_args);

    const auto* const reader_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/kernels/reader_reduce_nc.cpp";
    const auto* const writer_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/kernels/writer_reduce_nc.cpp";

    KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source = reader_kernel_file;
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = all_cores;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_kernel_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source = writer_kernel_file;
    writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = all_cores;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_kernel_desc.config = WriterConfigDescriptor{};

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> compute_args_group_1 = {
        num_cols_per_core_group_1, num_reduce_input_tile, input_granularity};
    KernelDescriptor::Defines compute_defines;
    if (fp32_dest_acc_en) {
        compute_defines.emplace_back("FP32_DEST_ACC_EN", "1");
    }
    const auto* const compute_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/kernels/reduce_nc.cpp";

    KernelDescriptor compute_kernel_1_desc;
    compute_kernel_1_desc.kernel_source = compute_kernel_file;
    compute_kernel_1_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_1_desc.core_ranges = core_group_1;
    compute_kernel_1_desc.compile_time_args = compute_args_group_1;
    compute_kernel_1_desc.defines = compute_defines;
    compute_kernel_1_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    std::optional<KernelDescriptor> compute_kernel_2_desc;
    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2 = {
            num_cols_per_core_group_2, num_reduce_input_tile, input_granularity};
        KernelDescriptor k2;
        k2.kernel_source = compute_kernel_file;
        k2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        k2.core_ranges = core_group_2;
        k2.compile_time_args = compute_args_group_2;
        k2.defines = compute_defines;
        k2.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
        };
        compute_kernel_2_desc = std::move(k2);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    // Each core is assigned an output work unit in a row wise round robin
    // fashion. For a given core, the first index is i, and all subsequent
    // indices are increments of num_cores_to_be_used. The total number of
    // units is num_tiles_per_group times num_cores_to_be_used.
    // For example, with 130 output tiles to be processed and no shards (shard
    // factor is 1) on an 8x8 grid
    // - the increment is 64
    // - the first 2 cores will have num_tiles_per_core 3 and the rest 2
    // - core x=0,y=0 will process output tiles 0, 64, and 128
    // - core x=1,y=0 will process output tiles 1, 65, and 129
    // - core x=2,y=0 will process output tiles 2 and 66
    // - core x=3,y=0 will process output tiles 3 and 67
    // - etc
    // The first tile that needs to be reduced has the same as the output tile.
    // That is the starting point for the reader, which then processes all
    // subsequent tiles to be reduced. The increment for the input indices is
    // the size of the inner dimensions in tiles (inner_tile_size). The number
    // of tiles to process is the size of the reduce dimension in tiles
    // (reduce_tile_size).
    // The shard factor is used to iterate over shards instead of tiles.
    // It is taken into account in the num_cols_per_core_group variables and
    // the tile_offset is incremented by it for the reader to adjust it's
    // reading pattern.
    std::vector<CoreCoord> ordered_cores;
    if (use_sub_core_grids) {
        for (const auto& range : all_cores.ranges()) {
            for (auto y = range.start_coord.y; y <= range.end_coord.y; ++y) {
                for (auto x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                    ordered_cores.emplace_back(x, y);
                }
            }
        }
    } else {
        for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
            ordered_cores.emplace_back(i % num_cores_x, i / num_cores_x);
        }
    }

    auto* const input_buffer = tensor_args.input.buffer();
    auto* const output_buffer = tensor_return_value.buffer();

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = ordered_cores[i];

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        reader_kernel_desc.emplace_runtime_args(
            core,
            {input_buffer,
             num_reduce_input_tile,
             /*id_range_length=*/num_tiles_per_core * num_cores_to_be_used,
             tile_offset,
             static_cast<uint32_t>(operation_attributes.dim),
             reduce_tile_size,
             inner_tile_size});

        writer_kernel_desc.emplace_runtime_args(
            core,
            {output_buffer,
             /*id_range_length=*/num_tiles_per_core * num_cores_to_be_used,
             tile_offset});

        tile_offset += shard_factor;
    }

    desc.kernels.push_back(std::move(reader_kernel_desc));
    desc.kernels.push_back(std::move(writer_kernel_desc));
    desc.kernels.push_back(std::move(compute_kernel_1_desc));
    if (compute_kernel_2_desc.has_value()) {
        desc.kernels.push_back(std::move(*compute_kernel_2_desc));
    }

    return desc;
}

}  // namespace ttnn::experimental::prim
