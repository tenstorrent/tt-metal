// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "narrow.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::data_movement {

ttnn::Tensor NarrowOperation::invoke(
    const ttnn::Tensor& input_tensor, const int32_t narrow_dim, const int32_t narrow_start, const uint32_t length) {
    auto input_tensor_shape = input_tensor.padded_shape();
    uint32_t dim = input_tensor_shape.get_normalized_index(narrow_dim);
    uint32_t start = wrap_index(narrow_start, input_tensor_shape[dim]);

    // Early return if no narrowing needed
    if (start == 0 && length == input_tensor_shape[dim]) {
        return input_tensor;
    }

    // Validate input parameters
    TT_FATAL(
        start < input_tensor_shape[dim],
        "Narrow start index {} out of bounds for dimension {} with size {}",
        start,
        dim,
        input_tensor_shape[dim]);
    TT_FATAL(
        length > 0 && start + length <= input_tensor_shape[dim],
        "Narrow length {} out of bounds for dimension {} with start index {} and size {}",
        length,
        dim,
        start,
        input_tensor_shape[dim]);

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "NarrowOperation currently only supports DEVICE tensors, got {}",
        input_tensor.storage_type());

    auto output_tensor_shape = input_tensor.logical_shape();
    output_tensor_shape[dim] = length;
    bool narrow_width = dim == input_tensor_shape.size() - 1;
    bool narrow_height = dim == input_tensor_shape.size() - 2;
    tt::tt_metal::DeviceStorage storage = std::get<tt::tt_metal::DeviceStorage>(input_tensor.storage());
    Buffer* buffer = storage.get_buffer();

    TT_FATAL(
        input_tensor.layout() != ttnn::TILE_LAYOUT || !narrow_width ||
            (start % tt::constants::TILE_WIDTH == 0 && length % tt::constants::TILE_WIDTH == 0),
        "Narrowing on width for TILE layout requires start and length to be multiples of TILE_WIDTH ({}), "
        "got start {} and length {}",
        tt::constants::TILE_WIDTH,
        start,
        length);
    TT_FATAL(
        input_tensor.layout() != ttnn::TILE_LAYOUT || !narrow_height ||
            (start % tt::constants::TILE_HEIGHT == 0 && length % tt::constants::TILE_HEIGHT == 0),
        "Narrowing on height for TILE layout requires start and length to be multiples of TILE_HEIGHT ({}), "
        "got start {} and length {}",
        tt::constants::TILE_HEIGHT,
        start,
        length);
    TT_FATAL(
        input_tensor.layout() != ttnn::ROW_MAJOR_LAYOUT || !narrow_width || start % buffer->alignment() == 0,
        "Narrowing on width for ROW_MAJOR layout requires start ({}) to be aligned to buffer alignment ({})",
        start,
        buffer->alignment());

    // Update global config for narrowed tensor
    // Note: ShardedBufferConfig variant is not supported due to lack of examples
    TT_FATAL(
        storage.mesh_buffer->global_layout() == tt::tt_metal::distributed::MeshBufferLayout::REPLICATED,
        "NarrowOperation currently only supports REPLICATED global layouts");

    tt::tt_metal::distributed::ReplicatedBufferConfig replicated_config =
        std::get<tt::tt_metal::distributed::ReplicatedBufferConfig>(storage.mesh_buffer->global_config());
    uint32_t reduction_factor = input_tensor_shape[dim] / length;
    replicated_config.size /= reduction_factor;
    tt::tt_metal::distributed::MeshBufferConfig narrowed_global_config = replicated_config;

    // Handle INTERLEAVED DRAM buffers
    if (input_tensor.memory_config().buffer_type() == ttnn::BufferType::DRAM &&
        input_tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED) {
        // For DRAM interleaved, narrowing is only supported on the first non-trivial dimension
        // (dimensions with size > 1). All preceding dimensions must be of size 1.
        bool all_preceding_dims_trivial = true;
        for (uint32_t i = 0; i < dim; ++i) {
            if (input_tensor_shape[i] != 1) {
                all_preceding_dims_trivial = false;
                break;
            }
        }
        TT_FATAL(
            all_preceding_dims_trivial,
            "Narrowing on dimension {} for DRAM INTERLEAVED tensors requires all preceding dimensions to be trivial "
            "(size 1). Got non-trivial dimensions before dimension {}.",
            dim,
            dim);

        // Compute internal block size (product of dimensions after narrowing dimension)
        uint64_t elements_per_block = 1;
        for (uint32_t i = dim + 1; i < input_tensor_shape.size(); ++i) {
            elements_per_block *= input_tensor_shape[i];
        }

        uint64_t element_size_bytes = input_tensor.element_size();
        uint32_t num_banks = buffer->allocator()->get_num_banks(buffer->buffer_type());

        // Calculate starting page ID and verify bank alignment
        uint64_t logical_page_size = (input_tensor.layout() == ttnn::TILE_LAYOUT)
                                         ? tt::constants::TILE_HW * element_size_bytes
                                         : buffer->page_size();
        uint64_t start_page_id = start * element_size_bytes * elements_per_block / logical_page_size;

        TT_FATAL(
            start_page_id % num_banks == 0,
            "Start page ID must be aligned with number of banks for INTERLEAVED DRAM buffer");

        uint64_t offset_bytes = (start_page_id / num_banks) * buffer->aligned_page_size();
        auto device_local_config = storage.mesh_buffer->device_local_config();

        auto subtensor_mesh = tt::tt_metal::distributed::MeshBuffer::create(
            narrowed_global_config,
            device_local_config,
            storage.mesh_buffer->device(),
            storage.mesh_buffer->address() + offset_bytes);

        tt::tt_metal::DeviceStorage subtensor_storage(subtensor_mesh, storage.coords);
        TensorSpec subtensor_spec = TensorSpec(
            output_tensor_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.dtype(),
                input_tensor.tensor_spec().page_config(),
                input_tensor.tensor_spec().memory_config()));

        return Tensor(subtensor_storage, subtensor_spec, input_tensor.tensor_topology());
    }

    // Handle sharded L1 buffers
    if (input_tensor.memory_config().buffer_type() == ttnn::BufferType::L1 && input_tensor.is_sharded()) {
        // Compute internal block size for non-last dimensions
        uint64_t elements_per_row = 1;
        for (uint32_t i = dim + 1; i < input_tensor_shape.size() - 1; ++i) {
            elements_per_row *= input_tensor_shape[i];
        }

        auto current_shard_shape = input_tensor.shard_spec()->shape;
        auto narrowed_shard_shape = current_shard_shape;
        uint32_t page_offset = 0;

        // Extract device-local configuration early to get page_shape
        auto device_local_config = storage.mesh_buffer->device_local_config();
        auto& sharding_args = device_local_config.sharding_args;
        auto shard_spec_buffer = sharding_args.shard_spec().value();
        auto page_shape = shard_spec_buffer.page_shape;

        // Process narrowing based on dimension
        if (narrow_width) {
            // Narrowing on last dimension
            if (current_shard_shape[0] == page_shape[0] &&
                (start % current_shard_shape[1] + length) <= current_shard_shape[1]) {
                // Splitting shard on last dimension
                narrowed_shard_shape[1] = length;
                page_offset = (start % current_shard_shape[1]) / page_shape[1];
            } else if (start % current_shard_shape[1] == 0 && length % current_shard_shape[1] == 0) {
                // Using full shards on last dimension (no-op for shard shape)
            } else {
                TT_FATAL(
                    false,
                    "Narrowing on last dimension for L1 sharded tensor only supports full shards or splitting "
                    "within single shard. Got start {} and length {} with shard shape ({}, {})",
                    start,
                    length,
                    current_shard_shape[0],
                    current_shard_shape[1]);
            }
        } else {
            // Narrowing on non-last dimension
            uint64_t start_offset = start * elements_per_row;
            uint64_t length_offset = length * elements_per_row;

            if ((start_offset % current_shard_shape[0] + length_offset) <= current_shard_shape[0]) {
                // Splitting shard on non-last dimension
                narrowed_shard_shape[0] = length_offset;
                page_offset =
                    (start_offset % current_shard_shape[0]) / page_shape[0] * current_shard_shape[1] / page_shape[1];
            } else if (start_offset % current_shard_shape[0] == 0 && length_offset % current_shard_shape[0] == 0) {
                // Using full shards on non-last dimension (no-op for shard shape)
            } else {
                TT_FATAL(
                    false,
                    "Narrowing on non-last dimension for L1 sharded tensor only supports full shards or splitting "
                    "within single shard. Got start {} and length {} with shard shape ({}, {})",
                    start,
                    length,
                    current_shard_shape[0],
                    current_shard_shape[1]);
            }
        }

        // Extract device-local configuration
        auto buffer_distribution_spec = sharding_args.buffer_distribution_spec().value();
        const auto& cores_with_data = buffer_distribution_spec.cores_with_data();
        auto tensor_pages_shape = shard_spec_buffer.tensor2d_shape_in_pages;
        auto shard_pages_shape = shard_spec_buffer.shape_in_pages();

        uint32_t grid_width = tensor_pages_shape[1] / shard_pages_shape[1];

        // Filter cores that contain needed pages
        std::vector<CoreCoord> filtered_cores;
        filtered_cores.reserve(cores_with_data.size());

        for (uint32_t core_id = 0; core_id < cores_with_data.size(); ++core_id) {
            bool is_in_range = false;

            if (narrow_width) {
                // Last dimension narrowing: check page ID based on width
                uint32_t core_width = core_id % grid_width;
                uint64_t page_id = (core_width * current_shard_shape[1] / page_shape[1]) + page_offset;
                is_in_range = (page_id >= (start / page_shape[1])) && (page_id < ((start + length) / page_shape[1]));
            } else {
                // Non-last dimension narrowing: check page ID based on height
                uint32_t core_height = core_id / grid_width;
                uint64_t page_id = (core_height * current_shard_shape[0] / page_shape[0] +
                                    page_offset / (current_shard_shape[1] / page_shape[1])) %
                                   (input_tensor_shape[dim] * elements_per_row / page_shape[0]);
                uint64_t start_page = start * elements_per_row / page_shape[0];
                uint64_t end_page = (start + length) * elements_per_row / page_shape[0];
                is_in_range = (page_id >= start_page) && (page_id < end_page);
            }

            if (is_in_range) {
                filtered_cores.push_back(cores_with_data[core_id]);
            }
        }

        // Create new core grid
        CoreRangeSet new_core_grid;
        if (shard_spec_buffer.orientation() == ShardOrientation::ROW_MAJOR) {
            new_core_grid = CoreRangeSet(filtered_cores);
        } else {
            std::vector<CoreRange> core_ranges;
            core_ranges.reserve(filtered_cores.size());
            for (const auto& core : filtered_cores) {
                core_ranges.push_back(CoreRange(core));
            }
            new_core_grid = CoreRangeSet(core_ranges);
        }

        // Update tensor shape in pages
        auto narrowed_pages_shape = tensor_pages_shape;
        if (narrow_width) {
            narrowed_pages_shape[1] /= reduction_factor;
        } else {
            narrowed_pages_shape[0] /= reduction_factor;
        }

        // Create new shard specifications
        tt::tt_metal::ShardSpec narrowed_shard_spec(new_core_grid, narrowed_shard_shape, ShardOrientation::ROW_MAJOR);
        tt::tt_metal::ShardSpecBuffer narrowed_shard_spec_buffer =
            tt::tt_metal::ShardSpecBuffer(narrowed_shard_spec, shard_spec_buffer.page_shape, narrowed_pages_shape);

        tt::tt_metal::Shape tensor_shape_pages(narrowed_pages_shape);
        tt::tt_metal::Shape shard_shape_pages(narrowed_shard_spec_buffer.shape_in_pages());
        tt::tt_metal::BufferDistributionSpec narrowed_buffer_dist_spec =
            tt::tt_metal::BufferDistributionSpec(tensor_shape_pages, shard_shape_pages, filtered_cores);

        tt::tt_metal::BufferShardingArgs narrowed_sharding_args(
            narrowed_buffer_dist_spec, narrowed_shard_spec_buffer, sharding_args.buffer_layout());

        // Create new device-local configuration
        tt::tt_metal::distributed::DeviceLocalBufferConfig narrowed_device_config = {
            .page_size = device_local_config.page_size,
            .buffer_type = device_local_config.buffer_type,
            .sharding_args = narrowed_sharding_args,
            .bottom_up = device_local_config.bottom_up};

        auto subtensor_mesh = tt::tt_metal::distributed::MeshBuffer::create(
            narrowed_global_config,
            narrowed_device_config,
            storage.mesh_buffer->device(),
            storage.mesh_buffer->address() + (page_offset * buffer->aligned_page_size()));

        tt::tt_metal::DeviceStorage subtensor_storage(subtensor_mesh, storage.coords);

        auto narrowed_memory_config =
            MemoryConfig(input_tensor.memory_config().memory_layout(), BufferType::L1, narrowed_shard_spec);

        TensorSpec subtensor_spec = TensorSpec(
            output_tensor_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.dtype(), input_tensor.tensor_spec().page_config(), narrowed_memory_config));

        return Tensor(subtensor_storage, subtensor_spec, input_tensor.tensor_topology());
    }

    // Unsupported tensor configuration
    TT_FATAL(
        false,
        "Narrowing for the given tensor configuration is not yet implemented."
        "Supported: DRAM INTERLEAVED tensors on first non-trivial dimension, L1 sharded tensors with restrictions.");
}

}  // namespace ttnn::operations::data_movement
