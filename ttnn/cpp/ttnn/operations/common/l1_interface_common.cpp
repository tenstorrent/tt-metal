#include "l1_interface_common.hpp"

#include "detail/util.hpp"
#include "impl/buffers/buffer.hpp"
#include "tt_metal/common/constants.hpp"
#include "ttnn/cpp/ttnn/tensor/tensor_impl.hpp"
#include "ttnn/cpp/ttnn/tensor/types.hpp"

uint32_t get_num_of_cores(const std::optional<tt::tt_metal::ShardSpec>& shard_spec) {
    if (shard_spec.has_value()) {
        return shard_spec.value().grid.num_cores();
    }
    return (uint32_t)64;  // Nebula_x1
};

uint32_t get_num_pages(const tt::tt_metal::ShardSpec& shard_spec) {
    return shard_spec.shape[0] * shard_spec.shape[1] / tt::constants::TILE_HW;
}

uint32_t calculate_circular_buffer_l1_allocation_size_per_core(
    const ttnn::types::Shape& shape,
    const tt::tt_metal::DataType data_type,
    const tt::tt_metal::Layout& layout,
    const tt::tt_metal::MemoryConfig& memory_config,
    const uint32_t max_block_size) {
    auto total_size_bytes =
        shape.with_tile_padding().volume() * tt::tt_metal::tensor_impl::element_size_bytes(data_type);
    auto page_size = tt::tt_metal::tensor_impl::get_page_size(data_type, layout, total_size_bytes, shape.value);
    auto num_pages = memory_config.is_sharded() ? get_num_pages(memory_config.shard_spec.value()) : 2 * max_block_size;

    return page_size * num_pages;
}

uint32_t calculate_repeat_circular_buffer_size(tt::tt_metal::DataType data_type) {
    const auto repeat_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(data_type);

    return 2 * tt::tt_metal::detail::TileSize(repeat_cb_data_format);
}

uint32_t calculate_max_block_size(const std::optional<tt::tt_metal::ShardSpec>& shard_spec) {
    if (!shard_spec.has_value()) {
        return (uint32_t)1;
    }

    const uint32_t num_tiles_per_shard = get_num_pages(shard_spec.value());
    const uint32_t max_block_size = 8;

    for (int find_divisor = max_block_size; find_divisor >= 1; find_divisor--) {
        if (num_tiles_per_shard % find_divisor == 0) {
            return find_divisor;
        }
    }

    return (uint32_t)1;
}

uint32_t calculate_tensor_l1_allocation_size_per_core(
    const ttnn::types::Shape& shape,
    const tt::tt_metal::DataType data_type,
    const tt::tt_metal::Layout& layout,
    const tt::tt_metal::MemoryConfig& memory_config) {
    if (memory_config.is_l1()) {
        if (memory_config.is_sharded()) {
            tt::tt_metal::ShardSpecBuffer
                shard_spec_buffer(  // this structure is not used inside validate_sharded_buffer_allocation
                                    // assembling it with data from memory_config
                    memory_config.shard_spec.value().grid,
                    memory_config.shard_spec.value().shape,
                    memory_config.shard_spec.value().orientation,
                    memory_config.shard_spec.value().halo,
                    {32, 32},  //
                    {32, 32}   //
                );
            tt::tt_metal::tensor_impl::validate_sharded_buffer_allocation(
                shape.value, layout, data_type, shard_spec_buffer, memory_config);

            auto total_size_bytes =
                shape.with_tile_padding().volume() * tt::tt_metal::tensor_impl::element_size_bytes(data_type);
            auto num_of_cores = memory_config.shard_spec.value().grid.num_cores();
            return total_size_bytes / num_of_cores;
        } else {
            //          Banks are ½ size of L1 (732KB)
            auto total_size_bytes =
                shape.with_tile_padding().volume() * tt::tt_metal::tensor_impl::element_size_bytes(data_type);
            auto num_of_cores = 64;  // Nebula_x1
            return total_size_bytes / num_of_cores;
        }
    }
    return (uint32_t)0;  // dram not implemented yet
}

bool is_sharded(const L1InterfaceOperandParams& operand) {
    return std::get<tt::tt_metal::MemoryConfig>(operand).is_sharded();
}

uint32_t get_tile_size(const L1InterfaceOperandParams& operand) {
    return tt::tt_metal::detail::TileSize(
        tt::tt_metal::datatype_to_dataformat_converter(std::get<tt::tt_metal::DataType>(operand)));
}

tt::tt_metal::Shape get_legacy_shape(const L1InterfaceOperandParams& operand) {
    return std::get<ttnn::Shape>(operand).value;
}

bool has_layout(const L1InterfaceOperandParams& operand, TensorMemoryLayout layout) {
    return std::get<tt::tt_metal::MemoryConfig>(operand).memory_layout == layout;
}

std::optional<tt::tt_metal::ShardSpec> get_shard_spec(const L1InterfaceOperandParams& operand) {
    return std::get<tt::tt_metal::MemoryConfig>(operand).shard_spec;
}
