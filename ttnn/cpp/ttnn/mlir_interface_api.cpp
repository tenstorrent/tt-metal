#include "mlir_interface_api.hpp"
#include "types_str_wrapper.hpp"

#include "tensor/types.hpp" // DataType, Lauout, StorageType
#include "tt_metal/impl/buffers/buffer_constants.hpp" // TensorMemoryLayout, ShardOrientation
#include "tt_metal/impl/buffers/buffer.hpp" // BufferType

namespace ttnn::mlir_interface
{

int add(int a, int b) {
    return a + b;
}

int subtract(int a, int b) {
    return a - b;
}

// check if layout is dram interleaved or l1 sharded, returns false otherwise
bool dummy_check(const std::string& tensor_memory_layout_str, const std::string& buffer_type_str) {
    auto tensor_memory_layout = ttnn::str_wrapper::str_to_memory_layout(tensor_memory_layout_str);
    auto buffer_type = ttnn::str_wrapper::str_to_buffer_type(buffer_type_str);
    if (!tensor_memory_layout.has_value() || !buffer_type.has_value()) {
        return false;
    }

    if (tensor_memory_layout.value() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED && buffer_type.value() == tt::tt_metal::BufferType::DRAM) {
        return true;
    } else if (tensor_memory_layout.value() == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED && buffer_type.value() == tt::tt_metal::BufferType::L1) {
        return true;
    } else if (tensor_memory_layout.value() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED && buffer_type.value() == tt::tt_metal::BufferType::L1) {
        return true;
    } else if (tensor_memory_layout.value() == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED && buffer_type.value() == tt::tt_metal::BufferType::L1) {
        return true;
    }

    return false;
}

} // namespace ttnn_mlir_interface
