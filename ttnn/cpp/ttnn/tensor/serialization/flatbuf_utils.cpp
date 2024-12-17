#include "buffers/buffer_constants.hpp"
#include "flatbuffers/flatbuffer_builder.h"
#include "tensor_generated.h"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal {
namespace {

flatbuf::Layout to_flatbuf_layout(Layout layout) {
    switch (layout) {
        case Layout::ROW_MAJOR: return flatbuf::Layout::ROW_MAJOR;
        case Layout::TILE: return flatbuf::Layout::TILE;
        case Layout::INVALID: return flatbuf::Layout::INVALID;
    }
}

flatbuf::DataType to_flatbuf_data_type(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return flatbuf::DataType::FLOAT32;
        case DataType::BFLOAT16: return flatbuf::DataType::BFLOAT16;
        case DataType::UINT8: return flatbuf::DataType::UINT8;
        case DataType::UINT16: return flatbuf::DataType::UINT16;
        case DataType::INT32: return flatbuf::DataType::INT32;
        case DataType::UINT32: return flatbuf::DataType::UINT32;
        case DataType::BFLOAT8_B: return flatbuf::DataType::BFLOAT8_B;
        case DataType::BFLOAT4_B: return flatbuf::DataType::BFLOAT4_B;
        case DataType::INVALID: return flatbuf::DataType::INVALID;
    }
}

Layout from_flatbuf_layout(flatbuf::Layout layout) {
    switch (layout) {
        case flatbuf::Layout::ROW_MAJOR: return Layout::ROW_MAJOR;
        case flatbuf::Layout::TILE: return Layout::TILE;
        case flatbuf::Layout::INVALID: return Layout::INVALID;
    }
}

DataType from_flatbuf_data_type(flatbuf::DataType dtype) {
    switch (dtype) {
        case flatbuf::DataType::FLOAT32: return DataType::FLOAT32;
        case flatbuf::DataType::BFLOAT16: return DataType::BFLOAT16;
        case flatbuf::DataType::UINT8: return DataType::UINT8;
        case flatbuf::DataType::UINT16: return DataType::UINT16;
        case flatbuf::DataType::INT32: return DataType::INT32;
        case flatbuf::DataType::UINT32: return DataType::UINT32;
        case flatbuf::DataType::BFLOAT8_B: return DataType::BFLOAT8_B;
        case flatbuf::DataType::BFLOAT4_B: return DataType::BFLOAT4_B;
        case flatbuf::DataType::INVALID: return DataType::INVALID;
    }
}

flatbuf::TensorMemoryLayout to_flatbuf_memory_layout(TensorMemoryLayout layout) {
    switch (layout) {
        case TensorMemoryLayout::INTERLEAVED: return flatbuf::TensorMemoryLayout::Interleaved;
        case TensorMemoryLayout::SINGLE_BANK: return flatbuf::TensorMemoryLayout::SingleBank;
        case TensorMemoryLayout::HEIGHT_SHARDED: return flatbuf::TensorMemoryLayout::HeightSharded;
        case TensorMemoryLayout::WIDTH_SHARDED: return flatbuf::TensorMemoryLayout::WidthSharded;
        case TensorMemoryLayout::BLOCK_SHARDED: return flatbuf::TensorMemoryLayout::BlockSharded;
    }
}

flatbuf::BufferType to_flatbuf_buffer_type(BufferType buffer_type) {
    switch (buffer_type) {
        case BufferType::DRAM: return flatbuf::BufferType::DRAM;
        case BufferType::L1: return flatbuf::BufferType::L1;
        case BufferType::SYSTEM_MEMORY: return flatbuf::BufferType::SYSTEM_MEMORY;
        case BufferType::L1_SMALL: return flatbuf::BufferType::L1_SMALL;
        case BufferType::TRACE: return flatbuf::BufferType::TRACE;
    }
}

}  // namespace

flatbuffers::Offset<flatbuf::Tensor> to_flatbuf_tensor(const Tensor& tensor, flatbuffers::FlatBufferBuilder& builder) {
    std::vector<uint32_t> shape_values(tensor.get_logical_shape().cbegin(), tensor.get_logical_shape().cend());

    auto shape_fb = flatbuf::CreateTensorShape(builder, builder.CreateVector(shape_values));

    const auto& tile_spec = tensor.tensor_spec().page_config().get_tile();
    auto tile_fb = flatbuf::Tile(tile_spec.get_height(), tile_spec.get_width(), tile_spec.get_transpose_within_face());
    auto page_config_fb = flatbuf::CreatePageConfig(builder, to_flatbuf_layout(tensor.layout()), &tile_fb);

    auto memory_config_fb = flatbuf::CreateMemoryConfig(
        builder,
        to_flatbuf_memory_layout(tensor.memory_config().memory_layout),
        to_flatbuf_buffer_type(tensor.memory_config().buffer_type));

    auto tensor_spec_fb = flatbuf::CreateTensorSpec(
        builder, shape_fb, to_flatbuf_data_type(tensor.dtype()), page_config_fb, memory_config_fb);

    // TODO: finish with the tensor data.

    return flatbuf::CreateTensor(builder, tensor_spec_fb);
}

Tensor from_flatbuf_tensor(const flatbuf::Tensor* fb_tensor) {
    // TODO: Implement.
    return Tensor();
}

}  // namespace tt::tt_metal
