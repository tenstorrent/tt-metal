#include "flatbuffers/flatbuffer_builder.h"
#include "tensor_generated.h"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal {

flatbuffers::Offset<flatbuf::Tensor> to_flatbuf_tensor(const Tensor& tensor, flatbuffers::FlatBufferBuilder& builder);

Tensor from_flatbuf_tensor(const flatbuf::Tensor& tensor);

}  // namespace tt::tt_metal
