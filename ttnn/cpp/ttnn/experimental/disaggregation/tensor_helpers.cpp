// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/experimental/disaggregation/tensor_helpers.hpp"

#include <cstring>
#include <utility>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/shape.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>

#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::experimental_disaggregation {

ttnn::Tensor tensor_from_bfp8_bytes(std::span<const uint8_t> raw_bytes, const std::vector<uint32_t>& shape) {
    TT_FATAL(
        raw_bytes.size() % sizeof(uint32_t) == 0,
        "tensor_from_bfp8_bytes: raw byte size {} is not a multiple of 4 (bfp8 storage is uint32-packed)",
        raw_bytes.size());

    size_t n_words = raw_bytes.size() / sizeof(uint32_t);
    std::vector<uint32_t> packed(n_words);
    std::memcpy(packed.data(), raw_bytes.data(), raw_bytes.size());

    tt::tt_metal::Shape tensor_shape(ttsl::Span<const uint32_t>(shape.data(), shape.size()));
    tt::tt_metal::TensorLayout layout(
        tt::tt_metal::DataType::BFLOAT8_B,
        tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
        tt::tt_metal::MemoryConfig{});
    tt::tt_metal::TensorSpec spec(tensor_shape, layout);

    tt::tt_metal::HostBuffer host_buffer(std::move(packed));
    return ttnn::Tensor(std::move(host_buffer), std::move(spec));
}

ttnn::Tensor tensor_from_bf16_bytes(std::span<const uint8_t> raw_bytes, const std::vector<uint32_t>& shape) {
    TT_FATAL(
        raw_bytes.size() % sizeof(bfloat16) == 0,
        "tensor_from_bf16_bytes: raw byte size {} is not a multiple of {} (bfloat16 is 2 bytes)",
        raw_bytes.size(),
        sizeof(bfloat16));

    size_t n_elems = raw_bytes.size() / sizeof(bfloat16);
    size_t shape_volume = 1;
    for (uint32_t dim : shape) {
        shape_volume *= dim;
    }
    TT_FATAL(
        n_elems == shape_volume,
        "tensor_from_bf16_bytes: {} bf16 elements decoded from {} raw bytes do not match the requested "
        "shape's volume {} — the shape/byte count are inconsistent",
        n_elems,
        raw_bytes.size(),
        shape_volume);

    std::vector<bfloat16> data(n_elems);
    std::memcpy(data.data(), raw_bytes.data(), raw_bytes.size());

    tt::tt_metal::Shape tensor_shape(ttsl::Span<const uint32_t>(shape.data(), shape.size()));
    tt::tt_metal::TensorLayout layout(
        tt::tt_metal::DataType::BFLOAT16,
        tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
        tt::tt_metal::MemoryConfig{});
    tt::tt_metal::TensorSpec spec(tensor_shape, layout);

    tt::tt_metal::HostBuffer host_buffer(std::move(data));
    return ttnn::Tensor(std::move(host_buffer), std::move(spec));
}

}  // namespace ttnn::experimental_disaggregation
