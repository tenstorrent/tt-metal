// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"

namespace tt::tt_metal {

class TensorSpec final {
public:
    TensorSpec(ttnn::SimpleShape logical_shape, TensorLayout tensor_layout): logical_shape_(std::move(logical_shape)), tensor_layout_(std::move(tensor_layout)) {}
    TensorSpec(TensorSpec&&) noexcept = default;
    TensorSpec& operator=(TensorSpec&&) = default;
    TensorSpec(const TensorSpec&) = default;
    TensorSpec& operator=(const TensorSpec&) = default;

    const ttnn::SimpleShape& logical_shape() const {
        return logical_shape_;
    }
    const TensorLayout& tensor_layout() const {
        return tensor_layout_;
    }
    ttnn::SimpleShape compute_padded_shape() const {
        return tensor_layout_.compute_padded_shape(logical_shape_);
    }
    ttnn::Shape compute_shape() const {
        return ttnn::Shape(logical_shape_.view(), compute_padded_shape().view());
    }

    Strides compute_strides() const {
        return tensor_layout_.compute_strides(logical_shape_);
    }
    std::optional<ShardSpecBuffer> compute_shard_spec_buffer() const {
        return tensor_layout_.compute_shard_spec_buffer(logical_shape_);
    }
    size_t compute_packed_buffer_size_bytes() const {
        return tensor_layout_.compute_packed_buffer_size_bytes(logical_shape_);
    }
    size_t compute_page_size_bytes() const {
        return tensor_layout_.compute_page_size_bytes(logical_shape_);
    }
    Size compute_physical_shape() const {
        return tensor_layout_.compute_physical_shape(logical_shape_);
    }

    static constexpr auto attribute_names = std::forward_as_tuple("logical_shape", "tensor_layout");
    const auto attribute_values() const {
        return std::forward_as_tuple(logical_shape_, tensor_layout_);
    }

private:
    ttnn::SimpleShape logical_shape_;
    TensorLayout tensor_layout_;
};

}
