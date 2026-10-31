// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/dynamic_length.hpp"

#include <tt_stl/assert.hpp>

namespace ttnn::dynamic_length {

using tt::tt_metal::DataType;

SplitLength split(const ValidLength& length, std::size_t capacity) {
    if (const auto* scalar = std::get_if<std::size_t>(&length)) {
        return {*scalar, std::nullopt};
    }
    return {capacity, std::get<Tensor>(length)};
}

void validate_axes(std::span<const DynamicAxis> axes, std::string_view operation_name) {
    for (const auto& axis : axes) {
        TT_FATAL(axis.alignment > 0, "{}: {} alignment must be positive", operation_name, axis.name);
        if (axis.tensor != nullptr) {
            const auto& t = *axis.tensor;
            TT_FATAL(
                t.dtype() == DataType::UINT32 || t.dtype() == DataType::INT32,
                "{}: {} tensor must be UINT32 or INT32 (the kernels read element 0 as a raw 32-bit word). Got {}",
                operation_name,
                axis.name,
                t.dtype());
            TT_FATAL(
                t.storage_type() == StorageType::DEVICE, "{}: {} tensor must be on device", operation_name, axis.name);
            TT_FATAL(t.is_allocated(), "{}: {} tensor must be allocated on device", operation_name, axis.name);
            TT_FATAL(
                t.logical_volume() == 1,
                "{}: {} tensor must hold exactly one value (the kernels read page 0, element 0). Got volume {}",
                operation_name,
                axis.name,
                t.logical_volume());
            // The word stored in the tensor is not read. The host placeholder (`value`) is still bounded, so a
            // direct caller cannot pair an oversized scalar with a tensor and have the factory size past capacity.
        }
        TT_FATAL(
            axis.value <= axis.capacity,
            "{}: {} ({}) exceeds the capacity ({}) the program is sized for. Allocate the inputs at a larger "
            "capacity",
            operation_name,
            axis.name,
            axis.value,
            axis.capacity);
        TT_FATAL(
            axis.value % axis.alignment == 0,
            "{}: {} ({}) must be a multiple of {}",
            operation_name,
            axis.name,
            axis.value,
            axis.alignment);
    }
}

DynamicKey& DynamicKey::dim(const Tensor& tensor, int dim) {
    dims_.push_back({&tensor, dim});
    return *this;
}

DynamicKey& DynamicKey::dim(const std::optional<Tensor>& tensor, int dim) {
    return tensor.has_value() ? this->dim(*tensor, dim) : *this;
}

namespace detail {

KeyEncoder::KeyEncoder(const DynamicKey& dynamic_key, std::string_view operation_name) :
    dynamic_key_(dynamic_key),
    operation_name_(operation_name),
    dim_matched_(dynamic_key.dims_.size(), false),
    value_matched_(dynamic_key.values_.size(), false) {}

bool KeyEncoder::is_left_out(const void* address, int type) {
    bool left_out = false;
    for (std::size_t i = 0; i < dynamic_key_.values_.size(); ++i) {
        if (dynamic_key_.values_[i].address == address && dynamic_key_.values_[i].type == type) {
            value_matched_[i] = true;
            left_out = true;
        }
    }
    return left_out;
}

// Mirrors the Tensor's own key (storage kind, logical shape, layout), with a declared dim encoded as absent.
// The padded shape follows from the logical shape and the layout's alignment, so it is left out as well.
void KeyEncoder::append_count(std::uint64_t count) {
    key_.append(reinterpret_cast<const char*>(&count), sizeof(count));
}

void KeyEncoder::append_tensor(const Tensor& tensor) {
    const auto& shape = tensor.logical_shape();
    const int rank = static_cast<int>(shape.rank());
    std::vector<std::optional<std::uint32_t>> extents(shape.cbegin(), shape.cend());
    for (std::size_t i = 0; i < dynamic_key_.dims_.size(); ++i) {
        const auto& declared = dynamic_key_.dims_[i];
        if (declared.tensor != &tensor) {
            continue;
        }
        const int dim = declared.dim < 0 ? declared.dim + rank : declared.dim;
        TT_FATAL(
            dim >= 0 && dim < rank,
            "{}: dynamic dim {} is out of range for a rank-{} tensor",
            operation_name_,
            declared.dim,
            rank);
        TT_FATAL(
            !tensor.memory_config().is_sharded(),
            "{}: dynamic dim {} is declared on a sharded tensor, whose shard spec still carries the extent",
            operation_name_,
            declared.dim);
        // A row-major row is the page. Dropping that dim drops the page size, which legacy factories compile
        // into TensorAccessor args and circular-buffer sizes.
        TT_FATAL(
            !(dim == rank - 1 && tensor.layout() == Layout::ROW_MAJOR),
            "{}: dynamic dim {} is the last dimension of a row-major tensor, so it is the page size and stays in "
            "the key",
            operation_name_,
            declared.dim);
        extents[dim] = std::nullopt;
        dim_matched_[i] = true;
    }
    key_ += ttsl::hash::canonical_key(tensor.storage_type(), extents, tensor.tensor_spec().tensor_layout());
}

std::string KeyEncoder::finish() && {
    TT_FATAL(
        std::ranges::all_of(dim_matched_, std::identity{}),
        "{}: a declared dynamic dim does not point at one of the op's tensor arguments",
        operation_name_);
    TT_FATAL(
        std::ranges::all_of(value_matched_, std::identity{}),
        "{}: a declared dynamic value is not an attribute reached by reference",
        operation_name_);
    // Length-prefixed, so the derived values cannot be read as part of the arguments' encoding.
    key_ += ttsl::hash::canonical_key(std::uint64_t{dynamic_key_.derived_.size()});
    key_ += dynamic_key_.derived_;
    return std::move(key_);
}

}  // namespace detail

}  // namespace ttnn::dynamic_length
