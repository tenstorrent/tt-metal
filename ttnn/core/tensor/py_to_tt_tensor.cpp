// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/tensor/py_to_tt_tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"

#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"

#include <tt_stl/unreachable.hpp>

#include <tracy/Tracy.hpp>

using namespace tt::tt_metal;

namespace {
auto get_datatype_tile_size(DataType dtype) { return tt::tile_size(datatype_to_dataformat_converter(dtype)); }

// Check if typecast and layout operations can be safely executed on the device
// for particular data type.
bool can_exec_ops_on_device(DataType type) {
    switch (type) {
        case DataType::UINT32:
        case DataType::INT32:
            // https://github.com/tenstorrent/tt-metal/issues/23407 (to_layout(RM) is not working for uint32/int32)
        case DataType::UINT16:
            // Tilize doesn't support uint16.
        case DataType::UINT8:
            // https://github.com/tenstorrent/tt-metal/issues/21682 (typecast doesn't support uint8)
            return false;
        default: return true;
    }
};

// Check if the tensor with the specified memory config and tiling can be
// constructed and used on the device, ignoring details of the type conversion.
bool can_construct_on_device(
    ttnn::distributed::MeshDevice* device, const std::optional<Tile>& optional_tile, const TensorSpec& tensor_spec) {
    bool res = device != nullptr &&
               // When on-device strategy is used, tensor spec needs a default alignment based on the target layout.
               // Otherwise, the tensor loses the data in the `to_layout` conversion and type conversion. But, if the
               // default alignment is used, the tensors of rank 5 and above are squeezed down to the rank 4 in
               // `build_ndiml_tilize`, which causes the padding loss, and subqequently the failure to validate
               // tilize operation, which requires `physical_volume() % tt::constants::TILE_HW == 0`
               tensor_spec.logical_shape().rank() <= 4 &&
               // Logical shape must match physical shape for the tensor to be constructed on the device.
               tt::tt_metal::logical_matches_physical(tensor_spec);

    if (optional_tile.has_value()) {
        // on-device tiling operation expects tiles to be divisible by 32x32.
        res &= ((optional_tile->get_width() % tt::constants::TILE_WIDTH) == 0) &&
               ((optional_tile->get_height() % tt::constants::TILE_HEIGHT) == 0);
    }
    return res;
}

Tensor create_tt_tensor_from_host_data(
    HostBuffer& host_buffer,
    DataType src_dtype,
    DataType dst_dtype,
    Layout layout,
    const ttnn::Shape& tensor_shape,
    const MemoryConfig& memory_config,
    const std::optional<Tile>& optional_tile,
    float pad_value,
    const ttnn::distributed::TensorToMesh* mesh_mapper,
    std::optional<ttnn::QueueId> cq_id,
    ttnn::distributed::MeshDevice* device,
    bool preserve_nan_values,
    bool fast_approx) {
    using namespace tt::tt_metal;
    auto create_tensor_from_host_buffer = [&]<typename T>() -> Tensor {
        TensorLayout dst_tensor_layout(dst_dtype, PageConfig(layout, optional_tile), memory_config);
        TensorLayout src_tensor_layout(src_dtype, PageConfig(ttnn::Layout::ROW_MAJOR), memory_config);
        if (mesh_mapper != nullptr) {
            const bool must_construct_on_host = device == nullptr || !fast_approx || preserve_nan_values;
            return ttnn::distributed::create_distributed_tensor(
                host_buffer.view_as<T>(),
                tensor_shape,
                host_buffer.pin(),
                must_construct_on_host ? dst_tensor_layout : src_tensor_layout,
                *mesh_mapper,
                device != nullptr ? std::make_optional(std::ref(*device)) : std::nullopt,
                cq_id,
                static_cast<T>(pad_value));
        }

        // If the memory config is sharded, the tensor must be constructed on the host. Even if we can borrow the
        // buffer, the sharding spec may require padding, including cases where the shard dimension is larger than the
        // shape dimension.
        const bool construct_on_device =
            can_exec_ops_on_device(dst_dtype) && can_exec_ops_on_device(src_dtype) && !memory_config.is_sharded() &&
            can_construct_on_device(device, optional_tile, TensorSpec(tensor_shape, src_tensor_layout));

        // Borrow the Python buffer directly when possible, otherwise copy via from_span.
        // TODO: Remove preserve_nan_values check after https://github.com/tenstorrent/tt-metal/issues/31406
        const bool can_borrow =
            src_dtype == convert_to_data_type<T>() && fast_approx && !preserve_nan_values && construct_on_device;

        if (can_borrow) {
            return Tensor::from_borrowed_data(host_buffer.view_as<T>(), tensor_shape, host_buffer.pin(), optional_tile);
        }

        return Tensor::from_span(
            ttsl::make_const_span(host_buffer.view_as<T>()),
            TensorSpec(tensor_shape, dst_tensor_layout),
            nullptr,
            std::nullopt,
            static_cast<T>(pad_value));
    };

    switch (src_dtype) {
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: return create_tensor_from_host_buffer.operator()<float>();
        case DataType::UINT32: return create_tensor_from_host_buffer.operator()<uint32_t>();
        case DataType::INT32: return create_tensor_from_host_buffer.operator()<int32_t>();
        case DataType::UINT8: return create_tensor_from_host_buffer.operator()<uint8_t>();
        case DataType::UINT16: return create_tensor_from_host_buffer.operator()<uint16_t>();
        case DataType::FLOAT32: return create_tensor_from_host_buffer.operator()<float>();
        case DataType::BFLOAT16: return create_tensor_from_host_buffer.operator()<bfloat16>();
        default: TT_THROW("Unsupported data type");
    }
}

DataType compute_host_dtype(ttnn::PyDType src_dtype, const DataType& dst_dtype, bool is_sharded) {
    auto to_ttnn_dtype = [](ttnn::PyDType type) {
        switch (type) {
            case ttnn::PyDType::FLOAT32: return DataType::FLOAT32;
            case ttnn::PyDType::BFLOAT16: return DataType::BFLOAT16;
            case ttnn::PyDType::INT32: return DataType::INT32;
            case ttnn::PyDType::UINT32: return DataType::UINT32;
            case ttnn::PyDType::UINT8: return DataType::UINT8;
            case ttnn::PyDType::UINT16: return DataType::UINT16;
            case ttnn::PyDType::BOOL: return DataType::UINT8;
            case ttnn::PyDType::UINT64:
            case ttnn::PyDType::FLOAT64:
            case ttnn::PyDType::FLOAT16:
            case ttnn::PyDType::INT64:
            case ttnn::PyDType::INT16:
            case ttnn::PyDType::INT8:
            default: return DataType::INVALID;
        }
        ttsl::unreachable();
    };

    const DataType mapped_dst_type =
        (dst_dtype == DataType::BFLOAT4_B or dst_dtype == DataType::BFLOAT8_B) ? DataType::BFLOAT16 : dst_dtype;

    if (to_ttnn_dtype(src_dtype) == DataType::INVALID) {
        return mapped_dst_type;
    }

    if (is_sharded && get_datatype_tile_size(dst_dtype) != get_datatype_tile_size(to_ttnn_dtype(src_dtype))) {
        // Sharded typecast does not support conversion between tensors with types of different tile size:
        // See explicit assertion in the `TypecastShardedProgramFactory::create` method implementation.
        return mapped_dst_type;
    }

    return to_ttnn_dtype(src_dtype);  // borrow pytensor by default.
}
}  // namespace

namespace ttnn {

Tensor convert_python_tensor_to_tt_tensor(
    const ttnn::Shape& tensor_shape,
    DataType dst_dtype,
    Layout layout,
    const std::optional<Tile>& optional_tile,
    const MemoryConfig& memory_config,
    ttnn::PyDType src_data_type,
    const std::function<HostBuffer(DataType)>& get_host_tensor,
    std::optional<distributed::MeshDevice*> device,
    std::optional<ttnn::QueueId> cq_id,
    const ttnn::distributed::TensorToMesh* mesh_mapper,
    std::optional<float> pad_value,
    bool preserve_nan_values,
    bool col_tilize,
    bool fast_approx) {
    ZoneScoped;
    if (dst_dtype == DataType::BFLOAT8_B || dst_dtype == DataType::BFLOAT4_B) {
        TT_FATAL(layout == Layout::TILE, "Layout must be Layout::TILE for bfloat8_b or bfloat4_b!");
    }

    GraphTracker::instance().track_function_start(
        "ttnn::convert_python_tensor_to_tt_tensor",
        dst_dtype,
        layout,
        optional_tile,
        memory_config,
        device,
        cq_id,
        mesh_mapper,
        pad_value);

    auto host_dtype = compute_host_dtype(src_data_type, dst_dtype, memory_config.is_sharded());
    if (col_tilize) {
        host_dtype = DataType::FLOAT32;
    }
    auto host_buffer = get_host_tensor(host_dtype);

    ttnn::Shape effective_shape = tensor_shape;
    if (col_tilize) {
        // Transpose the last two dims of the float32 host buffer by creating a new
        // buffer and replacing host_buffer, so that BFP exponent grouping happens
        // along columns instead of rows.
        auto rank = tensor_shape.rank();
        TT_FATAL(rank >= 2, "col_tilize requires tensor rank >= 2, got {}", rank);
        TT_FATAL(
            dst_dtype == DataType::BFLOAT8_B || dst_dtype == DataType::BFLOAT4_B,
            "col_tilize requires BFP dtype (BFLOAT8_B or BFLOAT4_B)");

        const auto K = tensor_shape[-2];
        const auto N = tensor_shape[-1];
        size_t batch_size = 1;
        for (size_t i = 0; i < rank - 2; ++i) {
            batch_size *= tensor_shape[i];
        }

        const float* src = host_buffer.view_as<float>().data();
        std::vector<float> transposed(batch_size * K * N);
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t i = 0; i < static_cast<size_t>(N); ++i) {
                for (size_t j = 0; j < static_cast<size_t>(K); ++j) {
                    transposed[(b * N * K) + (i * K) + j] = src[(b * K * N) + (j * N) + i];
                }
            }
        }
        host_buffer = HostBuffer(std::move(transposed));

        // Build transposed shape: swap last two dims
        std::vector<uint32_t> new_dims;
        new_dims.reserve(rank);
        for (size_t i = 0; i < rank - 2; ++i) {
            new_dims.push_back(tensor_shape[i]);
        }
        new_dims.push_back(N);
        new_dims.push_back(K);
        effective_shape = ttnn::Shape(ttsl::Span<const uint32_t>(new_dims.data(), new_dims.size()));
    }

    Tensor output = create_tt_tensor_from_host_data(
        host_buffer,
        host_dtype,
        dst_dtype,
        layout,
        effective_shape,
        memory_config,
        optional_tile,
        pad_value.value_or(0.0f),
        mesh_mapper,
        cq_id,
        device.value_or(nullptr),
        preserve_nan_values,
        fast_approx);

    auto set_layout = [&](Layout target) {
        if (output.layout() != target) {
            output = ttnn::to_layout(output, target);
        }
    };

    if (device) {
        output = output.to_device(device.value(), memory_config, cq_id);
        if (output.dtype() != dst_dtype) {
            // Need to perform final data conversion on device, typecast requires TILE layout.
            set_layout(Layout::TILE);
            output = ttnn::typecast(output, dst_dtype);
        }

        set_layout(layout);
    } else {
        set_layout(layout);
    }

    TT_FATAL(output.dtype() == dst_dtype, "Output dtype mismatch. Expected: {}, Got: {}", dst_dtype, output.dtype());
    GraphTracker::instance().track_function_end(output);
    return output;
}

}  // namespace ttnn
