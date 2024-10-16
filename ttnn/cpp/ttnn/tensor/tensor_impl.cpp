// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor_impl.hpp"

#include "ttnn/tensor/tensor_impl_wrapper.hpp"
#include "ttnn/distributed/api.hpp"

namespace tt {

namespace tt_metal {

namespace tensor_impl {

TensorPrintProfile TTNN_TENSOR_PRINT_PROFILE = TensorPrintProfile::Short;

std::ostream& operator<<(std::ostream& os, const DataType& dtype) {
    switch (dtype) {
        case DataType::BFLOAT8_B: os << "bfloat8_b"; break;
        case DataType::BFLOAT4_B: os << "bfloat4_b"; break;
        case DataType::BFLOAT16: os << "bfloat16"; break;
        case DataType::FLOAT32: os << "float32"; break;
        case DataType::UINT8: os << "uint8"; break;
        case DataType::UINT16: os << "uint16"; break;
        case DataType::UINT32: os << "uint32"; break;
        case DataType::INT32: os << "int32"; break;
        default: throw std::invalid_argument("Unknown data type");
    }
    return os;
}

uint32_t element_size_bytes(DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT16: return sizeof(bfloat16);
        case DataType::FLOAT32: return sizeof(float);
        case DataType::INT32: return sizeof(int32_t);
        case DataType::UINT32: return sizeof(uint32_t);
        case DataType::UINT16: return sizeof(uint16_t);
        case DataType::UINT8: return sizeof(uint8_t);
        case DataType::BFLOAT8_B: return sizeof(std::byte);
        case DataType::BFLOAT4_B: return sizeof(std::byte);
        default: TT_THROW("Unsupported data type");
    }
}

uint32_t get_page_size(DataType dtype, Layout layout, uint32_t total_size_bytes, const ttnn::SimpleShape& shape, const std::optional<Tile>& tile) {
    uint32_t W = shape[-1];
    uint32_t page_size = 0;
    const auto tile_HW = tile.has_value() ? tile->get_tile_hw() : constants::TILE_HW;
    const auto bfloat8b_tile_HW = tile.has_value() ? tile_HW + 64 : constants::BFLOAT8_B_TILE_HW;
    const auto bfloat4b_tile_HW = tile.has_value() ? tile_HW / 2 + 64 : constants::BFLOAT4_B_TILE_HW;
    switch (layout) {
        case Layout::ROW_MAJOR: {
            uint32_t size_of_element = element_size_bytes(dtype);
            page_size = W * size_of_element;
        } break;
        case Layout::TILE: {
            // TODO: Update to be generic for data type (issue 462)
            switch (dtype) {
                case DataType::BFLOAT16: {
                    // Float is converted to bfloat16 before being written to device
                    uint32_t size_of_element = element_size_bytes(DataType::BFLOAT16);
                    page_size = tile_HW * size_of_element;
                } break;
                case DataType::FLOAT32: {
                    uint32_t size_of_element = element_size_bytes(DataType::FLOAT32);
                    page_size = tile_HW * size_of_element;
                } break;
                case DataType::UINT32:
                case DataType::INT32:
                case DataType::UINT16:
                case DataType::UINT8:{
                    uint32_t size_of_element = element_size_bytes(dtype);
                    page_size = tile_HW * size_of_element;
                } break;
                case DataType::BFLOAT4_B: {
                    page_size = bfloat4b_tile_HW;
                } break;
                case DataType::BFLOAT8_B: {
                    page_size = bfloat8b_tile_HW;
                } break;
                default: TT_ASSERT(false && "Unsupported data type!");
            }
            TT_ASSERT(total_size_bytes % page_size == 0);
        } break;
        default: TT_ASSERT(false && "Unsupported layout to write to device");
    }
    TT_ASSERT(page_size != 0);
    return page_size;
}

std::array<uint32_t, 2> get_sharded_page_shape(Layout layout, DataType dtype, std::array<uint32_t, 2> shard_shape, const std::optional<Tile>& tile) {
    // Physical limitation in FD for now
    switch (layout) {
        case Layout::ROW_MAJOR:
            // TODO: Explore valid page shapes other than 1,W
            return {1, shard_shape[1]};
        case Layout::TILE: {
            auto tile_shape = tile.value_or(Tile{{constants::TILE_HEIGHT, constants::TILE_WIDTH}}).get_tile_shape();
            return {tile_shape[0], tile_shape[1]};
        }
        default: TT_THROW("Unsupported layout to write to device");
    }
}

void validate_sharded_buffer_allocation(
    const ttnn::SimpleShape& shape,
    Layout layout,
    DataType data_type,
    const ShardSpecBuffer& shard_params,
    const MemoryConfig& memory_config,
    const std::optional<Tile>& tile) {
    const auto& shard_spec = memory_config.shard_spec.value();
    const auto& shard_shape = shard_spec.shape;

    uint32_t num_cores = shard_spec.num_cores();

    uint32_t total_height = shape.volume() / shape[-1];
    uint32_t total_width = shape[-1];
    if (memory_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        TT_ASSERT(
            total_width == shard_shape[1],
                "Shard shape {} does not divide tensor shape {} correctly according to sharding scheme",
                shard_shape[1],
                total_width);
        uint32_t num_shards = div_up(total_height, shard_shape[0]);
        TT_ASSERT(
            num_shards <= num_cores,
            "Number of shards {} must match number of cores {}", num_shards, num_cores);
    } else if (memory_config.memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        TT_ASSERT(
            total_height == shard_shape[0],
            "Shard shape does not divide tensor shape correctly according to sharding scheme");
        uint32_t num_shards = div_up(total_width, shard_shape[1]);
        TT_ASSERT(
            num_shards <= num_cores,
            "Number of shards {} must match number of cores {}", num_shards, num_cores);
    } else if (memory_config.memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        TT_ASSERT(
            shard_spec.grid.ranges().size() == 1, "Shard grid must be one full rectangular grid for block sharded!");
        uint32_t num_shards_along_height = div_up(total_height, shard_shape[0]);
        uint32_t num_shards_along_width = div_up(total_width, shard_shape[1]);

        // Additionally check that number of cores along height and width matches shard grid
        const CoreCoord shard_grid = shard_spec.grid.bounding_box().grid_size();
        if (shard_spec.orientation == ShardOrientation::ROW_MAJOR) {
            TT_ASSERT(
                num_shards_along_height <= shard_grid.y,
                    "Number of shards along height {} must match number of rows {} for row major orientation!",
                    num_shards_along_height,
                    shard_grid.y);
            TT_ASSERT(
                num_shards_along_width <= shard_grid.x,
                    "Number of shards along width {} must match number of columns {} for row major orientation!",
                    num_shards_along_width,
                    shard_grid.x);
        } else {
            TT_ASSERT(
                num_shards_along_height <= shard_grid.x,
                    "Number of shards along height {} must match number of columns {} for column major orientation!",
                    num_shards_along_height,
                    shard_grid.x);
            TT_ASSERT(
                num_shards_along_width <= shard_grid.y,
                    "Number of shards along width {} must match number of rows {} for column major orientation!",
                    num_shards_along_width,
                    shard_grid.y);
        }
    } else {
        TT_THROW("Unsupported sharding scheme");
    }
    if (layout == Layout::TILE) {
        auto tile_shape = tile.value_or(Tile{{constants::TILE_HEIGHT, constants::TILE_WIDTH}}).get_tile_shape();
        TT_FATAL(
            (shard_shape[0] % tile_shape[0] == 0 && shard_shape[1] % tile_shape[1] == 0),
            "Shard shape {} must be tile {} sized!", shard_shape, tile_shape);
    } else if (layout == Layout::ROW_MAJOR) {
        TT_FATAL(shard_shape[1] * tensor_impl::element_size_bytes(data_type) % sizeof(uint32_t) == 0, "Error");
    }
}

namespace detail {

DeviceBuffer allocate_interleaved_buffer_on_device(
    size_t buffer_size_bytes,
    Device* device,
    const ttnn::SimpleShape& shape,
    DataType data_type,
    Layout layout,
    const MemoryConfig& memory_config,
    const std::optional<Tile>& tile) {
    uint32_t page_size = get_page_size(data_type, layout, buffer_size_bytes, shape, tile);
    return std::make_shared<Buffer>(device, buffer_size_bytes, page_size, memory_config.buffer_type);
}

DeviceBuffer allocate_contiguous_buffer_on_device(
    size_t buffer_size_bytes, Device* device, const MemoryConfig& memory_config) {
    return std::make_shared<Buffer>(device, buffer_size_bytes, buffer_size_bytes, memory_config.buffer_type);
}

DeviceBuffer allocate_sharded_buffer_on_device(
    size_t buffer_size_bytes,
    Device* device,
    const ttnn::SimpleShape& shape,
    DataType data_type,
    Layout layout,
    const ShardSpecBuffer& shard_params,
    const MemoryConfig& memory_config,
    const std::optional<Tile>& tile) {
    validate_sharded_buffer_allocation(shape, layout, data_type, shard_params, memory_config, tile);
    const auto& page_shape = ttnn::SimpleShape(shard_params.page_shape);
    uint32_t page_size = get_page_size(data_type, layout, buffer_size_bytes, page_shape, tile);

    return std::make_shared<Buffer>(
        device, buffer_size_bytes, page_size, memory_config.buffer_type, memory_config.memory_layout, shard_params);
}

}  // namespace detail

DeviceBuffer allocate_buffer_on_device(
    size_t buffer_size_bytes,
    Device* device,
    const ttnn::SimpleShape& shape,
    DataType data_type,
    Layout layout,
    const MemoryConfig& memory_config,
    const std::optional<ShardSpecBuffer>& shard_spec,
    const std::optional<Tile>& tile) {
    if (memory_config.memory_layout == tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
        return detail::allocate_interleaved_buffer_on_device(
            buffer_size_bytes, device, shape, data_type, layout, memory_config, tile);
    } else if (memory_config.memory_layout == tt::tt_metal::TensorMemoryLayout::SINGLE_BANK) {
        return detail::allocate_contiguous_buffer_on_device(buffer_size_bytes, device, memory_config);
    } else {
        TT_ASSERT(memory_config.is_sharded(), "Incorrect Memory Layout");
        return detail::allocate_sharded_buffer_on_device(
            buffer_size_bytes, device, shape, data_type, layout, shard_spec.value(), memory_config, tile);
    }
}

void validate_on_device_dtype_and_layout(Device* device, const ttnn::SimpleShape& shape, DataType dtype, Layout layout) {
    // TODO: Get supported layout and dtypes from device
    auto supported_dtype = [&dtype]() {
        TT_ASSERT(
            (dtype == DataType::UINT32 || dtype == DataType::INT32 || dtype == DataType::FLOAT32 ||
             dtype == DataType::UINT8 || dtype == DataType::UINT16 || dtype == DataType::BFLOAT16 ||
             dtype == DataType::BFLOAT8_B || dtype == DataType::BFLOAT4_B),
            "Only UINT32, INT32, FLOAT32, UINT16, UINT8, BFLOAT16, BFLOAT8_B, or BFLOAT4_B dtypes are supported on device!");
    };
    auto supported_layout = [&shape, &dtype, &layout]() {
        switch (dtype) {
            case DataType::UINT32:
            case DataType::INT32:
            case DataType::FLOAT32: break;
            case DataType::UINT8:
                if (layout == Layout::ROW_MAJOR) {
                    TT_ASSERT(
                        shape[-1] % 4 == 0,
                        "For ROW_MAJOR layout tensors with dtype UINT8, tensor width must be divisible by "
                        "4 since data is packed as uint32_t when creating buffers on device!");
                }
                break;
            case DataType::UINT16:
            case DataType::BFLOAT16:
                if (layout == Layout::ROW_MAJOR) {
                    TT_ASSERT(
                        shape[-1] % 2 == 0,
                        "For ROW_MAJOR layout tensors with dtype BFLOAT16 or UINT16, tensor width must be divisible by "
                        "2 since data is packed as uint32_t when creating buffers on device!");
                }
                break;
            case DataType::BFLOAT8_B:
            case DataType::BFLOAT4_B:
                TT_ASSERT(layout == Layout::TILE, "Only TILE layout is supported for BFLOAT8_B dtype!");
                break;
            default:
                TT_ASSERT(
                    false,
                    "Only UINT32, INT32, FLOAT32, UINT16, BFLOAT16, BFLOAT8_B, or BFLOAT4_B dtypes are supported on "
                    "device!");
                break;
        }
    };
    supported_dtype();
    supported_layout();
}

Tensor pad_bfloat8_b(
    const Tensor& tensor, const tt::tt_metal::LegacyShape& output_tensor_shape, const ttnn::SimpleShape& input_tensor_start, float pad_value) {
    // TODO(arakhmati): do not convert to FLOAT32

    // Convert to FLOAT32 tensor and pad
    auto input_packed_data = owned_buffer::get_as<uint32_t>(tensor).get();
    auto input_float_data =
        unpack_bfp8_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false);
    auto input_float_buffer = owned_buffer::create<float>(std::move(input_float_data));
    auto float_tensor =
        Tensor(OwnedStorage{input_float_buffer}, tensor.get_legacy_shape(), DataType::FLOAT32, tensor.get_layout(), tensor.get_tile())
            .pad(output_tensor_shape, input_tensor_start, pad_value);

    // Convert back to BFLOAT8_B
    auto output_float_data = owned_buffer::get_as<float>(float_tensor).get();
    auto output_packed_data =
        pack_fp32_vec_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
    auto output_uint32_buffer = owned_buffer::create<uint32_t>(std::move(output_packed_data));
    return Tensor(
        std::move(OwnedStorage{std::move(output_uint32_buffer)}),
        float_tensor.get_legacy_shape(),
        DataType::BFLOAT8_B,
        tensor.get_layout(),
        tensor.get_tile());
}

Tensor unpad_bfloat8_b(const Tensor& tensor, const ttnn::SimpleShape& output_tensor_start, const ttnn::SimpleShape& output_tensor_end) {
    // TODO(arakhmati): do not convert to FLOAT32

    // Convert to FLOAT32 tensor and unpad
    auto input_packed_data = owned_buffer::get_as<uint32_t>(tensor).get();
    auto input_float_data =
        unpack_bfp8_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false);
    auto input_float_buffer = owned_buffer::create<float>(std::move(input_float_data));
    auto float_tensor =
        Tensor(OwnedStorage{input_float_buffer}, tensor.get_legacy_shape(), DataType::FLOAT32, tensor.get_layout(), tensor.get_tile())
            .unpad(output_tensor_start, output_tensor_end);

    // Convert back to BFLOAT8_B
    auto output_float_data = owned_buffer::get_as<float>(float_tensor).get();
    auto output_packed_data =
        pack_fp32_vec_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
    auto output_uint32_buffer = owned_buffer::create<uint32_t>(std::move(output_packed_data));
    return Tensor(
        std::move(OwnedStorage{std::move(output_uint32_buffer)}),
        float_tensor.get_legacy_shape(),
        DataType::BFLOAT8_B,
        tensor.get_layout(),
        tensor.get_tile());
}

Tensor pad_bfloat4_b(
    const Tensor& tensor, const tt::tt_metal::LegacyShape& output_tensor_shape, const ttnn::SimpleShape& input_tensor_start, float pad_value) {
    // TODO(arakhmati): do not convert to FLOAT32

    // Convert to FLOAT32 tensor and pad
    auto input_packed_data = owned_buffer::get_as<uint32_t>(tensor).get();
    auto input_float_data =
        unpack_bfp4_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false);
    auto input_float_buffer = owned_buffer::create<float>(std::move(input_float_data));
    auto float_tensor =
        Tensor(OwnedStorage{input_float_buffer}, tensor.get_legacy_shape(), DataType::FLOAT32, tensor.get_layout(), tensor.get_tile())
            .pad(output_tensor_shape, input_tensor_start, pad_value);

    // Convert back to BFLOAT4_B
    auto output_float_data = owned_buffer::get_as<float>(float_tensor).get();
    auto output_packed_data =
        pack_fp32_vec_as_bfp4_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
    auto output_uint32_buffer = owned_buffer::create<uint32_t>(std::move(output_packed_data));
    return Tensor(
        std::move(OwnedStorage{std::move(output_uint32_buffer)}),
        float_tensor.get_legacy_shape(),
        DataType::BFLOAT4_B,
        tensor.get_layout(),
        tensor.get_tile());
}

Tensor unpad_bfloat4_b(const Tensor& tensor, const ttnn::SimpleShape& output_tensor_start, const ttnn::SimpleShape& output_tensor_end) {
    // TODO(arakhmati): do not convert to FLOAT32

    // Convert to FLOAT32 tensor and unpad
    auto input_packed_data = owned_buffer::get_as<uint32_t>(tensor).get();
    auto input_float_data =
        unpack_bfp4_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false);
    auto input_float_buffer = owned_buffer::create<float>(std::move(input_float_data));
    auto float_tensor =
        Tensor(OwnedStorage{input_float_buffer}, tensor.get_legacy_shape(), DataType::FLOAT32, tensor.get_layout(), tensor.get_tile())
            .unpad(output_tensor_start, output_tensor_end);

    // Convert back to BFLOAT4_B
    auto output_float_data = owned_buffer::get_as<float>(float_tensor).get();
    auto output_packed_data =
        pack_fp32_vec_as_bfp4_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
    auto output_uint32_buffer = owned_buffer::create<uint32_t>(std::move(output_packed_data));
    return Tensor(
        std::move(OwnedStorage{std::move(output_uint32_buffer)}),
        float_tensor.get_legacy_shape(),
        DataType::BFLOAT4_B,
        tensor.get_layout(),
        tensor.get_tile());
}

// ======================================================================================
//                                      .to_string()
// ======================================================================================

namespace detail {

struct DimensionShortener {
    size_t size;
    std::optional<std::size_t> max;

    bool print_parenthesis_and_advance_index_if_reached_half_of_max_and_check_if_loop_is_done(
        std::ostream& ss, std::size_t& index, const std::string& before, const std::string& after) const {
        if (this->max.has_value() and this->size >= this->max.value() and index == this->max.value() / 2) {
            ss << before << "...," << after;
            index = this->size - (this->max.value() / 2);
        }
        return index < this->size;
    }
};

inline DimensionShortener get_dimension_shortener(std::size_t size) {
    switch (TTNN_TENSOR_PRINT_PROFILE) {
        case TensorPrintProfile::Empty: return DimensionShortener{size, 0};
        case TensorPrintProfile::Short: return DimensionShortener{size, 4};
        case TensorPrintProfile::Full: return DimensionShortener{size, std::nullopt};
        default: TT_THROW("Unrecognized TTNN_TENSOR_PRINT_PROFILE {}", TTNN_TENSOR_PRINT_PROFILE);
    }
}

inline void print_trailing_comma(std::ostream& ss, std::size_t index, std::size_t size, const std::string& after) {
    if (index < size - 1) {
        ss << "," << after;
    }
}

template <typename T>
inline void print_datum(std::ostream& ss, T datum) {
    if (std::is_integral<T>::value) {
        ss << std::setw(5) << datum;
    } else {
        ss << std::fixed << std::setw(8) << std::setprecision(5) << datum;
    }
}

template <>
inline void print_datum(std::ostream& ss, bfloat16 datum) {
    print_datum(ss, datum.to_float());
}

template <>
inline void print_datum(std::ostream& ss, uint8_t datum) {
    print_datum<uint32_t>(ss, datum);
}

inline constexpr int constexpr_strlen(const char* str) { return *str ? 1 + constexpr_strlen(str + 1) : 0; }

constexpr auto TENSOR_TYPE_STRING = "ttnn.Tensor";
constexpr auto TENSOR_TYPE_STRING_PLUS_OPEN_PARENTHESIS_LENGTH = constexpr_strlen(TENSOR_TYPE_STRING) + 1;

static constexpr auto TAB = "    ";
static constexpr auto TAB_MINUS_1 = "   ";

template <typename BufferType, std::int64_t Rank, std::int64_t Dim = 0>
void to_string_row_major(
    std::stringstream& ss,
    const BufferType& buffer,
    const tt::tt_metal::LegacyShape& shape,
    std::size_t outer_index,
    const std::size_t buffer_offset) {
    auto stride = 1;
    for (auto index = Dim + 1; index < shape.rank(); index++) {
        stride *= shape[index];
    }

    std::string spaces = std::string(TENSOR_TYPE_STRING_PLUS_OPEN_PARENTHESIS_LENGTH + Dim, ' ');
    std::string before;
    std::string after;
    if constexpr (Rank == 1) {
        before = " ";
        after = " ";
    } else if constexpr (Rank == 2) {
        before = spaces + " ";
        after = "\n";
    } else {
        before = spaces + " ";
        after = "\n\n";
    }

    if (Dim > 0 and outer_index > 0) {
        ss << spaces;
    }
    ss << "[";
    auto dimension_shortener = get_dimension_shortener(shape[-Rank]);
    for (std::size_t index = 0;
         dimension_shortener.print_parenthesis_and_advance_index_if_reached_half_of_max_and_check_if_loop_is_done(
             ss, index, before, after);
         index++) {
        std::string after_comma;
        if constexpr (Rank == 1) {
            after_comma = " ";
        } else if constexpr (Rank == 2) {
            after_comma = "\n";
        } else {
            after_comma = after;
        }

        if constexpr (Rank > 1) {
            to_string_row_major<BufferType, Rank - 1, Dim + 1>(
                ss, buffer, shape, index, buffer_offset + index * stride);
        } else {
            print_datum(ss, buffer[buffer_offset + index]);
        }
        print_trailing_comma(ss, index, shape[-Rank], after_comma);
    }
    ss << "]";
}

template <typename BufferType, std::int64_t Rank, std::int64_t Dim = 0>
void to_string_tile(
    std::stringstream& ss,
    const BufferType& buffer,
    const tt::tt_metal::LegacyShape& shape,
    std::size_t outer_index,
    const std::size_t buffer_offset) {
    // For now, print it the same way as row-major
    return to_string_row_major<BufferType, Rank, Dim>(ss, buffer, shape, outer_index, buffer_offset);
}

template <typename BufferType>
std::string to_string(const BufferType& buffer, const tt::tt_metal::LegacyShape& shape, DataType dtype, Layout layout) {
    std::stringstream ss;
    ss << TENSOR_TYPE_STRING << "(";

    if (TTNN_TENSOR_PRINT_PROFILE == TensorPrintProfile::Empty) {
        ss << "...";
    } else if (layout == Layout::ROW_MAJOR) {
        switch (shape.rank()) {
            case 0: to_string_row_major<BufferType, 0>(ss, buffer, shape, 0, 0); break;
            case 1: to_string_row_major<BufferType, 1>(ss, buffer, shape, 0, 0); break;
            case 2: to_string_row_major<BufferType, 2>(ss, buffer, shape, 0, 0); break;
            case 3: to_string_row_major<BufferType, 3>(ss, buffer, shape, 0, 0); break;
            case 4: to_string_row_major<BufferType, 4>(ss, buffer, shape, 0, 0); break;
            case 5: to_string_row_major<BufferType, 5>(ss, buffer, shape, 0, 0); break;
            case 6: to_string_row_major<BufferType, 6>(ss, buffer, shape, 0, 0); break;
            case 7: to_string_row_major<BufferType, 7>(ss, buffer, shape, 0, 0); break;
            case 8: to_string_row_major<BufferType, 8>(ss, buffer, shape, 0, 0); break;
            default: TT_THROW("Unsupported Rank for printing tensor with ROW_MAJOR_LAYOUT!"); break;
        }
    } else if (layout == Layout::TILE) {
        switch (shape.rank()) {
            case 2: to_string_tile<BufferType, 2>(ss, buffer, shape, 0, 0); break;
            case 3: to_string_tile<BufferType, 3>(ss, buffer, shape, 0, 0); break;
            case 4: to_string_tile<BufferType, 4>(ss, buffer, shape, 0, 0); break;
            case 5: to_string_tile<BufferType, 5>(ss, buffer, shape, 0, 0); break;
            case 6: to_string_tile<BufferType, 6>(ss, buffer, shape, 0, 0); break;
            case 7: to_string_tile<BufferType, 7>(ss, buffer, shape, 0, 0); break;
            case 8: to_string_tile<BufferType, 8>(ss, buffer, shape, 0, 0); break;
            default: TT_THROW("Unsupported Rank for printing tensor with TILE_LAYOUT!"); break;
        }
    } else {
        TT_THROW("Unsupported Layout for printing tensor!");
    }
    ss << ", shape=" << fmt::format("{}", shape) << ", dtype=" << fmt::format("{}", dtype)
       << ", layout=" << fmt::format("{}", layout) << ")";
    return ss.str();
}

}  // namespace detail

template <typename T>
std::string to_string(const Tensor& tensor, std::optional<DataType> original_dtype) {
    const auto shape = tensor.get_legacy_shape();
    const auto dtype = original_dtype.value_or(tensor.get_dtype());
    const auto layout = tensor.get_layout();

    if (not tensor.is_allocated()) {
        return fmt::format(
            "{}(<buffer is not allocated>, shape={}, dtype={}, layout={})",
            detail::TENSOR_TYPE_STRING,
            shape,
            dtype,
            layout);
    }

    if (is_tensor_on_device(tensor)) {
        return to_string<T>(tensor.cpu());
    }

    return std::visit(
        [&](auto&& storage) -> std::string {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                if (dtype == DataType::BFLOAT8_B and original_dtype == std::nullopt) {
                    // Convert to FLOAT32 tensor before printing
                    auto input_packed_data = owned_buffer::get_as<uint32_t>(tensor).get();
                    auto input_float_data = unpack_bfp8_tiles_into_float_vec(
                        input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false);
                    auto input_float_buffer = owned_buffer::create<float>(std::move(input_float_data));
                    auto float_tensor = Tensor(
                        OwnedStorage{input_float_buffer},
                        tensor.get_legacy_shape(),
                        DataType::FLOAT32,
                        tensor.get_layout(),
                        tensor.get_tile());
                    return to_string<float>(float_tensor, tensor.get_dtype());
                }

                if (dtype == DataType::BFLOAT4_B and original_dtype == std::nullopt) {
                    // Convert to FLOAT32 tensor before printing
                    auto input_packed_data = owned_buffer::get_as<uint32_t>(tensor).get();
                    auto input_float_data = unpack_bfp4_tiles_into_float_vec(
                        input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false);
                    auto input_float_buffer = owned_buffer::create<float>(std::move(input_float_data));
                    auto float_tensor = Tensor(
                        OwnedStorage{input_float_buffer},
                        tensor.get_legacy_shape(),
                        DataType::FLOAT32,
                        tensor.get_layout(),
                        tensor.get_tile());
                    return to_string<float>(float_tensor, tensor.get_dtype());
                }
                const auto buffer = owned_buffer::get_as<T>(storage.buffer);
                return detail::to_string(buffer, shape, dtype, layout);
            } else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                const auto buffer = borrowed_buffer::get_as<T>(storage.buffer);
                return detail::to_string(buffer, shape, dtype, layout);
            } else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_THROW("Cannot print a device tensor!");
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceStorage>) {
                auto devices = get_devices(tensor);
                auto host_tensor = tensor.cpu();
                auto device_index = 0;
                std::stringstream ss;
                apply(host_tensor, [&](const Tensor& device_tensor) {
                    ss << "device_id:" << devices.at(device_index++)->id() << std::endl;
                    ss << to_string<T>(device_tensor) << std::endl;
                });
                return ss.str();
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceHostStorage>) {
                std::stringstream ss;
                apply(tensor, [&](const Tensor& device_tensor) { ss << to_string<T>(device_tensor) << std::endl; });
                return ss.str();
            } else {
                raise_unsupported_storage<StorageType>();
            }
        },
        tensor.get_storage());
}

template std::string to_string<bfloat16>(const Tensor& tensor, std::optional<DataType> original_dtype);
template std::string to_string<float>(const Tensor& tensor, std::optional<DataType> original_dtype);
template std::string to_string<int32_t>(const Tensor& tensor, std::optional<DataType> original_dtype);
template std::string to_string<uint32_t>(const Tensor& tensor, std::optional<DataType> original_dtype);
template std::string to_string<uint16_t>(const Tensor& tensor, std::optional<DataType> original_dtype);
template std::string to_string<uint8_t>(const Tensor& tensor, std::optional<DataType> original_dtype);

template <>
std::string to_string<bfloat8_b>(const Tensor& tensor, std::optional<DataType> original_dtype) {
    return to_string<uint32_t>(tensor, original_dtype);
}

template <>
std::string to_string<bfloat4_b>(const Tensor& tensor, std::optional<DataType> original_dtype) {
    return to_string<uint32_t>(tensor, original_dtype);
}

// ======================================================================================
//                                      .to_host()
// ======================================================================================

template <typename T>
Tensor to_host_helper(const Tensor& tensor, bool blocking = true, uint8_t cq_id = ttnn::DefaultQueueId) {
    TT_ASSERT(tensor.is_allocated(), "Buffer must be allocated on device!");
    auto device_buffer = tensor.device_buffer();
    auto device = tensor.device();
    TT_ASSERT(device != nullptr && "Need device to be set copy data from device to host!");
    uint32_t size_in_bytes = device_buffer->size();
    vector<T> data_vec;
    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        data_vec.resize(size_in_bytes / sizeof(T));
        read_data_from_device_buffer<T>(device->command_queue(cq_id), device_buffer, data_vec.data(), blocking);
    } else {
        read_data_from_device_buffer<T>(device_buffer, data_vec);
    }
    auto output_buffer = owned_buffer::create<T>(std::move(data_vec));
    return Tensor(OwnedStorage{output_buffer}, tensor.get_legacy_shape(), tensor.get_dtype(), tensor.get_layout(), tensor.get_tile());
}

template <typename T>
Tensor to_host(const Tensor& tensor, bool blocking, uint8_t cq_id) {
    if (tensor.storage_type() == StorageType::DEVICE) {
        return to_host_helper<T>(tensor, blocking, cq_id);
    } else if (tensor.storage_type() == StorageType::MULTI_DEVICE) {
        auto devices = get_devices(tensor);
        Tensor host_tensor({}, devices.size());
        for (int device_index = 0; device_index < devices.size(); ++device_index) {
            const auto& device = devices[device_index];
            auto shard = get_shard_for_device(tensor, device);
            shard = to_host_helper<T>(shard, blocking, cq_id);
            host_tensor.set_shape(tensor.get_shape());
            host_tensor.set_dtype(tensor.get_dtype());
            host_tensor.set_layout(tensor.get_layout());
            host_tensor.set_tile(tensor.get_tile());
            insert_buffer_and_shape_for_device(device, shard, host_tensor, device_index);
        }
        return host_tensor;
    } else {
        return tensor;
    }
}

template Tensor to_host<bfloat16>(const Tensor& tensor, bool blocking, uint8_t cq_id);
template Tensor to_host<float>(const Tensor& tensor, bool blocking, uint8_t cq_id);
template Tensor to_host<int32_t>(const Tensor& tensor, bool blocking, uint8_t cq_id);
template Tensor to_host<uint32_t>(const Tensor& tensor, bool blocking, uint8_t cq_id);
template Tensor to_host<uint16_t>(const Tensor& tensor, bool blocking, uint8_t cq_id);
template Tensor to_host<uint8_t>(const Tensor& tensor, bool blocking, uint8_t cq_id);

template <>
Tensor to_host<bfloat4_b>(const Tensor& tensor, bool blocking, uint8_t cq_id) {
    return to_host<uint32_t>(tensor, blocking, cq_id);
}

template <>
Tensor to_host<bfloat8_b>(const Tensor& tensor, bool blocking, uint8_t cq_id) {
    return to_host<uint32_t>(tensor, blocking, cq_id);
}

// ======================================================================================
//                                  .to_host_sharded()
// ======================================================================================

template <typename T>
Tensor to_host_sharded(const Tensor& tensor) {
    TT_ASSERT(tensor.is_allocated(), "Buffer must be allocated on device!");
    auto device_buffer = tensor.buffer();
    auto device = tensor.device();
    TT_ASSERT(device != nullptr && "Need device to be set copy data from device to host!");
    uint32_t size_in_bytes = device_buffer->size();
    std::vector<uint32_t> device_data;
    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        TT_THROW("FAST_DISPATCH is not supported for to_host_sharded!");
    }
    ::detail::ReadFromBuffer(*device_buffer, device_data, true);
    auto data_vec = unpack_uint32_vec<T>(device_data);
    auto output_buffer = owned_buffer::create<T>(std::move(data_vec));
    return Tensor(OwnedStorage{output_buffer}, tensor.get_legacy_shape(), tensor.get_dtype(), tensor.get_layout(), tensor.get_tile());
}

template Tensor to_host_sharded<bfloat16>(const Tensor& tensor);
template Tensor to_host_sharded<float>(const Tensor& tensor);
template Tensor to_host_sharded<int32_t>(const Tensor& tensor);
template Tensor to_host_sharded<uint32_t>(const Tensor& tensor);
template Tensor to_host_sharded<uint16_t>(const Tensor& tensor);
template Tensor to_host_sharded<uint8_t>(const Tensor& tensor);

template <>
Tensor to_host_sharded<bfloat4_b>(const Tensor& tensor) {
    return to_host_sharded<uint32_t>(tensor);
}

template <>
Tensor to_host_sharded<bfloat8_b>(const Tensor& tensor) {
    return to_host_sharded<uint32_t>(tensor);
}

// ======================================================================================
//                               .to_device() details
// ======================================================================================


template <typename T, template <typename> typename BufferType>
void write_data_to_device_buffer(
    CommandQueue& cq, const BufferType<T>& host_buffer, DeviceBuffer device_buffer) {
    ZoneScoped;
    // TODO(arakhmati): can we use generators in this function to go from `data_to_write` to `uint32_data`?
    // And effectively get rid of any additional allocation
    if (CommandQueue::default_mode() == CommandQueue::CommandQueueMode::ASYNC) {
        if constexpr (std::is_same_v<BufferType<T>, borrowed_buffer::Buffer<T>>) {
            // When writing borrowed storage asynchronously, we have no control over when host memory is deallocated by
            // the main thread. To ensure that worker threads enqueues the correct buffer, make a copy and caputre it in
            // an owned buffer.
            uint32_t borrowed_buf_size_words =
                device_buffer->num_pages() * device_buffer->page_size() / sizeof(uint32_t);
            const uint32_t* borrowed_buf_base = static_cast<const uint32_t*>(host_buffer.data());
            std::vector<uint32_t> owned_copy_vec(borrowed_buf_base, borrowed_buf_base + borrowed_buf_size_words);
            owned_buffer::Buffer<uint32_t> owned_copy(std::make_shared<std::vector<uint32_t>>(owned_copy_vec));
            EnqueueWriteBuffer(cq, device_buffer, owned_copy.get_ptr(), false);
        } else if constexpr (std::is_same_v<BufferType<T>, owned_buffer::Buffer<T>>) {
            EnqueueWriteBuffer(cq, device_buffer, host_buffer.get_ptr(), false);
        }
    } else {
        EnqueueWriteBuffer(cq, device_buffer, host_buffer.data(), false);
    }
}

template <typename T, template <typename> typename BufferType>
void write_data_to_device_buffer(const BufferType<T>& host_buffer, Buffer& device_buffer) {
    ZoneScoped;
    // TODO(arakhmati): can we use generators in this function to go from `data_to_write` to `uint32_data`?
    // And effectively get rid of any additional allocation

    auto uint32_data = pack_vec_into_uint32_vec<T>(host_buffer);
    ::detail::WriteToBuffer(device_buffer, uint32_data);
}

template <typename T, template <typename> typename BufferType>
DeviceBuffer initialize_data_on_device(
    BufferType<T>& data_to_write,
    Device* device,
    const ttnn::SimpleShape& shape,
    DataType data_type,
    Layout layout,
    const MemoryConfig& memory_config,
    const std::optional<ShardSpecBuffer>& shard_spec,
    const std::optional<Tile>& tile,
    std::optional<std::reference_wrapper<CommandQueue>> queue = std::nullopt) {
    ZoneScoped;
    TT_ASSERT(device != nullptr);
    auto packed_size_in_bytes = packed_buffer_size_bytes<T>(data_to_write.size());

    auto device_buffer =
        allocate_buffer_on_device(packed_size_in_bytes, device, shape, data_type, layout, memory_config, shard_spec, tile);
    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        write_data_to_device_buffer<T>(
            queue.has_value() ? queue.value().get() : device->command_queue(), data_to_write, device_buffer);
    } else {
        write_data_to_device_buffer<T>(data_to_write, *device_buffer);
    }
    return device_buffer;
}

template <typename T>
DeviceBuffer to_device_buffer(
    const Storage& storage,
    Device* device,
    const ttnn::SimpleShape& shape,
    DataType data_type,
    Layout layout,
    const MemoryConfig& memory_config,
    const std::optional<ShardSpecBuffer>& shard_spec,
    const std::optional<Tile>& tile,
    std::optional<std::reference_wrapper<CommandQueue>> queue) {
    return std::visit(
        [&device, &shape, &data_type, &layout, &tile, memory_config, shard_spec](auto&& storage) -> DeviceBuffer {
            using StorageType = std::decay_t<decltype(storage)>;
            if (memory_config.is_sharded()) {
                TT_ASSERT(shard_spec.has_value(), "If sharded must provide shard_spec");
            }
            if constexpr (std::is_same_v<StorageType, OwnedStorage> or std::is_same_v<StorageType, BorrowedStorage>) {
                auto data_to_write = host_buffer::get_as<T>(storage.buffer);
                TT_ASSERT(
                    compute_buffer_size(shape, data_type) == data_to_write.size(),
                        "Tensor buffer size and number of data elements does not match: {} != {}",
                        compute_buffer_size(shape, data_type),
                        data_to_write.size());
                if (layout == Layout::TILE) {
                    auto tile_shape = tile.value_or(Tile{{constants::TILE_HEIGHT, constants::TILE_WIDTH}}).get_tile_shape();
                    TT_ASSERT(
                        (shape[-2] % tile_shape[0] == 0 && shape[-1] % tile_shape[1] == 0),
                        "Tensor shape incompatible for specified layout");
                }
                return initialize_data_on_device<T>(
                    data_to_write, device, shape, data_type, layout, memory_config, shard_spec, tile);
            } else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_THROW("Device storage doesn't support to_device_buffer");
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceStorage>) {
                TT_THROW("MultiHostStorage storage doesn't support to_device_buffer");
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceHostStorage>) {
                TT_THROW("MultiDeviceStorage doesn't support to_device_buffer");
            } else {
                raise_unsupported_storage<StorageType>();
            }
        },
        storage);
}

// ======================================================================================
//                                  .to_device()
// ======================================================================================

template <typename T>
Tensor to_device(
    const Tensor& tensor,
    Device* target_device,
    const MemoryConfig& memory_config,
    std::optional<std::reference_wrapper<CommandQueue>> queue) {
    TT_ASSERT(tensor.storage_type() != StorageType::DEVICE);
    if (tensor.storage_type() == StorageType::OWNED) {
        TT_ASSERT(tensor.is_allocated(), "Need host buffer on device to exist to copy data to device!");
    }
    TT_ASSERT(target_device != nullptr && "Need target device in order to move tensor to device!");
    TT_ASSERT(tensor.is_allocated() && "Need data to exist in order to move it to device");

    auto shape = tensor.get_padded_shape();
    auto data_type = tensor.get_dtype();
    auto layout = tensor.get_layout();
    auto tile = tensor.get_tile();

    std::optional<ShardSpecBuffer> shard_spec_buffer_opt = std::nullopt;
    if (memory_config.is_sharded()) {
        auto page_shape = get_sharded_page_shape(layout, data_type, memory_config.shard_spec.value().shape, tile);

        auto width = shape[-1];
        auto other_dims = 1;
        for (int i = 0; i < shape.rank() - 1; i++) {
            other_dims *= shape[i];
        }

        std::array<uint32_t, 2> tensor2d_size = {other_dims / page_shape[0], width / page_shape[1]};
        shard_spec_buffer_opt = ShardSpecBuffer(memory_config.shard_spec.value(), page_shape, tensor2d_size);
    }

    auto device_buffer = tensor_impl::to_device_buffer<T>(
        tensor.get_storage(), target_device, shape, data_type, layout, memory_config, shard_spec_buffer_opt, tile, queue);
    return Tensor(DeviceStorage{device_buffer}, tensor.get_shape(), data_type, layout, tile);
}

template Tensor to_device<bfloat16>(
    const Tensor& tensor,
    Device* target_device,
    const MemoryConfig& memory_config,
    std::optional<std::reference_wrapper<CommandQueue>> queue);
template Tensor to_device<float>(
    const Tensor& tensor,
    Device* target_device,
    const MemoryConfig& memory_config,
    std::optional<std::reference_wrapper<CommandQueue>> queue);
template Tensor to_device<int32_t>(
    const Tensor& tensor,
    Device* target_device,
    const MemoryConfig& memory_config,
    std::optional<std::reference_wrapper<CommandQueue>> queue);
template Tensor to_device<uint32_t>(
    const Tensor& tensor,
    Device* target_device,
    const MemoryConfig& memory_config,
    std::optional<std::reference_wrapper<CommandQueue>> queue);
template Tensor to_device<uint16_t>(
    const Tensor& tensor,
    Device* target_device,
    const MemoryConfig& memory_config,
    std::optional<std::reference_wrapper<CommandQueue>> queue);
template Tensor to_device<uint8_t>(
    const Tensor& tensor,
    Device* target_device,
    const MemoryConfig& memory_config,
    std::optional<std::reference_wrapper<CommandQueue>> queue);

template <>
Tensor to_device<bfloat4_b>(
    const Tensor& tensor,
    Device* target_device,
    const MemoryConfig& memory_config,
    std::optional<std::reference_wrapper<CommandQueue>> queue) {
    return to_device<uint32_t>(tensor, target_device, memory_config, queue);
}

template <>
Tensor to_device<bfloat8_b>(
    const Tensor& tensor,
    Device* target_device,
    const MemoryConfig& memory_config,
    std::optional<std::reference_wrapper<CommandQueue>> queue) {
    return to_device<uint32_t>(tensor, target_device, memory_config, queue);
}

// ======================================================================================
//                                  .to_layout()
// ======================================================================================

template <typename T>
Tensor to_layout(const Tensor& tensor, Layout target_layout) {
    if (tensor.get_layout() == target_layout) {
        return tensor;
    }

    auto shape = tensor.get_legacy_shape();
    auto source_layout = tensor.get_layout();
    auto tile = tensor.tile();
    auto convert = [tile, &shape, source_layout, target_layout](const auto& input_data) -> std::vector<T> {
        switch (source_layout) {
            case Layout::ROW_MAJOR:
                if (target_layout == Layout::TILE) {
                    return convert_layout_row_major_to_tile(shape, tile, input_data);
                } else {
                    TT_THROW("Unsupported layout conversion");
                }
                break;
            case Layout::TILE:
                if (target_layout == Layout::ROW_MAJOR) {
                    return convert_layout_tile_to_row_major(shape, tile, input_data);
                } else {
                    TT_THROW("Unsupported layout conversion");
                }
                break;
            default: TT_THROW("Unsupported layout conversion");
        }
    };

    auto output_storage = std::visit(
        [&convert](auto&& storage) -> std::variant<OwnedStorage, MultiDeviceHostStorage> {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                const auto input_data = owned_buffer::get_as<T>(storage.buffer);
                auto output_buffer = owned_buffer::create<T>(std::move(convert(input_data)));
                return OwnedStorage{output_buffer};
            } else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                const auto input_data = borrowed_buffer::get_as<T>(storage.buffer);
                auto output_buffer = owned_buffer::create<T>(std::move(convert(input_data)));
                return OwnedStorage{output_buffer};
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceHostStorage>) {
                std::vector<OwnedBuffer> output_buffers;
                std::vector<ttnn::Shape> output_shapes;
                for (int i = 0; i < storage.num_buffers(); i++) {
                    const auto input_data = owned_buffer::get_as<T>(storage.get_buffer(i));
                    auto output_buffer = owned_buffer::create<T>(std::move(convert(input_data)));
                    output_buffers.push_back(output_buffer);
                    output_shapes.push_back(storage.shapes[i]);
                }
                return MultiDeviceHostStorage{storage.strategy, output_buffers, output_shapes};
            } else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceStorage>) {
                TT_THROW("On-device layout conversion for tensor with MultiDeviceStorage is not supported.");
            } else {
                raise_unsupported_storage<StorageType>();
            }
        },
        tensor.get_storage());

    return std::visit(
        [&tensor, &target_layout](auto&& storage) -> Tensor {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                return Tensor(storage, tensor.get_legacy_shape(), tensor.get_dtype(), target_layout, tensor.get_tile());
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceHostStorage>) {
                return Tensor(storage, tensor.get_legacy_shape(), tensor.get_dtype(), target_layout, tensor.get_tile());
            } else {
                raise_unsupported_storage<StorageType>();
            }
        },
        output_storage);
}

template Tensor to_layout<bfloat16>(const Tensor& tensor, Layout target_layout);
template Tensor to_layout<float>(const Tensor& tensor, Layout target_layout);
template Tensor to_layout<int32_t>(const Tensor& tensor, Layout target_layout);
template Tensor to_layout<uint32_t>(const Tensor& tensor, Layout target_layout);
template Tensor to_layout<uint16_t>(const Tensor& tensor, Layout target_layout);
template Tensor to_layout<uint8_t>(const Tensor& tensor, Layout target_layout);

// Template Specialization for unpack_bfloat_tiles_into_float {bfp4,bfp8}
template <typename... Args>
inline std::vector<float> unpack_bfloat_tiles_into_float_vec(const bfloat8_b&, Args&&... args) {
    return unpack_bfp8_tiles_into_float_vec(std::forward<Args>(args)...);
}
template <typename... Args>
inline std::vector<float> unpack_bfloat_tiles_into_float_vec(const bfloat4_b&, Args&&... args) {
    return unpack_bfp4_tiles_into_float_vec(std::forward<Args>(args)...);
}

// Template Specialization for pack_fp32_vec_as_bfp4_tiles {bfp4,bfp8}
template <typename... Args>
inline std::vector<uint32_t> pack_fp32_vec_as_bfloat_tiles(const bfloat8_b&, Args&&... args) {
    return pack_fp32_vec_as_bfp8_tiles(std::forward<Args>(args)...);
}
template <typename... Args>
inline std::vector<uint32_t> pack_fp32_vec_as_bfloat_tiles(const bfloat4_b&, Args&&... args) {
    return pack_fp32_vec_as_bfp4_tiles(std::forward<Args>(args)...);
}

// Template specialization for BFloatLayout based on type T
template <typename T>
struct bfloat_enum;

template <>
struct bfloat_enum<bfloat8_b> {
    static constexpr DataType value = DataType::BFLOAT8_B;
};

template <>
struct bfloat_enum<bfloat4_b> {
    static constexpr DataType value = DataType::BFLOAT4_B;
};

template <typename T>
Tensor to_layout_bfloat(const Tensor& tensor, Layout target_layout) {
    static_assert(std::is_same_v<T, bfloat8_b> || std::is_same_v<T, bfloat4_b>, "Invalid type T");

    // TODO(arakhmati): do not convert to FLOA32

    if (tensor.get_layout() == target_layout) {
        return tensor;
    }
    return std::visit(
        [&tensor, &target_layout](auto&& storage) -> Tensor {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, MultiDeviceHostStorage>) {
                std::vector<OwnedBuffer> output_buffers;
                for (int i = 0; i < storage.num_buffers(); i++) {
                    // Convert to FLOAT32 tensor and change layout
                    auto input_packed_data = owned_buffer::get_as<uint32_t>(storage.get_buffer(i)).get();
                    auto input_float_data = unpack_bfloat_tiles_into_float_vec(
                        T{}, input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false);
                    auto input_float_buffer = owned_buffer::create<float>(std::move(input_float_data));
                    auto float_tensor = Tensor(
                                            OwnedStorage{input_float_buffer},
                                            tensor.get_legacy_shape(),
                                            DataType::FLOAT32,
                                            tensor.get_layout(),
                                            tensor.get_tile())
                                            .to(target_layout);

                    // Convert back to BFLOAT8_B
                    auto output_float_data = owned_buffer::get_as<float>(float_tensor).get();
                    auto output_packed_data = pack_fp32_vec_as_bfloat_tiles(
                        T{}, output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
                    auto output_uint32_buffer = owned_buffer::create<uint32_t>(std::move(output_packed_data));
                    output_buffers.push_back(output_uint32_buffer);
                }
                return Tensor(
                    std::move(MultiDeviceHostStorage{storage.strategy, output_buffers, storage.shapes}),
                    tensor.get_legacy_shape(),
                    bfloat_enum<T>::value,
                    target_layout,
                    tensor.get_tile());

            } else {
                // Convert to FLOAT32 tensor and change layout
                auto input_packed_data = owned_buffer::get_as<uint32_t>(tensor).get();
                auto input_float_data = unpack_bfloat_tiles_into_float_vec(
                    T{}, input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false);
                auto input_float_buffer = owned_buffer::create<float>(std::move(input_float_data));
                auto float_tensor = Tensor(
                                        OwnedStorage{input_float_buffer},
                                        tensor.get_legacy_shape(),
                                        DataType::FLOAT32,
                                        tensor.get_layout(),
                                        tensor.get_tile())
                                        .to(target_layout);

                // Convert back to BFLOAT
                auto output_float_data = owned_buffer::get_as<float>(float_tensor).get();
                auto output_packed_data = pack_fp32_vec_as_bfloat_tiles(
                    T{}, output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
                auto output_uint32_buffer = owned_buffer::create<uint32_t>(std::move(output_packed_data));
                return Tensor(
                    std::move(OwnedStorage{std::move(output_uint32_buffer)}),
                    tensor.get_legacy_shape(),
                    bfloat_enum<T>::value,
                    target_layout,
                    tensor.get_tile());
            }
        },
        tensor.get_storage());
}

template <>
Tensor to_layout<bfloat8_b>(const Tensor& tensor, Layout target_layout) {
    return to_layout_bfloat<bfloat8_b>(tensor, target_layout);
}

template <>
Tensor to_layout<bfloat4_b>(const Tensor& tensor, Layout target_layout) {
    return to_layout_bfloat<bfloat4_b>(tensor, target_layout);
}

// ======================================================================================
//                                  .pad() and .unpad()
// ======================================================================================

template <typename T>
Tensor pad(const Tensor& tensor, const tt::tt_metal::LegacyShape& output_shape, const ttnn::SimpleShape& input_tensor_start, float pad_value) {
    if (ttnn::distributed::is_multi_device_tensor(tensor)) {
        return transform(tensor, [&](const Tensor& device_tensor) {
            return pad<T>(device_tensor, output_shape, input_tensor_start, pad_value);
        });
    }

    auto pad_value_ = static_cast<T>(pad_value);
    const auto input_shape = tensor.get_legacy_shape();
    const auto input_strides = tensor.strides();
    const auto input_data_type = tensor.get_dtype();

    auto pad = [&input_shape, &input_strides, &input_data_type, &output_shape, &input_tensor_start, &pad_value_](
                   const auto& input_buffer) {
        auto compute_stride = [](const tt::tt_metal::LegacyShape& shape, uint32_t index) {
            uint32_t stride = 1;
            for (auto i = index + 1; i < shape.rank(); i++) {
                stride *= shape[i];
            }
            return stride;
        };

        std::vector<std::array<uint32_t, 2>> pad_size{};
        std::vector<uint32_t> input_strides{};
        std::vector<uint32_t> output_strides{};
        std::vector<uint32_t> input_indices(input_shape.rank(), 0);

        for (auto index = 0; index < output_shape.rank(); index++) {
            // Check if input tensor fits in output tensor given the input tensor start indices
            TT_ASSERT(
                input_shape[index] + input_tensor_start[index] <= output_shape[index], "Input tensor is out of bounds");

            // Figure out pad size on each dim
            pad_size.push_back(
                {input_tensor_start[index], output_shape[index] - input_shape[index] - input_tensor_start[index]});

            input_strides.push_back(compute_stride(input_shape, index));
            output_strides.push_back(compute_stride(output_shape, index));
        }

        auto flat_output_index = 0;
        auto output_buffer = owned_buffer::create<T>(compute_volume(output_shape));
        std::function<void(std::size_t)> pad_to_tile = [&](std::size_t dim) -> void {
            for (auto i = 0; i < pad_size[dim][0] * output_strides[dim]; i++) {
                output_buffer[flat_output_index++] = pad_value_;
            }

            for (auto i = 0; i < input_shape[dim]; i++) {
                input_indices[dim] = i;
                if (dim == input_shape.rank() - 1) {
                    auto flat_input_index = compute_flat_input_index(input_indices, input_strides);
                    output_buffer[flat_output_index++] = input_buffer[flat_input_index];
                } else {
                    pad_to_tile(dim + 1);
                }
            }

            for (auto i = 0; i < pad_size[dim][1] * output_strides[dim]; i++) {
                output_buffer[flat_output_index++] = pad_value_;
            }
        };
        pad_to_tile(0);

        return output_buffer;
    };

    auto output_buffer = std::visit(
        [&pad](auto&& storage) -> owned_buffer::Buffer<T> {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                const auto input_data = owned_buffer::get_as<T>(storage.buffer);
                return pad(input_data);
            } else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                const auto input_data = borrowed_buffer::get_as<T>(storage.buffer);
                return pad(input_data);
            } else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceHostStorage>) {
                TT_THROW("Device storage isn't supported");
            } else {
                raise_unsupported_storage<StorageType>();
            }
        },
        tensor.get_storage());
    return Tensor(OwnedStorage{output_buffer}, output_shape, tensor.get_dtype(), tensor.get_layout(), tensor.get_tile());
}

template Tensor pad<bfloat16>(
    const Tensor& tensor, const tt::tt_metal::LegacyShape& output_shape, const ttnn::SimpleShape& input_tensor_start, float pad_value);
template Tensor pad<float>(
    const Tensor& tensor, const tt::tt_metal::LegacyShape& output_shape, const ttnn::SimpleShape& input_tensor_start, float pad_value);
template Tensor pad<int32_t>(
    const Tensor& tensor, const tt::tt_metal::LegacyShape& output_shape, const ttnn::SimpleShape& input_tensor_start, float pad_value);
template Tensor pad<uint32_t>(
    const Tensor& tensor, const tt::tt_metal::LegacyShape& output_shape, const ttnn::SimpleShape& input_tensor_start, float pad_value);
template Tensor pad<uint16_t>(
    const Tensor& tensor, const tt::tt_metal::LegacyShape& output_shape, const ttnn::SimpleShape& input_tensor_start, float pad_value);
template Tensor pad<uint8_t>(
    const Tensor& tensor, const tt::tt_metal::LegacyShape& output_shape, const ttnn::SimpleShape& input_tensor_start, float pad_value);

template <>
Tensor pad<bfloat8_b>(
    const Tensor& tensor, const tt::tt_metal::LegacyShape& output_shape, const ttnn::SimpleShape& input_tensor_start, float pad_value) {
    return pad_bfloat8_b(tensor, output_shape, input_tensor_start, pad_value);
}

template <>
Tensor pad<bfloat4_b>(
    const Tensor& tensor, const tt::tt_metal::LegacyShape& output_shape, const ttnn::SimpleShape& input_tensor_start, float pad_value) {
    return pad_bfloat4_b(tensor, output_shape, input_tensor_start, pad_value);
}

template <typename T>
Tensor unpad(const Tensor& tensor, const ttnn::SimpleShape& output_tensor_start, const ttnn::SimpleShape& output_tensor_end) {
    const auto input_shape = tensor.get_legacy_shape();
    const auto input_strides = tensor.strides();

    // Validate inputs and compute output shape
    std::vector<uint32_t> output_shape{};
    for (auto i = 0; i < input_shape.rank(); i++) {
        // Check if tensor start and end indices are within input tensor shape
        TT_ASSERT(output_tensor_start[i] < input_shape[i]);
        TT_ASSERT(output_tensor_end[i] <= input_shape[i]);
        // Check if start shape is < end shape
        TT_ASSERT(output_tensor_start[i] < output_tensor_end[i]);
        // Figure out output tensor shape
        output_shape.push_back(output_tensor_end[i] - output_tensor_start[i]);
    }

    auto unpad = [&input_shape, &input_strides, &output_shape, &output_tensor_start, &output_tensor_end](
                     const auto& input_buffer) {
        std::vector<uint32_t> input_indices(input_shape.rank(), 0);

        auto flat_output_index = 0;
        auto output_buffer = owned_buffer::create<T>(compute_volume(output_shape));

        std::function<void(std::size_t)> unpad_from_tile = [&](std::size_t dim) -> void {
            for (auto i = output_tensor_start[dim]; i < output_tensor_end[dim]; i++) {
                input_indices[dim] = i;
                if (dim == input_shape.rank() - 1) {
                    auto flat_input_index = compute_flat_input_index(input_indices, input_strides);
                    output_buffer[flat_output_index++] = input_buffer[flat_input_index];
                } else {
                    unpad_from_tile(dim + 1);
                }
            }
        };
        unpad_from_tile(0);

        return output_buffer;
    };

    auto output_buffer = std::visit(
        [&unpad](auto&& storage) -> owned_buffer::Buffer<T> {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                const auto input_data = owned_buffer::get_as<T>(storage.buffer);
                return unpad(input_data);
            } else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                const auto input_data = borrowed_buffer::get_as<T>(storage.buffer);
                return unpad(input_data);
            } else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceHostStorage>) {
                TT_THROW("Device storage isn't supported");
            } else {
                raise_unsupported_storage<StorageType>();
            }
        },
        tensor.get_storage());
    return Tensor(OwnedStorage{output_buffer}, output_shape, tensor.get_dtype(), tensor.get_layout(), tensor.get_tile());
}

template Tensor unpad<bfloat16>(const Tensor& tensor, const ttnn::SimpleShape& output_tensor_start, const ttnn::SimpleShape& output_tensor_end);
template Tensor unpad<float>(const Tensor& tensor, const ttnn::SimpleShape& output_tensor_start, const ttnn::SimpleShape& output_tensor_end);
template Tensor unpad<int32_t>(const Tensor& tensor, const ttnn::SimpleShape& output_tensor_start, const ttnn::SimpleShape& output_tensor_end);
template Tensor unpad<uint32_t>(const Tensor& tensor, const ttnn::SimpleShape& output_tensor_start, const ttnn::SimpleShape& output_tensor_end);
template Tensor unpad<uint16_t>(const Tensor& tensor, const ttnn::SimpleShape& output_tensor_start, const ttnn::SimpleShape& output_tensor_end);
template Tensor unpad<uint8_t>(const Tensor& tensor, const ttnn::SimpleShape& output_tensor_start, const ttnn::SimpleShape& output_tensor_end);

template <>
Tensor unpad<bfloat8_b>(const Tensor& tensor, const ttnn::SimpleShape& output_tensor_start, const ttnn::SimpleShape& output_tensor_end) {
    return unpad_bfloat8_b(tensor, output_tensor_start, output_tensor_end);
}

template <>
Tensor unpad<bfloat4_b>(const Tensor& tensor, const ttnn::SimpleShape& output_tensor_start, const ttnn::SimpleShape& output_tensor_end) {
    return unpad_bfloat4_b(tensor, output_tensor_start, output_tensor_end);
}

// ======================================================================================
//                                  .extract_shard()
// ======================================================================================

template <typename T>
Tensor extract_shard(const Tensor& tensor, const uint32_t& core_id) {
    auto buffer = tensor.buffer();
    auto buffer_shard_shape = buffer->shard_spec().shape();
    std::array<uint32_t, 4> shard_shape_array = {1, 1, buffer_shard_shape[0], buffer_shard_shape[1]};
    tt::tt_metal::LegacyShape shard_shape(shard_shape_array);
    std::vector<uint32_t> device_data;
    ::detail::ReadShard(*buffer, device_data, core_id);

    auto unpacked_data = tensor_impl::unpack_uint32_vec<T>(device_data);
    auto output_buffer = owned_buffer::create<T>(std::move(unpacked_data));
    return Tensor(OwnedStorage{output_buffer}, shard_shape, tensor.get_dtype(), tensor.get_layout(), tensor.get_tile());
}

template Tensor extract_shard<bfloat16>(const Tensor& tensor, const uint32_t& core_id);
template Tensor extract_shard<float>(const Tensor& tensor, const uint32_t& core_id);
template Tensor extract_shard<int32_t>(const Tensor& tensor, const uint32_t& core_id);
template Tensor extract_shard<uint32_t>(const Tensor& tensor, const uint32_t& core_id);
template Tensor extract_shard<uint16_t>(const Tensor& tensor, const uint32_t& core_id);
template Tensor extract_shard<uint8_t>(const Tensor& tensor, const uint32_t& core_id);

template <>
Tensor extract_shard<bfloat8_b>(const Tensor& tensor, const uint32_t& core_id) {
    return extract_shard<uint32_t>(tensor, core_id);
}

template <>
Tensor extract_shard<bfloat4_b>(const Tensor& tensor, const uint32_t& core_id) {
    return extract_shard<uint32_t>(tensor, core_id);
}


}  // namespace tensor_impl

}  // namespace tt_metal

}  // namespace tt
