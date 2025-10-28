#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "tt-metalium/shape.hpp"
#include "tt-metalium/tensor/types.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "tt-metalium/tensor/layout/tensor_layout.hpp"
#include <tt_stl/span.hpp>
#include <sstream>

namespace ttnn {
using namespace tt::tt_metal;
using namespace tt::tt_metal::tensor_impl;

PrintOptions TTNN_PRINT_OPTIONS;

// ======================================================================================
//                                      .to_string()
// ======================================================================================

namespace detail {

struct DimensionShortener {
    size_t size{};
    std::optional<std::size_t> max;

    bool print_parenthesis_and_advance_index_if_reached_half_of_max_and_check_if_loop_is_done(
        std::ostream& ss, std::size_t& index, const std::string& before, const std::string& after) const {
        if (this->max.has_value() and this->size > this->max.value() and index == this->max.value() / 2) {
            ss << before << "...," << after;
            index = this->size - (this->max.value() / 2);
        }
        return index < this->size;
    }
};

inline DimensionShortener get_dimension_shortener(std::size_t size) {
    switch (TTNN_PRINT_OPTIONS.profile) {
        case TensorPrintProfile::Empty: return DimensionShortener{size, 0};
        case TensorPrintProfile::Short: return DimensionShortener{size, 4};
        case TensorPrintProfile::Full: return DimensionShortener{size, std::nullopt};
        default: TT_THROW("Unrecognized TTNN_TENSOR_PRINT_PROFILE {}", TTNN_PRINT_OPTIONS.profile);
    }
}

inline void print_trailing_comma(std::ostream& ss, std::size_t index, std::size_t size, const std::string& after) {
    if (index < size - 1) {
        ss << "," << after;
    }
}

template <typename T>
inline void print_datum(std::ostream& ss, T datum, bool use_scientific = false) {
    if (std::is_integral<T>::value) {
        ss << std::setw(5) << datum;
    } else {
        int precision = TTNN_PRINT_OPTIONS.precision;
        if (use_scientific) {
            // Note: scientific required fixed width + 4 (e+/-AB, e.g. 1.23456e+08)
            ss << std::scientific << std::setw(precision + 7) << std::setprecision(precision) << datum;
        } else {
            ss << std::fixed << std::setw(precision + 3) << std::setprecision(precision) << datum;
        }
    }
}

template <>
inline void print_datum(std::ostream& ss, bfloat16 datum, bool use_scientific) {
    print_datum(ss, static_cast<float>(datum), use_scientific);
}

template <>
inline void print_datum(std::ostream& ss, uint8_t datum, bool use_scientific) {
    print_datum<uint32_t>(ss, datum, use_scientific);
}

// Helper function to determine if scientific notation should be used
template <typename T>
bool should_use_scientific_notation(tt::stl::Span<const T> buffer) {
    if (TTNN_PRINT_OPTIONS.sci_mode == SciMode::Enable) {
        return true;
    }
    if (TTNN_PRINT_OPTIONS.sci_mode == SciMode::Disable) {
        return false;
    }

    // SciMode::Default - auto-detect based on data range
    if constexpr (std::is_integral_v<T>) {
        return false;  // Never use scientific notation for integers
    } else {
        double nonzero_finite_min = std::numeric_limits<double>::max();
        double nonzero_finite_max = std::numeric_limits<double>::lowest();
        bool found_nonzero_finite = false;

        for (const auto& value : buffer) {
            double val = static_cast<double>(value);
            if (std::isfinite(val) && val != 0.0) {
                double abs_val = std::abs(val);
                nonzero_finite_min = std::min(nonzero_finite_min, abs_val);
                nonzero_finite_max = std::max(nonzero_finite_max, abs_val);
                found_nonzero_finite = true;
            }
        }

        if (!found_nonzero_finite) {
            return false;  // No nonzero finite values, don't use scientific notation
        }

        return (nonzero_finite_max / nonzero_finite_min > 1000.0) || (nonzero_finite_max > 1.0e8) ||
               (nonzero_finite_min < 1.0e-4);
    }
}

constexpr int constexpr_strlen(const char* str) { return *str ? 1 + constexpr_strlen(str + 1) : 0; }

constexpr auto TENSOR_TYPE_STRING = "ttnn.Tensor";
constexpr auto TENSOR_TYPE_STRING_PLUS_OPEN_PARENTHESIS_LENGTH = constexpr_strlen(TENSOR_TYPE_STRING) + 1;

template <typename T>
void to_string_row_major(
    std::stringstream& ss,
    tt::stl::Span<const T> buffer,
    const ttnn::Shape& shape,
    const tt::tt_metal::Strides& strides,
    std::size_t outer_index,
    const std::size_t buffer_offset,
    int64_t rank,
    int64_t dim,
    bool use_scientific) {
    auto stride = dim < strides.size() ? strides[dim] : 0;

    std::string spaces = std::string(TENSOR_TYPE_STRING_PLUS_OPEN_PARENTHESIS_LENGTH + dim, ' ');
    std::string before;
    std::string after;
    if (rank == 1) {
        before = " ";
        after = " ";
    } else if (rank == 2) {
        before = spaces + " ";
        after = "\n";
    } else {
        before = spaces + " ";
        after = "\n\n";
    }

    if (dim > 0 and outer_index > 0) {
        ss << spaces;
    }
    if (rank != 0) {
        ss << "[";
    }
    auto dimension_shortener = get_dimension_shortener(rank != 0 ? shape[-rank] : 1);
    for (std::size_t index = 0;
         dimension_shortener.print_parenthesis_and_advance_index_if_reached_half_of_max_and_check_if_loop_is_done(
             ss, index, before, after);
         index++) {
        std::string after_comma;
        if (rank == 1) {
            after_comma = " ";
        } else if (rank == 2) {
            after_comma = "\n";
        } else {
            after_comma = after;
        }

        if (rank > 1) {
            to_string_row_major(
                ss, buffer, shape, strides, index, buffer_offset + (index * stride), rank - 1, dim + 1, use_scientific);
        } else {
            print_datum(ss, buffer[buffer_offset + index], use_scientific);
        }
        print_trailing_comma(ss, index, rank != 0 ? shape[-rank] : 1, after_comma);
    }
    if (rank != 0) {
        ss << "]";
    }
}

template <typename T>
void to_string(
    std::stringstream& ss,
    tt::stl::Span<const T> buffer,
    const ttnn::Shape& shape,
    const tt::tt_metal::Strides& strides,
    DataType dtype,
    Layout layout) {
    ss << TENSOR_TYPE_STRING << "(";

    if (TTNN_PRINT_OPTIONS.profile == TensorPrintProfile::Empty) {
        ss << "...";
    } else {
        bool use_scientific = should_use_scientific_notation<T>(buffer);
        to_string_row_major<T>(ss, buffer, shape, strides, 0, 0, shape.rank(), 0, use_scientific);
    }
    ss << ", shape=" << fmt::format("{}", shape) << ", dtype=" << fmt::format("{}", dtype)
       << ", layout=" << fmt::format("{}", layout) << ")";
}

}  // namespace detail

template <typename T>
std::string to_string(const Tensor& tensor) {
    const auto& shape = tensor.logical_shape();

    if (!tensor.is_allocated()) {
        return fmt::format(
            "{}(<buffer is not allocated>, shape={}, dtype={}, layout={})",
            detail::TENSOR_TYPE_STRING,
            shape,
            tensor.dtype(),
            tensor.layout());
    }

    auto get_row_major_tensor = [&](const Tensor& tensor) -> Tensor {
        if (tensor.layout() == Layout::ROW_MAJOR) {
            return tensor;
        } else if (tensor.dtype() == DataType::BFLOAT8_B || tensor.dtype() == DataType::BFLOAT4_B) {
            return ttnn::to_layout(ttnn::to_dtype(tensor, DataType::FLOAT32), Layout::ROW_MAJOR);
        } else {
            return ttnn::to_layout(tensor, Layout::ROW_MAJOR);
        }
    };

    auto get_device_buffers = [&](const HostStorage& storage) {
        std::vector<HostBuffer> buffers;
        storage.buffer().apply([&](const HostBuffer& shard) { buffers.push_back(shard); });
        return buffers;
    };

    return std::visit(
        tt::stl::overloaded{
            [&](const HostStorage& storage) -> std::string {
                // TODO: Call tt-metal implementation
                return tensor.write_to_string();
            },
            [&](const DeviceStorage& storage) -> std::string {
                auto cpu_tensor = tensor.cpu();
                if (storage.mesh_buffer == nullptr) {
                    // Use owned buffer path above.
                    return to_string<T>(cpu_tensor);
                }

                auto* mesh_device = storage.mesh_buffer->device();
                if (mesh_device->num_devices() == 1) {
                    return to_string<T>(ttnn::distributed::get_device_tensors(cpu_tensor).at(0));
                }

                const Tensor row_major_tensor = get_row_major_tensor(cpu_tensor);
                const auto strides = row_major_tensor.tensor_spec().compute_strides();
                const auto& coords = storage.coords;
                auto coords_it = coords.begin();
                const std::vector<HostBuffer> buffers = get_device_buffers(row_major_tensor.host_storage());
                std::stringstream ss;
                for (size_t i = 0; i < buffers.size(); i++) {
                    const distributed::MeshCoordinate coord = *coords_it++;
                    if (mesh_device->is_local(coord)) {
                        ss << "device_id: " << mesh_device->get_device(coord)->id() << ", " << coord << std::endl;
                        detail::to_string(ss, buffers[i].view_as<T>(), shape, strides, tensor.dtype(), tensor.layout());
                    }
                    if (i + 1 != buffers.size()) {
                        ss << std::endl;
                    }
                }
                return ss.str();
            }},
        tensor.storage());
}

template std::string to_string<bfloat16>(const Tensor& tensor);
template std::string to_string<float>(const Tensor& tensor);
template std::string to_string<int32_t>(const Tensor& tensor);
template std::string to_string<uint32_t>(const Tensor& tensor);
template std::string to_string<uint16_t>(const Tensor& tensor);
template std::string to_string<uint8_t>(const Tensor& tensor);

template <>
std::string to_string<bfloat8_b>(const Tensor& tensor) {
    return to_string<float>(tensor);
}

template <>
std::string to_string<bfloat4_b>(const Tensor& tensor) {
    return to_string<float>(tensor);
}

}  // namespace ttnn
