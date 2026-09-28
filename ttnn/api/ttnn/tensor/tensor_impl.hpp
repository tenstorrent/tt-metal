// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <optional>
#include <span>
#include <utility>

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"

namespace ttnn::tensor_impl {

// Empty structs to facilitate Tensor template logic.
struct bfloat4_b {};
struct bfloat8_b {};
// One tag for all MX formats (packed MX tiles); the concrete format comes from the runtime DataType.
struct mx_tiles {};

// Utility to convert runtime DataType to compile-time constant and dispatch the function call
template <typename Func, typename... Args>
auto dispatch(tt::tt_metal::DataType dtype, Func&& func, Args&&... args) {
    switch (dtype) {
        case tt::tt_metal::DataType::BFLOAT16:
            return (std::forward<Func>(func)).template operator()<bfloat16>(std::forward<Args>(args)...);
        case tt::tt_metal::DataType::FLOAT32:
            return (std::forward<Func>(func)).template operator()<float>(std::forward<Args>(args)...);
        case tt::tt_metal::DataType::INT32:
            return (std::forward<Func>(func)).template operator()<int32_t>(std::forward<Args>(args)...);
        case tt::tt_metal::DataType::INT8:
            return (std::forward<Func>(func)).template operator()<int8_t>(std::forward<Args>(args)...);
        case tt::tt_metal::DataType::UINT32:
            return (std::forward<Func>(func)).template operator()<uint32_t>(std::forward<Args>(args)...);
        case tt::tt_metal::DataType::UINT16:
            return (std::forward<Func>(func)).template operator()<uint16_t>(std::forward<Args>(args)...);
        case tt::tt_metal::DataType::UINT8:
            return (std::forward<Func>(func)).template operator()<uint8_t>(std::forward<Args>(args)...);
        case tt::tt_metal::DataType::BFLOAT8_B:
            return (std::forward<Func>(func)).template operator()<bfloat8_b>(std::forward<Args>(args)...);
        case tt::tt_metal::DataType::BFLOAT4_B:
            return (std::forward<Func>(func)).template operator()<bfloat4_b>(std::forward<Args>(args)...);
        case tt::tt_metal::DataType::FP8_E4M3:
            return (std::forward<Func>(func)).template operator()<float8_e4m3>(std::forward<Args>(args)...);
        case tt::tt_metal::DataType::MXFP8_E4M3:
        case tt::tt_metal::DataType::MXFP8_E5M2:
        case tt::tt_metal::DataType::MXFP6_E2M3:
        case tt::tt_metal::DataType::MXFP6_E3M2:
        case tt::tt_metal::DataType::MXFP4:
        case tt::tt_metal::DataType::MXINT8:
        case tt::tt_metal::DataType::MXINT4:
        case tt::tt_metal::DataType::MXINT2:
            return (std::forward<Func>(func)).template operator()<mx_tiles>(std::forward<Args>(args)...);
        default: TT_THROW("Unsupported data type");
    }
}

// ===============================================================================================================================================
//                                                              High Level APIs
// ===============================================================================================================================================

// ======================================================================================
//                                  .view()
// ======================================================================================

tt::tt_metal::HostTensor view(
    const tt::tt_metal::HostTensor& tensor,
    const tt::tt_metal::Shape& new_logical_shape,
    const tt::tt_metal::Shape& new_padded_shape);

// ======================================================================================
//                                         Print
// ======================================================================================

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::DataType& dtype);

enum class TensorPrintProfile {
    Empty,
    Short,
    Full,
};

enum class SciMode {
    Enable,
    Disable,
    Default,
};

struct PrintOptions {
    TensorPrintProfile profile = TensorPrintProfile::Short;
    SciMode sci_mode = SciMode::Default;
    int precision = 4;
};

extern PrintOptions TTNN_PRINT_OPTIONS;

std::string to_string(const Tensor& tensor);

Tensor extract_shard(const Tensor& tensor, const uint32_t& core_id);

}  // namespace ttnn::tensor_impl
