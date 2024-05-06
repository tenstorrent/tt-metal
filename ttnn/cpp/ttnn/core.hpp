// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <csignal>
#include <optional>

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tensor/tensor_impl.hpp"  // TTNN_TENSOR_PRINT_PROFILE
#include "tt_eager/tensor/types.hpp"
#include "tt_eager/tt_dnn/op_library/operation.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

using tt::tt_metal::operation::OptionalConstTensors;
using tt::tt_metal::operation::OptionalTensors;
using tt::tt_metal::operation::Tensors;

using tt::tt_metal::any_tensor_on_multi_device;
using tt::tt_metal::is_tensor_on_device;
using tt::tt_metal::is_tensor_on_device_or_multidevice;
using tt::tt_metal::is_tensor_on_multi_device;
}  // namespace ttnn

namespace ttnn {

namespace core {

struct Config {
    std::string cache_path = "/home/.cache/ttnn";
    std::string model_cache_path = "/home/.cache/ttnn/models";
    std::string tmp_dir = "/tmp/ttnn";
    bool enable_model_cache = false;
    bool enable_fast_runtime_mode = false;
    bool throw_exception_on_fallback = false;
    bool enable_logging = false;
    bool enable_graph_report = false;
    bool enable_detailed_buffer_report = false;
    bool enable_detailed_tensor_report = false;
    bool enable_comparison_mode = false;
    float comparison_mode_pcc = 0.9999;
    std::string root_report_path = "generated/ttnn/reports";
    std::optional<std::string> report_name = std::nullopt;

    static constexpr auto attribute_names = std::make_tuple(
        "cache_path",
        "model_cache_path",
        "tmp_dir",
        "enable_model_cache",
        "enable_fast_runtime_mode",
        "throw_exception_on_fallback",
        "enable_logging",
        "enable_graph_report",
        "enable_detailed_buffer_report",
        "enable_detailed_tensor_report",
        "enable_comparison_mode",
        "comparison_mode_pcc",
        "root_report_path",
        "report_name");

    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->cache_path),
            std::cref(this->model_cache_path),
            std::cref(this->tmp_dir),
            std::cref(this->enable_model_cache),
            std::cref(this->enable_fast_runtime_mode),
            std::cref(this->throw_exception_on_fallback),
            std::cref(this->enable_logging),
            std::cref(this->enable_graph_report),
            std::cref(this->enable_detailed_buffer_report),
            std::cref(this->enable_detailed_tensor_report),
            std::cref(this->enable_comparison_mode),
            std::cref(this->comparison_mode_pcc),
            std::cref(this->root_report_path),
            std::cref(this->report_name));
    }
};

inline Config CONFIG{};

inline std::uint32_t pad_to_multiple_of_tile_size(std::uint32_t value) {
    return (value + (ttnn::TILE_SIZE - 1)) / ttnn::TILE_SIZE * ttnn::TILE_SIZE;
}

inline bool has_storage_type_of(const ttnn::Tensor& tensor, const ttnn::StorageType& storage_type) {
    return tensor.storage_type() == storage_type;
}

inline std::optional<ttnn::MemoryConfig> get_memory_config(const ttnn::Tensor& tensor) {
    if (not tensor.is_allocated() or not is_tensor_on_device_or_multidevice(tensor)) {
        return std::nullopt;
    }
    return tensor.memory_config();
}

inline void set_printoptions(const std::string& profile) {
    tt::tt_metal::tensor_impl::TTNN_TENSOR_PRINT_PROFILE =
        magic_enum::enum_cast<tt::tt_metal::tensor_impl::TensorPrintProfile>(profile, [](char lhs, char rhs) {
            return std::tolower(lhs) == std::tolower(rhs);
        }).value();
}

inline void segfault_handler(int sig) {
    std::cerr << tt::assert::backtrace_to_string() << std::endl;
    exit(EXIT_FAILURE);
}

inline void dump_stack_trace_on_segfault() {
    if (std::signal(SIGSEGV, segfault_handler) == SIG_ERR) {
        std::cerr << "Error: cannot handle SIGSEGV" << std::endl;
        exit(EXIT_FAILURE);
    }
}

}  // namespace core

using core::CONFIG;
using core::Config;
using core::get_memory_config;
using core::has_storage_type_of;
using core::pad_to_multiple_of_tile_size;
using core::set_printoptions;
}  // namespace ttnn
