// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <csignal>
#include <optional>

#include "magic_enum.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"  // TTNN_TENSOR_PRINT_PROFILE
#include "ttnn/tensor/types.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/config.hpp"
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

using core::get_memory_config;
using core::has_storage_type_of;
using core::pad_to_multiple_of_tile_size;
using core::set_printoptions;

class CoreIDs{
    public:
        static CoreIDs& instance() {
            static CoreIDs instance;
            return instance;
        }

        std::int64_t get_python_operation_id();
        void set_python_operation_id(std::int64_t operation_id);
        std::int64_t fetch_and_increment_python_operation_id();

        std::int64_t get_tensor_id();
        void set_tensor_id(std::int64_t tensor_id);
        std::int64_t fetch_and_increment_tensor_id();

        std::int64_t get_device_operation_id();
        void set_device_operation_id(std::int64_t device_operation_id);
        std::int64_t fetch_and_increment_device_operation_id();

    private:
        CoreIDs() = default;
        ~CoreIDs() = default;
        std::atomic<std::int64_t> tensor_id;
        std::atomic<std::int64_t> python_operation_id;
        std::atomic<std::int64_t> device_operation_id;
};


}  // namespace ttnn
