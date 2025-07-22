// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/core.hpp"
#include <tt_stl/caseless_comparison.hpp>
#include <enchantum/enchantum.hpp>

namespace ttnn::core {

bool has_storage_type_of(const ttnn::Tensor& tensor, const ttnn::StorageType& storage_type) {
    return tensor.storage_type() == storage_type;
}

std::optional<ttnn::MemoryConfig> get_memory_config(const ttnn::Tensor& tensor) {
    if (not tensor.is_allocated() or not is_device_tensor(tensor)) {
        return std::nullopt;
    }
    return tensor.memory_config();
}

void set_printoptions(const std::string& profile) {
    tt::tt_metal::tensor_impl::TTNN_TENSOR_PRINT_PROFILE =
        enchantum::cast<tt::tt_metal::tensor_impl::TensorPrintProfile>(profile,ttsl::ascii_caseless_comp).value();
}

void segfault_handler(int sig) {
    std::cerr << tt::assert::backtrace_to_string() << std::endl;
    exit(EXIT_FAILURE);
}

void dump_stack_trace_on_segfault() {
    if (std::signal(SIGSEGV, segfault_handler) == SIG_ERR) {
        std::cerr << "Error: cannot handle SIGSEGV" << std::endl;
        exit(EXIT_FAILURE);
    }
}
}  // namespace ttnn::core

namespace ttnn {

CoreIDs& CoreIDs::instance() {
    static CoreIDs instance;
    return instance;
}

std::int64_t CoreIDs::get_python_operation_id() { return python_operation_id.load(); }
void CoreIDs::set_python_operation_id(std::int64_t python_operation_id_) { python_operation_id = python_operation_id_; }
std::int64_t CoreIDs::fetch_and_increment_python_operation_id() { return python_operation_id.fetch_add(1); }

std::int64_t CoreIDs::get_tensor_id() { return tensor_id.load(); }
void CoreIDs::set_tensor_id(std::int64_t tensor_id_) { tensor_id = tensor_id_; }
std::int64_t CoreIDs::fetch_and_increment_tensor_id() { return tensor_id.fetch_add(1); }

std::int64_t CoreIDs::get_device_operation_id() { return device_operation_id.load(); }
void CoreIDs::set_device_operation_id(std::int64_t device_operation_id_) { device_operation_id = device_operation_id_; }
std::int64_t CoreIDs::fetch_and_increment_device_operation_id() { return device_operation_id.fetch_add(1); }

}  // namespace ttnn
