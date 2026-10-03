// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/core.hpp"
#include <tt_stl/caseless_comparison.hpp>
#include <enchantum/enchantum.hpp>

#include <tt_stl/assert.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace ttnn::core {

namespace {

// TTNN-owned per-thread stack of "current command queue id".
// Metal has no implicit queue state: this stack is only consulted by TTNN (see
// current_mesh_command_queue). It is deliberately not tied to any MetalContext/device, so it can be used
// before a device is opened and never creates a context as a side effect. An empty stack means cq 0.
thread_local std::vector<uint8_t> current_command_queue_id_stack;

}  // namespace

bool has_storage_type_of(const ttnn::Tensor& tensor, const ttnn::StorageType& storage_type) {
    return tensor.storage_type() == storage_type;
}

std::optional<ttnn::MemoryConfig> get_memory_config(const ttnn::Tensor& tensor) {
    if (not tensor.is_allocated() or not is_device_tensor(tensor)) {
        return std::nullopt;
    }
    return tensor.memory_config();
}

void set_printoptions(TensorPrintProfile print_profile, SciMode sci_mode, int precision) {
    ttnn::tensor_impl::TTNN_PRINT_OPTIONS.profile = print_profile;
    ttnn::tensor_impl::TTNN_PRINT_OPTIONS.sci_mode = sci_mode;
    ttnn::tensor_impl::TTNN_PRINT_OPTIONS.precision = precision;
}

void segfault_handler(int /*sig*/) {
    std::cerr << tt::assert::backtrace_to_string() << std::endl;
    exit(EXIT_FAILURE);
}

void dump_stack_trace_on_segfault() {
    if (std::signal(SIGSEGV, segfault_handler) == SIG_ERR) {
        std::cerr << "Error: cannot handle SIGSEGV" << std::endl;
        exit(EXIT_FAILURE);
    }
}

QueueId get_current_command_queue_id_for_thread() {
    if (current_command_queue_id_stack.empty()) {
        return QueueId(0);
    }
    return QueueId(current_command_queue_id_stack.back());
}

void push_current_command_queue_id_for_thread(QueueId cq_id) { current_command_queue_id_stack.push_back(cq_id.get()); }

QueueId pop_current_command_queue_id_for_thread() {
    TT_FATAL(!current_command_queue_id_stack.empty(), "Current command queue id stack is empty!");
    QueueId cq_id(current_command_queue_id_stack.back());
    current_command_queue_id_stack.pop_back();
    return cq_id;
}

ScopeGuard with_command_queue_id(QueueId cq_id) {
    push_current_command_queue_id_for_thread(cq_id);
    return make_guard([cq_id]() { pop_current_command_queue_id_for_thread(); });
}

tt::tt_metal::distributed::MeshCommandQueue& current_mesh_command_queue(
    tt::tt_metal::distributed::MeshDevice& mesh_device, std::optional<QueueId> cq_id) {
    return mesh_device.mesh_command_queue(cq_id.value_or(get_current_command_queue_id_for_thread()).get());
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

std::int64_t CoreIDs::get_device_operation_id() { return device_operation_id.load(); }
void CoreIDs::set_device_operation_id(std::int64_t device_operation_id_) { device_operation_id = device_operation_id_; }
std::int64_t CoreIDs::fetch_and_increment_device_operation_id() { return device_operation_id.fetch_add(1); }

}  // namespace ttnn
