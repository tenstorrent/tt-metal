// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/core.hpp"
#include <tt_stl/caseless_comparison.hpp>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/constants.hpp>

#include <tt-metalium/host_api.hpp>

namespace ttnn::core {

// returns true if padded tensor have other physical shape
bool is_padding_makes_sense(const ttnn::Tensor& tensor) {
    auto round_up = [](size_t value, size_t multiple) -> size_t {
        if (multiple == 0) {
            return value;
        }
        return ((value + multiple - 1) / multiple) * multiple;
    };

    auto [h, w] = tensor.tensor_spec().physical_shape().attribute_values();
    return (h > 0) && (w > 0) &&
           (!(h == round_up(h, tt::constants::TILE_HEIGHT) && w == round_up(w, tt::constants::TILE_WIDTH)));
}

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
    tt::tt_metal::tensor_impl::TTNN_PRINT_OPTIONS.profile = print_profile;
    tt::tt_metal::tensor_impl::TTNN_PRINT_OPTIONS.sci_mode = sci_mode;
    tt::tt_metal::tensor_impl::TTNN_PRINT_OPTIONS.precision = precision;
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

QueueId get_current_command_queue_id_for_thread() { return QueueId(tt::tt_metal::GetCurrentCommandQueueIdForThread()); }
void push_current_command_queue_id_for_thread(QueueId cq_id) {
    tt::tt_metal::PushCurrentCommandQueueIdForThread(cq_id.get());
}
QueueId pop_current_command_queue_id_for_thread() { return QueueId(tt::tt_metal::PopCurrentCommandQueueIdForThread()); }

ScopeGuard with_command_queue_id(QueueId cq_id) {
    push_current_command_queue_id_for_thread(cq_id);
    return make_guard([cq_id]() { pop_current_command_queue_id_for_thread(); });
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
