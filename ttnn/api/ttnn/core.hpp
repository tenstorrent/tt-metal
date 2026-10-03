// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <csignal>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <utility>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"  // TTNN_TENSOR_PRINT_PROFILE
#include "ttnn/tensor/types.hpp"
#include "ttnn/config.hpp"
#include "ttnn/types.hpp"
#include "ttnn/common/guard.hpp"
#include "ttnn/common/queue_id.hpp"

namespace tt::tt_metal::distributed {
class MeshCommandQueue;
class MeshDevice;
}  // namespace tt::tt_metal::distributed

namespace ttnn {

using OptionalConstTensors = std::vector<std::optional<const Tensor>>;
using OptionalTensors = std::vector<std::optional<Tensor>>;
using Tensors = std::vector<Tensor>;
using TensorPrintProfile = ttnn::tensor_impl::TensorPrintProfile;
using SciMode = ttnn::tensor_impl::SciMode;

}  // namespace ttnn

namespace ttnn {

namespace core {

bool has_storage_type_of(const ttnn::Tensor& tensor, const ttnn::StorageType& storage_type);

std::optional<ttnn::MemoryConfig> get_memory_config(const ttnn::Tensor& tensor);

void set_printoptions(TensorPrintProfile print_profile, SciMode sci_mode = SciMode::Default, int precision = 4);

void segfault_handler(int sig);

void dump_stack_trace_on_segfault();

// Thread-local "current command queue id" selection.
//
// TTNN owns this per-thread stack; Metal has no implicit queue state (MeshDevice::mesh_command_queue() with no
// argument always means cq 0). TTNN entry points that take an optional cq_id resolve a missing value through
// current_mesh_command_queue() below, which is what makes `with_command_queue_id` / `ttnn.command_queue` work.
// The stack is not tied to a device or MetalContext: get returns 0 when nothing has been pushed, and
// push/pop never create a context. pop on an empty stack is a fatal error.
QueueId get_current_command_queue_id_for_thread();
void push_current_command_queue_id_for_thread(QueueId cq_id);
QueueId pop_current_command_queue_id_for_thread();

ScopeGuard with_command_queue_id(QueueId cq_id);

template <typename T>
void with_command_queue_id(QueueId cq_id, T&& func) {
    auto guard = with_command_queue_id(cq_id);
    std::forward<T>(func)();
}

// Returns the mesh command queue TTNN should dispatch to: `cq_id` if provided, otherwise the thread's current
// command queue id (cq 0 when none has been selected). Use this instead of calling
// `mesh_device.mesh_command_queue()` without an explicit id.
tt::tt_metal::distributed::MeshCommandQueue& current_mesh_command_queue(
    tt::tt_metal::distributed::MeshDevice& mesh_device, std::optional<QueueId> cq_id = std::nullopt);

}  // namespace core

using core::current_mesh_command_queue;
using core::get_current_command_queue_id_for_thread;
using core::get_memory_config;
using core::has_storage_type_of;
using core::pop_current_command_queue_id_for_thread;
using core::push_current_command_queue_id_for_thread;
using core::set_printoptions;
using core::with_command_queue_id;

class CoreIDs {
public:
    static CoreIDs& instance();

    std::int64_t get_python_operation_id();
    void set_python_operation_id(std::int64_t python_operation_id_);
    std::int64_t fetch_and_increment_python_operation_id();

    std::int64_t get_device_operation_id();
    void set_device_operation_id(std::int64_t device_operation_id);
    std::int64_t fetch_and_increment_device_operation_id();

private:
    CoreIDs() = default;
    ~CoreIDs() = default;
    std::atomic<std::int64_t> python_operation_id;
    std::atomic<std::int64_t> device_operation_id = 1;
};

}  // namespace ttnn
