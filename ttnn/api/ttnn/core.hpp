// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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

namespace ttnn {

using OptionalConstTensors = std::vector<std::optional<const Tensor>>;
using OptionalTensors = std::vector<std::optional<Tensor>>;
using Tensors = std::vector<Tensor>;
using TensorPrintProfile = tt::tt_metal::tensor_impl::TensorPrintProfile;
using SciMode = tt::tt_metal::tensor_impl::SciMode;

}  // namespace ttnn

namespace ttnn {

namespace core {

bool has_storage_type_of(const ttnn::Tensor& tensor, const ttnn::StorageType& storage_type);

std::optional<ttnn::MemoryConfig> get_memory_config(const ttnn::Tensor& tensor);

void set_printoptions(TensorPrintProfile print_profile, SciMode sci_mode = SciMode::Default, int precision = 4);

void segfault_handler(int sig);

void dump_stack_trace_on_segfault();

QueueId get_current_command_queue_id_for_thread();
void push_current_command_queue_id_for_thread(QueueId cq_id);
QueueId pop_current_command_queue_id_for_thread();

ScopeGuard with_command_queue_id(QueueId cq_id);

template <typename T>
void with_command_queue_id(QueueId cq_id, T&& func) {
    auto guard = with_command_queue_id(cq_id);
    std::forward<T>(func)();
}

}  // namespace core

using core::get_current_command_queue_id_for_thread;
using core::get_memory_config;
using core::has_storage_type_of;
using core::pop_current_command_queue_id_for_thread;
using core::push_current_command_queue_id_for_thread;
using core::set_printoptions;
using core::with_command_queue_id;

}  // namespace ttnn
