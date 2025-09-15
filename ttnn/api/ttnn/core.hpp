// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <csignal>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"  // TTNN_TENSOR_PRINT_PROFILE
#include "ttnn/tensor/types.hpp"
#include "ttnn/config.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

using OptionalConstTensors = std::vector<std::optional<const Tensor>>;
using OptionalTensors = std::vector<std::optional<Tensor>>;
using Tensors = std::vector<Tensor>;

}  // namespace ttnn

namespace ttnn {

namespace core {

bool has_storage_type_of(const ttnn::Tensor& tensor, const ttnn::StorageType& storage_type);

std::optional<ttnn::MemoryConfig> get_memory_config(const ttnn::Tensor& tensor);

void set_printoptions(const std::string& profile);

void segfault_handler(int sig);

void dump_stack_trace_on_segfault();

}  // namespace core

using core::get_memory_config;
using core::has_storage_type_of;
using core::set_printoptions;

class CoreIDs {
public:
    static CoreIDs& instance();

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
    std::atomic<std::int64_t> device_operation_id = 1;
};

}  // namespace ttnn
