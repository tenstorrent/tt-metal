// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "ttnn/core.hpp"

namespace ttnn {

std::atomic<std::int64_t> TENSOR_ID = -1;
std::atomic<std::int64_t> PYTHON_OPERATION_ID = -1;
std::atomic<std::int64_t> DEVICE_OPERATION_ID = 0;

std::int64_t get_python_operation_id() {
    return PYTHON_OPERATION_ID.load();
}
void set_python_operation_id(std::int64_t python_operation_id) {
    PYTHON_OPERATION_ID = python_operation_id;
}
void increment_python_operation_id() {
    PYTHON_OPERATION_ID++;
}

std::int64_t get_tensor_id() {
    return TENSOR_ID.load();
}
void set_tensor_id(std::int64_t tensor_id) {
    TENSOR_ID = tensor_id;
}
void increment_tensor_id() {
    TENSOR_ID++;
}

std::int64_t get_device_operation_id() {
    return DEVICE_OPERATION_ID.load();
}
void set_device_operation_id(std::int64_t device_operation_id) {
    DEVICE_OPERATION_ID = device_operation_id;
}
std::int64_t fetch_and_increment_device_operation_id() {
    return DEVICE_OPERATION_ID.fetch_add(1);
}

}  // namespace ttnn
