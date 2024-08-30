// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "ttnn/core.hpp"

namespace ttnn {

std::int64_t CoreIDs::get_python_operation_id() {
    return python_operation_id.load();
}
void CoreIDs::set_python_operation_id(std::int64_t python_operation_id_) {
    python_operation_id = python_operation_id_;
}
std::int64_t CoreIDs::fetch_and_increment_python_operation_id() {
    return python_operation_id.fetch_add(1);
}

std::int64_t CoreIDs::get_tensor_id() {
    return tensor_id.load();
}
void CoreIDs::set_tensor_id(std::int64_t tensor_id_) {
    tensor_id = tensor_id_;
}
std::int64_t CoreIDs::fetch_and_increment_tensor_id() {
    return tensor_id.fetch_add(1);
}

std::int64_t CoreIDs::get_device_operation_id() {
    return device_operation_id.load();
}
void CoreIDs::set_device_operation_id(std::int64_t device_operation_id_) {
    device_operation_id = device_operation_id_;
}
std::int64_t CoreIDs::fetch_and_increment_device_operation_id() {
    return device_operation_id.fetch_add(1);
}

}  // namespace ttnn
