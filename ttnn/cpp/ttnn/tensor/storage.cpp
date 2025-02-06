// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/storage.hpp"

namespace tt::tt_metal {

std::vector<std::shared_ptr<Buffer>> MultiDeviceStorage::get_buffers() const {
    std::lock_guard<std::mutex> lock(buffer_mtx);
    std::vector<std::shared_ptr<Buffer>> buf_vec;
    buf_vec.reserve(buffers.size());
    for (const auto& pair : buffers) {
        buf_vec.push_back(pair.second);
    }
    return buf_vec;
}

}  // namespace tt::tt_metal
