// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

#include <tt-metalium/buffer_types.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace ttnn::reports {

struct DeviceInfo {
    size_t num_y_cores;
    size_t num_x_cores;
    size_t num_y_compute_cores;
    size_t num_x_compute_cores;
    size_t worker_l1_size;
    size_t l1_num_banks;
    size_t l1_bank_size;
    uint64_t address_at_first_l1_bank;
    uint64_t address_at_first_l1_cb_buffer;
    size_t num_banks_per_storage_core;
    size_t num_compute_cores;
    size_t num_storage_cores;
    size_t total_l1_memory;
    size_t total_l1_for_tensors;
    size_t total_l1_for_interleaved_buffers;
    size_t total_l1_for_sharded_buffers;
    size_t cb_limit;
};

DeviceInfo get_device_info(tt::tt_metal::distributed::MeshDevice* device);

struct BufferInfo {
    uint32_t device_id;
    uint32_t address;
    uint32_t max_size_per_bank;
    tt::tt_metal::BufferType buffer_type;
};

std::vector<BufferInfo> get_buffers(const std::vector<tt::tt_metal::distributed::MeshDevice*>& devices);

struct BufferPageInfo {
    uint32_t device_id;
    uint32_t address;
    uint32_t core_y;
    uint32_t core_x;
    uint32_t bank_id;
    uint32_t page_index;
    uint32_t page_address;
    uint32_t page_size;
    tt::tt_metal::BufferType buffer_type;
};

std::vector<BufferPageInfo> get_buffer_pages(const std::vector<tt::tt_metal::distributed::MeshDevice*>& devices);
}  // namespace ttnn::reports
