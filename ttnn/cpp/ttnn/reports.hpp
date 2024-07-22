// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <filesystem>
#include <optional>

#include "tt_metal/impl/buffers/buffer.hpp"

namespace ttnn {

namespace reports {

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

DeviceInfo get_device_info(const Device &device) {
    DeviceInfo info{};
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device.id());
    const auto descriptor = tt::get_core_descriptor_config(device.id(), device.num_hw_cqs(), dispatch_core_type);
    info.num_y_cores = device.logical_grid_size().y;
    info.num_x_cores = device.logical_grid_size().x;
    info.num_y_compute_cores = descriptor.compute_grid_size.y;
    info.num_x_compute_cores = descriptor.compute_grid_size.x;
    info.worker_l1_size = device.allocator_->config.worker_l1_size;
    info.l1_num_banks = device.num_banks(BufferType::L1);
    info.l1_bank_size = device.bank_size(BufferType::L1);
    info.address_at_first_l1_bank = device.allocator_->l1_manager.bank_offset(0);
    info.address_at_first_l1_cb_buffer = L1_UNRESERVED_BASE;
    info.num_banks_per_storage_core = device.allocator_->config.worker_l1_size / info.l1_bank_size;
    info.num_storage_cores = descriptor.relative_storage_cores.size();
    info.num_compute_cores = descriptor.relative_compute_cores.size();
    info.total_l1_memory = (info.num_storage_cores + info.num_compute_cores) * device.allocator_->config.worker_l1_size;
    info.total_l1_for_interleaved_buffers = (info.num_storage_cores + info.num_compute_cores + (info.num_banks_per_storage_core * info.num_storage_cores)) * info.l1_bank_size;
    info.total_l1_for_sharded_buffers = info.num_compute_cores * info.l1_bank_size;
    info.cb_limit = device.allocator_->config.worker_l1_size - L1_UNRESERVED_BASE;
    return info;
}

struct BufferInfo {
    uint32_t device_id;
    uint32_t address;
    uint32_t max_size_per_bank;
    BufferType buffer_type;
};

std::vector<BufferInfo> get_buffers() {
    std::vector<BufferInfo> buffer_infos;
    for (const auto &[key, buffer] : tt::tt_metal::detail::BUFFER_MAP.value()) {
        auto [device_id, address] = key;
        auto device = buffer->device();

        auto num_pages = buffer->num_pages();
        auto page_size = buffer->page_size();
        auto num_banks = device->num_banks(buffer->buffer_type());

        std::map<uint32_t, uint32_t> bank_to_num_pages;
        if (buffer->buffer_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
            uint32_t bank_id = 0;
            for (int page_index = 0; page_index < num_pages; page_index++) {
                if (bank_to_num_pages.find(bank_id) == bank_to_num_pages.end()) {
                    bank_to_num_pages[bank_id] = 0;
                }
                bank_to_num_pages[bank_id]++;
                bank_id = (bank_id + 1) % num_banks;
            }
        } else {
            auto buffer_page_mapping = generate_buffer_page_mapping(*buffer);
            for (int page_index = 0; page_index < num_pages; page_index++) {
                auto dev_page_index = buffer_page_mapping.host_page_to_dev_page_mapping_[page_index];
                auto core = buffer_page_mapping.all_cores_[buffer_page_mapping.dev_page_to_core_mapping_[dev_page_index]];
                auto bank_id = device->bank_ids_from_logical_core(buffer->buffer_type(), core)[0];

                if (bank_to_num_pages.find(bank_id) == bank_to_num_pages.end()) {
                    bank_to_num_pages[bank_id] = 0;
                }
                bank_to_num_pages[bank_id]++;
            }
        }

        auto max_num_pages =
            std::max_element(bank_to_num_pages.begin(), bank_to_num_pages.end(), [](const auto &a, const auto &b) {
                return a.second < b.second;
            });

        BufferInfo buffer_info = {};
        buffer_info.device_id = device_id;
        buffer_info.address = address;
        buffer_info.max_size_per_bank = (*max_num_pages).second * page_size;
        buffer_info.buffer_type = buffer->buffer_type();
        buffer_infos.push_back(buffer_info);
    }
    return buffer_infos;
}

struct BufferPageInfo {
    uint32_t device_id;
    uint32_t address;
    uint32_t core_y;
    uint32_t core_x;
    uint32_t bank_id;
    uint32_t page_index;
    uint32_t page_address;
    uint32_t page_size;
    BufferType buffer_type;
};

std::vector<BufferPageInfo> get_buffer_pages() {
    std::vector<BufferPageInfo> buffer_page_infos;
    for (const auto &[key, buffer] : tt::tt_metal::detail::BUFFER_MAP.value()) {
        if (not buffer->is_l1()) {
            continue;
        }

        auto [device_id, address] = key;
        auto device = buffer->device();

        uint32_t page_size = buffer->page_size();
        auto num_pages = buffer->num_pages();
        auto num_banks = device->num_banks(buffer->buffer_type());

        if (buffer->buffer_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
            uint32_t bank_id = 0;
            for (int page_index = 0; page_index < num_pages; page_index++) {
                auto page_address = buffer->page_address(bank_id, page_index);
                auto core = buffer->logical_core_from_bank_id(bank_id);

                BufferPageInfo buffer_page_info = {};
                buffer_page_info.device_id = device_id;
                buffer_page_info.address = address;
                buffer_page_info.core_y = core.y;
                buffer_page_info.core_x = core.x;
                buffer_page_info.bank_id = bank_id;
                buffer_page_info.page_index = page_index;
                buffer_page_info.page_address = page_address;
                buffer_page_info.page_size = page_size;
                buffer_page_info.buffer_type = buffer->buffer_type();
                buffer_page_infos.push_back(buffer_page_info);

                bank_id = (bank_id + 1) % num_banks;
            }
        } else {
            auto buffer_page_mapping = generate_buffer_page_mapping(*buffer);
            for (int page_index = 0; page_index < num_pages; page_index++) {
                auto dev_page_index = buffer_page_mapping.host_page_to_dev_page_mapping_[page_index];
                auto core = buffer_page_mapping.all_cores_[buffer_page_mapping.dev_page_to_core_mapping_[dev_page_index]];
                auto bank_id = device->bank_ids_from_logical_core(buffer->buffer_type(), core)[0];
                auto page_address = buffer->sharded_page_address(bank_id, dev_page_index);

                BufferPageInfo buffer_page_info = {};
                buffer_page_info.device_id = device_id;
                buffer_page_info.address = address;
                buffer_page_info.core_y = core.y;
                buffer_page_info.core_x = core.x;
                buffer_page_info.bank_id = bank_id;
                buffer_page_info.page_index = page_index;
                buffer_page_info.page_address = page_address;
                buffer_page_info.page_size = page_size;
                buffer_page_info.buffer_type = buffer->buffer_type();
                buffer_page_infos.push_back(buffer_page_info);
            }
        }
    }
    return buffer_page_infos;
}

}  // namespace reports

using reports::get_buffer_pages;
using reports::get_buffers;
using reports::get_device_info;

}  // namespace ttnn
