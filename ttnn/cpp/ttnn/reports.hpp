// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <filesystem>
#include <optional>

#include "tt_metal/impl/buffers/buffer.hpp"

namespace ttnn {

namespace reports {

struct DeviceInfo{
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
    const auto descriptor = tt::get_core_descriptor_config(device.id(), device.num_hw_cqs());

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


void write_l1_buffers(std::ostream &os) {
    std::map<uint32_t, std::map<uint32_t, Buffer *>> buffer_map_per_device;
    for (const auto &[key, buffer] : tt::tt_metal::detail::BUFFER_MAP) {
        auto [device_id, address] = key;
        buffer_map_per_device[device_id][address] = buffer;
    }

    os << "L1 Buffers:" << std::endl;
    for (const auto &[device_id, buffer_map] : buffer_map_per_device) {
        os << "Device: " << device_id << std::endl;
        auto device = buffer_map.begin()->second->device();

        auto core_to_page_map = std::map<CoreCoord, std::map<tt::tt_metal::detail::PageAddress, std::string>>{};
        int buffer_index = 0;
        for (const auto &[address, buffer] : buffer_map) {
            if (buffer->buffer_type() != BufferType::L1) {
                continue;
            }
            uint32_t page_size = buffer->page_size();
            auto num_pages = buffer->num_pages();
            auto num_banks = device->num_banks(buffer->buffer_type());

            if (buffer->buffer_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
                uint32_t bank_index = 0;
                for (int page_index = 0; page_index < num_pages; page_index++) {
                    auto page_address = buffer->page_address(bank_index, page_index);
                    auto core = buffer->logical_core_from_bank_id(bank_index);
                    bank_index = (bank_index + 1) % num_banks;
                    core_to_page_map[core][page_address] =
                        fmt::format("\tBuffer {:3}\tPage {:4}\tPage Size {:9}", buffer_index, page_index, page_size);
                }
            } else {
                for (int page_index = 0; page_index < num_pages; page_index++) {
                    auto dev_page_index = buffer->get_host_to_dev_mapped_page_id(page_index);
                    auto core = buffer->get_core_from_dev_page_id(dev_page_index);
                    auto bank_index = device->bank_ids_from_logical_core(core)[0];
                    auto page_address = buffer->sharded_page_address(bank_index, dev_page_index);
                    core_to_page_map[core][page_address] =
                        fmt::format("\tBuffer {:3}\tPage {:4}\tPage Size {:9}", buffer_index, page_index, page_size);
                }
            }
            buffer_index++;
        }

        for (const auto &[core, page_map] : core_to_page_map) {
            os << "Core: " << core.str() << std::endl;
            for (const auto &[address, page_name] : page_map) {
                os << "  " << fmt::format("Address {:9}", address) << ":" << page_name << std::endl;
            }
            os << std::endl;
        }
    }
    os << std::endl;
}

void print_l1_buffers(const std::optional<std::string> &file_name = std::nullopt) {
    if (file_name.has_value()) {
        tt::log_info("Writing L1 buffers to file: {}", file_name.value());
        std::ofstream os(file_name.value());
        write_l1_buffers(os);
    } else {
        write_l1_buffers(std::cout);
    }
}

struct BufferPage {
    uint32_t address;
    uint32_t device_id;
    uint32_t core_y;
    uint32_t core_x;
    uint32_t page_index;
    uint32_t page_address;
    uint32_t page_size;
    BufferType buffer_type;
};

std::vector<BufferPage> get_buffer_pages() {
    std::vector<BufferPage> pages;
    for (const auto &[key, buffer] : tt::tt_metal::detail::BUFFER_MAP) {
        if (buffer->buffer_type() != BufferType::L1) {
            continue;
        }

        auto [device_id, address] = key;
        auto device = buffer->device();

        uint32_t page_size = buffer->page_size();
        auto num_pages = buffer->num_pages();
        auto num_banks = device->num_banks(buffer->buffer_type());

        if (buffer->buffer_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
            uint32_t bank_index = 0;
            for (int page_index = 0; page_index < num_pages; page_index++) {
                auto page_address = buffer->page_address(bank_index, page_index);
                auto core = buffer->logical_core_from_bank_id(bank_index);
                bank_index = (bank_index + 1) % num_banks;

                BufferPage buffer_page = {};
                buffer_page.address = address;
                buffer_page.device_id = device_id;
                buffer_page.core_y = core.y;
                buffer_page.core_x = core.x;
                buffer_page.page_index = page_index;
                buffer_page.page_address = page_address;
                buffer_page.page_size = page_size;
                buffer_page.buffer_type = buffer->buffer_type();
                pages.push_back(buffer_page);
            }
        } else {
            for (int page_index = 0; page_index < num_pages; page_index++) {
                auto dev_page_index = buffer->get_host_to_dev_mapped_page_id(page_index);
                auto core = buffer->get_core_from_dev_page_id(dev_page_index);
                auto bank_index = device->bank_ids_from_logical_core(core)[0];
                auto page_address = buffer->sharded_page_address(bank_index, dev_page_index);

                BufferPage buffer_page = {};
                buffer_page.address = address;
                buffer_page.device_id = device_id;
                buffer_page.core_y = core.y;
                buffer_page.core_x = core.x;
                buffer_page.page_index = page_index;
                buffer_page.page_address = page_address;
                buffer_page.page_size = page_size;
                buffer_page.buffer_type = buffer->buffer_type();
                pages.push_back(buffer_page);
            }
        }
    }
    return pages;
}

}  // namespace reports

using reports::get_buffer_pages;
using reports::print_l1_buffers;

}  // namespace ttnn
