// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/reports.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_descriptor.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace ttnn::reports {

DeviceInfo get_device_info(tt::tt_metal::distributed::MeshDevice* device) {
    DeviceInfo info{};
    const auto& dispatch_core_config = tt::tt_metal::get_dispatch_core_config();
    const auto descriptor =
        tt::get_core_descriptor_config(device->get_device_ids().at(0), device->num_hw_cqs(), dispatch_core_config);
    const auto& device_allocator = device->allocator();
    info.num_y_cores = device->logical_grid_size().y;
    info.num_x_cores = device->logical_grid_size().x;
    info.num_y_compute_cores = descriptor.compute_grid_size.y;
    info.num_x_compute_cores = descriptor.compute_grid_size.x;
    info.worker_l1_size = device_allocator->get_config().worker_l1_size;
    info.l1_num_banks = device_allocator->get_num_banks(tt::tt_metal::BufferType::L1);
    info.l1_bank_size = device_allocator->get_bank_size(tt::tt_metal::BufferType::L1);
    info.address_at_first_l1_bank = device_allocator->get_bank_offset(tt::tt_metal::BufferType::L1, 0);
    info.address_at_first_l1_cb_buffer = device_allocator->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    info.num_banks_per_storage_core = device_allocator->get_config().worker_l1_size / info.l1_bank_size;
    info.num_storage_cores = descriptor.relative_storage_cores.size();
    info.num_compute_cores = descriptor.relative_compute_cores.size();
    info.total_l1_memory =
        (info.num_storage_cores + info.num_compute_cores) * device_allocator->get_config().worker_l1_size;
    info.total_l1_for_interleaved_buffers =
        (info.num_storage_cores + info.num_compute_cores + (info.num_banks_per_storage_core * info.num_storage_cores)) *
        info.l1_bank_size;
    info.total_l1_for_sharded_buffers = info.num_compute_cores * info.l1_bank_size;
    info.cb_limit = device_allocator->get_config().worker_l1_size -
                    device_allocator->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    return info;
}

std::vector<BufferInfo> get_buffers(const std::vector<tt::tt_metal::distributed::MeshDevice*>& devices) {
    std::vector<BufferInfo> buffer_infos;
    for (auto device : devices) {
        std::unordered_set<tt::tt_metal::Buffer*> allocated_buffers;
        {
            auto lock = device->allocator()->lock();
            allocated_buffers = device->allocator()->get_allocated_buffers();
        }
        for (const auto& buffer : allocated_buffers) {
            auto device_id = device->id();
            auto address = buffer->address();

            auto num_pages = buffer->num_pages();
            auto page_size = buffer->page_size();
            auto num_banks = device->allocator()->get_num_banks(buffer->buffer_type());

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
                const auto& buffer_page_mapping = *buffer->get_buffer_page_mapping();
                for (size_t core_index = 0; core_index < buffer_page_mapping.all_cores.size(); core_index++) {
                    auto core = buffer_page_mapping.all_cores[core_index];
                    auto bank_id = device->allocator()->get_bank_ids_from_logical_core(buffer->buffer_type(), core)[0];
                    uint32_t bank_num_pages = 0;
                    for (const auto& core_page_mapping : buffer_page_mapping.core_page_mappings[core_index]) {
                        bank_num_pages =
                            std::max(bank_num_pages, core_page_mapping.device_start_page + core_page_mapping.num_pages);
                    }
                    bank_to_num_pages[bank_id] = bank_num_pages;
                }
            }

            auto max_num_pages =
                std::max_element(bank_to_num_pages.begin(), bank_to_num_pages.end(), [](const auto& a, const auto& b) {
                    return a.second < b.second;
                });

            BufferInfo buffer_info = {};
            buffer_info.device_id = device_id;
            buffer_info.address = address;
            buffer_info.max_size_per_bank = (*max_num_pages).second * page_size;
            buffer_info.buffer_type = buffer->buffer_type();
            buffer_infos.push_back(buffer_info);
        }
    }
    return buffer_infos;
}

std::vector<BufferPageInfo> get_buffer_pages(const std::vector<tt::tt_metal::distributed::MeshDevice*>& devices) {
    std::vector<BufferPageInfo> buffer_page_infos;
    for (auto device : devices) {
        std::unordered_set<tt::tt_metal::Buffer*> allocated_buffers;
        {
            auto lock = device->allocator()->lock();
            allocated_buffers = device->allocator()->get_allocated_buffers();
        }
        for (const auto& buffer : allocated_buffers) {
            if (not buffer->is_l1()) {
                continue;
            }

            auto device_id = device->id();
            auto address = buffer->address();

            auto page_size = buffer->page_size();
            auto num_pages = buffer->num_pages();
            auto num_banks = device->allocator()->get_num_banks(buffer->buffer_type());
            auto buffer_type = buffer->buffer_type();

            if (is_sharded(buffer->buffer_layout())) {
                const auto& buffer_page_mapping = *buffer->get_buffer_page_mapping();
                for (auto mapped_page : buffer_page_mapping) {
                    auto core = buffer_page_mapping.all_cores[mapped_page.core_id];
                    auto bank_id = device->allocator()->get_bank_ids_from_logical_core(buffer_type, core)[0];
                    auto page_address = buffer->address() + mapped_page.device_page * buffer->aligned_page_size();
                    buffer_page_infos.push_back(BufferPageInfo{
                        .device_id = device_id,
                        .address = address,
                        .core_y = core.y,
                        .core_x = core.x,
                        .bank_id = bank_id,
                        .page_index = mapped_page.host_page,
                        .page_address = page_address,
                        .page_size = page_size,
                        .buffer_type = buffer_type,
                    });
                }
            } else {
                uint32_t bank_id = 0;
                for (uint32_t page_index = 0; page_index < num_pages; page_index++) {
                    auto page_address = buffer->page_address(bank_id, page_index);
                    CoreCoord core = buffer->allocator()->get_logical_core_from_bank_id(bank_id);
                    bank_id = (bank_id + 1) % num_banks;

                    buffer_page_infos.push_back(BufferPageInfo{
                        .device_id = device_id,
                        .address = address,
                        .core_y = core.y,
                        .core_x = core.x,
                        .bank_id = bank_id,
                        .page_index = page_index,
                        .page_address = page_address,
                        .page_size = page_size,
                        .buffer_type = buffer_type,
                    });
                }
            }
        }
    }
    return buffer_page_infos;
}

}  // namespace ttnn::reports
