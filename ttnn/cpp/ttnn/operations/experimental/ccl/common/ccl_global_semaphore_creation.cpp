// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/ccl/common/ccl_global_semaphore_creation.hpp"
#include "ttnn/cpp/ttnn/global_semaphore.hpp"

namespace ttnn::ccl::worker_detail {

std::vector<std::shared_ptr<const tt::tt_metal::GlobalSemaphore>> create_global_semaphores(
    const std::vector<Device*>& devices,
    const CoreRangeSet& worker_cores,
    std::optional<SubDeviceId> worker_subdevice_id_opt) {
    std::vector<std::shared_ptr<const tt::tt_metal::GlobalSemaphore>> semaphores;
    for (Device* d : devices) {
        CoreCoord grid_size = devices[0]->compute_with_storage_grid_size();
        auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
        auto worker_subdevice_id = worker_subdevice_id_opt.has_value()
                                       ? std::vector<SubDeviceId>{worker_subdevice_id_opt.value()}
                                       : std::vector<SubDeviceId>{};
        auto sem = std::make_shared<GlobalSemaphore>(
            global_semaphore::create_global_semaphore(d, core_grid, 0, BufferType::L1, worker_subdevice_id));
        semaphores.push_back(sem);
    }

    auto first_addr = semaphores.front()->address();
    bool all_same = std::all_of(
        semaphores.begin(), semaphores.end(), [first_addr](const auto& sem) { return sem->address() == first_addr; });

    if (!all_same) {
        DeviceAddr lowest_addr = semaphores.front()->address();
        for (auto i = 1; i < semaphores.size(); i++) {
            if (semaphores[i]->address() < lowest_addr) {
                lowest_addr = semaphores[i]->address();
            }
        };
        for (auto i = 0; i < semaphores.size(); i++) {
            size_t attempts = 1000;
            size_t attempt = 0;
            std::vector<std::shared_ptr<const tt::tt_metal::GlobalSemaphore>> garbage;
            while (semaphores[i]->address() != lowest_addr) {
                auto worker_subdevice_id = worker_subdevice_id_opt.has_value()
                                               ? std::vector<SubDeviceId>{worker_subdevice_id_opt.value()}
                                               : std::vector<SubDeviceId>{};
                auto sem = std::make_shared<GlobalSemaphore>(
                    global_semaphore::create_global_semaphore(devices[i], worker_cores, 0, BufferType::L1, worker_subdevice_id));
                if (sem->address() == lowest_addr) {
                    semaphores[i] = sem;
                } else {
                    garbage.push_back(std::move(sem));
                    attempt++;
                }

                if (attempt > attempts) {
                    TT_THROW("Failed to create global semaphores with the same address");
                }
            }
        }
    }
    return semaphores;
}

}  // namespace ttnn::ccl::worker_detail
