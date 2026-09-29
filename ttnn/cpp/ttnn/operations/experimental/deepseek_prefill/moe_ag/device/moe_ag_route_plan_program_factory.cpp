// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_ag_route_plan_device_operation.hpp"

#include "moe_ag_common.hpp"

using namespace tt::tt_metal;
using namespace ttnn::operations::experimental::deepseek_prefill::moe_ag;

namespace ttnn::prim {

// R = 64 cores (8 x 8 at the grid origin, core r = y * 8 + x owns tokens [r npr, r npr + npr) and, if r < EPC, local
// expert r); BRISC only. Common args: the six tensors, T, npr, 0, then the R cores' NoC xy (x << 16 | y).
// CBs (scratch): 0 idx pages, 1 local-slot map, 2 column, 3 message, 4 y_slot block, 5 expert list, 6 core 0 rows,
// 7 core 0 counts + a 1 word. Semaphores 0-5: three joins and three go signals.
ProgramDescriptor MoeAgRoutePlanDeviceOperation::ProgramFactory::create_descriptor(
    const MoeAgRoutePlanParams& args, const MoeAgRoutePlanInputs& tensor_args, std::vector<Tensor>& outputs) {
    const auto& idx = tensor_args.topk_indices;
    const auto& lmap = tensor_args.local_slot_map;
    constexpr uint32_t R = MOE_AG_ROUTE_PLAN_CORES, IDX_STRIDE = 64;
    const uint32_t EPC = args.experts_per_chip, NG = lmap.logical_shape()[-1];
    const uint32_t K = idx.logical_shape()[-1], T = rm_rows(idx);
    const uint32_t npr = round_up((T + R - 1) / R, token_align(K));  // 64 B aligned y_slot blocks

    const CoreRangeSet cores(CoreRange({0, 0}, {7, 7}));
    auto* device = idx.device();
    const auto p0 = device->worker_core_from_logical_core(CoreCoord{0, 0});
    const auto p1 = device->worker_core_from_logical_core(CoreCoord{7, 7});

    ProgramDescriptor desc;
    desc.cbs.push_back(scratch_cb(0, std::max(64u, npr * IDX_STRIDE), cores));
    desc.cbs.push_back(scratch_cb(1, NG * 4, cores));
    desc.cbs.push_back(scratch_cb(2, round_up(R * 4, 64), cores));
    desc.cbs.push_back(scratch_cb(3, round_up(3 * EPC * 4, 64), cores));
    desc.cbs.push_back(scratch_cb(4, std::max(64u, round_up(npr * K * 4, 64)), cores));
    desc.cbs.push_back(scratch_cb(5, round_up(T, 32) * 4, cores));
    desc.cbs.push_back(scratch_cb(6, 2 * NG * 4, cores));
    desc.cbs.push_back(scratch_cb(7, round_up(EPC * 4 + 16, 64), cores));

    auto kernel =
        kernel_desc("route_plan.cpp", cores, {R, EPC, NG, K, IDX_STRIDE, p0.x, p0.y, p1.x, p1.y}, dm_config(0, 0));
    KernelDescriptor::RTArgList common;
    common.push_back(idx.buffer());
    common.push_back(lmap.buffer());
    for (auto& t : outputs) {
        common.push_back(t.buffer());
    }
    common.push_back(T);
    common.push_back(npr);
    common.push_back(0u);
    for (uint32_t y = 0; y < 8; ++y) {
        for (uint32_t x = 0; x < 8; ++x) {
            const auto p = device->worker_core_from_logical_core(CoreCoord{x, y});
            common.push_back(static_cast<uint32_t>((p.x << 16) | p.y));
        }
    }
    kernel.emplace_common_runtime_args(common);
    desc.kernels.push_back(std::move(kernel));
    for (uint32_t i = 0; i < 6; ++i) {
        desc.semaphores.push_back(SemaphoreDescriptor{.id = i, .core_ranges = cores, .initial_value = 0});
    }
    return desc;
}

}  // namespace ttnn::prim
