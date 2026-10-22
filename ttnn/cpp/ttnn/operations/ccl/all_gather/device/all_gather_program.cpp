// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_program.hpp"

#include <array>
#include <set>
#include <unordered_map>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::operations::ccl::all_gather_program {

namespace {

// Slot payloads start on this boundary. It covers every read alignment (BH DRAM 64 B), so a payload offset
// is congruent to its source mod the source's read alignment whenever the cut quantum A is.
constexpr uint32_t kSlotAlign = 64;
constexpr uint32_t kSlotCb = tt::CBIndex::c_0;

}  // namespace

all_gather_plan::Plan make_device_plan(
    const AllGatherParams& args, const Tensor& input, const tt::tt_metal::TensorSpec& output_spec) {
    const uint32_t avail_cores =
        ::ttnn::ccl::num_available_worker_cores(input.device(), args.subdevice_id, args.sub_core_grid);
    return all_gather_plan::make_plan(
        all_gather_plan::make_plan_in(args, input.tensor_spec(), output_spec, avail_cores));
}

namespace {

Workers make_workers(const AllGatherParams& args, tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t n) {
    auto cores = ::ttnn::ccl::choose_worker_cores_across_rows(n, mesh_device, args.subdevice_id, args.sub_core_grid);
    TT_FATAL(cores.size() == n, "all_gather needs {} worker cores but got {}", n, cores.size());
    std::set<CoreRange> ranges(cores.begin(), cores.end());
    CoreRangeSet core_range(ranges);

    // Semaphores live with the cached workload, so they are never reset: every consumer subtracts what it
    // waited for. L1_SMALL keeps them from fragmenting the main L1 pool.
    const bool has_l1_small = mesh_device->allocator()->get_bank_size(tt::tt_metal::BufferType::L1_SMALL) > 0;
    const auto sem_type = has_l1_small ? tt::tt_metal::BufferType::L1_SMALL : tt::tt_metal::BufferType::L1;
    if (!has_l1_small) {
        log_warning(
            tt::LogOp,
            "Allocating semaphores in L1, which may fragment L1 and reduce headroom for subsequent op "
            "allocations. Configure an L1_SMALL region to mitigate this.");
    }
    auto init_sem = ttnn::global_semaphore::create_global_semaphore(mesh_device, core_range, 0, sem_type);
    auto sync_sem = ttnn::global_semaphore::create_global_semaphore(mesh_device, core_range, 0, sem_type);

    // Every device must hold its semaphores before any peer can increment them.
    const std::array<tt::tt_metal::SubDeviceId, 1> subdevices = {
        args.subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0))};
    tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, subdevices);

    return Workers{
        .cores = std::move(cores),
        .core_range = std::move(core_range),
        .init_sem = std::move(init_sem),
        .sync_sem = std::move(sync_sem)};
}

}  // namespace

SlotLayout make_slot_layout(uint32_t num_connections, uint32_t payload_bytes) {
    const auto hdr_bytes = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes());
    const uint32_t payload_off = tt::align(num_connections * hdr_bytes, kSlotAlign);
    return SlotLayout{
        .C = num_connections,
        .hdr_bytes = hdr_bytes,
        .payload_off = payload_off,
        .slot_bytes = tt::align(payload_off + payload_bytes, kSlotAlign)};
}

Kernels create_kernels(
    tt::tt_metal::Program& program,
    const CoreRangeSet& cores,
    const all_gather_plan::Plan& plan,
    const SlotLayout& slot,
    const KernelSpec& spec,
    const Tensor& input,
    const Tensor& output) {
    const auto& geo = plan.geo;
    auto cb_config =
        tt::tt_metal::CircularBufferConfig(kSlots * slot.slot_bytes, {{kSlotCb, tt::DataFormat::RawUInt32}})
            .set_page_size(kSlotCb, slot.slot_bytes);
    tt::tt_metal::CreateCircularBuffer(program, cores, cb_config);

    // Every kernel reads the slot layout; builders also the walk geometry (BuilderCt in kernels/slot_io.hpp).
    const NamedArgs common = {
        {"cb", kSlotCb},
        {"slots", kSlots},
        {"slot_bytes", slot.slot_bytes},
        {"payload_off", slot.payload_off},
        {"hdr_bytes", slot.hdr_bytes},
        {"C", slot.C},
        {"A", geo.A},
    };

    auto builder_named = common;
    builder_named.insert(spec.builder_args.begin(), spec.builder_args.end());
    builder_named.insert({
        {"P", geo.P},
        {"inflight", kInflight},
        {"Ui", geo.Ui},
        {"Uo", geo.Uo},
        {"S", geo.S},
        {"N", plan.N},
        {"E", geo.E},
        {"dst_cross", geo.dst_cross ? 1u : 0u},
        {"src_cross", geo.src_cross ? 1u : 0u},
        {"cross_row", 1u},  // measured on BH where it applies (S = 1 pages): 6-74% faster, never slower
    });
    std::vector<uint32_t> builder_ct;
    tt::tt_metal::TensorAccessorArgs(*input.buffer()).append_to(builder_ct);
    tt::tt_metal::TensorAccessorArgs(*output.buffer()).append_to(builder_ct);

    auto pump_named = common;
    pump_named.insert(spec.pump_args.begin(), spec.pump_args.end());

    const auto builder = tt::tt_metal::CreateKernel(
        program,
        spec.builder_path,
        cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_0,
            .compile_args = builder_ct,
            .named_compile_args = builder_named});
    const auto pump = tt::tt_metal::CreateKernel(
        program,
        spec.pump_path,
        cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::NOC_1,
            .named_compile_args = pump_named});
    return Kernels{.builder = builder, .pump = pump};
}

void append_route_args(
    std::vector<uint32_t>& args, const tt::tt_fabric::FabricNodeId& dst, uint32_t range_v0, uint32_t range_v1) {
    args.push_back(dst.chip_id);
    args.push_back(*dst.mesh_id);
    args.push_back(range_v0);
    args.push_back(range_v1);
}

cached_mesh_workload_t build_workload(
    const AllGatherParams& args,
    const ttnn::MeshCoordinateRangeSet& coords,
    const Tensor& input,
    const Tensor& output,
    all_gather_plan::AgAlgo algo,
    uint32_t cores_per_link,
    const DeviceProgramFn& fn) {
    const auto plan = make_device_plan(args, input, output.tensor_spec());
    TT_FATAL(plan.algo == algo, "all_gather factory does not match the plan's algorithm");
    const auto workers = make_workers(args, input.device(), cores_per_link * plan.links);

    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, SharedVariables> shared_variables;
    for (const auto& coord : coords.coords()) {
        tt::tt_metal::Program program{};
        const Kernels kernels = fn(program, plan, workers, coord);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(program));
        shared_variables.emplace(
            ttnn::MeshCoordinateRange(coord),
            SharedVariables{
                .cores = workers.cores,
                .builder_kernel_id = kernels.builder,
                .init_sem = workers.init_sem,
                .sync_sem = workers.sync_sem});
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

void override_runtime_arguments(cached_mesh_workload_t& workload, const Tensor& input, const Tensor& output) {
    const uint32_t input_addr = input.buffer()->address();
    const uint32_t output_addr = output.buffer()->address();
    for (auto& [range, program] : workload.workload.get_programs()) {
        const auto& shared = workload.shared_variables.at(range);
        auto& builder_args = tt::tt_metal::GetRuntimeArgs(program, shared.builder_kernel_id);
        for (const auto& core : shared.cores) {
            auto& args = builder_args[core.x][core.y];
            args[kBuilderRtInAddr] = input_addr;
            args[kBuilderRtOutAddr] = output_addr;
        }
    }
}

}  // namespace ttnn::operations::ccl::all_gather_program
