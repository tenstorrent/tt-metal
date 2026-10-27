// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_program_factory_common.hpp"

#include <set>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/descriptor_runtime_args.hpp"

namespace ttnn::operations::ccl::all_gather_program {

namespace {

constexpr const char* kReaderKernel = "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/all_gather_reader.cpp";
constexpr uint32_t kSlotCb = tt::CBIndex::c_0;

// The slot layout every kernel derives (ag::slot_offsets) and checks against slot_bytes.
tt::tt_metal::KernelDescriptor::NamedCompileTimeArgs slot_ct_args(const all_gather_plan::Plan& plan) {
    return {
        {"unit_bytes", plan.geo.unit_bytes},
        {"packet_cap", plan.geo.packet_cap},
        {"packets_per_slot", plan.choice.packets_per_slot},
        {"slot_bytes", plan.slot_bytes},
        {"slot_cb", kSlotCb},
    };
}

}  // namespace

KernelNocs kernel_nocs(const all_gather_plan::Plan& plan) {
    const size_t algo = all_gather_plan::algo_index(plan.choice.algo);
    const auto writer = all_gather_plan::tuning(plan.arch).writer_noc[algo][plan.ring ? 1 : 0];
    return {writer == tt::tt_metal::NOC::NOC_0 ? tt::tt_metal::NOC::NOC_1 : tt::tt_metal::NOC::NOC_0, writer};
}

WorkloadStart start_workload(
    const AllGatherParams& params,
    const Tensor& input,
    const Tensor& output,
    tt::tt_metal::WorkloadDescriptor& workload) {
    TT_FATAL(input.buffer() != nullptr && output.buffer() != nullptr, "all_gather needs allocated tensors");
    WorkloadStart start{.plan = all_gather_plan::expand(params, input.tensor_spec(), output.tensor_spec())};
    auto& mesh_device = *input.device();
    // Fabric has no in-kernel cross-device barrier, so the op syncs through global semaphores. One per purpose: on a
    // shared one, a fast device's next-launch init credit could stand in for a slow device's data or done credit.
    // They live on the worker cores only, in L1_SMALL when there is one: never on a mux core, whose L1 map starts at
    // the allocator base, and out of the large L1 pool.
    const auto sem_buffer_type = ::ttnn::ccl::prefer_l1_small_buffer_type(mesh_device);
    ::ttnn::ccl::warn_if_semaphores_fall_back_to_l1(sem_buffer_type);
    const CoreRangeSet cores = worker_cores(start.plan);
    for (int i = 0; i < 2; ++i) {
        workload.semaphores.push_back(
            ttnn::global_semaphore::create_global_semaphore(&mesh_device, cores, 0, sem_buffer_type));
    }
    // Every device must see the zeroed semaphores before any kernel can increment them.
    const ttsl::SmallVector<tt::tt_metal::SubDeviceId> subdevices = {
        params.subdevice_id.value_or(mesh_device.get_sub_device_ids().at(0))};
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, subdevices);
    start.init_sem = static_cast<uint32_t>(workload.semaphores[0].address());
    start.data_sem = static_cast<uint32_t>(workload.semaphores[1].address());
    return start;
}

CoreRangeSet worker_cores(const all_gather_plan::Plan& plan) {
    std::set<CoreRange> cores;
    for (const auto& group : plan.groups) {
        for (const auto& core : group.workers) {
            cores.emplace(core);
        }
    }
    for (const auto& core : plan.mcast_cores) {
        cores.emplace(core);
    }
    return CoreRangeSet(cores);
}

tt::tt_metal::CBDescriptor slot_cb(const all_gather_plan::Plan& plan, const CoreRangeSet& cores) {
    // The slot is untyped bytes (records, headers, payload); the format only sizes the page.
    return tt::tt_metal::CBDescriptor{
        .total_size = all_gather_plan::kCbSlots * plan.slot_bytes,
        .core_ranges = cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(kSlotCb),
            .data_format = tt::DataFormat::RawUInt8,
            .page_size = plan.slot_bytes,
        }}},
    };
}

tt::tt_metal::KernelDescriptor reader_kernel(
    const all_gather_plan::Plan& plan, const Tensor& input, const Tensor& output, const CoreRangeSet& cores) {
    const auto& geo = plan.geo;
    tt::tt_metal::KernelDescriptor reader;
    reader.kernel_source = kReaderKernel;
    reader.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    reader.named_compile_time_args = slot_ct_args(plan);
    reader.named_compile_time_args.insert(
        reader.named_compile_time_args.end(),
        {
            {"max_packet_bytes", geo.max_packet_bytes},
            {"in_page_units", geo.in_page_units},
            {"out_page_units", geo.out_page_units},
            {"stripe_units", geo.stripe_units},
            {"num_devices", plan.N},
            {"element_units", geo.element_units},
            {"dst_cross", geo.dst_cross ? 1u : 0u},
            {"src_cross", geo.src_cross ? 1u : 0u},
        });
    tt::tt_metal::TensorAccessorArgs(input.buffer()).append_to(reader.compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(reader.compile_time_args);
    reader.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = kernel_nocs(plan).reader};
    return reader;
}

tt::tt_metal::KernelDescriptor writer_kernel(
    const all_gather_plan::Plan& plan,
    const char* source,
    const CoreRangeSet& cores,
    const tt::tt_metal::KernelDescriptor::NamedCompileTimeArgs& extra) {
    tt::tt_metal::KernelDescriptor writer;
    writer.kernel_source = source;
    writer.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.named_compile_time_args = slot_ct_args(plan);
    writer.named_compile_time_args.insert(writer.named_compile_time_args.end(), extra.begin(), extra.end());
    writer.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = kernel_nocs(plan).writer};
    return writer;
}

void add_reader_rt_args(
    tt::tt_metal::KernelDescriptor& reader,
    const CoreCoord& core,
    const Tensor& input,
    const Tensor& output,
    const all_gather_plan::WorkerSched& ws,
    uint32_t data_sem) {
    const std::vector<uint32_t> args = {
        0u,  // input (Buffer* binding)
        0u,  // output (Buffer* binding)
        ws.e0,
        ws.ne,
        ws.iters,
        ws.stripe0,
        ws.stripe_step,
        ws.last_skip,
        ws.last_take,
        ws.iter_units,
        ws.end_wait,
        ws.local_copy ? 1u : 0u,
        data_sem,  // smuggled-rta-ok: persistent GlobalSemaphore (parked on the WorkloadDescriptor)
    };
    ::ttnn::ccl::emplace_runtime_args_with_leading_buffers(reader, core, {input.buffer(), output.buffer()}, args);
}

}  // namespace ttnn::operations::ccl::all_gather_program
