// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/d2d_stream_service.hpp"

#include <map>
#include <optional>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/internal/service/service_core_manager.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace tt::tt_metal {

namespace CMAKE_UNIQUE_NAMESPACE {

// Build a single-shard host tensor with zero-initialised data of size `spec`.
// Used purely to feed the mapper at construction time so we can extract a
// TensorTopology + per-shard spec before any user data exists. The bytes are
// never read. Mirrors the helper in socket_services.cpp (H2D).
// TODO: replace with a direct "topology from MeshMapperConfig + global shape"
Tensor make_zero_host_tensor(const TensorSpec& spec) {
    const size_t bytes = spec.compute_packed_buffer_size_bytes();
    switch (spec.data_type()) {
        case DataType::BFLOAT16:
            return Tensor::from_vector<bfloat16>(std::vector<bfloat16>(bytes / sizeof(bfloat16)), spec);
        case DataType::FLOAT32: return Tensor::from_vector<float>(std::vector<float>(bytes / sizeof(float)), spec);
        case DataType::INT32: return Tensor::from_vector<int32_t>(std::vector<int32_t>(bytes / sizeof(int32_t)), spec);
        case DataType::UINT8: return Tensor::from_vector<uint8_t>(std::vector<uint8_t>(bytes / sizeof(uint8_t)), spec);
        case DataType::UINT16:
            return Tensor::from_vector<uint16_t>(std::vector<uint16_t>(bytes / sizeof(uint16_t)), spec);
        case DataType::BFLOAT4_B:
        case DataType::BFLOAT8_B:
        case DataType::UINT32:
            return Tensor::from_vector<uint32_t>(std::vector<uint32_t>(bytes / sizeof(uint32_t)), spec);
        case DataType::INVALID: TT_THROW("D2DStreamService: invalid global_spec data type");
    }
    TT_THROW("Unreachable");
}

// Claim one service core per participating coord on `mesh`, recording the
// choice per coord. Mirrors the B3.5 block in socket_services.cpp.
std::map<distributed::MeshCoordinate, CoreCoord> claim_service_cores(
    const std::shared_ptr<distributed::MeshDevice>& mesh,
    const std::vector<distributed::MeshCoordinate>& coords,
    const char* side) {
    auto& svc = tt::tt_metal::internal::ServiceCoreManager::get();
    std::map<distributed::MeshCoordinate, CoreCoord> service_cores;
    for (const auto& coord : coords) {
        auto* d = mesh->get_device(coord);
        auto claimable = svc.get_claimable_cores(d);
        TT_FATAL(
            !claimable.empty(), "D2DStreamService: no claimable {} service core on device at coord {}", side, coord);
        const CoreCoord chosen = claimable.front();
        svc.claim(d, {chosen});
        service_cores.emplace(coord, chosen);
    }
    return service_cores;
}

// Chunking plan shared by both sides. Identical to the H2D derivation in
// socket_services.cpp: pack as many whole tensor pages into a socket page as
// the scratch CB budget allows, reduced to a divisor of the page count so a
// transfer is an integer number of socket pages.
struct ChunkPlan {
    uint32_t socket_page_size;  // bytes per socket page (== pages_per_chunk * tensor_page_size)
    uint32_t num_socket_pages;  // socket pages per full transfer (== tensor_num_pages / pages_per_chunk)
    uint32_t pages_per_chunk;   // tensor pages drained per socket page
};

ChunkPlan derive_chunk_plan(uint32_t tensor_page_size, uint32_t tensor_num_pages, uint32_t scratch_cb_size_bytes) {
    TT_FATAL(tensor_page_size > 0, "D2DStreamService: tensor page size must be > 0");
    TT_FATAL(tensor_num_pages > 0, "D2DStreamService: backing tensor must have at least one page");
    TT_FATAL(
        scratch_cb_size_bytes >= tensor_page_size,
        "D2DStreamService: scratch_cb_size_bytes ({} B) must be >= tensor page size ({} B)",
        scratch_cb_size_bytes,
        tensor_page_size);

    const uint32_t max_pages_per_chunk_by_cb = scratch_cb_size_bytes / tensor_page_size;
    uint32_t pages_per_chunk = std::min(tensor_num_pages, max_pages_per_chunk_by_cb);
    while (pages_per_chunk > 1 && (tensor_num_pages % pages_per_chunk) != 0) {
        --pages_per_chunk;
    }
    TT_FATAL(
        pages_per_chunk > 0,
        "D2DStreamService: derived pages_per_chunk == 0 (tensor_page_size={}, tensor_num_pages={}, "
        "scratch_cb_size_bytes={}); the socket FIFO must hold at least one tensor page",
        tensor_page_size,
        tensor_num_pages,
        scratch_cb_size_bytes);
    return ChunkPlan{
        .socket_page_size = pages_per_chunk * tensor_page_size,
        .num_socket_pages = tensor_num_pages / pages_per_chunk,
        .pages_per_chunk = pages_per_chunk,
    };
}

// Number of cores in an inclusive CoreRange (worker-grid mcast destination count
// + sync-arithmetic target).
uint32_t core_range_size(const CoreRange& range) {
    return (range.end_coord.x - range.start_coord.x + 1) * (range.end_coord.y - range.start_coord.y + 1);
}

// Allocate one zero-initialised uint32 L1 word on a service core, recording the
// address per coord. Used for the per-coord counters and termination words that
// can't be GlobalSemaphores (service cores differ per device, and
// GlobalSemaphore's reset path requires a MeshBuffer-backed buffer).
std::map<distributed::MeshCoordinate, DeviceAddr> allocate_service_core_words(
    const std::shared_ptr<distributed::MeshDevice>& mesh,
    const std::map<distributed::MeshCoordinate, CoreCoord>& service_cores) {
    auto& svc = tt::tt_metal::internal::ServiceCoreManager::get();
    std::vector<uint32_t> zero_word{0};
    std::map<distributed::MeshCoordinate, DeviceAddr> addrs;
    for (const auto& [coord, core] : service_cores) {
        auto* d = mesh->get_device(coord);
        const DeviceAddr addr = svc.allocate_l1(d, core, sizeof(uint32_t));
        tt::tt_metal::detail::WriteToDeviceL1(d, core, static_cast<uint32_t>(addr), zero_word);
        addrs.emplace(coord, addr);
    }
    return addrs;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE

// ===========================================================================
// Sender / Receiver Impl state
// ===========================================================================
//
// V0 staging: at step 2 these hold only the state produced by the validated +
// allocate + claim portion of create_pair (backing tensor, per-shard spec,
// worker cores, service-core map, mesh device handle). Sockets, worker-sync
// resources, termination semaphores, and the persistent MeshWorkload land in
// later steps.

struct D2DStreamServiceSender::Impl {
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    TensorSpec per_shard_spec;
    Tensor backing_tensor;
    CoreRange worker_cores;
    std::map<distributed::MeshCoordinate, CoreCoord> service_cores;
    // Sender endpoint of the MeshSocket pair (no data buffer — sender only owns
    // the config buffer). std::optional because MeshSocket has no default ctor.
    std::optional<distributed::MeshSocket> socket;

    // --- step 4: chunk plan + worker-sync resources ------------------------
    uint32_t socket_page_size = 0;
    uint32_t num_socket_pages = 0;
    uint32_t pages_per_chunk = 0;
    uint32_t num_workers = 0;
    // Per-coord service-core L1 words (service cores differ per device).
    std::map<distributed::MeshCoordinate, DeviceAddr> data_ready_counter_addrs;
    std::map<distributed::MeshCoordinate, DeviceAddr> termination_addrs;
    // Mesh-wide GlobalSemaphore on sender_worker_cores; the service kernel
    // multicast-incs it once per drained iter.
    std::optional<GlobalSemaphore> consumed_sem;

    // --- step 6: persistent sender workload --------------------------------
    std::unique_ptr<distributed::MeshWorkload> workload;
    bool launched = false;
};

struct D2DStreamServiceReceiver::Impl {
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    TensorSpec per_shard_spec;
    Tensor backing_tensor;
    CoreRange worker_cores;
    std::map<distributed::MeshCoordinate, CoreCoord> service_cores;
    // Receiver endpoint of the MeshSocket pair (owns the data FIFO + config
    // buffer). std::optional because MeshSocket has no default ctor.
    std::optional<distributed::MeshSocket> socket;

    // --- step 4: chunk plan + worker-sync resources ------------------------
    uint32_t socket_page_size = 0;
    uint32_t num_socket_pages = 0;
    uint32_t pages_per_chunk = 0;
    uint32_t num_workers = 0;
    // Per-coord service-core L1 words (service cores differ per device).
    std::map<distributed::MeshCoordinate, DeviceAddr> consumed_counter_addrs;
    std::map<distributed::MeshCoordinate, DeviceAddr> termination_addrs;
    // Mesh-wide GlobalSemaphore on receiver_worker_cores; the service kernel
    // multicast-incs it after the transfer has landed.
    std::optional<GlobalSemaphore> data_ready_sem;

    // --- step 5: persistent receiver workload ------------------------------
    std::unique_ptr<distributed::MeshWorkload> workload;
    bool launched = false;
};

// ===========================================================================
// Sender handle
// ===========================================================================

D2DStreamServiceSender::D2DStreamServiceSender(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

D2DStreamServiceSender::~D2DStreamServiceSender() {
    // Teardown (step 9 shape, minus the multi-side coordination): flip this
    // side's termination word so the persistent kernel exits at its next
    // data_ready poll, drain the kernel, free service-core L1, drop the socket,
    // and release the cores. The two handles tear down independently.
    try {
        if (impl_ == nullptr) {
            return;
        }
        auto& svc = tt::tt_metal::internal::ServiceCoreManager::get();
        auto* mesh = impl_->mesh_device.get();

        // 1. Signal termination + drain the persistent sender kernel.
        if (impl_->launched && mesh != nullptr) {
            std::vector<uint32_t> one_word{1};
            for (const auto& [coord, addr] : impl_->termination_addrs) {
                tt::tt_metal::detail::WriteToDeviceL1(
                    mesh->get_device(coord), impl_->service_cores.at(coord), static_cast<uint32_t>(addr), one_word);
            }
            distributed::Finish(mesh->mesh_command_queue());
            for (const auto& [coord, core] : impl_->service_cores) {
                svc.wait_done(mesh->get_device(coord), core);
            }
        }
        impl_->workload.reset();

        // 2. Free the service-core L1 words (termination + data_ready_counter).
        if (mesh != nullptr) {
            for (const auto& [coord, addr] : impl_->termination_addrs) {
                svc.deallocate_l1(mesh->get_device(coord), impl_->service_cores.at(coord), addr);
            }
            for (const auto& [coord, addr] : impl_->data_ready_counter_addrs) {
                svc.deallocate_l1(mesh->get_device(coord), impl_->service_cores.at(coord), addr);
            }
        }

        // 3. Drop the socket (frees its config buffer), then release the cores.
        impl_->socket.reset();
        if (mesh != nullptr) {
            for (const auto& [coord, core] : impl_->service_cores) {
                svc.release(mesh->get_device(coord), {core});
            }
        }
    } catch (const std::exception& e) {
        log_warning(tt::LogOp, "D2DStreamServiceSender: shutdown failed: {}", e.what());
    } catch (...) {
        log_warning(tt::LogOp, "D2DStreamServiceSender: shutdown failed with unknown exception");
    }
}

const Tensor& D2DStreamServiceSender::get_backing_tensor() const { return impl_->backing_tensor; }

const TensorSpec& D2DStreamServiceSender::get_per_shard_spec() const { return impl_->per_shard_spec; }

CoreRange D2DStreamServiceSender::get_worker_cores() const { return impl_->worker_cores; }

CoreCoord D2DStreamServiceSender::get_service_core(const distributed::MeshCoordinate& coord) const {
    auto it = impl_->service_cores.find(coord);
    TT_FATAL(
        it != impl_->service_cores.end(),
        "D2DStreamServiceSender::get_service_core: no service core claimed at coord {}",
        coord);
    return it->second;
}

DeviceAddr D2DStreamServiceSender::get_data_ready_counter_addr(const distributed::MeshCoordinate& coord) const {
    auto it = impl_->data_ready_counter_addrs.find(coord);
    TT_FATAL(
        it != impl_->data_ready_counter_addrs.end(),
        "D2DStreamServiceSender::get_data_ready_counter_addr: no counter at coord {}",
        coord);
    return it->second;
}

DeviceAddr D2DStreamServiceSender::get_consumed_sem_addr() const {
    TT_FATAL(
        impl_->consumed_sem.has_value(),
        "D2DStreamServiceSender::get_consumed_sem_addr: consumed_sem not allocated (sender_worker_cores empty?)");
    return impl_->consumed_sem->address();
}

// ===========================================================================
// Receiver handle
// ===========================================================================

D2DStreamServiceReceiver::D2DStreamServiceReceiver(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

D2DStreamServiceReceiver::~D2DStreamServiceReceiver() {
    // Mirror of the sender dtor. The receiver kernel exits at its next
    // socket_wait_for_pages_with_termination poll (its idle state between
    // transfers), so termination is clean once outstanding transfers have
    // drained.
    try {
        if (impl_ == nullptr) {
            return;
        }
        auto& svc = tt::tt_metal::internal::ServiceCoreManager::get();
        auto* mesh = impl_->mesh_device.get();

        // 1. Signal termination + drain the persistent receiver kernel.
        if (impl_->launched && mesh != nullptr) {
            std::vector<uint32_t> one_word{1};
            for (const auto& [coord, addr] : impl_->termination_addrs) {
                tt::tt_metal::detail::WriteToDeviceL1(
                    mesh->get_device(coord), impl_->service_cores.at(coord), static_cast<uint32_t>(addr), one_word);
            }
            distributed::Finish(mesh->mesh_command_queue());
            for (const auto& [coord, core] : impl_->service_cores) {
                svc.wait_done(mesh->get_device(coord), core);
            }
        }
        impl_->workload.reset();

        // 2. Free the service-core L1 words (termination + consumed_counter).
        if (mesh != nullptr) {
            for (const auto& [coord, addr] : impl_->termination_addrs) {
                svc.deallocate_l1(mesh->get_device(coord), impl_->service_cores.at(coord), addr);
            }
            for (const auto& [coord, addr] : impl_->consumed_counter_addrs) {
                svc.deallocate_l1(mesh->get_device(coord), impl_->service_cores.at(coord), addr);
            }
        }

        // 3. Drop the socket (frees its config + data FIFO buffers), then
        //    release the cores.
        impl_->socket.reset();
        if (mesh != nullptr) {
            for (const auto& [coord, core] : impl_->service_cores) {
                svc.release(mesh->get_device(coord), {core});
            }
        }
    } catch (const std::exception& e) {
        log_warning(tt::LogOp, "D2DStreamServiceReceiver: shutdown failed: {}", e.what());
    } catch (...) {
        log_warning(tt::LogOp, "D2DStreamServiceReceiver: shutdown failed with unknown exception");
    }
}

const Tensor& D2DStreamServiceReceiver::get_backing_tensor() const { return impl_->backing_tensor; }

const TensorSpec& D2DStreamServiceReceiver::get_per_shard_spec() const { return impl_->per_shard_spec; }

CoreRange D2DStreamServiceReceiver::get_worker_cores() const { return impl_->worker_cores; }

CoreCoord D2DStreamServiceReceiver::get_service_core(const distributed::MeshCoordinate& coord) const {
    auto it = impl_->service_cores.find(coord);
    TT_FATAL(
        it != impl_->service_cores.end(),
        "D2DStreamServiceReceiver::get_service_core: no service core claimed at coord {}",
        coord);
    return it->second;
}

DeviceAddr D2DStreamServiceReceiver::get_data_ready_sem_addr() const {
    TT_FATAL(
        impl_->data_ready_sem.has_value(),
        "D2DStreamServiceReceiver::get_data_ready_sem_addr: data_ready_sem not allocated (receiver_worker_cores "
        "empty?)");
    return impl_->data_ready_sem->address();
}

DeviceAddr D2DStreamServiceReceiver::get_consumed_counter_addr(const distributed::MeshCoordinate& coord) const {
    auto it = impl_->consumed_counter_addrs.find(coord);
    TT_FATAL(
        it != impl_->consumed_counter_addrs.end(),
        "D2DStreamServiceReceiver::get_consumed_counter_addr: no counter at coord {}",
        coord);
    return it->second;
}

// ===========================================================================
// Persistent program builders (steps 5 + 6)
// ===========================================================================

namespace CMAKE_UNIQUE_NAMESPACE {

// Both kernels run on the single service core, RISCV_0. CB indices are private
// to each program so the sender (scratch + headers) and receiver (headers only)
// don't need to agree.
constexpr tt::CBIndex kScratchCbIndex = tt::CBIndex::c_0;
constexpr tt::CBIndex kPacketHeaderCbIndex = tt::CBIndex::c_1;

// Per-coord worker-sync CT-arg block. `sem_addr` is the mesh-wide worker-grid
// GlobalSemaphore (consumed_sem on the sender, data_ready_sem on the receiver);
// `counter_addr` is the per-coord service-core L1 word (data_ready_counter on
// the sender, consumed_counter on the receiver). All zero when disabled.
struct WorkerSyncArgs {
    bool enabled = false;
    uint32_t sem_addr = 0;
    uint32_t counter_addr = 0;
    uint32_t mcast_noc_x_start = 0;
    uint32_t mcast_noc_y_start = 0;
    uint32_t mcast_noc_x_end = 0;
    uint32_t mcast_noc_y_end = 0;
    uint32_t num_workers = 0;
};

// Always populates the per-coord addresses, mcast bbox, and num_workers (the
// sender's data_ready_counter trigger needs counter_addr + num_workers even
// when the consumed_sem multicast is off). `enabled` only toggles the gated
// blocks: the consumed_sem multicast on the sender, the whole data_ready_sem /
// consumed_counter handshake on the receiver.
WorkerSyncArgs make_worker_sync_args(
    IDevice* device,
    const CoreRange& worker_cores,
    uint32_t num_workers,
    uint32_t sem_addr,
    uint32_t counter_addr,
    bool enabled) {
    WorkerSyncArgs ws;
    const auto start_phys = device->worker_core_from_logical_core(worker_cores.start_coord);
    const auto end_phys = device->worker_core_from_logical_core(worker_cores.end_coord);
    ws.enabled = enabled;
    ws.sem_addr = sem_addr;
    ws.counter_addr = counter_addr;
    ws.mcast_noc_x_start = static_cast<uint32_t>(start_phys.x);
    ws.mcast_noc_y_start = static_cast<uint32_t>(start_phys.y);
    ws.mcast_noc_x_end = static_cast<uint32_t>(end_phys.x);
    ws.mcast_noc_y_end = static_cast<uint32_t>(end_phys.y);
    ws.num_workers = num_workers;
    return ws;
}

// CT-arg layout must stay in sync with persistent_d2d_receiver.cpp.
Program build_receiver_program(
    const Buffer& output_buffer,
    const CoreCoord& service_core,
    uint32_t socket_config_addr,
    uint32_t termination_addr,
    const ChunkPlan& plan,
    uint32_t tensor_page_size,
    const WorkerSyncArgs& ws,
    const tt::tt_fabric::FabricNodeId& receiver_node,
    const tt::tt_fabric::FabricNodeId& sender_node,
    uint32_t link_index) {
    auto program = CreateProgram();

    // One packet header is enough for the receiver (control-flow notify only);
    // allocate two to mirror the sender and leave headroom.
    const uint32_t header_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    auto ph_cfg = CircularBufferConfig(2 * header_size, {{kPacketHeaderCbIndex, tt::DataFormat::UInt32}})
                      .set_page_size(kPacketHeaderCbIndex, header_size);
    CreateCircularBuffer(program, service_core, ph_cfg);

    auto accessor_ct = TensorAccessorArgs(output_buffer).get_compile_time_args();

    std::vector<uint32_t> ct_args = {
        socket_config_addr,
        termination_addr,
        plan.socket_page_size,
        plan.num_socket_pages,
        plan.pages_per_chunk,
        tensor_page_size,
        static_cast<uint32_t>(output_buffer.address()),
        static_cast<uint32_t>(kPacketHeaderCbIndex),
        ws.enabled ? 1u : 0u,
        ws.sem_addr,      // data_ready_sem_addr
        ws.counter_addr,  // consumed_counter_addr
        ws.mcast_noc_x_start,
        ws.mcast_noc_y_start,
        ws.mcast_noc_x_end,
        ws.mcast_noc_y_end,
        ws.num_workers,
    };
    ct_args.insert(ct_args.end(), accessor_ct.begin(), accessor_ct.end());

    auto kernel = CreateKernel(
        program,
        "ttnn/core/tensor/kernels/persistent_d2d_receiver.cpp",
        service_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = ct_args,
        });

    // The receiver returns socket credits to the sender, so its fabric
    // connection runs receiver -> sender.
    std::vector<uint32_t> rt_args;
    tt::tt_fabric::append_fabric_connection_rt_args(
        receiver_node, sender_node, link_index, program, service_core, rt_args);
    SetRuntimeArgs(program, kernel, service_core, rt_args);
    return program;
}

// CT-arg layout must stay in sync with persistent_d2d_sender.cpp.
Program build_sender_program(
    const Buffer& input_buffer,
    const CoreCoord& service_core,
    uint32_t socket_config_addr,
    uint32_t termination_addr,
    const ChunkPlan& plan,
    uint32_t tensor_page_size,
    uint32_t fabric_max_payload_size,
    DataType dtype,
    const WorkerSyncArgs& ws,
    const tt::tt_fabric::FabricNodeId& sender_node,
    const tt::tt_fabric::FabricNodeId& receiver_node,
    uint32_t link_index) {
    auto program = CreateProgram();

    // Single-slot scratch CB sized to one socket page (DRAM -> CB staging).
    auto scratch_cfg =
        CircularBufferConfig(plan.socket_page_size, {{kScratchCbIndex, datatype_to_dataformat_converter(dtype)}})
            .set_page_size(kScratchCbIndex, plan.socket_page_size);
    CreateCircularBuffer(program, service_core, scratch_cfg);

    const uint32_t header_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    auto ph_cfg = CircularBufferConfig(2 * header_size, {{kPacketHeaderCbIndex, tt::DataFormat::UInt32}})
                      .set_page_size(kPacketHeaderCbIndex, header_size);
    CreateCircularBuffer(program, service_core, ph_cfg);

    auto accessor_ct = TensorAccessorArgs(input_buffer).get_compile_time_args();

    std::vector<uint32_t> ct_args = {
        socket_config_addr,
        termination_addr,
        plan.socket_page_size,
        plan.num_socket_pages,
        plan.pages_per_chunk,
        tensor_page_size,
        static_cast<uint32_t>(input_buffer.address()),
        static_cast<uint32_t>(kScratchCbIndex),
        static_cast<uint32_t>(kPacketHeaderCbIndex),
        fabric_max_payload_size,
        ws.enabled ? 1u : 0u,
        ws.counter_addr,  // data_ready_counter_addr
        ws.sem_addr,      // consumed_sem_addr
        ws.mcast_noc_x_start,
        ws.mcast_noc_y_start,
        ws.mcast_noc_x_end,
        ws.mcast_noc_y_end,
        ws.num_workers,
    };
    ct_args.insert(ct_args.end(), accessor_ct.begin(), accessor_ct.end());

    auto kernel = CreateKernel(
        program,
        "ttnn/core/tensor/kernels/persistent_d2d_sender.cpp",
        service_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = ct_args,
        });

    // The sender writes bulk data downstream, so its fabric connection runs
    // sender -> receiver. The receiver NoC coords come from the socket's
    // downstream encoding at runtime, so the fabric-connection args are the only
    // runtime args.
    std::vector<uint32_t> rt_args;
    tt::tt_fabric::append_fabric_connection_rt_args(
        sender_node, receiver_node, link_index, program, service_core, rt_args);
    SetRuntimeArgs(program, kernel, service_core, rt_args);
    return program;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE

// ===========================================================================
// Factory
// ===========================================================================

std::pair<std::unique_ptr<D2DStreamServiceSender>, std::unique_ptr<D2DStreamServiceReceiver>>
D2DStreamService::create_pair(
    const std::shared_ptr<distributed::MeshDevice>& sender_mesh,
    const std::shared_ptr<distributed::MeshDevice>& receiver_mesh,
    D2DStreamConfig cfg) {
    // --- (a) validate shapes --------------------------------------------------
    TT_FATAL(sender_mesh != nullptr, "D2DStreamService: sender_mesh must not be null");
    TT_FATAL(receiver_mesh != nullptr, "D2DStreamService: receiver_mesh must not be null");
    TT_FATAL(
        sender_mesh->shape() == receiver_mesh->shape(),
        "D2DStreamService: sender_mesh shape {} must equal receiver_mesh shape {} (1:1 coord mapping)",
        sender_mesh->shape(),
        receiver_mesh->shape());

    // --- (b) validate mapper --------------------------------------------------
    TT_FATAL(cfg.mapper != nullptr, "D2DStreamService: cfg.mapper must not be null");

    // --- (c) run the mapper on a zero host tensor to derive per-shard spec +
    //         topology shared by both sides ------------------------------------
    auto mapper = std::move(cfg.mapper);
    const auto distributed_dummy = (*mapper)(CMAKE_UNIQUE_NAMESPACE::make_zero_host_tensor(cfg.global_spec));
    const auto& per_shard_spec = distributed_dummy.tensor_spec();
    const auto& topology = distributed_dummy.tensor_topology();
    const auto& coords = topology.mesh_coords();

    // --- (d) allocate sender + receiver backing tensors -----------------------
    // The per-shard spec is identical on both sides (symmetric-mapping
    // invariant); only the target mesh differs.
    Tensor sender_backing = create_device_tensor(per_shard_spec, sender_mesh.get(), topology);
    Tensor receiver_backing = create_device_tensor(per_shard_spec, receiver_mesh.get(), topology);

    // --- (e) claim one service core per coord on each side --------------------
    auto sender_service_cores = CMAKE_UNIQUE_NAMESPACE::claim_service_cores(sender_mesh, coords, "sender");
    auto receiver_service_cores = CMAKE_UNIQUE_NAMESPACE::claim_service_cores(receiver_mesh, coords, "receiver");

    // --- (f) build the SocketConnection list + create the MeshSocket pair -----
    // One connection per PARTICIPATING coord (topology.mesh_coords(), NOT the
    // full mesh range): a stray connection on an unparticipating coord asserts
    // inside MeshSocket::create_socket_pair. Each connection wires sender coord
    // (x, y) on its service core 1:1 to receiver coord (x, y) on its service
    // core.
    std::vector<distributed::SocketConnection> connections;
    connections.reserve(coords.size());
    for (const auto& coord : coords) {
        connections.emplace_back(
            distributed::MeshCoreCoord(coord, sender_service_cores.at(coord)),
            distributed::MeshCoreCoord(coord, receiver_service_cores.at(coord)));
    }

    // socket_mem_config is forwarded verbatim (socket_storage_type, fifo_size,
    // sub-device fields). create_socket_pair allocates the sender config buffer,
    // the receiver config buffer, and the receiver data FIFO, then runs the
    // same-mesh / cross-mesh handshake.
    distributed::SocketConfig socket_config(connections, cfg.socket_mem_config);
    auto socket_pair = distributed::MeshSocket::create_socket_pair(sender_mesh, receiver_mesh, socket_config);
    auto& sender_socket = socket_pair.first;
    auto& receiver_socket = socket_pair.second;

    // Sanity print: the active cores on each endpoint must be the claimed
    // service cores, 1:1 per coord.
    for (const auto& active : sender_socket.get_active_cores()) {
        log_info(
            tt::LogMetal,
            "D2DStreamService: sender socket active core coord={} logical=({}, {})",
            active.device_coord,
            active.core_coord.x,
            active.core_coord.y);
    }
    for (const auto& active : receiver_socket.get_active_cores()) {
        log_info(
            tt::LogMetal,
            "D2DStreamService: receiver socket active core coord={} logical=({}, {})",
            active.device_coord,
            active.core_coord.x,
            active.core_coord.y);
    }

    // --- (g) derive the shared chunk plan -------------------------------------
    // V0 supports L1 socket storage only — the sender's fabric write targets the
    // receiver service core's L1 bank (get_noc_addr_from_bank_id<false>).
    TT_FATAL(
        cfg.socket_mem_config.socket_storage_type == BufferType::L1,
        "D2DStreamService: V0 supports socket_storage_type == L1 only");

    const uint32_t l1_alignment = hal::get_l1_alignment();
    const uint32_t tensor_page_size = sender_backing.buffer()->aligned_page_size();
    const uint32_t tensor_num_pages = sender_backing.buffer()->num_pages();
    TT_FATAL(
        tensor_page_size % l1_alignment == 0,
        "D2DStreamService: tensor page size {} must be L1-aligned ({}). V0 supports UINT32 ROW_MAJOR DRAM where this "
        "holds.",
        tensor_page_size,
        l1_alignment);

    // The socket FIFO doubles as the scratch budget: pack as many tensor pages
    // into a socket page as the FIFO holds. socket_page_size <= fifo_size by
    // construction, so the receiver FIFO always holds a whole socket page.
    const auto plan =
        CMAKE_UNIQUE_NAMESPACE::derive_chunk_plan(tensor_page_size, tensor_num_pages, cfg.socket_mem_config.fifo_size);

    const auto fabric_max_payload_size = static_cast<uint32_t>(
        tt::round_down(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes(), static_cast<size_t>(l1_alignment)));
    TT_FATAL(fabric_max_payload_size > 0, "D2DStreamService: fabric max payload size rounded to zero");

    // --- (h) allocate the per-side worker-sync resources (step 4) -------------
    const uint32_t sender_num_workers = CMAKE_UNIQUE_NAMESPACE::core_range_size(cfg.sender_worker_cores);
    const uint32_t receiver_num_workers = CMAKE_UNIQUE_NAMESPACE::core_range_size(cfg.receiver_worker_cores);

    auto sender_termination_addrs =
        CMAKE_UNIQUE_NAMESPACE::allocate_service_core_words(sender_mesh, sender_service_cores);
    auto receiver_termination_addrs =
        CMAKE_UNIQUE_NAMESPACE::allocate_service_core_words(receiver_mesh, receiver_service_cores);
    auto sender_data_ready_counter_addrs =
        CMAKE_UNIQUE_NAMESPACE::allocate_service_core_words(sender_mesh, sender_service_cores);
    auto receiver_consumed_counter_addrs =
        CMAKE_UNIQUE_NAMESPACE::allocate_service_core_words(receiver_mesh, receiver_service_cores);

    // Mesh-wide worker-grid GlobalSemaphores (same L1 address on every
    // (device, worker core)). Allocated now so the getters can expose them to
    // the worker kernels landing in steps 7/8.
    auto consumed_sem = ttnn::global_semaphore::create_global_semaphore(
        sender_mesh.get(), CoreRangeSet(cfg.sender_worker_cores), /*initial_value=*/0, BufferType::L1);
    auto data_ready_sem = ttnn::global_semaphore::create_global_semaphore(
        receiver_mesh.get(), CoreRangeSet(cfg.receiver_worker_cores), /*initial_value=*/0, BufferType::L1);

    // --- (i) build the persistent receiver + sender workloads (steps 5 + 6) ---
    // Worker sync is always on (M4, steps 7-8): the receiver multicast-incs
    // data_ready_sem after each transfer lands and waits for num_workers acks on
    // consumed_counter; the sender multicast-incs consumed_sem after each drain.
    // The sender's data_ready_counter gate is always-on regardless. Tests that
    // run without real workers (M3) simulate the acks from host and rely on the
    // service kernels' termination poll to tear down cleanly.
    constexpr bool worker_sync_enabled = true;

    auto receiver_workload = std::make_unique<distributed::MeshWorkload>();
    auto sender_workload = std::make_unique<distributed::MeshWorkload>();

    const uint32_t sender_socket_config_addr = static_cast<uint32_t>(sender_socket.get_config_buffer()->address());
    const uint32_t receiver_socket_config_addr = static_cast<uint32_t>(receiver_socket.get_config_buffer()->address());

    for (const auto& coord : coords) {
        // ---- receiver program ----
        const Buffer* recv_buf = receiver_backing.mesh_buffer().get_device_buffer(coord);
        TT_FATAL(recv_buf != nullptr, "D2DStreamService: receiver device buffer missing for coord {}", coord);
        auto* recv_device = receiver_mesh->get_device(coord);
        const CoreCoord recv_service_core = receiver_service_cores.at(coord);

        const auto receiver_node = receiver_mesh->get_fabric_node_id(coord);
        const auto recv_upstream_sender_node =
            receiver_socket.get_fabric_node_id(distributed::SocketEndpoint::SENDER, coord);
        const auto recv_links = tt::tt_fabric::get_forwarding_link_indices(receiver_node, recv_upstream_sender_node);
        TT_FATAL(!recv_links.empty(), "D2DStreamService: no fabric link receiver->sender at coord {}", coord);

        const auto recv_ws = CMAKE_UNIQUE_NAMESPACE::make_worker_sync_args(
            recv_device,
            cfg.receiver_worker_cores,
            receiver_num_workers,
            static_cast<uint32_t>(data_ready_sem.address()),
            static_cast<uint32_t>(receiver_consumed_counter_addrs.at(coord)),
            worker_sync_enabled);

        receiver_workload->add_program(
            distributed::MeshCoordinateRange(coord),
            CMAKE_UNIQUE_NAMESPACE::build_receiver_program(
                *recv_buf,
                recv_service_core,
                receiver_socket_config_addr,
                static_cast<uint32_t>(receiver_termination_addrs.at(coord)),
                plan,
                tensor_page_size,
                recv_ws,
                receiver_node,
                recv_upstream_sender_node,
                recv_links.front()));

        // ---- sender program ----
        const Buffer* send_buf = sender_backing.mesh_buffer().get_device_buffer(coord);
        TT_FATAL(send_buf != nullptr, "D2DStreamService: sender device buffer missing for coord {}", coord);
        auto* send_device = sender_mesh->get_device(coord);
        const CoreCoord send_service_core = sender_service_cores.at(coord);

        const auto sender_node = sender_mesh->get_fabric_node_id(coord);
        const auto send_downstream_receiver_node =
            sender_socket.get_fabric_node_id(distributed::SocketEndpoint::RECEIVER, coord);
        const auto send_links = tt::tt_fabric::get_forwarding_link_indices(sender_node, send_downstream_receiver_node);
        TT_FATAL(!send_links.empty(), "D2DStreamService: no fabric link sender->receiver at coord {}", coord);

        const auto send_ws = CMAKE_UNIQUE_NAMESPACE::make_worker_sync_args(
            send_device,
            cfg.sender_worker_cores,
            sender_num_workers,
            static_cast<uint32_t>(consumed_sem.address()),
            static_cast<uint32_t>(sender_data_ready_counter_addrs.at(coord)),
            worker_sync_enabled);

        sender_workload->add_program(
            distributed::MeshCoordinateRange(coord),
            CMAKE_UNIQUE_NAMESPACE::build_sender_program(
                *send_buf,
                send_service_core,
                sender_socket_config_addr,
                static_cast<uint32_t>(sender_termination_addrs.at(coord)),
                plan,
                tensor_page_size,
                fabric_max_payload_size,
                sender_backing.dtype(),
                send_ws,
                sender_node,
                send_downstream_receiver_node,
                send_links.front()));
    }

    // --- (j) stash everything into the two Impls ------------------------------
    // CoreRange / TensorSpec aren't default-constructible, so build each Impl
    // via aggregate init (all fields at once).
    auto sender_impl = std::make_unique<D2DStreamServiceSender::Impl>(D2DStreamServiceSender::Impl{
        .mesh_device = sender_mesh,
        .per_shard_spec = per_shard_spec,
        .backing_tensor = sender_backing,
        .worker_cores = cfg.sender_worker_cores,
        .service_cores = std::move(sender_service_cores),
        .socket = std::move(sender_socket),
        .socket_page_size = plan.socket_page_size,
        .num_socket_pages = plan.num_socket_pages,
        .pages_per_chunk = plan.pages_per_chunk,
        .num_workers = sender_num_workers,
        .data_ready_counter_addrs = std::move(sender_data_ready_counter_addrs),
        .termination_addrs = std::move(sender_termination_addrs),
        .consumed_sem = std::move(consumed_sem),
        .workload = std::move(sender_workload),
        .launched = false,
    });

    auto receiver_impl = std::make_unique<D2DStreamServiceReceiver::Impl>(D2DStreamServiceReceiver::Impl{
        .mesh_device = receiver_mesh,
        .per_shard_spec = per_shard_spec,
        .backing_tensor = receiver_backing,
        .worker_cores = cfg.receiver_worker_cores,
        .service_cores = std::move(receiver_service_cores),
        .socket = std::move(receiver_socket),
        .socket_page_size = plan.socket_page_size,
        .num_socket_pages = plan.num_socket_pages,
        .pages_per_chunk = plan.pages_per_chunk,
        .num_workers = receiver_num_workers,
        .consumed_counter_addrs = std::move(receiver_consumed_counter_addrs),
        .termination_addrs = std::move(receiver_termination_addrs),
        .data_ready_sem = std::move(data_ready_sem),
        .workload = std::move(receiver_workload),
        .launched = false,
    });

    auto sender_handle = std::unique_ptr<D2DStreamServiceSender>(new D2DStreamServiceSender(std::move(sender_impl)));
    auto receiver_handle =
        std::unique_ptr<D2DStreamServiceReceiver>(new D2DStreamServiceReceiver(std::move(receiver_impl)));

    // --- (k) launch the persistent kernels (non-blocking) ---------------------
    // Receiver first so it's parked on its socket wait before the sender starts
    // pushing pages.
    EnqueueMeshWorkload(
        receiver_handle->impl_->mesh_device->mesh_command_queue(),
        *receiver_handle->impl_->workload,
        /*blocking=*/false);
    receiver_handle->impl_->launched = true;
    EnqueueMeshWorkload(
        sender_handle->impl_->mesh_device->mesh_command_queue(), *sender_handle->impl_->workload, /*blocking=*/false);
    sender_handle->impl_->launched = true;

    return {std::move(sender_handle), std::move(receiver_handle)};
}

}  // namespace tt::tt_metal
