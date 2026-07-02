// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/services/d2h_socket_service.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <exception>
#include <thread>
#include <unordered_set>

#include <tt_stl/assert.hpp>
#include <tt_stl/reflection.hpp>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <internal/service/service_core_manager.hpp>
#include <tt-metalium/memory_pin.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "socket_service_common.hpp"
#include "tensor/tensor_ops.hpp"
#include "tt_metal/distributed/d2h_stream_service_descriptor.hpp"
#include "tt_metal/distributed/hd_socket_descriptor.hpp"
#include "tt_metal/distributed/shm_resource_tracker.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/global_semaphore.hpp"

namespace tt::tt_metal {

namespace {

distributed::MeshComposerConfig derive_composer_config(const distributed::MeshMapperConfig& mapper_config) {
    ttsl::SmallVector<int> dims;
    for (const auto& p : mapper_config.placements) {
        std::visit(
            [&dims](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, distributed::MeshMapperConfig::Shard>) {
                    dims.push_back(arg.dim);
                }
            },
            p);
    }
    return distributed::MeshComposerConfig{.dims = dims, .mesh_shape_override = mapper_config.mesh_shape_override};
}

struct D2HWorkerSyncArgs {
    bool enabled = false;
    uint32_t transfer_done_sem_addr = 0;
    uint32_t write_ack_counter_addr = 0;
    uint32_t mcast_noc_x_start = 0;
    uint32_t mcast_noc_y_start = 0;
    uint32_t mcast_noc_x_end = 0;
    uint32_t mcast_noc_y_end = 0;
    uint32_t num_workers = 0;
};

struct D2HMetadataArgs {
    bool enabled = false;
    uint32_t metadata_size_bytes = 0;
    uint32_t metadata_l1_addr = 0;
};

Program build_persistent_d2h_program(
    const Buffer& device_buffer,
    const CoreCoord& sender_core,
    uint32_t socket_config_buffer_address,
    uint32_t termination_semaphore_addr,
    const ChunkPlan& plan,
    uint32_t tensor_page_size,
    DataType dtype,
    const D2HWorkerSyncArgs& worker_sync,
    const D2HMetadataArgs& metadata) {
    auto program = CreateProgram();

    constexpr tt::CBIndex scratch_cb_index = tt::CBIndex::c_0;
    auto cb_cfg =
        CircularBufferConfig(plan.socket_page_size, {{scratch_cb_index, datatype_to_dataformat_converter(dtype)}})
            .set_page_size(scratch_cb_index, plan.socket_page_size);
    CreateCircularBuffer(program, sender_core, cb_cfg);

    auto tensor_accessor_args = TensorAccessorArgs(device_buffer);
    auto tensor_accessor_compile_args = tensor_accessor_args.get_compile_time_args();

    std::vector<uint32_t> ct_args = {
        socket_config_buffer_address,
        termination_semaphore_addr,
        plan.socket_page_size,
        plan.num_socket_pages,
        static_cast<uint32_t>(device_buffer.address()),
        tensor_page_size,
        plan.pages_per_chunk,
        static_cast<uint32_t>(scratch_cb_index),
        static_cast<uint32_t>(worker_sync.enabled ? 1u : 0u),
        worker_sync.transfer_done_sem_addr,
        worker_sync.write_ack_counter_addr,
        worker_sync.mcast_noc_x_start,
        worker_sync.mcast_noc_y_start,
        worker_sync.mcast_noc_x_end,
        worker_sync.mcast_noc_y_end,
        worker_sync.num_workers,
        static_cast<uint32_t>(metadata.enabled ? 1u : 0u),
        metadata.metadata_size_bytes,
        metadata.metadata_l1_addr,
    };
    ct_args.insert(ct_args.end(), tensor_accessor_compile_args.begin(), tensor_accessor_compile_args.end());

    CreateKernel(
        program,
        "models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/persistent_d2h_sender.cpp",
        sender_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = ct_args,
        });

    return program;
}

inline void require_d2h_owner(bool is_owner, const char* api) {
    TT_FATAL(
        is_owner,
        "{}: this getter is owner-only. The connector-mode service (built via "
        "D2HStreamService::connect) has no MeshDevice and cannot dispatch worker "
        "workloads, so the worker-sync addresses it would return are "
        "meaningless. Call this from the owner process instead.",
        api);
}

}  // namespace

D2HStreamService::D2HStreamService(const std::shared_ptr<distributed::MeshDevice>& mesh_device, Config cfg) :
    mesh_device_(mesh_device), cfg_(std::move(cfg)) {
    TT_FATAL(mesh_device_ != nullptr, "D2HStreamService: mesh_device must not be null");
    TT_FATAL(cfg_.fifo_size_bytes > 0, "D2HStreamService: fifo_size_bytes must be > 0");
    TT_FATAL(cfg_.scratch_cb_size_bytes > 0, "D2HStreamService: scratch_cb_size_bytes must be > 0");
    TT_FATAL(
        cfg_.metadata_size_bytes == 0 || cfg_.worker_cores.has_value(),
        "D2HStreamService: metadata_size_bytes={} requires Config::worker_cores to be set",
        cfg_.metadata_size_bytes);
    if (cfg_.worker_cores.has_value()) {
        const auto& wr = cfg_.worker_cores.value();
        num_workers_ = (wr.end_coord.x - wr.start_coord.x + 1) * (wr.end_coord.y - wr.start_coord.y + 1);
        if (cfg_.metadata_size_bytes > 0) {
            TT_FATAL(
                cfg_.metadata_master_core.has_value(),
                "D2HStreamService: Config::metadata_master_core is required when metadata_size_bytes > 0");
            const auto& mf = cfg_.metadata_master_core.value();
            TT_FATAL(
                mf.x >= wr.start_coord.x && mf.x <= wr.end_coord.x && mf.y >= wr.start_coord.y &&
                    mf.y <= wr.end_coord.y,
                "D2HStreamService: metadata_master_core ({}, {}) must lie within worker_cores {}",
                mf.x,
                mf.y,
                wr);
            TT_FATAL(num_workers_ >= 1, "D2HStreamService: worker_cores must contain at least one core");
            metadata_master_core_ = mf;
        } else {
            TT_FATAL(num_workers_ >= 1, "D2HStreamService: worker_cores must contain at least one core");
        }
    }

    if (cfg_.mapper == nullptr) {
        ttsl::SmallVector<distributed::MeshMapperConfig::Placement> replicate_all(
            mesh_device_->shape().dims(), distributed::MeshMapperConfig::Replicate{});
        cfg_.mapper = ttnn::distributed::create_mesh_mapper(
            *mesh_device_, distributed::MeshMapperConfig{.placements = std::move(replicate_all)});
    }
    mapper_ = std::move(cfg_.mapper);

    const auto distributed_dummy = (*mapper_)(make_zero_host_tensor(cfg_.global_spec));
    const auto& per_shard_spec = distributed_dummy.tensor_spec();
    const auto& topology = distributed_dummy.tensor_topology();

    device_tensor_ = create_device_tensor(per_shard_spec, mesh_device_.get(), topology);
    per_shard_spec_ = device_tensor_.tensor_spec();

    if (cfg_.composer_config.has_value()) {
        composer_ = ttnn::distributed::create_mesh_composer(*mesh_device_, cfg_.composer_config.value());
    } else {
        const auto comp_cfg = derive_composer_config(mapper_->config());
        // Skip auto-composer when partial-shard mapper dims != mesh dims.
        const auto effective_mesh = mapper_->config().mesh_shape_override.value_or(mesh_device_->shape());
        if (!comp_cfg.dims.empty() && comp_cfg.dims.size() == effective_mesh.dims()) {
            composer_ = ttnn::distributed::create_mesh_composer(*mesh_device_, comp_cfg);
        }
    }

    auto& svc = tt::tt_metal::internal::service_core_manager();
    const auto& coords = topology.mesh_coords();
    for (const auto& coord : coords) {
        auto* d = mesh_device_->get_device(coord);
        auto claimable = svc.get_claimable_cores(d);
        TT_FATAL(!claimable.empty(), "D2HStreamService: no claimable service core on device at coord {}", coord);
        const CoreCoord chosen = claimable.front();
        svc.claim(d, {chosen});
        service_cores_.emplace(coord, chosen);
    }

    // Also claim service cores under MeshDevice id for CB validation across mesh reopens.
    {
        const auto mesh_claimed = svc.claimed_cores(mesh_device_->id());
        std::unordered_set<CoreCoord> registered;
        for (const auto& [coord, core] : service_cores_) {
            if (!mesh_claimed.contains(core) && !registered.contains(core)) {
                svc.claim(mesh_device_.get(), {core});
                registered.insert(core);
            }
        }
        mesh_id_claimed_cores_.assign(registered.begin(), registered.end());
    }

    sockets_.reserve(coords.size());
    for (const auto& coord : coords) {
        sockets_.push_back(std::make_unique<distributed::D2HSocket>(
            mesh_device_, distributed::MeshCoreCoord(coord, service_cores_.at(coord)), cfg_.fifo_size_bytes));
    }

    const uint32_t tensor_page_size = device_tensor_.buffer()->page_size();
    const uint32_t tensor_num_pages = device_tensor_.buffer()->num_pages();
    const ChunkPlan plan = derive_chunk_plan(tensor_page_size, tensor_num_pages, cfg_.scratch_cb_size_bytes);
    socket_page_size_ = plan.socket_page_size;
    num_socket_pages_ = plan.num_socket_pages;

    TT_FATAL(
        cfg_.metadata_size_bytes <= socket_page_size_,
        "D2HStreamService: metadata_size_bytes={} exceeds derived socket_page_size={}",
        cfg_.metadata_size_bytes,
        socket_page_size_);

    for (auto& s : sockets_) {
        s->set_page_size(plan.socket_page_size);
    }

    std::vector<uint32_t> zero_word{0};
    for (const auto& coord : coords) {
        auto* d = mesh_device_->get_device(coord);
        const CoreCoord chosen = service_cores_.at(coord);
        const DeviceAddr sem_addr = svc.allocate_l1(d, chosen, sizeof(uint32_t));
        termination_addrs_.emplace(coord, sem_addr);
        tt::tt_metal::detail::WriteToDeviceL1(d, chosen, static_cast<uint32_t>(sem_addr), zero_word);
    }

    for (const auto& coord : coords) {
        auto* d = mesh_device_->get_device(coord);
        const CoreCoord chosen = service_cores_.at(coord);
        const DeviceAddr addr = svc.allocate_l1(d, chosen, sizeof(uint32_t));
        write_ack_addrs_.emplace(coord, addr);
        tt::tt_metal::detail::WriteToDeviceL1(d, chosen, static_cast<uint32_t>(addr), zero_word);
    }

    if (cfg_.worker_cores.has_value()) {
        const auto& worker_range = cfg_.worker_cores.value();

        transfer_done_sem_.emplace(ttnn::global_semaphore::create_global_semaphore(
            mesh_device_.get(), CoreRangeSet(worker_range), /*initial_value=*/0, BufferType::L1));

        if (cfg_.metadata_size_bytes > 0) {
            const uint32_t l1_align = hal::get_l1_alignment();

            const DeviceAddr aligned_shard_size =
                tt::align(static_cast<DeviceAddr>(cfg_.metadata_size_bytes), static_cast<DeviceAddr>(l1_align));
            const CoreRangeSet shard_grid(worker_range);
            distributed::DeviceLocalBufferConfig worker_metadata_local = {
                .page_size = aligned_shard_size,
                .buffer_type = BufferType::L1,
                .sharding_args = BufferShardingArgs(
                    ShardSpecBuffer(shard_grid, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_workers_, 1}),
                    TensorMemoryLayout::HEIGHT_SHARDED),
                .bottom_up = std::nullopt,
                .sub_device_id = std::nullopt,
            };
            distributed::MeshBufferConfig worker_metadata_mesh = distributed::ReplicatedBufferConfig{
                .size = aligned_shard_size * static_cast<DeviceAddr>(num_workers_),
            };
            metadata_worker_buffer_ =
                distributed::MeshBuffer::create(worker_metadata_mesh, worker_metadata_local, mesh_device_.get());
            metadata_worker_l1_addr_ = metadata_worker_buffer_->address();
        }
    } else {
        num_workers_ = 1;
    }

    if (cfg_.metadata_size_bytes > 0) {
        const uint32_t l1_align = hal::get_l1_alignment();
        const DeviceAddr aligned_staging_size =
            tt::align(static_cast<DeviceAddr>(socket_page_size_), static_cast<DeviceAddr>(l1_align));
        for (const auto& coord : coords) {
            auto* d = mesh_device_->get_device(coord);
            const CoreCoord chosen = service_cores_.at(coord);
            const DeviceAddr addr = svc.allocate_l1(d, chosen, aligned_staging_size);
            metadata_input_addrs_.emplace(coord, addr);
            std::vector<uint8_t> zero_meta(aligned_staging_size, 0);
            tt::tt_metal::detail::WriteToDeviceL1(d, chosen, static_cast<uint32_t>(addr), zero_meta, CoreType::WORKER);
        }
        metadata_scratch_.assign(socket_page_size_, std::byte{0});
    }

    workload_ = std::make_unique<distributed::MeshWorkload>();
    for (auto& s : sockets_) {
        const auto core = s->get_active_cores()[0];
        const Buffer* dbuf = device_tensor_.mesh_buffer().get_device_buffer(core.device_coord);
        TT_FATAL(dbuf != nullptr, "D2HStreamService: device buffer missing for coord {}", core.device_coord);
        const uint32_t term_addr = static_cast<uint32_t>(termination_addrs_.at(core.device_coord));

        D2HWorkerSyncArgs worker_sync;
        worker_sync.write_ack_counter_addr = static_cast<uint32_t>(write_ack_addrs_.at(core.device_coord));
        // Host-only path: default num_workers=0 would free-run the sender.
        worker_sync.num_workers = num_workers_;
        if (cfg_.worker_cores.has_value()) {
            const auto& worker_range = cfg_.worker_cores.value();
            auto* d = mesh_device_->get_device(core.device_coord);
            const auto start_phys = d->worker_core_from_logical_core(worker_range.start_coord);
            const auto end_phys = d->worker_core_from_logical_core(worker_range.end_coord);
            worker_sync.enabled = true;
            worker_sync.transfer_done_sem_addr = static_cast<uint32_t>(transfer_done_sem_->address());
            worker_sync.mcast_noc_x_start = static_cast<uint32_t>(start_phys.x);
            worker_sync.mcast_noc_y_start = static_cast<uint32_t>(start_phys.y);
            worker_sync.mcast_noc_x_end = static_cast<uint32_t>(end_phys.x);
            worker_sync.mcast_noc_y_end = static_cast<uint32_t>(end_phys.y);
        }

        D2HMetadataArgs metadata;
        if (cfg_.metadata_size_bytes > 0) {
            metadata.enabled = true;
            metadata.metadata_size_bytes = cfg_.metadata_size_bytes;
            metadata.metadata_l1_addr = static_cast<uint32_t>(metadata_input_addrs_.at(core.device_coord));
        }

        auto program = build_persistent_d2h_program(
            *dbuf,
            core.core_coord,
            s->get_config_buffer_address(),
            term_addr,
            plan,
            tensor_page_size,
            device_tensor_.dtype(),
            worker_sync,
            metadata);
        workload_->add_program(distributed::MeshCoordinateRange(core.device_coord), std::move(program));
    }

    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *workload_, /*blocking=*/false);
}

D2HStreamService::D2HStreamService(
    Config cfg,
    std::vector<std::unique_ptr<distributed::D2HSocket>> sockets,
    uint32_t socket_page_size,
    uint32_t num_socket_pages) :
    is_owner_(false), cfg_(std::move(cfg)) {
    TT_FATAL(!sockets.empty(), "D2HStreamService(connector): sockets vector must not be empty");
    TT_FATAL(cfg_.mapper != nullptr, "D2HStreamService(connector): mapper must be pre-built and supplied");
    TT_FATAL(socket_page_size > 0, "D2HStreamService(connector): socket_page_size must be > 0");
    TT_FATAL(num_socket_pages > 0, "D2HStreamService(connector): num_socket_pages must be > 0");

    mapper_ = std::move(cfg_.mapper);
    socket_page_size_ = socket_page_size;
    num_socket_pages_ = num_socket_pages;
    sockets_ = std::move(sockets);
    for (auto& s : sockets_) {
        s->set_page_size(socket_page_size_);
    }

    const auto distributed_dummy = (*mapper_)(make_zero_host_tensor(cfg_.global_spec));
    per_shard_spec_ = distributed_dummy.tensor_spec();

    if (cfg_.metadata_size_bytes > 0) {
        metadata_scratch_.assign(socket_page_size_, std::byte{0});
    }
}

D2HStreamService::~D2HStreamService() {
    try {
        if (!is_owner_) {
            sockets_.clear();
            return;
        }

        barrier();
        signal_termination();

        if (mesh_device_) {
            distributed::Finish(mesh_device_->mesh_command_queue());
        }

        auto& svc = tt::tt_metal::internal::service_core_manager();
        if (mesh_device_) {
            for (const auto& [coord, core] : service_cores_) {
                auto* d = mesh_device_->get_device(coord);
                svc.wait_done(d, core);
            }
        }

        if (mesh_device_) {
            for (const auto& [coord, addr] : termination_addrs_) {
                auto* d = mesh_device_->get_device(coord);
                svc.deallocate_l1(d, service_cores_.at(coord), addr);
            }
            termination_addrs_.clear();

            for (const auto& [coord, addr] : write_ack_addrs_) {
                auto* d = mesh_device_->get_device(coord);
                svc.deallocate_l1(d, service_cores_.at(coord), addr);
            }
            write_ack_addrs_.clear();

            metadata_worker_buffer_.reset();
            metadata_worker_l1_addr_ = 0;

            for (const auto& [coord, addr] : metadata_input_addrs_) {
                auto* d = mesh_device_->get_device(coord);
                svc.deallocate_l1(d, service_cores_.at(coord), addr);
            }
            metadata_input_addrs_.clear();

            sockets_.clear();

            for (const auto& [coord, core] : service_cores_) {
                auto* d = mesh_device_->get_device(coord);
                svc.release(d, {core});
            }

            if (!mesh_id_claimed_cores_.empty()) {
                svc.release(mesh_device_.get(), mesh_id_claimed_cores_);
                mesh_id_claimed_cores_.clear();
            }
        }

        if (!descriptor_path_.empty()) {
            if (std::remove(descriptor_path_.c_str()) == 0 || errno == ENOENT) {
                distributed::ShmResourceTracker::instance().untrack_file(descriptor_path_);
            }
            descriptor_path_.clear();
        }
    } catch (const std::exception& e) {
        log_warning(tt::LogOp, "D2HStreamService: shutdown failed: {}", e.what());
    } catch (...) {
        log_warning(tt::LogOp, "D2HStreamService: shutdown failed with unknown exception");
    }
}

void D2HStreamService::notify_backing_ready() {
    require_d2h_owner(is_owner_, "D2HStreamService::notify_backing_ready");
    TT_FATAL(
        !cfg_.worker_cores.has_value(),
        "D2HStreamService::notify_backing_ready: workers ack write_ack directly when "
        "Config::worker_cores is set");
    TT_FATAL(mesh_device_ != nullptr, "D2HStreamService::notify_backing_ready: mesh device unavailable");
    for (const auto& [coord, addr] : write_ack_addrs_) {
        auto* d = mesh_device_->get_device(coord);
        const CoreCoord chosen = service_cores_.at(coord);
        std::vector<uint32_t> cur(1);
        tt::tt_metal::detail::ReadFromDeviceL1(d, chosen, static_cast<uint32_t>(addr), sizeof(uint32_t), cur);
        std::vector<uint32_t> next{cur[0] + 1};
        tt::tt_metal::detail::WriteToDeviceL1(d, chosen, static_cast<uint32_t>(addr), next);
    }
}

void D2HStreamService::barrier() {
    for (auto& s : sockets_) {
        s->barrier();
    }
}

void D2HStreamService::signal_termination() {
    if (mesh_device_ == nullptr) {
        return;
    }
    std::vector<uint32_t> one_word{1};
    for (const auto& [coord, addr] : termination_addrs_) {
        auto* d = mesh_device_->get_device(coord);
        const CoreCoord chosen = service_cores_.at(coord);
        tt::tt_metal::detail::WriteToDeviceL1(d, chosen, static_cast<uint32_t>(addr), one_word);
    }
}

std::vector<distributed::D2HSocket*> D2HStreamService::get_sockets() const {
    std::vector<distributed::D2HSocket*> out;
    out.reserve(sockets_.size());
    for (const auto& s : sockets_) {
        out.push_back(s.get());
    }
    return out;
}

CoreRange D2HStreamService::get_worker_cores() const {
    TT_FATAL(
        cfg_.worker_cores.has_value(),
        "D2HStreamService::get_worker_cores: worker-sync was not configured (Config::worker_cores unset).");
    return *cfg_.worker_cores;
}

CoreCoord D2HStreamService::get_metadata_master_core() const {
    TT_FATAL(
        cfg_.metadata_size_bytes > 0 && cfg_.metadata_master_core.has_value(),
        "D2HStreamService::get_metadata_master_core: metadata master is only configured when metadata is enabled.");
    return *cfg_.metadata_master_core;
}

DeviceAddr D2HStreamService::get_write_ack_counter_addr(const distributed::MeshCoordinate& coord) const {
    require_d2h_owner(is_owner_, "D2HStreamService::get_write_ack_counter_addr");
    auto it = write_ack_addrs_.find(coord);
    TT_FATAL(
        it != write_ack_addrs_.end(),
        "D2HStreamService::get_write_ack_counter_addr: no write-ack counter at coord {}",
        coord);
    return it->second;
}

DeviceAddr D2HStreamService::get_worker_metadata_addr() const {
    require_d2h_owner(is_owner_, "D2HStreamService::get_worker_metadata_addr");
    TT_FATAL(metadata_worker_l1_addr_ != 0, "D2HStreamService::get_worker_metadata_addr: metadata was not configured.");
    return metadata_worker_l1_addr_;
}

DeviceAddr D2HStreamService::get_metadata_input_addr(const distributed::MeshCoordinate& coord) const {
    require_d2h_owner(is_owner_, "D2HStreamService::get_metadata_input_addr");
    TT_FATAL(cfg_.metadata_size_bytes > 0, "D2HStreamService::get_metadata_input_addr: metadata was not configured.");
    auto it = metadata_input_addrs_.find(coord);
    TT_FATAL(
        it != metadata_input_addrs_.end(),
        "D2HStreamService::get_metadata_input_addr: no metadata input region at coord {}",
        coord);
    return it->second;
}

DeviceAddr D2HStreamService::get_transfer_done_sem_addr() const {
    require_d2h_owner(is_owner_, "D2HStreamService::get_transfer_done_sem_addr");
    TT_FATAL(
        transfer_done_sem_.has_value(),
        "D2HStreamService::get_transfer_done_sem_addr: worker-sync was not configured.");
    return transfer_done_sem_->address();
}

CoreCoord D2HStreamService::get_service_core(const distributed::MeshCoordinate& coord) const {
    require_d2h_owner(is_owner_, "D2HStreamService::get_service_core");
    auto it = service_cores_.find(coord);
    TT_FATAL(it != service_cores_.end(), "D2HStreamService::get_service_core: no service core at coord {}", coord);
    return it->second;
}

DeviceAddr D2HStreamService::get_metadata_addr(const distributed::MeshCoordinate& coord) const {
    return get_metadata_input_addr(coord);
}

const TensorSpec& D2HStreamService::get_per_shard_spec() const {
    TT_FATAL(per_shard_spec_.has_value(), "D2HStreamService::get_per_shard_spec: per-shard spec not derived");
    return *per_shard_spec_;
}

const Tensor& D2HStreamService::get_backing_tensor() const {
    require_d2h_owner(is_owner_, "D2HStreamService::get_backing_tensor");
    return device_tensor_;
}

std::size_t D2HStreamService::payload_size_bytes() const { return cfg_.global_spec.compute_packed_buffer_size_bytes(); }

std::size_t D2HStreamService::metadata_size_bytes() const { return cfg_.metadata_size_bytes; }

std::string D2HStreamService::export_descriptor(const std::string& service_id) {
    TT_FATAL(is_owner_, "D2HStreamService::export_descriptor: only owner-side services can be exported");
    TT_FATAL(mesh_device_ != nullptr, "D2HStreamService::export_descriptor: mesh device unavailable");
    TT_FATAL(mapper_ != nullptr, "D2HStreamService::export_descriptor: mapper unavailable");

    distributed::D2HStreamServiceDescriptor desc;
    desc.global_shape = cfg_.global_spec.logical_shape();
    desc.global_dtype = cfg_.global_spec.data_type();
    desc.mesh_shape = mesh_device_->shape();
    desc.mapper_config = mapper_->config();
    if (cfg_.composer_config.has_value()) {
        desc.composer_config = cfg_.composer_config.value();
    } else {
        // Mirror ctor: omit composer_config when auto-derived dims are malformed.
        auto comp_cfg = derive_composer_config(mapper_->config());
        const auto effective_mesh = mapper_->config().mesh_shape_override.value_or(mesh_device_->shape());
        if (comp_cfg.dims.size() == effective_mesh.dims()) {
            desc.composer_config = std::move(comp_cfg);
        }
    }
    desc.socket_page_size = socket_page_size_;
    desc.num_socket_pages = num_socket_pages_;
    desc.metadata_size_bytes = cfg_.metadata_size_bytes;

    desc.per_coord_entries.reserve(sockets_.size());
    for (auto& s : sockets_) {
        const auto coord = s->get_active_cores()[0].device_coord;
        desc.per_coord_entries.emplace_back(coord, s->populate_descriptor());
    }

    auto path = distributed::descriptor_path_for_d2h_service(service_id);
    desc.write_to_file(path);
    distributed::ShmResourceTracker::instance().track_file(path);
    descriptor_path_ = path;
    return path;
}

std::unique_ptr<D2HStreamService> D2HStreamService::connect(
    const std::string& service_id, std::optional<uint32_t> timeout_ms) {
    auto desc = distributed::D2HStreamServiceDescriptor::wait_and_read(
        distributed::descriptor_path_for_d2h_service(service_id), timeout_ms.value_or(10000));

    const TensorLayout tensor_layout(
        desc.global_dtype,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});
    TensorSpec global_spec(desc.global_shape, tensor_layout);

    auto mapper = ttnn::distributed::create_mesh_mapper(desc.mesh_shape, desc.mapper_config);

    std::vector<std::unique_ptr<distributed::D2HSocket>> sockets;
    sockets.reserve(desc.per_coord_entries.size());
    for (const auto& [coord, socket_desc] : desc.per_coord_entries) {
        (void)coord;
        sockets.push_back(distributed::D2HSocket::connect_from_descriptor(socket_desc));
    }

    Config cfg{
        .global_spec = std::move(global_spec),
        .mapper = std::move(mapper),
        .composer_config = desc.composer_config,
        .fifo_size_bytes = 0,
        .scratch_cb_size_bytes = 0,
        .worker_cores = std::nullopt,
        .metadata_master_core = std::nullopt,
        .metadata_size_bytes = desc.metadata_size_bytes,
    };

    auto service = std::unique_ptr<D2HStreamService>(
        new D2HStreamService(std::move(cfg), std::move(sockets), desc.socket_page_size, desc.num_socket_pages));
    return service;
}

void D2HStreamService::read_from_tensor(ttsl::Span<std::byte> bytes, ttsl::Span<std::byte> metadata) {
    const size_t expected = cfg_.global_spec.compute_packed_buffer_size_bytes();
    TT_FATAL(
        bytes.size() == expected,
        "D2HStreamService::read_from_tensor: span size {} B does not match global_spec packed size {} B",
        bytes.size(),
        expected);
    TT_FATAL(
        metadata.size() == cfg_.metadata_size_bytes,
        "D2HStreamService::read_from_tensor: metadata span size {} B does not match Config::metadata_size_bytes={}",
        metadata.size(),
        cfg_.metadata_size_bytes);
    TT_FATAL(
        cfg_.global_spec.layout() == Layout::ROW_MAJOR,
        "D2HStreamService::read_from_tensor(span): global_spec must be ROW_MAJOR");

    Tensor host_tensor = (*mapper_)(make_zero_host_tensor(cfg_.global_spec));
    read_from_tensor(host_tensor, metadata);

    const size_t per_shard_bytes = per_shard_spec_->compute_packed_buffer_size_bytes();
    if (per_shard_bytes == expected) {
        const auto coord = sockets_[0]->get_active_cores()[0].device_coord;
        auto shard_opt = host_tensor.host_storage().host_tensor().buffer().get_shard(coord);
        TT_FATAL(shard_opt.has_value(), "D2HStreamService::read_from_tensor: shard missing after read");
        std::memcpy(bytes.data(), shard_opt->view_bytes().data(), expected);
        return;
    }

    TT_FATAL(composer_ != nullptr, "D2HStreamService::read_from_tensor: composer unavailable");
    Tensor composed = composer_->compose(host_tensor);

    // View composed bytes directly; to_vector<uint8_t>() rejects non-UINT8 dtypes.
    const auto& composed_host = composed.host_storage().host_tensor();
    const auto& composed_dhb = composed_host.buffer();
    const auto& composed_coords = composed_dhb.shard_coords();
    TT_FATAL(
        composed_coords.size() == 1,
        "D2HStreamService::read_from_tensor: composed tensor has {} shards, expected a single aggregated shard",
        composed_coords.size());
    auto composed_shard = composed_dhb.get_shard(*composed_coords.begin());
    TT_FATAL(composed_shard.has_value(), "D2HStreamService::read_from_tensor: composed shard not populated");
    const auto composed_bytes = composed_shard->view_bytes();
    TT_FATAL(
        composed_bytes.size() == bytes.size(),
        "D2HStreamService::read_from_tensor: composed size {} != expected {}",
        composed_bytes.size(),
        bytes.size());
    std::memcpy(bytes.data(), composed_bytes.data(), bytes.size());
}

void D2HStreamService::read_from_tensor(Tensor& host_tensor, ttsl::Span<std::byte> metadata) {
    TT_FATAL(
        metadata.size() == cfg_.metadata_size_bytes,
        "D2HStreamService::read_from_tensor: metadata span size {} B does not match Config::metadata_size_bytes={}",
        metadata.size(),
        cfg_.metadata_size_bytes);
    TT_FATAL(per_shard_spec_.has_value(), "D2HStreamService::read_from_tensor: per-shard spec not derived");

    if (is_owner_ && mesh_device_ && !cfg_.worker_cores.has_value()) {
        notify_backing_ready();
    }

    TT_FATAL(
        host_tensor.storage_type() == StorageType::HOST,
        "D2HStreamService::read_from_tensor: expected a preallocated host tensor");

    const auto& host_mesh_tensor = host_tensor.host_storage().host_tensor();
    const auto& dhb = host_mesh_tensor.buffer();
    TT_FATAL(
        host_mesh_tensor.tensor_spec() == *per_shard_spec_,
        "D2HStreamService::read_from_tensor: host tensor per-shard spec mismatch");

    const uint64_t expected_shard_bytes =
        static_cast<uint64_t>(num_socket_pages_) * static_cast<uint64_t>(socket_page_size_);

    std::vector<std::byte*> bases;
    bases.reserve(sockets_.size());
    for (auto& s : sockets_) {
        const auto coord = s->get_active_cores()[0].device_coord;
        TT_FATAL(dhb.is_local(coord), "D2HStreamService::read_from_tensor: no local shard for coord {}", coord);
        auto shard_opt = dhb.get_shard(coord);
        TT_FATAL(shard_opt.has_value(), "D2HStreamService::read_from_tensor: shard not populated at coord {}", coord);
        auto shard_span = shard_opt->view_bytes();
        TT_FATAL(
            shard_span.size() == expected_shard_bytes,
            "D2HStreamService::read_from_tensor: shard at coord {} has {} B, expected {}",
            coord,
            shard_span.size(),
            expected_shard_bytes);
        bases.push_back(shard_span.data());
    }

    std::vector<std::thread> read_threads;
    std::vector<std::exception_ptr> read_errors(sockets_.size());
    read_threads.reserve(sockets_.size());
    for (size_t s = 0; s < sockets_.size(); ++s) {
        read_threads.emplace_back([&, s] {
            try {
                for (uint32_t i = 0; i < num_socket_pages_; ++i) {
                    const size_t offset = static_cast<size_t>(i) * socket_page_size_;
                    sockets_[s]->read(bases[s] + offset, /*num_pages=*/1);
                }
            } catch (...) {
                read_errors[s] = std::current_exception();
            }
        });
    }
    for (auto& thread : read_threads) {
        thread.join();
    }
    for (const auto& error : read_errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }

    if (cfg_.metadata_size_bytes > 0) {
        TT_FATAL(!sockets_.empty(), "D2HStreamService::read_from_tensor: expected at least one socket for metadata");
        sockets_.front()->read(metadata_scratch_.data(), /*num_pages=*/1);
        std::memcpy(metadata.data(), metadata_scratch_.data(), metadata.size());
        for (size_t s = 1; s < sockets_.size(); ++s) {
            sockets_[s]->read(metadata_scratch_.data(), /*num_pages=*/1);
            TT_FATAL(
                std::memcmp(metadata.data(), metadata_scratch_.data(), metadata.size()) == 0,
                "D2HStreamService::read_from_tensor: metadata mismatch across sockets (socket index {})",
                s);
        }
    }
}

}  // namespace tt::tt_metal
