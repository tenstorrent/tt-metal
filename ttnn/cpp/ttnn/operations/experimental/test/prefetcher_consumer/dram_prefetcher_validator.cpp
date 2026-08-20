// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher_validator.hpp"

#include <tt_stl/assert.hpp>
#include <tt_stl/reflection.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal.hpp>

#include <unordered_map>

#include "ttnn/operations/experimental/tensor_prefetcher/tensor_prefetcher.hpp"

namespace ttnn::operations::experimental::test {

namespace {
constexpr uint32_t kValidatorRemoteCBId = 31;
constexpr uint32_t kValidatorScratchCBId = 0;

// Everything both validator paths derive from the delivery target's topology plus the source
// tensor's shape. Kept in one place because the whole point of the validator is that the host
// re-derives the expected (receiver, block) -> tile mapping independently of the prefetcher; two
// copies of that derivation could drift apart and start agreeing with a bug.
struct ValidatorReceiverPlan {
    CoreCoord core;
    uint32_t bank_id;
    uint32_t bank_local_recv;
    uint32_t ring_pos;
};

struct ValidatorGeometry {
    uint32_t num_blocks = 0;
    uint32_t num_senders = 0;
    uint32_t k_block_w_tiles = 0;
    uint32_t total_n_tiles = 0;
    uint32_t n_per_recv_tiles = 0;
    uint32_t page_bytes_per_recv = 0;
    tt::DataFormat dataformat = tt::DataFormat::Invalid;
    std::vector<ValidatorReceiverPlan> receivers;
};

ValidatorGeometry compute_validator_geometry(
    const ttnn::Tensor& source_tensor, const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sr_mapping) {
    using namespace tt::tt_metal;
    ValidatorGeometry geom;

    // Ring topology. Both a GlobalCircularBuffer and a PersistentDFB expose this same mapping with
    // sender.x == bank id, and both DRAM-sender factories place senders identically.
    geom.num_senders = static_cast<uint32_t>(sr_mapping.size());
    uint32_t max_bank_id = 0;
    for (const auto& [sender_logical, receivers] : sr_mapping) {
        const uint32_t bank_id = static_cast<uint32_t>(sender_logical.x);
        max_bank_id = bank_id > max_bank_id ? bank_id : max_bank_id;
        geom.num_blocks += receivers.num_cores();
    }
    const uint32_t num_dram_banks = max_bank_id + 1;
    TT_FATAL(
        num_dram_banks > 0 && geom.num_blocks % num_dram_banks == 0,
        "Validator: total receiver count ({}) must divide evenly across {} DRAM banks",
        geom.num_blocks,
        num_dram_banks);
    const uint32_t receivers_per_bank = geom.num_blocks / num_dram_banks;

    // Per-tensor geometry (single-tensor path; see prefetcher_matmul_design.md §3).
    // Read shape from the tensor's logical (padded) shape so this works for both legacy
    // WIDTH_SHARDED (one shard per bank) and ND_SHARDED (num_shards = ring_size,
    // receiver-contiguous) layouts. The per-(ring_pos, block) tile mapping is
    // layout-mode-invariant; only the ring_pos -> (bank, recv_idx) formula differs.
    Buffer* tensor_buffer = source_tensor.buffer();
    const auto& padded_shape = source_tensor.padded_shape();
    TT_FATAL(
        padded_shape.rank() >= 2,
        "Validator: source tensor padded shape must be at least rank 2; got rank {}",
        padded_shape.rank());
    const auto& tile_spec = source_tensor.tensor_spec().tile();
    const auto tile_shape = tile_spec.get_tile_shape();
    const uint32_t tile_h = tile_shape[0];
    const uint32_t tile_w = tile_shape[1];
    const uint32_t K_elems = padded_shape[-2];
    const uint32_t N_elems = padded_shape[-1];
    TT_FATAL(
        K_elems % tile_h == 0 && N_elems % tile_w == 0,
        "Validator: tensor padded shape ({}, {}) must be tile-aligned (tile {}x{})",
        K_elems,
        N_elems,
        tile_h,
        tile_w);
    const uint32_t k_tiles = K_elems / tile_h;
    geom.total_n_tiles = N_elems / tile_w;
    TT_FATAL(
        k_tiles % geom.num_blocks == 0,
        "Validator: k_tiles ({}) must be divisible by num_blocks ({})",
        k_tiles,
        geom.num_blocks);
    const uint32_t ring_size = geom.num_blocks;
    TT_FATAL(
        geom.total_n_tiles % ring_size == 0,
        "Validator: total_n_tiles ({}) must be divisible by ring_size ({})",
        geom.total_n_tiles,
        ring_size);
    geom.n_per_recv_tiles = geom.total_n_tiles / ring_size;
    geom.k_block_w_tiles = k_tiles / geom.num_blocks;
    geom.dataformat = datatype_to_dataformat_converter(source_tensor.dtype());
    geom.page_bytes_per_recv = geom.k_block_w_tiles * geom.n_per_recv_tiles * tile_spec.get_tile_size(geom.dataformat);

    // Layout detection: ND_SHARDED tensors with `num_shards == ring_size` are the
    // receiver-contiguous DRAM-core layout. Under ROUND_ROBIN_1D shard distribution the natural
    // pairing is strided (bank b feeds ring positions b, b + num_senders, ...). Under
    // CONTIGUOUS_1D, bank b owns a contiguous run of shards, so it feeds the contiguous ring arc
    // b*R .. b*R+R-1 (same pairing as legacy WIDTH_SHARDED).
    const bool is_recv_contig = source_tensor.memory_config().memory_layout() == TensorMemoryLayout::ND_SHARDED &&
                                tensor_buffer->buffer_distribution_spec().has_value() &&
                                tensor_buffer->buffer_distribution_spec()->num_shards() == ring_size;
    const bool is_shard_contiguous_recv_contig =
        is_recv_contig && tensor_buffer->buffer_distribution_spec()->shard_distribution_strategy() ==
                              ShardDistributionStrategy::CONTIGUOUS_1D;
    const bool strided_pairing = is_recv_contig && !is_shard_contiguous_recv_contig;

    // Per-receiver plan. Receiver enumeration within a sender's CoreRangeSet must match the order
    // the sender addresses them in its NOC XY table (row-major).
    std::unordered_map<uint32_t, uint32_t> receivers_seen_by_bank;
    for (uint32_t s = 0; s < geom.num_senders; ++s) {
        const auto& [sender_logical, receivers_set] = sr_mapping[s];
        const uint32_t bank_id = sender_logical.x;
        const auto recv_cores = corerange_to_cores(receivers_set, std::nullopt, /*row_wise=*/true);
        const uint32_t recv_index_base = receivers_seen_by_bank[bank_id];
        receivers_seen_by_bank[bank_id] = recv_index_base + static_cast<uint32_t>(recv_cores.size());
        for (uint32_t r = 0; r < recv_cores.size(); ++r) {
            const uint32_t bank_local_recv = recv_index_base + r;
            TT_FATAL(
                bank_local_recv < receivers_per_bank,
                "Sender {} on bank {} maps receiver {} past receivers_per_bank {}",
                s,
                bank_id,
                bank_local_recv,
                receivers_per_bank);
            const uint32_t ring_pos = strided_pairing ? (bank_id + bank_local_recv * num_dram_banks)
                                                      : (bank_id * receivers_per_bank + bank_local_recv);
            geom.receivers.push_back(ValidatorReceiverPlan{
                .core = recv_cores[r], .bank_id = bank_id, .bank_local_recv = bank_local_recv, .ring_pos = ring_pos});
        }
    }
    return geom;
}
}  // namespace

void DramPrefetcherValidatorDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    TT_FATAL(attrs.num_layers > 0, "num_layers must be > 0");
    const auto* tensor_buffer = tensor_args.source_tensor.buffer();
    TT_FATAL(tensor_buffer != nullptr, "source_tensor must be on device");
    TT_FATAL(tensor_buffer->is_dram(), "source_tensor must be a DRAM buffer");
    TT_FATAL(attrs.global_cb.has_value(), "global_cb required");
    TT_FATAL(attrs.global_cb->receiver_cores().num_cores() > 0, "GCB has no receiver cores");

    const auto& sr_mapping = attrs.global_cb->sender_receiver_core_mapping();
    TT_FATAL(!sr_mapping.empty(), "GCB has no senders");
}

void DramPrefetcherValidatorDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {}

DramPrefetcherValidatorDeviceOperation::spec_return_value_t
DramPrefetcherValidatorDeviceOperation::compute_output_specs(const operation_attributes_t&, const tensor_args_t&) {
    return std::vector<tt::tt_metal::TensorSpec>{};
}

DramPrefetcherValidatorDeviceOperation::tensor_return_value_t
DramPrefetcherValidatorDeviceOperation::create_output_tensors(const operation_attributes_t&, const tensor_args_t&) {
    return std::vector<ttnn::Tensor>{};
}

ttsl::hash::hash_t DramPrefetcherValidatorDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // GlobalCircularBuffer / Tensor aren't reflection-hashable here; pick the bits that
    // determine Program shape: scalar attrs, GCB identity, the source tensor's DRAM
    // address (compile-time arg via TensorAccessorArgs), and its dataformat.
    const auto* tensor_buffer = tensor_args.source_tensor.buffer();
    const tt::DataFormat dataformat = tt::tt_metal::datatype_to_dataformat_converter(tensor_args.source_tensor.dtype());
    return ttsl::hash::hash_objects_with_default_seed(
        ttsl::hash::type_hash<DramPrefetcherValidatorDeviceOperation>,
        attrs.num_layers,
        attrs.print_stride,
        attrs.streaming,
        attrs.rotation,
        static_cast<uint64_t>(attrs.global_cb->config_address()),
        static_cast<uint64_t>(tensor_buffer != nullptr ? tensor_buffer->address() : 0),
        static_cast<uint32_t>(dataformat));
}

ttnn::device_operation::CachedProgram<DramPrefetcherValidatorDeviceOperation::ProgramFactory::shared_variables_t>
DramPrefetcherValidatorDeviceOperation::ProgramFactory::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& /*mesh_coordinate*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/) {
    using namespace tt::tt_metal;

    const auto& source_tensor = tensor_args.source_tensor;
    const auto& global_cb = operation_attributes.global_cb.value();

    const ValidatorGeometry geom = compute_validator_geometry(source_tensor, global_cb.sender_receiver_core_mapping());
    Buffer* tensor_buffer = source_tensor.buffer();
    const CoreRangeSet receiver_cores = global_cb.receiver_cores();

    Program program = CreateProgram();

    // Receiver-side remote CB: wait_front/pop_front units are one full per-receiver block.
    CircularBufferConfig remote_cfg(geom.page_bytes_per_recv);
    remote_cfg.remote_index(kValidatorRemoteCBId)
        .set_page_size(geom.page_bytes_per_recv)
        .set_data_format(geom.dataformat);
    tt::tt_metal::experimental::CreateCircularBuffer(program, receiver_cores, remote_cfg, global_cb);

    // Scratch CB: holds the expected page bytes during a single block comparison.
    CircularBufferConfig scratch_cfg(geom.page_bytes_per_recv, {{kValidatorScratchCBId, geom.dataformat}});
    scratch_cfg.set_page_size(kValidatorScratchCBId, geom.page_bytes_per_recv);
    CreateCircularBuffer(program, receiver_cores, scratch_cfg);

    // Compile-time args: scalars, then the TensorAccessor args for the source tensor.
    std::vector<uint32_t> compile_args = {
        kValidatorRemoteCBId,
        kValidatorScratchCBId,
        operation_attributes.num_layers,
        geom.num_blocks,
        geom.num_senders,
        operation_attributes.print_stride,
        operation_attributes.streaming ? 1u : 0u,
    };
    TensorAccessorArgs(*tensor_buffer).append_to(compile_args);

    KernelHandle kernel_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/gcb_validator_receiver.cpp",
        receiver_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = compile_args});

    const uint32_t bank_base_addr = static_cast<uint32_t>(tensor_buffer->address());
    for (const auto& plan : geom.receivers) {
        // Lead physical block this receiver expects at FIFO position 0 under streaming:
        // rotation[ring_pos] when a rotation was supplied (must match the prefetcher), else
        // ring_pos for the identity (natural topology) order.
        uint32_t lead_block = plan.ring_pos;
        if (!operation_attributes.rotation.empty()) {
            TT_FATAL(
                plan.ring_pos < operation_attributes.rotation.size(),
                "Validator rotation has {} entries but ring_pos {} indexes past it",
                operation_attributes.rotation.size(),
                plan.ring_pos);
            lead_block = operation_attributes.rotation[plan.ring_pos];
        }
        const std::vector<uint32_t> rt_args = {
            plan.bank_id,
            plan.bank_local_recv,
            bank_base_addr,
            geom.k_block_w_tiles,
            geom.total_n_tiles,
            geom.n_per_recv_tiles,
            plan.ring_pos * geom.n_per_recv_tiles,
            lead_block,
        };
        SetRuntimeArgs(program, kernel_id, plan.core, rt_args);
    }

    return {std::move(program), shared_variables_t{}};
}

void DramPrefetcherValidatorDeviceOperation::ProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& /*cached_workload*/,
    const operation_attributes_t& /*attrs*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    // Nothing to override — buffer address is part of the cache key via compute_program_hash.
}

void test_dram_prefetcher_validator(
    tt::tt_metal::distributed::MeshDevice* /*mesh_device*/,
    const ttnn::Tensor& source_tensor,
    uint32_t num_layers,
    uint32_t print_stride,
    const tt::tt_metal::experimental::GlobalCircularBuffer& global_cb,
    bool streaming,
    const std::vector<uint32_t>& rotation) {
    using OperationType = DramPrefetcherValidatorDeviceOperation;
    OperationType::operation_attributes_t attrs{
        .num_layers = num_layers,
        .print_stride = print_stride,
        .global_cb = global_cb,
        .streaming = streaming,
        .rotation = rotation,
    };
    OperationType::tensor_args_t tensor_args{.source_tensor = source_tensor};
    ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

void test_tensor_prefetcher_pdfb_validator(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const ttnn::Tensor& source_tensor,
    uint32_t num_layers,
    uint32_t print_stride,
    const ttnn::operations::experimental::PersistentDFBHandle& persistent_dfb_handle) {
    using namespace tt::tt_metal;
    namespace metal_exp = tt::tt_metal::experimental;

    TT_FATAL(persistent_dfb_handle.pdfb != nullptr, "persistent_dfb must not be null");
    auto& persistent_dfb = *persistent_dfb_handle.pdfb;
    TT_FATAL(num_layers > 0, "num_layers must be > 0");
    Buffer* tensor_buffer = source_tensor.buffer();
    TT_FATAL(tensor_buffer != nullptr, "source_tensor must be on device");
    TT_FATAL(tensor_buffer->is_dram(), "source_tensor must be a DRAM buffer");

    const auto& sr_mapping = metal_exp::persistent_dfb_sender_receiver_core_mapping(persistent_dfb);
    TT_FATAL(!sr_mapping.empty(), "PersistentDFB has no senders");
    const CoreRangeSet receiver_cores = metal_exp::persistent_dfb_receiver_cores(persistent_dfb);
    TT_FATAL(receiver_cores.num_cores() > 0, "PersistentDFB has no receiver cores");

    const ValidatorGeometry geom = compute_validator_geometry(source_tensor, sr_mapping);
    const uint32_t entry_size = metal_exp::persistent_dfb_entry_size(persistent_dfb);
    TT_FATAL(
        geom.page_bytes_per_recv == entry_size,
        "Validator: this tensor delivers {} B per receiver per block, but the PersistentDFB was created with "
        "entry_size {} B. Create it with entry_size == the per-receiver block size.",
        geom.page_bytes_per_recv,
        entry_size);

    Program program = CreateProgram();

    // No remote CB: a PersistentDFB consumer Attaches instead, which hands the kernel a
    // program-local slot id. No entry_size override, so the device constructor stays on its
    // same-epoch fast path and publishes no padding credits.
    const uint8_t persistent_dfb_id =
        metal_exp::AttachPersistentDFB(program, persistent_dfb, receiver_cores, std::nullopt);

    // Scratch CB: holds the expected entry bytes during a single block comparison.
    CircularBufferConfig scratch_cfg(geom.page_bytes_per_recv, {{kValidatorScratchCBId, geom.dataformat}});
    scratch_cfg.set_page_size(kValidatorScratchCBId, geom.page_bytes_per_recv);
    CreateCircularBuffer(program, receiver_cores, scratch_cfg);

    std::vector<uint32_t> compile_args = {
        static_cast<uint32_t>(persistent_dfb_id),
        kValidatorScratchCBId,
        num_layers,
        geom.num_blocks,
        print_stride,
    };
    TensorAccessorArgs(*tensor_buffer).append_to(compile_args);

    KernelHandle kernel_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/pdfb_validator_receiver.cpp",
        receiver_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = compile_args});

    const uint32_t bank_base_addr = static_cast<uint32_t>(tensor_buffer->address());
    for (const auto& plan : geom.receivers) {
        const std::vector<uint32_t> rt_args = {
            plan.bank_id,
            plan.bank_local_recv,
            bank_base_addr,
            geom.k_block_w_tiles,
            geom.total_n_tiles,
            geom.n_per_recv_tiles,
            plan.ring_pos * geom.n_per_recv_tiles,
        };
        SetRuntimeArgs(program, kernel_id, plan.core, rt_args);
    }

    tt::tt_metal::distributed::MeshWorkload workload;
    workload.add_program(tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    tt::tt_metal::distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/false);
    tt::tt_metal::distributed::Finish(mesh_device->mesh_command_queue());
}

}  // namespace ttnn::operations::experimental::test
