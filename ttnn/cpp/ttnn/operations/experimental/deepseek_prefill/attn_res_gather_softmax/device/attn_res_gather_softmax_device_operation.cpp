// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "attn_res_gather_softmax_device_operation.hpp"

#include <array>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>

#include "ttnn/device.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

using namespace tt::constants;

namespace {

std::vector<std::pair<const ttnn::Tensor*, const char*>> named_operands(const AttnResGatherSoftmaxInputs& tensor_args) {
    std::vector<std::pair<const ttnn::Tensor*, const char*>> operands = {
        {&tensor_args.partial, "partial"},
        {&tensor_args.running_sum, "running_sum"},
        {&tensor_args.shift, "shift"},
        {&tensor_args.mass, "mass"},
        {&tensor_args.q, "q"},
        {&tensor_args.stats, "stats"},
    };
    if (tensor_args.pending.has_value()) {
        operands.emplace_back(&tensor_args.pending.value(), "pending");
    }
    return operands;
}

// An operand's spec is part of the program cache key but its device is not — device
// storage contributes no attributes at all — while every dispatch patches the operand's
// raw address into the program. So an operand from another mesh reaches a cached program
// as a local address holding something else, which is why this belongs on the hit path as
// much as the miss path. The program itself runs on `partial`'s device: that is the
// tensor the dispatch takes the mesh from.
void validate_device_affinity(const AttnResGatherSoftmaxInputs& tensor_args) {
    for (const auto& [tensor, name] : named_operands(tensor_args)) {
        TT_FATAL(
            tensor->storage_type() == StorageType::DEVICE,
            "AttnResGatherSoftmax requires {} on device, got {}",
            name,
            tensor->storage_type());
        TT_FATAL(
            tensor->device() == tensor_args.partial.device(),
            "AttnResGatherSoftmax requires {} on the same device as partial",
            name);
    }
}

// `site` selects a plane of every batched operand, and the factory turns it into page
// offsets with no further checking, so a value past the batch reads whatever follows
// the buffer.
void validate_site(const AttnResGatherSoftmaxParams& args, const AttnResGatherSoftmaxInputs& tensor_args) {
    const std::array<std::pair<const ttnn::Tensor*, const char*>, 3> batched = {{
        {&tensor_args.partial, "partial"},
        {&tensor_args.shift, "shift"},
        {&tensor_args.mass, "mass"},
    }};
    for (const auto& [tensor, name] : batched) {
        const auto sites = tensor->padded_shape()[0];
        TT_FATAL(
            sites == 1 || args.site < sites,
            "AttnResGatherSoftmax site {} is past {}'s dim 0 of {}",
            args.site,
            name,
            sites);
    }
}

// The semaphore is excluded from the cache key and its address is patched per dispatch,
// so a cache hit can be carrying a semaphore the program was never built against. It
// reaches the exchange as a bare L1 address, waited on by the last core of the grid and
// written by every peer at that same coordinate; built without that core, or in DRAM, it
// resolves to an address nothing arrives at and the wait never returns — a hang rather
// than a wrong answer, and one that looks like a fabric fault.
void validate_semaphore(const AttnResGatherSoftmaxParams& args, const AttnResGatherSoftmaxInputs& tensor_args) {
    TT_FATAL(
        args.semaphore.device() == tensor_args.partial.device(),
        "AttnResGatherSoftmax requires the semaphore on the same device as its operands");
    // By value: attribute_values() returns a temporary tuple.
    const auto [semaphore_cores, semaphore_buffer_type] = args.semaphore.attribute_values();
    const auto grid = tensor_args.partial.device()->compute_with_storage_grid_size();
    const CoreCoord gather_core{grid.x - 1, grid.y - 1};
    TT_FATAL(
        semaphore_cores.contains(gather_core),
        "AttnResGatherSoftmax exchanges on core {} and needs the semaphore built over it, but the semaphore covers {}",
        gather_core,
        semaphore_cores);
    TT_FATAL(
        semaphore_buffer_type == tt::tt_metal::BufferType::L1,
        "AttnResGatherSoftmax waits on the semaphore in L1, got buffer type {}",
        semaphore_buffer_type);
}

}  // namespace

void AttnResGatherSoftmaxDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_device_affinity(tensor_args);
    validate_site(args, tensor_args);
    validate_semaphore(args, tensor_args);
}

void AttnResGatherSoftmaxDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& partial = tensor_args.partial;
    const auto& running_sum = tensor_args.running_sum;
    const auto& stats = tensor_args.stats;

    // The compute config defaults to HiFi4 with fp32 dest accumulation, which is only
    // correct on Blackhole; elsewhere the op compiles, runs, and returns silently wrong
    // values.
    const tt::ARCH arch = tt::tt_metal::hal::get_arch();
    TT_FATAL(arch == tt::ARCH::BLACKHOLE, "AttnResGatherSoftmax is only supported on Blackhole, got {}", arch);

    operations::check_tensor(partial, "AttnResGatherSoftmax", "partial", {DataType::BFLOAT16});
    operations::check_tensor(running_sum, "AttnResGatherSoftmax", "running_sum", {DataType::BFLOAT16});
    operations::check_tensor(tensor_args.q, "AttnResGatherSoftmax", "q", {DataType::BFLOAT16});
    if (tensor_args.pending.has_value()) {
        operations::check_tensor(*tensor_args.pending, "AttnResGatherSoftmax", "pending", {DataType::BFLOAT16});
    }

    // shift, mass and the gathered statistics reach the weight derivation through one
    // circular buffer, so they are read through a single unpack configuration.
    const std::array<std::pair<const ttnn::Tensor*, const char*>, 3> scalars = {{
        {&tensor_args.shift, "shift"},
        {&tensor_args.mass, "mass"},
        {&stats, "stats"},
    }};
    for (const auto& [tensor, name] : scalars) {
        operations::check_tensor(*tensor, "AttnResGatherSoftmax", name, {DataType::BFLOAT16, DataType::FLOAT32});
        TT_FATAL(
            tensor->dtype() == tensor_args.shift.dtype(),
            "AttnResGatherSoftmax requires one dtype across shift, mass and stats; {} is {} against shift's {}",
            name,
            tensor->dtype(),
            tensor_args.shift.dtype());
    }

    validate_device_affinity(tensor_args);
    for (const auto& [tensor, name] : named_operands(tensor_args)) {
        TT_FATAL(tensor->layout() == Layout::TILE, "AttnResGatherSoftmax requires TILE layout for {}", name);
        TT_FATAL(
            !tensor->memory_config().is_sharded(), "AttnResGatherSoftmax supports interleaved operands only, {}", name);
        TT_FATAL(
            tensor->padded_shape().rank() == 4,
            "AttnResGatherSoftmax requires rank-4 operands, {} has rank {}",
            name,
            tensor->padded_shape().rank());
    }
    TT_FATAL(!args.output_mem_config.is_sharded(), "AttnResGatherSoftmax supports an interleaved output only");

    const auto& partial_shape = partial.padded_shape();
    const auto& partial_logical = partial.logical_shape();
    const auto& running_sum_shape = running_sum.padded_shape();
    const auto& running_sum_logical = running_sum.logical_shape();
    // Logical as well as padded, here and in every cross-operand check below. The inner
    // dims are tile-padded, so two different row counts in one tile bucket — 100 and 120,
    // both padding to 128 — compare equal padded. The output is built from partial's
    // logical shape, so the shorter operand's padding would be folded into rows the op
    // still returns as data.
    TT_FATAL(
        running_sum_shape[0] == 1 && running_sum_shape[1] == partial_shape[1] &&
            running_sum_shape[2] == partial_shape[2] && running_sum_shape[3] == partial_shape[3] &&
            running_sum_logical[1] == partial_logical[1] && running_sum_logical[2] == partial_logical[2] &&
            running_sum_logical[3] == partial_logical[3],
        "AttnResGatherSoftmax requires an unbatched running_sum matching partial's plane, got {} ({} padded) against "
        "{} ({} padded)",
        running_sum_logical,
        running_sum_shape,
        partial_logical,
        partial_shape);
    if (tensor_args.pending.has_value()) {
        // The settled stream replaces `running_sum` everywhere the op uses it, so it has
        // to be the same tensor shape, and it shares the fold's unpacker configuration
        // with `partial` like `running_sum` does.
        TT_FATAL(
            tensor_args.pending->padded_shape() == running_sum_shape &&
                tensor_args.pending->logical_shape() == running_sum_logical,
            "AttnResGatherSoftmax pending is {} ({} padded) but settles into running_sum's {} ({} padded)",
            tensor_args.pending->logical_shape(),
            tensor_args.pending->padded_shape(),
            running_sum_logical,
            running_sum_shape);
    }
    TT_FATAL(partial_shape[1] == 1, "AttnResGatherSoftmax requires a candidate dim of 1, got {}", partial_shape[1]);
    validate_site(args, tensor_args);
    validate_semaphore(args, tensor_args);
    // The logical inner dims, not the padded ones. The statistics reduce runs the whole
    // padded row, so a logically narrower row folds its own padding into both the sum of
    // squares and the dot product that set the live score.
    TT_FATAL(
        partial_logical[-1] % TILE_WIDTH == 0 && partial_logical[-2] % TILE_HEIGHT == 0,
        "AttnResGatherSoftmax requires tile-aligned inner dims, got {} x {}",
        partial_logical[-2],
        partial_logical[-1]);

    // The logical height, not the padded one: the reader fetches pages 0..Wt-1 and stops,
    // so a query taller than the tile it pads into would be silently truncated to its
    // first row rather than rejected.
    const auto& q_shape = tensor_args.q.padded_shape();
    const auto& q_logical = tensor_args.q.logical_shape();
    TT_FATAL(
        q_shape[0] == 1 && q_shape[1] == 1 && q_logical[-2] == 1 && q_shape[-1] == partial_shape[-1] &&
            q_logical[-1] == partial_logical[-1],
        "AttnResGatherSoftmax takes the statistics against a single query row of partial's width, got {} ({} padded) "
        "against {} ({} padded)",
        q_logical,
        q_shape,
        partial_logical,
        partial_shape);

    for (const auto* tensor : {&tensor_args.shift, &tensor_args.mass}) {
        TT_FATAL(
            tensor->logical_shape()[-1] == 1,
            "AttnResGatherSoftmax shift and mass must carry one scalar per row, got a logical last dim of {}",
            tensor->logical_shape()[-1]);
        for (int i = 1; i < 3; ++i) {
            TT_FATAL(
                tensor->padded_shape()[i] == partial_shape[i] && tensor->logical_shape()[i] == partial_logical[i],
                "AttnResGatherSoftmax shift and mass dim {} is {} ({} padded) but partial's is {} ({} padded)",
                i,
                tensor->logical_shape()[i],
                tensor->padded_shape()[i],
                partial_logical[i],
                partial_shape[i]);
        }
    }

    // The exchange writes a peer's slot by naming a page of that peer's own copy of this
    // tensor, so the allocation has to be the same shape and address on every chip of
    // the axis — which is what a mesh-wide device tensor is — and it has to be one plane
    // pair per rank so a rank addresses its slot by rank alone.
    const auto& stats_shape = stats.padded_shape();
    TT_FATAL(
        stats_shape[0] == 1 && stats_shape[1] == 2 * args.ring_size && stats_shape[2] == partial_shape[2] &&
            stats.logical_shape()[2] == partial_logical[2] && stats.logical_shape()[-1] == 1,
        "AttnResGatherSoftmax at ring size {} needs stats shaped [1, {}, {}, 1], got {} ({} padded)",
        args.ring_size,
        2 * args.ring_size,
        partial_logical[2],
        stats.logical_shape(),
        stats_shape);

    TT_FATAL(
        args.inv_hidden_size > 0.0f,
        "AttnResGatherSoftmax derives the live score itself and needs a positive inv_hidden_size, got {}",
        args.inv_hidden_size);
    TT_FATAL(
        args.ring_size > 1,
        "AttnResGatherSoftmax at ring size {} has nothing to gather: the exchange is what the op is built "
        "around, and at one rank the statistics are already complete",
        args.ring_size);
    TT_FATAL(
        args.cluster_axis < 2, "AttnResGatherSoftmax takes a 2D mesh axis, got cluster_axis {}", args.cluster_axis);

    // The exchange is one core opening one connection, because the whole payload is a
    // handful of scalar tiles — a second link would have nothing to carry. Rejected
    // rather than ignored, since `num_links` does key the program cache.
    TT_FATAL(args.num_links == 1, "AttnResGatherSoftmax sends over a single link, got num_links {}", args.num_links);

    // The factory seats its two passes on the compute grid directly, so a subdevice that
    // does not cover it would put this program on cores the caller kept for something
    // else.
    auto* mesh_device = partial.device();
    const auto grid = mesh_device->compute_with_storage_grid_size();
    const auto sub_device_id = args.sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0));
    TT_FATAL(
        mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id)
            .contains(CoreRangeSet(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}))),
        "AttnResGatherSoftmax spans the whole compute grid and cannot be confined to a subdevice");
}

std::vector<tt::tt_metal::TensorSpec> AttnResGatherSoftmaxDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // From the logical shape, not the padded one, so that tile padding stays labelled as
    // padding rather than becoming readable data. Dim 0 is the read-site axis and `site`
    // picks one of them, so the output is a single plane however many the caller batched.
    auto output_shape = tensor_args.partial.logical_shape();
    output_shape[0] = 1;
    const tt::tt_metal::TensorSpec merged(
        output_shape,
        operations::TensorLayout(
            tensor_args.partial.dtype(), operations::PageConfig(Layout::TILE), args.output_mem_config));

    if (!tensor_args.pending.has_value()) {
        return {merged};
    }
    // The settled stream, which the fold reads back and the caller carries forward. It
    // takes running_sum's spec rather than the merged output's — the two happen to agree
    // here, but it is running_sum this replaces.
    return {
        merged,
        tt::tt_metal::TensorSpec(
            tensor_args.running_sum.logical_shape(),
            operations::TensorLayout(
                tensor_args.running_sum.dtype(), operations::PageConfig(Layout::TILE), args.output_mem_config))};
}

std::vector<Tensor> AttnResGatherSoftmaxDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    std::vector<Tensor> outputs;
    for (const auto& spec : compute_output_specs(args, tensor_args)) {
        outputs.push_back(create_device_tensor(spec, tensor_args.partial.device()));
    }
    return outputs;
}

ttsl::hash::hash_t AttnResGatherSoftmaxDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto* mesh_device = tensor_args.partial.device();
    const auto sd_id = args.sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0));
    const auto subdevice_core_range_set =
        mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    // Neither `site` nor the semaphore appears here: both are runtime-arg values this
    // op's factory rewrites on every cache hit, and a walk of 186 read sites has to
    // land on one cached program. Whether a write was handed in does shape the kernels,
    // and it is stated rather than left to how an absent operand hashes.
    return tt::tt_metal::operation::hash_operation<AttnResGatherSoftmaxDeviceOperation>(
        tensor_args.pending.has_value(),
        args.inv_hidden_size,
        args.eps,
        args.output_mem_config,
        args.compute_kernel_config,
        args.topology,
        args.num_links,
        args.ring_size,
        args.cluster_axis,
        subdevice_core_range_set,
        tensor_args);
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> attn_res_gather_softmax(
    const Tensor& partial,
    const Tensor& running_sum,
    const Tensor& shift,
    const Tensor& mass,
    const Tensor& q,
    const Tensor& stats,
    const std::optional<Tensor>& pending,
    uint32_t site,
    float inv_hidden_size,
    float eps,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const GlobalSemaphore& semaphore,
    ttnn::ccl::Topology topology,
    uint32_t num_links,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = ttnn::experimental::prim::AttnResGatherSoftmaxDeviceOperation;

    const auto& mesh_view = mesh_device.get_view();
    const uint32_t ring_size = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    auto operation_attributes = OperationType::operation_attributes_t(
        site,
        inv_hidden_size,
        eps,
        output_mem_config,
        compute_kernel_config,
        topology,
        num_links,
        ring_size,
        cluster_axis,
        semaphore,
        sub_device_id);

    auto tensor_args = OperationType::tensor_args_t{
        .partial = partial,
        .running_sum = running_sum,
        .shift = shift,
        .mass = mass,
        .q = q,
        .stats = stats,
        .pending = pending};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
