// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_padding_config_device_operation.hpp"

#include <cstdint>
#include <utility>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_padding_config {

using namespace tt::tt_metal;
using namespace tt::constants;

namespace {

constexpr auto kWriterKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config/device/kernels/dataflow/"
    "writer_moe_padding_config.cpp";

constexpr uint32_t kOutCbIndex = 0;
constexpr uint32_t kMetaCbIndex = 1;
// L1-scratch slot the kernel reads each 1-element metadata tensor into (element [0], 4 bytes).
// Rounded up to a 16B alignment-friendly slot.
constexpr uint32_t kMetadataBytes = 16;

// Index of the first per-call common runtime arg (the two metadata tensor addresses).
constexpr uint32_t kArgActualStartAddr = 3;
constexpr uint32_t kArgActualEndAddr = 4;

// Checks re-run on BOTH the cache-miss and cache-hit paths: none of these are hashed, so a caller
// could otherwise change them behind a cached program. The per-chunk VALUES live in device tensors
// and cannot be read host-side without a read-back, so they stay the caller's responsibility (the
// kernel's arithmetic is total over uint32 regardless — an out-of-range start just yields 0).
void validate_runtime_args(
    const MoePaddingConfigDeviceOperation::operation_attributes_t& args,
    const MoePaddingConfigDeviceOperation::tensor_args_t& tensor_args) {
    TT_FATAL(args.cluster_axis == 0 || args.cluster_axis == 1, "cluster_axis ({}) must be 0 or 1", args.cluster_axis);
    TT_FATAL(args.tokens_per_chip > 0, "tokens_per_chip must be positive");
    TT_FATAL(args.pad_side == 0 || args.pad_side == 1, "pad_side ({}) must be 0 (right) or 1 (left)", args.pad_side);

    const auto& config = tensor_args.config;
    const auto& mesh_view = config.device()->get_view();
    TT_FATAL(mesh_view.is_mesh_2d(), "moe_padding_config requires a 2D mesh");

    // Output: what the consumers (moe_grouped_topk / dispatch) require of padding_config, asserted
    // here so a malformed buffer fails with a clear message instead of a bad on-device read.
    TT_FATAL(config.storage_type() == StorageType::DEVICE, "config must be on device");
    TT_FATAL(config.buffer() != nullptr, "config must be allocated");
    TT_FATAL(config.dtype() == DataType::UINT32, "config must be UINT32");
    TT_FATAL(config.layout() == Layout::ROW_MAJOR, "config must be ROW_MAJOR");
    TT_FATAL(
        config.logical_shape()[-1] >= 2,
        "config last dim ({}) must hold at least [local_real_tokens, pad_side]",
        config.logical_shape()[-1]);
    TT_FATAL(!config.is_sharded(), "config must not be L1-sharded (it is a per-device DRAM row)");

    // Structural properties the kernel assumes of each metadata tensor.
    auto validate_meta = [&config](const Tensor& meta, const char* name) {
        TT_FATAL(meta.storage_type() == StorageType::DEVICE, "metadata tensor {} must be on device", name);
        TT_FATAL(meta.buffer() != nullptr, "metadata tensor {} must be allocated", name);
        TT_FATAL(meta.dtype() == DataType::UINT32, "metadata tensor {} must be UINT32", name);
        TT_FATAL(meta.layout() == Layout::ROW_MAJOR, "metadata tensor {} must be ROW_MAJOR", name);
        TT_FATAL(
            meta.logical_volume() == 1,
            "metadata tensor {} must be a single element (got {})",
            name,
            meta.logical_volume());
        TT_FATAL(!meta.is_sharded(), "metadata tensor {} must not be sharded", name);
        // The kernel resolves meta.buffer()->address() against config.device(); a tensor on a
        // different mesh device would bake the wrong address and fail obscurely on device.
        TT_FATAL(meta.device() == config.device(), "metadata tensor {} must be on the same device as config", name);
    };
    validate_meta(tensor_args.actual_start, "actual_start");
    validate_meta(tensor_args.actual_end, "actual_end");
}

}  // namespace

MoePaddingConfigDeviceOperation::program_factory_t MoePaddingConfigDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return MeshWorkloadFactory{};
}

void MoePaddingConfigDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_runtime_args(args, tensor_args);
}

void MoePaddingConfigDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_runtime_args(args, tensor_args);
}

MoePaddingConfigDeviceOperation::spec_return_value_t MoePaddingConfigDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    // In-place: output spec = config spec.
    return tensor_args.config.tensor_spec();
}

MoePaddingConfigDeviceOperation::tensor_return_value_t MoePaddingConfigDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    // In-place: return a handle to the caller-owned config tensor.
    return tensor_args.config;
}

ttsl::hash::hash_t MoePaddingConfigDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // The per-chunk values are NEVER hashed: they are read on-device from the metadata tensors, whose
    // raw addresses live in common runtime args refreshed by override_runtime_arguments. That is the
    // whole point — one cached program serves every chunk, so it can be captured once and replayed.
    const auto& config = tensor_args.config;
    return tt::tt_metal::operation::hash_operation<MoePaddingConfigDeviceOperation>(
        args.tokens_per_chip,
        args.pad_side,
        args.cluster_axis,
        config.dtype(),
        config.layout(),
        config.memory_config(),
        config.padded_shape());
}

tt::tt_metal::ProgramDescriptor MoePaddingConfigDeviceOperation::ProgramFactory::create_descriptor(
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*output*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(
        mesh_dispatch_coordinate.has_value(), "MoePaddingConfig::create_descriptor requires a mesh dispatch coordinate");
    const auto& coord = mesh_dispatch_coordinate.value();

    const auto& config = tensor_args.config;
    auto* device = config.device();

    // This chip's position along the SP axis — the only thing that distinguishes one chip's config row
    // from another's. Same derivation as update_padded_kv_cache's writer, so the two cannot drift.
    const auto& mesh_view = device->get_view();
    const uint32_t sp_factor = (args.cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    const uint32_t my_sp_coord =
        ::ttnn::ccl::get_linearized_index_from_physical_coord(config, coord, args.cluster_axis);

    // One row of output on one core: there is nothing to parallelize.
    const CoreCoord core{0, 0};
    const CoreRangeSet single_core{CoreRange{core, core}};

    // Write the config row a full page at a time (the row may be padded up to the buffer's aligned
    // page size); the kernel zeroes the slot first so the padding bytes are deterministic.
    const uint32_t out_page_size = config.buffer()->aligned_page_size();

    tt::tt_metal::ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = out_page_size,
        .core_ranges = single_core,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kOutCbIndex,
            .data_format = tt::DataFormat::UInt32,
            .page_size = out_page_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = kMetadataBytes,
        .core_ranges = single_core,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kMetaCbIndex,
            .data_format = tt::DataFormat::UInt32,
            .page_size = kMetadataBytes,
        }}},
    });

    // Compile args: [0]=cb_out, [1]=cb_meta, [2]=pad_side, [3..]=config accessor, then ONE metadata
    // accessor (both 1-element tensors share an identical layout, so one accessor serves both reads).
    KernelDescriptor::CompileTimeArgs writer_compile_args = {kOutCbIndex, kMetaCbIndex, args.pad_side};
    TensorAccessorArgs(config.buffer()).append_to(writer_compile_args);
    TensorAccessorArgs(tensor_args.actual_start.buffer()).append_to(writer_compile_args);

    KernelDescriptor writer_kernel;
    writer_kernel.kernel_source = kWriterKernelPath;
    writer_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel.core_ranges = single_core;
    writer_kernel.compile_time_args = std::move(writer_compile_args);
    writer_kernel.config = WriterConfigDescriptor{};

    // Common rt-args: [0..2] structural (this chip's mesh position + the rotation period), [3..4] the
    // per-call metadata tensor addresses that override_runtime_arguments refreshes on cache hits.
    writer_kernel.emplace_common_runtime_args({
        my_sp_coord,
        sp_factor,
        args.tokens_per_chip,
        tensor_args.actual_start.buffer()->address(),  // smuggled-rta-ok: 1-element metadata tensor DRAM
                                                       // addr; read on-device (trace-safe, unhashed)
        tensor_args.actual_end.buffer()->address(),    // smuggled-rta-ok: as above
    });

    // The config buffer is passed as a Buffer* binding (not a raw address) so cache hits take the fast
    // path that patches its address and skips create_descriptor.
    writer_kernel.emplace_runtime_args(core, {config.buffer()});

    desc.kernels.push_back(std::move(writer_kernel));
    return desc;
}

MoePaddingConfigDeviceOperation::MeshWorkloadFactory::cached_mesh_workload_t
MoePaddingConfigDeviceOperation::MeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    return descriptor_adapter_t::create_mesh_workload(args, tensor_coords, tensor_args, output);
}

void MoePaddingConfigDeviceOperation::MeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    // Default adapter behaviour: patch operand buffer-binding addresses on cache hits.
    descriptor_adapter_t::apply_descriptor(cached_workload, args, tensor_args, output);
    // The metadata addresses are raw scalars in common runtime args, which the buffer-binding fast
    // path does not refresh — patch them on every cached program or the kernel would keep reading a
    // stale (possibly freed) address.
    constexpr uint32_t kWriterKernelHandle = 0;  // the only kernel pushed in create_descriptor
    const uint32_t start_addr = tensor_args.actual_start.buffer()->address();
    const uint32_t end_addr = tensor_args.actual_end.buffer()->address();
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& writer_common = GetCommonRuntimeArgs(program, kWriterKernelHandle);
        TT_FATAL(
            kArgActualEndAddr < writer_common.size(),
            "moe_padding_config writer is missing its per-call common runtime args");
        writer_common[kArgActualStartAddr] = start_addr;
        writer_common[kArgActualEndAddr] = end_addr;
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_padding_config

namespace ttnn::prim {

ttnn::Tensor moe_padding_config(
    const ttnn::Tensor& config,
    const ttnn::Tensor& actual_start,
    const ttnn::Tensor& actual_end,
    uint32_t tokens_per_chip,
    uint32_t pad_side,
    uint32_t cluster_axis) {
    using OperationType =
        ttnn::operations::experimental::deepseek_prefill::moe_padding_config::MoePaddingConfigDeviceOperation;
    auto attrs = OperationType::operation_attributes_t{
        .tokens_per_chip = tokens_per_chip,
        .pad_side = pad_side,
        .cluster_axis = cluster_axis,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .config = config,
        .actual_start = actual_start,
        .actual_end = actual_end,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
