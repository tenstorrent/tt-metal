// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "zero_padded_kv_cache_device_operation.hpp"

#include <cstdint>
#include <utility>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::zero_padded_kv_cache {

using namespace tt::tt_metal;
using namespace tt::constants;

namespace {

// Three kernels on a single core per chip: the reader brings the boundary (partial) tile in from the
// cache and builds the row-mask tile in L1; the compute multiplies them; the writer writes the masked
// partial back and zeros the full pad tiles from the L1 zeros buffer. Each chip computes its share of
// the global pad window on-device from `my_sp_coord` + the per-call `valid_global` (patched scalar),
// so the window spilling across a chip boundary is handled by every chip doing its own slice.
constexpr auto kReaderKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/kernels/dataflow/"
    "reader_zero_padded_kv_cache.cpp";
constexpr auto kComputeKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/kernels/compute/"
    "zero_padded_kv_cache.cpp";
constexpr auto kWriterKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/kernels/dataflow/"
    "writer_zero_padded_kv_cache.cpp";

constexpr uint32_t kSrcCbIndex = 0;   // partial tile read from cache (cache dtype)
constexpr uint32_t kMaskCbIndex = 1;  // row-mask tile built in the reader (bf16)
constexpr uint32_t kOutCbIndex = 2;   // masked partial tile from compute (cache dtype)
constexpr uint32_t kZeroCbIndex = 3;  // pre-zeroed scratch for full-tile writes (cache dtype)
constexpr uint32_t kMetaCbIndex = 4;  // tensor path only: L1 scratch the reader/writer read each
                                      // 1-element uint32 metadata tensor page into (reused for both
                                      // the slot_idx and valid_global tensors -- identical layout)
constexpr uint32_t kMetadataBytes = 16;

// Common runtime arg layout. Index 3 is valid_global and index 9 is slot_idx -- the per-call scalars
// patched by override_runtime_arguments on the SCALAR path. On the TENSOR path index 10 is the
// slot_idx tensor's raw DRAM address and index 11 is the valid_global tensor's, both patched there (the
// reader/writer read element 0 of each).
constexpr uint32_t kValidGlobalCommonArgIdx = 3;
constexpr uint32_t kSlotIdxCommonArgIdx = 9;
constexpr uint32_t kSlotIdxAddrCommonArgIdx = 10;
constexpr uint32_t kValidGlobalAddrCommonArgIdx = 11;

// Per-call scalar checks shared by the cache-miss and cache-hit paths.
void validate_runtime_args(
    const ZeroPaddedKvCacheDeviceOperation::operation_attributes_t& args,
    const ZeroPaddedKvCacheDeviceOperation::tensor_args_t& tensor_args) {
    TT_FATAL(args.cluster_axis == 0 || args.cluster_axis == 1, "cluster_axis ({}) must be 0 or 1", args.cluster_axis);
    const auto& cache = tensor_args.cache;
    // slot_idx is a host value only on the scalar path; on the tensor path it lives in the device
    // tensor (read on-device) and is the caller's responsibility.
    if (!tensor_args.slot_idx.has_value()) {
        const uint32_t num_slots = cache.padded_shape()[0] / args.num_layers;
        TT_FATAL(args.slot_idx < num_slots, "slot_idx ({}) out of range for num_slots ({})", args.slot_idx, num_slots);
    }

    const auto& mesh_view = cache.device()->get_view();
    TT_FATAL(mesh_view.is_mesh_2d(), "zero_padded_kv_cache requires a 2D mesh");
    const uint32_t sp_factor = (args.cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    TT_FATAL(
        args.chunk_size_global % sp_factor == 0,
        "chunk_size_global ({}) must be a multiple of sp_factor ({})",
        args.chunk_size_global,
        sp_factor);
    const uint32_t chunk_local = args.chunk_size_global / sp_factor;
    TT_FATAL(args.pad_align > 0, "pad_align must be positive");
    // The pad window MAY cross chip boundaries (each chip zeroes its own contiguous-local slice), but
    // it must not cross a slab boundary -- so it stays within one block-cyclic cycle. Holds when
    // pad_align divides chunk_size_global. And each chip boundary must be tile-aligned.
    TT_FATAL(
        args.chunk_size_global % args.pad_align == 0,
        "chunk_size_global ({}) must be a multiple of pad_align ({}) so the window stays within one slab",
        args.chunk_size_global,
        args.pad_align);
    TT_FATAL(
        chunk_local % TILE_HEIGHT == 0,
        "chunk_local ({}) must be tile-aligned (multiple of {})",
        chunk_local,
        TILE_HEIGHT);
    const uint32_t global_capacity = sp_factor * cache.padded_shape()[-2];
    // A pad_align-aligned capacity guarantees ceil_pad_align(valid_global) <= capacity whenever
    // valid_global <= capacity, so the window never rounds past the cache (well-formed block-cyclic
    // caches hold whole slabs, so capacity is a multiple of chunk_size_global and thus of pad_align).
    TT_FATAL(
        global_capacity % args.pad_align == 0,
        "global cache capacity ({}) must be a multiple of pad_align ({})",
        global_capacity,
        args.pad_align);
    // valid_global is a host value only on the scalar path.
    if (!tensor_args.valid_global.has_value()) {
        TT_FATAL(
            args.valid_global <= global_capacity,
            "valid_global ({}) exceeds cache capacity ({})",
            args.valid_global,
            global_capacity);
    }
}

}  // namespace

ZeroPaddedKvCacheDeviceOperation::program_factory_t ZeroPaddedKvCacheDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return MeshWorkloadFactory{};
}

void ZeroPaddedKvCacheDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& cache = tensor_args.cache;
    TT_FATAL(cache.storage_type() == StorageType::DEVICE, "cache must be on device");
    TT_FATAL(cache.layout() == Layout::TILE, "cache must be TILE layout");
    const auto& cache_shape = cache.padded_shape();
    TT_FATAL(cache_shape.rank() == 4, "cache must be 4D (got rank {})", cache_shape.rank());
    TT_FATAL(cache_shape[1] == 1, "cache num-heads dim must be 1 (got {})", cache_shape[1]);
    TT_FATAL(args.num_layers > 0, "num_layers must be positive");
    TT_FATAL(
        cache_shape[0] % args.num_layers == 0,
        "cache batch dim ({}) must be a multiple of num_layers ({})",
        cache_shape[0],
        args.num_layers);
    TT_FATAL(
        args.layer_idx < args.num_layers,
        "layer_idx {} out of range for num_layers {}",
        args.layer_idx,
        args.num_layers);
    validate_runtime_args(args, tensor_args);
}

void ZeroPaddedKvCacheDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_runtime_args(args, tensor_args);
}

ZeroPaddedKvCacheDeviceOperation::spec_return_value_t ZeroPaddedKvCacheDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    return tensor_args.cache.tensor_spec();  // in-place
}

ZeroPaddedKvCacheDeviceOperation::tensor_return_value_t ZeroPaddedKvCacheDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    return tensor_args.cache;  // in-place
}

ttsl::hash::hash_t ZeroPaddedKvCacheDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // slot_idx and valid_global are per-call scalars (patched, NOT hashed). layer_idx, num_layers,
    // cluster_axis, chunk_size_global and pad_align are structural (hashed). Hash the full cache shape.
    const auto& cache = tensor_args.cache;
    // The tensor-vs-scalar choice changes the reader/writer programs (compile args + which kernel
    // branch compiles), so hash slot_idx.has_value() to keep the two variants distinct; slot_idx and
    // valid_global themselves are never hashed on either path.
    return tt::tt_metal::operation::hash_operation<ZeroPaddedKvCacheDeviceOperation>(
        tensor_args.slot_idx.has_value(),
        args.layer_idx,
        args.num_layers,
        args.cluster_axis,
        args.chunk_size_global,
        args.pad_align,
        cache.dtype(),
        cache.memory_config(),
        cache.padded_shape());
}

tt::tt_metal::ProgramDescriptor ZeroPaddedKvCacheDeviceOperation::ProgramFactory::create_descriptor(
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*output*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(mesh_dispatch_coordinate.has_value(), "ZeroPaddedKvCache::create_descriptor needs a mesh coordinate");
    const auto& coord = mesh_dispatch_coordinate.value();

    const auto& cache = tensor_args.cache;
    auto* device = cache.device();
    const auto& cache_shape = cache.padded_shape();
    const bool has_metadata = tensor_args.slot_idx.has_value();
    const uint32_t slot_idx_addr = has_metadata ? static_cast<uint32_t>(tensor_args.slot_idx->buffer()->address()) : 0u;
    const uint32_t valid_global_addr =
        has_metadata ? static_cast<uint32_t>(tensor_args.valid_global->buffer()->address()) : 0u;

    const tt::DataFormat cache_format = datatype_to_dataformat_converter(cache.dtype());
    const uint32_t cache_tile_size = tt::tile_size(cache_format);
    const tt::DataFormat mask_format = tt::DataFormat::Float16_b;
    const uint32_t mask_tile_size = tt::tile_size(mask_format);

    const uint32_t Wt = cache_shape[-1] / TILE_WIDTH;
    const uint32_t cache_HtWt = cache_shape[-2] * Wt / TILE_HEIGHT;
    const uint32_t cache_CHtWt = cache_shape[1] * cache_HtWt;

    const auto& mesh_view = device->get_view();
    const uint32_t sp_factor = (args.cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    const uint32_t my_sp_coord = ::ttnn::ccl::get_linearized_index_from_physical_coord(cache, coord, args.cluster_axis);
    const uint32_t chunk_local = args.chunk_size_global / sp_factor;  // tokens

    // Single core per chip: the pad window is at most pad_align tokens (a few tiles).
    CoreRangeSet all_cores(CoreRange({0, 0}, {0, 0}));

    tt::tt_metal::ProgramDescriptor desc;

    // CBs: src (partial tile read), mask (bf16 row-mask), out (masked partial), zero (write scratch).
    auto add_cb = [&](uint32_t index, tt::DataFormat fmt, uint32_t page, uint32_t npages) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = npages * page,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{.buffer_index = index, .data_format = fmt, .page_size = page}}},
        });
    };
    add_cb(kSrcCbIndex, cache_format, cache_tile_size, Wt);
    add_cb(kMaskCbIndex, mask_format, mask_tile_size, 1);
    add_cb(kOutCbIndex, cache_format, cache_tile_size, Wt);
    add_cb(kZeroCbIndex, cache_format, cache_tile_size, 1);
    if (has_metadata) {
        // reader/writer metadata read: one CB page reused for both 1-element uint32 tensor reads
        add_cb(kMetaCbIndex, tt::DataFormat::UInt32, kMetadataBytes, 1);
    }

    // Common runtime args, shared layout across all three kernels (each recomputes its share of the
    // window from these). Index 3 = valid_global, 9 = slot_idx are the per-call scalars patched on
    // cache hits by override_runtime_arguments.
    // Layout: 0 my_sp_coord, 1 sp_factor, 2 chunk_local(tokens), 3 valid_global, 4 pad_align,
    //         5 layer_idx, 6 num_layers, 7 Wt, 8 cache_CHtWt, 9 slot_idx, 10 slot_idx_addr,
    //         11 valid_global_addr.
    // Index 3/9 are the scalar-path per-call values; indices 10/11 are the slot_idx / valid_global
    // tensors' raw DRAM addresses (tensor path). override_runtime_arguments patches whichever applies.
#define ZP_COMMON_ARGS  \
    {my_sp_coord,       \
     sp_factor,         \
     chunk_local,       \
     args.valid_global, \
     args.pad_align,    \
     args.layer_idx,    \
     args.num_layers,   \
     Wt,                \
     cache_CHtWt,       \
     args.slot_idx,     \
     slot_idx_addr,     \
     valid_global_addr}

    // Reader: reads cache (TensorAccessor) + builds mask.
    KernelDescriptor reader;
    reader.kernel_source = kReaderKernelPath;
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    // [3]=has_metadata, [4]=metadata CB index (placeholder 0 on the scalar path). Cache accessor at
    // <5>; the metadata accessor (when present) is appended after it so the cache-accessor offset is
    // fixed at <5> for both paths.
    reader.compile_time_args = {
        kSrcCbIndex,
        kMaskCbIndex,
        cache_tile_size,
        static_cast<uint32_t>(has_metadata),
        has_metadata ? kMetaCbIndex : 0u};
    TensorAccessorArgs(cache.buffer()).append_to(reader.compile_time_args);
    if (has_metadata) {
        // One metadata accessor, reused for both 1-element tensors (identical layout); the kernel reads
        // each from its own DRAM address (common args 10/11).
        TensorAccessorArgs(tensor_args.slot_idx->buffer()).append_to(reader.compile_time_args);
    }
    reader.config = ReaderConfigDescriptor{};
    reader.emplace_common_runtime_args(ZP_COMMON_ARGS);
    reader.emplace_runtime_args(CoreCoord{0, 0}, {cache.buffer()});

    // Compute: partial x mask -> out.
    KernelDescriptor compute;
    compute.kernel_source = kComputeKernelPath;
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.compile_time_args = {kSrcCbIndex, kMaskCbIndex, kOutCbIndex};
    compute.config = ComputeConfigDescriptor{};
    compute.emplace_common_runtime_args(ZP_COMMON_ARGS);
    compute.emplace_runtime_args(CoreCoord{0, 0}, {0u});  // compute reads only common args; dummy per-core arg

    // Writer: masked partial back + zero full tiles from the zero scratch.
    KernelDescriptor writer;
    writer.kernel_source = kWriterKernelPath;
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    // Same leading layout as the reader: [3]=has_metadata, [4]=metadata CB index, cache accessor at
    // <5>, then the metadata accessor when present.
    writer.compile_time_args = {
        kOutCbIndex,
        kZeroCbIndex,
        cache_tile_size,
        static_cast<uint32_t>(has_metadata),
        has_metadata ? kMetaCbIndex : 0u};
    TensorAccessorArgs(cache.buffer()).append_to(writer.compile_time_args);
    if (has_metadata) {
        // One metadata accessor, reused for both 1-element tensors (identical layout); the kernel reads
        // each from its own DRAM address (common args 10/11).
        TensorAccessorArgs(tensor_args.slot_idx->buffer()).append_to(writer.compile_time_args);
    }
    writer.config = WriterConfigDescriptor{};
    writer.emplace_common_runtime_args(ZP_COMMON_ARGS);
    writer.emplace_runtime_args(CoreCoord{0, 0}, {cache.buffer()});
#undef ZP_COMMON_ARGS

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(compute));
    desc.kernels.push_back(std::move(writer));
    return desc;
}

ZeroPaddedKvCacheDeviceOperation::MeshWorkloadFactory::cached_mesh_workload_t
ZeroPaddedKvCacheDeviceOperation::MeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    return descriptor_adapter_t::create_mesh_workload(args, tensor_coords, tensor_args, output);
}

void ZeroPaddedKvCacheDeviceOperation::MeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    descriptor_adapter_t::apply_descriptor(cached_workload, args, tensor_args, output);
    // Patch the per-call common args the buffer-binding fast path would otherwise leave stale.
    //   - tensor path: the reader (0) and writer (2) read slot_idx/valid_global on-device from the two
    //     1-element tensors, so only their raw DRAM addresses (indices 10/11) need patching there. The
    //     compute (1) reads only structural args, so it needs no patch.
    //   - scalar path: all three kernels read slot_idx (9) / valid_global (3) from common args.
    if (tensor_args.slot_idx.has_value()) {
        const uint32_t slot_idx_addr = static_cast<uint32_t>(tensor_args.slot_idx->buffer()->address());
        const uint32_t valid_global_addr = static_cast<uint32_t>(tensor_args.valid_global->buffer()->address());
        for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
            for (uint32_t kernel_handle : {0u, 2u}) {  // reader, writer
                auto& common = GetCommonRuntimeArgs(program, kernel_handle);
                TT_FATAL(
                    kValidGlobalAddrCommonArgIdx < common.size(),
                    "zero_padded_kv_cache kernel missing the metadata-tensor addr common args");
                common[kSlotIdxAddrCommonArgIdx] = slot_idx_addr;
                common[kValidGlobalAddrCommonArgIdx] = valid_global_addr;
            }
        }
    } else {
        for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
            for (uint32_t kernel_handle : {0u, 1u, 2u}) {
                auto& common = GetCommonRuntimeArgs(program, kernel_handle);
                TT_FATAL(
                    kSlotIdxCommonArgIdx < common.size(), "zero_padded_kv_cache kernel missing per-call common args");
                common[kValidGlobalCommonArgIdx] = args.valid_global;
                common[kSlotIdxCommonArgIdx] = args.slot_idx;
            }
        }
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::zero_padded_kv_cache

namespace ttnn::prim {

ttnn::Tensor zero_padded_kv_cache(
    const ttnn::Tensor& cache,
    const std::optional<ttnn::Tensor>& slot_idx_tensor,
    const std::optional<ttnn::Tensor>& valid_global_tensor,
    uint32_t slot_idx,
    uint32_t layer_idx,
    uint32_t num_layers,
    uint32_t valid_global,
    uint32_t chunk_size_global,
    uint32_t cluster_axis,
    uint32_t pad_align) {
    using OperationType =
        ttnn::operations::experimental::deepseek_prefill::zero_padded_kv_cache::ZeroPaddedKvCacheDeviceOperation;
    auto attrs = OperationType::operation_attributes_t{
        .slot_idx = slot_idx,
        .valid_global = valid_global,
        .chunk_size_global = chunk_size_global,
        .pad_align = pad_align,
        .layer_idx = layer_idx,
        .num_layers = num_layers,
        .cluster_axis = cluster_axis,
    };
    auto tensor_args =
        OperationType::tensor_args_t{.cache = cache, .slot_idx = slot_idx_tensor, .valid_global = valid_global_tensor};
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
