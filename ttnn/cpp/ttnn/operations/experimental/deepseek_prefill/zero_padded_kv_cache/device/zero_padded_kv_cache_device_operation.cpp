// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "zero_padded_kv_cache_device_operation.hpp"

#include <cstdint>
#include <utility>
#include <vector>

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
constexpr auto kRowMajorWriterKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/kernels/dataflow/"
    "writer_zero_padded_kv_cache_row_major.cpp";

constexpr uint32_t kSrcCbIndex = 0;   // partial tile read from cache (cache dtype)
constexpr uint32_t kMaskCbIndex = 1;  // row-mask tile built in the reader (bf16)
constexpr uint32_t kOutCbIndex = 2;   // masked partial tile from compute (cache dtype)
constexpr uint32_t kZeroCbIndex = 3;  // pre-zeroed scratch for full-tile writes (cache dtype)

// Common runtime arg layout shared by the reader and writer (the compute reads a subset). Index 3 is
// valid_global and index 9 is slot_idx -- the per-call scalars patched by override_runtime_arguments.
constexpr uint32_t kValidGlobalCommonArgIdx = 3;
constexpr uint32_t kSlotIdxCommonArgIdx = 9;

// Per-call scalar checks shared by the cache-miss and cache-hit paths.
void validate_runtime_args(
    const ZeroPaddedKvCacheDeviceOperation::operation_attributes_t& args,
    const ZeroPaddedKvCacheDeviceOperation::tensor_args_t& tensor_args) {
    TT_FATAL(args.cluster_axis == 0 || args.cluster_axis == 1, "cluster_axis ({}) must be 0 or 1", args.cluster_axis);
    const auto& cache = tensor_args.cache;
    const uint32_t num_slots = cache.padded_shape()[0] / args.num_layers;
    TT_FATAL(args.slot_idx < num_slots, "slot_idx ({}) out of range for num_slots ({})", args.slot_idx, num_slots);

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
    TT_FATAL(
        args.valid_global <= global_capacity,
        "valid_global ({}) exceeds cache capacity ({})",
        args.valid_global,
        global_capacity);
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
    TT_FATAL(cache.buffer()->buffer_type() == BufferType::DRAM, "zero_padded_kv_cache requires a DRAM-backed cache");
    TT_FATAL(
        cache.layout() == Layout::TILE || cache.layout() == Layout::ROW_MAJOR,
        "cache layout must be TILE or ROW_MAJOR");
    if (cache.layout() == Layout::ROW_MAJOR) {
        TT_FATAL(
            cache.dtype() == DataType::BFLOAT16 || cache.dtype() == DataType::FP8_E4M3,
            "ROW_MAJOR zero_padded_kv_cache supports bfloat16 or fp8_e4m3 (got {})",
            cache.dtype());
    }
    if (cache.dtype() == DataType::FP8_E4M3) {
        TT_FATAL(cache.layout() == Layout::ROW_MAJOR, "fp8_e4m3 cache must be ROW_MAJOR");
    }
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

    const tt::DataFormat cache_format = datatype_to_dataformat_converter(cache.dtype());
    const bool is_row_major = cache.layout() == Layout::ROW_MAJOR;
    const uint32_t Wt = is_row_major ? 1 : cache_shape[-1] / TILE_WIDTH;
    const uint32_t cache_H_pages = is_row_major ? cache_shape[-2] : cache_shape[-2] * Wt / TILE_HEIGHT;
    const uint32_t cache_CH_pages = cache_shape[1] * cache_H_pages;

    const auto& mesh_view = device->get_view();
    const uint32_t sp_factor = (args.cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    const uint32_t my_sp_coord = ::ttnn::ccl::get_linearized_index_from_physical_coord(cache, coord, args.cluster_axis);
    const uint32_t chunk_local = args.chunk_size_global / sp_factor;  // tokens

    // Single core per chip: the pad window is at most pad_align tokens (a few tiles).
    CoreRangeSet all_cores(CoreRange({0, 0}, {0, 0}));

    tt::tt_metal::ProgramDescriptor desc;

    // Keep one common-argument layout for both descriptors. Page units are native to the layout:
    // width-tiles for TILE, one complete token row for ROW_MAJOR.
    const std::vector<uint32_t> common_runtime_args = {
        my_sp_coord,
        sp_factor,
        chunk_local,
        args.valid_global,
        args.pad_align,
        args.layer_idx,
        args.num_layers,
        Wt,
        cache_CH_pages,
        args.slot_idx,
    };

    if (is_row_major) {
        // A row is one opaque DRAM page. Use a dataflow-only zero writer so FP8 never enters the
        // unpack/compute engine (and therefore needs no fp32 destination accumulator setting).
        const uint32_t row_page_size = cache.buffer()->aligned_page_size();
        desc.cbs.push_back(CBDescriptor{
            .total_size = row_page_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = kZeroCbIndex,
                .data_format = cache_format,
                .page_size = row_page_size,
            }}},
        });

        KernelDescriptor writer;
        writer.kernel_source = kRowMajorWriterKernelPath;
        writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer.core_ranges = all_cores;
        writer.compile_time_args = {kZeroCbIndex, row_page_size};
        TensorAccessorArgs(cache.buffer()).append_to(writer.compile_time_args);
        writer.config = WriterConfigDescriptor{};
        writer.common_runtime_args = common_runtime_args;
        writer.emplace_runtime_args(CoreCoord{0, 0}, {cache.buffer()});
        desc.kernels.push_back(std::move(writer));
        return desc;
    }

    const uint32_t cache_tile_size = tt::tile_size(cache_format);
    const tt::DataFormat mask_format = tt::DataFormat::Float16_b;
    const uint32_t mask_tile_size = tt::tile_size(mask_format);

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

    // Reader: reads cache (TensorAccessor) + builds mask.
    KernelDescriptor reader;
    reader.kernel_source = kReaderKernelPath;
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = {kSrcCbIndex, kMaskCbIndex, cache_tile_size};
    TensorAccessorArgs(cache.buffer()).append_to(reader.compile_time_args);
    reader.config = ReaderConfigDescriptor{};
    reader.common_runtime_args = common_runtime_args;
    reader.emplace_runtime_args(CoreCoord{0, 0}, {cache.buffer()});

    // Compute: partial x mask -> out.
    KernelDescriptor compute;
    compute.kernel_source = kComputeKernelPath;
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.compile_time_args = {kSrcCbIndex, kMaskCbIndex, kOutCbIndex};
    compute.config = ComputeConfigDescriptor{};
    compute.common_runtime_args = common_runtime_args;
    compute.emplace_runtime_args(CoreCoord{0, 0}, {0u});  // compute reads only common args; dummy per-core arg

    // Writer: masked partial back + zero full tiles from the zero scratch.
    KernelDescriptor writer;
    writer.kernel_source = kWriterKernelPath;
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = {kOutCbIndex, kZeroCbIndex, cache_tile_size};
    TensorAccessorArgs(cache.buffer()).append_to(writer.compile_time_args);
    writer.config = WriterConfigDescriptor{};
    writer.common_runtime_args = common_runtime_args;
    writer.emplace_runtime_args(CoreCoord{0, 0}, {cache.buffer()});
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
    // TILE has reader/compute/writer; ROW_MAJOR is a dataflow-only writer. Patch every kernel in the
    // selected descriptor without assuming one fixed kernel count.
    const uint32_t num_kernels = tensor_args.cache.layout() == Layout::ROW_MAJOR ? 1u : 3u;
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        for (uint32_t kernel_handle = 0; kernel_handle < num_kernels; ++kernel_handle) {
            auto& common = GetCommonRuntimeArgs(program, kernel_handle);
            TT_FATAL(kSlotIdxCommonArgIdx < common.size(), "zero_padded_kv_cache kernel missing per-call common args");
            common[kValidGlobalCommonArgIdx] = args.valid_global;
            common[kSlotIdxCommonArgIdx] = args.slot_idx;
        }
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::zero_padded_kv_cache

namespace ttnn::prim {

ttnn::Tensor zero_padded_kv_cache(
    const ttnn::Tensor& cache,
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
        .cache_dtype = cache.dtype(),
        .cache_layout = cache.layout(),
        .cache_memory_config = cache.memory_config(),
        .cache_padded_shape = cache.padded_shape(),
    };
    auto tensor_args = OperationType::tensor_args_t{.cache = cache};
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
