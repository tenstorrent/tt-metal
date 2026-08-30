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
// Tensor path only: L1 scratch each kernel reads the 1-element uint32 metadata tensor page into (reused
// within a kernel for its two sequential slot_idx/valid_global reads -- identical layout). The reader
// (BRISC) and writer (NCRISC) run on the SAME core, so they MUST use SEPARATE CBs: a shared CB is one L1
// page both kernels NoC-read into concurrently, and a skew across the slot->valid read transition could
// let one kernel observe the other's value (wrong pad window -> silent cache corruption).
constexpr uint32_t kMetaCbIndex = 4;        // reader's metadata scratch
constexpr uint32_t kMetaCbIndexWriter = 5;  // writer's metadata scratch (disjoint from the reader's)
constexpr uint32_t kPageTableCbIndex = 6;
constexpr uint32_t kPageTableCbIndexWriter = 7;
constexpr uint32_t kMetadataBytes = 16;

// Common runtime arg layout. Index 3 is valid_global and index 9 is slot_idx -- the per-call scalars
// patched by override_runtime_arguments on the SCALAR path. On the TENSOR path index 10 is the
// slot_idx tensor's raw DRAM address and index 11 is the valid_global tensor's, both patched there (the
// reader/writer read element 0 of each).
constexpr uint32_t kValidGlobalCommonArgIdx = 3;
constexpr uint32_t kSlotIdxCommonArgIdx = 9;
constexpr uint32_t kSlotIdxAddrCommonArgIdx = 10;
constexpr uint32_t kValidGlobalAddrCommonArgIdx = 11;

uint32_t logical_local_cache_tokens(
    const ZeroPaddedKvCacheDeviceOperation::operation_attributes_t& args,
    const ZeroPaddedKvCacheDeviceOperation::tensor_args_t& tensor_args) {
    if (tensor_args.has_paged_cache()) {
        return static_cast<uint32_t>(
            tensor_args.page_bundle_indices->logical_volume() * static_cast<uint64_t>(args.kv_cache_page_size));
    }
    return tensor_args.cache.padded_shape()[-2];
}

void validate_paged_cache(
    const ZeroPaddedKvCacheDeviceOperation::operation_attributes_t& args,
    const ZeroPaddedKvCacheDeviceOperation::tensor_args_t& tensor_args) {
    if (!tensor_args.has_paged_cache()) {
        return;
    }
    const auto& cache = tensor_args.cache;
    const auto& table = *tensor_args.page_bundle_indices;
    TT_FATAL(
        args.kv_cache_page_size > 0 && args.kv_cache_page_size % TILE_HEIGHT == 0,
        "kv_cache_page_size must be a positive multiple of {} (got {})",
        TILE_HEIGHT,
        args.kv_cache_page_size);
    TT_FATAL(
        table.storage_type() == StorageType::DEVICE && table.buffer() != nullptr,
        "page_bundle_indices must be allocated on device");
    TT_FATAL(table.device() == cache.device(), "page_bundle_indices must be on the same device as cache");
    TT_FATAL(table.dtype() == DataType::UINT16, "page_bundle_indices must have uint16 dtype");
    TT_FATAL(table.layout() == Layout::ROW_MAJOR, "page_bundle_indices must use ROW_MAJOR layout");
    TT_FATAL(table.padded_shape() == table.logical_shape(), "page_bundle_indices must not be padded");
    TT_FATAL(
        table.memory_config().buffer_type() == BufferType::DRAM && !table.memory_config().is_sharded(),
        "page_bundle_indices must be DRAM interleaved");
    const auto& table_shape = table.logical_shape();
    TT_FATAL(
        table_shape.rank() == 4 && table_shape[0] == 1 && table_shape[1] == 1 && table_shape[2] == 1 &&
            table_shape[3] > 0,
        "page_bundle_indices must have shape [1,1,1,num_logical_bundles] (got {})",
        table_shape);

    const auto& cache_shape = cache.logical_shape();
    TT_FATAL(
        cache_shape.rank() == 4 && cache_shape[1] == 1 && cache_shape[2] == args.kv_cache_page_size,
        "Paged cache must have shape [physical_bundles*num_layers,1,kv_cache_page_size,D] (got {})",
        cache_shape);
    TT_FATAL(
        cache_shape[0] > 0 && cache_shape[0] % args.num_layers == 0,
        "Paged cache flat page count {} must be positive and divisible by num_layers {}",
        cache_shape[0],
        args.num_layers);
    const uint32_t physical_bundles = cache_shape[0] / args.num_layers;
    TT_FATAL(
        physical_bundles <= (1u << 16),
        "uint16 page_bundle_indices support at most 65536 physical bundles (got {})",
        physical_bundles);
    const auto nd = cache.nd_shard_spec();
    TT_FATAL(nd.has_value(), "Paged cache must use an ND-sharded memory config");
    const auto& shard = nd->shard_shape;
    TT_FATAL(
        shard.rank() == 4 && shard[0] == 1 && shard[1] == 1 && shard[2] == args.kv_cache_page_size &&
            shard[3] == cache_shape[3],
        "Each paged cache ND shard must be exactly [1,1,{},{}] (got {})",
        args.kv_cache_page_size,
        cache_shape[3],
        shard);
}

// Per-call scalar checks shared by the cache-miss and cache-hit paths.
void validate_runtime_args(
    const ZeroPaddedKvCacheDeviceOperation::operation_attributes_t& args,
    const ZeroPaddedKvCacheDeviceOperation::tensor_args_t& tensor_args) {
    TT_FATAL(args.cluster_axis == 0 || args.cluster_axis == 1, "cluster_axis ({}) must be 0 or 1", args.cluster_axis);
    const auto& cache = tensor_args.cache;
    validate_paged_cache(args, tensor_args);

    // Metadata-path invariant + tensor validation. The path is selected on slot_idx.has_value(), but the
    // factory dereferences valid_global->buffer() whenever slot_idx is set, so a mismatched optional would
    // null-deref -- reject it up front. Then validate each metadata tensor's structural properties (device,
    // UINT32, ROW_MAJOR, single-element) the reader/writer assume, so a malformed tensor fails host-side
    // with a clear message instead of a silent 4-byte on-device misread. Values stay off the dispatch path.
    TT_FATAL(
        tensor_args.slot_idx.has_value() == tensor_args.valid_global.has_value(),
        "metadata tensors slot_idx and valid_global must be supplied together (got slot_idx={}, valid_global={})",
        tensor_args.slot_idx.has_value(),
        tensor_args.valid_global.has_value());
    if (tensor_args.slot_idx.has_value()) {
        auto validate_meta = [&cache](const Tensor& meta, const char* name) {
            TT_FATAL(meta.storage_type() == StorageType::DEVICE, "metadata tensor {} must be on device", name);
            TT_FATAL(meta.dtype() == DataType::UINT32, "metadata tensor {} must be UINT32", name);
            TT_FATAL(meta.layout() == Layout::ROW_MAJOR, "metadata tensor {} must be ROW_MAJOR", name);
            TT_FATAL(
                meta.logical_volume() == 1,
                "metadata tensor {} must be a single element (got {})",
                name,
                meta.logical_volume());
            TT_FATAL(meta.device() == cache.device(), "metadata tensor {} must be on the same device as cache", name);
        };
        validate_meta(tensor_args.slot_idx.value(), "slot_idx");
        validate_meta(tensor_args.valid_global.value(), "valid_global");
    }

    // slot_idx is a host value only on the scalar path; on the tensor path it lives in the device
    // tensor (read on-device) and is the caller's responsibility.
    if (!tensor_args.slot_idx.has_value()) {
        if (tensor_args.has_paged_cache()) {
            TT_FATAL(
                args.slot_idx == 0, "Paged cache uses page_bundle_indices to select the request; slot_idx must be 0");
        } else {
            const uint32_t num_slots = cache.padded_shape()[0] / args.num_layers;
            TT_FATAL(
                args.slot_idx < num_slots, "slot_idx ({}) out of range for num_slots ({})", args.slot_idx, num_slots);
        }
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
    const uint32_t global_capacity = sp_factor * logical_local_cache_tokens(args, tensor_args);
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
    // The per-element-tensor (metadata) path is TILE-only: it patches the reader/writer kernels (0/2),
    // which only exist for TILE (ROW_MAJOR is a dataflow-only single writer). ROW_MAJOR must use the
    // scalar signature. (Guard added when rebasing the metadata overload onto main's ROW_MAJOR support.)
    if (tensor_args.slot_idx.has_value()) {
        TT_FATAL(
            cache.layout() == Layout::TILE,
            "zero_padded_kv_cache per-element-tensor (metadata) path supports TILE layout only; use the "
            "scalar signature for ROW_MAJOR");
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
        args.kv_cache_page_size,
        cache.dtype(),
        cache.layout(),
        cache.memory_config(),
        cache.padded_shape(),
        tensor_args.has_paged_cache(),
        tensor_args.has_paged_cache() ? tensor_args.page_bundle_indices->memory_config() : cache.memory_config(),
        tensor_args.has_paged_cache() ? tensor_args.page_bundle_indices->padded_shape() : tt::tt_metal::Shape{});
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
    const bool has_paged_cache = tensor_args.has_paged_cache();
    const uint32_t slot_idx_addr = has_metadata ? static_cast<uint32_t>(tensor_args.slot_idx->buffer()->address()) : 0u;
    const uint32_t valid_global_addr =
        has_metadata ? static_cast<uint32_t>(tensor_args.valid_global->buffer()->address()) : 0u;

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
    // Layout: 0 my_sp_coord, 1 sp_factor, 2 chunk_local(tokens), 3 valid_global, 4 pad_align,
    //         5 layer_idx, 6 num_layers, 7 Wt, 8 cache_CH_pages (== cache_CHtWt for TILE), 9 slot_idx,
    //         10 slot_idx_addr, 11 valid_global_addr.
    // Indices 3/9 are the scalar-path per-call values; 10/11 are the slot_idx/valid_global tensors' raw
    // DRAM addresses (metadata path, 0 on the scalar path). override_runtime_arguments patches whichever
    // applies. The metadata (per-element-tensor) path is TILE-only; ROW_MAJOR uses the scalar values.
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
        slot_idx_addr,
        valid_global_addr,
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
        if (has_paged_cache) {
            const uint32_t table_bytes = tensor_args.page_bundle_indices->logical_volume() * sizeof(uint16_t);
            const uint32_t aligned_table_bytes = (table_bytes + 31u) & ~31u;
            desc.cbs.push_back(CBDescriptor{
                .total_size = aligned_table_bytes,
                .core_ranges = all_cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = kPageTableCbIndexWriter,
                    .data_format = tt::DataFormat::RawUInt16,
                    .page_size = aligned_table_bytes,
                }}},
            });
        }

        KernelDescriptor writer;
        writer.kernel_source = kRowMajorWriterKernelPath;
        writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer.core_ranges = all_cores;
        writer.compile_time_args = {kZeroCbIndex, row_page_size};
        TensorAccessorArgs(cache.buffer()).append_to(writer.compile_time_args);
        TensorAccessorArgs(has_paged_cache ? tensor_args.page_bundle_indices->buffer() : cache.buffer())
            .append_to(writer.compile_time_args);
        writer.compile_time_args.insert(
            writer.compile_time_args.end(),
            {static_cast<uint32_t>(has_paged_cache),
             has_paged_cache ? kPageTableCbIndexWriter : 0u,
             args.kv_cache_page_size,
             args.num_layers,
             args.layer_idx,
             has_paged_cache ? static_cast<uint32_t>(tensor_args.page_bundle_indices->logical_volume()) : 0u});
        writer.config = WriterConfigDescriptor{};
        writer.common_runtime_args = common_runtime_args;
        writer.emplace_runtime_args(
            CoreCoord{0, 0},
            {cache.buffer(), has_paged_cache ? tensor_args.page_bundle_indices->buffer() : cache.buffer()});
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
    if (has_metadata) {
        // Metadata read scratch: a SEPARATE L1 page per kernel. The reader and writer run on the same
        // core, so sharing one CB would race their concurrent NoC reads (see kMetaCbIndex/...Writer).
        add_cb(kMetaCbIndex, tt::DataFormat::UInt32, kMetadataBytes, 1);
        add_cb(kMetaCbIndexWriter, tt::DataFormat::UInt32, kMetadataBytes, 1);
    }
    if (has_paged_cache) {
        const uint32_t table_bytes = tensor_args.page_bundle_indices->logical_volume() * sizeof(uint16_t);
        const uint32_t aligned_table_bytes = (table_bytes + 31u) & ~31u;
        add_cb(kPageTableCbIndex, tt::DataFormat::RawUInt16, aligned_table_bytes, 1);
        add_cb(kPageTableCbIndexWriter, tt::DataFormat::RawUInt16, aligned_table_bytes, 1);
    }

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
    TensorAccessorArgs(has_paged_cache ? tensor_args.page_bundle_indices->buffer() : cache.buffer())
        .append_to(reader.compile_time_args);
    reader.compile_time_args.insert(
        reader.compile_time_args.end(),
        {static_cast<uint32_t>(has_paged_cache),
         has_paged_cache ? kPageTableCbIndex : 0u,
         args.kv_cache_page_size / TILE_HEIGHT,
         args.num_layers,
         args.layer_idx,
         has_paged_cache ? static_cast<uint32_t>(tensor_args.page_bundle_indices->logical_volume()) : 0u});
    reader.config = ReaderConfigDescriptor{};
    reader.common_runtime_args = common_runtime_args;
    reader.emplace_runtime_args(
        CoreCoord{0, 0},
        {cache.buffer(), has_paged_cache ? tensor_args.page_bundle_indices->buffer() : cache.buffer()});

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
    // Same leading layout as the reader: [3]=has_metadata, [4]=metadata CB index, cache accessor at
    // <5>, then the metadata accessor when present.
    writer.compile_time_args = {
        kOutCbIndex,
        kZeroCbIndex,
        cache_tile_size,
        static_cast<uint32_t>(has_metadata),
        has_metadata ? kMetaCbIndexWriter : 0u};  // writer's OWN metadata CB (disjoint from the reader's)
    TensorAccessorArgs(cache.buffer()).append_to(writer.compile_time_args);
    if (has_metadata) {
        // One metadata accessor, reused for both 1-element tensors (identical layout); the kernel reads
        // each from its own DRAM address (common args 10/11).
        TensorAccessorArgs(tensor_args.slot_idx->buffer()).append_to(writer.compile_time_args);
    }
    TensorAccessorArgs(has_paged_cache ? tensor_args.page_bundle_indices->buffer() : cache.buffer())
        .append_to(writer.compile_time_args);
    writer.compile_time_args.insert(
        writer.compile_time_args.end(),
        {static_cast<uint32_t>(has_paged_cache),
         has_paged_cache ? kPageTableCbIndexWriter : 0u,
         args.kv_cache_page_size / TILE_HEIGHT,
         args.num_layers,
         args.layer_idx,
         has_paged_cache ? static_cast<uint32_t>(tensor_args.page_bundle_indices->logical_volume()) : 0u});
    writer.config = WriterConfigDescriptor{};
    writer.common_runtime_args = common_runtime_args;
    writer.emplace_runtime_args(
        CoreCoord{0, 0},
        {cache.buffer(), has_paged_cache ? tensor_args.page_bundle_indices->buffer() : cache.buffer()});
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
    //   - metadata (tensor) path [TILE-only]: reader (0) and writer (2) read slot_idx/valid_global
    //     on-device from the two 1-element tensors, so only their raw DRAM addresses (indices 10/11)
    //     need patching; compute (1) reads only structural args.
    //   - scalar path: every kernel reads slot_idx (9)/valid_global (3). TILE has reader/compute/writer
    //     (3 kernels); ROW_MAJOR is a dataflow-only writer (1 kernel).
    if (tensor_args.slot_idx.has_value()) {
        const uint32_t slot_idx_addr = static_cast<uint32_t>(tensor_args.slot_idx->buffer()->address());
        const uint32_t valid_global_addr = static_cast<uint32_t>(tensor_args.valid_global->buffer()->address());
        for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
            for (uint32_t kernel_handle : {0u, 2u}) {  // reader, writer (metadata path is TILE-only)
                auto& common = GetCommonRuntimeArgs(program, kernel_handle);
                TT_FATAL(
                    kValidGlobalAddrCommonArgIdx < common.size(),
                    "zero_padded_kv_cache kernel missing the metadata-tensor addr common args");
                common[kSlotIdxAddrCommonArgIdx] = slot_idx_addr;
                common[kValidGlobalAddrCommonArgIdx] = valid_global_addr;
            }
        }
    } else {
        const uint32_t num_kernels = tensor_args.cache.layout() == Layout::ROW_MAJOR ? 1u : 3u;
        for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
            for (uint32_t kernel_handle = 0; kernel_handle < num_kernels; ++kernel_handle) {
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
    uint32_t pad_align,
    const std::optional<ttnn::Tensor>& page_bundle_indices,
    uint32_t kv_cache_page_size) {
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
        .kv_cache_page_size = kv_cache_page_size,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .cache = cache,
        .slot_idx = slot_idx_tensor,
        .valid_global = valid_global_tensor,
        .page_bundle_indices = page_bundle_indices};
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
