// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "update_padded_kv_cache_device_operation.hpp"

#include <cstdint>
#include <utility>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache {

using namespace tt::tt_metal;
using namespace tt::constants;

namespace {

// Reader kernel is reused from the kv_cache fill path — purely (src_addr, num_tiles, src_start) rt-args.
// Writer is a forked variant that derives `start_id` on-device from the per-call `slot_idx` and
// `kv_actual_global` (common runtime args patched on cache hits) plus structural common rt-args
// (`my_sp_coord`/`sp_factor`/`layer_idx` etc.). The per-call scalars are patched by the op's
// MeshWorkloadFactory::override_runtime_arguments, so the buffer-binding fast cache-hit path stays
// correct while those values remain out of the program hash.
constexpr auto kReaderKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/kernels/dataflow/"
    "reader_update_padded_kv_cache.cpp";
constexpr auto kWriterKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/kernels/dataflow/"
    "writer_update_padded_kv_cache.cpp";

constexpr uint32_t kSrcCbIndex = 0;
constexpr uint32_t kNumInputPagesDoubleBuffered = 2;

// Per-call scalar checks shared by the cache-miss and cache-hit paths. slot_idx and kv_actual_global
// are now host-side scalars (common runtime args patched on cache hits), so the value-range checks
// dropped when they lived in device tensors can be enforced here again.
void validate_runtime_args(
    const UpdatePaddedKvCacheDeviceOperation::operation_attributes_t& args,
    const UpdatePaddedKvCacheDeviceOperation::tensor_args_t& tensor_args) {
    TT_FATAL(args.cluster_axis == 0 || args.cluster_axis == 1, "cluster_axis ({}) must be 0 or 1", args.cluster_axis);
    // The writer divides kv_actual_global by TILE_HEIGHT to get its tile offset, so it must be aligned.
    TT_FATAL(
        args.kv_actual_global % TILE_HEIGHT == 0,
        "kv_actual_global ({}) must be tile-aligned (a multiple of {})",
        args.kv_actual_global,
        TILE_HEIGHT);
    const auto& cache = tensor_args.cache;
    const uint32_t num_slots = cache.padded_shape()[0] / args.num_layers;
    TT_FATAL(args.slot_idx < num_slots, "slot_idx ({}) out of range for num_slots ({})", args.slot_idx, num_slots);

    // This chunk is written at a per-chip offset derived from kv_actual_global; the prior valid KV
    // plus this chunk must fit the global cache capacity (sp_factor slabs of cache_seq tokens each),
    // else the write spills past the cache. sp_factor = mesh extent along cluster_axis.
    const auto& mesh_view = cache.device()->get_view();
    TT_FATAL(mesh_view.is_mesh_2d(), "update_padded_kv_cache requires a 2D mesh");
    const uint32_t sp_factor = (args.cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    const uint32_t chunk_global_tokens = sp_factor * tensor_args.input.padded_shape()[-2];
    const uint32_t global_cache_capacity = sp_factor * cache.padded_shape()[-2];
    TT_FATAL(
        args.kv_actual_global + chunk_global_tokens <= global_cache_capacity,
        "kv_actual_global ({}) + chunk_global ({}) would overflow global cache capacity ({})",
        args.kv_actual_global,
        chunk_global_tokens,
        global_cache_capacity);
}

}  // namespace

UpdatePaddedKvCacheDeviceOperation::program_factory_t UpdatePaddedKvCacheDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return MeshWorkloadFactory{};
}

void UpdatePaddedKvCacheDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& cache = tensor_args.cache;
    const auto& input = tensor_args.input;

    TT_FATAL(cache.storage_type() == StorageType::DEVICE, "cache must be on device");
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "input must be on device");
    TT_FATAL(cache.dtype() == input.dtype(), "cache and input dtype must match");

    // Layout / dtype gating. The op is a pure page copy, so it supports both TILE and ROW_MAJOR;
    // the page-unit math in create_descriptor branches on layout. The two formats are mutually
    // exclusive per dtype family:
    //   - block-float (bfloat8_b/bfloat4_b) carries a per-face shared exponent and is tile-only.
    //   - fp8_e4m3 is ROW_MAJOR-only (Blackhole) in ttnn today.
    TT_FATAL(cache.layout() == input.layout(), "cache and input layout must match");
    TT_FATAL(input.layout() == Layout::TILE || input.layout() == Layout::ROW_MAJOR, "layout must be TILE or ROW_MAJOR");
    if (tt::tt_metal::is_block_float(input.dtype())) {
        TT_FATAL(input.layout() == Layout::TILE, "block-float dtypes (bfloat8_b/bfloat4_b) require TILE layout");
    }
    if (input.dtype() == DataType::FP8_E4M3) {
        TT_FATAL(input.layout() == Layout::ROW_MAJOR, "FP8_E4M3 requires ROW_MAJOR layout");
    }

    const auto& cache_shape = cache.padded_shape();
    const auto& input_shape = input.padded_shape();
    TT_FATAL(cache_shape.rank() == 4, "cache must be 4D (got rank {})", cache_shape.rank());
    TT_FATAL(input_shape.rank() == 4, "input must be 4D (got rank {})", input_shape.rank());
    TT_FATAL(cache_shape[-1] == input_shape[-1], "cache and input head dim must match");
    TT_FATAL(cache_shape[1] == input_shape[1], "cache and input num-heads dim must match");

    const uint32_t cache_seq = cache_shape[-2];
    const uint32_t input_seq = input_shape[-2];
    // Seq / offset arithmetic stays tile-granular (multiples of 32) in BOTH layouts: the writer's
    // update_idxt boundary math counts tile-rows even when ROW_MAJOR makes each page a single token
    // row, so input/cache seq must be 32-aligned regardless of layout.
    TT_FATAL(input_seq % TILE_HEIGHT == 0, "input seq dim ({}) must be tile-aligned", input_seq);
    TT_FATAL(cache_seq % TILE_HEIGHT == 0, "cache seq dim ({}) must be tile-aligned", cache_seq);
    TT_FATAL(cache_seq % input_seq == 0, "cache seq ({}) must be a multiple of input seq ({})", cache_seq, input_seq);

    TT_FATAL(args.num_layers > 0, "num_layers must be positive");
    TT_FATAL(
        cache_shape[0] % args.num_layers == 0,
        "cache batch dim ({}) must be a multiple of num_layers ({})",
        cache_shape[0],
        args.num_layers);
    // layer_idx is hashed (structural), so this is a miss-only check and stays valid for the cached
    // program's lifetime. slot_idx / kv_actual_global value checks live in validate_runtime_args.
    TT_FATAL(
        args.layer_idx < args.num_layers,
        "layer_idx {} out of range for num_layers {}",
        args.layer_idx,
        args.num_layers);

    // The 2D-mesh requirement and the per-call value checks (kv tile-alignment, slot range, cache
    // capacity) are enforced by validate_runtime_args, run on both the cache-miss and cache-hit paths.
    validate_runtime_args(args, tensor_args);
}

void UpdatePaddedKvCacheDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // slot_idx and kv_actual_global are per-call scalars (not hashed) and can differ from the compiled
    // program's call, so re-validate their ranges every hit. Structural constraints are hashed.
    validate_runtime_args(args, tensor_args);
}

UpdatePaddedKvCacheDeviceOperation::spec_return_value_t UpdatePaddedKvCacheDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    // In-place: output spec = cache spec.
    return tensor_args.cache.tensor_spec();
}

UpdatePaddedKvCacheDeviceOperation::tensor_return_value_t UpdatePaddedKvCacheDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    // In-place: return a handle to the cache.
    return tensor_args.cache;
}

ttsl::hash::hash_t UpdatePaddedKvCacheDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // slot_idx and kv_actual_global are per-call scalars held in common runtime args and patched on
    // cache hits, so they are intentionally NOT hashed and successive users/chunks reuse the cached
    // program. layer_idx IS hashed: it takes only num_layers distinct values, so one program per layer
    // is reused across users/chunks, and a hashed scalar stays correct on the fast cache-hit path.
    // num_layers and cluster_axis stay IN: both are structural — they govern the cache slot
    // linearization and which mesh dim is sp — not per-call data.
    const auto& cache = tensor_args.cache;
    const auto& input = tensor_args.input;
    // Hash the full padded shapes, not just their volumes: the descriptor derives Wt, input_Ht,
    // cache_HtWt/CHtWt and the work split from specific dimensions, so two differently-shaped
    // tensors that happen to share a volume must NOT collide onto the same cached program.
    return tt::tt_metal::operation::hash_operation<UpdatePaddedKvCacheDeviceOperation>(
        args.layer_idx,
        args.num_layers,
        args.cluster_axis,
        input.dtype(),
        input.layout(),  // TILE vs ROW_MAJOR drives the page-unit math; must not collide
        input.memory_config(),
        input.padded_shape(),
        cache.memory_config(),
        cache.padded_shape());
}

tt::tt_metal::ProgramDescriptor UpdatePaddedKvCacheDeviceOperation::ProgramFactory::create_descriptor(
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*output*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(
        mesh_dispatch_coordinate.has_value(),
        "UpdatePaddedKvCache::create_descriptor requires a mesh dispatch coordinate");
    const auto& coord = mesh_dispatch_coordinate.value();

    const auto& cache = tensor_args.cache;
    const auto& input = tensor_args.input;
    auto* device = input.device();

    const auto& cache_shape = cache.padded_shape();
    const auto& input_shape = input.padded_shape();

    const tt::DataFormat data_format = datatype_to_dataformat_converter(input.dtype());

    // The op moves the cache as opaque pages: in TILE layout a page is a 32x32 tile; in ROW_MAJOR a
    // page is one token row (head_dim wide). All of the writer's start_id / update_idxt arithmetic is
    // expressed in pages, so switching layout is purely a reinterpretation of these page-unit counts
    // (plus the page byte size and the writer's tile_height compile arg). The seq/offset asserts keep
    // everything 32-row-aligned in both layouts, so the boundary math is identical.
    const bool is_row_major = input.layout() == Layout::ROW_MAJOR;

    uint32_t single_page_size;  // bytes per page / CB page
    uint32_t Wt;                // width pages per (head, seq-row)
    uint32_t input_Ht;          // input seq in page-rows
    uint32_t cache_HtWt;        // cache page-rows per head
    uint32_t writer_tile_height;
    if (is_row_major) {
        // ROW_MAJOR: page = one token row; use the buffer's aligned page size (handles row padding).
        single_page_size = cache.buffer()->aligned_page_size();
        Wt = 1;
        input_Ht = input_shape[-2];
        cache_HtWt = cache_shape[-2];
        writer_tile_height = 1;
    } else {
        single_page_size = tt::tile_size(data_format);
        Wt = cache_shape[-1] / TILE_WIDTH;
        input_Ht = input_shape[-2] / TILE_HEIGHT;
        cache_HtWt = cache_shape[-2] * Wt / TILE_HEIGHT;
        writer_tile_height = TILE_HEIGHT;
    }
    const uint32_t cache_CHtWt = cache_shape[1] * cache_HtWt;

    // Per-chip kernel inputs: kernel does the update_idxt + start_id math itself from these.
    // sp_factor is the mesh extent along the cluster axis (validated 2D in validate_runtime_args).
    const auto& mesh_view = device->get_view();
    const uint32_t sp_factor = (args.cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    const uint32_t my_sp_coord = ::ttnn::ccl::get_linearized_index_from_physical_coord(cache, coord, args.cluster_axis);

    // Work split: one tile per "block". num_blocks_of_work = input_C * input_Ht (= num_heads * seq_tiles).
    const uint32_t num_blocks_of_work = input_shape[1] * input_Ht;

    const auto compute_grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_g1, num_blocks_per_core_g2] =
        tt::tt_metal::split_work_to_cores(compute_grid, num_blocks_of_work, /*row_wise=*/true);

    tt::tt_metal::ProgramDescriptor desc;

    // CB for the input pages (a page is a tile in TILE layout, a token row in ROW_MAJOR).
    desc.cbs.push_back(CBDescriptor{
        .total_size = kNumInputPagesDoubleBuffered * single_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kSrcCbIndex,
            .data_format = data_format,
            .page_size = single_page_size,
        }}},
    });

    // Reader kernel descriptor.
    KernelDescriptor::CompileTimeArgs reader_compile_args;
    TensorAccessorArgs(input.buffer()).append_to(reader_compile_args);

    KernelDescriptor reader_kernel;
    reader_kernel.kernel_source = kReaderKernelPath;
    reader_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel.core_ranges = all_cores;
    reader_kernel.compile_time_args = std::move(reader_compile_args);
    reader_kernel.config = ReaderConfigDescriptor{};

    // Writer kernel descriptor. Compile args: src CB (read), tile_height (divides kv tokens into the
    // page-row unit: TILE_HEIGHT for TILE, 1 for ROW_MAJOR), then the cache tensor accessor.
    KernelDescriptor::CompileTimeArgs writer_compile_args = {kSrcCbIndex, writer_tile_height};
    TensorAccessorArgs(cache.buffer()).append_to(writer_compile_args);

    KernelDescriptor writer_kernel;
    writer_kernel.kernel_source = kWriterKernelPath;
    writer_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel.core_ranges = all_cores;
    writer_kernel.compile_time_args = std::move(writer_compile_args);
    writer_kernel.config = WriterConfigDescriptor{};

    // Common rt-args: per-chip kernel inputs for on-device update_idxt + start_id derivation. Indices
    // 0-7 are structural (constant for this cached program): my_sp_coord/sp_factor are this chip's mesh
    // position, layer_idx is hashed. slot_idx (8) and kv_actual_global (9) are the per-call values
    // patched on cache hits by override_runtime_arguments. The kernel composes
    // batch_idx = slot_idx*num_layers + layer_idx.
    writer_kernel.emplace_common_runtime_args({
        my_sp_coord,
        sp_factor,
        input_Ht,
        args.layer_idx,
        args.num_layers,
        Wt,
        cache_HtWt,
        cache_CHtWt,
        args.slot_idx,
        args.kv_actual_global,
    });

    // Per-core runtime args. Buffers are passed as Buffer* bindings (not raw addresses) so cache hits
    // take the fast path that patches addresses and skips create_descriptor. The per-call scalars
    // (slot_idx, kv_actual_global) are common args patched by override_runtime_arguments, so no per-core
    // scalar goes stale.
    auto* src_buffer = input.buffer();
    auto* dst_buffer = cache.buffer();
    const uint32_t g1_numcores = core_group_1.num_cores();

    const auto cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);
    reader_kernel.runtime_args.reserve(num_cores);
    writer_kernel.runtime_args.reserve(num_cores);

    uint32_t num_blocks_written = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord& core = cores.at(i);
        const uint32_t num_blocks_per_core = (i < g1_numcores) ? num_blocks_per_core_g1 : num_blocks_per_core_g2;

        // Reader: (src_addr, num_tiles, src_start_tile_id)
        reader_kernel.emplace_runtime_args(core, {src_buffer, num_blocks_per_core * Wt, num_blocks_written * Wt});

        // Writer: (dst_addr, num_pages, core_blocks_written) — kernel derives update_idxt + head
        // offset from the slot_idx/kv_actual_global common args.
        writer_kernel.emplace_runtime_args(core, {dst_buffer, num_blocks_per_core * Wt, num_blocks_written});

        num_blocks_written += num_blocks_per_core;
    }

    desc.kernels.push_back(std::move(reader_kernel));
    desc.kernels.push_back(std::move(writer_kernel));
    return desc;
}

UpdatePaddedKvCacheDeviceOperation::MeshWorkloadFactory::cached_mesh_workload_t
UpdatePaddedKvCacheDeviceOperation::MeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    return descriptor_adapter_t::create_mesh_workload(args, tensor_coords, tensor_args, output);
}

void UpdatePaddedKvCacheDeviceOperation::MeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    // Default adapter behaviour: patch operand buffer-binding addresses on cache hits.
    descriptor_adapter_t::apply_descriptor(cached_workload, args, tensor_args, output);
    // The writer's common runtime args 8/9 hold slot_idx/kv_actual_global -- the per-call values the
    // buffer-binding fast path would otherwise leave stale. Patch them on every cached program.
    constexpr uint32_t kWriterKernelHandle = 1;  // writer is pushed second in create_descriptor
    constexpr uint32_t kSlotIdxCommonArgIdx = 8;
    constexpr uint32_t kKvActualGlobalCommonArgIdx = 9;
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& writer_common = GetCommonRuntimeArgs(program, kWriterKernelHandle);
        TT_FATAL(
            kKvActualGlobalCommonArgIdx < writer_common.size(),
            "update_padded_kv_cache writer is missing the slot_idx/kv_actual_global common args");
        writer_common[kSlotIdxCommonArgIdx] = args.slot_idx;
        writer_common[kKvActualGlobalCommonArgIdx] = args.kv_actual_global;
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache

namespace ttnn::prim {

ttnn::Tensor update_padded_kv_cache(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& input,
    uint32_t slot_idx,
    uint32_t layer_idx,
    uint32_t num_layers,
    uint32_t kv_actual_global,
    uint32_t cluster_axis) {
    using OperationType =
        ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache::UpdatePaddedKvCacheDeviceOperation;
    auto attrs = OperationType::operation_attributes_t{
        .slot_idx = slot_idx,
        .kv_actual_global = kv_actual_global,
        .layer_idx = layer_idx,
        .num_layers = num_layers,
        .cluster_axis = cluster_axis,
    };
    auto tensor_args = OperationType::tensor_args_t{.cache = cache, .input = input};
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
