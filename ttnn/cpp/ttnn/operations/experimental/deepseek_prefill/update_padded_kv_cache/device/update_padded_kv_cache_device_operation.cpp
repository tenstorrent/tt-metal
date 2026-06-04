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
// Writer is a forked variant that derives `start_id` on-device: it reads the per-call `slot_idx` and
// `kv_actual_global` from single-element device tensors and takes `my_sp_coord`/`sp_factor`/`layer_idx`
// etc. as common rt-args, so no per-call scalar lives in the runtime args. That keeps the
// buffer-binding fast cache-hit path correct (it patches addresses but skips create_descriptor).
constexpr auto kReaderKernelPath =
    "ttnn/cpp/ttnn/operations/kv_cache/device/kernels/dataflow/reader_fill_cache_interleaved_start_id.cpp";
constexpr auto kWriterKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/kernels/dataflow/"
    "writer_update_padded_kv_cache.cpp";

constexpr uint32_t kSrcCbIndex = 0;
constexpr uint32_t kSlotCbIndex = 1;
constexpr uint32_t kKvCbIndex = 2;
constexpr uint32_t kNumInputTilesDoubleBuffered = 2;

// Structural validation of an index tensor (slot_idx / kv_actual_global): a single-element ROW_MAJOR
// uint32 datum on the same device as the cache, read on-device by the writer. Its value is unknown
// host-side (it is intentionally out of the program hash), so value-range checks that the old scalar
// path did (kv_actual tile-alignment, slot_idx < num_slots, global capacity overflow) are dropped --
// the same trade-off as the indexed-RoPE op; only structure is checked, and only on the cache miss.
void validate_index_tensor(const Tensor& t, const Tensor& input, const char* name) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(t.buffer() != nullptr, "{} must be allocated in a buffer on device", name);
    TT_FATAL(t.device() == input.device(), "{} must be on the same device as input", name);
    TT_FATAL(t.layout() == Layout::ROW_MAJOR, "{} must be ROW_MAJOR layout", name);
    TT_FATAL(t.dtype() == DataType::UINT32, "{} must be uint32 (got {})", name, t.dtype());
}

}  // namespace

UpdatePaddedKvCacheDeviceOperation::program_factory_t UpdatePaddedKvCacheDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return ProgramFactory{};
}

void UpdatePaddedKvCacheDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& cache = tensor_args.cache;
    const auto& input = tensor_args.input;

    TT_FATAL(cache.storage_type() == StorageType::DEVICE, "cache must be on device");
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "input must be on device");
    TT_FATAL(cache.layout() == Layout::TILE, "cache must be TILE layout");
    TT_FATAL(input.layout() == Layout::TILE, "input must be TILE layout");
    TT_FATAL(cache.dtype() == input.dtype(), "cache and input dtype must match");

    const auto& cache_shape = cache.padded_shape();
    const auto& input_shape = input.padded_shape();
    TT_FATAL(cache_shape.rank() == 4, "cache must be 4D (got rank {})", cache_shape.rank());
    TT_FATAL(input_shape.rank() == 4, "input must be 4D (got rank {})", input_shape.rank());
    TT_FATAL(cache_shape[-1] == input_shape[-1], "cache and input head dim must match");
    TT_FATAL(cache_shape[1] == input_shape[1], "cache and input num-heads dim must match");

    const uint32_t cache_seq = cache_shape[-2];
    const uint32_t input_seq = input_shape[-2];
    TT_FATAL(input_seq % TILE_HEIGHT == 0, "input seq dim ({}) must be tile-aligned", input_seq);
    TT_FATAL(cache_seq % input_seq == 0, "cache seq ({}) must be a multiple of input seq ({})", cache_seq, input_seq);

    TT_FATAL(args.num_layers > 0, "num_layers must be positive");
    TT_FATAL(
        cache_shape[0] % args.num_layers == 0,
        "cache batch dim ({}) must be a multiple of num_layers ({})",
        cache_shape[0],
        args.num_layers);
    // layer_idx is hashed (structural), so this is a miss-only check and stays valid for the cached
    // program's lifetime; slot_idx's value-range check is dropped (its value is on-device only).
    TT_FATAL(
        args.layer_idx < args.num_layers,
        "layer_idx {} out of range for num_layers {}",
        args.layer_idx,
        args.num_layers);

    // 2D mesh is required for the per-chip sp offset derivation (sp_factor = mesh extent on cluster_axis).
    const auto& mesh_view = cache.device()->get_view();
    TT_FATAL(mesh_view.is_mesh_2d(), "update_padded_kv_cache requires a 2D mesh");

    // slot_idx and kv_actual_global are single-element device tensors read on-device; check structure.
    validate_index_tensor(tensor_args.slot_idx, input, "slot_idx");
    validate_index_tensor(tensor_args.kv_actual_global, input, "kv_actual_global");
}

void UpdatePaddedKvCacheDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    // Nothing to re-validate per hit: slot_idx and kv_actual_global are device tensors whose values
    // are unknown host-side (and out of the hash), and layer_idx is hashed (so structural and
    // guaranteed unchanged for this cached program). All structural constraints were checked on miss.
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
    // slot_idx and kv_actual_global are per-call device tensors read on-device and intentionally
    // NOT hashed by value, so successive users/chunks reuse the cached program (their addresses are
    // patched on cache hits via the buffer-binding fast path). Only their structure (dtype/mem_config)
    // is hashed. layer_idx IS hashed: it takes only num_layers distinct values, so one program per
    // layer is reused across users/chunks, and a hashed scalar (unlike a non-hashed common rt-arg)
    // stays correct on the fast cache-hit path. num_layers and cluster_axis stay IN: both are
    // structural — they govern the cache slot linearization and which mesh dim is sp — not per-call data.
    const auto& cache = tensor_args.cache;
    const auto& input = tensor_args.input;
    const auto& slot_idx = tensor_args.slot_idx;
    const auto& kv_actual_global = tensor_args.kv_actual_global;
    // Hash the full padded shapes, not just their volumes: the descriptor derives Wt, input_Ht,
    // cache_HtWt/CHtWt and the work split from specific dimensions, so two differently-shaped
    // tensors that happen to share a volume must NOT collide onto the same cached program.
    return tt::tt_metal::operation::hash_operation<UpdatePaddedKvCacheDeviceOperation>(
        args.layer_idx,
        args.num_layers,
        args.cluster_axis,
        input.dtype(),
        input.memory_config(),
        input.padded_shape(),
        cache.memory_config(),
        cache.padded_shape(),
        slot_idx.dtype(),
        slot_idx.memory_config(),
        kv_actual_global.dtype(),
        kv_actual_global.memory_config());
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
    const auto& slot_idx = tensor_args.slot_idx;
    const auto& kv_actual_global = tensor_args.kv_actual_global;
    auto* device = input.device();

    const auto& cache_shape = cache.padded_shape();
    const auto& input_shape = input.padded_shape();

    const tt::DataFormat data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t single_tile_size = tt::tile_size(data_format);

    // slot_idx and kv_actual_global are single ROW_MAJOR uint32 data, each read as one page into a
    // small CB. Their values are read on-device by the writer, so they never enter the program hash.
    const tt::DataFormat slot_cb_data_format = datatype_to_dataformat_converter(slot_idx.dtype());
    const uint32_t slot_page_size = slot_idx.buffer()->aligned_page_size();
    const tt::DataFormat kv_cb_data_format = datatype_to_dataformat_converter(kv_actual_global.dtype());
    const uint32_t kv_page_size = kv_actual_global.buffer()->aligned_page_size();

    const uint32_t Wt = cache_shape[-1] / TILE_WIDTH;
    const uint32_t input_Ht = input_shape[-2] / TILE_HEIGHT;
    const uint32_t cache_HtWt = cache_shape[-2] * Wt / TILE_HEIGHT;
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

    // CB for the input tiles.
    desc.cbs.push_back(CBDescriptor{
        .total_size = kNumInputTilesDoubleBuffered * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kSrcCbIndex,
            .data_format = data_format,
            .page_size = single_tile_size,
        }}},
    });

    // Small CBs the writer reads slot_idx and kv_actual_global into (one uint32 page each).
    desc.cbs.push_back(CBDescriptor{
        .total_size = slot_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kSlotCbIndex,
            .data_format = slot_cb_data_format,
            .page_size = slot_page_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = kv_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kKvCbIndex,
            .data_format = kv_cb_data_format,
            .page_size = kv_page_size,
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

    // Writer kernel descriptor. Compile args: src CB (read), slot/kv CBs (on-device index reads),
    // TILE_HEIGHT (divides kv tokens into tiles), then the cache/slot/kv tensor accessors.
    KernelDescriptor::CompileTimeArgs writer_compile_args = {kSrcCbIndex, kSlotCbIndex, kKvCbIndex, TILE_HEIGHT};
    TensorAccessorArgs(cache.buffer()).append_to(writer_compile_args);
    TensorAccessorArgs(slot_idx.buffer()).append_to(writer_compile_args);
    TensorAccessorArgs(kv_actual_global.buffer()).append_to(writer_compile_args);

    KernelDescriptor writer_kernel;
    writer_kernel.kernel_source = kWriterKernelPath;
    writer_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel.core_ranges = all_cores;
    writer_kernel.compile_time_args = std::move(writer_compile_args);
    writer_kernel.config = WriterConfigDescriptor{};

    // Common rt-args: per-chip kernel inputs for on-device update_idxt + start_id derivation. These
    // are structural (constant across calls for this cached program): my_sp_coord/sp_factor are this
    // chip's mesh position, and layer_idx is hashed. slot_idx and kv_actual_global are NOT here --
    // they arrive via device tensors read on-device, so no per-call scalar is left to go stale on the
    // buffer-binding fast cache-hit path. The kernel composes batch_idx = slot_idx*num_layers+layer_idx.
    writer_kernel.emplace_common_runtime_args({
        my_sp_coord,
        sp_factor,
        input_Ht,
        args.layer_idx,
        args.num_layers,
        Wt,
        cache_HtWt,
        cache_CHtWt,
    });

    // Per-core runtime args. Buffers are passed as Buffer* bindings (not raw addresses) so cache hits
    // take the fast path that patches addresses and skips create_descriptor — correct now that the
    // per-call values (slot_idx, kv_actual_global) live in device tensors, leaving no stale scalar.
    auto* src_buffer = input.buffer();
    auto* dst_buffer = cache.buffer();
    auto* slot_buffer = slot_idx.buffer();
    auto* kv_buffer = kv_actual_global.buffer();
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

        // Writer: (dst_addr, slot_addr, kv_addr, num_pages, core_blocks_written) — kernel reads
        // slot/kv on-device and adds update_idxt + head offset itself.
        writer_kernel.emplace_runtime_args(
            core, {dst_buffer, slot_buffer, kv_buffer, num_blocks_per_core * Wt, num_blocks_written});

        num_blocks_written += num_blocks_per_core;
    }

    desc.kernels.push_back(std::move(reader_kernel));
    desc.kernels.push_back(std::move(writer_kernel));
    return desc;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache

namespace ttnn::prim {

ttnn::Tensor update_padded_kv_cache(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& input,
    const ttnn::Tensor& slot_idx,
    uint32_t layer_idx,
    uint32_t num_layers,
    const ttnn::Tensor& kv_actual_global,
    uint32_t cluster_axis) {
    using OperationType =
        ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache::UpdatePaddedKvCacheDeviceOperation;
    auto attrs = OperationType::operation_attributes_t{
        .layer_idx = layer_idx,
        .num_layers = num_layers,
        .cluster_axis = cluster_axis,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .cache = cache, .input = input, .slot_idx = slot_idx, .kv_actual_global = kv_actual_global};
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
