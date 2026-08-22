// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sort_device_operation.hpp"
#include "tt_stl/assert.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

constexpr uint32_t SORT_WT_THRESHOLD = 64;
// UINT16 + ROW_MAJOR uses a lower SingleCore threshold: the SingleCore RM path
// promotes the value CBs (rm_input_cb, rm_value_output_cb) from UInt16 to
// Float32 (2× storage) and its per-row page size scales with the full W (Wt *
// TILE_W * 4 B), so at Wt = 64 the sum of static CBs exceeds the ~1.5 MB L1
// budget.  The MultiCore factory's UINT16 RM path uses per-tile pages and does
// not scale with Wt, so we route Wt > 32 UINT16 RM inputs there.  Empirically
// Wt = 32 stays comfortably below the L1 cap (~1.1 MB static CBs); Wt = 64
// OOMs during program allocation.
constexpr uint32_t SORT_WT_THRESHOLD_UINT16_ROW_MAJOR = 32;

// Mergesort row-engine eligibility (issue #33492 roadmap): stable bfloat16 rows of padded
// width 2048 or 4096 on Blackhole run a full per-row sort on the TopK XL SFPU kernels
// (fused linearly-tagged keys — stability is structural, no comparator), one row per core.
// Blackhole-only because the TopK XL LLKs have no Wormhole tree. The engine's fused tags
// live in 32-bit DEST words, so its indices ride the UINT32 transport (the u16-in-32-bit-
// DEST combination has no working pack path); a caller who preallocates UINT16 index
// tensors opts out to the previous routing in select_program_factory.
static bool mergesort_row_engine_eligible(
    const SortDeviceOperation::operation_attributes_t& attributes,
    const SortDeviceOperation::tensor_args_t& tensor_args,
    uint32_t sort_w_dim) {
    return attributes.stable && tensor_args.input_tensor.dtype() == DataType::BFLOAT16 &&
           tensor_args.input_tensor.device()->arch() == tt::ARCH::BLACKHOLE &&
           (sort_w_dim == 2048 || sort_w_dim == 4096);
}

SortDeviceOperation::program_factory_t SortDeviceOperation::select_program_factory(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const bool is_row_major = (tensor_args.input_tensor.layout() == Layout::ROW_MAJOR);
    const uint32_t w_dim =
        is_row_major ? tensor_args.input_tensor.logical_shape()[3] : tensor_args.input_tensor.padded_shape()[3];
    const auto tile_width = tensor_args.input_tensor.tensor_spec().tile().get_width();
    const uint32_t Wt = w_dim / tile_width;

    auto* const device = tensor_args.input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t total_number_of_cores = compute_with_storage_grid_size.y * compute_with_storage_grid_size.x;

    const auto input_dtype = tensor_args.input_tensor.dtype();
    const auto output_specs = compute_output_specs(attributes, tensor_args);
    const auto index_dtype = output_specs[1].data_type();

    const uint32_t total_number_of_tiles_for_hybrid_approach =
        total_number_of_cores *
        SortProgramFactoryCrossCoreDataExchange::get_number_of_tiles_per_core(
            total_number_of_cores,
            Wt,
            input_dtype,
            index_dtype,
            SortProgramFactoryCrossCoreDataExchange::CrossCoreDataExchangeSortSlicingStrategy::USE_AS_MANY_CORES);

    const bool is_uint16 = (input_dtype == DataType::UINT16);
    const uint32_t single_core_wt_threshold =
        (is_uint16 && is_row_major) ? SORT_WT_THRESHOLD_UINT16_ROW_MAJOR : SORT_WT_THRESHOLD;

    // Mergesort row engine: eligible stable bf16 cells with UINT32 indices (the dtype the
    // gate in compute_output_specs selects; a preallocated UINT16 index tensor falls back).
    if (mergesort_row_engine_eligible(attributes, tensor_args, w_dim) && index_dtype == DataType::UINT32) {
        return SortProgramFactoryMergesortRowParallel{};
    }

    if (Wt <= single_core_wt_threshold) {
        // Small-Ht wide-tile reroute (#33492 roadmap, routing fix): the single-core factory
        // processes one full tile-row's bitonic tile network per core per loop (all 32 tensor
        // rows of a tile-row ride the SFPU lanes concurrently), so its wall time is
        // ceil(Ht / num_cores) x T_sc(Wt) and the grid idles when Ht is small. The CrossCore
        // factory splits the same tile-row's Wt tiles over ~Wt/2 cores and costs
        // Ht x T_cc(Wt). Measured on p150a (tie-heavy bf16, Tracy device-kernel duration,
        // n=10, spread < 0.5%):
        //   Wt=64: T_sc 1982 us (stable) / 1562 (unstable) vs T_cc 170-186 / 150  (10-12x)
        //   Wt=32: T_sc  805 (stable)                      vs T_cc  97          (8x)
        //   Wt=16: T_sc  314 (stable)                      vs T_cc  66          (5x)
        // Route to CrossCore only where it wins with margin: Ht <= Wt/8 keeps every routed
        // cell >= 2x under the single-core wall at the measured ratios, and leaves large-Ht
        // workloads (where per-core tile-row fanning is already optimal) untouched.
        // STABLE ONLY: the CrossCore unstable exchange resolves ties positionally (raw
        // SFPSWAP index tracking) and the two peers of a spanning tile pair merge with
        // swapped operand order while keeping opposite halves, so on a tie both peers can
        // emit the SAME index — duplicate indices inside tie groups, indices not a
        // permutation (pre-existing CrossCore behavior wherever it serves unstable sorts;
        // measured on silicon at W=512-4096 on plain randn, issue #54043). Unstable cells
        // therefore keep the single-core engine, which moves
        // value+index atomically per SFPSWAP and always emits a valid permutation. The
        // stable comparator (topk_cmp_swap_stable_directional) is operand-order-independent
        // and silicon-verified exact on the rerouted band. BLACKHOLE ONLY: the reroute was
        // silicon-validated on p150a only (same posture as the mergesort row engine above);
        // Wormhole keeps the previous routing untouched. UINT16 stays single-core (the
        // CrossCore factory has no u16<->fp32 conversion path); ROW_MAJOR keeps the
        // existing routing (the CrossCore RM path is unmeasured at these widths). The
        // index-dtype contract in compute_output_specs is deliberately NOT
        // routing-dependent: stable bf16 cells at Wt <= 64 keep UINT32 indices on both
        // factories so the dtype a user sees never depends on Ht.
        const auto tile_height = tensor_args.input_tensor.tensor_spec().tile().get_height();
        const auto& pshape = tensor_args.input_tensor.padded_shape();
        const uint32_t Ht = (pshape[0] * pshape[1] * pshape[2]) / tile_height;
        const bool cross_core_wins_small_ht = attributes.stable && !is_uint16 && !is_row_major && Wt >= 16 &&
                                              Ht <= Wt / 8 && device->arch() == tt::ARCH::BLACKHOLE;
        if (!cross_core_wins_small_ht) {
            // Single-core implementation
            return SortProgramFactorySingleRowSingleCore{};
        }
        // Fall through to the CrossCore eligibility check below.
    }
    // UINT16 support in the CrossCore factory would require Float32 intermediate,
    // peer, and rm_value_output CBs (c_4, c_6, c_8, c_13) plus reader/writer
    // element-wise UInt16↔Float32 conversion loops.  Until that is wired up,
    // route UINT16 inputs above the SingleCore threshold through the MultiCore
    // DRAM factory, which already has both reader and writer UInt16↔Float32
    // conversion paths for TILE and ROW_MAJOR (see
    // SortProgramFactorySingleRowMultiCore and its dataflow kernels).
    if (!is_uint16 && Wt <= total_number_of_tiles_for_hybrid_approach) {
        // Hybrid implementation
        return SortProgramFactoryCrossCoreDataExchange{};
    }
    // DRAM implementation
    return SortProgramFactorySingleRowMultiCore{};
}

void SortDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    const auto input_lshape = input.logical_shape();
    const auto input_pshape = input.padded_shape();

    TT_FATAL(input.buffer() != nullptr, "Operands need to be allocated in buffers on the device. Buffer is null.");
    TT_FATAL(
        input.storage_type() == StorageType::DEVICE,
        "Operation requires input to be on Device. Input storage type: {}",
        static_cast<int>(input.storage_type()));

    TT_FATAL(input_pshape.rank() == 4, "Input shape must be 4D, got {}", input_pshape.rank());

    const int8_t rank = static_cast<int8_t>(input_pshape.rank());
    const int8_t dim = operation_attributes.dim;
    TT_FATAL(
        dim == -1 || dim == rank - 1,
        "Sort device op requires dim to be the last axis (-1 or {}), got {}. "
        "The composite sort() layer must transpose before dispatching.",
        rank - 1,
        dim);

    TT_FATAL(
        input.dtype() == DataType::BFLOAT16 || input.dtype() == DataType::UINT16 || input.dtype() == DataType::FLOAT32,
        "Input tensor data type must be BFLOAT16, UINT16, or FLOAT32, got {}",
        input.dtype());

    const bool is_row_major = (input.layout() == Layout::ROW_MAJOR);

    // UINT16 support: the reader/writer kernels of both the SingleCore and
    // MultiCore factories perform an element-wise UInt16↔Float32 software
    // conversion for both TILE and ROW_MAJOR layouts, so any Wt is accepted.
    // The CrossCore factory does NOT yet include the equivalent conversion;
    // select_program_factory routes UINT16 with Wt > SORT_WT_THRESHOLD to
    // MultiCore to work around that.

    // Width must be a power of two >= 64 regardless of layout: the bitonic
    // engines have no j < Wt partner guard and truncate log2(Wt), so a
    // non-power-of-two width (e.g. 192 — a multiple of 64) silently produces
    // garbage rather than failing. The public ttnn.sort composite always pads
    // the sort dim to the next power of two >= 64 with +/-inf sentinels, so
    // this only rejects direct prim calls.
    // For TILE the relevant dimension is the padded width; for ROW_MAJOR it is
    // the logical width (padding was already applied in pre_sort_transform_tensor).
    const uint32_t checked_w = is_row_major ? input_lshape[-1] : input_pshape[-1];
    TT_FATAL(
        checked_w >= 64 && (checked_w & (checked_w - 1)) == 0,
        "Input shape inner dim {} must be a power of two >= 64. Use ttnn.sort, which pads the sort dimension "
        "with +/-infinity to the next power of two.",
        checked_w);

    // Height constraint: the kernel always works on TILE_HEIGHT (32) row groups.
    // For TILE layout: padded_shape height is tile-aligned by construction.
    // For ROW_MAJOR layout: pre_sort_transform_tensor in sort.cpp pads the H
    //   dimension automatically, so combined_h is always a multiple of 32 here.
    const uint32_t combined_h = input_pshape[0] * input_pshape[1] * input_pshape[2];
    // Empty tensors must not reach a program factory: the mergesort factory
    // derives its active-core count from the row count and would divide by
    // zero; the other factories would build zero-work programs.
    TT_FATAL(
        combined_h > 0,
        "Sort device op requires a non-empty input tensor (shape[0]*shape[1]*shape[2] must be > 0), got shape {}.",
        input_pshape);
    TT_FATAL(
        combined_h % tt::constants::TILE_HEIGHT == 0,
        "Input combined height (shape[0]*shape[1]*shape[2] = {}) must be a multiple of 32.",
        combined_h);

    if (tensor_args.output_tensors.size() == 2) {
        if (tensor_args.output_tensors.at(0).has_value() && tensor_args.output_tensors.at(1).has_value()) {
            const auto output_tensor_shape = tensor_args.output_tensors.at(0)->padded_shape();
            TT_FATAL(
                output_tensor_shape == input_pshape,
                "Output tensor shape must be the same as input tensor shape. Got output tensor shape: {} and input "
                "tensor shape: {}",
                output_tensor_shape,
                input_pshape);
            const auto output_indices_shape = tensor_args.output_tensors.at(1)->padded_shape();
            TT_FATAL(
                output_indices_shape == input_pshape,
                "Output tensor indices shape must be the same as input tensor shape. Got output indices tensor shape: "
                "{} and "
                "input tensor shape: {}",
                output_indices_shape,
                input_pshape);
            TT_FATAL(
                tensor_args.output_tensors.at(0)->dtype() == tensor_args.input_tensor.dtype(),
                "Output values tensor dtype must be the same as input tensor dtype. Got output values tensor dtype: {} "
                "and input tensor dtype: {}",
                tensor_args.output_tensors.at(0)->dtype(),
                tensor_args.input_tensor.dtype());
            TT_FATAL(
                tensor_args.output_tensors.at(1)->dtype() == DataType::UINT16 ||
                    tensor_args.output_tensors.at(1)->dtype() == DataType::UINT32,
                "Output indices tensor dtype must be UINT16 or UINT32. Got output indices tensor dtype: {}",
                tensor_args.output_tensors.at(1)->dtype());
            if (tensor_args.input_tensor.dtype() == DataType::FLOAT32 ||
                tensor_args.input_tensor.dtype() == DataType::UINT16) {
                TT_FATAL(
                    tensor_args.output_tensors.at(1)->dtype() == DataType::UINT32,
                    "Output indices tensor dtype must be UINT32 when input dtype is FLOAT32 or UINT16 "
                    "(fp32_dest_acc_en forces 32-bit index tiles). Got: {}",
                    tensor_args.output_tensors.at(1)->dtype());
            }
        }
    }
}

SortDeviceOperation::spec_return_value_t SortDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensors.size() == 2) {
        if (tensor_args.output_tensors.at(0).has_value() && tensor_args.output_tensors.at(1).has_value()) {
            return {tensor_args.output_tensors[0]->tensor_spec(), tensor_args.output_tensors[1]->tensor_spec()};
        }
    }
    // Create output tensors specs
    auto output_shape = tensor_args.input_tensor.logical_shape();

    // Indices are always stored as unsigned integers.  The topk LLK uses LO16
    // (uint16) or INT32 (uint32) mode in the SFPU to track indices, so the CB
    // format must match.
    //
    // When fp32_dest_acc_en is enabled the DEST registers are 32-bit and topk
    // reads/writes indices via the INT32 path; UINT16 index tiles (2 KB) would
    // not match the 4 KB tiles that mode produces, overrunning the index CB.
    // Force UINT32 indices whenever the sort runs in 32-bit DEST mode, which
    // currently happens for:
    //   • FLOAT32 input (direct fp32 comparison)
    //   • UINT16 input  (uint16 int → fp32 via hardware unpack, exact 0..65535)
    //   • stable BFLOAT16 on the SingleRowSingleCore factory (issue #33492): the fused-key
    //     stable engine tags the value words with the true index and needs the 32-bit DEST; its
    //     index tiles ride the proven UINT32 transport (the u16-index-in-32-bit-DEST combination
    //     has no working pack path — the tile-size rationale above). The width guard uses the
    //     single-core width threshold, NOT the exact factory routing: small-Ht cells inside this
    //     width band may be rerouted to the CrossCore factory for speed (see
    //     select_program_factory), and those keep UINT32 indices too so the index dtype a user
    //     sees never depends on Ht. Wider stable sorts run the comparator on the DM-bound
    //     factories with their ordinary index dtype.
    const bool input_is_fp32 = (tensor_args.input_tensor.dtype() == DataType::FLOAT32);
    const bool input_is_uint16 = (tensor_args.input_tensor.dtype() == DataType::UINT16);
    const bool input_is_row_major = (tensor_args.input_tensor.layout() == Layout::ROW_MAJOR);
    const uint32_t sort_w_dim =
        input_is_row_major ? tensor_args.input_tensor.logical_shape()[3] : tensor_args.input_tensor.padded_shape()[3];
    const uint32_t sort_wt = sort_w_dim / tensor_args.input_tensor.tensor_spec().tile().get_width();
    const bool stable_bf16_single_core =
        attributes.stable && (tensor_args.input_tensor.dtype() == DataType::BFLOAT16) && (sort_wt <= SORT_WT_THRESHOLD);
    // Mergesort row engine (Blackhole, stable bf16, W in {2048, 4096}): fused u16 tags in
    // 32-bit DEST, so its indices are UINT32 for the same pack-path reason. W=2048 is
    // already covered by the single-core gate above; this adds W=4096 on Blackhole.
    const bool stable_bf16_mergesort = mergesort_row_engine_eligible(attributes, tensor_args, sort_w_dim);
    DataType index_dtype = DataType::UINT16;
    if (output_shape[-1] >= std::numeric_limits<uint16_t>::max() || input_is_fp32 || input_is_uint16 ||
        stable_bf16_single_core || stable_bf16_mergesort) {
        index_dtype = DataType::UINT32;
    }

    // Output layout always mirrors the input layout.  For ROW_MAJOR inputs the
    // DRAM multi-core factory (SortProgramFactorySingleRowMultiCore) processes
    // data natively in ROW_MAJOR: each worker tilizes its pair in L1, sorts,
    // then untilizes back, so the DRAM scratch and the output buffer remain RM.
    const Layout out_layout = tensor_args.input_tensor.layout();

    // If the requested output memory config is sharded, the W-padded intermediate
    // tensor's shape may not be compatible with the original shard spec (e.g.
    // shard_width=32 for a padded 64-wide tensor).  Fall back to DRAM interleaved
    // so that the device op always produces a valid tensor spec.  The caller
    // (sort.cpp) is responsible for converting to the user's sharded config after
    // the post-transform slice restores the original W dimension.
    const MemoryConfig effective_mem_cfg = attributes.output_mem_config.is_sharded()
                                               ? MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}
                                               : attributes.output_mem_config;

    auto values_spec = tt::tt_metal::TensorSpec(
        output_shape, TensorLayout(tensor_args.input_tensor.dtype(), PageConfig(out_layout), effective_mem_cfg));
    auto index_spec =
        tt::tt_metal::TensorSpec(output_shape, TensorLayout(index_dtype, PageConfig(out_layout), effective_mem_cfg));

    return {values_spec, index_spec};
}

SortDeviceOperation::tensor_return_value_t SortDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensors.size() == 2) {
        if (tensor_args.output_tensors.at(0).has_value() && tensor_args.output_tensors.at(1).has_value()) {
            return {tensor_args.output_tensors[0].value(), tensor_args.output_tensors[1].value()};
        }
    }
    auto output_specs = compute_output_specs(attributes, tensor_args);
    return {
        create_device_tensor(output_specs[0], tensor_args.input_tensor.device()),  // Value tensor
        create_device_tensor(output_specs[1], tensor_args.input_tensor.device()),  // Index tensor
    };
}
}  // namespace ttnn::prim

namespace ttnn::prim {
ttnn::prim::SortDeviceOperation::tensor_return_value_t sort(
    const Tensor& input_tensor,
    int8_t dim,
    bool descending,
    bool stable,
    const MemoryConfig& output_memory_config,
    const std::vector<std::optional<Tensor>>& output_tensors) {
    using OperationType = ttnn::prim::SortDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{dim, descending, stable, output_memory_config},
        OperationType::tensor_args_t{input_tensor, output_tensors});
}
}  // namespace ttnn::prim
