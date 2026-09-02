// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "device/argmax_device_operation.hpp"
#include "device/argmax_nc_device_operation.hpp"
#include "device/argmax_utils.hpp"
#include "ttnn/operations/reduction/argmax/argmax.hpp"
#include "argmax_force.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/core/to_layout/to_layout_op.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

#include <tt-metalium/hal.hpp>

#include <cstdint>
#include <utility>

namespace ttnn {

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;

namespace {

bool should_row_major_h_via_tile(
    const Tensor& input, const std::optional<int>& dim, const MemoryConfig& output_memory_config) {
    if (!dim.has_value() || input.layout() != Layout::ROW_MAJOR) {
        return false;
    }
    const int32_t rank = static_cast<int32_t>(input.logical_shape().rank());
    if (rank < 2) {
        return false;
    }
    const int32_t normalized_dim = dim.value() < 0 ? dim.value() + rank : dim.value();
    if (normalized_dim != rank - 2) {
        return false;
    }
    if (input.dtype() != DataType::BFLOAT16 && input.dtype() != DataType::FLOAT32) {
        return false;
    }
    if (input.memory_config().memory_layout() != tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
        return false;
    }
    if (output_memory_config.memory_layout() != tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
        return false;
    }
    return true;
}

// Returns true if we should dispatch the reduction to the register-based NC
// path (compute-kernel + DST accumulation). The NC path supports reducing
// along any non-HW dim (i.e., dim < rank - 2) and assumes BFLOAT16 / FLOAT32
// values.
bool should_use_nc_path(const Tensor& input, const std::optional<int>& dim, const MemoryConfig& output_memory_config) {
    if (!dim.has_value()) {
        return false;
    }
    const auto rank = static_cast<int32_t>(input.logical_shape().rank());
    if (rank < 3) {
        // Rank 2 only has H and W; nothing is "non-HW".
        return false;
    }
    const int32_t normalized_dim = dim.value() < 0 ? dim.value() + rank : dim.value();
    if (normalized_dim < 0 || normalized_dim >= rank - 2) {
        return false;
    }
    const auto dtype = input.dtype();
    if (dtype != tt::tt_metal::DataType::BFLOAT16 && dtype != tt::tt_metal::DataType::FLOAT32) {
        return false;
    }
    if (input.memory_config().memory_layout() != tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
        return false;
    }
    if (output_memory_config.memory_layout() != tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
        return false;
    }
    return true;
}

// Run the register-based argmax for a non-HW dim. Returns a ROW_MAJOR UINT32
// tensor with the user-visible logical output shape.
Tensor run_argmax_nc(
    const Tensor& input_tensor,
    int dim,
    bool keepdim,
    const MemoryConfig& output_memory_config,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using tt::tt_metal::Layout;

    // 1) Ensure the input is in TILE layout for the compute kernel.
    Tensor tiled_input = input_tensor;
    if (tiled_input.layout() != Layout::TILE) {
        tiled_input = ttnn::to_layout(tiled_input, Layout::TILE);
    }

    // 2) Run the NC device op. Returns a TILE UINT32 tensor with the reduced
    //    dim's logical size collapsed to 1 (keepdim=True semantics).
    auto compute_kernel_config = init_device_compute_kernel_config(
        tiled_input.device()->arch(), std::nullopt, tt::tt_metal::MathFidelity::HiFi4);
    Tensor tile_out = ttnn::prim::argmax_nc(
        tiled_input,
        /*dim=*/dim,
        /*preallocated_output=*/std::nullopt,
        /*output_mem_config=*/output_memory_config,
        compute_kernel_config,
        sub_core_grids);

    // 3) Convert the TILE UINT32 output back to ROW_MAJOR UINT32.
    Tensor row_major_out = ttnn::to_layout(tile_out, Layout::ROW_MAJOR);

    // 4) Apply keepdim semantics: if keepdim=false, remove the reduced dim.
    if (!keepdim) {
        const auto& logical_shape = row_major_out.logical_shape();
        const int32_t rank = static_cast<int32_t>(logical_shape.rank());
        const int32_t normalized_dim = dim < 0 ? dim + rank : dim;
        ttsl::SmallVector<uint32_t> new_shape;
        new_shape.reserve(rank - 1);
        for (int32_t i = 0; i < rank; ++i) {
            if (i == normalized_dim) {
                continue;
            }
            new_shape.push_back(logical_shape[i]);
        }
        row_major_out = ttnn::reshape(row_major_out, ttnn::Shape(new_shape));
    }

    return row_major_out;
}

// Path selection: the one place that decides which argmax path runs.
// Everything downstream consumes ArgmaxParams::path.

using ttnn::prim::ArgMaxPath;

// Correctness gate shared by both accelerated paths: the preconditions under
// which the Blackhole TILE-layout last-dim kernels are defined at all. Mirrors
// the per-path TT_FATALs in ArgMaxDeviceOperation::validate_on_program_cache_miss
// and uses the same hal::get_arch(), so the two cannot drift silently.
bool accelerated_paths_can_serve(
    const Tensor& input,
    const std::optional<int>& dim,
    bool keepdim,
    const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    if (tt::tt_metal::hal::get_arch() != tt::ARCH::BLACKHOLE) {
        return false;
    }
    if (input.layout() != Layout::TILE || input.dtype() != DataType::BFLOAT16) {
        return false;
    }
    if (!dim.has_value()) {
        return false;
    }
    const auto& logical_shape = input.logical_shape();
    const int32_t rank = static_cast<int32_t>(logical_shape.rank());
    if (input.logical_volume() == 0) {
        return false;
    }
    // Rank 0 is a correctness gate, not a preference: its logical_volume() is 1
    // so the check above admits it, normalize_dim(-1, 0) == -1 compares equal
    // to rank - 1 just below so the dim check admits it too, and
    // logical_shape[-1] would then index out of bounds. Rank 1 IS served, by
    // the RVV path: both accelerated factories fold it to h_logical == 1, so a
    // [V] input compiles exactly like a [1, 1, 1, V] one, and only these paths
    // can fill a max-value output for it. select_argmax_path, not this gate,
    // keeps rank 1 off the SFPU path.
    if (rank < 1) {
        return false;
    }
    if (ttnn::prim::normalize_dim(static_cast<int32_t>(dim.value()), rank) != rank - 1) {
        return false;
    }
    const auto& tile = input.tensor_spec().tile();
    if (tile.get_width() != 32 || tile.get_height() != 32) {
        return false;
    }
    // No width padding: the reduction dim must fill whole tiles.
    if (logical_shape[-1] % tile.get_width() != 0) {
        return false;
    }
    if (input.memory_config().memory_layout() != tt::tt_metal::TensorMemoryLayout::INTERLEAVED ||
        output_memory_config.memory_layout() != tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
        return false;
    }
    // The accelerated readers page their writes off the preallocated output, so
    // a prealloc whose logical shape is not the reduction output shape leaves
    // results unwritten or writes past its page count. This demotes rather than
    // raising: the scalar readers mis-handle the same shapes -- they page off
    // the reduction output shape and ignore the prealloc's own, silently
    // short-writing when its TRAILING dim differs (Blackhole, [1, 1, 32, 2048],
    // keepdim = false, [1, 1, 32, 1] prealloc: element 0 correct, the other 31
    // left at zero) -- while three reshape-equivalent preallocs do return
    // correct results today, which raising here would turn into hard errors on
    // this arch/layout/dtype/dim alone. Making the scalar readers page off the
    // prealloc they were given is the actual fix.
    if (optional_output_tensor.has_value() &&
        optional_output_tensor->logical_shape() != ttnn::Shape(ttnn::prim::get_output_shape(input, dim, keepdim))) {
        return false;
    }
    return true;
}

// Valid rows per tile-row pass -- the H dim, which is what separates the two
// accelerated paths (see select_argmax_path). Rank 1 has no second-to-last dim
// and answers 1, the same value both accelerated factories bind as `h_logical`,
// so routing and kernels agree on what rank 1 means.
uint32_t argmax_rows_per_tile_row(const Tensor& input) {
    const auto& logical_shape = input.logical_shape();
    const int32_t rank = static_cast<int32_t>(logical_shape.rank());
    return rank >= 2 ? logical_shape[rank - 2] : 1u;
}

// Smallest H at which the SFPU path is measured to win. H, not the reduction
// width, is the discriminator: the SFPU pass reduces all 32 rows of a tile-row
// at once and is therefore flat in H, while the RVV scan revisits each tile
// once per valid row, so below this boundary RVV wins -- by 3-14x at the low
// core counts where the paths are actually distinguishable. The boundary is
// mildly V-dependent, so no H-only rule is right everywhere; 32 keeps every
// unambiguous case on RVV.
//
// Compare the paths only at equal core counts, and never only at the top of the
// grid: past their knees both converge on a shared per-core program dispatch
// floor, so numbers taken with the whole grid measure that floor rather than
// the paths. The per-core measurements, the methodology and the full tables are
// printed by (and documented in)
// tests/ttnn/unit_tests/operations/reduce/test_argmax_path_crossover_bench.py.
constexpr uint32_t kSfpuMinRows = 32;

ArgMaxPath select_argmax_path(
    const Tensor& input,
    const std::optional<int>& dim,
    bool keepdim,
    const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    bool exact_special_values) {
    if (!accelerated_paths_can_serve(input, dim, keepdim, output_memory_config, optional_output_tensor)) {
        return ArgMaxPath::ScalarReader;
    }

    // Both accelerated paths split the reduction dim's tiles across cores and
    // honour any sub_core_grids the caller supplies, so the grid does not
    // constrain the choice -- it is purely the H comparison below. The SFPU
    // path diverges from the scalar readers on special values, so a caller that
    // asked for their exact behaviour must never be routed to it. The
    // `rank >= 2` term is redundant today (rank 1 answers H == 1, which cannot
    // clear kSfpuMinRows) and states the intent: rank 1 has never run on the
    // SFPU kernels, and at H == 1 the SFPU is the measured loser.
    const int32_t rank = static_cast<int32_t>(input.logical_shape().rank());
    const bool sfpu_can_serve = !exact_special_values && rank >= 2;

    if (sfpu_can_serve && argmax_rows_per_tile_row(input) >= kSfpuMinRows) {
        return ArgMaxPath::Sfpu;
    }
    // Everything else the preconditions admit goes to RVV: it serves every
    // rank >= 1 and every grid, and is bit-identical to the scalar readers.
    return ArgMaxPath::Rvv;
}

}  // namespace

static Tensor zero_volume_argmax(
    const Tensor& input_tensor,
    const std::optional<int>& dim,
    bool keepdim,
    const MemoryConfig& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    auto output_shape = ttnn::Shape(ttnn::prim::get_output_shape(input_tensor, dim, keepdim));
    if (!optional_output_tensor.has_value()) {
        return ttnn::full(
            output_shape,
            0,  // fill_value doesn't matter for zero-volume tensor.
            tt::tt_metal::DataType::UINT32,
            input_tensor.layout(),
            *input_tensor.device(),
            memory_config);
    }

    Tensor& preallocated_tensor = optional_output_tensor.value();
    TT_FATAL(
        preallocated_tensor.logical_shape() == output_shape,
        "Preallocated output tensor has incorrect shape! Got : {}, expected: {}",
        preallocated_tensor.logical_shape(),
        output_shape);

    // Creating result tensor on host and copying to device (there is no direct way to write
    // to a device tensor with a scalar value).
    // Unspecified contents are fine here because the tensor is 0-volume (i.e., it has no elements).
    const tt::tt_metal::TensorSpec& tensor_spec = preallocated_tensor.tensor_spec();
    Tensor host_tensor(tt::tt_metal::HostTensor::allocate_for_overwrite(tensor_spec));
    copy_to_device(host_tensor, preallocated_tensor);

    return preallocated_tensor;
}

namespace {

// The whole of ttnn::argmax, plus a hook for the verification-only forced
// entries (argmax_force.hpp). `forced_path` is std::nullopt for every public
// call.
Tensor argmax_impl(
    const Tensor& input_tensor,
    const std::optional<int>& dim,
    bool keepdim,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor,
    std::optional<Tensor> optional_maxval_tensor,
    bool exact_special_values,
    std::optional<ArgMaxPath> forced_path) {
    auto output_memory_config = memory_config.value_or(input_tensor.memory_config());

    TT_FATAL(is_device_tensor(input_tensor), "Input tensor must be on device");
    TT_FATAL(
        !optional_output_tensor.has_value() || is_device_tensor(optional_output_tensor.value()),
        "Preallocated output tensor must be on device");

    const auto& input_shape = input_tensor.logical_shape();
    const auto rank = input_shape.size();
    if (dim.has_value()) {
        if (rank > 0) {
            const int32_t r = static_cast<int32_t>(rank);
            TT_FATAL(
                dim.value() >= -r && dim.value() < r,
                "argmax: Dimension out of range (expected to be in range of [{}, {}], but got {})",
                -r,
                r - 1,
                dim.value());
        } else {
            // Rank 0 (scalar): only the virtual axis 0 / -1 is valid.
            TT_FATAL(
                dim.value() == 0 || dim.value() == -1,
                "argmax: Dimension out of range for scalar tensor (expected 0 or -1, but got {})",
                dim.value());
        }
    }

    // Chosen once, after the dim range check above (select_argmax_path
    // normalizes dim against the rank).
    const ArgMaxPath path =
        forced_path.has_value()
            ? forced_path.value()
            : select_argmax_path(
                  input_tensor, dim, keepdim, output_memory_config, optional_output_tensor, exact_special_values);

    // Must precede the zero-volume and rank-0 early returns below: a scalar or
    // empty-tensor call that supplied a maxval_tensor has to fail loudly rather
    // than no-op through the host-side fallback and leave it stale.
    TT_FATAL(
        !optional_maxval_tensor.has_value() || path != ArgMaxPath::ScalarReader,
        "argmax: the max-value output tensor (maxval_tensor) is only produced by the accelerated paths, and this "
        "call was routed to the scalar reader path. Those paths require a Blackhole device, a BFLOAT16 TILE-layout "
        "input of rank >= 1 in INTERLEAVED memory, an explicit last-dim reduction, standard 32x32 tiles, a reduction "
        "dim that is a multiple of 32, an INTERLEAVED output, and (if a preallocated output tensor is supplied) that "
        "its logical shape is the reduction output shape. Drop maxval_tensor, or use ttnn.max separately.");
    if (path != ArgMaxPath::ScalarReader) {
        TT_FATAL(
            rank > 0 && input_tensor.logical_volume() > 0,
            "argmax: the accelerated paths support only non-empty tensors of rank >= 1 (got rank {}, logical "
            "volume {})",
            rank,
            input_tensor.logical_volume());
    }

    if (input_tensor.logical_volume() == 0) [[unlikely]] {
        return zero_volume_argmax(input_tensor, dim, keepdim, output_memory_config, optional_output_tensor);
    }

    if (rank == 0) [[unlikely]] {
        if (!optional_output_tensor.has_value()) {
            return full(
                input_shape,
                /*fill_value=*/0,
                tt::tt_metal::DataType::UINT32,
                input_tensor.layout(),
                *input_tensor.device(),
                output_memory_config);
        }

        Tensor& preallocated_tensor = optional_output_tensor.value();
        TT_FATAL(
            preallocated_tensor.logical_shape() == input_shape,
            "Preallocated output tensor has incorrect shape! Got : {}, expected: {}",
            preallocated_tensor.logical_shape(),
            input_shape);
        // Creating result tensor on host and copying to device (there is no direct way to write
        // to a device tensor with a scalar value).
        const tt::tt_metal::TensorSpec& preallocated_spec = preallocated_tensor.tensor_spec();
        TT_FATAL(
            preallocated_spec.data_type() == DataType::UINT32,
            "Preallocated output tensor must be UINT32 for rank 0 input tensor");
        // Although we only need to store one value, have to account for extra padding
        // in possible tile layout. So host buffer size needs to match device buffer size.
        auto result_vec = std::vector<uint32_t>(
            preallocated_spec.physical_shape().height() * preallocated_spec.physical_shape().width(), 0);
        Tensor host_indices(
            tt::tt_metal::HostBuffer(std::move(result_vec)), input_shape, DataType::UINT32, preallocated_spec.layout());
        copy_to_device(host_indices, preallocated_tensor);

        return preallocated_tensor;
    }

    // Accelerated paths (see ArgMaxPath): Blackhole TILE-layout last-dim
    // argmax, taking TILE input directly -- no to_layout / untilize hop -- and
    // optionally returning the max values. Eligibility is re-checked by the
    // device op.
    if (path != ArgMaxPath::ScalarReader) {
        // Automatic dispatch demotes a wrong-shaped prealloc to the scalar
        // readers before reaching here (see accelerated_paths_can_serve); a
        // forced path has to be told no.
        if (optional_output_tensor.has_value()) {
            const auto expected_shape = ttnn::Shape(ttnn::prim::get_output_shape(input_tensor, dim, keepdim));
            TT_FATAL(
                optional_output_tensor->logical_shape() == expected_shape,
                "argmax: preallocated output tensor has shape {}, expected the reduction output shape {}",
                optional_output_tensor->logical_shape(),
                expected_shape);
        }
        return prim::argmax(
            input_tensor,
            tt::tt_metal::DataType::UINT32,
            dim,
            keepdim,
            sub_core_grids,
            output_memory_config,
            std::move(optional_output_tensor),
            path,
            std::move(optional_maxval_tensor));
    }

    // Register-based NC path for reductions along any non-HW dimension.
    // Uses DST accumulation (similar to fast_reduce_nc). Supports sub_core_grids.
    if (should_use_nc_path(input_tensor, dim, output_memory_config)) {
        Tensor nc_result = run_argmax_nc(input_tensor, dim.value(), keepdim, output_memory_config, sub_core_grids);
        if (optional_output_tensor.has_value()) {
            // nc_result is already on device; copy_to_device is host -> device only.
            ttnn::copy(nc_result, optional_output_tensor.value());
            return optional_output_tensor.value();
        }
        return nc_result;
    }

    if (should_row_major_h_via_tile(input_tensor, dim, output_memory_config)) {
        const Tensor tiled_input = ttnn::to_layout(input_tensor, Layout::TILE);
        return prim::argmax(
            tiled_input,
            DataType::UINT32,
            dim,
            keepdim,
            sub_core_grids,
            output_memory_config,
            std::move(optional_output_tensor));
    }

    return prim::argmax(
        input_tensor,
        tt::tt_metal::DataType::UINT32,
        dim,
        keepdim,
        sub_core_grids,
        output_memory_config,
        std::move(optional_output_tensor));
}

}  // namespace

Tensor argmax(
    const Tensor& input_tensor,
    const std::optional<int>& dim,
    bool keepdim,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor,
    bool exact_special_values,
    std::optional<Tensor> optional_maxval_tensor) {
    return argmax_impl(
        input_tensor,
        dim,
        keepdim,
        sub_core_grids,
        memory_config,
        std::move(optional_output_tensor),
        std::move(optional_maxval_tensor),
        exact_special_values,
        /*forced_path=*/std::nullopt);
}

namespace operations::reduction::detail {

// Verification-only; see argmax_force.hpp for why these exist and why they do
// not fall back. exact_special_values is irrelevant once the path is pinned.

Tensor argmax_force_scalar_reader(
    const Tensor& input_tensor,
    const std::optional<int>& dim,
    bool keepdim,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor,
    std::optional<Tensor> optional_maxval_tensor) {
    return argmax_impl(
        input_tensor,
        dim,
        keepdim,
        sub_core_grids,
        memory_config,
        std::move(optional_output_tensor),
        std::move(optional_maxval_tensor),
        /*exact_special_values=*/false,
        prim::ArgMaxPath::ScalarReader);
}

Tensor argmax_force_rvv(
    const Tensor& input_tensor,
    const std::optional<int>& dim,
    bool keepdim,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor,
    std::optional<Tensor> optional_maxval_tensor) {
    return argmax_impl(
        input_tensor,
        dim,
        keepdim,
        sub_core_grids,
        memory_config,
        std::move(optional_output_tensor),
        std::move(optional_maxval_tensor),
        /*exact_special_values=*/false,
        prim::ArgMaxPath::Rvv);
}

Tensor argmax_force_sfpu(
    const Tensor& input_tensor,
    const std::optional<int>& dim,
    bool keepdim,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor,
    std::optional<Tensor> optional_maxval_tensor) {
    return argmax_impl(
        input_tensor,
        dim,
        keepdim,
        sub_core_grids,
        memory_config,
        std::move(optional_output_tensor),
        std::move(optional_maxval_tensor),
        /*exact_special_values=*/false,
        prim::ArgMaxPath::Sfpu);
}

}  // namespace operations::reduction::detail

}  // namespace ttnn
