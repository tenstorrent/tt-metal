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

// ---------------------------------------------------------------------------
// Engine selection
// ---------------------------------------------------------------------------
// The one place that decides which argmax engine runs. Everything downstream
// consumes ArgmaxParams::engine; nothing re-derives the choice.

using ttnn::prim::ArgMaxEngine;

// Correctness gate shared by both accelerated engines: the preconditions under
// which the Blackhole TILE-layout last-dim kernels are defined at all. Mirrors
// -- and must stay in sync with -- the per-engine TT_FATALs in
// ArgMaxDeviceOperation::validate_on_program_cache_miss, which is what catches
// it if the two ever drift. Uses the same hal::get_arch() the validator does,
// so the two cannot disagree about the architecture.
bool accelerated_engines_can_serve(
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
    // Rank 0 (a scalar) is the only rank the accelerated engines cannot take,
    // and this is a correctness gate, not a preference: a rank-0 shape has no
    // reduction axis, normalize_dim(-1, 0) == -1 compares equal to rank - 1
    // just below, and logical_shape[-1] would then index out of bounds.
    //
    // Rank 1 IS served, by the RVV engine. A rank-1 logical shape has no H
    // dim, but that is not a gap in the kernels: both accelerated program
    // factories already fold it to H == 1 explicitly
    // (argmax_rvv_tile_program_factory.cpp h_logical,
    // argmax_sfpu_tile_program_factory.cpp h_logical), which makes a rank-1
    // [V] input produce exactly the compile-time arguments of a [1, 1, 1, V]
    // input -- the H == 1 shape the RVV battery already covers. The scalar
    // readers cannot produce a max-value output, so demoting rank 1 here would
    // turn `argmax(rank1, dim=-1, maxval_tensor=...)` into a hard error; the
    // RVV engine served exactly that call before engine selection moved
    // in-tree. Rank 1 is kept away from the SFPU engine in
    // select_argmax_engine, not here.
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
    // a prealloc whose logical shape is not the reduction output shape would
    // leave results unwritten or write past the tensor's page count. The scalar
    // readers tolerate it, so demote instead of turning a call that used to
    // work into an error.
    if (optional_output_tensor.has_value() &&
        optional_output_tensor->logical_shape() != ttnn::Shape(ttnn::prim::get_output_shape(input, dim, keepdim))) {
        return false;
    }
    return true;
}

// Valid rows per tile-row pass -- the H dim. This, not the reduction width, is
// what separates the two accelerated engines (see select_argmax_engine).
//
// Rank 1 has no second-to-last dim and answers 1: its single logical row sits
// in row 0 of one tile-row whose other 31 rows are padding. That is the same
// value both accelerated program factories bind as `h_logical` and pass to
// their kernels as `logical_height` -- the bound of the per-pass row loop --
// so the routing decision and the kernels agree on what rank 1 means.
uint32_t argmax_rows_per_tile_row(const Tensor& input) {
    const auto& logical_shape = input.logical_shape();
    const int32_t rank = static_cast<int32_t>(logical_shape.rank());
    return rank >= 2 ? logical_shape[rank - 2] : 1u;
}

// Smallest H at which the SFPU engine is MEASURED to win. See the table below.
constexpr uint32_t kSfpuMinRows = 8;

// Blackhole P150 device-kernel medians measured for this change (microseconds;
// last-dim argmax over V columns of B = H rows):
//
//   V         B    incumbent      RVV     SFPU (multicore)
//   262144    1        9128       462     loses -- 32 lanes for 1 valid row
//    32768    8        5253       192      35
//   131072   32       77762      2712     136
//   262144   32      155561      5275     190
//
// Both accelerated engines beat the scalar readers by 10-30x at every measured
// point, so the only real decision is RVV vs SFPU. The SFPU recipe reduces all
// 32 rows of a tile-row in one lane-parallel pass (~802 cycles per tile,
// whether 1 or 32 of those rows are real), while the RVV scan costs 63-100
// cycles per tile PER ROW. The discriminator is therefore the number of valid
// rows per tile-row -- the H dim: at H == 1 the SFPU pays for 32 lanes to
// serve one row and loses, at H >= 8 it wins decisively.
//
// 1 < H < 8 IS NOT MEASURED. The boundary sits at the smallest measured SFPU
// win rather than at H == 2, so the whole unmeasured band goes to RVV -- the
// engine that is bit-identical to the scalar readers. Being wrong there costs
// throughput on shapes nobody benchmarked; being wrong the other way would
// silently change special-value results on them.
ArgMaxEngine select_argmax_engine(
    const Tensor& input,
    const std::optional<int>& dim,
    bool keepdim,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    bool exact_special_values) {
    if (!accelerated_engines_can_serve(input, dim, keepdim, output_memory_config, optional_output_tensor)) {
        return ArgMaxEngine::Incumbent;
    }

    // The RVV engine is single core and currently hard-wired to core (0, 0):
    // an explicit grid that excludes it would make the engine throw, so such a
    // call is not RVV's to take. The SFPU engine honours any grid.
    const bool rvv_can_serve = !sub_core_grids.has_value() || sub_core_grids->contains(CoreCoord{0, 0});
    // The SFPU engine's special-value divergence is documented and measured,
    // but it is still a divergence: a caller that asked for the scalar
    // readers' exact behaviour must never be routed to it.
    //
    // Rank 1 is excluded for a different reason. It reaches the accelerated
    // engines because the RVV engine served it before (see
    // accelerated_engines_can_serve), and nothing has ever run it on the SFPU
    // kernels -- not a test, not a measurement. It cannot reach the H >= 8
    // branch below (rank 1 is H == 1 by construction), so the only way it
    // could land on the SFPU is the last-resort branch, i.e. an explicit
    // sub_core_grids that excludes core (0, 0) and so puts RVV out of reach.
    // At H == 1 the SFPU is the measured LOSER anyway (see the table above),
    // so there is nothing to win by making an unexercised shape class the
    // first caller of that combination: rank 1 falls back to the incumbent
    // instead. That is not a regression -- the old use_rvv API raised on a
    // grid excluding (0, 0) rather than serving it.
    const int32_t rank = static_cast<int32_t>(input.logical_shape().rank());
    const bool sfpu_can_serve = !exact_special_values && rank >= 2;

    if (sfpu_can_serve && argmax_rows_per_tile_row(input) >= kSfpuMinRows) {
        return ArgMaxEngine::Sfpu;
    }
    if (rvv_can_serve) {
        return ArgMaxEngine::Rvv;
    }
    if (sfpu_can_serve) {
        return ArgMaxEngine::Sfpu;
    }
    return ArgMaxEngine::Incumbent;
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
// entries (argmax_force.hpp). `forced_engine` is std::nullopt for every public
// call: the engine is then chosen by select_argmax_engine and nothing else in
// the op revisits that decision.
Tensor argmax_impl(
    const Tensor& input_tensor,
    const std::optional<int>& dim,
    bool keepdim,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor,
    std::optional<Tensor> optional_maxval_tensor,
    bool exact_special_values,
    std::optional<ArgMaxEngine> forced_engine) {
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

    // Engine choice happens here, once, and only after the dim has been range
    // checked (select_argmax_engine normalizes it against the rank).
    const ArgMaxEngine engine = forced_engine.has_value() ? forced_engine.value()
                                                          : select_argmax_engine(
                                                                input_tensor,
                                                                dim,
                                                                keepdim,
                                                                sub_core_grids,
                                                                output_memory_config,
                                                                optional_output_tensor,
                                                                exact_special_values);

    // The maxval contract check must precede the zero-volume and rank-0 early
    // returns below: a scalar or empty-tensor call that supplied a
    // maxval_tensor must fail loudly rather than silently no-op through the
    // host-side fallback, which would leave that tensor untouched (stale).
    TT_FATAL(
        !optional_maxval_tensor.has_value() || engine != ArgMaxEngine::Incumbent,
        "argmax: the max-value output tensor (maxval_tensor) is only produced by the accelerated engines, and this "
        "call was routed to the scalar reader path. Those engines require a Blackhole device, a BFLOAT16 TILE-layout "
        "input of rank >= 1 in INTERLEAVED memory, an explicit last-dim reduction, standard 32x32 tiles, a reduction "
        "dim that is a multiple of 32, an INTERLEAVED output, and (if a preallocated output tensor is supplied) that "
        "its logical shape is the reduction output shape. A rank-1 input additionally needs a sub_core_grids that "
        "contains core (0, 0), or none at all, because only the single-core RVV engine serves rank 1. Drop "
        "maxval_tensor, or use ttnn.max separately.");
    if (engine != ArgMaxEngine::Incumbent) {
        TT_FATAL(
            rank > 0 && input_tensor.logical_volume() > 0,
            "argmax: the accelerated engines support only non-empty tensors of rank >= 1 (got rank {}, logical "
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

    // Accelerated engines: TILE-layout last-dim argmax (Blackhole). Both take
    // TILE input DIRECTLY -- no to_layout / untilize hop -- and optionally
    // return the max values alongside the indices. Rvv scans on the pack
    // RISC's vector unit (single core, the engine at H == 1); Sfpu reduces all
    // 32 rows of each tile-row in one lane-parallel pass and runs multicore
    // (flat in H, the batch-shape engine). Eligibility is re-checked by the
    // device op.
    if (engine != ArgMaxEngine::Incumbent) {
        // The reader kernel derives its output paging from the preallocated
        // tensor, so a wrong logical shape means unwritten results or writes
        // past the tensor's page count -- reject it up front. Automatic
        // dispatch demotes such a call to the incumbent instead of reaching
        // here; a forced engine has to be told no.
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
            engine,
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
        /*forced_engine=*/std::nullopt);
}

namespace operations::reduction::detail {

// Verification-only; see argmax_force.hpp for why these exist and why they do
// not fall back. exact_special_values is irrelevant once the engine is pinned.

Tensor argmax_force_incumbent(
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
        prim::ArgMaxEngine::Incumbent);
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
        prim::ArgMaxEngine::Rvv);
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
        prim::ArgMaxEngine::Sfpu);
}

}  // namespace operations::reduction::detail

}  // namespace ttnn
