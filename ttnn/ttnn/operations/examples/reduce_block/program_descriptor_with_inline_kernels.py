# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""A small ProgramDescriptor example driven entirely by the host reduce planner.

Both a standalone reduction and a cross-CB accumulating reduction use the same
compute source. The example places its own iteration count first, then asks the
reduce plan to append a call count followed by complete, independently decodable
calls. It issues those calls back to back; a fused kernel may instead place
unrelated work between them.

The legacy ``variant``, ``policy``, and reload-shaped arguments remain accepted
while callers migrate.  They no longer select compute implementations: shape,
tensor specs, reduction semantics, hardware, and the input-CB L1 constraint are
the inputs to the C++ planner.
"""

import ttnn

TILE = 32

# CB assignment (semantic names). Each sequence input receives a distinct ID.
CB_IN = 0
CB_AUXILIARY = 1
CB_ACCUMULATOR = 2
CB_OUT = 16

# Automatic is the only advertised mode. The older labels remain accepted by
# create_program_descriptor while benchmark callers migrate; they are aliases
# and cannot override a planner decision.
VARIANTS = ("automatic",)
_LEGACY_VARIANTS = ("reduce_tile", "accumulate_via_add", "accumulate_via_add_inline", "dispatch")
BASELINE = "automatic"
DIMS = ("row", "col", "scalar")
DTYPES = ("fp32", "bf16")
POLICIES = ("bulk", "stream", "wait_upfront", "no_wait")
RELOADS = ("fold", "copy_pairs", "copy_uniform", "copy_sfpu", "copy_zero")
WITHIN_TILE = ("collapse", "skip")

_PLANNER = ttnn.reduce_planner
_REDUCE_DIM = {
    "row": _PLANNER.ReduceDimension.ROW,
    "col": _PLANNER.ReduceDimension.COLUMN,
    "scalar": _PLANNER.ReduceDimension.SCALAR,
}


def dispatch_min(dim):
    """Historical crossover, retained for report formatting only."""
    return {"row": 4, "col": 8, "scalar": 8}[dim]


def reduced_count(dim, Ht, Wt):
    if dim == "row":
        return Wt
    if dim == "col":
        return Ht
    return Ht * Wt


def input_shape(Ht, Wt, NC=1):
    return (NC * Ht * TILE, Wt * TILE)


def output_shape(dim, Ht, Wt, NC=1):
    if dim == "row":
        return (NC * Ht * TILE, TILE)
    if dim == "col":
        return (TILE, NC * Wt * TILE)
    if dim == "scalar":
        return (NC * TILE, TILE)
    raise ValueError(f"dim must be one of {DIMS}, got {dim!r}")


def out_tile_count(dim, Ht, Wt, NC=1):
    if dim == "row":
        return NC * Ht
    if dim == "col":
        return NC * Wt
    return NC


def elements_reduced(dim, Ht, Wt):
    if dim == "row":
        return Wt * TILE
    if dim == "col":
        return Ht * TILE
    return Ht * Wt * TILE * TILE


# The only compute source in this module. Each ReduceCallArgs is data only: this
# kernel owns startup, CB lifetime, and call placement.
_PLANNED_REDUCE_KERNEL = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"

namespace {
constexpr uint32_t kernel_iters = get_compile_time_arg_val(0);
constexpr uint32_t reduce_args_offset = 1;
constexpr uint32_t num_calls = get_compile_time_arg_val(reduce_args_offset);
constexpr uint32_t first_call_args_offset =
    reduce_args_offset + ttnn::kernel_lib::reduce_plan_args::call_count_word_count;
static_assert(num_calls > 0, "A reduction plan must contain at least one call");
static_assert(kernel_iters > 0, "The kernel iteration count must be positive");

template <uint32_t CALL_INDEX>
using CallAt = ttnn::kernel_lib::ReduceCallAtT<first_call_args_offset, CALL_INDEX>;

using LastCall = CallAt<num_calls - 1>;

template <typename Call>
constexpr uint32_t input_tile_count() {
    constexpr uint32_t row_pitch = Call::row_stride == 0 ? Call::columns : Call::row_stride;
    return Call::rows * row_pitch * Call::batches;
}

template <typename Call>
constexpr uint32_t output_tile_count() {
    if constexpr (Call::reduce_dim == ckernel::ReduceDim::REDUCE_ROW) {
        return Call::rows * Call::batches;
    } else if constexpr (Call::reduce_dim == ckernel::ReduceDim::REDUCE_COL) {
        return Call::columns * Call::batches;
    } else {
        static_assert(Call::reduce_dim == ckernel::ReduceDim::REDUCE_SCALAR, "Unknown reduction dimension");
        return Call::batches;
    }
}

template <typename Call>
ALWI void arm_input() {
    DataflowBuffer input(Call::input_cb_id);
    input.reserve_back(input_tile_count<Call>());
    input.push_back(input_tile_count<Call>());
}

template <uint32_t CALL_INDEX = 0>
ALWI void arm_persistent_inputs() {
    if constexpr (CALL_INDEX < num_calls) {
        using Call = CallAt<CALL_INDEX>;
        if constexpr (
            Call::input_policy == compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop ||
            Call::input_policy == compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop) {
            arm_input<Call>();
        }
        arm_persistent_inputs<CALL_INDEX + 1>();
    }
}

template <typename Call>
ALWI void issue_call() {
    static_assert(Call::path == ttnn::kernel_lib::ReducePath::Tiled,
                  "This tiled example cannot execute a dense row-major plan");

    if constexpr (
        Call::input_policy != compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop &&
        Call::input_policy != compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop) {
        arm_input<Call>();
    }

    DataflowBuffer auxiliary(Call::auxiliary_cb_id);
    auxiliary.wait_front(Call::auxiliary_tile_count);

    constexpr auto shape =
        compute_kernel_lib::ReduceInputBlockShape::of(Call::rows, Call::columns, Call::batches);
    constexpr auto layout =
        Call::row_stride == 0
            ? compute_kernel_lib::ReduceInputMemoryLayout::contiguous()
            : compute_kernel_lib::ReduceInputMemoryLayout::with_row_stride(Call::row_stride);
    constexpr auto chunk =
        compute_kernel_lib::ReduceInputChunk::of(Call::reduce_axis_chunk_tiles, Call::output_chunk_tiles);

    auto post_scale = [](uint32_t dst_index) {
        if constexpr (Call::post_scale_bits != ttnn::kernel_lib::reduce_plan_args::float_one_bits) {
            constexpr DataFormat input_format = static_cast<DataFormat>(unpack_src_format[Call::input_cb_id]);
            compute_kernel_lib::detail::reduce_post_mul_tile<input_format>(dst_index, Call::post_scale_bits);
        }
    };

    if constexpr (Call::accumulation_mode == ttnn::kernel_lib::ReduceAccumulationMode::None) {
        compute_kernel_lib::reduce<
            Call::reduce_type,
            Call::reduce_dim,
            Call::input_cb_id,
            Call::auxiliary_cb_id,
            Call::output_cb_id,
            Call::input_policy,
            Call::reconfig_mode,
            Call::fp32_mode,
            Call::algorithm,
            Call::within_tile,
            Call::reduce_factor>(
                shape,
                layout,
                compute_kernel_lib::NoAccumulation{},
                post_scale,
                Call::partial_mode,
                chunk);
    } else if constexpr (Call::accumulation_mode == ttnn::kernel_lib::ReduceAccumulationMode::Final) {
        compute_kernel_lib::reduce<
            Call::reduce_type,
            Call::reduce_dim,
            Call::input_cb_id,
            Call::auxiliary_cb_id,
            Call::output_cb_id,
            Call::input_policy,
            Call::reconfig_mode,
            Call::fp32_mode,
            Call::algorithm,
            Call::within_tile,
            Call::reduce_factor>(
                shape,
                layout,
                compute_kernel_lib::Accumulate::at_last(Call::accumulator_cb_id, Call::accumulation_index)
                    .with_reload(Call::reload_mode),
                post_scale,
                Call::partial_mode,
                chunk);
    } else {
        static_assert(
            Call::accumulation_mode == ttnn::kernel_lib::ReduceAccumulationMode::Intermediate,
            "Unknown reduction accumulation mode");
        compute_kernel_lib::reduce<
            Call::reduce_type,
            Call::reduce_dim,
            Call::input_cb_id,
            Call::auxiliary_cb_id,
            Call::output_cb_id,
            Call::input_policy,
            Call::reconfig_mode,
            Call::fp32_mode,
            Call::algorithm,
            Call::within_tile,
            Call::reduce_factor>(
                shape,
                layout,
                compute_kernel_lib::Accumulate::at(Call::accumulator_cb_id, Call::accumulation_index)
                    .with_reload(Call::reload_mode),
                compute_kernel_lib::NoOp{},
                Call::partial_mode,
                chunk);
    }

    auxiliary.pop_front(Call::auxiliary_tile_count);
}

// This walk belongs to the example kernel, not the reduce library. A fused
// consumer can instead issue CallAt<I> at arbitrary points in its own control
// flow and interleave unrelated work between calls.
template <uint32_t CALL_INDEX = 0>
ALWI void issue_calls() {
    if constexpr (CALL_INDEX < num_calls) {
        issue_call<CallAt<CALL_INDEX>>();
        issue_calls<CALL_INDEX + 1>();
    }
}
}  // namespace

void kernel_main() {
    using First = CallAt<0>;
    constexpr uint32_t startup_src_b =
        First::algorithm == compute_kernel_lib::ReduceAlgorithm::AccumulateViaAdd
            ? First::input_cb_id
            : First::auxiliary_cb_id;
    compute_kernel_hw_startup(First::input_cb_id, startup_src_b, First::output_cb_id);

    arm_persistent_inputs();
    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        issue_calls();
        if (iter + 1 < kernel_iters) {
            DataflowBuffer output(LastCall::output_cb_id);
            output.wait_front(output_tile_count<LastCall>());
            output.pop_front(output_tile_count<LastCall>());
        }
    }
}
"""


# The only dataflow source in this module. It sees a physical tile recipe, not
# reduction algorithms, masks/scalers, call ordering, or accumulation policy.
_AUXILIARY_KERNEL = r"""
#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {
constexpr uint32_t kernel_iters = get_compile_time_arg_val(0);
constexpr uint32_t reduce_args_offset = 1;
constexpr uint32_t num_calls = get_compile_time_arg_val(reduce_args_offset);
constexpr uint32_t first_call_args_offset =
    reduce_args_offset + ttnn::kernel_lib::reduce_plan_args::call_count_word_count;
static_assert(num_calls > 0, "A reduction plan must contain at least one call");
static_assert(kernel_iters > 0, "The kernel iteration count must be positive");

template <uint32_t CALL_INDEX>
using CallAt = ttnn::kernel_lib::ReduceCallAtT<first_call_args_offset, CALL_INDEX>;

template <uint32_t CALL_INDEX = 0>
FORCE_INLINE void prepare_call_auxiliaries() {
    if constexpr (CALL_INDEX < num_calls) {
        using Call = CallAt<CALL_INDEX>;
        dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<Call>();
        prepare_call_auxiliaries<CALL_INDEX + 1>();
    }
}
}  // namespace

void kernel_main() {
    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        prepare_call_auxiliaries();
    }
}
"""


def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def create_sharded_memory_config(shape):
    return ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _dtype_of(name):
    return ttnn.float32 if name == "fp32" else ttnn.bfloat16


def _scratch_cb(cb_id, data_format, num=1):
    page_size = ttnn.tile_size(data_format)
    fmt = ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=data_format, page_size=page_size)
    return ttnn.CBDescriptor(
        total_size=page_size * num,
        core_ranges=_single_core(),
        format_descriptors=[fmt],
    )


def _planner_tensor_spec(tensor, logical_shape):
    """Keep native dtype/layout/memory metadata while restoring the logical NC dimension."""
    memory = tensor.memory_config()
    return ttnn.TensorSpec(
        ttnn.Shape(list(logical_shape)),
        tensor.dtype,
        tensor.layout,
        memory.memory_layout,
        memory.shard_spec,
        memory.buffer_type,
        tensor.spec.tile,
    )


def _logical_input_shape(dim, Ht, Wt, NC, partial_elems):
    height = Ht * TILE
    width = Wt * TILE
    if partial_elems:
        if dim == "row":
            width = (Wt - 1) * TILE + partial_elems
        elif dim == "col":
            height = (Ht - 1) * TILE + partial_elems
    return (NC, height, width)


def _logical_output_shape(dim, Ht, Wt, NC):
    if dim == "row":
        return (NC, Ht * TILE, TILE)
    if dim == "col":
        return (NC, TILE, Wt * TILE)
    return (NC, TILE, TILE)


def _mean_n(dim, Ht, Wt, partial_elems):
    if partial_elems:
        return (reduced_count(dim, Ht, Wt) - 1) * TILE + partial_elems
    return elements_reduced(dim, Ht, Wt)


def _hardware(input_tensor, fp32_dest):
    return _PLANNER.ReduceHardwareConfig(
        arch=input_tensor.device().arch(),
        fp32_dest_acc_en=fp32_dest,
        dst_full_sync_en=False,
        available_l1_bytes=ttnn.get_max_worker_l1_unreserved_size(),
    )


def _natural_input_cb_bytes(input_tensor, Ht, Wt, NC, row_stride=0):
    physical_wt = row_stride or Wt
    return NC * Ht * physical_wt * ttnn.tile_size(input_tensor.dtype)


def _legacy_policy_cap(input_tensor, dim, Ht, Wt, NC, policy, row_stride=0):
    """Translate old benchmark labels into the planner's supported L1 constraint."""
    natural = _natural_input_cb_bytes(input_tensor, Ht, Wt, NC, row_stride)
    if policy == "stream":
        return 2 * ttnn.tile_size(input_tensor.dtype)
    if policy == "no_wait":
        memory_layout = input_tensor.memory_config().memory_layout
        directly_aliasable = (dim == "row" and memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED) or (
            dim == "col" and memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
        )
        if directly_aliasable:
            return 0
    return natural


def _input_cb_ids(count):
    ids = [cb_id for cb_id in range(64) if cb_id not in (CB_AUXILIARY, CB_ACCUMULATOR, CB_OUT)]
    if count > len(ids):
        raise ValueError(f"at most {len(ids)} input CBs fit in this example, got {count}")
    return ids[:count]


def _find_cb_requirement(call, role):
    return next(requirement for requirement in call.plan.cb_requirements if requirement.role == role)


def _make_sequence_plan(
    input_tensor,
    output_tensor,
    *,
    dim,
    Ht,
    Wt,
    NC,
    fp32_dest,
    input_cb_ids,
    reduce_math,
    scalar,
    partial_elems,
    policy,
    max_input_cb_bytes=None,
):
    input_spec = _planner_tensor_spec(input_tensor, _logical_input_shape(dim, Ht, Wt, NC, partial_elems))
    output_spec = _planner_tensor_spec(output_tensor, _logical_output_shape(dim, Ht, Wt, NC))
    cap = (
        max_input_cb_bytes
        if max_input_cb_bytes is not None
        else _legacy_policy_cap(input_tensor, dim, Ht, Wt, NC, policy)
    )
    configs = [
        (
            cb_id,
            _PLANNER.ReduceCallConfig(
                input_spec=input_spec,
                output_spec=output_spec,
                reduce_math=reduce_math,
                reduce_dim=_REDUCE_DIM[dim],
                scalar=scalar,
                fp32_mode=_PLANNER.ReduceFp32Mode.FAST,
                max_input_cb_bytes=cap,
            ),
        )
        for cb_id in input_cb_ids
    ]
    return _PLANNER.make_reduce_sequence_plan(
        reductions=configs,
        cb_ids=_PLANNER.ReduceSequenceCbIds(
            auxiliary_cb_id=CB_AUXILIARY,
            accumulator_cb_id=CB_ACCUMULATOR,
            output_cb_id=CB_OUT,
        ),
        hardware=_hardware(input_tensor, fp32_dest),
    )


def _planned_kernels(sequence, *, kernel_iters, fidelity, fp32_dest, compute_cfg=None):
    compile_time_args = [kernel_iters]
    sequence.append_to(compile_time_args)
    reader = ttnn.KernelDescriptor(
        kernel_source=_AUXILIARY_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=compile_time_args,
        runtime_args=[],
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=_PLANNED_REDUCE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=compile_time_args,
        config=compute_cfg or ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=fp32_dest),
    )
    return [reader, compute]


def _planned_cbs(input_bindings, output_tensor, sequence, accumulation_dtype):
    auxiliary_pages = max(
        _find_cb_requirement(call, _PLANNER.ReduceCbRole.AUXILIARY).page_count for call in sequence.calls
    )
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(cb_id, tensor) for cb_id, tensor in input_bindings]
    cbs.extend(
        [
            ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output_tensor),
            _scratch_cb(CB_AUXILIARY, input_bindings[0][1].dtype, auxiliary_pages),
        ]
    )
    if len(sequence.calls) > 1:
        cbs.append(_scratch_cb(CB_ACCUMULATOR, accumulation_dtype, out_tile_count_from_plan(sequence.calls[0])))
    return cbs


def out_tile_count_from_plan(call):
    if call.plan.reduce_dim == _PLANNER.ReduceDimension.ROW:
        return call.plan.Ht * call.plan.batches
    if call.plan.reduce_dim == _PLANNER.ReduceDimension.COLUMN:
        return call.plan.Wt * call.plan.batches
    return call.plan.batches


def create_program_descriptor(
    input_tensor,
    output_tensor,
    *,
    variant="automatic",
    dim,
    Ht,
    Wt,
    NC=1,
    accum="fp32",
    kernel_iters=1,
    math_fidelity=None,
    partial_elems=0,
    stream=False,
    policy="bulk",
    avg_post_op=False,
    reconfig=None,
    row_stride=0,
    within_tile="collapse",
    max_input_cb_bytes=None,
):
    if variant not in VARIANTS + _LEGACY_VARIANTS:
        raise ValueError(f"variant must be 'automatic', got {variant!r}")
    if dim not in DIMS:
        raise ValueError(f"dim must be one of {DIMS}, got {dim!r}")
    if accum not in DTYPES:
        raise ValueError(f"accum must be one of {DTYPES}, got {accum!r}")
    if min(Ht, Wt, NC, kernel_iters) < 1:
        raise ValueError("Ht, Wt, NC and kernel_iters must be positive")
    if input_tensor.dtype != ttnn.bfloat16 or input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("input must be bfloat16 TILE_LAYOUT")
    if output_tensor.dtype != ttnn.float32 or output_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("output must be float32 TILE_LAYOUT")
    if partial_elems and (not 1 <= partial_elems < TILE or dim == "scalar"):
        raise ValueError("partial_elems must be in [1, 31] and is supported for row/col only")
    if stream:
        policy = "stream"
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    if row_stride:
        raise ValueError("row_stride is no longer a manual kernel lever; describe it in the tensor layout")
    if within_tile != "collapse":
        raise ValueError("within_tile is selected by the host planner")
    if reconfig is not None:
        raise ValueError("reconfiguration policy is selected by the host planner")

    fp32_dest = accum == "fp32"
    local_elements = _mean_n(dim, Ht, Wt, partial_elems)
    scalar = (2.0 if avg_post_op else 1.0) / local_elements
    sequence = _make_sequence_plan(
        input_tensor,
        output_tensor,
        dim=dim,
        Ht=Ht,
        Wt=Wt,
        NC=NC,
        fp32_dest=fp32_dest,
        input_cb_ids=[CB_IN],
        reduce_math=_PLANNER.ReduceMath.AVG,
        scalar=scalar,
        partial_elems=partial_elems,
        policy=policy,
        max_input_cb_bytes=max_input_cb_bytes,
    )
    fidelity = math_fidelity or ttnn.MathFidelity.HiFi4
    kernels = _planned_kernels(sequence, kernel_iters=kernel_iters, fidelity=fidelity, fp32_dest=fp32_dest)
    cbs = _planned_cbs([(CB_IN, input_tensor)], output_tensor, sequence, _dtype_of(accum))
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def run_op(input_tensor, **kwargs):
    dim, Ht, Wt = kwargs["dim"], kwargs["Ht"], kwargs["Wt"]
    NC = kwargs.get("NC", 1)
    out_hw = output_shape(dim, Ht, Wt, NC)
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(out_hw)),
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        create_sharded_memory_config(out_hw),
    )
    descriptor = create_program_descriptor(input_tensor, output, **kwargs)
    return ttnn.generic_op([input_tensor, output], descriptor)


def create_accumulate_program_descriptor(
    input_tensor,
    output_tensor,
    *,
    dim,
    Ht,
    Wt,
    NC=1,
    accum="fp32",
    kernel_iters=1,
    num_chunks=2,
    mean=False,
    math_fidelity=None,
    partial_elems=0,
    row_stride=0,
    reload="copy_pairs",
    acc_unpack_to_dest=False,
    max_input_cb_bytes=None,
):
    if dim not in DIMS:
        raise ValueError(f"dim must be one of {DIMS}, got {dim!r}")
    if reload not in RELOADS:
        raise ValueError(f"reload must be one of {RELOADS}, got {reload!r}")
    # Compatibility-only label: reload policy is part of the returned plan.
    if min(Ht, Wt, NC, kernel_iters, num_chunks) < 1:
        raise ValueError("Ht, Wt, NC, kernel_iters and num_chunks must be positive")
    if input_tensor.dtype != ttnn.bfloat16 or input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("input must be bfloat16 TILE_LAYOUT")
    if output_tensor.dtype != ttnn.float32 or output_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("output must be float32 TILE_LAYOUT")
    if partial_elems and (not 1 <= partial_elems < TILE or dim == "scalar"):
        raise ValueError("partial_elems must be in [1, 31] and is supported for row/col only")
    if row_stride:
        raise ValueError("row_stride is no longer a manual kernel lever; describe it in the tensor layout")
    if acc_unpack_to_dest and accum != "fp32":
        raise ValueError("acc_unpack_to_dest requires accum='fp32'")

    fp32_dest = accum == "fp32"
    local_elements = _mean_n(dim, Ht, Wt, partial_elems)
    reduce_math = _PLANNER.ReduceMath.AVG if mean else _PLANNER.ReduceMath.SUM
    scalar = 1.0 / local_elements if mean else 1.0
    input_cb_ids = _input_cb_ids(num_chunks)
    sequence = _make_sequence_plan(
        input_tensor,
        output_tensor,
        dim=dim,
        Ht=Ht,
        Wt=Wt,
        NC=NC,
        fp32_dest=fp32_dest,
        input_cb_ids=input_cb_ids,
        reduce_math=reduce_math,
        scalar=scalar,
        partial_elems=partial_elems,
        policy="bulk",
        max_input_cb_bytes=max_input_cb_bytes,
    )

    fidelity = math_fidelity or ttnn.MathFidelity.HiFi4
    compute_cfg = ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=fp32_dest)
    if acc_unpack_to_dest:
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[CB_ACCUMULATOR] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_cfg.unpack_to_dest_mode = modes
    kernels = _planned_kernels(
        sequence,
        kernel_iters=kernel_iters,
        fidelity=fidelity,
        fp32_dest=fp32_dest,
        compute_cfg=compute_cfg,
    )
    input_bindings = [(cb_id, input_tensor) for cb_id in input_cb_ids]
    cbs = _planned_cbs(input_bindings, output_tensor, sequence, _dtype_of(accum))
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def run_accumulate(input_tensor, **kwargs):
    dim, Ht, Wt = kwargs["dim"], kwargs["Ht"], kwargs["Wt"]
    NC = kwargs.get("NC", 1)
    out_hw = output_shape(dim, Ht, Wt, NC)
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(out_hw)),
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        create_sharded_memory_config(out_hw),
    )
    descriptor = create_accumulate_program_descriptor(input_tensor, output, **kwargs)
    return ttnn.generic_op([input_tensor, output], descriptor)
