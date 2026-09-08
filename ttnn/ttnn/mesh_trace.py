# Scratchpad for refactoring the TTNN mesh-trace API.
#
# Not imported by ttnn. The walkthrough below is one stack, top to bottom.
# Legacy begin/end_trace copies follow as an appendix.


NEW_TRACE_API = r"""
# =============================================================================
# ttnn.scale_mask_softmax + MeshTrace
#
# SoftmaxProgramFactoryAttentionOptimized (ScaleMaskSoftmax + default config):
#   TensorParamName  SRC{"src"}  DST{"dst"}  MASK{"mask"}
#   KernelSpecName   READER{"reader"}
#   scale --bit_cast<uint32_t>--> reader RTA "pre_scale"
#   is_causal_mask / numeric_stable are compile-time defines — not patchable
#
# Stack:
#   builder.add(op, IN, SCALE, mask)
#     unwrap
#     compile(op, call)                    → CompiledMeshWorkload
#     record_compiled(builder, compiled, pending)
#       compiled.map_trace_parameters
#         Adapter::map_trace_parameters
#           Softmax::map_trace_parameters
#       metal MeshTraceBuilder::add
#   builder.build / update_args / replay
# Hot path (unchanged cost): launch = compile + dispatch
# =============================================================================

# -----------------------------------------------------------------------------
# 1. User script
# -----------------------------------------------------------------------------
attn_scores = ttnn.from_torch(torch.randn(1, 8, 32, 32), device=mesh_device, layout=ttnn.TILE_LAYOUT)
attn_mask = ttnn.from_torch(torch.zeros(1, 8, 32, 32), device=mesh_device, layout=ttnn.TILE_LAYOUT)
new_scores = ttnn.from_torch(torch.randn(1, 8, 32, 32), device=mesh_device, layout=ttnn.TILE_LAYOUT)

IN = ttnn.TraceParam(default=attn_scores, name="input")  # object is the identity; name is the registry key
SCALE = ttnn.TraceParam(default=0.125, name="scale")

builder = ttnn.MeshTraceBuilder(mesh_device)
# IN, SCALE registered. attn_mask and is_causal_mask=True are plain values (not patchable).
out = builder.add(ttnn.scale_mask_softmax, IN, SCALE, attn_mask, is_causal_mask=True)
for _ in range(iters - 1):
    out = builder.add(ttnn.scale_mask_softmax, IN, SCALE, attn_mask, is_causal_mask=True)

trace_cq0 = builder.build(cq_id=0)
trace_cq1 = builder.build(cq_id=1)

# Pass in TraceParam directly or string name directly for updating
trace_cq0.update_args({IN: new_scores, "scale": 0.25})
trace_cq1.update_args({IN: new_scores, "scale": 0.50})

trace_cq0.replay(blocking=False)
trace_cq1.replay(blocking=False)

ttnn.synchronize_device(mesh_device)
trace_cq0.deallocate()
trace_cq1.deallocate()


# -----------------------------------------------------------------------------
# 2. Python MeshTraceBuilder.add
# -----------------------------------------------------------------------------
import inspect


class TraceParam:
    def __init__(self, default, name=None):
        self.default = default
        self.name = name  # required for intern; omit → "param_<id>"


class _PendingBind:
    __slots__ = ("param", "invoke_name")

    def __init__(self, param, invoke_name):
        self.param = param
        self.invoke_name = invoke_name  # signature name: "input_tensor", "scale"


class MeshTraceBuilder:
    def __init__(self, mesh_device):
        self._cxx = experimental.MeshTraceBuilder(mesh_device)
        self._interned: dict[int, str] = {}
        self._used_names: set[str] = set()

    def add(self, op, *args, **kwargs):
        pending, call = self._unwrap(op, args, kwargs)
        cxx_pending = [self._to_cxx_bind(b) for b in pending]
        compiled = experimental.compile(op, call)  # no enqueue
        return experimental.record_compiled(self._cxx, compiled, cxx_pending)

    def build(self, cq_id):
        return self._cxx.build(self._cxx.device().mesh_command_queue(cq_id))

    def _unwrap(self, op, args, kwargs):
        # invoke_name comes from nanobind nb::arg names via inspect.signature.
        bound = inspect.signature(op).bind(*args, **kwargs)
        bound.apply_defaults()
        pending, call = [], {}
        for name, value in bound.arguments.items():
            if isinstance(value, TraceParam):
                self._intern(value)
                pending.append(_PendingBind(value, invoke_name=name))
                value = value.default
            call[name] = value
        return pending, call

    def _intern(self, param: TraceParam) -> str:
        key = id(param)
        if key in self._interned:
            return self._interned[key]
        name = param.name if param.name is not None else f"param_{key}"
        if name in self._used_names:
            raise ValueError(f"TraceParam name {name!r} is already registered")
        self._used_names.add(name)
        self._interned[key] = name
        return name

    def _to_cxx_bind(self, bind: _PendingBind):
        return experimental.PendingBind(
            registry_name=self._interned[id(bind.param)],
            invoke_name=bind.invoke_name,
        )


# After unwrap:
#   pending = [IN → "input_tensor", SCALE → "scale"]
#   call    = {input_tensor=attn_scores, scale=0.125, mask=attn_mask, is_causal_mask=True}


# -----------------------------------------------------------------------------
# 3. compile / dispatch / launch
# -----------------------------------------------------------------------------
# Today's create_and_cache ≈ compile; enqueue_mesh_workload ≈ dispatch.
# launch calls both in one C++ stack (inlinable). Trace calls compile only.
# Python cannot name the factory type, so compile closes over it on the handle.

struct PendingBind {
    std::string registry_name; // Name specified within TraceParam constructor
    std::string invoke_name; // Op signature name, ex: "input_tensor", "scale"
};

struct CompiledMeshWorkload {
    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::experimental::ProgramRunArgs run_params;
    typename mesh_device_operation_t::tensor_return_value_t outputs;
    bool has_trace_parameters = false;
    std::function<TraceParameters(const Program&, const ProgramRunArgs&,
                                  const std::vector<PendingBind>&)>
        map_trace_parameters;
};

template <DeviceOperationWithMeshDeviceAdapter mesh_device_operation_t>
CompiledMeshWorkload compile(
    const typename mesh_device_operation_t::operation_attributes_t& attrs,
    const typename mesh_device_operation_t::tensor_args_t& tensor_args,
    typename mesh_device_operation_t::tensor_return_value_t& outputs,
    ttnn::MeshDevice* mesh_device) {
    // create_and_cache / cache-hit, minus enqueue
    CompiledMeshWorkload compiled;
    compiled.workload = /* cached or newly created */;
    compiled.run_params = /* from artifacts */;
    compiled.outputs = outputs;

    auto program_factory = mesh_device_operation_t::select_program_factory(attrs, tensor_args);
    dispatch_to_mesh_workload_factory<mesh_device_operation_t>(
        program_factory, [&]<typename WorkloadFactory>() {
            compiled.has_trace_parameters =
                requires { requires WorkloadFactory::has_trace_parameters; };
            if constexpr (requires { WorkloadFactory::map_trace_parameters; }) {
                compiled.map_trace_parameters = WorkloadFactory::map_trace_parameters;
            }
        });
    return compiled;
}

void dispatch(CompiledMeshWorkload& compiled, ttnn::MeshDevice* mesh_device) {
    enqueue_mesh_workload(/* attrs / tensors as today */, mesh_device, compiled.workload);
}

template <DeviceOperationWithMeshDeviceAdapter mesh_device_operation_t>
void launch(...) {
    auto compiled = compile<mesh_device_operation_t>(attrs, tensor_args, outputs, mesh_device);
    dispatch(compiled, mesh_device);
}


# -----------------------------------------------------------------------------
# 4. record_compiled
# -----------------------------------------------------------------------------
# Both this function and the adapter throw if the inner factory did not opt in.

auto record_compiled(
    tt::tt_metal::distributed::experimental::MeshTraceBuilder& metal_builder,
    CompiledMeshWorkload& compiled,
    const std::vector<PendingBind>& pending) {
    TraceParameters parameters;
    if (!pending.empty()) {
        if (compiled.has_trace_parameters) {
            for (auto& [range, program] : compiled.workload.get_programs()) {
                merge_trace_parameters(
                    parameters,
                    compiled.map_trace_parameters(program, compiled.run_params, pending));
            }
        } else {
            TT_THROW(
                "This op's factory does not declare map_trace_parameters; "
                "TraceParam arguments are unsupported");
        }
    }
    metal_builder.add(compiled.workload, parameters);
    return compiled.outputs;
    // dispatch is not called
}


# -----------------------------------------------------------------------------
# 5. Adapter — ProgramSpecMeshWorkloadFactoryAdapter<SpecFactory>
# -----------------------------------------------------------------------------
# compile's WorkloadFactory is this wrapper, never softmax by name.
# Method always exists. Forwards if SpecFactory opted in, otherwise throws.

template <typename SpecFactory>
struct ProgramSpecMeshWorkloadFactoryAdapter {
    static constexpr bool has_trace_parameters =
        requires { SpecFactory::map_trace_parameters; };

    static TraceParameters map_trace_parameters(
        const Program& program,
        const ProgramRunArgs& run_params,
        const std::vector<PendingBind>& pending) {
        if constexpr (has_trace_parameters) {
            return SpecFactory::map_trace_parameters(program, run_params, pending);
        } else {
            TT_THROW(
                "This op's factory does not declare map_trace_parameters; "
                "TraceParam arguments are unsupported");
        }
    }
};


# -----------------------------------------------------------------------------
# 6. SoftmaxProgramFactoryAttentionOptimized — the 1:1
# -----------------------------------------------------------------------------
# Opt-in. Adding this static makes the adapter's has_trace_parameters true.
# Names the writes create_program_artifacts already made:
#   run_args.tensor_args.emplace(SRC, input_tensor.mesh_tensor());
#   AddRuntimeArgsForNode(reader, node, {{"pre_scale", bit_cast(scale)}, ...});

struct SoftmaxProgramFactoryAttentionOptimized {
    static TraceParameters map_trace_parameters(
        const Program& program,
        const ProgramRunArgs& run_params,
        const std::vector<PendingBind>& pending) {
        using namespace tt::tt_metal::distributed::experimental;
        using namespace tt::tt_metal::experimental;

        const KernelSpecName READER{"reader"};
        const TensorParamName SRC{"src"};

        std::vector<NodeCoord> pre_scale_nodes;
        for (const auto& kra : run_params.kernel_run_args) {
            if (kra.kernel != READER) {
                continue;
            }
            if (auto* node_vals = kra.runtime_arg_values.get("pre_scale")) {
                for (const auto& [node, _] : *node_vals) {
                    pre_scale_nodes.push_back(node);
                }
            }
        }

        TraceParameters parameters;
        for (const auto& bind : pending) {
            if (bind.invoke_name == "input_tensor") {
                parameters.tensor_parameters[TraceTensorArgName(bind.registry_name)].push_back(
                    TraceTensorArgPath{.program = program, .param_name = SRC});
            } else if (bind.invoke_name == "scale") {
                TT_FATAL(!pre_scale_nodes.empty(), "scale is only a reader RTA when a mask is fused");
                parameters.runtime_parameters[TraceRuntimeArgName(bind.registry_name)].push_back(
                    TraceRuntimeArgPath{
                        .program = program,
                        .kernel_name = READER,
                        .arg_name = "pre_scale",
                        .nodes = pre_scale_nodes});
            } else {
                TT_THROW(
                    "scale_mask_softmax does not expose invoke '{}' as a trace parameter "
                    "(supported: input_tensor, scale)",
                    bind.invoke_name);
            }
        }
        return parameters;
    }
};

// This call: tensor_parameters["input"] = {program, "src"}
//            runtime_parameters["scale"] = {program, "reader", "pre_scale", nodes…}


# -----------------------------------------------------------------------------
# 7. update_args / replay
# -----------------------------------------------------------------------------
static uint32_t encode_trace_runtime(std::string_view invoke_name, const SoftmaxParams& attrs) {
    if (invoke_name == "scale") {
        return std::bit_cast<std::uint32_t>(attrs.scale.value_or(1.0f));
    }
    TT_THROW("scale_mask_softmax has no runtime encode for '{}'", invoke_name);
}

# trace_cq0.update_args({IN: new_scores, "scale": 0.25})
#   "input" → TraceArgPatch.tensor_args["input"] = new_scores.mesh_tensor()
#   "scale" → encode_trace_runtime("scale", attrs) = bit_cast(0.25f) = 0x3E800000
# MeshTrace::update_args(patch); replay uses the patched words.
"""


# =============================================================================
# Appendix — legacy begin/end_trace API (current tree, not the walkthrough)
# =============================================================================

TTNN_TRACE_HPP = r"""
// ttnn/cpp/ttnn/operations/trace.hpp

#pragma once

#include <tt-metalium/mesh_trace_id.hpp>

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ttnn/types.hpp"

namespace ttnn {

using MeshTraceId = tt::tt_metal::distributed::MeshTraceId;

namespace operations::trace {

MeshTraceId begin_trace_capture(MeshDevice* device, std::optional<QueueId> cq_id);
void end_trace_capture(MeshDevice* device, MeshTraceId trace_id, std::optional<QueueId> cq_id);
void execute_trace(MeshDevice* device, MeshTraceId trace_id, std::optional<QueueId> cq_id, bool blocking);
void release_trace(MeshDevice* device, MeshTraceId trace_id);

// Unsafe allocation tracking
std::unordered_map<size_t, std::string> get_unsafe_tracked_ids(MeshDevice* device, MeshTraceId trace_id);
void remove_unsafe_tracked_id(MeshDevice* device, size_t buffer_unique_id);
std::vector<size_t> drain_pending_traceback_ids();
std::vector<size_t> drain_retired_traceback_ids();
void push_corruptible_allocation_scope(MeshDevice* device);
void pop_corruptible_allocation_scope(MeshDevice* device);

// Thread-local allocation context stack (delegates to tt::tt_metal:: free functions)
void push_allocation_context(const std::string& ctx);
void pop_allocation_context();

}  // namespace operations::trace

}  // namespace ttnn
"""


TTNN_TRACE_CPP = r"""
// ttnn/cpp/ttnn/operations/trace.cpp

#include "ttnn/operations/trace.hpp"

#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/allocation_context.hpp>
#include "tt_metal/distributed/trace_allocation_tracker.hpp"

#include <tracy/Tracy.hpp>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/core.hpp"

namespace ttnn::operations::trace {

namespace tracker = tt::tt_metal::distributed::trace_allocation_tracker;

MeshTraceId begin_trace_capture(MeshDevice* device, std::optional<QueueId> cq_id) {
    ZoneScoped;
    QueueId cq_id_value = cq_id.value_or(get_current_command_queue_id_for_thread());
    return device->begin_mesh_trace(device->mesh_command_queue(cq_id_value.get()));
}

void end_trace_capture(MeshDevice* device, MeshTraceId trace_id, std::optional<QueueId> cq_id) {
    ZoneScoped;
    QueueId cq_id_value = cq_id.value_or(get_current_command_queue_id_for_thread());
    device->end_mesh_trace(device->mesh_command_queue(cq_id_value.get()), trace_id);
}

void execute_trace(MeshDevice* device, MeshTraceId trace_id, std::optional<QueueId> cq_id, bool blocking) {
    ZoneScoped;
    QueueId cq_id_value = cq_id.value_or(get_current_command_queue_id_for_thread());
    device->replay_mesh_trace(device->mesh_command_queue(cq_id_value.get()), trace_id, blocking);
}

void release_trace(MeshDevice* device, MeshTraceId trace_id) {
    ZoneScoped;
    device->release_mesh_trace(trace_id);
}

std::unordered_map<size_t, std::string> get_unsafe_tracked_ids(MeshDevice* device, MeshTraceId trace_id) {
    return tracker::get_unsafe_tracked_ids(device, trace_id);
}

void remove_unsafe_tracked_id(MeshDevice* device, size_t buffer_unique_id) {
    tracker::remove_unsafe_tracked_id(device, buffer_unique_id);
}

std::vector<size_t> drain_pending_traceback_ids() { return tracker::drain_pending_traceback_ids(); }
std::vector<size_t> drain_retired_traceback_ids() { return tracker::drain_retired_traceback_ids(); }
void push_corruptible_allocation_scope(MeshDevice* device) { tracker::push_corruptible_allocation_scope(device); }
void pop_corruptible_allocation_scope(MeshDevice* device) { tracker::pop_corruptible_allocation_scope(device); }

void push_allocation_context(const std::string& ctx) { tt::tt_metal::push_allocation_context(ctx); }
void pop_allocation_context() { tt::tt_metal::pop_allocation_context(); }

}  // namespace ttnn::operations::trace
"""


TTNN_TRACE_NANOBIND_HPP = r"""
// ttnn/cpp/ttnn-nanobind/operations/trace.hpp

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::trace {

namespace nb = nanobind;
void py_module_types(nb::module_& mod);
void py_module(nb::module_& mod);

}  // namespace ttnn::operations::trace
"""


TTNN_TRACE_NANOBIND_CPP = r"""
// ttnn/cpp/ttnn-nanobind/operations/trace.cpp

#include "trace.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/trace.hpp"

namespace ttnn::operations::trace {

void py_module_types(nb::module_& mod) {
    nb::class_<ttnn::MeshTraceId>(mod, "MeshTraceId")
        .def(nb::init<uint32_t>())
        .def("__int__", [](const ttnn::MeshTraceId& self) { return static_cast<int>(*self); })
        .def(
            "__repr__",
            [](const ttnn::MeshTraceId& self) {
                return "MeshTraceId(" + std::to_string(static_cast<int>(*self)) + ")";
            })
        .def(nb::self == nb::self);
}

void py_module(nb::module_& mod) {
    mod.def(
        "begin_trace_capture",
        [](MeshDevice* device, std::optional<ttnn::QueueId> cq_id) {
            return ttnn::operations::trace::begin_trace_capture(device, cq_id);
        },
        nb::arg("mesh_device"),
        nb::kw_only(),
        nb::arg("cq_id") = nb::none(),
        nb::call_guard<nb::gil_scoped_release>());

    mod.def(
        "end_trace_capture",
        [](MeshDevice* device, MeshTraceId trace_id, std::optional<ttnn::QueueId> cq_id) {
            ttnn::operations::trace::end_trace_capture(device, trace_id, cq_id);
        },
        nb::arg("mesh_device"),
        nb::arg("trace_id"),
        nb::kw_only(),
        nb::arg("cq_id") = nb::none(),
        nb::call_guard<nb::gil_scoped_release>());

    mod.def(
        "execute_trace",
        [](MeshDevice* device, MeshTraceId trace_id, std::optional<QueueId> cq_id, bool blocking) {
            ttnn::operations::trace::execute_trace(device, trace_id, cq_id, blocking);
        },
        nb::arg("mesh_device"),
        nb::arg("trace_id"),
        nb::kw_only(),
        nb::arg("cq_id") = nb::none(),
        nb::arg("blocking") = true,
        nb::call_guard<nb::gil_scoped_release>());

    mod.def(
        "release_trace",
        [](MeshDevice* device, MeshTraceId trace_id) { ttnn::operations::trace::release_trace(device, trace_id); },
        nb::arg("mesh_device"),
        nb::arg("trace_id"),
        nb::call_guard<nb::gil_scoped_release>());

    // Unsafe allocation tracking
    mod.def(
        "get_unsafe_tracked_ids",
        [](MeshDevice* device, MeshTraceId trace_id) {
            return ttnn::operations::trace::get_unsafe_tracked_ids(device, trace_id);
        },
        nb::arg("mesh_device"),
        nb::arg("trace_id"));
    mod.def(
        "remove_unsafe_tracked_id",
        [](MeshDevice* device, size_t buffer_unique_id) {
            ttnn::operations::trace::remove_unsafe_tracked_id(device, buffer_unique_id);
        },
        nb::arg("mesh_device"),
        nb::arg("buffer_unique_id"));
    mod.def("drain_pending_traceback_ids", []() { return ttnn::operations::trace::drain_pending_traceback_ids(); });
    mod.def("drain_retired_traceback_ids", []() { return ttnn::operations::trace::drain_retired_traceback_ids(); });
    mod.def(
        "push_corruptible_allocation_scope",
        [](MeshDevice* device) { ttnn::operations::trace::push_corruptible_allocation_scope(device); },
        nb::arg("mesh_device"));
    mod.def(
        "pop_corruptible_allocation_scope",
        [](MeshDevice* device) { ttnn::operations::trace::pop_corruptible_allocation_scope(device); },
        nb::arg("mesh_device"));

    // Allocation context stack
    mod.def(
        "push_allocation_context",
        [](const std::string& ctx) { ttnn::operations::trace::push_allocation_context(ctx); },
        nb::arg("context"));
    mod.def("pop_allocation_context", []() { ttnn::operations::trace::pop_allocation_context(); });
}

}  // namespace ttnn::operations::trace
"""


NANOBIND_MODULE_REGISTRATION = r"""
// Relevant pieces of ttnn/cpp/ttnn-nanobind/__init__.cpp

#include "ttnn-nanobind/operations/trace.hpp"

namespace ttnn::operations {

void py_module(nb::module_& mod) {
    auto m_trace = mod.def_submodule("trace", "trace operations");
    trace::py_module_types(m_trace);
    trace::py_module(m_trace);
}

}  // namespace ttnn::operations
"""


PYTHON_PUBLIC_EXPORTS = r'''
# Relevant pieces of ttnn/ttnn/__init__.py

from ttnn._ttnn.operations.trace import (
    MeshTraceId,
    begin_trace_capture,
    end_trace_capture,
    execute_trace as _ttnn_execute_trace,
    release_trace,
)

from ttnn.trace_allocation_config import TRACE_ALLOC_TRACKING

if TRACE_ALLOC_TRACKING:
    from ttnn._ttnn.operations.trace import (
        pop_corruptible_allocation_scope as _pop_corruptible_allocation_scope,
        push_corruptible_allocation_scope as _push_corruptible_allocation_scope,
    )

    @contextlib.contextmanager
    def corruptible_allocation_scope(mesh_device):
        """Suppress accounting for intentionally corruptible allocations in this scope."""
        _push_corruptible_allocation_scope(mesh_device)
        try:
            yield
        finally:
            _pop_corruptible_allocation_scope(mesh_device)

else:

    @contextlib.contextmanager
    def corruptible_allocation_scope(mesh_device):
        """No-op when trace allocation tracking is disabled."""
        yield


if TRACE_ALLOC_TRACKING:

    def execute_trace(device, trace_id, *, cq_id=None, blocking=True):
        """Execute a captured trace, with automatic allocation-safety verification."""
        from ttnn.unsafe_allocation_tracker import UnsafeAllocationTracker

        UnsafeAllocationTracker(device).verify_before_replay(trace_id)
        return _ttnn_execute_trace(device, trace_id, cq_id=cq_id, blocking=blocking)

else:
    execute_trace = _ttnn_execute_trace


def mark_corruptible(tensor):
    """Mark a tensor buffer as intentionally corruptible for trace allocation checks."""
    from ttnn.unsafe_allocation_tracker import UnsafeAllocationTracker

    return UnsafeAllocationTracker.mark_corruptible(tensor)
'''


TRACE_ALLOCATION_CONFIG = r"""
# ttnn/ttnn/trace_allocation_config.py

import os
import warnings


def _env_enabled(name: str) -> bool:
    return os.environ.get(name) == "1"


def _env_nonnegative_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
        if parsed < 0:
            raise ValueError
        return parsed
    except ValueError:
        warnings.warn(f"{name} must be a non-negative integer; using {default}", stacklevel=2)
        return default


TRACE_ALLOC_TRACKING = _env_enabled("TT_METAL_TRACE_ALLOC_TRACKING")
TRACE_ALLOC_DIAGNOSTICS = TRACE_ALLOC_TRACKING and _env_enabled("TT_METAL_TRACE_ALLOC_TRACEBACKS")
TRACE_ALLOC_REFERRER_DEPTH = (
    _env_nonnegative_int("TT_METAL_TRACE_ALLOC_REFERRER_DEPTH", 10) if TRACE_ALLOC_DIAGNOSTICS else 10
)
"""


DECORATOR_ALLOCATION_HOOKS = r"""
# Relevant pieces of ttnn/ttnn/decorators.py.
# This wrapping exists twice: once for FastOperation.__call__ and once for
# Operation.__call__.

if TRACE_ALLOC_TRACKING:
    from ttnn._ttnn.operations.trace import pop_allocation_context, push_allocation_context

    _untracked_operation_call = Operation.__call__

    if TRACE_ALLOC_DIAGNOSTICS:

        @wraps(_untracked_operation_call)
        def _tracked_operation_call(self, *function_args, **function_kwargs):
            _drain_traceback_ids(source="op_start", op_name=self.python_fully_qualified_name)
            push_allocation_context(self.python_fully_qualified_name)
            try:
                result = _untracked_operation_call(self, *function_args, **function_kwargs)
                _drain_traceback_ids(source="op_end", op_name=self.python_fully_qualified_name)
                return result
            finally:
                pop_allocation_context()

    else:

        @wraps(_untracked_operation_call)
        def _tracked_operation_call(self, *function_args, **function_kwargs):
            push_allocation_context(self.python_fully_qualified_name)
            try:
                return _untracked_operation_call(self, *function_args, **function_kwargs)
            finally:
                pop_allocation_context()

    Operation.__call__ = _tracked_operation_call
"""


BUILD_REGISTRATION = r"""
# Relevant pieces of ttnn/sources.cmake

set(TTNNCPP_SRCS
    cpp/ttnn/operations/trace.cpp
)

set(TTNN_SRC_PYBIND
    cpp/ttnn-nanobind/__init__.cpp
    cpp/ttnn-nanobind/operations/trace.cpp
)

# Public-header list elsewhere in this file:
cpp/ttnn/operations/trace.hpp
"""


RELATED_TTNN_FILES_NOT_INLINED = r"""
ttnn/ttnn/unsafe_allocation_tracker.py
    Python-only diagnostics for allocations made while a legacy trace is live.
    It consumes the allocation-tracker bindings copied above. It is orthogonal
    to the public API shape, so keep or replace it when deciding how the new
    MeshTrace object's replay path performs safety verification.

ttnn/api/ttnn/graph/graph_query_op_runtime.hpp
    Internal benchmark helper that captures, executes, and releases a trace.
    This is a call site rather than API plumbing.
"""
