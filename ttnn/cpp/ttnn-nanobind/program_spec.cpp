// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Python bindings for the Metal 2.0 host API (ProgramSpec and friends).
//
// DELIBERATELY DUMB. This mirrors tt_metal/api/tt-metalium/experimental/metal2_host_api/*
// one-to-one and knows nothing about any programming model built on top of it. The
// unified model's own vocabulary -- which buffer is an INPUT, which DM thread produces it
// -- belongs in unified_harness.py, where it is cheap to change; see unified_metal2_spec.md.
//
// The counterpart for the legacy path is program_descriptors.cpp, which binds
// ProgramDescriptor for ttnn.generic_op. The two are independent: a program is built
// either way, not both.
//
// TensorParameters are the one place this is not a straight mirror. ProgramRunArgs holds
// TensorArguments as reference_wrappers into MeshTensors it does not own, which is not a
// thing Python can safely hold, so `run_program_spec` takes the ttnn Tensors as a separate
// argument and fills the table itself. The Tensors stay alive in the caller's arguments
// for the duration of the (blocking) call, which is what makes the references valid.

#include "program_spec.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt_stl/strong_type.hpp>

#include "ttnn-nanobind/export_enum.hpp"
#include "ttnn/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace nb = nanobind;
namespace exp = tt::tt_metal::experimental;

// ---------------------------------------------------------------------------
// Type casters
//
// Two of the Metal 2.0 vocabulary types have no natural Python spelling, and both have an
// obvious one: a StrongType<std::string> is a str, and a Table<K, V> is a dict. Casting
// them here rather than binding them as classes is what keeps every spec field below
// readable from Python without a wrapper for each.
// ---------------------------------------------------------------------------

namespace nanobind::detail {

// KernelSpecName, DFBSpecName, SemaphoreSpecName, TensorParamName -- all
// ttsl::StrongType<std::string, Tag>, all just names.
template <typename Tag>
struct type_caster<ttsl::StrongType<std::string, Tag>> {
    using Strong = ttsl::StrongType<std::string, Tag>;
    using StrCaster = make_caster<std::string>;
    NB_TYPE_CASTER(Strong, const_name("str"))

    bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept {
        StrCaster caster;
        if (!caster.from_python(src, flags, cleanup)) {
            return false;
        }
        value = Strong(caster.operator cast_t<std::string>());
        return true;
    }

    static handle from_cpp(const Strong& v, rv_policy policy, cleanup_list* cleanup) noexcept {
        return StrCaster::from_cpp(v.get(), policy, cleanup);
    }
};

// Table<K, V> <-> dict. Insertion order is not meaningful to a Table (it documents its
// iteration order as unspecified), so a dict is an honest representation rather than a
// lossy one.
template <typename K, typename V>
struct type_caster<tt::tt_metal::experimental::Table<K, V>> {
    using MapT = tt::tt_metal::experimental::Table<K, V>;
    using KeyCaster = make_caster<K>;
    using ValueCaster = make_caster<V>;
    NB_TYPE_CASTER(MapT, const_name("dict[") + KeyCaster::Name + const_name(", ") + ValueCaster::Name + const_name("]"))

    bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept {
        dict d;
        if (!try_borrow(src, d)) {
            return false;
        }
        value.clear();
        for (auto [k, v] : d) {
            KeyCaster kc;
            ValueCaster vc;
            if (!kc.from_python(k, flags, cleanup) || !vc.from_python(v, flags, cleanup)) {
                return false;
            }
            value.insert({kc.operator cast_t<K>(), vc.operator cast_t<V>()});
        }
        return true;
    }

    static handle from_cpp(const MapT& v, rv_policy policy, cleanup_list* cleanup) noexcept {
        object out = steal(PyDict_New());
        if (!out.is_valid()) {
            return handle();
        }
        for (const auto& [k, val] : v) {
            handle kh = KeyCaster::from_cpp(k, policy, cleanup);
            handle vh = ValueCaster::from_cpp(val, policy, cleanup);
            if (!kh.is_valid() || !vh.is_valid() || PyDict_SetItem(out.ptr(), kh.ptr(), vh.ptr()) != 0) {
                return handle();
            }
            kh.dec_ref();
            vh.dec_ref();
        }
        return out.release();
    }

private:
    static bool try_borrow(handle src, dict& out) noexcept {
        if (!PyDict_Check(src.ptr())) {
            return false;
        }
        out = borrow<dict>(src);
        return true;
    }
};

}  // namespace nanobind::detail

namespace ttnn::program_spec {

namespace {

// Fill the run args' tensor table from ttnn Tensors and dispatch.
//
// Split out from the binding so the reference_wrappers are formed and consumed inside one
// scope: `tensors` is a caller argument, so every MeshTensor referenced here outlives the
// enqueue below.
void run_program_spec(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const exp::ProgramSpec& spec,
    exp::ProgramRunArgs run_args,
    const std::vector<std::pair<std::string, ttnn::Tensor>>& tensors,
    bool blocking) {
    for (const auto& [name, tensor] : tensors) {
        run_args.tensor_args.insert(
            {exp::TensorParamName{name}, exp::ProgramRunArgs::TensorArgument{std::cref(tensor.mesh_tensor())}});
    }

    // No caching: the workload is rebuilt on every call, so a kernel edit is picked up and
    // nothing stale can be dispatched. That costs a spec validation and a JIT-cache lookup
    // per launch, which is the right trade for a correctness harness and the wrong one for
    // a benchmark. Caching belongs here when the benchmarks move over.
    tt::tt_metal::distributed::MeshWorkload workload = exp::MakeMeshWorkloadFromSpec(mesh_device, spec);
    tt::tt_metal::Program& program = workload.get_programs().begin()->second;
    exp::SetProgramRunArgs(program, run_args);

    // A runtime id, WITHOUT WHICH THE PROGRAM IS INVISIBLE TO THE REAL-TIME PROFILER. It
    // defaults to 0, and 0 is REALTIME_PROFILER_UNPROFILED_PROGRAM_HOST_ID
    // (dispatch/kernels/realtime_profiler.hpp:23) -- so the dispatch kernel takes a
    // zero-id program for one the host asked not to profile and never appends it to the
    // record FIFO. No error, no records, and a profiler that reports itself active.
    //
    // Nothing else assigns one: the only setter in the tree outside tests is ttnn's
    // device-operation path (device_operation.hpp:186), which every ttnn op goes through and
    // a program built straight from a ProgramSpec does not. Same counter as ttnn's, so ids
    // stay unique across both.
    const auto runtime_id = static_cast<uint64_t>(CoreIDs::instance().fetch_and_increment_device_operation_id());
    for (auto& [_, p] : workload.get_programs()) {
        p.set_runtime_id(runtime_id);
    }

    tt::tt_metal::distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, blocking);
}

}  // namespace

void py_module_types(nb::module_& mod) {
    // ---------------------------------------------------------------- enums
    export_enum<exp::DFBEndpointType>(mod, "DFBEndpointType");
    export_enum<exp::DFBAccessPattern>(mod, "DFBAccessPattern");
    // Precision is part of ComputeGen1Config's surface and is not bound anywhere else in
    // ttnn, so it is exported here rather than left unreachable. (The legacy path spells the
    // same choice as ComputeConfigDescriptor's math_approx_mode bool.)
    export_enum<tt::tt_metal::Precision>(mod, "Precision");
    // Also unbound elsewhere in ttnn, and it MATTERS: KernelSpec defaults every kernel to O2,
    // where the legacy ComputeConfig defaulted compute to O3 (kernel_types.hpp:132). A compute
    // kernel built at O2 is slower, and for an LLK-heavy one it can fail to link outright --
    // constant propagation no longer reaches the addrmod immediates and LTO reports
    // "impossible constraint in 'asm'".
    export_enum<tt::tt_metal::KernelBuildOptLevel>(mod, "KernelBuildOptLevel");

    // ------------------------------------------------------------- KernelSpec
    auto kernel_spec = nb::class_<exp::KernelSpec>(mod, "KernelSpec", R"pbdoc(
        A compiled kernel: source, compiler options, resource bindings and argument schema.

        One source may back several KernelSpecs in one ProgramSpec, each compiled and placed
        independently. That is what lets a single file be compiled for every projection of a
        node.
    )pbdoc");

    nb::class_<exp::KernelSpec::SourceCode>(kernel_spec, "SourceCode")
        .def(nb::init<>())
        .def(
            "__init__",
            [](exp::KernelSpec::SourceCode* self, std::string code) {
                new (self) exp::KernelSpec::SourceCode{std::move(code)};
            })
        .def_rw("code", &exp::KernelSpec::SourceCode::code);

    nb::class_<exp::KernelSpec::CompilerOptions>(kernel_spec, "CompilerOptions")
        .def(nb::init<>())
        .def_rw("include_paths", &exp::KernelSpec::CompilerOptions::include_paths)
        .def_rw("defines", &exp::KernelSpec::CompilerOptions::defines)
        .def_rw("opt_level", &exp::KernelSpec::CompilerOptions::opt_level);

    nb::class_<exp::DFBBinding>(kernel_spec, "DFBBinding", R"pbdoc(
        Declares that this kernel is the producer or the consumer of a dataflow buffer.

        A DFB has exactly two endpoint roles and both are exclusive per node, which is why a
        buffer cannot be bound to every kernel of a node. See unified_metal2_spec.md 7.1.
    )pbdoc")
        .def(nb::init<>())
        .def_rw("dfb_spec_name", &exp::DFBBinding::dfb_spec_name)
        .def_rw("accessor_name", &exp::DFBBinding::accessor_name)
        .def_rw("endpoint_type", &exp::DFBBinding::endpoint_type)
        .def_rw("access_pattern", &exp::DFBBinding::access_pattern);

    mod.def(
        "producer_of",
        &exp::ProducerOf,
        nb::arg("dfb_spec_name"),
        nb::arg("accessor_name"),
        "A PRODUCER binding of the named dataflow buffer.");
    mod.def(
        "consumer_of",
        &exp::ConsumerOf,
        nb::arg("dfb_spec_name"),
        nb::arg("accessor_name"),
        "A CONSUMER binding of the named dataflow buffer.");

    nb::class_<exp::SemaphoreBinding>(kernel_spec, "SemaphoreBinding")
        .def(nb::init<>())
        .def_rw("semaphore_spec_name", &exp::SemaphoreBinding::semaphore_spec_name)
        .def_rw("accessor_name", &exp::SemaphoreBinding::accessor_name);

    nb::class_<exp::TensorBinding>(kernel_spec, "TensorBinding", R"pbdoc(
        Declares that this kernel accesses a tensor parameter, under a local accessor name.

        Unlike a DFB binding a tensor binding carries no exclusive role, so the same tensor
        may be bound to every kernel of a node -- which is what lets one shared source name
        tensor::<accessor_name> on every projection.
    )pbdoc")
        .def(nb::init<>())
        .def_rw("tensor_parameter_name", &exp::TensorBinding::tensor_parameter_name)
        .def_rw("accessor_name", &exp::TensorBinding::accessor_name);

    nb::class_<exp::KernelSpec::RuntimeArgSchema>(kernel_spec, "RuntimeArgSchema")
        .def(nb::init<>())
        .def_rw("runtime_arg_names", &exp::KernelSpec::RuntimeArgSchema::runtime_arg_names)
        .def_rw("common_runtime_arg_names", &exp::KernelSpec::RuntimeArgSchema::common_runtime_arg_names);

    kernel_spec.def(nb::init<>())
        .def_rw("unique_id", &exp::KernelSpec::unique_id)
        .def_rw("source", &exp::KernelSpec::source)
        .def_rw("num_threads", &exp::KernelSpec::num_threads)
        .def_rw("compiler_options", &exp::KernelSpec::compiler_options)
        .def_rw("dfb_bindings", &exp::KernelSpec::dfb_bindings)
        .def_rw("semaphore_bindings", &exp::KernelSpec::semaphore_bindings)
        .def_rw("tensor_bindings", &exp::KernelSpec::tensor_bindings)
        .def_rw("compile_time_args", &exp::KernelSpec::compile_time_args)
        .def_rw("runtime_arg_schema", &exp::KernelSpec::runtime_arg_schema)
        .def_rw("hw_config", &exp::KernelSpec::hw_config)
        .def_rw("advanced_options", &exp::KernelSpec::advanced_options);

    // ------------------------------------------------------- hardware configs
    nb::class_<exp::DataMovementGen1Config>(mod, "DataMovementGen1Config")
        .def(nb::init<>())
        .def_rw("processor", &exp::DataMovementGen1Config::processor)
        .def_rw("noc", &exp::DataMovementGen1Config::noc)
        .def_rw("noc_mode", &exp::DataMovementGen1Config::noc_mode);

    nb::class_<exp::ComputeGen1Config>(mod, "ComputeGen1Config")
        .def(nb::init<>())
        .def_rw("fpu_math_fidelity", &exp::ComputeGen1Config::fpu_math_fidelity)
        .def_rw("sfpu_precision_mode", &exp::ComputeGen1Config::sfpu_precision_mode)
        .def_rw("bfp_pack_precision_mode", &exp::ComputeGen1Config::bfp_pack_precision_mode);

    nb::class_<exp::KernelAdvancedOptions>(mod, "KernelAdvancedOptions")
        .def(nb::init<>())
        .def_rw("num_runtime_varargs", &exp::KernelAdvancedOptions::num_runtime_varargs)
        .def_rw("num_common_runtime_varargs", &exp::KernelAdvancedOptions::num_common_runtime_varargs);

    // --------------------------------------------------- program-scope resources
    nb::class_<exp::DataflowBufferSpec>(mod, "DataflowBufferSpec", R"pbdoc(
        A software FIFO between a producer kernel and a consumer kernel. The Metal 2.0
        replacement for a circular buffer.
    )pbdoc")
        .def(nb::init<>())
        .def_rw("unique_id", &exp::DataflowBufferSpec::unique_id)
        .def_rw("entry_size", &exp::DataflowBufferSpec::entry_size)
        .def_rw("num_entries", &exp::DataflowBufferSpec::num_entries)
        // tt::DataFormat is not bound in nanobind, so this takes a ttnn DataType and
        // converts, exactly as CBFormatDescriptor does on the legacy path. The getter hands
        // back the raw enum value rather than a DataType, because the mapping is not
        // invertible.
        .def_prop_rw(
            "data_format_metadata",
            [](const exp::DataflowBufferSpec& d) -> std::optional<uint8_t> {
                if (!d.data_format_metadata.has_value()) {
                    return std::nullopt;
                }
                return static_cast<uint8_t>(*d.data_format_metadata);
            },
            [](exp::DataflowBufferSpec& d, std::optional<tt::tt_metal::DataType> dtype) {
                d.data_format_metadata =
                    dtype.has_value()
                        ? std::optional<tt::DataFormat>(tt::tt_metal::datatype_to_dataformat_converter(*dtype))
                        : std::nullopt;
            },
            "Entry data format, required for any DFB bound to a compute kernel. Set with a ttnn "
            "DataType (e.g. ttnn.bfloat16); reads back as the raw tt::DataFormat enum value.");

    nb::class_<exp::SemaphoreSpec>(mod, "SemaphoreSpec")
        .def(nb::init<>())
        .def_rw("unique_id", &exp::SemaphoreSpec::unique_id)
        .def_rw("target_nodes", &exp::SemaphoreSpec::target_nodes)
        .def_prop_rw(
            "initial_value",
            // Reading and writing a deprecated member is the whole point of exposing it: the
            // legacy harness takes an initial value per semaphore, so the shim has to be able
            // to carry one. Gen2 rejects a non-zero value; Gen1 does not.
            [](const exp::SemaphoreSpec& s) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
                return s.advanced_options.initial_value;
#pragma GCC diagnostic pop
            },
            [](exp::SemaphoreSpec& s, uint32_t v) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
                s.advanced_options.initial_value = v;
#pragma GCC diagnostic pop
            });

    // No default constructor: TensorSpec has none, so a TensorParameter is built from a
    // name and a spec, and the spec comes off a ttnn.Tensor the caller already allocated.
    nb::class_<exp::TensorParameter>(mod, "TensorParameter")
        .def(
            "__init__",
            [](exp::TensorParameter* self, std::string name, tt::tt_metal::TensorSpec spec) {
                new (self) exp::TensorParameter{exp::TensorParamName{std::move(name)}, std::move(spec), {}};
            },
            nb::arg("unique_id"),
            nb::arg("spec"))
        .def_rw("unique_id", &exp::TensorParameter::unique_id)
        .def_rw("spec", &exp::TensorParameter::spec);

    nb::class_<exp::WorkUnitSpec>(mod, "WorkUnitSpec", R"pbdoc(
        A set of kernels that run together on a set of nodes. Every node in target_nodes runs
        an identical set of kernel instances.
    )pbdoc")
        .def(nb::init<>())
        .def_rw("name", &exp::WorkUnitSpec::name)
        .def_rw("kernels", &exp::WorkUnitSpec::kernels)
        .def_rw("target_nodes", &exp::WorkUnitSpec::target_nodes);

    nb::class_<exp::ProgramSpec>(mod, "ProgramSpec", R"pbdoc(
        Everything immutable about a program: its kernels, its program-scope resources, its
        tensor parameter declarations and where they all run.
    )pbdoc")
        .def(nb::init<>())
        .def_rw("name", &exp::ProgramSpec::name)
        .def_rw("kernels", &exp::ProgramSpec::kernels)
        .def_rw("dataflow_buffers", &exp::ProgramSpec::dataflow_buffers)
        .def_rw("semaphores", &exp::ProgramSpec::semaphores)
        .def_rw("tensor_parameters", &exp::ProgramSpec::tensor_parameters)
        .def_rw("work_units", &exp::ProgramSpec::work_units);

    // ------------------------------------------------------------ run args
    auto run_args = nb::class_<exp::ProgramRunArgs>(mod, "ProgramRunArgs", R"pbdoc(
        Everything mutable about one execution of a program.

        Tensor arguments are NOT held here: they are passed to run_program_spec directly, so
        the Tensors stay owned by the caller for the duration of the call.
    )pbdoc");

    nb::class_<exp::ProgramRunArgs::KernelRunArgs>(run_args, "KernelRunArgs")
        .def(nb::init<>())
        .def_rw("kernel", &exp::ProgramRunArgs::KernelRunArgs::kernel)
        .def_rw("runtime_arg_values", &exp::ProgramRunArgs::KernelRunArgs::runtime_arg_values)
        .def_rw("common_runtime_arg_values", &exp::ProgramRunArgs::KernelRunArgs::common_runtime_arg_values);

    run_args.def(nb::init<>()).def_rw("kernel_run_args", &exp::ProgramRunArgs::kernel_run_args);

    // --------------------------------------------------------------- dispatch
    mod.def(
        "run_program_spec",
        &run_program_spec,
        nb::arg("mesh_device"),
        nb::arg("spec"),
        nb::arg("run_args"),
        nb::arg("tensors"),
        nb::arg("blocking") = true,
        R"pbdoc(
            Build a program from `spec`, bind `run_args` and `tensors`, and enqueue it.

            `tensors` is a list of (tensor_parameter_name, ttnn.Tensor) pairs, one for every
            TensorParameter the spec declares.

            The workload is rebuilt on every call. That is deliberate for a correctness
            harness -- nothing stale can be dispatched -- and wrong for a benchmark.
        )pbdoc");
}

}  // namespace ttnn::program_spec
