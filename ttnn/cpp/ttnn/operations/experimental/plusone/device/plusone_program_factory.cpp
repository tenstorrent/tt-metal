// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <filesystem>

#include "plusone_program_factory.hpp"
#include "plusone_device_operation_types.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace {

constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/experimental/plusone/device/kernels/reader_plusone_interleaved.cpp";

// Work geometry — a pure function of (attrs, input layout/spec), all of which are in the cache key. Per-row
// stick processing: W = last dim (elements per stick), H = number of sticks (1 when sharded, matching the
// descriptor-era behavior). The same single reader runs on every target core.
struct PlusOneWorkGeometry {
    CoreRangeSet all_cores;
    std::vector<CoreCoord> cores;
    uint32_t W = 0;
    uint32_t H = 0;
    uint32_t aligned_page_size = 0;
    bool src_is_dram = false;
    bool is_sharded = false;
    DataFormat data_format = DataFormat::Invalid;
};

PlusOneWorkGeometry compute_geometry(const PlusoneParams& attrs, const Tensor& input) {
    PlusOneWorkGeometry g;
    g.data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t unit_size = input.element_size();

    g.all_cores = CoreRangeSet(std::vector{CoreRange({0, 0}, {0, 0})});
    uint32_t num_cores = 1;
    if (attrs.sub_core_grids.has_value()) {
        g.all_cores = attrs.sub_core_grids.value();
        num_cores = g.all_cores.num_cores();
    }

    const auto& shape = input.padded_shape();
    g.W = shape[-1];
    g.H = 1;
    g.is_sharded = input.is_sharded();
    if (!g.is_sharded && shape.size() > 1) {
        for (uint32_t i = 0; i < shape.size() - 1; ++i) {
            g.H *= shape[i];
        }
    }

    auto* buf = input.buffer();
    g.src_is_dram = buf->buffer_type() == BufferType::DRAM;
    const uint32_t page_alignment = g.src_is_dram ? hal::get_dram_alignment() : hal::get_l1_alignment();
    g.aligned_page_size = tt::align(g.W * unit_size, page_alignment);
    g.cores = corerange_to_cores(g.all_cores, num_cores, true);
    return g;
}

}  // namespace

// create_program_spec — the immutable blueprint only (DFB, reader kernel, tensor parameter, work unit).
// Spec-keyed: the default reflection hash (op type + attrs + tensor spec) is the cache key, so a custom
// compute_program_hash is neither needed nor allowed.
m2::ProgramSpec PlusOneProgramFactory::create_program_spec(
    const PlusoneParams& attrs, const Tensor& input, [[maybe_unused]] Tensor& output) {
    const auto g = compute_geometry(attrs, input);

    // Scratch CB for one stick. For a sharded input the CB is borrowed from the input buffer (the modify
    // is in place); for DRAM it is plain L1 scratch the reader NOC-copies through.
    m2::DataflowBufferSpec scratch_dfb{
        .unique_id = m2::DFBSpecName{"plusone_scratch"},
        .entry_size = g.aligned_page_size,
        .num_entries = 1,
        .data_format_metadata = g.data_format,
    };
    if (g.is_sharded) {
        scratch_dfb.borrowed_from = m2::TensorParamName{"input"};
    }

    m2::KernelSpec::CompilerOptions reader_opts;
    if (g.src_is_dram) {
        reader_opts.defines.emplace("SRC_IS_DRAM", "1");
    }
    if (attrs.skip_negative_entries) {
        reader_opts.defines.emplace("SKIP_NEGATIVE_ENTRIES", "1");
    }

    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{READER_KERNEL_PATH},
        .compiler_options = std::move(reader_opts),
        .dfb_bindings =
            {m2::ProducerOf(m2::DFBSpecName{"plusone_scratch"}, "plusone_scratch"),
             m2::ConsumerOf(m2::DFBSpecName{"plusone_scratch"}, "plusone_scratch")},
        .runtime_arg_schema = {.runtime_arg_names = {"W", "H", "stick_size"}},
        .hw_config = m2::DataMovementHardwareConfig{.gen1_config = m2::DataMovementHardwareConfig::Gen1Config{}},
        .advanced_options = m2::KernelAdvancedOptions{.enqueue_invariant_runtime_args = {"W", "H", "stick_size"}},
    };
    // The reader only uses the TensorAccessor (ta::input) on the DRAM path; a sharded input reaches the
    // kernel through the borrowed scratch CB instead, so no kernel-side tensor binding is needed there.
    if (g.src_is_dram) {
        reader.tensor_bindings = {
            m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"input"}, .accessor_name = "input"}};
    }

    m2::ProgramSpec spec;
    spec.name = "plusone";
    spec.kernels = {std::move(reader)};
    spec.dataflow_buffers = {std::move(scratch_dfb)};
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = input.tensor_spec()}};
    spec.work_units = {m2::WorkUnitSpec{
        .name = "plusone_work", .kernels = {m2::KernelSpecName{"reader"}}, .target_nodes = g.all_cores}};

    return spec;
}

// create_invariant_run_args — the enqueue-invariant per-core work geometry (W/H/stick_size); identical
// across cores and a pure function of shape, which is in the cache key. Set once on a miss and retained.
m2::ProgramRunArgs PlusOneProgramFactory::create_invariant_run_args(
    const PlusoneParams& attrs, const Tensor& input, [[maybe_unused]] Tensor& output) {
    const auto g = compute_geometry(attrs, input);
    m2::ProgramRunArgs::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
    for (const auto& core : g.cores) {
        reader_args.runtime_arg_values.push_back({core, {{"W", g.W}, {"H", g.H}, {"stick_size", g.aligned_page_size}}});
    }
    m2::ProgramRunArgs invariant_args;
    invariant_args.kernel_run_args.push_back(std::move(reader_args));
    return invariant_args;
}

// create_per_enqueue_args — the per-call dynamic: the input tensor binding, re-applied via
// UpdateProgramRunArgs on every dispatch. plusone is in-place, so the input is both read and written.
m2::ProgramRunArgs PlusOneProgramFactory::create_per_enqueue_args(
    [[maybe_unused]] const PlusoneParams& attrs,
    const Tensor& input,
    [[maybe_unused]] Tensor& output,
    [[maybe_unused]] const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    m2::ProgramRunArgs run_args;
    run_args.tensor_args.emplace(
        m2::TensorParamName{"input"}, m2::ProgramRunArgs::TensorArgument{std::cref(input.mesh_tensor())});
    return run_args;
}

}  // namespace ttnn::experimental::prim
