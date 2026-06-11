// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "plusone_program_factory.hpp"
#include "plusone_device_operation_types.hpp"

#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operation.hpp"

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace {

constexpr const char* PLUSONE_READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/experimental/plusone/device/kernels/reader_plusone_interleaved.cpp";

}  // namespace

ttnn::device_operation::ProgramArtifacts PlusOneProgramFactory::create_program_artifacts(
    const PlusoneParams& operation_attributes, const Tensor& input, Tensor& /*tensor_return_value*/) {
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_unit_size = input.element_size();

    CoreRangeSet all_cores = CoreRangeSet(std::vector{CoreRange({0, 0}, {0, 0})});
    uint32_t num_cores = 1;  // single-core

    if (operation_attributes.sub_core_grids.has_value()) {
        all_cores = operation_attributes.sub_core_grids.value();
        num_cores = all_cores.num_cores();
    }

    const auto& input_shape = input.padded_shape();
    uint32_t W = input_shape[-1];
    uint32_t H = 1;
    if (!input.is_sharded() && input_shape.size() > 1) {
        for (uint32_t i = 0; i < input_shape.size() - 1; ++i) {
            H *= input_shape[i];
        }
    }

    uint32_t num_input_units = W;
    auto* src_buffer = input.buffer();
    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const uint32_t page_alignment =
        src_is_dram ? tt::tt_metal::hal::get_dram_alignment() : tt::tt_metal::hal::get_l1_alignment();
    uint32_t aligned_input_page_size = tt::align(num_input_units * input_unit_size, page_alignment);

    m2::ProgramSpec spec;
    spec.name = "plusone";

    // The input tensor parameter — the reader's TensorAccessor base address is filled from it on cache
    // miss and refreshed on a cache hit (UpdateTensorArgs). The op is in-place, so this same tensor is
    // both read and written.
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = input.tensor_spec()}};

    // The "in" DFB. When the input is sharded, borrow the input buffer's L1 storage so the framework can
    // re-apply the globally-allocated address on a program-cache hit and the increment happens in place
    // with no NoC traffic. For the interleaved path the DFB is plain L1 scratch and the buffer address is
    // taken from the TensorAccessor("input") binding inside the reader instead.
    m2::DataflowBufferSpec in_dfb{
        .unique_id = m2::DFBSpecName{"in"},
        .entry_size = aligned_input_page_size,
        .num_entries = 1,
        .data_format_metadata = input_cb_data_format,
    };
    if (input.is_sharded()) {
        in_dfb.borrowed_from = m2::TensorParamName{"input"};
    }
    spec.dataflow_buffers = {std::move(in_dfb)};

    // The single reader kernel both NoC-reads and NoC-writes the SAME buffer (in-place), so it binds the
    // one DFB as BOTH producer and consumer (single kernel, both roles OK). Formerly-positional
    // compile-time args become a named Table; the TensorAccessor args that were appended positionally are
    // now carried by the tensor binding.
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{PLUSONE_READER_KERNEL_PATH},
        .dfb_bindings =
            {m2::ProducerOf(m2::DFBSpecName{"in"}, "cb_id_in0"), m2::ConsumerOf(m2::DFBSpecName{"in"}, "cb_id_in0")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"input"}, .accessor_name = "s0_args"}},
        .compile_time_args =
            {{"src0_is_dram", static_cast<uint32_t>(src_is_dram)},
             {"stick_size", aligned_input_page_size},
             {"W", W},
             {"H", H},
             {"skip_negative_entries", static_cast<uint32_t>(operation_attributes.skip_negative_entries)}},
        .hw_config = m2::DataMovementHardwareConfig{.gen1_config = m2::DataMovementHardwareConfig::Gen1Config{}},
    };

    spec.kernels = {std::move(reader)};
    spec.work_units = {
        m2::WorkUnitSpec{.name = "plusone", .kernels = {m2::KernelSpecName{"reader"}}, .target_nodes = all_cores}};

    // The reader declares no runtime args (everything is a compile-time arg or comes from the tensor
    // binding), but a KernelRunArgs must still be present for every kernel; emit an empty per-core entry
    // on each target core to mirror the legacy SetRuntimeArgs over all_cores.
    m2::ProgramRunArgs run_args;
    m2::ProgramRunArgs::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
    for (const auto& core : corerange_to_cores(all_cores, num_cores, /*row_wise=*/true)) {
        reader_args.runtime_arg_values.push_back({core, {}});
    }
    run_args.kernel_run_args.push_back(std::move(reader_args));

    // Tensor argument: the framework fills the reader's tensor accessor base address from this, and
    // refreshes it on a cache hit (UpdateTensorArgs).
    run_args.tensor_args.emplace(
        m2::TensorParamName{"input"}, m2::ProgramRunArgs::TensorArgument{std::cref(input.mesh_tensor())});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
