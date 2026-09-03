// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_planner_nanobind.hpp"

#include <cstddef>
#include <cstdint>
#include <new>
#include <optional>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include "ttnn/cpp/ttnn/kernel_lib/host/reduce_host.hpp"

namespace ttnn::operations::reduction::detail {
namespace {

namespace host = ttnn::kernel_lib::host;

}  // namespace

void bind_reduce_planner(nb::module_& mod) {
    auto planner = mod.def_submodule(
        "planner",
        "Host-side reduction planning and compile-time argument serialization. Planning never executes a kernel.");

    nb::enum_<tt::tt_metal::ReduceOpMath>(planner, "ReduceMath")
        .value("SUM", tt::tt_metal::ReduceOpMath::SUM)
        .value("AVG", tt::tt_metal::ReduceOpMath::AVG)
        .value("MAX", tt::tt_metal::ReduceOpMath::MAX)
        .value("MIN", tt::tt_metal::ReduceOpMath::MIN);
    nb::enum_<tt::tt_metal::ReduceOpDim>(planner, "ReduceDimension")
        .value("ROW", tt::tt_metal::ReduceOpDim::W)
        .value("COLUMN", tt::tt_metal::ReduceOpDim::H)
        .value("SCALAR", tt::tt_metal::ReduceOpDim::HW);
    nb::enum_<ReduceFp32Mode>(planner, "ReduceFp32Mode")
        .value("FAST", ReduceFp32Mode::Fast)
        .value("ACCURATE", ReduceFp32Mode::Accurate);
    nb::enum_<ttnn::kernel_lib::ReducePath>(planner, "ReducePath")
        .value("TILED", ttnn::kernel_lib::ReducePath::Tiled)
        .value("DENSE_ROW_MAJOR", ttnn::kernel_lib::ReducePath::DenseRowMajor);
    nb::enum_<ttnn::kernel_lib::ReduceAccumulationMode>(planner, "ReduceAccumulationMode")
        .value("NONE", ttnn::kernel_lib::ReduceAccumulationMode::None)
        .value("INTERMEDIATE", ttnn::kernel_lib::ReduceAccumulationMode::Intermediate)
        .value("FINAL", ttnn::kernel_lib::ReduceAccumulationMode::Final);
    nb::enum_<compute_kernel_lib::ReduceAlgorithm>(planner, "ReduceAlgorithm")
        .value("REDUCE_TILE", compute_kernel_lib::ReduceAlgorithm::ReduceTile)
        .value("ACCUMULATE_VIA_ADD", compute_kernel_lib::ReduceAlgorithm::AccumulateViaAdd);
    nb::enum_<compute_kernel_lib::ReduceInputPolicy>(planner, "ReduceInputPolicy")
        .value("WAIT_AND_POP_PER_TILE", compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile)
        .value("BULK_WAIT_BULK_POP", compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop)
        .value("WAIT_UPFRONT_NO_POP", compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop)
        .value("NO_WAIT_NO_POP", compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop)
        .value("CHUNKED_WAIT_CHUNKED_POP", compute_kernel_lib::ReduceInputPolicy::ChunkedWaitChunkedPop);
    nb::enum_<compute_kernel_lib::AccumulateReloadMode>(planner, "AccumulateReloadMode")
        .value("FOLD_VIA_ADD", compute_kernel_lib::AccumulateReloadMode::FoldViaAdd)
        .value("COPY_SEED_PAIRS", compute_kernel_lib::AccumulateReloadMode::CopySeedPairs)
        .value("COPY_SEED_UNIFORM", compute_kernel_lib::AccumulateReloadMode::CopySeedUniform)
        .value("COPY_SEED_SFPU_ADD", compute_kernel_lib::AccumulateReloadMode::CopySeedSfpuAdd)
        .value("COPY_SEED_ZERO_PAIR", compute_kernel_lib::AccumulateReloadMode::CopySeedZeroPair);
    nb::enum_<compute_kernel_lib::ReduceDataFormatReconfigMode>(planner, "ReduceDataFormatReconfigMode")
        .value("NONE", compute_kernel_lib::ReduceDataFormatReconfigMode::NONE)
        .value("INPUT", compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT)
        .value("OUTPUT", compute_kernel_lib::ReduceDataFormatReconfigMode::OUTPUT)
        .value("INPUT_AND_OUTPUT", compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT);
    nb::enum_<compute_kernel_lib::ReduceWithinTile>(planner, "ReduceWithinTile")
        .value("COLLAPSE", compute_kernel_lib::ReduceWithinTile::Collapse)
        .value("SKIP", compute_kernel_lib::ReduceWithinTile::Skip);
    nb::enum_<compute_kernel_lib::ReducePartialMode>(planner, "ReducePartialMode")
        .value("NONE", compute_kernel_lib::ReducePartialMode::None)
        .value("SCALER", compute_kernel_lib::ReducePartialMode::Scaler)
        .value("MASK", compute_kernel_lib::ReducePartialMode::Mask);
    nb::enum_<host::ReduceCbRole>(planner, "ReduceCbRole")
        .value("INPUT", host::ReduceCbRole::Input)
        .value("OUTPUT", host::ReduceCbRole::Output)
        .value("AUXILIARY", host::ReduceCbRole::Auxiliary)
        .value("ROW_MAJOR_STAGING", host::ReduceCbRole::RowMajorStaging)
        .value("TILED_SCRATCH", host::ReduceCbRole::TiledScratch)
        .value("ACCUMULATOR", host::ReduceCbRole::Accumulator)
        .value("PADDING_IDENTITY", host::ReduceCbRole::PaddingIdentity);
    nb::enum_<host::ReduceCbAlias>(planner, "ReduceCbAlias")
        .value("NONE", host::ReduceCbAlias::None)
        .value("INPUT_TENSOR", host::ReduceCbAlias::InputTensor)
        .value("OUTPUT_TENSOR", host::ReduceCbAlias::OutputTensor);
    nb::enum_<ttnn::kernel_lib::ReduceAuxiliaryTileType>(planner, "ReduceAuxiliaryTileType")
        .value("FIRST_ROW", ttnn::kernel_lib::ReduceAuxiliaryTileType::FirstRow)
        .value("FIRST_COLUMN", ttnn::kernel_lib::ReduceAuxiliaryTileType::FirstColumn)
        .value("FIRST_ROW_PER_FACE_ROW", ttnn::kernel_lib::ReduceAuxiliaryTileType::FirstRowPerFaceRow)
        .value("ZERO", ttnn::kernel_lib::ReduceAuxiliaryTileType::Zero);

    nb::class_<host::ReduceHardwareConfig>(planner, "ReduceHardwareConfig")
        .def(
            "__init__",
            [](host::ReduceHardwareConfig* self,
               tt::ARCH arch,
               bool fp32_dest_acc_en,
               bool dst_full_sync_en,
               std::size_t available_l1_bytes) {
                new (self) host::ReduceHardwareConfig{
                    .arch = arch,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .dst_full_sync_en = dst_full_sync_en,
                    .available_l1_bytes = available_l1_bytes};
            },
            nb::arg("arch"),
            nb::arg("fp32_dest_acc_en"),
            nb::arg("dst_full_sync_en"),
            nb::arg("available_l1_bytes"))
        .def_rw("arch", &host::ReduceHardwareConfig::arch)
        .def_rw("fp32_dest_acc_en", &host::ReduceHardwareConfig::fp32_dest_acc_en)
        .def_rw("dst_full_sync_en", &host::ReduceHardwareConfig::dst_full_sync_en)
        .def_rw("available_l1_bytes", &host::ReduceHardwareConfig::available_l1_bytes);

    nb::class_<host::ReduceChunkPlan>(planner, "ReduceChunkPlan")
        .def_ro("reduce_axis_tiles", &host::ReduceChunkPlan::reduce_axis_tiles)
        .def_ro("output_tiles", &host::ReduceChunkPlan::output_tiles)
        .def_ro("buffers", &host::ReduceChunkPlan::buffers)
        .def_prop_ro("input_tiles", &host::ReduceChunkPlan::input_tiles);

    nb::class_<host::ReduceCbRequirement>(planner, "ReduceCbRequirement")
        .def_ro("role", &host::ReduceCbRequirement::role)
        .def_prop_ro(
            "data_format",
            [](const host::ReduceCbRequirement& self) { return static_cast<std::uint8_t>(self.data_format); },
            "Raw tt::DataFormat value; DataFormat is not otherwise exposed to Python.")
        .def_ro("page_size", &host::ReduceCbRequirement::page_size)
        .def_ro("page_count", &host::ReduceCbRequirement::page_count)
        .def_ro("total_size_bytes", &host::ReduceCbRequirement::total_size_bytes)
        .def_ro("alias", &host::ReduceCbRequirement::alias)
        .def_prop_ro("owns_l1", &host::ReduceCbRequirement::owns_l1);

    nb::class_<host::ReduceAuxiliaryTileSpec>(planner, "ReduceAuxiliaryTileSpec")
        .def_ro("value", &host::ReduceAuxiliaryTileSpec::value)
        .def_ro("type", &host::ReduceAuxiliaryTileSpec::type)
        .def_ro("num_valid_elements", &host::ReduceAuxiliaryTileSpec::num_valid_elements);

    nb::class_<host::DenseRowMajorPlan>(planner, "DenseRowMajorPlan")
        .def_ro("H_logical", &host::DenseRowMajorPlan::H_logical)
        .def_ro("W_logical", &host::DenseRowMajorPlan::W_logical)
        .def_ro("Ht_rm", &host::DenseRowMajorPlan::Ht_rm)
        .def_ro("Wt", &host::DenseRowMajorPlan::Wt)
        .def_ro("rm_rows_per_tile", &host::DenseRowMajorPlan::rm_rows_per_tile)
        .def_ro("wt_tiles_per_chunk", &host::DenseRowMajorPlan::wt_tiles_per_chunk)
        .def_ro("ht_tiles_per_chunk", &host::DenseRowMajorPlan::ht_tiles_per_chunk)
        .def_ro("chunk_row_bytes", &host::DenseRowMajorPlan::chunk_row_bytes)
        .def_ro("rm_staging_page_size", &host::DenseRowMajorPlan::rm_staging_page_size)
        .def_ro("padding_identity_bits", &host::DenseRowMajorPlan::padding_identity_bits)
        .def_ro("src_datum_size", &host::DenseRowMajorPlan::src_datum_size)
        .def_ro("dst_datum_size", &host::DenseRowMajorPlan::dst_datum_size)
        .def_ro("staging_buffers", &host::DenseRowMajorPlan::staging_buffers);

    auto py_plan = nb::class_<host::ReducePlan>(planner, "ReducePlan");
    py_plan.def_ro("path", &host::ReducePlan::path)
        .def_ro("reduce_math", &host::ReducePlan::reduce_math)
        .def_ro("reduce_dim", &host::ReducePlan::reduce_dim)
        .def_ro("fp32_mode", &host::ReducePlan::fp32_mode)
        .def_ro("algorithm", &host::ReducePlan::algorithm)
        .def_ro("input_policy", &host::ReducePlan::input_policy)
        .def_ro("reload_mode", &host::ReducePlan::reload_mode)
        .def_ro("reconfig_mode", &host::ReducePlan::reconfig_mode)
        .def_ro("within_tile", &host::ReducePlan::within_tile)
        .def_ro("chunk", &host::ReducePlan::chunk)
        .def_ro("Ht", &host::ReducePlan::Ht)
        .def_ro("Wt", &host::ReducePlan::Wt)
        .def_ro("batches", &host::ReducePlan::batches)
        .def_ro("input_row_stride_tiles", &host::ReducePlan::input_row_stride_tiles)
        .def_ro("reduce_factor", &host::ReducePlan::reduce_factor)
        .def_ro("post_scale", &host::ReducePlan::post_scale)
        .def_ro("partial_mode", &host::ReducePlan::partial_mode)
        .def_ro("auxiliary_tiles", &host::ReducePlan::auxiliary_tiles)
        .def_ro("partial_reduce_axis_elements", &host::ReducePlan::partial_reduce_axis_elements)
        .def_ro("row_major", &host::ReducePlan::row_major)
        .def_ro("cb_requirements", &host::ReducePlan::cb_requirements)
        .def_ro("total_owned_l1_bytes", &host::ReducePlan::total_owned_l1_bytes)
        .def(
            "compile_time_args",
            [](const host::ReducePlan& self,
               std::uint32_t input_cb_id,
               std::uint32_t auxiliary_cb_id,
               std::uint32_t output_cb_id) {
                return host::ReduceCallArgs(
                           self,
                           {.input_cb_id = input_cb_id,
                            .auxiliary_cb_id = auxiliary_cb_id,
                            .output_cb_id = output_cb_id})
                    .get_compile_time_args();
            },
            nb::arg("input_cb_id"),
            nb::arg("auxiliary_cb_id"),
            nb::arg("output_cb_id"),
            "Serialize one complete non-accumulating call for the caller's CB namespace.");

    nb::class_<host::ReduceCallConfig>(planner, "ReduceCallConfig")
        .def(
            "__init__",
            [](host::ReduceCallConfig* self,
               const tt::tt_metal::TensorSpec& input_spec,
               const tt::tt_metal::TensorSpec& output_spec,
               tt::tt_metal::ReduceOpMath reduce_math,
               tt::tt_metal::ReduceOpDim reduce_dim,
               float scalar,
               ReduceFp32Mode fp32_mode,
               std::optional<std::size_t> max_input_cb_bytes) {
                new (self) host::ReduceCallConfig{
                    .input_spec = input_spec,
                    .output_spec = output_spec,
                    .reduce_math = reduce_math,
                    .reduce_dim = reduce_dim,
                    .scalar = scalar,
                    .fp32_mode = fp32_mode,
                    .max_input_cb_bytes = max_input_cb_bytes};
            },
            nb::arg("input_spec"),
            nb::arg("output_spec"),
            nb::arg("reduce_math"),
            nb::arg("reduce_dim"),
            nb::arg("scalar"),
            nb::arg("fp32_mode"),
            nb::arg("max_input_cb_bytes") = nb::none())
        .def_rw("input_spec", &host::ReduceCallConfig::input_spec)
        .def_rw("output_spec", &host::ReduceCallConfig::output_spec)
        .def_rw("reduce_math", &host::ReduceCallConfig::reduce_math)
        .def_rw("reduce_dim", &host::ReduceCallConfig::reduce_dim)
        .def_rw("scalar", &host::ReduceCallConfig::scalar)
        .def_rw("fp32_mode", &host::ReduceCallConfig::fp32_mode)
        .def_rw("max_input_cb_bytes", &host::ReduceCallConfig::max_input_cb_bytes);

    nb::class_<host::ReduceSequenceCbIds>(planner, "ReduceSequenceCbIds")
        .def(
            "__init__",
            [](host::ReduceSequenceCbIds* self,
               std::uint32_t auxiliary_cb_id,
               std::uint32_t accumulator_cb_id,
               std::uint32_t output_cb_id) {
                new (self) host::ReduceSequenceCbIds{
                    .auxiliary_cb_id = auxiliary_cb_id,
                    .accumulator_cb_id = accumulator_cb_id,
                    .output_cb_id = output_cb_id};
            },
            nb::arg("auxiliary_cb_id"),
            nb::arg("accumulator_cb_id"),
            nb::arg("output_cb_id"))
        .def_rw("auxiliary_cb_id", &host::ReduceSequenceCbIds::auxiliary_cb_id)
        .def_rw("accumulator_cb_id", &host::ReduceSequenceCbIds::accumulator_cb_id)
        .def_rw("output_cb_id", &host::ReduceSequenceCbIds::output_cb_id);

    nb::class_<host::ReduceCallPlan>(planner, "ReduceCallPlan")
        .def_ro("input_cb_id", &host::ReduceCallPlan::input_cb_id)
        .def_ro("auxiliary_cb_id", &host::ReduceCallPlan::auxiliary_cb_id)
        .def_ro("output_cb_id", &host::ReduceCallPlan::output_cb_id)
        .def_ro("accumulator_cb_id", &host::ReduceCallPlan::accumulator_cb_id)
        .def_ro("accumulation_mode", &host::ReduceCallPlan::accumulation_mode)
        .def_ro("accumulation_index", &host::ReduceCallPlan::accumulation_index)
        .def_ro("plan", &host::ReduceCallPlan::plan)
        .def_prop_ro(
            "compile_time_args",
            [](const host::ReduceCallPlan& self) { return host::ReduceCallArgs(self).get_compile_time_args(); },
            "One complete, independently decodable reduce call record.");

    nb::class_<host::ReduceSequencePlan>(planner, "ReduceSequencePlan")
        .def_ro("calls", &host::ReduceSequencePlan::calls)
        .def_prop_ro("call_count", [](const host::ReduceSequencePlan& self) { return self.calls.size(); })
        .def_prop_ro(
            "compile_time_args",
            &host::ReduceSequencePlan::get_compile_time_args,
            "The flat call-count-plus-calls suffix without any kernel-owned prefix.")
        .def(
            "append_to",
            [](const host::ReduceSequencePlan& self, nb::list compile_time_args) {
                for (const auto arg : self.get_compile_time_args()) {
                    compile_time_args.append(nb::cast(arg));
                }
            },
            nb::arg("compile_time_args"),
            "Append call count and complete calls to a caller-owned compile-time argument list.")
        .def("__len__", [](const host::ReduceSequencePlan& self) { return self.calls.size(); });

    planner.def(
        "make_reduce_plan",
        nb::overload_cast<
            const tt::tt_metal::TensorSpec&,
            const tt::tt_metal::TensorSpec&,
            tt::tt_metal::ReduceOpMath,
            tt::tt_metal::ReduceOpDim,
            float,
            ReduceFp32Mode,
            const host::ReduceHardwareConfig&,
            std::optional<std::size_t>>(&host::make_reduce_plan),
        nb::arg("input_spec"),
        nb::arg("output_spec"),
        nb::arg("reduce_math"),
        nb::arg("reduce_dim"),
        nb::arg("scalar"),
        nb::arg("fp32_mode"),
        nb::arg("hardware"),
        nb::arg("max_input_cb_bytes") = nb::none(),
        "Plan one reduction without executing it.");
    planner.def(
        "make_reduce_sequence_plan",
        nb::overload_cast<
            const std::vector<host::ReduceCbConfig>&,
            const host::ReduceSequenceCbIds&,
            const host::ReduceHardwareConfig&>(&host::make_reduce_sequence_plan),
        nb::arg("reductions"),
        nb::arg("cb_ids"),
        nb::arg("hardware"),
        "Plan an explicitly ordered sequence of reductions over distinct input CB IDs.");
}

}  // namespace ttnn::operations::reduction::detail
