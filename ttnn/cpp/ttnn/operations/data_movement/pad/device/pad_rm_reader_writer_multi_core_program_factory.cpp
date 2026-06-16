// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_rm_reader_writer_multi_core_program_factory.hpp"
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::tt_metal;
using namespace tt::constants;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

namespace {
// This is currently mostly hardcoded for resnet shapes
inline std::tuple<uint32_t, uint32_t, uint32_t, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t, uint32_t, uint32_t>
split_across_cores(CoreCoord grid_size, uint32_t nbatch, uint32_t ntiles_h, uint32_t ntiles_w) {
    uint32_t ncores, ncores_h, ncores_w, ntiles_per_core_h, ntiles_per_core_w, nbatch_per_core_h, ncores_per_batch_h;

    // each batch needs to be padded independently
    switch (nbatch) {
        case 1:
            ncores_h = 1;
            nbatch_per_core_h = 1;
            ntiles_per_core_h = 1;
            switch (ntiles_h) {
                case 2:
                    ncores_h = 2;
                    ntiles_per_core_h = 1;
                    break;
                case 4:
                    ncores_h = 4;
                    ntiles_per_core_h = 1;
                    break;
                case 8:
                    ncores_h = 8;
                    ntiles_per_core_h = 1;
                    break;
                case 64:
                    ncores_h = 8;
                    ntiles_per_core_h = 8;
                    break;
                default: TT_THROW("Unsupported ntiles_h value {}", ntiles_h);
            }
            ncores_per_batch_h = ncores_h;
            break;

        case 2:
            ncores_h = 1;
            ncores_per_batch_h = 1;
            nbatch_per_core_h = 1;
            ntiles_per_core_h = 1;
            switch (ntiles_h) {
                case 2:
                    ncores_per_batch_h = 2;
                    ncores_h = ncores_per_batch_h * nbatch;
                    ntiles_per_core_h = 1;
                    break;
                case 4:
                    ncores_per_batch_h = 4;
                    ncores_h = ncores_per_batch_h * nbatch;
                    ntiles_per_core_h = 1;
                    break;
                case 8:
                    ncores_per_batch_h = 4;
                    ncores_h = ncores_per_batch_h * nbatch;
                    ntiles_per_core_h = 2;
                    break;
                case 64:
                    ncores_per_batch_h = 4;
                    ncores_h = ncores_per_batch_h * nbatch;
                    ntiles_per_core_h = 16;
                    break;
                default: TT_THROW("Unsupported ntiles_h value {}", ntiles_h);
            }
            break;

        case 8:
            ncores_h = 8;
            ncores_per_batch_h = 1;
            nbatch_per_core_h = 1;
            ntiles_per_core_h = ntiles_h;
            break;

        default:
            TT_THROW("Unsupported nbatch value {} for pad operation. Supported values are 1, 2, and 8.", nbatch);

            // generic case -- TODO

            // one of the following will be 0 when grid_size.y != nbatch
            nbatch_per_core_h = nbatch / grid_size.y;   // floor
            ncores_per_batch_h = grid_size.y / nbatch;  // floor
            if (nbatch == grid_size.y) {
                nbatch_per_core_h = 1;
                ncores_per_batch_h = 1;
            }

            // currently uses hardcoded values for resnet50
            // TT_ASSERT(ntiles_h == 1 || ntiles_h == 2 || ntiles_h == 4 || ntiles_h == 16, "Only Resnet50 shapes are
            // supported in multicore version for now."); TT_ASSERT(ntiles_w == 64, "Only Resnet50 shapes are supported
            // in multicore version for now.");

            TT_ASSERT(nbatch <= grid_size.y, "Unsupported case with nbatch > grid_size.y!");

            if (nbatch_per_core_h == 0) {
                // there are multiple cores along h per batch
                nbatch_per_core_h = 1;
            } else if (ncores_per_batch_h == 0) {
                // unsupported case. TODO.
                TT_THROW(
                    "Unsupported configuration: multiple batches per core along height dimension "
                    "(nbatch={}, grid_size.y={})",
                    nbatch,
                    grid_size.y);
                // there are multiple batch per core along h
                // ncores_per_batch_h = 1;
            } else {
                TT_THROW("Something went terribly wrong in splitting across cores");
            }
            break;
    }

    switch (ntiles_w) {
        case 2: ncores_w = 2; break;
        case 4: ncores_w = 4; break;
        case 8:
        case 64: ncores_w = 8; break;
        default: TT_THROW("Unsupported ntiles_w value {}", ntiles_w);
    }
    ncores = ncores_h * ncores_w;
    ntiles_per_core_w = ntiles_w / ncores_w;
    std::set<CoreRange> all_cores;
    std::set<CoreRange> core_range;

    all_cores.insert(CoreRange(CoreCoord(0, 0), CoreCoord(ncores_w - 1, ncores_h - 1)));
    core_range.insert(CoreRange(CoreCoord(0, 0), CoreCoord(ncores_w - 1, ncores_h - 1)));

    return std::make_tuple(
        ncores,
        ncores_h,
        ncores_w,
        all_cores,
        core_range,
        ntiles_per_core_h,
        ntiles_per_core_w,
        nbatch_per_core_h,
        ncores_per_batch_h);
}

// Allocate the on-device pad-value const tensor.  Returned as an op-owned tensor in
// ProgramArtifacts::op_owned_tensors: the framework parks it at a stable address for the cached
// Program's life (allocated once on a cache miss, reused on a hit).  The kernel reads it through a
// TensorAccessor (ta::pad) exactly like the io tensors.
// NOTE: The const buffer is always in L1.
// TODO: make a local buffer for each core?
Tensor build_pad_value_const_tensor_mc(const PadInputs& tensor_args, float pad_value) {
    MeshDevice* device = tensor_args.input.device();
    uint32_t pad_value_const_buffer_size = 32;  // noc transfers in chunks of 32
    auto pad_value_const_buffer =
        tt::tt_metal::HostBuffer(std::vector<bfloat16>(pad_value_const_buffer_size, bfloat16(pad_value)));
    return Tensor(
               std::move(pad_value_const_buffer),
               ttnn::Shape({1, 1, 1, pad_value_const_buffer_size}),
               DataType::BFLOAT16,
               Layout::ROW_MAJOR)
        .to_device(device, MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1});
}

}  // namespace

ttnn::device_operation::ProgramArtifacts PadRmReaderWriterMultiCoreProgramFactory::create_program_spec(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& output_padded_shape = operation_attributes.output_padded_shape;
    const auto& pad_value = operation_attributes.pad_value;

    auto output_shape = output_padded_shape;

    uint32_t unpadded_row_size_nbytes = a.padded_shape()[3] * a.element_size();
    uint32_t padded_row_size_nbytes = output_shape[3] * a.element_size();  // Assuming output is same datatype as input
    TT_ASSERT(
        unpadded_row_size_nbytes <= padded_row_size_nbytes, "Padded output tensor size should be >= input tensor size");

    distributed::MeshDevice* device = a.device();

    // uint32_t ntiles_h = output_tensor_shape[0] * output_tensor_shape[1] * output_tensor_shape[2] / TILE_HEIGHT;
    uint32_t ntiles_h = output_padded_shape[2] / TILE_HEIGHT;
    uint32_t ntiles_w = output_padded_shape[3] / TILE_WIDTH;

    auto grid_size = device->compute_with_storage_grid_size();
    uint32_t nbatch = output_padded_shape[0];
    // first the batch dim is distributed along H, and within each batch then the tiles are distributed.
    auto
        [ncores,
         ncores_h,
         ncores_w,
         all_cores,
         core_range,
         ntiles_per_core_h,
         ntiles_per_core_w,
         nbatch_per_core_h,
         ncores_per_batch_h] = split_across_cores(grid_size, nbatch, ntiles_h, ntiles_w);

    [[maybe_unused]] int32_t src_nbytes_per_core_w = ntiles_per_core_w * TILE_WIDTH * a.element_size();
    int32_t dst_nbytes_per_core_w = ntiles_per_core_w * TILE_WIDTH * output.element_size();

    Buffer* src0_buffer = a.buffer();
    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    uint32_t cb_npages = 16;  // multibuffering for perf
    // uint32_t cb_npages = 1; // multibuffering for perf
    uint32_t cb_page_alignment = std::max(tt::constants::TILE_WIDTH, src0_buffer->alignment());
    uint32_t cb_pagesize =
        static_cast<uint32_t>(std::ceil((float)dst_nbytes_per_core_w / cb_page_alignment)) * cb_page_alignment;
    tt::DataFormat in_df = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());

    uint32_t packed_pad_value;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32) {
        packed_pad_value = pad_value;
    } else if (a.dtype() == DataType::UINT16) {
        packed_pad_value = pack_two_uint16_into_uint32({0, float_to_uint16(pad_value)});
    } else {
        packed_pad_value = pack_two_bfloat16_into_uint32({bfloat16(0.0f), bfloat16(pad_value)});
    }

    // ---- Op-owned pad-value const tensor (allocate first; bind against the vector element) ----
    std::vector<Tensor> op_owned;
    op_owned.reserve(1);
    op_owned.push_back(build_pad_value_const_tensor_mc(tensor_args, pad_value));
    const Tensor& pad_const = op_owned[0];

    // ---- ProgramSpec (immutable) ----
    m2::ProgramSpec spec;
    spec.name = "pad_rm_reader_writer_multi_core";

    // c_0 (in0): row buffer the reader fills (data + pad) and the writer drains to the output.
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"in0"},
            .entry_size = cb_pagesize,
            .num_entries = cb_npages,
            .data_format_metadata = in_df},
    };

    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"src"}, .spec = a.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"dst"}, .spec = output.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"pad"}, .spec = pad_const.tensor_spec()},
    };

    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/"
                                        "reader_pad_dims_rm_interleaved_m2.cpp"},
        .dfb_bindings = {m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"in0"},
            .accessor_name = "in0",
            .endpoint_type = m2::DFBEndpointType::PRODUCER}},
        .tensor_bindings =
            {m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"src"}, .accessor_name = "src"},
             m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"pad"}, .accessor_name = "pad"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_unpadded_W",
                  "num_total_W",
                  "num_unpadded_Z",
                  "num_total_Z",
                  "num_unpadded_Y",
                  "num_total_Y",
                  "unpadded_X_nbytes",
                  "padded_X_nbytes",
                  "padded_X_diff_nbytes",
                  "pad_value_packed",
                  "start_src_stick_id",
                  "start_src_stick_wi",
                  "start_src_stick_offset",
                  "num_local_Y",
                  "num_local_unpadded_Y",
                  "full_unpadded_X_nbytes",
                  "num_local_W"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::READER},
    };

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/"
                                        "writer_pad_dims_rm_interleaved_m2.cpp"},
        .dfb_bindings = {m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"in0"},
            .accessor_name = "in0",
            .endpoint_type = m2::DFBEndpointType::CONSUMER}},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"dst"}, .accessor_name = "dst"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_total_W",
                  "num_total_Z",
                  "num_total_Y",
                  "num_total_X",
                  "padded_X_nbytes",
                  "start_dst_stick_id",
                  "start_dst_stick_wi",
                  "num_local_Y",
                  "num_local_unpadded_Y",
                  "full_padded_X_nbytes",
                  "dst_stick_offset",
                  "num_local_W"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
    };

    spec.kernels = {reader, writer};

    // Local DFB (in0): reader PRODUCER -> writer CONSUMER.  Both kernels run on all_cores, so a
    // single WorkUnitSpec hosts both endpoints on every node.
    spec.work_units = std::vector<m2::WorkUnitSpec>{
        m2::WorkUnitSpec{
            .name = "multi_core",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}},
            .target_nodes = all_cores},
    };

    // ---- ProgramRunArgs (mutable) ----
    m2::ProgramRunArgs run;
    m2::KernelRunArgs reader_run{.kernel = m2::KernelSpecName{"reader"}};
    m2::KernelRunArgs writer_run{.kernel = m2::KernelSpecName{"writer"}};

    uint32_t start_src_stick_id = 0;
    uint32_t start_dst_stick_id = 0;
    uint32_t start_src_stick_wi = 0;  // start of stick segment for 2d decomp
    uint32_t start_dst_stick_wi = 0;
    int32_t local_nsticks = ntiles_per_core_h * TILE_HEIGHT;
    for (int32_t b = 0; b < nbatch; ++b) {
        int32_t rem_src_nsticks = a.padded_shape()[2];
        for (uint32_t j = 0; j < ncores_per_batch_h; ++j) {
            uint32_t num_local_unpadded_nsticks = local_nsticks;
            if (rem_src_nsticks - local_nsticks >= 0) {
                // not reached padding sticks yet
                rem_src_nsticks -= local_nsticks;
            } else {
                num_local_unpadded_nsticks = rem_src_nsticks;
                rem_src_nsticks = 0;
            }
            start_src_stick_wi = 0;
            start_dst_stick_wi = 0;
            int32_t rem_src_stick_size_nbytes = unpadded_row_size_nbytes;
            for (uint32_t i = 0; i < ncores_w; ++i) {
                CoreCoord core = {i, (b * ncores_per_batch_h) + j};
                const m2::NodeCoord node{core};
                uint32_t curr_stick_size_nbytes = 0;
                int32_t curr_stick_diff_nbytes = 0;
                if (rem_src_stick_size_nbytes - dst_nbytes_per_core_w >= 0) {
                    // no padding on this core
                    curr_stick_size_nbytes = dst_nbytes_per_core_w;
                    rem_src_stick_size_nbytes -= dst_nbytes_per_core_w;
                } else {
                    // this core has padding
                    curr_stick_size_nbytes = rem_src_stick_size_nbytes;
                    curr_stick_diff_nbytes = dst_nbytes_per_core_w - curr_stick_size_nbytes;
                    rem_src_stick_size_nbytes = 0;
                }
                // The legacy buffer-address RTA slots (0=src, 1=dst, 13=pad-value const) are now
                // carried by the src/dst/pad TensorBindings; only the scalar per-core args survive.
                reader_run.runtime_arg_values.push_back(
                    {node,
                     {{"num_unpadded_W", static_cast<uint32_t>(a.padded_shape()[0])},
                      {"num_total_W", static_cast<uint32_t>(output_shape[0])},
                      {"num_unpadded_Z", static_cast<uint32_t>(a.padded_shape()[1])},
                      {"num_total_Z", static_cast<uint32_t>(output_shape[1])},
                      {"num_unpadded_Y", static_cast<uint32_t>(a.padded_shape()[2])},
                      {"num_total_Y", static_cast<uint32_t>(output_shape[2])},
                      {"unpadded_X_nbytes", curr_stick_size_nbytes},
                      {"padded_X_nbytes", static_cast<uint32_t>(dst_nbytes_per_core_w)},
                      {"padded_X_diff_nbytes", static_cast<uint32_t>(curr_stick_diff_nbytes)},
                      {"pad_value_packed", packed_pad_value},
                      {"start_src_stick_id", start_src_stick_id},
                      {"start_src_stick_wi", start_src_stick_wi},
                      {"start_src_stick_offset", start_src_stick_wi * a.element_size()},
                      {"num_local_Y", static_cast<uint32_t>(local_nsticks)},
                      {"num_local_unpadded_Y", num_local_unpadded_nsticks},
                      {"full_unpadded_X_nbytes", unpadded_row_size_nbytes},
                      {"num_local_W", nbatch_per_core_h}}});

                writer_run.runtime_arg_values.push_back(
                    {node,
                     {{"num_total_W", static_cast<uint32_t>(output_shape[0])},
                      {"num_total_Z", static_cast<uint32_t>(output_shape[1])},
                      {"num_total_Y", static_cast<uint32_t>(output_shape[2])},
                      {"num_total_X", static_cast<uint32_t>(output_shape[3])},
                      {"padded_X_nbytes", static_cast<uint32_t>(dst_nbytes_per_core_w)},
                      {"start_dst_stick_id", start_dst_stick_id},
                      {"start_dst_stick_wi", start_dst_stick_wi},
                      {"num_local_Y", static_cast<uint32_t>(local_nsticks)},
                      {"num_local_unpadded_Y", num_local_unpadded_nsticks},
                      {"full_padded_X_nbytes", padded_row_size_nbytes},
                      {"dst_stick_offset", start_dst_stick_wi * output.element_size()},
                      {"num_local_W", nbatch_per_core_h}}});

                start_src_stick_wi += ntiles_per_core_w * TILE_WIDTH;
                start_dst_stick_wi += ntiles_per_core_w * TILE_WIDTH;
            }  // for ncores_w
            start_src_stick_id += num_local_unpadded_nsticks;
            start_dst_stick_id += local_nsticks;
        }  // for ncores_h
    }

    run.kernel_run_args = {reader_run, writer_run};
    run.tensor_args = {
        {m2::TensorParamName{"src"}, a.mesh_tensor()},
        {m2::TensorParamName{"dst"}, output.mesh_tensor()},
        {m2::TensorParamName{"pad"}, pad_const.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec), .run_params = std::move(run), .op_owned_tensors = std::move(op_owned)};
}

}  // namespace ttnn::prim
