// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_rm_reader_writer_multi_core_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>

#include <cmath>
#include <filesystem>
#include <set>
#include <tuple>
#include <vector>

#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
using namespace tt::constants;

namespace ttnn::prim::qsr {
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

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
        CoreRangeSet(all_cores),
        CoreRangeSet(core_range),
        ntiles_per_core_h,
        ntiles_per_core_w,
        nbatch_per_core_h,
        ncores_per_batch_h);
}

// Allocate and fill the op-owned pad-value const tensor.  Mirrors the legacy
// build_pad_value_const_tensor_mc(): build a host tensor holding the pad value, then write it to
// an L1 interleaved device allocation.  enqueue_write_tensor() returns a sole-owner MeshTensor,
// which is what ProgramArtifacts::op_owned_tensors requires (the framework keeps it alive at a
// stable address for the cached Program; see #44565 for why sole ownership matters).
MeshTensor build_pad_value_const_mesh_tensor(const PadInputs& tensor_args, float pad_value) {
    MeshDevice* device = tensor_args.input.device();
    uint32_t pad_value_const_buffer_size = 32;  // noc transfers in chunks of 32
    auto host_buffer =
        tt::tt_metal::HostBuffer(std::vector<bfloat16>(pad_value_const_buffer_size, bfloat16(pad_value)));
    Tensor host_pad(
        std::move(host_buffer),
        ttnn::Shape({1, 1, 1, pad_value_const_buffer_size}),
        DataType::BFLOAT16,
        Layout::ROW_MAJOR);
    auto& cq = device->mesh_command_queue();
    // NOTE: The const buffer is always in L1 (mirrors the legacy factory).
    const MemoryConfig mem_cfg{TensorMemoryLayout::INTERLEAVED, BufferType::L1};
    return tt::tt_metal::enqueue_write_tensor(cq, host_pad.host_tensor(), *device, mem_cfg);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::ProgramArtifacts PadRmReaderWriterMultiCoreProgramFactory::create_program_artifacts(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& output) {
    using namespace CMAKE_UNIQUE_NAMESPACE;  // resolve the file-local ids/helpers below
    // Metal 2.0 named resource handles (locals to avoid unity-build name collisions).
    const DFBSpecName CB_IN0{"cb_in0"};  // legacy c_0: input stream (reader produces, writer consumes)

    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};
    const TensorParamName PAD_TENSOR{"pad"};  // op-owned pad-value const tensor

    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};

    constexpr const char* READER_PATH =
        "ttnn/cpp/ttnn/operations/experimental/quasar/pad/device/kernels/dataflow/"
        "reader_pad_dims_rm_interleaved_sc.cpp";
    constexpr const char* WRITER_PATH =
        "ttnn/cpp/ttnn/operations/experimental/quasar/pad/device/kernels/dataflow/"
        "writer_pad_dims_rm_interleaved_sc.cpp";

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
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // ------------------------------------------------------------------------
    // Op-owned pad-value const tensor: allocate + fill, park on the artifact, then bind it like an
    // io tensor (TensorParameter + TensorArgument).  Build op_owned_tensors first so the
    // TensorArgument references the parked element (the adapter matches by pointer identity; a
    // vector move keeps the address).
    // ------------------------------------------------------------------------
    std::vector<MeshTensor> op_owned_tensors;
    op_owned_tensors.reserve(1);
    op_owned_tensors.push_back(build_pad_value_const_mesh_tensor(tensor_args, pad_value));
    const MeshTensor& pad_const = op_owned_tensors[0];

    uint32_t pad_value_const_buffer_size = 32;  // noc transfers in chunks of 32
    uint32_t pad_value_const_buffer_nbytes = pad_value_const_buffer_size * a.element_size();

    // ------------------------------------------------------------------------
    // DataflowBuffer (legacy CB c_0): one input-row stream, multibuffered.
    // ------------------------------------------------------------------------
    uint32_t cb_npages = 16;  // multibuffering for perf
    uint32_t cb_page_alignment = std::max(tt::constants::TILE_WIDTH, src0_buffer->alignment());
    uint32_t cb_pagesize =
        static_cast<uint32_t>(std::ceil((float)dst_nbytes_per_core_w / cb_page_alignment)) * cb_page_alignment;
    tt::DataFormat in_df = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    DataflowBufferSpec cb_in0_spec{
        .unique_id = CB_IN0,
        .entry_size = cb_pagesize,
        .num_entries = cb_npages,
        .data_format_metadata = in_df,
    };

    // ------------------------------------------------------------------------
    // Tensor parameters (Case-1 page access).  src + pad on the reader, dst on the writer.
    // ------------------------------------------------------------------------
    TensorParameter input_param{.unique_id = INPUT_TENSOR, .spec = a.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_TENSOR, .spec = output.tensor_spec()};
    TensorParameter pad_param{.unique_id = PAD_TENSOR, .spec = pad_const.tensor_spec()};

    // ------------------------------------------------------------------------
    // Pad value packed exactly as the legacy factory did (preserve dtype handling).
    // ------------------------------------------------------------------------
    uint32_t packed_pad_value;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32) {
        packed_pad_value = pad_value;
    } else if (a.dtype() == DataType::UINT16) {
        packed_pad_value = pack_two_uint16_into_uint32({0, float_to_uint16(pad_value)});
    } else {
        packed_pad_value = pack_two_bfloat16_into_uint32({bfloat16(0.0f), bfloat16(pad_value)});
    }

    // ------------------------------------------------------------------------
    // Reader: streams unpadded input rows into cb_in0, filling padding from the
    // op-owned pad-value const tensor (tensor::pad).  The legacy reader's slot-0 src-address and
    // slot-13 pad-const-address RTAs are dropped (reached via TensorBindings); the remaining slots
    // become named scalar RTAs.
    // ------------------------------------------------------------------------
    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path{READER_PATH},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = CB_IN0, .accessor_name = "cb_in0", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "src"},
             TensorBinding{.tensor_parameter_name = PAD_TENSOR, .accessor_name = "pad"}},
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
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // ------------------------------------------------------------------------
    // Writer: pulls padded rows from cb_in0 and writes them out page-by-page (Case-1).
    // ------------------------------------------------------------------------
    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{WRITER_PATH},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = CB_IN0, .accessor_name = "cb_in0", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "dst"}},
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
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    log_debug(tt::LogOp, "ncores: {}", ncores);
    log_debug(tt::LogOp, "ncores_h: {}", ncores_h);
    log_debug(tt::LogOp, "ncores_w: {}", ncores_w);
    log_debug(tt::LogOp, "ntiles_per_core_h: {}", ntiles_per_core_h);
    log_debug(tt::LogOp, "ntiles_per_core_w: {}", ntiles_per_core_w);
    log_debug(tt::LogOp, "src0_buffer_addr: {}", src0_buffer->address());
    log_debug(tt::LogOp, "dst_buffer_addr: {}", dst_buffer->address());
    log_debug(tt::LogOp, "unpadded_row_size_nbytes: {}", unpadded_row_size_nbytes);
    log_debug(tt::LogOp, "padded_row_size_nbytes: {}", padded_row_size_nbytes);
    log_debug(tt::LogOp, "pad_value_const_buffer_nbytes: {}", pad_value_const_buffer_nbytes);
    log_debug(tt::LogOp, "packed_pad_value: {}", packed_pad_value);
    log_debug(tt::LogOp, "src_nbytes_per_core_w: {}", src_nbytes_per_core_w);
    log_debug(tt::LogOp, "dst_nbytes_per_core_w: {}", dst_nbytes_per_core_w);
    log_debug(tt::LogOp, "nbatch_per_core_h: {}", nbatch_per_core_h);
    log_debug(tt::LogOp, "ncores_per_batch_h: {}", ncores_per_batch_h);

    // ------------------------------------------------------------------------
    // Per-core runtime args.  Mirrors the legacy descriptor's get_runtime_args accounting: the input
    // region is walked stick-by-stick to derive each core's start ids/offsets and unpadded extents.
    // ------------------------------------------------------------------------
    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};

    uint32_t start_src_stick_id = 0;
    uint32_t start_dst_stick_id = 0;
    uint32_t start_src_stick_wi = 0;  // start of stick segment for 2d decomp
    uint32_t start_dst_stick_wi = 0;
    int32_t local_nsticks = ntiles_per_core_h * TILE_HEIGHT;
    for (int32_t b = 0; b < static_cast<int32_t>(nbatch); ++b) {
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

                const NodeCoord node = core;
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

    WorkUnitSpec wu{
        .name = "pad_rm_multicore",
        .kernels = {READER_KERNEL, WRITER_KERNEL},
        .target_nodes = all_cores,
    };

    ProgramSpec spec{
        .name = "pad_rm_multicore",
        .kernels = {reader_spec, writer_spec},
        .dataflow_buffers = {cb_in0_spec},
        .tensor_parameters = {input_param, output_param, pad_param},
        .work_units = {wu},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run};
    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(a.mesh_tensor())}},
        {OUTPUT_TENSOR, TensorArgument{std::cref(output.mesh_tensor())}},
        {PAD_TENSOR, TensorArgument{std::cref(pad_const)}}};

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec), .run_params = std::move(run_args), .op_owned_tensors = std::move(op_owned_tensors)};
}

}  // namespace ttnn::prim::qsr
