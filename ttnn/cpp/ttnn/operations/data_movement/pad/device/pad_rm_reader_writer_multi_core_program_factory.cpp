// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_rm_reader_writer_multi_core_program_factory.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
using namespace tt::constants;

namespace ttnn::prim {
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

namespace {
// Names are prefixed per factory: all seven pad factories land in one unity-build
// translation unit, where every anonymous namespace is merged into a single scope.
const KernelSpecName RM_MC_READER{"reader"};
const KernelSpecName RM_MC_WRITER{"writer"};
const DFBSpecName RM_MC_IN0{"in0"};
const TensorParamName RM_MC_INPUT{"input"};
const TensorParamName RM_MC_OUTPUT{"output"};
const TensorParamName RM_MC_PAD_VALUE{"pad_value_const"};

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

// Allocate the on-device pad-value const tensor.  Pulled out so
// create_program_artifacts() can build it once on cache miss and hand its
// MeshTensor to the framework as an op-owned tensor, deferring the device
// deallocation until the cached Program is evicted (see #44565).
Tensor build_pad_value_const_tensor_mc(const PadInputs& tensor_args, float pad_value) {
    MeshDevice* device = tensor_args.input.device();
    uint32_t pad_value_const_buffer_size = 32;  // noc transfers in chunks of 32
    auto pad_value_const_buffer =
        tt::tt_metal::HostBuffer(std::vector<bfloat16>(pad_value_const_buffer_size, bfloat16(pad_value)));
    // NOTE: The const buffer is always in L1
    // TODO: make a local buffer for each core?
    return Tensor(
               std::move(pad_value_const_buffer),
               ttnn::Shape({1, 1, 1, pad_value_const_buffer_size}),
               DataType::BFLOAT16,
               Layout::ROW_MAJOR)
        .to_device(device, MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1});
}

}  // namespace

ttnn::device_operation::ProgramArtifacts PadRmReaderWriterMultiCoreProgramFactory::create_program_artifacts(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& input_mesh_tensor = a.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    // Build the pad-value const tensor once on cache miss and release its owning MeshTensor into
    // the artifact.  The framework parks it in the cache entry, so its address stays valid for
    // every dispatch that hits this cached Program.
    std::vector<tt::tt_metal::MeshTensor> op_owned;
    op_owned.reserve(1);
    Tensor pad_value_const_tensor = build_pad_value_const_tensor_mc(tensor_args, operation_attributes.pad_value);
    op_owned.push_back(pad_value_const_tensor.device_storage().release_mesh_tensor());
    const auto& pad_value_mesh_tensor = op_owned.back();

    const auto& output_padded_shape = operation_attributes.output_padded_shape;
    const auto& pad_value = operation_attributes.pad_value;

    auto output_shape = output_padded_shape;

    uint32_t unpadded_row_size_nbytes = a.padded_shape()[3] * a.element_size();
    uint32_t padded_row_size_nbytes = output_shape[3] * a.element_size();  // Assuming output is same datatype as input
    TT_ASSERT(
        unpadded_row_size_nbytes <= padded_row_size_nbytes, "Padded output tensor size should be >= input tensor size");

    distributed::MeshDevice* device = a.device();

    uint32_t pad_value_const_buffer_size = 32;  // noc transfers in chunks of 32
    uint32_t pad_value_const_buffer_nbytes = pad_value_const_buffer_size * a.element_size();

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

    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    uint32_t dfb_npages = 16;  // multibuffering for perf
    // uint32_t dfb_npages = 1; // multibuffering for perf
    uint32_t dfb_page_alignment = std::max(tt::constants::TILE_WIDTH, a.buffer()->alignment());
    uint32_t dfb_pagesize =
        static_cast<uint32_t>(std::ceil((float)dst_nbytes_per_core_w / dfb_page_alignment)) * dfb_page_alignment;
    tt::DataFormat in_df = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    DataflowBufferSpec in0_dfb{
        .unique_id = RM_MC_IN0,
        .entry_size = dfb_pagesize,
        .num_entries = dfb_npages,
        .data_format_metadata = in_df,
    };

    uint32_t packed_pad_value;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32) {
        packed_pad_value = pad_value;
    } else if (a.dtype() == DataType::UINT16) {
        packed_pad_value = pack_two_uint16_into_uint32({0, float_to_uint16(pad_value)});
    } else {
        packed_pad_value = pack_two_bfloat16_into_uint32({bfloat16(0.0f), bfloat16(pad_value)});
    }

    KernelSpec reader{
        .unique_id = RM_MC_READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/"
            "reader_pad_dims_rm_interleaved.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = RM_MC_IN0,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = RM_MC_INPUT,
                    .accessor_name = "src",
                },
                TensorBinding{
                    .tensor_parameter_name = RM_MC_PAD_VALUE,
                    .accessor_name = "pad_value",
                },
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"num_unpadded_W",
                     "num_unpadded_Z",
                     "num_total_Z",
                     "unpadded_X_nbytes",
                     "padded_X_nbytes",
                     "padded_X_diff_nbytes",
                     "pad_value_packed",
                     "start_src_stick_id",
                     "start_src_stick_offset",
                     "num_local_Y",
                     "num_local_unpadded_Y",
                     "num_local_W"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelSpec writer{
        .unique_id = RM_MC_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/"
            "writer_pad_dims_rm_interleaved.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = RM_MC_IN0,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = RM_MC_OUTPUT,
                    .accessor_name = "dst",
                },
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"num_total_Z",
                     "padded_X_nbytes",
                     "start_dst_stick_id",
                     "num_local_Y",
                     "dst_stick_offset",
                     "num_local_W"},
            },
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    KernelRunArgs reader_run_args{.kernel = RM_MC_READER};
    KernelRunArgs writer_run_args{.kernel = RM_MC_WRITER};

#if 1
    {
        log_debug(tt::LogOp, "ncores: {}", ncores);
        log_debug(tt::LogOp, "ncores_h: {}", ncores_h);
        log_debug(tt::LogOp, "ncores_w: {}", ncores_w);
        log_debug(tt::LogOp, "ntiles_per_core_h: {}", ntiles_per_core_h);
        log_debug(tt::LogOp, "ntiles_per_core_w: {}", ntiles_per_core_w);
        log_debug(tt::LogOp, "a.shape[0]: {}", a.padded_shape()[0]);
        log_debug(tt::LogOp, "out.shape[0]: {}", output_shape[0]);
        log_debug(tt::LogOp, "a.shape[1]: {}", a.padded_shape()[1]);
        log_debug(tt::LogOp, "out.shape[1]: {}", output_shape[1]);
        log_debug(tt::LogOp, "a.shape[2]: {}", a.padded_shape()[2]);
        log_debug(tt::LogOp, "out.shape[2]: {}", output_shape[2]);
        log_debug(tt::LogOp, "s.shape[3]: {}", a.padded_shape()[3]);
        log_debug(tt::LogOp, "out.shape[3]: {}", output_shape[3]);
        log_debug(tt::LogOp, "unpadded_row_size_nbytes: {}", unpadded_row_size_nbytes);
        log_debug(tt::LogOp, "padded_row_size_nbytes: {}", padded_row_size_nbytes);
        // log_debug(tt::LogOp, "padded_row_diff_size_nbytes: {}", padded_row_diff_size_nbytes);
        log_debug(tt::LogOp, "pad_value_const_buffer_nbytes: {}", pad_value_const_buffer_nbytes);
        log_debug(tt::LogOp, "packed_pad_value: {}", packed_pad_value);
        log_debug(tt::LogOp, "src_nbytes_per_core_w: {}", src_nbytes_per_core_w);
        log_debug(tt::LogOp, "dst_nbytes_per_core_w: {}", dst_nbytes_per_core_w);
        log_debug(tt::LogOp, "nbatch_per_core_h: {}", nbatch_per_core_h);
        log_debug(tt::LogOp, "ncores_per_batch_h: {}", ncores_per_batch_h);
    }
#endif

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
                AddRuntimeArgsForNode(
                    reader_run_args.runtime_arg_values,
                    core,
                    {{"num_unpadded_W", static_cast<uint32_t>(a.padded_shape()[0])},
                     {"num_unpadded_Z", static_cast<uint32_t>(a.padded_shape()[1])},
                     {"num_total_Z", static_cast<uint32_t>(output_shape[1])},
                     {"unpadded_X_nbytes", curr_stick_size_nbytes},
                     {"padded_X_nbytes", static_cast<uint32_t>(dst_nbytes_per_core_w)},
                     {"padded_X_diff_nbytes", static_cast<uint32_t>(curr_stick_diff_nbytes)},
                     {"pad_value_packed", packed_pad_value},
                     {"start_src_stick_id", start_src_stick_id},
                     {"start_src_stick_offset", start_src_stick_wi * a.element_size()},
                     {"num_local_Y", static_cast<uint32_t>(local_nsticks)},
                     {"num_local_unpadded_Y", num_local_unpadded_nsticks},
                     {"num_local_W", nbatch_per_core_h}});

                AddRuntimeArgsForNode(
                    writer_run_args.runtime_arg_values,
                    core,
                    {{"num_total_Z", static_cast<uint32_t>(output_shape[1])},
                     {"padded_X_nbytes", static_cast<uint32_t>(dst_nbytes_per_core_w)},
                     {"start_dst_stick_id", start_dst_stick_id},
                     {"num_local_Y", static_cast<uint32_t>(local_nsticks)},
                     {"dst_stick_offset", start_dst_stick_wi * output.element_size()},
                     {"num_local_W", nbatch_per_core_h}});

                start_src_stick_wi += ntiles_per_core_w * TILE_WIDTH;
                start_dst_stick_wi += ntiles_per_core_w * TILE_WIDTH;
            }  // for ncores_w
            start_src_stick_id += num_local_unpadded_nsticks;
            start_dst_stick_id += local_nsticks;
        }  // for ncores_h
    }

    ProgramSpec spec{
        .name = "pad_rm_reader_writer_multi_core",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = {std::move(in0_dfb)},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = RM_MC_INPUT, .spec = input_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = RM_MC_OUTPUT, .spec = output_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = RM_MC_PAD_VALUE, .spec = pad_value_mesh_tensor.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {RM_MC_READER, RM_MC_WRITER},
                    .target_nodes = all_cores,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {RM_MC_INPUT, TensorArgument{input_mesh_tensor}},
        {RM_MC_OUTPUT, TensorArgument{output_mesh_tensor}},
        {RM_MC_PAD_VALUE, TensorArgument{pad_value_mesh_tensor}},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
        .op_owned_tensors = std::move(op_owned),
    };
}

}  // namespace ttnn::prim
