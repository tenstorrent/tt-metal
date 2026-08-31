// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sort_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace ttnn::prim {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

// The ROW_MAJOR and TILE configurations bind different sets of dataflow buffers, so the layout gate
// has to reach the kernels at preprocessor level: a `dfb::` handle exists only where the host
// actually binds it, and `if constexpr` still resolves names in its discarded branch.
//
// The same reasoning applies to IS_UINT16_FP32_MODE: the UInt16 staging DFBs are only bound
// when the input dtype is UInt16, so the reader/writer software-conversion loops (which
// reference `dfb::uint16_input_stage` etc.) must also be preprocessor-gated to avoid the
// undeclared handles from tripping name lookup on the non-UINT16 path.
KernelSpec::CompilerOptions::Defines sort_kernel_defines(bool is_row_major, bool is_uint16_input) {
    KernelSpec::CompilerOptions::Defines defines;
    if (is_row_major) {
        defines.insert({"IS_ROW_MAJOR", "1"});
    }
    if (is_uint16_input) {
        defines.insert({"IS_UINT16_FP32_MODE", "1"});
    }
    return defines;
}

// Layout-only overload kept for the CrossCore factory, which routes UINT16 inputs to
// a different program via select_program_factory and therefore never sets
// IS_UINT16_FP32_MODE (its reader/writer have no software-conversion loops).
KernelSpec::CompilerOptions::Defines layout_defines(bool is_row_major) {
    return sort_kernel_defines(is_row_major, /*is_uint16_input=*/false);
}

}  // namespace

// Single row - single core
ttnn::device_operation::ProgramArtifacts SortProgramFactorySingleRowSingleCore::create_program_artifacts(
    const SortParams& attributes, const SortInputs& tensor_args, std::vector<Tensor>& output_tensors) {
    const bool is_row_major = (tensor_args.input_tensor.layout() == Layout::ROW_MAJOR);

    const tt::DataFormat input_tensor_cb_data_format =
        datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());
    // Value output dtype is always equal to input dtype in compute_output_specs (see
    // sort_device_operation.cpp), and the DFB sizing uses sort_value_cb_data_format
    // (Float32 for UINT16) rather than the output tensor's raw dtype, so
    // output_tensors.at(0).dtype() / value_tensor_cb_data_format aren't needed here.
    const tt::DataFormat index_tensor_cb_data_format = datatype_to_dataformat_converter(output_tensors.at(1).dtype());

    // Only the index tile size is needed directly; INPUT_TENSOR / INPUT_TRANSPOSED /
    // VALUE_TENSOR / RM_* all use sort_value_tile_size (below) so the fp32 sort path
    // works uniformly for UINT16 (where sort_value_cb_data_format is promoted to
    // Float32) and non-UINT16 inputs.
    const uint32_t index_tensor_tile_size = tile_size(index_tensor_cb_data_format);

    const auto& input_mesh_tensor = tensor_args.input_tensor.mesh_tensor();
    const auto& value_mesh_tensor = output_tensors.at(0).mesh_tensor();
    const auto& index_mesh_tensor = output_tensors.at(1).mesh_tensor();

    const auto input_shape =
        is_row_major ? tensor_args.input_tensor.logical_shape() : tensor_args.input_tensor.padded_shape();
    const uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / tt::constants::TILE_HEIGHT;
    const uint32_t Wt = input_shape[3] / tt::constants::TILE_WIDTH;

    const uint32_t element_size_bytes = tt::datum_size(input_tensor_cb_data_format);
    const uint32_t index_element_size_bytes = tt::datum_size(index_tensor_cb_data_format);
    const uint32_t W_value_bytes = input_shape[3] * element_size_bytes;
    const uint32_t W_index_bytes = input_shape[3] * index_element_size_bytes;

    constexpr uint32_t num_cb_unit = 2;
    constexpr uint32_t cb_in_units = 2 * num_cb_unit;

    auto* device = tensor_args.input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t total_number_of_cores = compute_with_storage_grid_size.y * compute_with_storage_grid_size.x;

    const uint32_t all_core_utilization_loop_count = Ht / total_number_of_cores;
    const uint32_t all_core_utilization_loop_residuum = Ht % total_number_of_cores;

    const bool is_32_bit_index = index_tensor_cb_data_format == tt::DataFormat::UInt32;
    // UINT16 keys must also run in fp32_dest_acc_en mode so the SFPU compares
    // them as 32-bit floats.  The hardware unpack converts the uint16 integer
    // value to an exact float32 (all 0..65535 are representable in 24-bit
    // mantissa), avoiding the bf16 precision loss that collapses integers > 256.
    const bool is_32_bit_data = is_32_bit_index || input_tensor_cb_data_format == tt::DataFormat::Float32 ||
                                input_tensor_cb_data_format == tt::DataFormat::UInt16;

    // With fp32_dest_acc_en=True, pack_tile writes 32-bit floats. Intermediate value DFBs
    // (INPUT_TRANSPOSED, VALUE_TENSOR / RM_VALUE_OUTPUT) must use Float32 so that fp32
    // values from the DEST register are stored and read back without truncation.  The
    // writer then converts Float32 → UInt16 element-by-element when sending to DRAM.
    const bool is_uint16_input = (input_tensor_cb_data_format == tt::DataFormat::UInt16);
    const tt::DataFormat sort_value_cb_data_format =
        is_uint16_input ? tt::DataFormat::Float32 : input_tensor_cb_data_format;
    const uint32_t sort_value_tile_size = tile_size(sort_value_cb_data_format);
    // Number of bytes a Float32-promoted RM value row occupies.  For UINT16 inputs
    // the reader software-converts each UInt16 row into a Float32 row before handing
    // it to the compute kernel (via RM_INPUT / RM_VALUE_OUTPUT), so this is
    // 2 × W_value_bytes.  For non-UINT16 dtypes it equals W_value_bytes.
    const uint32_t W_sort_value_bytes = input_shape[3] * tt::datum_size(sort_value_cb_data_format);

    CoreRangeSet core_range;
    if (Ht >= total_number_of_cores) {
        core_range = CoreRangeSet(
            CoreRange({0, 0}, {compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1}));
    } else {
        const uint32_t core_grid_calculated_rows_number = Ht / compute_with_storage_grid_size.x;
        const uint32_t core_grid_calculated_columns_number = Ht % compute_with_storage_grid_size.x;

        if (core_grid_calculated_rows_number == 0 && core_grid_calculated_columns_number == 0) {
            core_range = CoreRangeSet(CoreCoord(0, 0));
        } else if (core_grid_calculated_rows_number == 0) {
            core_range = CoreRangeSet(CoreRange({0, 0}, {core_grid_calculated_columns_number - 1, 0}));
        } else {
            core_range = CoreRangeSet(
                CoreRange({0, 0}, {compute_with_storage_grid_size.x - 1, core_grid_calculated_rows_number - 1}));
            if (core_grid_calculated_columns_number != 0) {
                const CoreRange additional_range(
                    {0, core_grid_calculated_rows_number},
                    {core_grid_calculated_columns_number - 1, core_grid_calculated_rows_number});
                core_range = core_range.merge(CoreRangeSet(additional_range));
            }
        }
    }

    // -----------------------------------------------------------------------
    // Resource names
    // -----------------------------------------------------------------------
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    const DFBSpecName INPUT_TENSOR{"input_tensor"};
    const DFBSpecName INDEX_TENSOR{"index_tensor"};
    const DFBSpecName INPUT_TRANSPOSED{"input_tensor_transposed"};
    const DFBSpecName INDEX_TRANSPOSED{"index_tensor_transposed"};
    const DFBSpecName VALUE_TENSOR{"value_tensor"};
    const DFBSpecName INDEX_OUTPUT{"index_tensor_output"};
    const DFBSpecName SYNCHRONIZATION{"synchronization"};
    const DFBSpecName RM_INPUT{"rm_input"};
    const DFBSpecName RM_VALUE_OUTPUT{"rm_value_output"};
    const DFBSpecName RM_INDEX_OUTPUT{"rm_index_output"};
    const DFBSpecName RM_POST_SORT_INDEX{"rm_post_sort_index"};
    // UINT16 conversion staging DFBs — only bound when is_uint16_input.
    const DFBSpecName UINT16_INPUT_STAGE{"uint16_input_stage"};
    const DFBSpecName UINT16_CONV{"uint16_conv"};
    const DFBSpecName RM_UINT16_INPUT_STAGE{"rm_uint16_input_stage"};
    const DFBSpecName RM_UINT16_OUTPUT_STAGE{"rm_uint16_output_stage"};

    const TensorParamName INPUT_PARAM{"input"};
    const TensorParamName VALUE_PARAM{"value_output"};
    const TensorParamName INDEX_PARAM{"index_output"};

    ProgramSpec spec;
    spec.name = "sort_single_row_single_core";

    // -----------------------------------------------------------------------
    // Dataflow buffers
    //
    // When the input is UInt16 the hardware unpack cannot numerically convert
    // UInt16 → Float32 (the ISA only allows UInt16 → UInt16 destination).  The
    // reader kernel software-converts each element on the RISC-V core: DMA the
    // raw UInt16 tile from DRAM into a staging DFB (uint16_input_stage /
    // rm_uint16_input_stage) and then emit float(uint16_val) into INPUT_TENSOR /
    // RM_INPUT — which are promoted to Float32 for the UINT16 mode via
    // sort_value_cb_data_format.  Compute then sorts on Float32 exactly.
    // Symmetrically the writer software-converts sorted Float32 → UInt16 in
    // uint16_conv / rm_uint16_output_stage before DMA back to DRAM.  Non-UINT16
    // dtypes do not bind these staging DFBs.
    // -----------------------------------------------------------------------
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INPUT_TENSOR,
        // INPUT_TENSOR carries the values the compute kernel sorts. With
        // is_uint16_input the reader software-converts UInt16 → Float32 into
        // this buffer, so it needs to be sized and formatted for Float32; for
        // non-UINT16 dtypes sort_value_* equals input_tensor_* (see the
        // sort_value_cb_data_format derivation above).
        .entry_size = sort_value_tile_size,
        .num_entries = is_row_major ? Wt : cb_in_units,
        .data_format_metadata = sort_value_cb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INDEX_TENSOR,
        .entry_size = index_tensor_tile_size,
        .num_entries = is_row_major ? Wt : cb_in_units,
        .data_format_metadata = index_tensor_cb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INPUT_TRANSPOSED,
        // The transposed value buffer sits between pack_tile and the next
        // sort iteration and must round-trip fp32 without truncation when
        // fp32_dest_acc_en is on (Float32 / UINT16 inputs).
        .entry_size = sort_value_tile_size,
        .num_entries = Wt,
        .data_format_metadata = sort_value_cb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INDEX_TRANSPOSED,
        .entry_size = index_tensor_tile_size,
        .num_entries = Wt,
        .data_format_metadata = index_tensor_cb_data_format,
    });
    if (!is_row_major) {
        // The ROW_MAJOR path routes its output through rm_value_output / rm_index_output instead, so
        // these two have no endpoint there and are declared for TILE only.
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = VALUE_TENSOR,
            // Same reasoning as INPUT_TRANSPOSED — this receives pack_tile
            // output before the writer converts back to UInt16 for DRAM.
            .entry_size = sort_value_tile_size,
            .num_entries = num_cb_unit,
            .data_format_metadata = sort_value_cb_data_format,
        });
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = INDEX_OUTPUT,
            .entry_size = index_tensor_tile_size,
            .num_entries = num_cb_unit,
            .data_format_metadata = index_tensor_cb_data_format,
        });
    }
    constexpr uint32_t synchronization_entry_size = tt::constants::TILE_HW * sizeof(uint8_t);
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = SYNCHRONIZATION,
        .entry_size = synchronization_entry_size,
        .num_entries = 1,
        .data_format_metadata = tt::DataFormat::UInt8,
    });

    if (is_uint16_input && !is_row_major) {
        // TILE path staging DFBs for the UInt16 <-> Float32 software conversion.
        // Reader stages one raw UInt16 tile per iteration in uint16_input_stage,
        // then converts into INPUT_TENSOR (Float32).  Writer stages one converted
        // UInt16 tile in uint16_conv before DMA to DRAM.  Both are single-tile CBs.
        const uint32_t uint16_one_tile_size = tile_size(tt::DataFormat::UInt16);
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = UINT16_INPUT_STAGE,
            .entry_size = uint16_one_tile_size,
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::UInt16,
        });
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = UINT16_CONV,
            .entry_size = uint16_one_tile_size,
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::UInt16,
        });
    }

    if (is_row_major) {
        // rm_input: reader pushes TILE_HEIGHT pages of sorted-value dtype.  For
        // UINT16 this is Float32 (W_sort_value_bytes == 2 × W_value_bytes) after
        // the reader's software conversion; for other dtypes it equals
        // W_value_bytes.
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = RM_INPUT,
            .entry_size = W_sort_value_bytes,
            .num_entries = tt::constants::TILE_HEIGHT,
            .data_format_metadata = sort_value_cb_data_format,
        });
        // rm_value_output: compute pushes untilized value rows; writer drains
        // them.  Sized for sort_value_cb_data_format so the pack_untilize output
        // fits when running the fp32 sort path (UINT16 inputs land here as
        // Float32; writer converts back to UInt16 before DMA to DRAM).
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = RM_VALUE_OUTPUT,
            .entry_size = W_sort_value_bytes,
            .num_entries = tt::constants::TILE_HEIGHT,
            .data_format_metadata = sort_value_cb_data_format,
        });
        // rm_index_output: compute pushes untilized index rows; reader drains them.
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = RM_INDEX_OUTPUT,
            .entry_size = W_index_bytes,
            .num_entries = tt::constants::TILE_HEIGHT,
            .data_format_metadata = index_tensor_cb_data_format,
        });
        // rm_post_sort_index: holds Wt un-transposed sorted index tiles.
        // PACK is the sole producer (compute kernel only) and UNPACK is the sole
        // consumer (also compute), so the push_back / wait_front pair use
        // matched semantics and the mixed-producer counter race that affects
        // index_tensor (BRISC + PACK) does not occur here.
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = RM_POST_SORT_INDEX,
            .entry_size = index_tensor_tile_size,
            .num_entries = Wt,
            .data_format_metadata = index_tensor_cb_data_format,
        });
        if (is_uint16_input) {
            // ROW_MAJOR UInt16 staging DFBs — one raw UInt16 row each.  Reader
            // stages incoming rows in rm_uint16_input_stage before software-
            // converting to Float32 into RM_INPUT.  Writer software-converts
            // Float32 rows from RM_VALUE_OUTPUT into rm_uint16_output_stage
            // before DMA to DRAM.
            spec.dataflow_buffers.push_back(DataflowBufferSpec{
                .unique_id = RM_UINT16_INPUT_STAGE,
                .entry_size = W_value_bytes,
                .num_entries = 1,
                .data_format_metadata = tt::DataFormat::UInt16,
            });
            spec.dataflow_buffers.push_back(DataflowBufferSpec{
                .unique_id = RM_UINT16_OUTPUT_STAGE,
                .entry_size = W_value_bytes,
                .num_entries = 1,
                .data_format_metadata = tt::DataFormat::UInt16,
            });
        }
    }

    // -----------------------------------------------------------------------
    // Tensor parameters
    // -----------------------------------------------------------------------
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = INPUT_PARAM, .spec = input_mesh_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = VALUE_PARAM, .spec = value_mesh_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = INDEX_PARAM, .spec = index_mesh_tensor.tensor_spec()});

    // -----------------------------------------------------------------------
    // Kernel resource bindings
    // -----------------------------------------------------------------------
    Group<DFBBinding> reader_dfb_bindings;
    Group<DFBBinding> writer_dfb_bindings;
    Group<DFBBinding> compute_dfb_bindings;

    if (is_row_major) {
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = RM_INPUT,
            .accessor_name = "rm_input",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = RM_INDEX_OUTPUT,
            .accessor_name = "rm_index_output",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        if (is_uint16_input) {
            // Reader also owns the raw-UInt16 staging DFB in ROW_MAJOR + UINT16
            // mode (see reader_single_row_single_core.cpp's #ifdef
            // IS_UINT16_FP32_MODE branch, which produces + consumes it inline).
            reader_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = RM_UINT16_INPUT_STAGE,
                .accessor_name = "rm_uint16_input_stage",
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
            reader_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = RM_UINT16_INPUT_STAGE,
                .accessor_name = "rm_uint16_input_stage",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
        }
    } else {
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = INPUT_TENSOR,
            .accessor_name = "input_tensor",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = INDEX_OUTPUT,
            .accessor_name = "index_tensor_output",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        if (is_uint16_input) {
            // TILE-path staging DFB — same producer/consumer pattern as the
            // ROW_MAJOR one above.
            reader_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = UINT16_INPUT_STAGE,
                .accessor_name = "uint16_input_stage",
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
            reader_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = UINT16_INPUT_STAGE,
                .accessor_name = "uint16_input_stage",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
        }
    }

    // The writer generates the index tiles the compute kernel sorts, in both configurations.
    writer_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = INDEX_TENSOR,
        .accessor_name = "index_tensor",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    if (is_row_major) {
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = RM_VALUE_OUTPUT,
            .accessor_name = "rm_value_output",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        if (is_uint16_input) {
            // Writer stages the software-converted UInt16 row before DMA.
            writer_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = RM_UINT16_OUTPUT_STAGE,
                .accessor_name = "rm_uint16_output_stage",
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
            writer_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = RM_UINT16_OUTPUT_STAGE,
                .accessor_name = "rm_uint16_output_stage",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
        }
    } else {
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = VALUE_TENSOR,
            .accessor_name = "value_tensor",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        if (is_uint16_input) {
            // TILE-path Float32 → UInt16 conversion staging (matches
            // writer_single_row_single_core.cpp #ifdef IS_UINT16_FP32_MODE).
            writer_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = UINT16_CONV,
                .accessor_name = "uint16_conv",
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
            writer_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = UINT16_CONV,
                .accessor_name = "uint16_conv",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
        }
    }

    if (is_row_major) {
        // In ROW_MAJOR the compute kernel fills input_tensor itself from the tilize, so it holds
        // both ends of that buffer; in TILE the reader fills it.
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = INPUT_TENSOR,
            .accessor_name = "input_tensor",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
    }
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = INPUT_TENSOR,
        .accessor_name = "input_tensor",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = INDEX_TENSOR,
        .accessor_name = "index_tensor",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = INPUT_TRANSPOSED,
        .accessor_name = "input_tensor_transposed",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = INPUT_TRANSPOSED,
        .accessor_name = "input_tensor_transposed",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = INDEX_TRANSPOSED,
        .accessor_name = "index_tensor_transposed",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = INDEX_TRANSPOSED,
        .accessor_name = "index_tensor_transposed",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = SYNCHRONIZATION,
        .accessor_name = "synchronization",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = SYNCHRONIZATION,
        .accessor_name = "synchronization",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    if (is_row_major) {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = RM_INPUT,
            .accessor_name = "rm_input",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = RM_VALUE_OUTPUT,
            .accessor_name = "rm_value_output",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = RM_INDEX_OUTPUT,
            .accessor_name = "rm_index_output",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = RM_POST_SORT_INDEX,
            .accessor_name = "rm_post_sort_index",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = RM_POST_SORT_INDEX,
            .accessor_name = "rm_post_sort_index",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    } else {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = VALUE_TENSOR,
            .accessor_name = "value_tensor",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = INDEX_OUTPUT,
            .accessor_name = "index_tensor_output",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
    }

    // -----------------------------------------------------------------------
    // Kernels
    //
    // The IS_UINT16_FP32_MODE compile-time define below gates the reader/writer
    // kernel software-conversion loops.  See the DFB spec block above for the
    // matching UINT16_INPUT_STAGE / UINT16_CONV / RM_UINT16_* staging DFBs.
    // -----------------------------------------------------------------------
    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/"
                  "reader_single_row_single_core.cpp",
        .compiler_options = {.defines = sort_kernel_defines(is_row_major, is_uint16_input)},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = INPUT_PARAM, .accessor_name = "input_tensor"},
                TensorBinding{.tensor_parameter_name = INDEX_PARAM, .accessor_name = "index_tensor"},
            },
        .compile_time_args =
            {
                {"Wt", Wt},
                {"Ht", Ht},
                {"total_number_of_cores", total_number_of_cores},
                {"compute_with_storage_grid_size_x", static_cast<uint32_t>(compute_with_storage_grid_size.x)},
                {"compute_with_storage_grid_size_y", static_cast<uint32_t>(compute_with_storage_grid_size.y)},
                {"W_value_bytes", W_value_bytes},
                {"W_index_bytes", W_index_bytes},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"core_loop_count"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    });

    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/"
                  "writer_single_row_single_core.cpp",
        .compiler_options = {.defines = sort_kernel_defines(is_row_major, is_uint16_input)},
        .dfb_bindings = std::move(writer_dfb_bindings),
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = VALUE_PARAM, .accessor_name = "value_tensor"},
            },
        .compile_time_args =
            {
                {"Wt", Wt},
                {"Ht", Ht},
                {"total_number_of_cores", total_number_of_cores},
                {"compute_with_storage_grid_size_x", static_cast<uint32_t>(compute_with_storage_grid_size.x)},
                {"compute_with_storage_grid_size_y", static_cast<uint32_t>(compute_with_storage_grid_size.y)},
                {"is_32_bit_data", static_cast<uint32_t>(is_32_bit_data)},
                {"W_value_bytes", W_value_bytes},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"core_loop_count"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    });

    ComputeGen1Config compute_hw_config{.enable_32_bit_dest = is_32_bit_data};
    // UINT16 keys also route through the fp32 sort path (sort_value_cb_data_format
    // is Float32 for UINT16 — the reader's software conversion writes Float32 into
    // INPUT_TENSOR), so treat UINT16 the same as Float32 here: the value buffers
    // that the compute kernel unpacks must go UnpackToDest to preserve the full
    // 32-bit mantissa.  Without UnpackToDest, the hardware unpack would see a
    // Float32 CB and still load it in a reduced-precision (BF16/TF32) mode,
    // losing bits > 255 and defeating the software conversion.  Index buffers
    // stay on the default path.
    if (input_tensor_cb_data_format == tt::DataFormat::Float32 || is_uint16_input) {
        compute_hw_config.unpack_modes.insert({INPUT_TENSOR, UnpackMode::UnpackToDest});
        compute_hw_config.unpack_modes.insert({INPUT_TRANSPOSED, UnpackMode::UnpackToDest});
        if (is_row_major) {
            // rm_input carries Float32 for both native Float32 RM inputs and
            // UINT16 RM inputs (after reader software conversion), so
            // tilize_block must load the full 32-bit mantissa here.
            compute_hw_config.unpack_modes.insert({RM_INPUT, UnpackMode::UnpackToDest});
        } else {
            compute_hw_config.unpack_modes.insert({VALUE_TENSOR, UnpackMode::UnpackToDest});
        }
    }

    spec.kernels.push_back(KernelSpec{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/compute/"
                  "sort_single_row_single_core.cpp",
        // Compute kernels build at O3; the compiler-options default is O2, so the level is stated
        // explicitly rather than inherited.
        .compiler_options =
            {.defines = sort_kernel_defines(is_row_major, is_uint16_input),
             .opt_level = KernelSpec::CompilerOptions::OptLevel::O3},
        .dfb_bindings = std::move(compute_dfb_bindings),
        .compile_time_args =
            {
                {"Wt", Wt},
                {"descending", static_cast<uint32_t>(attributes.descending)},
                {"stable", static_cast<uint32_t>(attributes.stable)},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"core_loop_count"}},
        .hw_config = ComputeHardwareConfig{compute_hw_config},
    });

    spec.work_units.push_back(WorkUnitSpec{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = core_range,
    });

    // -----------------------------------------------------------------------
    // Run args
    // -----------------------------------------------------------------------
    // When the tile rows don't divide evenly across the cores, the first `residuum` cores in
    // row-major order each take one extra loop iteration to absorb the remainder.
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};

    const uint32_t default_loop_count = all_core_utilization_loop_count ? all_core_utilization_loop_count : 1;
    const bool has_residuum = (all_core_utilization_loop_residuum != 0) && (all_core_utilization_loop_count != 0);
    uint32_t residuum_used = 0;
    for (uint32_t core_y = 0; core_y < compute_with_storage_grid_size.y; core_y++) {
        for (uint32_t core_x = 0; core_x < compute_with_storage_grid_size.x; core_x++) {
            const CoreCoord core = {core_x, core_y};
            if (!core_range.contains(core)) {
                continue;
            }
            uint32_t core_loop_count = default_loop_count;
            if (has_residuum && residuum_used < all_core_utilization_loop_residuum) {
                core_loop_count = all_core_utilization_loop_count + 1;
                residuum_used++;
            }
            AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"core_loop_count", core_loop_count}});
            AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, core, {{"core_loop_count", core_loop_count}});
            AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"core_loop_count", core_loop_count}});
        }
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));
    run_args.tensor_args.emplace(INPUT_PARAM, input_mesh_tensor);
    run_args.tensor_args.emplace(VALUE_PARAM, value_mesh_tensor);
    run_args.tensor_args.emplace(INDEX_PARAM, index_mesh_tensor);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

// SortProgramFactoryCrossCoreDataExchange - single row, multi core with processing multiple tiles on one core with
// cross core data exchange

namespace {

CoreRangeSet compute_cross_core_range(
    uint32_t all_core_utilization_count,
    uint32_t total_number_of_cores_physical,
    uint32_t total_number_of_cores_virtual,
    const CoreCoord& compute_with_storage_grid_size) {
    /**
     * Calculates the core range based on the number of work units (all_core_utilization_count) and the total number of
     * available cores in the device's compute grid. The core range determines which cores will be utilized for
     * computation.
     *
     * The calculation works as follows:
     * 1. If all available cores are needed (all_core_utilization_count == total_number_of_cores), the core range covers
     * the entire grid.
     * 2. Otherwise, the number of rows (core_grid_calculated_rows_number) and columns
     * (core_grid_calculated_columns_number) required to cover all_core_utilization_count are calculated based on the
     * grid dimensions.
     *    - If both rows and columns are zero, only a single core is used.
     *    - If only rows are zero, the core range is set to cover the required number of columns in the first row.
     *    - Otherwise, the core range is set to cover the required rows, and if there are remaining columns,
     *      an additional range is added to cover those columns in the next row.
     *
     * The resulting core range is represented as a `CoreRangeSet`, which may consist of one or more `CoreRange`
     * objects depending on the configuration.
     */
    CoreRangeSet core_range;
    if (all_core_utilization_count == total_number_of_cores_physical) {
        core_range = CoreRangeSet(
            CoreRange({0, 0}, {compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1}));
    } else if (all_core_utilization_count == total_number_of_cores_virtual) {
        const uint32_t core_grid_calculated_rows_number =
            (all_core_utilization_count / compute_with_storage_grid_size.x) - 1;
        const uint32_t core_grid_calculated_columns_number =
            all_core_utilization_count % compute_with_storage_grid_size.x;
        core_range =
            CoreRangeSet(CoreRange({0, 0}, {compute_with_storage_grid_size.x - 1, core_grid_calculated_rows_number}));
        if (core_grid_calculated_columns_number != 0) {
            const CoreRange additional_range(
                {0, core_grid_calculated_rows_number + 1},
                {core_grid_calculated_columns_number - 1, core_grid_calculated_rows_number + 1});
            core_range = core_range.merge(CoreRangeSet(additional_range));
        }
    } else {
        const uint32_t core_grid_calculated_rows_number = all_core_utilization_count / compute_with_storage_grid_size.x;
        const uint32_t core_grid_calculated_columns_number =
            all_core_utilization_count % compute_with_storage_grid_size.x;

        if (core_grid_calculated_rows_number == 0 && core_grid_calculated_columns_number == 0) {
            core_range = CoreRangeSet(CoreCoord(0, 0));
        } else if (core_grid_calculated_rows_number == 0) {
            core_range = CoreRangeSet(CoreRange({0, 0}, {core_grid_calculated_columns_number - 1, 0}));
        } else {
            core_range = CoreRangeSet(
                CoreRange({0, 0}, {compute_with_storage_grid_size.x - 1, core_grid_calculated_rows_number - 1}));
            if (core_grid_calculated_columns_number != 0) {
                const CoreRange additional_range(
                    {0, core_grid_calculated_rows_number},
                    {core_grid_calculated_columns_number - 1, core_grid_calculated_rows_number});
                core_range = core_range.merge(CoreRangeSet(additional_range));
            }
        }
    }
    return core_range;
}

// The worker grid layout is needed twice: once to populate the physical-core lookup table and once
// to build the spec. Both derive it from here so the table's row order always matches the core order
// the kernels index it by.
struct CrossCoreLayout {
    CoreRangeSet core_range;
    uint32_t all_core_utilization_count;
    uint32_t total_number_of_cores_physical;
    uint32_t total_number_of_cores_virtual;
    uint32_t number_of_tiles_per_core;
    CoreCoord compute_with_storage_grid_size;
    uint32_t Ht;
    uint32_t Wt;
};

CrossCoreLayout compute_cross_core_layout(const SortInputs& tensor_args, const std::vector<Tensor>& output_tensors) {
    CrossCoreLayout layout{};
    auto* const device = tensor_args.input_tensor.device();
    layout.compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    layout.total_number_of_cores_physical =
        layout.compute_with_storage_grid_size.y * layout.compute_with_storage_grid_size.x;
    layout.total_number_of_cores_virtual =
        SortProgramFactoryCrossCoreDataExchange::rounddown_pow2(layout.total_number_of_cores_physical);

    const auto tile_width = tensor_args.input_tensor.tensor_spec().tile().get_width();
    const auto tile_height = tensor_args.input_tensor.tensor_spec().tile().get_height();
    const auto input_shape = tensor_args.input_tensor.padded_shape();
    layout.Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / tile_height;
    layout.Wt = input_shape[3] / tile_width;

    uint32_t number_of_tiles_per_core = SortProgramFactoryCrossCoreDataExchange::get_number_of_tiles_per_core(
        layout.total_number_of_cores_virtual,
        layout.Wt,
        tensor_args.input_tensor.dtype(),
        output_tensors.at(1).dtype(),
        SortProgramFactoryCrossCoreDataExchange::CrossCoreDataExchangeSortSlicingStrategy::USE_AS_MANY_CORES);
    layout.number_of_tiles_per_core = std::min(number_of_tiles_per_core, layout.Wt);

    layout.all_core_utilization_count =
        (layout.Wt + layout.number_of_tiles_per_core - 1) / layout.number_of_tiles_per_core;

    layout.core_range = compute_cross_core_range(
        layout.all_core_utilization_count,
        layout.total_number_of_cores_physical,
        layout.total_number_of_cores_virtual,
        layout.compute_with_storage_grid_size);

    return layout;
}

// Build the op-owned physical-core lookup table tensor.
Tensor build_physical_core_lookup_table_tensor(const SortInputs& tensor_args, std::vector<Tensor>& output_tensors) {
    auto* const device = tensor_args.input_tensor.device();
    const auto layout = compute_cross_core_layout(tensor_args, output_tensors);

    // Physical core coordinates, in the order the kernels index them by. The contents depend only
    // on the worker grid, so this is built once when the program is created; the framework keeps the
    // tensor alive and re-binds its address to the reader on every dispatch.
    std::vector<uint32_t> physical_core_lookup_table_data;
    for (const auto& core_range : layout.core_range.ranges()) {
        for (const auto& core_coord : core_range) {
            const auto physical_core = device->worker_core_from_logical_core(core_coord);
            physical_core_lookup_table_data.emplace_back(physical_core.x);
            physical_core_lookup_table_data.emplace_back(physical_core.y);
        }
    }
    const tt::tt_metal::TensorSpec physical_core_lookup_table_spec(
        ttnn::Shape{1, physical_core_lookup_table_data.size()},
        TensorLayout{DataType::UINT32, PageConfig{Layout::ROW_MAJOR}, MemoryConfig()});
    Tensor physical_core_lookup_table_tensor =
        Tensor::from_vector(std::move(physical_core_lookup_table_data), physical_core_lookup_table_spec);
    return physical_core_lookup_table_tensor.to_device(device);
}

}  // namespace

uint32_t SortProgramFactoryCrossCoreDataExchange::get_number_of_tiles_per_core(
    uint32_t total_number_of_cores,
    uint32_t Wt,
    const DataType& input_dtype,
    const DataType& index_dtype,
    CrossCoreDataExchangeSortSlicingStrategy slicing_strategy) {
    TT_FATAL(total_number_of_cores != 0, "number of cores cannot be 0");
    switch (slicing_strategy) {
        case CrossCoreDataExchangeSortSlicingStrategy::USE_AS_MANY_CORES: {
            constexpr uint32_t MIN_TILES_PER_CORE = 2;
            constexpr uint32_t MAX_TILES_PER_CORE = 128;
            const auto max_val = std::max(Wt / total_number_of_cores, MIN_TILES_PER_CORE);
            return std::min(MAX_TILES_PER_CORE, max_val);
        }
        case CrossCoreDataExchangeSortSlicingStrategy::FILL_CORES_FIRST:
        default: {
            if (input_dtype == DataType::FLOAT32 || input_dtype == DataType::UINT32 || input_dtype == DataType::INT32 ||
                index_dtype == DataType::INT32 || index_dtype == DataType::UINT32) {
                return 64;
            }
            break;
        }
    }

    return 128;
}

uint32_t SortProgramFactoryCrossCoreDataExchange::rounddown_pow2(uint32_t n) {
    if (n == 0) {
        return 0;
    }
    return 1 << (31 - std::countl_zero(n));
}

ttnn::device_operation::ProgramArtifacts SortProgramFactoryCrossCoreDataExchange::create_program_artifacts(
    const SortParams& attributes, const SortInputs& tensor_args, std::vector<Tensor>& output_tensors) {
    const tt::DataFormat input_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());
    const tt::DataFormat value_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(output_tensors.at(0).dtype());
    const tt::DataFormat index_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(output_tensors.at(1).dtype());
    const tt::DataFormat packer_unpacker_sync_cb_data_format = tt::DataFormat::Float16_b;

    const uint32_t input_tensor_tile_size = tile_size(input_tensor_cb_data_format);
    const uint32_t value_tensor_tile_size = tile_size(value_tensor_cb_data_format);
    const uint32_t index_tensor_tile_size = tile_size(index_tensor_cb_data_format);
    const uint32_t packer_unpacker_sync_tile_size = tile_size(packer_unpacker_sync_cb_data_format);

    const auto& input_mesh_tensor = tensor_args.input_tensor.mesh_tensor();
    const auto& value_mesh_tensor = output_tensors.at(0).mesh_tensor();
    const auto& index_mesh_tensor = output_tensors.at(1).mesh_tensor();

    const auto layout = compute_cross_core_layout(tensor_args, output_tensors);
    const auto& core_range = layout.core_range;
    const uint32_t Ht = layout.Ht;
    const uint32_t Wt = layout.Wt;
    const uint32_t number_of_tiles_per_core = layout.number_of_tiles_per_core;
    const uint32_t all_core_utilization_count = layout.all_core_utilization_count;
    const uint32_t total_number_of_cores_virtual = layout.total_number_of_cores_virtual;
    const auto& compute_with_storage_grid_size = layout.compute_with_storage_grid_size;

    TT_FATAL(
        all_core_utilization_count <= total_number_of_cores_virtual,
        "All core utilization count exceeds total number of cores. Utilized cores: {}, Total cores: {}",
        all_core_utilization_count,
        total_number_of_cores_virtual);

    // UINT16 inputs are excluded from this factory by
    // SortDeviceOperation::select_program_factory: any UINT16 above the SingleCore
    // threshold is routed to SortProgramFactorySingleRowMultiCore
    // (`!is_uint16 && Wt <= total_number_of_tiles_for_hybrid_approach`) because the
    // CrossCore factory has no UInt16 <-> Float32 conversion DFBs.  Guard with
    // TT_FATAL (not TT_ASSERT — which compiles out in release builds) so a broken
    // routing invariant fails loudly instead of silently sorting UINT16 keys in
    // bf16 mode and returning wrong indices.
    TT_FATAL(
        input_tensor_cb_data_format != tt::DataFormat::UInt16,
        "Invariant violated: UINT16 input reached the CrossCore sort factory.  "
        "SortDeviceOperation::select_program_factory routes UINT16 to "
        "SortProgramFactorySingleRowMultiCore for Wt above the SingleCore "
        "threshold; this factory does not implement the UInt16 <-> Float32 "
        "conversion loops.");

    // uint32 index tensor support
    const bool is_32_bit_index = index_tensor_cb_data_format == tt::DataFormat::UInt32;
    const bool is_32_bit_data = is_32_bit_index || input_tensor_cb_data_format == tt::DataFormat::Float32;

    const bool is_row_major = (tensor_args.input_tensor.layout() == Layout::ROW_MAJOR);
    const auto tile_width = tensor_args.input_tensor.tensor_spec().tile().get_width();
    const uint32_t value_element_bytes = tt::datum_size(value_tensor_cb_data_format);
    const uint32_t index_element_bytes = tt::datum_size(index_tensor_cb_data_format);
    const uint32_t W_value_slice_bytes = number_of_tiles_per_core * tile_width * value_element_bytes;
    const uint32_t W_index_slice_bytes = number_of_tiles_per_core * tile_width * index_element_bytes;

    // The lookup table is a device tensor this factory allocates for itself, beyond the op's declared
    // io. Moving the owning MeshTensor into the artifact hands its lifetime to the framework, which
    // keeps the allocation at a stable address for as long as the program stays cached.
    Tensor lookup_build_tensor = build_physical_core_lookup_table_tensor(tensor_args, output_tensors);
    std::vector<tt::tt_metal::MeshTensor> op_owned;
    op_owned.reserve(1);
    op_owned.push_back(lookup_build_tensor.device_storage().release_mesh_tensor());
    const auto& lookup_mesh_tensor = op_owned.back();

    const tt::DataFormat physical_core_lookup_table_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(lookup_mesh_tensor.tensor_spec().data_type());
    const uint32_t physical_core_lookup_table_tile_size = tile_size(physical_core_lookup_table_cb_data_format);

    // -----------------------------------------------------------------------
    // Resource names
    // -----------------------------------------------------------------------
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    const DFBSpecName INPUT_TENSOR{"input_tensor"};
    const DFBSpecName INDEX_TENSOR{"index_tensor"};
    const DFBSpecName INPUT_TRANSPOSED{"input_tensor_transposed"};
    const DFBSpecName INDEX_TRANSPOSED{"index_tensor_transposed"};
    const DFBSpecName VALUE_TENSOR{"value_tensor"};
    const DFBSpecName INDEX_OUTPUT{"index_tensor_output"};
    const DFBSpecName VALUE_INTERMEDIATE{"value_tensor_intermediate"};
    const DFBSpecName INDEX_INTERMEDIATE{"index_tensor_intermediate"};
    const DFBSpecName VALUE_PEER{"value_tensor_peer"};
    const DFBSpecName INDEX_PEER{"index_tensor_peer"};
    const DFBSpecName LOOKUP_TABLE{"physical_core_lookup_table"};
    const DFBSpecName PACKER_UNPACKER_SYNC{"packer_unpacker_sync"};
    const DFBSpecName RM_INPUT{"rm_input"};
    const DFBSpecName RM_VALUE_OUTPUT{"rm_value_output"};
    const DFBSpecName RM_INDEX_OUTPUT{"rm_index_output"};
    const DFBSpecName RM_POST_SORT_INDEX{"rm_post_sort_index"};

    const SemaphoreSpecName SEM_EXCHANGE{"exchange_readers"};
    const SemaphoreSpecName SEM_UNUSED{"unused"};
    const SemaphoreSpecName SEM_BARRIER{"barrier"};

    const TensorParamName INPUT_PARAM{"input"};
    const TensorParamName VALUE_PARAM{"value_output"};
    const TensorParamName INDEX_PARAM{"index_output"};
    const TensorParamName LOOKUP_PARAM{"physical_core_lookup_table"};

    ProgramSpec spec;
    spec.name = "sort_cross_core_data_exchange";

    // -----------------------------------------------------------------------
    // Dataflow buffers
    // -----------------------------------------------------------------------
    constexpr uint32_t cb_scale_factor = 2;

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INPUT_TENSOR,
        .entry_size = input_tensor_tile_size,
        .num_entries = is_row_major ? number_of_tiles_per_core : cb_scale_factor,
        .data_format_metadata = input_tensor_cb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INDEX_TENSOR,
        .entry_size = index_tensor_tile_size,
        .num_entries = cb_scale_factor,
        .data_format_metadata = index_tensor_cb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INPUT_TRANSPOSED,
        .entry_size = input_tensor_tile_size,
        .num_entries = number_of_tiles_per_core,
        .data_format_metadata = input_tensor_cb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INDEX_TRANSPOSED,
        .entry_size = index_tensor_tile_size,
        .num_entries = number_of_tiles_per_core,
        .data_format_metadata = index_tensor_cb_data_format,
    });
    if (!is_row_major) {
        // The ROW_MAJOR path routes its output through rm_value_output / rm_index_output instead, so
        // these two have no endpoint there and are declared for TILE only.
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = VALUE_TENSOR,
            .entry_size = value_tensor_tile_size,
            .num_entries = cb_scale_factor,
            .data_format_metadata = value_tensor_cb_data_format,
        });
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = INDEX_OUTPUT,
            .entry_size = index_tensor_tile_size,
            .num_entries = cb_scale_factor,
            .data_format_metadata = index_tensor_cb_data_format,
        });
    }
    // The two value-carrying exchange buffers hold value tiles but are paged at the index tile size,
    // so their entry count is the value bytes they must hold divided by that page size. The division
    // is exact for every dtype the op accepts: value and index tiles are both either 2048 or 4096
    // bytes, so one size is always a whole multiple of the other.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = VALUE_INTERMEDIATE,
        .entry_size = index_tensor_tile_size,
        .num_entries = (cb_scale_factor * value_tensor_tile_size) / index_tensor_tile_size,
        .data_format_metadata = value_tensor_cb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INDEX_INTERMEDIATE,
        .entry_size = index_tensor_tile_size,
        .num_entries = cb_scale_factor,
        .data_format_metadata = index_tensor_cb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = VALUE_PEER,
        .entry_size = index_tensor_tile_size,
        .num_entries = (cb_scale_factor * value_tensor_tile_size) / index_tensor_tile_size,
        .data_format_metadata = value_tensor_cb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INDEX_PEER,
        .entry_size = index_tensor_tile_size,
        .num_entries = cb_scale_factor,
        .data_format_metadata = index_tensor_cb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = LOOKUP_TABLE,
        .entry_size = physical_core_lookup_table_tile_size,
        .num_entries = 1,
        .data_format_metadata = physical_core_lookup_table_cb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = PACKER_UNPACKER_SYNC,
        .entry_size = packer_unpacker_sync_tile_size,
        .num_entries = 1,
        .data_format_metadata = packer_unpacker_sync_cb_data_format,
    });

    if (is_row_major) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = RM_INPUT,
            .entry_size = W_value_slice_bytes,
            .num_entries = tt::constants::TILE_HEIGHT,
            .data_format_metadata = input_tensor_cb_data_format,
        });
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = RM_VALUE_OUTPUT,
            .entry_size = W_value_slice_bytes,
            .num_entries = tt::constants::TILE_HEIGHT,
            .data_format_metadata = value_tensor_cb_data_format,
        });
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = RM_INDEX_OUTPUT,
            .entry_size = W_index_slice_bytes,
            .num_entries = tt::constants::TILE_HEIGHT,
            .data_format_metadata = index_tensor_cb_data_format,
        });
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = RM_POST_SORT_INDEX,
            .entry_size = index_tensor_tile_size,
            .num_entries = number_of_tiles_per_core,
            .data_format_metadata = index_tensor_cb_data_format,
        });
    }

    // -----------------------------------------------------------------------
    // Semaphores. SEM_UNUSED is bound by no kernel. It is declared so the program's semaphore
    // footprint matches what the cross-core exchange has always allocated, and can be dropped once
    // that is confirmed to be safe.
    // -----------------------------------------------------------------------
    spec.semaphores.push_back(SemaphoreSpec{.unique_id = SEM_EXCHANGE, .target_nodes = core_range});
    spec.semaphores.push_back(SemaphoreSpec{.unique_id = SEM_UNUSED, .target_nodes = core_range});
    spec.semaphores.push_back(SemaphoreSpec{.unique_id = SEM_BARRIER, .target_nodes = core_range});

    // -----------------------------------------------------------------------
    // Tensor parameters
    // -----------------------------------------------------------------------
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = INPUT_PARAM, .spec = input_mesh_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = VALUE_PARAM, .spec = value_mesh_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = INDEX_PARAM, .spec = index_mesh_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = LOOKUP_PARAM, .spec = lookup_mesh_tensor.tensor_spec()});

    // -----------------------------------------------------------------------
    // Kernel resource bindings
    // -----------------------------------------------------------------------
    Group<DFBBinding> reader_dfb_bindings;
    Group<DFBBinding> writer_dfb_bindings;
    Group<DFBBinding> compute_dfb_bindings;

    if (is_row_major) {
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = RM_INPUT,
            .accessor_name = "rm_input",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = RM_INDEX_OUTPUT,
            .accessor_name = "rm_index_output",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    } else {
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = INPUT_TENSOR,
            .accessor_name = "input_tensor",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = INDEX_OUTPUT,
            .accessor_name = "index_tensor_output",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }
    // The peer exchange runs in both configurations: the reader drains the two intermediate buffers
    // out to the peer core and fills the two peer buffers from it.
    reader_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = VALUE_INTERMEDIATE,
        .accessor_name = "value_tensor_intermediate",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    reader_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = INDEX_INTERMEDIATE,
        .accessor_name = "index_tensor_intermediate",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    reader_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = VALUE_PEER,
        .accessor_name = "value_tensor_peer",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    reader_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = INDEX_PEER,
        .accessor_name = "index_tensor_peer",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    // The reader is the lookup table's real producer: it fills the table from DRAM and then reads it
    // back through a raw pointer peek, which either endpoint role permits.
    reader_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = LOOKUP_TABLE,
        .accessor_name = "physical_core_lookup_table",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });

    writer_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = INDEX_TENSOR,
        .accessor_name = "index_tensor",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    if (is_row_major) {
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = RM_VALUE_OUTPUT,
            .accessor_name = "rm_value_output",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    } else {
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = VALUE_TENSOR,
            .accessor_name = "value_tensor",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }
    // No kernel ever waits on or pops the lookup table: the reader fills it and then reads it back
    // through a raw pointer, and the writer's only touch is one bare push_back with no matching
    // reserve. Every buffer still needs one producer and one consumer on each node it lives on, so
    // the writer takes the consumer role. On Gen1 a dataflow buffer lowers to a plain circular
    // buffer whose counters any core can drive, so the push_back behaves the same either way.
    writer_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = LOOKUP_TABLE,
        .accessor_name = "physical_core_lookup_table",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });

    if (is_row_major) {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = INPUT_TENSOR,
            .accessor_name = "input_tensor",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
    }
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = INPUT_TENSOR,
        .accessor_name = "input_tensor",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = INDEX_TENSOR,
        .accessor_name = "index_tensor",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = INPUT_TRANSPOSED,
        .accessor_name = "input_tensor_transposed",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = INPUT_TRANSPOSED,
        .accessor_name = "input_tensor_transposed",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = INDEX_TRANSPOSED,
        .accessor_name = "index_tensor_transposed",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = INDEX_TRANSPOSED,
        .accessor_name = "index_tensor_transposed",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = VALUE_INTERMEDIATE,
        .accessor_name = "value_tensor_intermediate",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = INDEX_INTERMEDIATE,
        .accessor_name = "index_tensor_intermediate",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = VALUE_PEER,
        .accessor_name = "value_tensor_peer",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = INDEX_PEER,
        .accessor_name = "index_tensor_peer",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = PACKER_UNPACKER_SYNC,
        .accessor_name = "packer_unpacker_sync",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = PACKER_UNPACKER_SYNC,
        .accessor_name = "packer_unpacker_sync",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    if (is_row_major) {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = RM_INPUT,
            .accessor_name = "rm_input",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = RM_VALUE_OUTPUT,
            .accessor_name = "rm_value_output",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = RM_INDEX_OUTPUT,
            .accessor_name = "rm_index_output",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = RM_POST_SORT_INDEX,
            .accessor_name = "rm_post_sort_index",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = RM_POST_SORT_INDEX,
            .accessor_name = "rm_post_sort_index",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    } else {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = VALUE_TENSOR,
            .accessor_name = "value_tensor",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = INDEX_OUTPUT,
            .accessor_name = "index_tensor_output",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
    }

    // -----------------------------------------------------------------------
    // Kernels
    // -----------------------------------------------------------------------
    auto* const device = tensor_args.input_tensor.device();

    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/"
                  "reader_cross_core_data_exchange.cpp",
        .compiler_options = {.defines = layout_defines(is_row_major)},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .semaphore_bindings =
            {
                SemaphoreBinding{.semaphore_spec_name = SEM_EXCHANGE, .accessor_name = "exchange"},
                SemaphoreBinding{.semaphore_spec_name = SEM_BARRIER, .accessor_name = "barrier"},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = INPUT_PARAM, .accessor_name = "input_tensor"},
                TensorBinding{.tensor_parameter_name = INDEX_PARAM, .accessor_name = "index_tensor"},
                TensorBinding{.tensor_parameter_name = LOOKUP_PARAM, .accessor_name = "physical_core_lookup_table"},
            },
        .compile_time_args =
            {
                {"compute_with_storage_grid_size_x", static_cast<uint32_t>(compute_with_storage_grid_size.x)},
                {"compute_with_storage_grid_size_y", static_cast<uint32_t>(compute_with_storage_grid_size.y)},
                {"Ht", Ht},
                {"Wt", Wt},
                {"number_of_tiles_per_core", number_of_tiles_per_core},
                {"number_of_cores_used", all_core_utilization_count},
                {"ascending", static_cast<uint32_t>(!attributes.descending)},
                {"W_value_slice_bytes", W_value_slice_bytes},
                {"W_index_slice_bytes", W_index_slice_bytes},
                // The peer exchange moves TILE-format tiles in both configurations, but in ROW_MAJOR
                // the reader binds no buffer paged at those sizes, so they travel as scalars.
                {"input_tensor_tile_size_bytes", input_tensor_tile_size},
                {"index_tensor_tile_size_bytes", index_tensor_tile_size},
            },
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    });

    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/"
                  "writer_cross_core_data_exchange.cpp",
        .compiler_options = {.defines = layout_defines(is_row_major)},
        .dfb_bindings = std::move(writer_dfb_bindings),
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = VALUE_PARAM, .accessor_name = "value_tensor"},
            },
        .compile_time_args =
            {
                {"compute_with_storage_grid_size_x", static_cast<uint32_t>(compute_with_storage_grid_size.x)},
                {"compute_with_storage_grid_size_y", static_cast<uint32_t>(compute_with_storage_grid_size.y)},
                {"Wt", Wt},
                {"Ht", Ht},
                {"number_of_tiles_per_core", number_of_tiles_per_core},
                {"number_of_cores_used", total_number_of_cores_virtual},
                {"is_32_bit_data", static_cast<uint32_t>(is_32_bit_data)},
                {"W_value_slice_bytes", W_value_slice_bytes},
            },
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    });

    ComputeGen1Config compute_hw_config{.enable_32_bit_dest = is_32_bit_data};
    if (input_tensor_cb_data_format == tt::DataFormat::Float32) {
        // Only buffers this configuration actually binds may appear here, and an absent entry
        // means UnpackToSrc. Float32 operands are unpacked straight to Dest so the sort keeps full
        // precision; the index buffers stay on the default path.
        compute_hw_config.unpack_modes.insert({INPUT_TENSOR, UnpackMode::UnpackToDest});
        compute_hw_config.unpack_modes.insert({INPUT_TRANSPOSED, UnpackMode::UnpackToDest});
        compute_hw_config.unpack_modes.insert({VALUE_INTERMEDIATE, UnpackMode::UnpackToDest});
        compute_hw_config.unpack_modes.insert({VALUE_PEER, UnpackMode::UnpackToDest});
        if (is_row_major) {
            compute_hw_config.unpack_modes.insert({RM_INPUT, UnpackMode::UnpackToDest});
        } else {
            compute_hw_config.unpack_modes.insert({VALUE_TENSOR, UnpackMode::UnpackToDest});
        }
    }

    spec.kernels.push_back(KernelSpec{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/compute/"
                  "sort_cross_core_data_exchange.cpp",
        // Compute kernels build at O3; the compiler-options default is O2, so the level is stated
        // explicitly rather than inherited.
        .compiler_options =
            {.defines = layout_defines(is_row_major), .opt_level = KernelSpec::CompilerOptions::OptLevel::O3},
        .dfb_bindings = std::move(compute_dfb_bindings),
        .compile_time_args =
            {
                {"compute_with_storage_grid_size_x", static_cast<uint32_t>(compute_with_storage_grid_size.x)},
                {"compute_with_storage_grid_size_y", static_cast<uint32_t>(compute_with_storage_grid_size.y)},
                {"Ht", Ht},
                {"Wt", Wt},
                {"number_of_tiles_per_core", number_of_tiles_per_core},
                {"number_of_cores_used", all_core_utilization_count},
                {"ascending", static_cast<uint32_t>(!attributes.descending)},
            },
        .hw_config = ComputeHardwareConfig{compute_hw_config},
    });

    spec.work_units.push_back(WorkUnitSpec{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = core_range,
    });

    // -----------------------------------------------------------------------
    // Run args
    // -----------------------------------------------------------------------
    // Every kernel derives its per-core work from its own node coordinates, so no runtime args
    // remain once the buffer addresses become tensor bindings.
    ProgramRunArgs run_args;
    run_args.tensor_args.emplace(INPUT_PARAM, input_mesh_tensor);
    run_args.tensor_args.emplace(VALUE_PARAM, value_mesh_tensor);
    run_args.tensor_args.emplace(INDEX_PARAM, index_mesh_tensor);
    run_args.tensor_args.emplace(LOOKUP_PARAM, lookup_mesh_tensor);

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec), .run_params = std::move(run_args), .op_owned_tensors = std::move(op_owned)};
}

// Single row - multi core
ttnn::device_operation::ProgramArtifacts SortProgramFactorySingleRowMultiCore::create_program_artifacts(
    const SortParams& attributes, const SortInputs& tensor_args, std::vector<Tensor>& output_tensors) {
    const tt::DataFormat input_tensor_cb_data_format =
        datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());
    const tt::DataFormat index_tensor_cb_data_format = datatype_to_dataformat_converter(output_tensors.at(1).dtype());

    const uint32_t input_tensor_tile_size = tile_size(input_tensor_cb_data_format);
    const uint32_t index_tensor_tile_size = tile_size(index_tensor_cb_data_format);

    const auto& input_mesh_tensor = tensor_args.input_tensor.mesh_tensor();
    const auto& value_mesh_tensor = output_tensors.at(0).mesh_tensor();
    const auto& index_mesh_tensor = output_tensors.at(1).mesh_tensor();

    const auto tile_width = tensor_args.input_tensor.tensor_spec().tile().get_width();
    const auto tile_height = tensor_args.input_tensor.tensor_spec().tile().get_height();

    const bool is_row_major = (tensor_args.input_tensor.layout() == Layout::ROW_MAJOR);

    const auto input_shape =
        is_row_major ? tensor_args.input_tensor.logical_shape() : tensor_args.input_tensor.padded_shape();
    const uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / tile_height;
    const uint32_t Wt = input_shape[3] / tile_width;

    const uint32_t value_element_size = tt::datum_size(input_tensor_cb_data_format);
    const uint32_t W_tile_bytes = tile_width * value_element_size;
    const uint32_t index_element_size = tt::datum_size(index_tensor_cb_data_format);
    const uint32_t W_index_bytes = tile_width * index_element_size;

    auto* device = tensor_args.input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t total_number_of_cores = compute_with_storage_grid_size.y * compute_with_storage_grid_size.x;

    const uint32_t total_work_units = Wt / 2;
    const uint32_t number_of_available_cores = total_number_of_cores - 1;

    const uint32_t all_core_utilization_loop_count = total_work_units / number_of_available_cores;

    const bool is_32_bit_index = index_tensor_cb_data_format == tt::DataFormat::UInt32;
    // UINT16 keys must also run in fp32_dest_acc_en mode so the SFPU compares
    // them as 32-bit floats.  See comment in SortProgramFactorySingleRowSingleCore
    // for the full rationale.
    const bool is_32_bit_data = is_32_bit_index || input_tensor_cb_data_format == tt::DataFormat::Float32 ||
                                input_tensor_cb_data_format == tt::DataFormat::UInt16;

    const bool is_uint16_input = (input_tensor_cb_data_format == tt::DataFormat::UInt16);
    const tt::DataFormat sort_value_cb_data_format =
        is_uint16_input ? tt::DataFormat::Float32 : input_tensor_cb_data_format;
    const uint32_t sort_value_tile_size = tile_size(sort_value_cb_data_format);
    // For UINT16 inputs, c_0 must be Float32 because the reader does a software
    // UInt16 → Float32 conversion (hardware unpack cannot convert UInt16 → Float32).
    // See SortProgramFactorySingleRowSingleCore for the full rationale.
    const tt::DataFormat input_cb_data_format = is_uint16_input ? tt::DataFormat::Float32 : input_tensor_cb_data_format;
    const uint32_t input_cb_tile_size = is_uint16_input ? sort_value_tile_size : input_tensor_tile_size;

    const uint32_t log2Wt = std::log2(Wt);

    CoreCoord coordinator_core = {compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1};
    CoreRangeSet core_range;
    if (all_core_utilization_loop_count > 0) {
        core_range = CoreRangeSet(
            CoreRange({0, 0}, {compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 2}));
        core_range = core_range.merge<CoreRangeSet>(CoreRangeSet(CoreRange(
            {0, compute_with_storage_grid_size.y - 1},
            {compute_with_storage_grid_size.x - 2, compute_with_storage_grid_size.y - 1})));
    } else {
        const uint32_t core_grid_calculated_rows_number = total_work_units / compute_with_storage_grid_size.x;
        const uint32_t core_grid_calculated_columns_number = total_work_units % compute_with_storage_grid_size.x;

        if (core_grid_calculated_rows_number == 0 && core_grid_calculated_columns_number == 0) {
            core_range = CoreRangeSet(CoreCoord(0, 0));
        } else if (core_grid_calculated_rows_number == 0) {
            core_range = CoreRangeSet(CoreRange({0, 0}, {core_grid_calculated_columns_number - 1, 0}));
        } else {
            core_range = CoreRangeSet(
                CoreRange({0, 0}, {compute_with_storage_grid_size.x - 1, core_grid_calculated_rows_number - 1}));
            if (core_grid_calculated_columns_number != 0) {
                const CoreRange additional_range(
                    {0, core_grid_calculated_rows_number},
                    {core_grid_calculated_columns_number - 1, core_grid_calculated_rows_number});
                core_range = core_range.merge(CoreRangeSet(additional_range));
            }
        }
    }
    CoreRangeSet all_core_set({CoreRange(coordinator_core)});
    all_core_set = all_core_set.merge<CoreRangeSet>(core_range);

    // -----------------------------------------------------------------------
    // Resource names
    //
    // The coordinator and the workers run different kernels on disjoint nodes and share no dataflow
    // buffer, so each side gets its own buffer specs. Placement is derived from the bindings, which
    // is what keeps the coordinator node from carrying the worker buffers it never touches.
    // -----------------------------------------------------------------------
    const KernelSpecName COORDINATOR{"coordinator"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    const DFBSpecName COORD_INPUT{"coord_input_tensor"};
    const DFBSpecName COORD_INDEX{"coord_index_tensor"};
    const DFBSpecName COORD_VALUE_ROW{"rm_coord_value_row"};
    const DFBSpecName COORD_INDEX_ROW{"rm_coord_index_row"};

    const DFBSpecName WORKER_INPUT{"input_tensor"};
    const DFBSpecName WORKER_INDEX{"index_tensor"};
    const DFBSpecName WORKER_INPUT_TRANSPOSED{"input_tensor_transposed"};
    const DFBSpecName WORKER_INDEX_TRANSPOSED{"index_tensor_transposed"};
    const DFBSpecName WORKER_VALUE_OUTPUT{"input_tensor_output"};
    const DFBSpecName WORKER_INDEX_OUTPUT{"index_tensor_output"};
    const DFBSpecName WORKER_RM_IN_VALUE{"rm_worker_input_value"};
    const DFBSpecName WORKER_RM_IN_INDEX{"rm_worker_input_index"};
    const DFBSpecName WORKER_RM_OUT_VALUE{"rm_worker_output_value"};
    const DFBSpecName WORKER_RM_OUT_INDEX{"rm_worker_output_index"};

    // Staging buffers for the UInt16 software conversion. Both are one raw UInt16 tile, which is
    // wide enough to also hold a pair-row on the row-major path, so one buffer per direction serves
    // both layouts. They exist only when the input dtype is UInt16.
    const DFBSpecName WORKER_UINT16_INPUT_STAGE{"uint16_input_stage"};
    const DFBSpecName WORKER_UINT16_CONV{"uint16_conv"};

    const SemaphoreSpecName SEM_COORD_TO_CORES{"coordinator_to_cores"};
    const SemaphoreSpecName SEM_CORES_TO_COORD_READY{"cores_to_coordinator_ready"};
    const SemaphoreSpecName SEM_CORES_TO_COORD_DONE{"cores_to_coordinator_done"};

    const TensorParamName INPUT_PARAM{"input"};
    const TensorParamName VALUE_PARAM{"value_output"};
    const TensorParamName INDEX_PARAM{"index_output"};

    ProgramSpec spec;
    spec.name = "sort_single_row_multi_core";

    // -----------------------------------------------------------------------
    // Dataflow buffers
    // -----------------------------------------------------------------------
    constexpr uint32_t buffer_scale_factor = 2;

    if (!is_row_major) {
        // The coordinator stages one tile at a time through these while copying the input to the
        // output in DRAM and generating the index tensor there.
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = COORD_INPUT,
            .entry_size = input_cb_tile_size,
            .num_entries = buffer_scale_factor,
            .data_format_metadata = input_cb_data_format,
        });
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = COORD_INDEX,
            .entry_size = index_tensor_tile_size,
            .num_entries = buffer_scale_factor,
            .data_format_metadata = index_tensor_cb_data_format,
        });
    }

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = WORKER_INPUT,
        .entry_size = input_cb_tile_size,
        .num_entries = buffer_scale_factor,
        .data_format_metadata = input_cb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = WORKER_INDEX,
        .entry_size = index_tensor_tile_size,
        .num_entries = buffer_scale_factor,
        .data_format_metadata = index_tensor_cb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = WORKER_INPUT_TRANSPOSED,
        .entry_size = sort_value_tile_size,
        .num_entries = buffer_scale_factor,
        .data_format_metadata = sort_value_cb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = WORKER_INDEX_TRANSPOSED,
        .entry_size = index_tensor_tile_size,
        .num_entries = buffer_scale_factor,
        .data_format_metadata = index_tensor_cb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = WORKER_VALUE_OUTPUT,
        .entry_size = sort_value_tile_size,
        .num_entries = buffer_scale_factor,
        .data_format_metadata = sort_value_cb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = WORKER_INDEX_OUTPUT,
        .entry_size = index_tensor_tile_size,
        .num_entries = buffer_scale_factor,
        .data_format_metadata = index_tensor_cb_data_format,
    });

    // The worker row-major value and index buffers hold pair-rows: each entry is 2 * TILE_W wide,
    // the concatenation of the left tile's row and the right tile's row. That is what
    // tilize_block(buffer, 2, out) expects, namely TILE_H rows of 2 * TILE_W elements laid out
    // contiguously. The reader builds this layout by reading the left half and the right half of
    // each row into a single reserved entry.
    //
    // The coordinator's own row buffers stay at the raw input dtype and one raw row per entry: its
    // row-major branch only passes raw rows from the input tensor through to the output tensor.
    const uint32_t W_sort_value_bytes = tile_width * tt::datum_size(sort_value_cb_data_format);
    const uint32_t W_pair_value_bytes = 2 * W_sort_value_bytes;
    const uint32_t W_pair_index_bytes = 2 * W_index_bytes;

    if (is_row_major) {
        constexpr uint32_t TILE_H = tt::constants::TILE_HEIGHT;

        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = COORD_VALUE_ROW,
            .entry_size = W_tile_bytes,
            .num_entries = 1,
            .data_format_metadata = input_tensor_cb_data_format,
        });
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = COORD_INDEX_ROW,
            .entry_size = W_index_bytes,
            .num_entries = 1,
            .data_format_metadata = index_tensor_cb_data_format,
        });
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = WORKER_RM_IN_VALUE,
            .entry_size = W_pair_value_bytes,
            .num_entries = TILE_H,
            .data_format_metadata = sort_value_cb_data_format,
        });
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = WORKER_RM_IN_INDEX,
            .entry_size = W_pair_index_bytes,
            .num_entries = TILE_H,
            .data_format_metadata = index_tensor_cb_data_format,
        });
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = WORKER_RM_OUT_VALUE,
            .entry_size = W_pair_value_bytes,
            .num_entries = TILE_H,
            .data_format_metadata = sort_value_cb_data_format,
        });
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = WORKER_RM_OUT_INDEX,
            .entry_size = W_pair_index_bytes,
            .num_entries = TILE_H,
            .data_format_metadata = index_tensor_cb_data_format,
        });
    }

    if (is_uint16_input) {
        // The reader stages raw UInt16 data here before converting it to Float32, and the writer
        // converts sorted Float32 values back into its own staging buffer before writing to DRAM.
        // Each is both produced and consumed by the single kernel that owns it.
        const uint32_t uint16_one_tile_size = tile_size(tt::DataFormat::UInt16);
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = WORKER_UINT16_INPUT_STAGE,
            .entry_size = uint16_one_tile_size,
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::UInt16,
        });
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = WORKER_UINT16_CONV,
            .entry_size = uint16_one_tile_size,
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::UInt16,
        });
    }

    // -----------------------------------------------------------------------
    // Semaphores. The cores to coordinator channel uses two separate semaphores so a fast reader's
    // next-row readiness increment can never be miscounted as a sub-stage confirmation: on one
    // shared counter it could overshoot the coordinator's exact-match wait and deadlock the op at
    // Ht >= 2. Readiness goes to the ready semaphore, per-pair confirmations to the done semaphore.
    // -----------------------------------------------------------------------
    spec.semaphores.push_back(SemaphoreSpec{.unique_id = SEM_COORD_TO_CORES, .target_nodes = all_core_set});
    spec.semaphores.push_back(SemaphoreSpec{.unique_id = SEM_CORES_TO_COORD_READY, .target_nodes = all_core_set});
    spec.semaphores.push_back(SemaphoreSpec{.unique_id = SEM_CORES_TO_COORD_DONE, .target_nodes = all_core_set});

    // -----------------------------------------------------------------------
    // Tensor parameters
    // -----------------------------------------------------------------------
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = INPUT_PARAM, .spec = input_mesh_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = VALUE_PARAM, .spec = value_mesh_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = INDEX_PARAM, .spec = index_mesh_tensor.tensor_spec()});

    // -----------------------------------------------------------------------
    // Kernel resource bindings
    // -----------------------------------------------------------------------
    Group<DFBBinding> coordinator_dfb_bindings;
    Group<DFBBinding> reader_dfb_bindings;
    Group<DFBBinding> writer_dfb_bindings;
    Group<DFBBinding> compute_dfb_bindings;

    // The coordinator both fills its staging buffers from DRAM and drains them back, so it holds
    // both ends of each.
    if (is_row_major) {
        coordinator_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = COORD_VALUE_ROW,
            .accessor_name = "rm_coord_value_row",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        coordinator_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = COORD_VALUE_ROW,
            .accessor_name = "rm_coord_value_row",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        coordinator_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = COORD_INDEX_ROW,
            .accessor_name = "rm_coord_index_row",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        coordinator_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = COORD_INDEX_ROW,
            .accessor_name = "rm_coord_index_row",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    } else {
        coordinator_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = COORD_INPUT,
            .accessor_name = "input_tensor",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        coordinator_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = COORD_INPUT,
            .accessor_name = "input_tensor",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        coordinator_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = COORD_INDEX,
            .accessor_name = "index_tensor",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        coordinator_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = COORD_INDEX,
            .accessor_name = "index_tensor",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }

    if (is_row_major) {
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WORKER_RM_IN_VALUE,
            .accessor_name = "rm_input_value",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WORKER_RM_IN_INDEX,
            .accessor_name = "rm_input_index",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
    } else {
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WORKER_INPUT,
            .accessor_name = "input_tensor",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WORKER_INDEX,
            .accessor_name = "index_tensor",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
    }

    if (is_uint16_input) {
        // The reader fills the staging buffer over the NoC and then reads it back to convert each
        // element, so it holds both ends by itself.
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WORKER_UINT16_INPUT_STAGE,
            .accessor_name = "uint16_input_stage",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WORKER_UINT16_INPUT_STAGE,
            .accessor_name = "uint16_input_stage",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }

    if (is_row_major) {
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WORKER_RM_OUT_VALUE,
            .accessor_name = "rm_output_value",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WORKER_RM_OUT_INDEX,
            .accessor_name = "rm_output_index",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    } else {
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WORKER_VALUE_OUTPUT,
            .accessor_name = "input_tensor_output",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WORKER_INDEX_OUTPUT,
            .accessor_name = "index_tensor_output",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }

    if (is_uint16_input) {
        // Mirror of the reader's staging buffer: the writer fills it with converted values and then
        // sources its own NoC write from it.
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WORKER_UINT16_CONV,
            .accessor_name = "uint16_conv",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WORKER_UINT16_CONV,
            .accessor_name = "uint16_conv",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }

    // In ROW_MAJOR the compute kernel fills the tile-format buffers itself from the tilize and
    // drains the output ones through pack_untilize, so it holds both ends of all four.
    if (is_row_major) {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WORKER_INPUT,
            .accessor_name = "input_tensor",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WORKER_INDEX,
            .accessor_name = "index_tensor",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
    }
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = WORKER_INPUT,
        .accessor_name = "input_tensor",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = WORKER_INDEX,
        .accessor_name = "index_tensor",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = WORKER_INPUT_TRANSPOSED,
        .accessor_name = "input_tensor_transposed",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = WORKER_INPUT_TRANSPOSED,
        .accessor_name = "input_tensor_transposed",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = WORKER_INDEX_TRANSPOSED,
        .accessor_name = "index_tensor_transposed",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = WORKER_INDEX_TRANSPOSED,
        .accessor_name = "index_tensor_transposed",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = WORKER_VALUE_OUTPUT,
        .accessor_name = "input_tensor_output",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    compute_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = WORKER_INDEX_OUTPUT,
        .accessor_name = "index_tensor_output",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    if (is_row_major) {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WORKER_VALUE_OUTPUT,
            .accessor_name = "input_tensor_output",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WORKER_INDEX_OUTPUT,
            .accessor_name = "index_tensor_output",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WORKER_RM_IN_VALUE,
            .accessor_name = "rm_input_value",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WORKER_RM_IN_INDEX,
            .accessor_name = "rm_input_index",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WORKER_RM_OUT_VALUE,
            .accessor_name = "rm_output_value",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WORKER_RM_OUT_INDEX,
            .accessor_name = "rm_output_index",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
    }

    // -----------------------------------------------------------------------
    // Kernels
    // -----------------------------------------------------------------------
    const auto coordinator_core_physical_coord = device->worker_core_from_logical_core(coordinator_core);
    const auto start_core_logical = core_range.ranges()[0].start_coord;
    const auto start_core_physical_coord = device->worker_core_from_logical_core(start_core_logical);

    // Coordinator's `set_multicast(..., num_dests)` requires `num_dests` to match the
    // number of actual destinations in the multicast rectangle (per dataflow_api.h:
    // "when mcasting to an 8x8 grid that includes self, num_dests should be 63").
    //
    // In the FULL-GRID case core_range spans (0,0)..(grid_x-1,grid_y-1) minus the
    // coord corner (63 workers on an 8x8 grid).  The bounding box then covers the
    // full grid including the coord, so multicasting to it with num_dests == 63
    // matches: rect size (64) − src (coord in rect) = 63.
    //
    // In the PARTIAL-GRID case core_range is a strict sub-rectangle of the grid
    // that does NOT contain the coord (e.g. Wt=64 → workers at (0,0)-(7,3), coord
    // at (7,7)).  Previously we still passed the coord as the multicast end coord,
    // which produced a rectangle covering all 63 non-coord cores while
    // `num_dests` counted only the 32 workers.  The NoC then acknowledged 63
    // destinations while the coord's barrier only accounted for 32 → NoC ack
    // counter drift → workers never get the "go" semaphore → hang.
    //
    // Fix: derive the multicast rectangle from `core_range.bounding_box()` and
    // compute `num_multicast_dests` from the bounding-box size (subtracting one
    // when the coord — the multicast source — is inside the bbox, since
    // `noc_semaphore_set_multicast` does not deliver to self).  `num_workers`
    // (used for the coord's `wait()` on per-worker signals) stays as
    // `core_range.num_cores()`.
    const CoreRange multicast_bbox = core_range.bounding_box();
    const auto end_core_physical_coord = device->worker_core_from_logical_core(multicast_bbox.end_coord);
    const bool coord_in_multicast_bbox = multicast_bbox.contains(coordinator_core);
    const uint32_t num_multicast_dests =
        static_cast<uint32_t>(multicast_bbox.size()) - (coord_in_multicast_bbox ? 1u : 0u);
    const uint32_t num_workers = static_cast<uint32_t>(core_range.num_cores());

    spec.kernels.push_back(KernelSpec{
        .unique_id = COORDINATOR,
        .source = "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/"
                  "coordinator_single_row_multi_core.cpp",
        .compiler_options = {.defines = sort_kernel_defines(is_row_major, is_uint16_input)},
        .dfb_bindings = std::move(coordinator_dfb_bindings),
        .semaphore_bindings =
            {
                SemaphoreBinding{.semaphore_spec_name = SEM_COORD_TO_CORES, .accessor_name = "coordinator_to_cores"},
                SemaphoreBinding{
                    .semaphore_spec_name = SEM_CORES_TO_COORD_READY, .accessor_name = "cores_to_coordinator_ready"},
                SemaphoreBinding{
                    .semaphore_spec_name = SEM_CORES_TO_COORD_DONE, .accessor_name = "cores_to_coordinator_done"},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = INPUT_PARAM, .accessor_name = "input_tensor"},
                TensorBinding{.tensor_parameter_name = VALUE_PARAM, .accessor_name = "output_tensor"},
                TensorBinding{.tensor_parameter_name = INDEX_PARAM, .accessor_name = "output_index_tensor"},
            },
        .compile_time_args =
            {
                {"Wt", Wt},
                {"Ht", Ht},
                {"is_32_bit_data", static_cast<uint32_t>(is_32_bit_data)},
                {"W_tile_bytes", W_tile_bytes},
                {"W_index_bytes", W_index_bytes},
                {"tile_width", static_cast<uint32_t>(tile_width)},
            },
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"start_core_physical_coord_x",
                  "start_core_physical_coord_y",
                  "end_core_physical_coord_x",
                  "end_core_physical_coord_y",
                  "number_of_workers",
                  "num_multicast_dests"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    });

    // The workers read their input from the value-output buffer: the coordinator has already copied
    // the input tensor there and generated the index tensor alongside it, and the sort then runs in
    // place in those two output buffers.
    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/"
                  "reader_single_row_multi_core.cpp",
        .compiler_options = {.defines = sort_kernel_defines(is_row_major, is_uint16_input)},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .semaphore_bindings =
            {
                SemaphoreBinding{.semaphore_spec_name = SEM_COORD_TO_CORES, .accessor_name = "coordinator_to_cores"},
                SemaphoreBinding{
                    .semaphore_spec_name = SEM_CORES_TO_COORD_READY, .accessor_name = "cores_to_coordinator_ready"},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = VALUE_PARAM, .accessor_name = "input_tensor"},
                TensorBinding{.tensor_parameter_name = INDEX_PARAM, .accessor_name = "index_tensor"},
            },
        .compile_time_args =
            {
                {"Wt", Wt},
                {"Ht", Ht},
                {"compute_with_storage_grid_size_x", static_cast<uint32_t>(compute_with_storage_grid_size.x)},
                {"number_of_available_cores", number_of_available_cores},
                {"W_tile_bytes", W_tile_bytes},
                {"W_index_bytes", W_index_bytes},
            },
        // The coordinator sits on one fixed core, so every worker addresses the same pair of
        // coordinates. Broadcasting them as common arguments keeps one copy per program instead
        // of one per worker node.
        .runtime_arg_schema =
            {.common_runtime_arg_names = {"coordinator_core_physical_coord_x", "coordinator_core_physical_coord_y"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    });

    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/"
                  "writer_single_row_multi_core.cpp",
        .compiler_options = {.defines = sort_kernel_defines(is_row_major, is_uint16_input)},
        .dfb_bindings = std::move(writer_dfb_bindings),
        .semaphore_bindings =
            {
                SemaphoreBinding{
                    .semaphore_spec_name = SEM_CORES_TO_COORD_DONE, .accessor_name = "cores_to_coordinator_done"},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = VALUE_PARAM, .accessor_name = "input_tensor"},
                TensorBinding{.tensor_parameter_name = INDEX_PARAM, .accessor_name = "index_tensor"},
            },
        .compile_time_args =
            {
                {"Wt", Wt},
                {"Ht", Ht},
                {"compute_with_storage_grid_size_x", static_cast<uint32_t>(compute_with_storage_grid_size.x)},
                {"number_of_available_cores", number_of_available_cores},
                {"W_tile_bytes", W_tile_bytes},
                {"W_index_bytes", W_index_bytes},
            },
        // See the reader above: one fixed coordinator core means one value for every worker.
        .runtime_arg_schema =
            {.common_runtime_arg_names = {"coordinator_core_physical_coord_x", "coordinator_core_physical_coord_y"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    });

    ComputeGen1Config compute_hw_config{.enable_32_bit_dest = is_32_bit_data};
    if (input_tensor_cb_data_format == tt::DataFormat::Float32 || is_uint16_input) {
        // Only buffers this configuration actually binds may appear here, and an absent entry
        // means UnpackToSrc. Float32 operands are unpacked straight to Dest so the sort keeps full
        // precision; the index buffers stay on the default path.
        compute_hw_config.unpack_modes.insert({WORKER_INPUT, UnpackMode::UnpackToDest});
        compute_hw_config.unpack_modes.insert({WORKER_INPUT_TRANSPOSED, UnpackMode::UnpackToDest});
        compute_hw_config.unpack_modes.insert({WORKER_VALUE_OUTPUT, UnpackMode::UnpackToDest});
        if (is_row_major) {
            compute_hw_config.unpack_modes.insert({WORKER_RM_IN_VALUE, UnpackMode::UnpackToDest});
        }
    }

    spec.kernels.push_back(KernelSpec{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/compute/"
                  "sort_single_row_multi_core.cpp",
        // Compute kernels build at O3; the compiler-options default is O2, so the level is stated
        // explicitly rather than inherited.
        .compiler_options =
            {.defines = sort_kernel_defines(is_row_major, is_uint16_input),
             .opt_level = KernelSpec::CompilerOptions::OptLevel::O3},
        .dfb_bindings = std::move(compute_dfb_bindings),
        .compile_time_args =
            {
                {"Wt", Wt},
                {"Ht", Ht},
                {"number_of_available_cores", number_of_available_cores},
                {"compute_with_storage_grid_size_x", static_cast<uint32_t>(compute_with_storage_grid_size.x)},
                {"descending", static_cast<uint32_t>(attributes.descending)},
                {"stable", static_cast<uint32_t>(attributes.stable)},
                {"log2Wt", log2Wt},
            },
        .hw_config = ComputeHardwareConfig{compute_hw_config},
    });

    spec.work_units.push_back(WorkUnitSpec{
        .name = "coordinator",
        .kernels = {COORDINATOR},
        .target_nodes = CoreRangeSet(CoreRange(coordinator_core)),
    });
    spec.work_units.push_back(WorkUnitSpec{
        .name = "workers",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = core_range,
    });

    // -----------------------------------------------------------------------
    // Run args
    // -----------------------------------------------------------------------
    KernelRunArgs coordinator_run_args{
        .kernel = COORDINATOR,
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(
            coordinator_core,
            {{"start_core_physical_coord_x", static_cast<uint32_t>(start_core_physical_coord.x)},
             {"start_core_physical_coord_y", static_cast<uint32_t>(start_core_physical_coord.y)},
             {"end_core_physical_coord_x", static_cast<uint32_t>(end_core_physical_coord.x)},
             {"end_core_physical_coord_y", static_cast<uint32_t>(end_core_physical_coord.y)},
             {"number_of_workers", num_workers},
             {"num_multicast_dests", num_multicast_dests}}),
    };

    const KernelRunArgs::CommonRuntimeArgValues coordinator_coord_args{
        {"coordinator_core_physical_coord_x", static_cast<uint32_t>(coordinator_core_physical_coord.x)},
        {"coordinator_core_physical_coord_y", static_cast<uint32_t>(coordinator_core_physical_coord.y)}};

    KernelRunArgs reader_run_args{.kernel = READER, .common_runtime_arg_values = coordinator_coord_args};
    KernelRunArgs writer_run_args{.kernel = WRITER, .common_runtime_arg_values = coordinator_coord_args};

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(coordinator_run_args));
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.tensor_args.emplace(INPUT_PARAM, input_mesh_tensor);
    run_args.tensor_args.emplace(VALUE_PARAM, value_mesh_tensor);
    run_args.tensor_args.emplace(INDEX_PARAM, index_mesh_tensor);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}
}  // namespace ttnn::prim
