// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>
#include <variant>

#include <tt-metalium/experimental/mesh_program_descriptor.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/utility/table.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::generic {

// Metal 2.0 form of a caller-supplied program. tensor_args on the ProgramRunArgs is left empty
// and reconstructed by the spec factory from tensor_arg_indices: a TensorArgument must reference,
// by pointer identity, a MeshTensor reachable from the op's tensor_args, so it cannot be handed
// in from Python.
struct SpecProgram {
    tt::tt_metal::experimental::ProgramSpec spec;
    tt::tt_metal::experimental::ProgramRunArgs run_args;
    tt::tt_metal::experimental::Table<tt::tt_metal::experimental::TensorParamName, uint32_t> tensor_arg_indices;
};

// A generic_op call carries either a descriptor program or a spec program; the alternative
// selects the program factory.
struct operation_attributes_t {
    std::variant<tt::tt_metal::experimental::MeshProgramDescriptor, SpecProgram> program;

    bool is_spec() const { return std::holds_alternative<SpecProgram>(program); }
    const tt::tt_metal::experimental::MeshProgramDescriptor& mesh_program_descriptor() const {
        return std::get<tt::tt_metal::experimental::MeshProgramDescriptor>(program);
    }
    const SpecProgram& spec_program() const { return std::get<SpecProgram>(program); }

    // Descriptors and specs are both too large for reflection inline storage; keying is done by
    // GenericOpDeviceOperation::compute_program_hash.
    static constexpr auto attribute_names = std::forward_as_tuple("program_kind");
    auto attribute_values() const { return std::make_tuple(program.index()); }
};

using tensor_return_value_t = Tensor;
using spec_return_value_t = tt::tt_metal::TensorSpec;

// NOTE: output tensor is the last element in the vector io_tensors. io_tensors may hold exactly one
// tensor (a generator program, or a single-tensor in-place program), in which case it is both the
// only argument and the output.
struct tensor_args_t {
    const std::vector<Tensor>& io_tensors;
    const Tensor& output_tensor;
};

}  // namespace ttnn::operations::generic
