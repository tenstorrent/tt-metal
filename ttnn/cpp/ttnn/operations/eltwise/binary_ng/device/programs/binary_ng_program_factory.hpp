#pragma once

#include "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation_types.hpp"

namespace ttnn::operations::binary_ng::program {

struct BinaryNgProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id;
        tt::tt_metal::KernelHandle writer_kernel_id;
        tt::tt_metal::KernelHandle compute_kernel_id;
        tt::tt_metal::CBHandle cb_src_a;
        tt::tt_metal::CBHandle cb_src_b;
        tt::tt_metal::CBHandle cb_src_c;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const BinaryNgParams& operation_attributes, const BinaryNgInputs& tensor_args, Tensor& c);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const BinaryNgParams& operation_attributes,
        const BinaryNgInputs& tensor_args,
        Tensor& c);
};
}  // namespace ttnn::operations::binary_ng::program
