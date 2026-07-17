include(sources.cmake)

add_library(ttnn_op_moreh ${LIB_TYPE})
add_library(TTNN::Ops::Moreh ALIAS ttnn_op_moreh)

tt_reuse_precompile_headers(ttnn_op_moreh TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_moreh)
set_target_properties(
    ttnn_op_moreh
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(
    GLOB_RECURSE kernels
    moreh_abs_pow/device/kernels/*.cpp
    moreh_adam/device/kernels/*.cpp
    moreh_adamw/device/kernels/*.cpp
    moreh_arange/device/kernels/*.cpp
    moreh_clip_grad_norm/moreh_clip_grad_norm_step1/device/kernels/*.cpp
    moreh_clip_grad_norm/moreh_clip_grad_norm_step2/device/kernels/*.cpp
    moreh_clip_grad_norm/moreh_clip_grad_norm_step3/device/kernels/*.cpp
    moreh_dot/device/kernels/*.cpp
    moreh_dot_backward/device/kernels/*.cpp
    moreh_fold/device/kernels/*.cpp
    moreh_getitem/device/moreh_getitem_kernels/*.cpp
    moreh_getitem/device/moreh_getitem_tilized_kernels/*.cpp
    moreh_getitem/device/moreh_getitem_tilized_kernels/*.hpp
    moreh_group_norm/device/kernels/*.cpp
    moreh_group_norm_backward/device/gamma_beta_grad/kernels/*.cpp
    moreh_group_norm_backward/device/input_grad/kernels/*.cpp
    moreh_layer_norm/device/kernels/*.cpp
    moreh_layer_norm_backward/device/kernels/*.cpp
    moreh_linear_backward/device/kernels/*.cpp
    moreh_matmul/device/kernels/*.cpp
    moreh_mean/device/kernels/*.cpp
    moreh_mean_backward/device/kernels/*.cpp
    moreh_nll_loss/moreh_nll_loss_step1/device/kernels/*.cpp
    moreh_nll_loss/moreh_nll_loss_step2/device/kernels/*.cpp
    moreh_nll_loss_backward/device/kernels/*.cpp
    moreh_nll_loss_unreduced_backward/device/kernels/*.cpp
    moreh_norm/device/moreh_norm_h/kernels/*.cpp
    moreh_norm/device/moreh_norm_other/kernels/*.cpp
    moreh_norm/device/moreh_norm_w/kernels/*.cpp
    moreh_norm/device/ord_other/moreh_norm_h/kernels/*.cpp
    moreh_norm/device/ord_other/moreh_norm_nc/kernels/*.cpp
    moreh_norm/device/ord_other/moreh_norm_w/kernels/*.cpp
    moreh_norm_backward/device/kernels/*.cpp
    moreh_sgd/device/kernels/*.cpp
    moreh_softmax/device/kernels/*.cpp
    moreh_softmax_backward/device/kernels/*.cpp
    moreh_sum/device/moreh_sum_h_impl_kernels/*.cpp
    moreh_sum/device/moreh_sum_nc_impl_kernels/*.cpp
    moreh_sum/device/moreh_sum_w_impl_kernels/*.cpp
    moreh_sum_backward/device/kernels/*.cpp
)
target_sources(
    ttnn_op_moreh
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_MOREH_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            ${kernels}
            # moreh common
            moreh_helper_functions.hpp
            moreh_nanobind.hpp
            # moreh_abs_pow/
            moreh_abs_pow/moreh_abs_pow.hpp
            moreh_abs_pow/moreh_abs_pow_nanobind.hpp
            moreh_abs_pow/device/moreh_abs_pow_device_operation.hpp
            # moreh_adam/
            moreh_adam/moreh_adam.hpp
            moreh_adam/moreh_adam_nanobind.hpp
            moreh_adam/device/moreh_adam_device_operation.hpp
            # moreh_adamw/
            moreh_adamw/moreh_adamw.hpp
            moreh_adamw/moreh_adamw_nanobind.hpp
            moreh_adamw/device/moreh_adamw_device_operation.hpp
            # moreh_arange/
            moreh_arange/moreh_arange.hpp
            moreh_arange/moreh_arange_nanobind.hpp
            moreh_arange/device/moreh_arange_device_operation.hpp
            # moreh_bmm/
            moreh_bmm/moreh_bmm.hpp
            moreh_bmm/moreh_bmm_nanobind.hpp
            # moreh_bmm_backward/
            moreh_bmm_backward/moreh_bmm_backward.hpp
            moreh_bmm_backward/moreh_bmm_backward_nanobind.hpp
            # moreh_clip_grad_norm/
            moreh_clip_grad_norm/moreh_clip_grad_norm.hpp
            moreh_clip_grad_norm/moreh_clip_grad_norm_nanobind.hpp
            moreh_clip_grad_norm/moreh_clip_grad_norm_step1/device/moreh_clip_grad_norm_step1_device_operation.hpp
            moreh_clip_grad_norm/moreh_clip_grad_norm_step2/device/moreh_clip_grad_norm_step2_device_operation.hpp
            moreh_clip_grad_norm/moreh_clip_grad_norm_step3/device/moreh_clip_grad_norm_step3_device_operation.hpp
            # moreh_cumsum/
            moreh_cumsum/moreh_cumsum.hpp
            moreh_cumsum/moreh_cumsum_nanobind.hpp
            # moreh_dot/
            moreh_dot/moreh_dot.hpp
            moreh_dot/moreh_dot_nanobind.hpp
            moreh_dot/device/moreh_dot_device_operation.hpp
            # moreh_dot_backward/
            moreh_dot_backward/moreh_dot_backward.hpp
            moreh_dot_backward/moreh_dot_backward_nanobind.hpp
            moreh_dot_backward/device/moreh_dot_backward_device_operation.hpp
            # moreh_fold/
            moreh_fold/fold.hpp
            moreh_fold/fold_nanobind.hpp
            moreh_fold/device/fold_device_operation.hpp
            # moreh_getitem/
            moreh_getitem/moreh_getitem.hpp
            moreh_getitem/moreh_getitem_nanobind.hpp
            moreh_getitem/device/moreh_getitem_device_operation.hpp
            # moreh_group_norm/
            # moreh_group_norm_backward/
            # moreh_layer_norm/
            # moreh_layer_norm_backward/
            # moreh_linear/
            moreh_linear/moreh_linear.hpp
            moreh_linear/moreh_linear_nanobind.hpp
            # moreh_linear_backward/
            moreh_linear_backward/moreh_linear_backward.hpp
            moreh_linear_backward/moreh_linear_backward_nanobind.hpp
            moreh_linear_backward/device/moreh_linear_backward_device_operation.hpp
            # moreh_matmul/
            moreh_matmul/moreh_matmul.hpp
            moreh_matmul/moreh_matmul_nanobind.hpp
            moreh_matmul/device/moreh_matmul_device_operation.hpp
            # moreh_matmul_backward/
            moreh_matmul_backward/moreh_matmul_backward.hpp
            moreh_matmul_backward/moreh_matmul_backward_nanobind.hpp
            # moreh_mean/
            moreh_mean/moreh_mean.hpp
            moreh_mean/moreh_mean_nanobind.hpp
            moreh_mean/device/moreh_mean_device_operation.hpp
            # moreh_mean_backward/
            moreh_mean_backward/moreh_mean_backward.hpp
            moreh_mean_backward/moreh_mean_backward_nanobind.hpp
            moreh_mean_backward/device/moreh_mean_backward_device_operation.hpp
            # moreh_nll_loss/
            moreh_nll_loss/moreh_nll_loss.hpp
            moreh_nll_loss/moreh_nll_loss_nanobind.hpp
            moreh_nll_loss/moreh_nll_loss_helper.hpp
            moreh_nll_loss/moreh_nll_loss_step1/device/moreh_nll_loss_step1_device_operation.hpp
            moreh_nll_loss/moreh_nll_loss_step2/device/moreh_nll_loss_step2_device_operation.hpp
            # moreh_nll_loss_backward/
            moreh_nll_loss_backward/moreh_nll_loss_backward.hpp
            moreh_nll_loss_backward/moreh_nll_loss_backward_nanobind.hpp
            moreh_nll_loss_backward/device/moreh_nll_loss_backward_device_operation.hpp
            # moreh_nll_loss_unreduced_backward/
            moreh_nll_loss_unreduced_backward/moreh_nll_loss_unreduced_backward.hpp
            moreh_nll_loss_unreduced_backward/moreh_nll_loss_unreduced_backward_nanobind.hpp
            moreh_nll_loss_unreduced_backward/device/moreh_nll_loss_unreduced_backward_device_operation.hpp
            # moreh_norm/
            moreh_norm/moreh_norm.hpp
            moreh_norm/moreh_norm_nanobind.hpp
            moreh_norm/device/moreh_norm_device_operation.hpp
            # moreh_norm_backward/
            moreh_norm_backward/moreh_norm_backward.hpp
            moreh_norm_backward/moreh_norm_backward_nanobind.hpp
            moreh_norm_backward/device/moreh_norm_backward_device_operation.hpp
            # moreh_sgd/
            moreh_sgd/moreh_sgd.hpp
            moreh_sgd/moreh_sgd_nanobind.hpp
            moreh_sgd/device/moreh_sgd_device_operation.hpp
            # moreh_softmax/
            # moreh_softmax_backward/
            # moreh_sum/
            moreh_sum/moreh_sum.hpp
            moreh_sum/moreh_sum_nanobind.hpp
            moreh_sum/device/moreh_sum_device_operation.hpp
            # moreh_sum_backward/
            moreh_sum_backward/moreh_sum_backward.hpp
            moreh_sum_backward/moreh_sum_backward_nanobind.hpp
            moreh_sum_backward/device/moreh_sum_backward_device_operation.hpp
    PRIVATE
        ${TTNN_OP_MOREH_SRCS}
)

target_include_directories(ttnn_op_moreh PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_moreh PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_moreh
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/moreh
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_moreh LIBRARY COMPONENT tar)

# Python bindings for this op are registered directly on the shared `ttnn`
# Python module target here, rather than in the centrally-owned
# ttnn/sources.cmake — the target itself is necessarily shared (nanobind
# builds one Python extension module), but which file lists a given op's
# own registration source doesn't need to be. `ttnn` only exists when
# WITH_PYTHON_BINDINGS is on, hence the guard.
#
# The actual file list (TTNN_OP_MOREH_NANOBIND_SRCS) lives in sources.cmake, not here: this
# file carries metalium-developers-infra as a required co-owner (so infra
# can land sweeping build-system changes), and routine add/remove/rename of
# a nanobind source shouldn't need to go through infra. This wiring line
# itself is build logic and is expected to change rarely.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_MOREH_NANOBIND_SRCS})
endif()
