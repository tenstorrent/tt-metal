add_library(ttnn_op_moreh ${LIB_TYPE})
add_library(TTNN::Ops::Moreh ALIAS ttnn_op_moreh)

target_precompile_headers(ttnn_op_moreh REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_moreh)
set_target_properties(
    ttnn_op_moreh
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(
    GLOB_RECURSE kernels
    moreh_dot/device/kernels/*
    moreh_getitem/device/moreh_getitem_kernels/*
    moreh_getitem/device/moreh_getitem_tilized_kernels/*
)
target_sources(
    ttnn_op_moreh
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            ${kernels}
            moreh_sum/device/moreh_sum_w_impl_kernels/moreh_sum_w.cpp
            moreh_sum/device/moreh_sum_w_impl_kernels/reader_moreh_sum_w.cpp
            moreh_sum/device/moreh_sum_w_impl_kernels/writer_moreh_sum_w.cpp
    PRIVATE
        moreh_abs_pow/device/moreh_abs_pow_device_operation.cpp
        moreh_abs_pow/device/moreh_abs_pow_program_factory.cpp
        moreh_abs_pow/moreh_abs_pow.cpp
        moreh_adam/device/moreh_adam_device_operation.cpp
        moreh_adam/device/moreh_adam_program_factory.cpp
        moreh_adam/moreh_adam.cpp
        moreh_adamw/device/moreh_adamw_device_operation.cpp
        moreh_adamw/device/multi_core_program_factory.cpp
        moreh_adamw/moreh_adamw.cpp
        moreh_arange/device/moreh_arange_device_operation.cpp
        moreh_arange/device/moreh_arange_program_factory.cpp
        moreh_arange/moreh_arange.cpp
        moreh_bmm_backward/moreh_bmm_backward.cpp
        moreh_bmm/moreh_bmm.cpp
        moreh_cumsum/moreh_cumsum.cpp
        moreh_dot_backward/device/moreh_dot_backward_device_operation.cpp
        moreh_dot_backward/device/moreh_dot_backward_program_factory.cpp
        moreh_dot_backward/moreh_dot_backward.cpp
        moreh_dot/device/moreh_dot_device_operation.cpp
        moreh_dot/device/moreh_dot_device_operation.cpp
        moreh_dot/device/moreh_dot_program_factory.cpp
        moreh_dot/device/moreh_dot_program_factory.cpp
        moreh_dot/moreh_dot.cpp
        moreh_dot/moreh_dot.cpp
        moreh_getitem/device/moreh_getitem_device_operation.cpp
        moreh_getitem/device/moreh_getitem_rm_factory.cpp
        moreh_getitem/device/moreh_getitem_tilized_factory.cpp
        moreh_getitem/moreh_getitem.cpp
        moreh_group_norm_backward/device/gamma_beta_grad/moreh_group_norm_backward_gamma_beta_grad_device_operation.cpp
        moreh_group_norm_backward/device/gamma_beta_grad/moreh_group_norm_backward_gamma_beta_grad_factory.cpp
        moreh_group_norm_backward/device/input_grad/moreh_group_norm_backward_input_grad_device_operation.cpp
        moreh_group_norm_backward/device/input_grad/moreh_group_norm_backward_input_grad_factory.cpp
        moreh_group_norm_backward/moreh_group_norm_backward.cpp
        moreh_group_norm/device/moreh_group_norm_device_operation.cpp
        moreh_group_norm/device/moreh_group_norm_program_factory.cpp
        moreh_group_norm/moreh_group_norm.cpp
        moreh_layer_norm_backward/device/moreh_layer_norm_backward_gamma_beta_grad_device_operation.cpp
        moreh_layer_norm_backward/device/moreh_layer_norm_backward_gamma_beta_grad_program_factory.cpp
        moreh_layer_norm_backward/device/moreh_layer_norm_backward_input_grad_device_operation.cpp
        moreh_layer_norm_backward/device/moreh_layer_norm_backward_input_grad_program_factory.cpp
        moreh_layer_norm_backward/moreh_layer_norm_backward.cpp
        moreh_layer_norm/device/moreh_layer_norm_device_operation.cpp
        moreh_layer_norm/device/moreh_layer_norm_program_factory.cpp
        moreh_layer_norm/moreh_layer_norm.cpp
        moreh_linear_backward/device/moreh_linear_backward_device_operation.cpp
        moreh_linear_backward/device/moreh_linear_backward_multi_core_program_factory.cpp
        moreh_linear_backward/device/moreh_linear_backward_single_core_program_factory.cpp
        moreh_linear_backward/moreh_linear_backward.cpp
        moreh_linear/moreh_linear.cpp
        moreh_matmul_backward/moreh_matmul_backward.cpp
        moreh_matmul/device/moreh_matmul_device_operation.cpp
        moreh_matmul/device/moreh_matmul_program_factory.cpp
        moreh_matmul/moreh_matmul.cpp
        moreh_mean_backward/device/moreh_mean_backward_device_operation.cpp
        moreh_mean_backward/device/moreh_mean_backward_program_factory.cpp
        moreh_mean_backward/moreh_mean_backward.cpp
        moreh_mean/device/moreh_mean_device_operation.cpp
        moreh_mean/device/moreh_mean_h_program_factory.cpp
        moreh_mean/device/moreh_mean_nc_program_factory.cpp
        moreh_mean/device/moreh_mean_w_program_factory.cpp
        moreh_mean/moreh_mean.cpp
        moreh_nll_loss_backward/device/moreh_nll_loss_backward_device_operation.cpp
        moreh_nll_loss_backward/device/moreh_nll_loss_backward_program_factory.cpp
        moreh_nll_loss_backward/moreh_nll_loss_backward.cpp
        moreh_nll_loss_unreduced_backward/device/moreh_nll_loss_unreduced_backward_device_operation.cpp
        moreh_nll_loss_unreduced_backward/device/moreh_nll_loss_unreduced_backward_program_factory.cpp
        moreh_nll_loss_unreduced_backward/moreh_nll_loss_unreduced_backward.cpp
        moreh_nll_loss/moreh_nll_loss_step1/device/moreh_nll_loss_step1_device_operation.cpp
        moreh_nll_loss/moreh_nll_loss_step1/device/moreh_nll_loss_step1_program_factory.cpp
        moreh_nll_loss/moreh_nll_loss_step2/device/moreh_nll_loss_step2_device_operation.cpp
        moreh_nll_loss/moreh_nll_loss_step2/device/moreh_nll_loss_step2_program_factory.cpp
        moreh_nll_loss/moreh_nll_loss.cpp
        moreh_norm_backward/device/moreh_norm_backward_device_operation.cpp
        moreh_norm_backward/device/moreh_norm_backward_program_factory.cpp
        moreh_norm_backward/moreh_norm_backward.cpp
        moreh_norm/device/moreh_norm_device_operation.cpp
        moreh_norm/device/ord_other/moreh_norm_program_factory_h_other.cpp
        moreh_norm/device/ord_other/moreh_norm_program_factory_nc_other.cpp
        moreh_norm/device/ord_other/moreh_norm_program_factory_w_other.cpp
        moreh_norm/moreh_norm.cpp
        moreh_sgd/device/moreh_sgd_device_operation.cpp
        moreh_sgd/device/moreh_sgd_program_factory.cpp
        moreh_sgd/moreh_sgd.cpp
        moreh_softmax_backward/device/moreh_softmax_backward_device_operation.cpp
        moreh_softmax_backward/device/softmax_backward_c_large/softmax_backward_c_large.cpp
        moreh_softmax_backward/device/softmax_backward_h_large/softmax_backward_h_large.cpp
        moreh_softmax_backward/device/softmax_backward_h_small/softmax_backward_h_small.cpp
        moreh_softmax_backward/device/softmax_backward_w_large/softmax_backward_w_large.cpp
        moreh_softmax_backward/device/softmax_backward_w_small/softmax_backward_w_small.cpp
        moreh_softmax_backward/moreh_softmax_backward.cpp
        moreh_softmax/device/moreh_softmax_device_operation.cpp
        moreh_softmax/device/moreh_softmax_device_operation.cpp
        moreh_softmax/device/softmax_c_large/softmax_c_large.cpp
        moreh_softmax/device/softmax_c_large/softmax_c_large.cpp
        moreh_softmax/device/softmax_h_large/softmax_h_large.cpp
        moreh_softmax/device/softmax_h_large/softmax_h_large.cpp
        moreh_softmax/device/softmax_h_small/softmax_h_small.cpp
        moreh_softmax/device/softmax_h_small/softmax_h_small.cpp
        moreh_softmax/device/softmax_w_large/softmax_w_large.cpp
        moreh_softmax/device/softmax_w_large/softmax_w_large.cpp
        moreh_softmax/device/softmax_w_small/softmax_w_small.cpp
        moreh_softmax/device/softmax_w_small/softmax_w_small.cpp
        moreh_softmax/moreh_softmax.cpp
        moreh_softmax/moreh_softmax.cpp
        moreh_sum_backward/device/moreh_sum_backward_device_operation.cpp
        moreh_sum_backward/device/moreh_sum_backward_program_factory.cpp
        moreh_sum_backward/moreh_sum_backward.cpp
        moreh_sum/device/moreh_int_sum_h_program_factory.cpp
        moreh_sum/device/moreh_int_sum_nc_program_factory.cpp
        moreh_sum/device/moreh_int_sum_w_program_factory.cpp
        moreh_sum/device/moreh_sum_device_operation.cpp
        moreh_sum/device/moreh_sum_h_program_factory.cpp
        moreh_sum/device/moreh_sum_nc_program_factory.cpp
        moreh_sum/device/moreh_sum_w_program_factory.cpp
        moreh_sum/moreh_sum.cpp
        moreh_fold/fold.cpp
        moreh_fold/device/fold_device_operation.cpp
        moreh_fold/device/fold_program_factory_rm.cpp
        moreh_helper_functions.cpp
        moreh_clip_grad_norm/moreh_clip_grad_norm_step1/device/moreh_clip_grad_norm_step1_device_operation.cpp
        moreh_clip_grad_norm/moreh_clip_grad_norm_step1/device/moreh_clip_grad_norm_step1_program_factory.cpp
        moreh_clip_grad_norm/moreh_clip_grad_norm_step2/device/moreh_clip_grad_norm_step2_device_operation.cpp
        moreh_clip_grad_norm/moreh_clip_grad_norm_step2/device/moreh_clip_grad_norm_step2_program_factory.cpp
        moreh_clip_grad_norm/moreh_clip_grad_norm_step3/device/moreh_clip_grad_norm_step3_device_operation.cpp
        moreh_clip_grad_norm/moreh_clip_grad_norm_step3/device/moreh_clip_grad_norm_step3_program_factory.cpp
        moreh_clip_grad_norm/moreh_clip_grad_norm.cpp
)

target_include_directories(ttnn_op_moreh PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_moreh
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_moreh
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/moreh
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_moreh LIBRARY COMPONENT tar)
