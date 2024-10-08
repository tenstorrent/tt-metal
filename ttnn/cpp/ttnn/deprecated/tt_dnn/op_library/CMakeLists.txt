
# We do not use GLOB here since build system won't be able to pick up changes to the file list generated
set(TT_DNN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/sharding_utilities.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_helper_functions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_clip_grad_norm/moreh_clip_grad_norm_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_clip_grad_norm/moreh_clip_grad_norm_step1/moreh_clip_grad_norm_step1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_clip_grad_norm/moreh_clip_grad_norm_step2/moreh_clip_grad_norm_step2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_clip_grad_norm/moreh_clip_grad_norm_step3/moreh_clip_grad_norm_step3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_sum/moreh_sum_h_impl/moreh_sum_h_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_sum/moreh_sum_h_impl/moreh_int_sum_h_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_sum/moreh_sum_w_impl/moreh_sum_w_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_sum/moreh_sum_w_impl/moreh_int_sum_w_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_sum/moreh_sum_nc_impl/moreh_sum_nc_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_sum/moreh_sum_nc_impl/moreh_int_sum_nc_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_sum/moreh_sum_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_sum/moreh_sum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_sum_backward/moreh_sum_backward_impl/moreh_sum_backward_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_sum_backward/moreh_sum_backward_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_bmm/moreh_bmm_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_bmm_backward/moreh_bmm_backward_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_matmul/multi_core/moreh_matmul_op_multi_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_matmul/moreh_matmul_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_matmul_backward/moreh_matmul_backward_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_dot/single_core/moreh_dot_op_single_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_dot/moreh_dot_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_dot_backward/single_core/moreh_dot_backward_op_single_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_dot_backward/moreh_dot_backward_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_layernorm/moreh_layernorm_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_layernorm_backward/moreh_layernorm_backward_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_layernorm_backward/input_grad/moreh_layernorm_backward_input_grad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_layernorm_backward/gamma_beta_grad/moreh_layernorm_backward_gamma_beta_grad.cpp

    CACHE INTERNAL "tt_dnn sources to reuse in ttnn build"
)