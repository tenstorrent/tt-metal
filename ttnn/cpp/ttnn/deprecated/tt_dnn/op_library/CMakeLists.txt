# We do not use GLOB here since build system won't be able to pick up changes to the file list generated
set(TT_DNN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_clip_grad_norm/moreh_clip_grad_norm_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_clip_grad_norm/moreh_clip_grad_norm_step1/moreh_clip_grad_norm_step1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_clip_grad_norm/moreh_clip_grad_norm_step2/moreh_clip_grad_norm_step2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moreh_clip_grad_norm/moreh_clip_grad_norm_step3/moreh_clip_grad_norm_step3.cpp
    CACHE INTERNAL
    "tt_dnn sources to reuse in ttnn build"
)
