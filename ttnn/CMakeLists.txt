####################################################################################################
# Build TTNN and TTNNCPP
####################################################################################################

##################################################
# Set options
##################################################
if(ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "Enabling code coverage flags for all ttnn targets")
    add_compile_options(--coverage)
    add_link_options(--coverage)
endif()

set(LIB_TYPE OBJECT)
if(ENABLE_TTNN_SHARED_SUBLIBS)
    set(LIB_TYPE SHARED)
endif()

add_library(ttnn_core ${LIB_TYPE})
add_library(TT::NN::Core ALIAS ttnn_core)

target_compile_definitions(ttnn_core PUBLIC "$<$<CXX_COMPILER_ID:GNU>:DISABLE_NAMESPACE_STATIC_ASSERT>")

GENERATE_FBS_HEADER(${CMAKE_CURRENT_SOURCE_DIR}/core/tensor/flatbuffer/tensor_types.fbs)
target_sources(
    ttnn_core
    PRIVATE
        ${FBS_GENERATED_HEADER_FILE}
        core/async_runtime.cpp
        core/cluster.cpp
        core/config.cpp
        core/core.cpp
        core/device.cpp
        core/distributed/api.cpp
        core/distributed/distributed_tensor.cpp
        core/distributed/distributed_tensor_config.cpp
        core/distributed/distributed_tensor_config.cpp
        core/events.cpp
        core/global_circular_buffer.cpp
        core/global_semaphore.cpp
        core/graph/graph_argument_serializer.cpp
        core/graph/graph_processor.cpp
        core/graph/graph_trace_utils.cpp
        core/old_infra_device_operation.cpp
        core/run_operation.cpp
        core/tensor/flatbuffer/tensor_types_from_flatbuffer.cpp
        core/tensor/flatbuffer/tensor_types_to_flatbuffer.cpp
        core/tensor/host_buffer/functions.cpp
        core/tensor/layout/alignment.cpp
        core/tensor/layout/page_config.cpp
        core/tensor/layout/tensor_layout.cpp
        core/tensor/serialization.cpp
        core/tensor/storage.cpp
        core/tensor/tensor.cpp
        core/tensor/tensor_attributes.cpp
        core/tensor/tensor_impl.cpp
        core/tensor/tensor_ops.cpp
        core/tensor/tensor_spec.cpp
        core/tensor/tensor_utils.cpp
        core/tensor/types.cpp
        core/tensor/xtensor/partition.cpp
)
target_include_directories(
    ttnn_core
    PUBLIC
        api
    PRIVATE
        api/ttnn
        core
        ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers
        cpp # FIXME: Depend on the Ops needed (make the circular obvious)
        . # FIXME: Depend on the Ops needed (make the circular obvious)
)
target_link_libraries(
    ttnn_core
    PRIVATE
        TT::Metalium
        xtensor
        FlatBuffers::FlatBuffers
        Boost::algorithm
)
install(TARGETS ttnn_core LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

##################################################
# Define a method build_library to define all the options and properties for a library
##################################################
function(build_library LIBRARY_NAME ALIAS)
    set(options
        SHARED
        STATIC
    )
    set(oneValueArgs) # No single-value args
    set(multiValueArgs
        SOURCES
        PUBLIC_LINK_DIRECTORIES
        INCLUDE_DIRECTORIES
        TARGET_LINK_LIBRARIES
    ) # All are multi-value

    cmake_parse_arguments(BUILD_LIB "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Determine library type
    if(BUILD_LIB_SHARED)
        set(LIBRARY_TYPE "SHARED")
    elseif(BUILD_LIB_STATIC)
        set(LIBRARY_TYPE "STATIC")
    else()
        message(FATAL_ERROR "You must specify SHARED or STATIC for library type")
    endif()

    # Assign values
    set(SOURCES "${BUILD_LIB_SOURCES}")
    set(TARGET_LINK_LIBRARIES "${BUILD_LIB_TARGET_LINK_LIBRARIES}")
    set(LINK_DIRECTORIES "${BUILD_LIB_LINK_DIRECTORIES}")
    set(INCLUDE_DIRS "${BUILD_LIB_INCLUDE_DIRECTORIES}")

    ##################################################
    # Declare and define the library
    ##################################################
    #todo: Ideally declaring the library should be the first thing to do, sadly making it shared requires to add sources
    # Assign values to variables for clarity

    add_library(
        ${LIBRARY_NAME}
        ${LIBRARY_TYPE}
        ${SOURCES}
    )

    add_library(Metalium::${ALIAS} ALIAS ${LIBRARY_NAME})
    target_include_directories(${LIBRARY_NAME} PUBLIC "${INCLUDE_DIRS}")
    target_link_directories(${LIBRARY_NAME} PUBLIC ${LINK_DIRECTORIES})

    if("${build_library}" STREQUAL "ttnncpp")
        target_precompile_headers(${LIBRARY_NAME} REUSE_FROM ${PRECOMPILED_HEADER_TARGET})
    endif()

    target_link_libraries(
        ${LIBRARY_NAME}
        PUBLIC
            ${TARGET_LINK_LIBRARIES}
            Boost::algorithm
    )

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_definitions(${LIBRARY_NAME} PUBLIC DISABLE_NAMESPACE_STATIC_ASSERT)
    endif()

    TT_ENABLE_UNITY_BUILD(${LIBRARY_NAME})

    #We move the library binaries to a different path rather than PROJECT_BINARY_DIR
    #in the Python wheel
    set(TTNN_INSTALL_RPATH
        "${PROJECT_BINARY_DIR}/lib"
        "$ORIGIN/build/lib"
        "$ORIGIN"
    )

    if(BUILD_LIB_SHARED)
        #Make sure library built is _ttnn.so / _ttnnpycpp.so and that it can find all it's linked libraries
        #ttnn breaks if - fvisibility = hidden, so CXX_VISIBILITY_PRESET set to default
        set_target_properties(
            ${LIBRARY_NAME}
            PROPERTIES
                OUTPUT_NAME
                    "_${LIBRARY_NAME}"
                PREFIX
                    ""
                SUFFIX
                    ".so"
                BUILD_RPATH
                    "${PROJECT_BINARY_DIR}/tt_metal;${PROJECT_BINARY_DIR}/ttnn"
                INSTALL_RPATH
                    "${TTNN_INSTALL_RPATH}"
                CXX_VISIBILITY_PRESET
                    "default"
                ADDITIONAL_CLEAN_FILES
                    "${PROJECT_SOURCE_DIR}/ttnn/ttnn/_'${LIBRARY_NAME}'.so;${PROJECT_SOURCE_DIR}/ttnn/'${LIBRARY_NAME}'.egg-info"
        )
    endif()
endfunction()

##################################################
# Define auxiliar functions
##################################################
function(use_precompiled_header TARGET)
    if("${PRECOMPILED_HEADER_TARGET}" STREQUAL "")
        set(PRECOMPILED_HEADER_TARGET "${TARGET}" PARENT_SCOPE)
        target_precompile_headers(${TARGET} PUBLIC ${TTNN_PRECOMPILED_HEADERS})
    else()
        target_precompile_headers(${TARGET} REUSE_FROM ${PRECOMPILED_HEADER_TARGET})
    endif()
endfunction()

function(add_ttnn_sublibrary SUBLIBRARY_NAME)
    add_library(
        ${SUBLIBRARY_NAME}
        ${LIB_TYPE}
        ${ARGN}
    )
    install(TARGETS ${SUBLIBRARY_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

    TT_ENABLE_UNITY_BUILD(${SUBLIBRARY_NAME})
    target_include_directories(${SUBLIBRARY_NAME} PUBLIC ${TTNN_PUBLIC_INCLUDE_DIRS})
    target_link_libraries(${SUBLIBRARY_NAME} PUBLIC ${TTNN_PUBLIC_LINK_LIBRARIES})
    target_link_directories(${SUBLIBRARY_NAME} PUBLIC ${TTNN_PUBLIC_LINK_DIRS})
    use_precompiled_header(${SUBLIBRARY_NAME})
    set(PRECOMPILED_HEADER_TARGET "${PRECOMPILED_HEADER_TARGET}" PARENT_SCOPE)

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_definitions(${SUBLIBRARY_NAME} PRIVATE DISABLE_NAMESPACE_STATIC_ASSERT)
    endif()
endfunction()

##################################################
# Collect all the files and folders that TTNNCPP will use
##################################################
set(TTNN_PUBLIC_INCLUDE_DIRS
    # ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/ttnn
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
)

set(TTNN_BASE_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/compute_throttle_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/sharding_utilities.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/trace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/sharding_addrgen_helper.cpp
)

set(TTNN_OP_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/generic/generic_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/generic/device/generic_op_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/padded_slice/device/padded_slice_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/padded_slice/device/padded_slice_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/padded_slice/padded_slice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/slice_write/device/slice_write_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/slice_write/device/slice_write_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/slice_write/slice_write.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/gather/gather.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/gather/device/gather_device_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/gather/device/gather_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/gather/tosa/gather_tosa.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/rms_allgather_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/rms_allgather/rms_allgather.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/multi_core/rms_allgather_pf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_pre_all_gather.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_post_all_gather.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/softmax/softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/softmax/device/multi_core/softmax_op_multi_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/softmax/device/softmax_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/copy/typecast/device/typecast_device_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/copy/typecast/device/typecast_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/copy/typecast/device/typecast_sharded_program_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/copy/typecast/typecast.cpp
)

set(TTNN_SRC)
list(
    APPEND
    TTNN_SRC
    ${TTNN_BASE_SRCS}
    ${TTNN_OP_SRCS}
    ${TENSOR_SRC}
)

set(TTNN_SUBLIBRARIES ttnn/operations/experimental/experimental_${PY_BINDING})

set(TTNN_PUBLIC_LINK_LIBRARIES
    metal_common_libs
    Metalium::Metal
    TT::STL
    xtensor
    xtensor-blas
    xtl
    TT::NN::Core
)

set(PRECOMPILED_HEADER_TARGET "TT::CommonPCH")

##################################################
# Collect all the files and folders that TTNN will use
##################################################
set(TTNN_SRC_PYBIND
    ${CMAKE_CURRENT_SOURCE_DIR}/core/distributed/distributed_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/graph/graph_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/bernoulli/bernoulli_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/conv/conv1d/conv1d_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/conv/conv2d/conv2d_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/conv/conv_transpose2d/conv_transpose2d_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/bcast/bcast_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/clone/clone_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/copy/copy_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/chunk/chunk_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/expand/expand_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/fill_pad/fill_pad_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/fill_rm/fill_rm_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/fold/fold_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/indexed_fill/indexed_fill_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/move/move_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/non_zero_indices/non_zero_indices_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/pad/pad_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/permute/permute_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/repeat/repeat_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/reshape_view/reshape_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/reshape_on_device/reshape_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/roll/roll_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/stack/stack_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/sharded/reshard/reshard_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/interleaved_to_sharded_partial_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial/sharded_to_interleaved_partial_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/squeeze/squeeze_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/transpose/transpose_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/unsqueeze/unsqueeze_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/data_movement/view/view_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/binary/binary_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/eltwise/quantization/quantization_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/embedding_backward/embedding_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/examples/example_multiple_return/example_multiple_return_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/examples/examples_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/conv3d/conv3d_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/cnn/convert_to_chw/convert_to_chw_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/copy/typecast/typecast_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/dropout/dropout_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/bcast_to/bcast_to_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/experimental_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/matmul/attn_matmul/attn_matmul_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/group_attn_matmul_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/paged_cache/paged_cache_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/plusone/plusone_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/reduction/cumprod/cumprod_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/fast_reduce_nc_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/fast_reduce_nc_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/reduction/cumsum/cumsum_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/reshape/view_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/slice_write/slice_write_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/padded_slice/padded_slice_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ssm/hc_sum_reduce/hc_sum_reduce_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ssm/prefix_scan/prefix_scan_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul/repeat_and_interleave_eltwise_mul_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/create_qkv_heads/create_qkv_heads_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/create_qkv_heads_from_separate_tensors/create_qkv_heads_from_separate_tensors_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/nlp_concat_heads_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/nlp_concat_heads_decode_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/nlp_create_qkv_heads_decode_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_falcon7b/nlp_create_qkv_heads_falcon7b_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_segformer/nlp_create_qkv_heads_segformer_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_vit/nlp_create_qkv_heads_vit_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/nlp_kv_cache_load_slice/nlp_kv_cache_load_slice_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotary_embedding/rotary_embedding_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/rotary_embedding_llama_fused_qk_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/rotate_half/rotate_half_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/gelu_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/full/full_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/full_like/full_like_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/generic/generic_op_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/index_fill/index_fill_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/kv_cache/kv_cache_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/loss/loss_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/matmul/matmul_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_abs_pow/moreh_abs_pow_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_adam/moreh_adam_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_adamw/moreh_adamw_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_arange/moreh_arange_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_bmm/moreh_bmm_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_bmm_backward/moreh_bmm_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_cumsum/moreh_cumsum_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_dot/moreh_dot_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_dot/moreh_dot_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_dot_backward/moreh_dot_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_fold/fold_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_getitem/moreh_getitem_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_group_norm/moreh_group_norm_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_group_norm_backward/moreh_group_norm_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_layer_norm/moreh_layer_norm_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/moreh_layer_norm_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_linear/moreh_linear_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_linear_backward/moreh_linear_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_matmul/moreh_matmul_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_matmul_backward/moreh_matmul_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_mean/moreh_mean_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_mean_backward/moreh_mean_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_nll_loss_backward/moreh_nll_loss_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_nll_loss_unreduced_backward/moreh_nll_loss_unreduced_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_norm/moreh_norm_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_norm_backward/moreh_norm_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_sgd/moreh_sgd_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_softmax/moreh_softmax_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_softmax/moreh_softmax_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_softmax_backward/moreh_softmax_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_sum/moreh_sum_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/moreh/moreh_sum_backward/moreh_sum_backward_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/batch_norm/batch_norm_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/groupnorm/groupnorm_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/layernorm/layernorm_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/layernorm_distributed/layernorm_distributed_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/rmsnorm/rmsnorm_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/rmsnorm/rmsnorm_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_distributed_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/normalization/softmax/softmax_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/downsample/downsample_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/generic/generic_pools_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/upsample/upsample_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/prefetcher/prefetcher/dram_prefetcher_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/prefetcher/prefetcher_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/reduction_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/reduction/sampling/sampling_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/sliding_window/sliding_window_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/transformer/attention_softmax/attention_softmax_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/transformer/concatenate_heads/concatenate_heads_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/transformer/sdpa/sdpa_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/transformer/split_query_key_value_and_split_heads/split_query_key_value_and_split_heads_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/transformer/transformer_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/uniform/uniform_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/all_reduce_create_qkv_heads_${PY_BINDING}.cpp
)

set(CCL_EXPERIMENTAL_TTNN_SRCS_PYBIND
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/all_gather_async/all_gather_async_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/all_gather_concat_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/all_gather_matmul/all_gather_matmul_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/all_reduce/all_reduce_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/ccl_experimental_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/reduce_scatter_async/reduce_scatter_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/rms_allgather/rms_allgather_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/reduction/sort/sort_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/gather/gather_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/gather/tosa/gather_tosa_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/llama_reduce_scatter_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/llama_reduce_scatter_create_heads_${PY_BINDING}.cpp
)

set(CCL_TTNN_SRCS_PYBIND
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/ccl_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/all_gather/all_gather_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/reduce_scatter/reduce_scatter_${PY_BINDING}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/ccl/barrier/barrier_${PY_BINDING}.cpp
)

#TODO : should be using pybind11_add_module, but right now it introduces many build problems
#pybinds will always be built as a shared library
list(
    APPEND
    TTNN_SRC_PYBIND
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/__init__.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/types.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/events.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/global_circular_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/global_semaphore.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/cluster.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/fabric.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/profiler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/program_descriptors.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn-${PY_BINDING}/pytensor.cpp
)

set(TTNN_PUBLIC_INCLUDE_DIRS_PYBIND ${Python3_INCLUDE_DIRS})

##################################################
# Handle sublibraries
##################################################
foreach(SUBLIBRARY_PATH ${TTNN_SUBLIBRARIES})
    set(SUBLIBRARY_SRCS ${TTNN_SRC})
    list(FILTER SUBLIBRARY_SRCS INCLUDE REGEX "${SUBLIBRARY_PATH}")
    list(FILTER TTNN_SRC EXCLUDE REGEX "${SUBLIBRARY_PATH}")

    if(SUBLIBRARY_SRCS)
        string(REPLACE "/" "_" SUBLIBRARY_NAME ${SUBLIBRARY_PATH})

        add_ttnn_sublibrary(${SUBLIBRARY_NAME} ${SUBLIBRARY_SRCS})
        list(APPEND ALL_TTNN_SUBLIBRARIES ${SUBLIBRARY_NAME})
    endif()
endforeach(
    SUBLIBRARY_PATH
    ${TTNN_SUBLIBRARIES}
)

##################################################
# Prepare the variables to define the library to be built
##################################################
set(SOURCES_PYBIND
    ${SRC_PYBIND}
    ${TTNN_SRC_PYBIND}
    ${CCL_TTNN_SRCS_PYBIND}
    ${CCL_EXPERIMENTAL_TTNN_SRCS_PYBIND}
    ${Python3_LIBRARY_DIRS}
)

set(SOURCES_CPP
    ${TTNN_SRC}
    ${QUEUE_SRCS}
)

set(TARGET_LINK_LIBRARIES_PYBIND
    ttnncpp
    ${Python3_LIBRARIES}
    pybind11::module
)

set(TARGET_LINK_LIBRARIES_CPP
    ${TTNN_PUBLIC_LINK_LIBRARIES}
    ${ALL_TTNN_SUBLIBRARIES}
)

list(APPEND TTNN_PUBLIC_INCLUDE_DIRS_PYBIND ${TTNN_PUBLIC_INCLUDE_DIRS})

##################################################
# Build the libraries!
##################################################

set(TTNN_CPP_BUILD_TYPE SHARED)
if(WITH_PYTHON_BINDINGS)
    #ttnncpp will be static when building python bindings
    set(TTNN_CPP_BUILD_TYPE STATIC)
endif()

#there are two ways to build TTNN
# 1) Without Python bindings
#        ttnncpp will be a dynamic library that can be linked to any project
# 2) With python bindings
#        ttnn will be a dynamic library with ttnncpp statically linked to it
# so if a user wants to consume ttnncpp for a project but also is interested on having
# ttnn as well, it should build it first without python bindings and then with them.
# Most of the use cases will link to ttnn without caring a lot about the backend, but
# for those who need this second use case, now it is officially supported

build_library(
    "ttnncpp"                        #LIBRARY_NAME
    "TTNNCPP"                        #ALIAS
    ${TTNN_CPP_BUILD_TYPE}
    SOURCES
        "${SOURCES_CPP}"
    PUBLIC_LINK_DIRECTORIES
        "${TTNN_PUBLIC_LINK_DIRS}"
    INCLUDE_DIRECTORIES
        "${TTNN_PUBLIC_INCLUDE_DIRS}"
    TARGET_LINK_LIBRARIES
        "${TARGET_LINK_LIBRARIES_CPP}"
        TT::NN::Ops::Bernoulli
        TT::NN::Ops::CCL
        TT::NN::Ops::Conv
        TT::NN::Ops::Core
        TT::NN::Ops::DataMovement
        TT::NN::Ops::Eltwise::Binary
        TT::NN::Ops::Eltwise::Binary::Backward
        TT::NN::Ops::Eltwise::Binary::NG
        TT::NN::Ops::Eltwise::Complex
        TT::NN::Ops::Eltwise::Complex::Binary
        TT::NN::Ops::Eltwise::Complex::Unary
        TT::NN::Ops::Eltwise::Complex::Unary::Backward
        TT::NN::Ops::Eltwise::Quantization
        TT::NN::Ops::Eltwise::Ternary
        TT::NN::Ops::Eltwise::Ternary::Backward
        TT::NN::Ops::Eltwise::Unary
        TT::NN::Ops::Eltwise::Unary::Backward
        TT::NN::Ops::Embedding
        TT::NN::Ops::Embedding::Backward
        TT::NN::Ops::Examples
        TT::NN::Ops::Experimental::AutoFormat
        TT::NN::Ops::Experimental::BcastTo
        TT::NN::Ops::Experimental::CCL
        TT::NN::Ops::Experimental::CNN
        TT::NN::Ops::Experimental::Conv3d
        TT::NN::Ops::Experimental::Copy
        TT::NN::Ops::Experimental::Dropout
        TT::NN::Ops::Experimental::Matmul
        TT::NN::Ops::Experimental::PagedCache
        TT::NN::Ops::Experimental::PlusOne
        TT::NN::Ops::Experimental::Reduction
        TT::NN::Ops::Experimental::Reshape
        TT::NN::Ops::Experimental::SSM
        TT::NN::Ops::Experimental::Transformer
        TT::NN::Ops::Experimental::UnaryBackward
        TT::NN::Ops::Full
        TT::NN::Ops::FullLike
        TT::NN::Ops::IndexFill
        TT::NN::Ops::KvCache
        TT::NN::Ops::Loss
        TT::NN::Ops::Matmul
        TT::NN::Ops::Moreh
        TT::NN::Ops::Prefetcher
        TT::NN::Ops::Reduction
        TT::NN::Ops::SlidingWindow
        TT::NN::Ops::Transformer
        TT::NN::Ops::Uniform
        TT::NN::Ops::Pool
        TT::NN::Ops::Normalization
)

if(WITH_PYTHON_BINDINGS)
    build_library(
        "ttnn"                           #LIBRARY_NAME
        "TTNN"	                         #ALIAS
        SHARED
        SOURCES
            "${SOURCES_PYBIND}"
        PUBLIC_LINK_DIRECTORIES
            "${TTNN_PUBLIC_LINK_DIRS_PYBIND}"
        INCLUDE_DIRECTORIES
            "${TTNN_PUBLIC_INCLUDE_DIRS_PYBIND}"
        TARGET_LINK_LIBRARIES
            "${TARGET_LINK_LIBRARIES_PYBIND}"
    )
endif()

if(WITH_PYTHON_BINDINGS)
    # Install .so into src files for pybinds implementation
    install(
        TARGETS
            ttnn
        ARCHIVE
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT tar
    )

    install(
        TARGETS
            ttnn
            DESTINATION
            ${PROJECT_SOURCE_DIR}/ttnn/ttnn
            COMPONENT
            tt_pybinds
    )
else()
    #when we don't build python bindings, we generate a dynamic library ttnncpp
    install(
        TARGETS
            ttnncpp
        ARCHIVE
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT tar
    )
endif()

set(FixmeOpIncDirs
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
)
add_subdirectory(cpp/ttnn/operations/bernoulli)
add_subdirectory(cpp/ttnn/operations/ccl)
add_subdirectory(cpp/ttnn/operations/conv)
add_subdirectory(cpp/ttnn/operations/core)
add_subdirectory(cpp/ttnn/operations/data_movement)
add_subdirectory(cpp/ttnn/operations/eltwise/binary)
add_subdirectory(cpp/ttnn/operations/eltwise/binary_backward)
add_subdirectory(cpp/ttnn/operations/eltwise/binary_ng)
add_subdirectory(cpp/ttnn/operations/eltwise/complex)
add_subdirectory(cpp/ttnn/operations/eltwise/complex_binary)
add_subdirectory(cpp/ttnn/operations/eltwise/complex_unary)
add_subdirectory(cpp/ttnn/operations/eltwise/complex_unary_backward)
add_subdirectory(cpp/ttnn/operations/eltwise/quantization)
add_subdirectory(cpp/ttnn/operations/eltwise/ternary)
add_subdirectory(cpp/ttnn/operations/eltwise/ternary_backward)
add_subdirectory(cpp/ttnn/operations/eltwise/unary)
add_subdirectory(cpp/ttnn/operations/eltwise/unary_backward)
add_subdirectory(cpp/ttnn/operations/embedding)
add_subdirectory(cpp/ttnn/operations/embedding_backward)
add_subdirectory(cpp/ttnn/operations/examples)
add_subdirectory(cpp/ttnn/operations/experimental/auto_format)
add_subdirectory(cpp/ttnn/operations/experimental/bcast_to)
add_subdirectory(cpp/ttnn/operations/experimental/ccl)
add_subdirectory(cpp/ttnn/operations/experimental/cnn)
add_subdirectory(cpp/ttnn/operations/experimental/conv3d)
add_subdirectory(cpp/ttnn/operations/experimental/copy)
add_subdirectory(cpp/ttnn/operations/experimental/dropout)
add_subdirectory(cpp/ttnn/operations/experimental/matmul)
add_subdirectory(cpp/ttnn/operations/experimental/paged_cache)
add_subdirectory(cpp/ttnn/operations/experimental/plusone)
add_subdirectory(cpp/ttnn/operations/experimental/reduction)
add_subdirectory(cpp/ttnn/operations/experimental/reshape)
add_subdirectory(cpp/ttnn/operations/experimental/ssm)
add_subdirectory(cpp/ttnn/operations/experimental/transformer)
add_subdirectory(cpp/ttnn/operations/experimental/unary_backward)
add_subdirectory(cpp/ttnn/operations/full)
add_subdirectory(cpp/ttnn/operations/full_like)
add_subdirectory(cpp/ttnn/operations/index_fill)
add_subdirectory(cpp/ttnn/operations/kv_cache)
add_subdirectory(cpp/ttnn/operations/loss)
add_subdirectory(cpp/ttnn/operations/matmul)
add_subdirectory(cpp/ttnn/operations/moreh)
add_subdirectory(cpp/ttnn/operations/prefetcher)
add_subdirectory(cpp/ttnn/operations/reduction)
add_subdirectory(cpp/ttnn/operations/sliding_window)
add_subdirectory(cpp/ttnn/operations/transformer)
add_subdirectory(cpp/ttnn/operations/uniform)
add_subdirectory(cpp/ttnn/operations/pool)
add_subdirectory(cpp/ttnn/operations/normalization)
