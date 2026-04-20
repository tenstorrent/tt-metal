####################################################################################################
# Build TTNN and TTNNCPP
####################################################################################################

include(sources.cmake)

add_compile_options("$<$<CXX_COMPILER_ID:Clang>:-Wunused-parameter>")

# FIXME: Many lambda functions in ttnn capture variables and don't use them.
add_compile_options("$<$<CXX_COMPILER_ID:Clang>:-Wno-unused-lambda-capture>")

# Suppress narrowing warnings for ttnn (being cleaned up in tt_metal first)
add_compile_options("$<$<CXX_COMPILER_ID:Clang>:-Wno-c++11-narrowing>")
add_compile_options("$<$<CXX_COMPILER_ID:GNU>:-Wno-narrowing>")

##################################################
# TTNN-specific precompiled header
# Extends common PCH with frequently used TTNN headers to speed up compilation
##################################################

add_library(ttnn_pch STATIC)
add_library(TTNN::PCH ALIAS ttnn_pch)
file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ttnn_pch.cpp CONTENT "/*dummy ttnn pch file*/\n")
target_sources(ttnn_pch PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/ttnn_pch.cpp)

target_precompile_headers(
    ttnn_pch
    PUBLIC
        # Reuse common PCH headers
        <fmt/core.h>
        <fmt/format.h>
        <nlohmann/json.hpp>
        <algorithm>
        <cstddef>
        <cstdint>
        <functional>
        <map>
        <memory>
        <tuple>
        <unordered_map>
        <variant>
        <vector>
        # TTNN-specific headers that are included frequently
        <ttnn/device_operation.hpp>
        <ttnn/operation.hpp>
)

target_link_libraries(
    ttnn_pch
    PUBLIC
        TT::Metalium
        nlohmann_json::nlohmann_json
        fmt::fmt-header-only
)

target_include_directories(ttnn_pch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/api)
if(COMMAND tt_disable_ccache_for_pch)
    tt_disable_ccache_for_pch(ttnn_pch)
endif()

##################################################
# Set options
##################################################

set(LIB_TYPE OBJECT)
if(ENABLE_TTNN_SHARED_SUBLIBS)
    # Building various components as .so reduces the final link of ttnn.so (and will make loading a bit slower).
    # This is important for development iterations in TT-NN.
    # TODO: Measure whether linker improvements (choice of linker + parallelism flags) render this obsolete.
    set(LIB_TYPE SHARED)
endif()

add_library(ttnn_core ${LIB_TYPE})
add_library(TTNN::Core ALIAS ttnn_core)

# Enable unity builds for faster compilation
TT_ENABLE_UNITY_BUILD(ttnn_core)

target_compile_definitions(ttnn_core PUBLIC "$<$<CXX_COMPILER_ID:GNU>:DISABLE_NAMESPACE_STATIC_ASSERT>")
tt_reuse_precompile_headers(ttnn_core TTNN::PCH)

set(TENSOR_FLATBUFFER_SCHEMAS
    ${CMAKE_CURRENT_SOURCE_DIR}/core/tensor/flatbuffer/mesh_shape.fbs
    ${CMAKE_CURRENT_SOURCE_DIR}/core/tensor/flatbuffer/tensor_spec.fbs
    ${CMAKE_CURRENT_SOURCE_DIR}/core/tensor/flatbuffer/tensor_topology.fbs
    ${CMAKE_CURRENT_SOURCE_DIR}/core/tensor/flatbuffer/tensor.fbs
    ${CMAKE_CURRENT_SOURCE_DIR}/core/tensor/flatbuffer/overlapped_tensor.fbs
)

set(TENSOR_FLATBUFFER_GENERATED_HEADERS)
foreach(FBS_FILE ${TENSOR_FLATBUFFER_SCHEMAS})
    GENERATE_FBS_HEADER(${FBS_FILE} TARGET TTNN::Core)
    list(APPEND TENSOR_FLATBUFFER_GENERATED_HEADERS ${FBS_GENERATED_HEADER_FILE})
endforeach()

set_target_properties(
    ttnn_core
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)
target_sources(
    ttnn_core
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
        FILE_SET jit_api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${TTNN_CORE_JIT_API_HEADERS}
    PRIVATE
        ${TENSOR_FLATBUFFER_GENERATED_HEADERS}
        ${TTNN_CORE_SRCS}
)

target_include_directories(
    ttnn_core
    PUBLIC
        api
    PRIVATE
        api/ttnn
        core
        cpp # FIXME: Depend on the Ops needed (make the circular obvious)
)
target_include_directories(ttnn_core SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers)
target_link_libraries(
    ttnn_core
    PUBLIC
        TT::Metalium
    PRIVATE
        xtensor
        FlatBuffers::FlatBuffers
        Boost::algorithm
)

install(
    TARGETS
        ttnn_core
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FILE_SET
    jit_api
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn
        COMPONENT ttnn-runtime
)

##################################################
# Build the libraries!
##################################################

#there are two ways to build TTNN
# 1) Without Python bindings
#        ttnncpp will be a dynamic library that can be linked to any project
# 2) With python bindings
#        ttnn will be a dynamic library with ttnncpp statically linked to it
# so if a user wants to consume ttnncpp for a project but also is interested on having
# ttnn as well, it should build it first without python bindings and then with them.
# Most of the use cases will link to ttnn without caring a lot about the backend, but
# for those who need this second use case, now it is officially supported

add_library(ttnncpp SHARED)
add_library(TTNN::CPP ALIAS ttnncpp)
add_library(TTNN::TTNN ALIAS ttnncpp)
add_library(Metalium::TTNNCPP ALIAS ttnncpp) # backwards compatibility. FIXME: remove

target_compile_definitions(ttnncpp PUBLIC "$<$<CXX_COMPILER_ID:GNU>:DISABLE_NAMESPACE_STATIC_ASSERT>")
tt_reuse_precompile_headers(ttnncpp TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnncpp)

set_target_properties(
    ttnncpp
    PROPERTIES
        EXPORT_NAME
            TTNN
        PREFIX
            "_"
        BUILD_RPATH
            "${PROJECT_BINARY_DIR}/tt_metal;${PROJECT_BINARY_DIR}/ttnn"
        INSTALL_RPATH
            "${PROJECT_BINARY_DIR}/lib;$ORIGIN/build/lib;$ORIGIN" # FIXME: Install RPATH should not have a build path!
        CXX_VISIBILITY_PRESET
            "default" # FIXME: Export the symbols we want, and use hidden visibility instead of default
        ADDITIONAL_CLEAN_FILES
            "${PROJECT_SOURCE_DIR}/ttnn/ttnn/_'${LIBRARY_NAME}'.so;${PROJECT_SOURCE_DIR}/ttnn/'${LIBRARY_NAME}'.egg-info" # FIXME: Confirm this is a relic of the past and can be deleted; No generated files should land in the source tree!
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE ttnn_kernels cpp/ttnn/operations/copy/typecast/device/kernels/*)
target_sources(
    ttnncpp
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/api ${CMAKE_CURRENT_SOURCE_DIR}/cpp
        FILES ${TTNNCPP_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${ttnn_kernels}
    PRIVATE
        ${TTNNCPP_SRCS}
)

target_include_directories(
    ttnncpp
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/api>"
        # FIXME: decide what to do about this dir. Device includes complicate matters.
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp>"
)

target_link_libraries(
    ttnncpp
    PUBLIC
        TT::Metalium
        xtensor
    PRIVATE
        TTNN::Core
        TTNN::Ops::Bernoulli
        TTNN::Ops::CCL
        TTNN::Ops::Conv
        TTNN::Ops::Core
        TTNN::Ops::Creation
        TTNN::Ops::DataMovement
        TTNN::Ops::Eltwise::Binary
        TTNN::Ops::Eltwise::Binary::Backward
        TTNN::Ops::Eltwise::Binary::NG
        TTNN::Ops::Eltwise::Complex
        TTNN::Ops::Eltwise::Complex::Binary
        TTNN::Ops::Eltwise::Complex::Unary
        TTNN::Ops::Eltwise::Complex::Unary::Backward
        TTNN::Ops::Eltwise::Quantization
        TTNN::Ops::Eltwise::Ternary
        TTNN::Ops::Eltwise::Ternary::Backward
        TTNN::Ops::Eltwise::Unary
        TTNN::Ops::Eltwise::UnaryNG
        TTNN::Ops::Eltwise::Unary::Backward
        TTNN::Ops::Embedding
        TTNN::Ops::Embedding::Backward
        TTNN::Ops::Examples
        TTNN::Ops::Experimental::AdaptivePool
        TTNN::Ops::Experimental::BcastTo
        TTNN::Ops::Experimental::CCL
        TTNN::Ops::Experimental::CNN
        TTNN::Ops::Experimental::Conv3d
        TTNN::Ops::Experimental::Copy
        TTNN::Ops::Experimental::Dropout
        TTNN::Ops::Experimental::IsIn
        TTNN::Ops::Experimental::Matmul
        TTNN::Ops::Experimental::PaddedSlice
        TTNN::Ops::Experimental::MinimalMatmul
        TTNN::Ops::Experimental::DeepseekMoEPostCombineTilize
        TTNN::Ops::Experimental::DeepSeekPrefill::PostCombineReduce
        TTNN::Ops::Experimental::DeepSeek::MoE::MoEGateMM
        TTNN::Ops::Experimental::TopkRouterGpt
        TTNN::Ops::Experimental::Deepseek::MLA::Matmul_WO
        TTNN::Ops::Experimental::MoEGPT
        TTNN::Ops::Experimental::DeepSeekPrefill::Dispatch
        TTNN::Ops::Experimental::DeepSeekPrefill::Combine
        TTNN::Ops::Experimental::DeepSeekPrefill::MaskedBincount
        TTNN::Ops::Experimental::DeepSeekPrefill::OffsetCumsum
        TTNN::Ops::Experimental::DeepSeekPrefill::Extract
        TTNN::Ops::Experimental::DeepSeekPrefill::Insert
        TTNN::Ops::Experimental::DeepSeekPrefill::MoeGroupedTopk
        TTNN::Ops::Experimental::PagedCache
        TTNN::Ops::Experimental::PlusOne
        TTNN::Ops::Experimental::Reduction
        TTNN::Ops::Experimental::Reshape
        TTNN::Ops::Experimental::SliceWrite
        TTNN::Ops::Experimental::SSM
        TTNN::Ops::Experimental::Transformer
        TTNN::Ops::Experimental::UnaryBackward
        TTNN::Ops::Experimental::Test
        TTNN::Ops::Full
        TTNN::Ops::FullLike
        TTNN::Ops::IndexFill
        TTNN::Ops::KvCache
        TTNN::Ops::Loss
        TTNN::Ops::Matmul
        TTNN::Ops::Moreh
        TTNN::Ops::Prefetcher
        TTNN::Ops::Reduction
        TTNN::Ops::SlidingWindow
        TTNN::Ops::Transformer
        TTNN::Ops::Uniform
        TTNN::Ops::Debug
        TTNN::Ops::PointToPoint
        TTNN::Ops::Pool
        TTNN::Ops::Normalization
        TTNN::Ops::Rand
        TTNN::Ops::Randn
)
if(TT_INSTALL)
    install(
        TARGETS
            ttnncpp
        LIBRARY
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT tar
    )
    install(
        TARGETS
            ttnncpp
        EXPORT TT-NN
        FILE_SET
        api
            COMPONENT ttnn-dev
        FILE_SET
        kernels
            DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn
            COMPONENT ttnn-runtime
        LIBRARY
            COMPONENT ttnn-runtime
    )
endif()

if(WITH_PYTHON_BINDINGS)
    add_library(ttnn SHARED)
    add_library(TTNN::PYTHON ALIAS ttnn)
    add_library(Metalium::TTNN ALIAS ttnn) # backwards compatibility. FIXME: remove
    TT_ENABLE_UNITY_BUILD(ttnn)

    set_target_properties(
        ttnn
        PROPERTIES
            EXPORT_NAME
                Python
            PREFIX
                "_"
            BUILD_RPATH
                "${PROJECT_BINARY_DIR}/tt_metal;${PROJECT_BINARY_DIR}/ttnn"
            INSTALL_RPATH
                "${PROJECT_BINARY_DIR}/lib;$ORIGIN/build/lib;$ORIGIN" # FIXME: Install RPATH should not have a build path!
            CXX_VISIBILITY_PRESET
                "default" # FIXME: Export the symbols we want, and use hidden visibility instead of default
            ADDITIONAL_CLEAN_FILES
                "${PROJECT_SOURCE_DIR}/ttnn/ttnn/_'${LIBRARY_NAME}'.so;${PROJECT_SOURCE_DIR}/ttnn/'${LIBRARY_NAME}'.egg-info" # FIXME: Confirm this is a relic of the past and can be deleted; No generated files should land in the source tree!
    )

    target_sources(
        ttnn
        PRIVATE
            ${SRC_PYBIND}
            ${TTNN_SRC_PYBIND}
            ${CCL_TTNN_SRCS_PYBIND}
            ${CCL_EXPERIMENTAL_TTNN_SRCS_PYBIND}
            ${DEBUG_TTNN_SRCS_PYBIND}
            ${TTNN_P2P_PYBIND}
    )
    target_include_directories(
        ttnn
        PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp>"
    )

    # TODO: This block was removed in main. Figure out how to interop with
    # manylinux work
    # nanobind requires this to be invoked before nanobind is included
    if(CMAKE_VERSION VERSION_LESS 3.18)
        set(DEV_MODULE Development)
    else()
        set(DEV_MODULE Development.Module)
    endif()

    set(Python3_FIND_STRATEGY LOCATION)
    find_package(
        Python3
        REQUIRED
        COMPONENTS
            Interpreter
            ${DEV_MODULE}
        OPTIONAL_COMPONENTS
            Development.SABIModule
    )
    message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")

    # https://nanobind.readthedocs.io/en/latest/api_cmake.html#low-level-interface
    # -static Perform a static library build (without this suffix, a shared build is used)
    # -abi3   Perform a stable ABI build targeting Python v3.12+.
    # -ft     Perform a build that opts into the Python 3.13+ free-threaded behavior.

    set(NANOBIND_LIBNAME "nanobind-static-abi3")
    nanobind_build_library(${NANOBIND_LIBNAME})
    get_target_property(NANOBIND_INCLUDES ${NANOBIND_LIBNAME} INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(
        ${NANOBIND_LIBNAME}
        PROPERTIES
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                "${NANOBIND_INCLUDES}"
    )
    add_library(TT::Nanobind ALIAS ${NANOBIND_LIBNAME})

    # GCC warns about visibility mismatches when using nanobind with -fvisibility=default
    # This affects code that uses nanobind types (not just nanobind headers themselves)
    target_compile_options(${NANOBIND_LIBNAME} INTERFACE "$<$<CXX_COMPILER_ID:GNU>:-Wno-attributes>")

    # find_package(Python3) required for here.
    # nanobind doesn't build otherwise.
    # PUBLIC on the headers also required
    target_include_directories(${NANOBIND_LIBNAME} PUBLIC ${Python3_INCLUDE_DIRS})

    target_link_libraries(
        ttnn
        PUBLIC
            "$<BUILD_INTERFACE:TTNN::CPP>"
        PRIVATE
            TT::Nanobind
            Boost::algorithm
            TT::Metalium
    )
    # .. enable size optimizations
    nanobind_opt_size(ttnn)

    # .. enable link time optimization
    nanobind_lto(ttnn)

    # .. set the default symbol visibility to 'hidden'
    #nanobind_set_visibility(ttnn)

    # .. strip unneeded symbols and debug info from the binary (only active in release builds)
    #nanobind_strip(ttnn)

    # .. disable the stack protector
    #nanobind_disable_stack_protector(ttnn)

    # .. set the Python extension suffix
    #nanobind_extension(ttnn)

    # .. set important compilation flags
    nanobind_compile_options(ttnn)

    # .. set important linker flags
    nanobind_link_options(ttnn)

    if(TT_INSTALL)
        install(TARGETS ttnn EXPORT TT-NN LIBRARY COMPONENT ttnn-runtime)
    endif()

    # Install .so into src files for nanobind implementation
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
endif()
if(TT_INSTALL)
    install(EXPORT TT-NN NAMESPACE TTNN:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tt-nn COMPONENT ttnn-dev)
endif()

if(TTNN_BUILD_TESTS)
    add_subdirectory(test)
endif()

set(FixmeOpAPIDir ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
set(FixmeOpIncDirs
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
)

add_subdirectory(cpp/ttnn/operations/bernoulli)
add_subdirectory(cpp/ttnn/operations/ccl)
add_subdirectory(cpp/ttnn/operations/conv)
add_subdirectory(cpp/ttnn/operations/creation)
add_subdirectory(cpp/ttnn/operations/core)
add_subdirectory(cpp/ttnn/operations/data_movement)
add_subdirectory(cpp/ttnn/operations/debug)
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
add_subdirectory(cpp/ttnn/operations/eltwise/unary_ng)
add_subdirectory(cpp/ttnn/operations/eltwise/unary_backward)
add_subdirectory(cpp/ttnn/operations/embedding)
add_subdirectory(cpp/ttnn/operations/embedding_backward)
add_subdirectory(cpp/ttnn/operations/examples)
add_subdirectory(cpp/ttnn/operations/experimental/adaptive_pool)
add_subdirectory(cpp/ttnn/operations/experimental/bcast_to)
add_subdirectory(cpp/ttnn/operations/experimental/ccl)
add_subdirectory(cpp/ttnn/operations/experimental/cnn)
add_subdirectory(cpp/ttnn/operations/experimental/conv3d)
add_subdirectory(cpp/ttnn/operations/experimental/copy)
add_subdirectory(cpp/ttnn/operations/experimental/deepseek_moe_post_combine_tilize)
add_subdirectory(cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce)
add_subdirectory(cpp/ttnn/operations/experimental/dropout)
add_subdirectory(cpp/ttnn/operations/experimental/isin)
add_subdirectory(cpp/ttnn/operations/experimental/matmul)
add_subdirectory(cpp/ttnn/operations/experimental/minimal_matmul)
add_subdirectory(cpp/ttnn/operations/experimental/deepseek/moe/moe_gate_mm)
add_subdirectory(cpp/ttnn/operations/experimental/topk_router_gpt)
add_subdirectory(cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo)
add_subdirectory(cpp/ttnn/operations/experimental/ccl/moe_gpt)
add_subdirectory(cpp/ttnn/operations/experimental/deepseek_prefill/dispatch)
add_subdirectory(cpp/ttnn/operations/experimental/deepseek_prefill/combine)
add_subdirectory(cpp/ttnn/operations/experimental/deepseek_prefill/masked_bincount)
add_subdirectory(cpp/ttnn/operations/experimental/deepseek_prefill/offset_cumsum)
add_subdirectory(cpp/ttnn/operations/experimental/deepseek_prefill/extract)
add_subdirectory(cpp/ttnn/operations/experimental/deepseek_prefill/insert)
add_subdirectory(cpp/ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk)
add_subdirectory(cpp/ttnn/operations/experimental/padded_slice)
add_subdirectory(cpp/ttnn/operations/experimental/paged_cache)
add_subdirectory(cpp/ttnn/operations/experimental/plusone)
add_subdirectory(cpp/ttnn/operations/experimental/reduction)
add_subdirectory(cpp/ttnn/operations/experimental/reshape)
add_subdirectory(cpp/ttnn/operations/experimental/slice_write)
add_subdirectory(cpp/ttnn/operations/experimental/ssm)
add_subdirectory(cpp/ttnn/operations/experimental/test)
add_subdirectory(cpp/ttnn/operations/experimental/transformer)
add_subdirectory(cpp/ttnn/operations/experimental/unary_backward)
add_subdirectory(cpp/ttnn/operations/full)
add_subdirectory(cpp/ttnn/operations/full_like)
add_subdirectory(cpp/ttnn/operations/index_fill)
add_subdirectory(cpp/ttnn/operations/kernel_helper_functions)
add_subdirectory(cpp/ttnn/operations/kv_cache)
add_subdirectory(cpp/ttnn/operations/loss)
add_subdirectory(cpp/ttnn/operations/matmul)
add_subdirectory(cpp/ttnn/operations/moreh)
add_subdirectory(cpp/ttnn/operations/normalization)
add_subdirectory(cpp/ttnn/operations/point_to_point)
add_subdirectory(cpp/ttnn/operations/pool)
add_subdirectory(cpp/ttnn/operations/prefetcher)
add_subdirectory(cpp/ttnn/operations/rand)
add_subdirectory(cpp/ttnn/operations/randn)
add_subdirectory(cpp/ttnn/operations/reduction)
add_subdirectory(cpp/ttnn/operations/sliding_window)
add_subdirectory(cpp/ttnn/operations/transformer)
add_subdirectory(cpp/ttnn/operations/uniform)
add_subdirectory(examples)
