# Derived ("forked") TTNN ops of agentic model bring-ups; see INDEX.md. Every subfolder with a CMakeLists.txt is
# one fork, made by fork_op.py (target ttnn_op_bringup_<folder>). ttnncpp links TTNN::Ops::Bringup.
file(GLOB children RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*/CMakeLists.txt)
list(SORT children)
set(bringup_targets "")
foreach(child ${children})
    get_filename_component(dir ${child} DIRECTORY)
    add_subdirectory(${dir})
    list(APPEND bringup_targets ttnn_op_bringup_${dir})
endforeach()

add_library(ttnn_op_bringup INTERFACE)
add_library(TTNN::Ops::Bringup ALIAS ttnn_op_bringup)
foreach(t ${bringup_targets})
    target_link_libraries(ttnn_op_bringup INTERFACE ${t})
    # OBJECT leaves only pass usage requirements through an INTERFACE library; splice their objects in (see
    # ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/CMakeLists.txt).
    target_sources(ttnn_op_bringup INTERFACE $<TARGET_OBJECTS:${t}>)
endforeach()

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bringup_nanobind.cpp)
endif()
