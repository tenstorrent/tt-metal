include(flatbuffers)

set(OP_PERF_FBS ${CMAKE_CURRENT_SOURCE_DIR}/schemas/reshard.fbs)

foreach(FBS_FILE ${OP_PERF_FBS})
    GENERATE_FBS_HEADER(${FBS_FILE})
    list(APPEND OP_PERF_GENERATED_HEADER ${FBS_GENERATED_HEADER_FILE})
endforeach()

add_executable(
    flatbuffers_example
    test_fb_des.cpp
    ${OP_PERF_GENERATED_HEADER}
)
target_include_directories(
    flatbuffers_example
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${FLATBUFFERS_INCLUDE_DIR}
)
target_link_libraries(flatbuffers_example PRIVATE flatbuffers)
