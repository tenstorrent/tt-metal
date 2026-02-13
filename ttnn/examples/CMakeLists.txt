add_executable(example_add)
target_sources(example_add PRIVATE add/add.cpp)
target_link_libraries(example_add PRIVATE TTNN::CPP)

add_executable(example_lab_eltwise_binary)
target_sources(example_lab_eltwise_binary PRIVATE lab_eltwise_binary/lab_eltwise_binary.cpp)
target_link_libraries(
    example_lab_eltwise_binary
    PRIVATE
        TT::Metalium
        TTNN::CPP
)

add_executable(example_lab_multicast)
target_sources(example_lab_multicast PRIVATE lab_multicast/lab_multicast.cpp)
target_link_libraries(
    example_lab_multicast
    PRIVATE
        TT::Metalium
        TTNN::CPP
)

add_executable(example_lab3_multicast_ex1)
target_sources(example_lab3_multicast_ex1 PRIVATE lab3_multicast_ex1/lab3_multicast_ex1.cpp)
target_link_libraries(
    example_lab3_multicast_ex1
    PRIVATE
        TT::Metalium
        TTNN::CPP
)

add_executable(example_lab3_ex2)
target_sources(example_lab3_ex2 PRIVATE lab3_ex2/lab3_ex2.cpp)
target_link_libraries(
    example_lab3_ex2
    PRIVATE
        TT::Metalium
        TTNN::CPP
)

add_executable(example_lab3_matmul_ex1)
target_sources(example_lab3_matmul_ex1 PRIVATE lab3_matmul_ex1/lab3_matmul_ex1.cpp)
target_link_libraries(
    example_lab3_matmul_ex1
    PRIVATE
        TT::Metalium
        TTNN::CPP
)

add_executable(example_lab3_matmul_ex3)
target_sources(example_lab3_matmul_ex3 PRIVATE lab3_matmul_ex3/lab3_matmul_ex3.cpp)
target_link_libraries(
    example_lab3_matmul_ex3
    PRIVATE
        TT::Metalium
        TTNN::CPP
)

add_executable(example_lab2_matmul_ex2)
target_sources(example_lab2_matmul_ex2 PRIVATE lab2_matmul_ex2/lab2_matmul_ex2.cpp)
target_link_libraries(
    example_lab2_matmul_ex2
    PUBLIC
        TT::Metalium
        TTNN::TTNN
)

add_executable(example_lab2_matmul_ex2_alt)
target_sources(example_lab2_matmul_ex2_alt PRIVATE lab2_matmul_ex2_alt/lab2_matmul_ex2_alt.cpp)
target_link_libraries(
    example_lab2_matmul_ex2_alt
    PUBLIC
        TT::Metalium
        TTNN::TTNN
)

install(
    DIRECTORY
        add
    # DESTINATION ${CMAKE_INSTALL_DOCDIR}/examples
    # FIXME(afuller): Something funky is happening when installing files into /usr/share/doc on a default Docker image.
    #                 Speculation: some dependency for magic doc handling is missing.
    #                 For now keep it out of 'doc'.
    DESTINATION "${CMAKE_INSTALL_DATADIR}/tt-nn/examples"
    COMPONENT ttnn-examples
)
