# Smoke
add_executable(tt-nn-validation-smoke)

target_link_libraries(tt-nn-validation-smoke PRIVATE TT::NN::Test::Smoke)

install(TARGETS tt-nn-validation-smoke RUNTIME COMPONENT ttnn-validation)

# Basic
add_executable(tt-nn-validation-basic)

target_link_libraries(tt-nn-validation-basic PRIVATE TT::NN::Test::Basic)

install(TARGETS tt-nn-validation-basic RUNTIME COMPONENT ttnn-validation)
