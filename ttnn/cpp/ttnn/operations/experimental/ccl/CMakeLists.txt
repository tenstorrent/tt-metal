include(sources.cmake)

# The experimental CCL ops were removed for agent evaluation, so there is no
# ttnn_op_experimental_ccl library any more. The Python bindings that remain
# (fabric_mux and the empty ttnn.experimental CCL submodule) are registered
# directly on the shared `ttnn` Python module target, which only exists when
# WITH_PYTHON_BINDINGS is on, hence the guard.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_CCL_NANOBIND_SRCS})
endif()
