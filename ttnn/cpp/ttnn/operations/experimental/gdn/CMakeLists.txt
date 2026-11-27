# GDN-domain build. The operation leaves reuse the KDA validators (KDA::Common) and
# are exported through one domain umbrella.
add_subdirectory(gdn_decode_step)
add_subdirectory(gdn_spec_step)

add_library(ttnn_op_experimental_gdn INTERFACE)
add_library(TTNN::Ops::Experimental::GDN ALIAS ttnn_op_experimental_gdn)
target_link_libraries(
    ttnn_op_experimental_gdn
    INTERFACE
        TTNN::Ops::Experimental::GDN::GdnDecodeStep
        TTNN::Ops::Experimental::GDN::GdnSpecStep
)

# OBJECT libraries do not propagate their compiled objects through a nested
# INTERFACE link. Inject each GDN object set here exactly once; these generator
# expressions are empty when the corresponding target is a shared library.
target_sources(
    ttnn_op_experimental_gdn
    INTERFACE
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::GDN::GdnDecodeStep>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::GDN::GdnSpecStep>
)
