# Split-TU fixture for testing ttsl::indirect / ttsl::PimplBase against an incomplete type.
# Built as its own library (not folded into the smoke OBJECT library) so a unity/jumbo build
# cannot merge widget.cpp into the same TU as the tests and silently defeat the incomplete-type
# coverage.
add_library(test_indirect_util STATIC widget.cpp)
target_link_libraries(test_indirect_util PRIVATE TT::STL)
