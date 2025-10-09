#!/bin/bash

echo "========================================"
echo "Race Condition Fix Verification"
echo "========================================"
echo ""

# Check if source files have the fixes
echo "1. Checking source code for fixes..."
echo ""

if grep -q "g_device_mutex_map_lock" tt_metal/impl/buffers/buffer.cpp; then
    echo "✅ buffer.cpp: Per-device mutex fix FOUND"
else
    echo "❌ buffer.cpp: Per-device mutex fix NOT FOUND"
fi

if grep -q "get_device_lifecycle_mutex" tt_metal/impl/buffers/buffer.cpp; then
    echo "✅ buffer.cpp: Helper function FOUND"
else
    echo "❌ buffer.cpp: Helper function NOT FOUND"
fi

if grep -q "g_allocation_tracking_mutex" tt_metal/graph/graph_tracking.cpp; then
    echo "✅ graph_tracking.cpp: Global tracking mutex FOUND"
else
    echo "❌ graph_tracking.cpp: Global tracking mutex NOT FOUND"
fi

echo ""
echo "2. Checking if code needs rebuild..."
echo ""

# Check if object files exist and are newer than source
BUFFER_CPP="tt_metal/impl/buffers/buffer.cpp"
GRAPH_CPP="tt_metal/graph/graph_tracking.cpp"

# Find compiled objects
BUFFER_OBJ=$(find build -name "buffer.cpp.o" 2>/dev/null | head -1)
GRAPH_OBJ=$(find build -name "graph_tracking.cpp.o" 2>/dev/null | head -1)

if [ -f "$BUFFER_OBJ" ]; then
    if [ "$BUFFER_CPP" -nt "$BUFFER_OBJ" ]; then
        echo "⚠️  buffer.cpp is NEWER than compiled object - REBUILD NEEDED"
    else
        echo "✅ buffer.cpp.o is up to date"
    fi
else
    echo "⚠️  buffer.cpp.o NOT FOUND - BUILD NEEDED"
fi

if [ -f "$GRAPH_OBJ" ]; then
    if [ "$GRAPH_CPP" -nt "$GRAPH_OBJ" ]; then
        echo "⚠️  graph_tracking.cpp is NEWER than compiled object - REBUILD NEEDED"
    else
        echo "✅ graph_tracking.cpp.o is up to date"
    fi
else
    echo "⚠️  graph_tracking.cpp.o NOT FOUND - BUILD NEEDED"
fi

echo ""
echo "3. Recommendations..."
echo ""
echo "To rebuild TT-Metal with the fixes:"
echo "  cd /workspace/tt-metal-apv"
echo "  cmake --build build -j\$(nproc)"
echo ""
echo "To test the fix:"
echo "  export TT_BUFFER_DEBUG_LOG=0"
echo "  export TT_ALLOC_TRACKING_ENABLED=1"
echo "  # Start server"
echo "  ./allocation_server_poc &"
echo "  # Run your test"
echo "  python your_test.py"
echo "  # Check results"
echo "  grep -c 'unknown buffer' debug.log"
echo ""
