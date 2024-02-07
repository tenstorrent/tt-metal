# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

ttnn/dev_install: python_env/dev
	echo "Installing editable dev version of ttnn package..."
	bash -c "source $(PYTHON_ENV)/bin/activate && pip install -e ttnn"

TTNN_LIB = $(LIBDIR)/libttnn.so
TTNN_PYBIND_LIB = $(LIBDIR)/_ttnn.so

TTNN_DEFINES =

TTNN_INCLUDES = -Itt_eager/tt_dnn $(TT_EAGER_INCLUDES) $(TT_LIB_INCLUDES)
TTNN_PYBIND11_INCLUDES = $(TTNN_INCLUDES) $(shell python3-config --includes) -Itt_metal/third_party/pybind11/include

TTNN_LDFLAGS = -ltt_dnn -ldtx -ltensor -ltt_metal -lyaml-cpp $(LDFLAGS)
TTNN_LDFLAGS += -L$(LIBDIR) -ldevice -lqueue -ltracy -ltt_lib_csrc
TTNN_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast -fno-var-tracking

TTNN_SRCS =

TTNN_PYBIND_SRCS = \
    ttnn/cpp/module.cpp

TTNN_OBJS = $(addprefix $(OBJDIR)/, $(TTNN_SRCS:.cpp=.o))
TTNN_PYBIND_OBJS = $(addprefix $(OBJDIR)/, $(TTNN_PYBIND_SRCS:.cpp=.o))

TTNN_DEPS = $(addprefix $(OBJDIR)/, $(TTNN_SRCS:.cpp=.d))
-include $(TTNN_DEPS)

-include $(TTNN_PYBIND_SRCS:.cpp=.d)


ttnn: $(TTNN_LIB) $(TTNN_PYBIND_LIB)

$(TTNN_LIB): $(TTNN_OBJS) $(TT_DNN_LIB) $(TENSOR_LIB) $(DTX_LIB) $(TT_METAL_LIB) $(TT_LIBS_TO_BUILD)
	@mkdir -p $(LIBDIR)
	$(CXX) $(TTNN_CFLAGS) $(CXXFLAGS) $(SHARED_LIB_FLAGS) -o $@ $(TTNN_OBJS) $(TTNN_LDFLAGS)

$(TTNN_PYBIND_LIB): $(TTNN_PYBIND_OBJS)
	@mkdir -p $(LIBDIR)
	$(CXX) $(TTNN_CFLAGS) $(CXXFLAGS) $(SHARED_LIB_FLAGS) -o $@ $(TTNN_PYBIND_OBJS) $(TTNN_LDFLAGS)

.PHONY: ttnn/setup_local_so
ttnn/setup_local_so: $(TTNN_PYBIND_LIB)
	rm -f ttnn/ttnn/_ttnn.so
	cp $^ ttnn/ttnn/_ttnn.so


$(OBJDIR)/ttnn/cpp/%.o: ttnn/cpp/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(TTNN_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TTNN_INCLUDES) -c -o $@ $<
