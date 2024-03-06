TTNN_LIB = $(LIBDIR)/libttnn.so
TTNN_PYBIND11_LIB = $(LIBDIR)/_ttnn.so
TTNN_PYBIND11_LOCAL_SO = ttnn/ttnn/_ttnn.so

TTNN_DEFINES =

TTNN_INCLUDES = $(TT_EAGER_INCLUDES) $(TT_LIB_INCLUDES) -Ittnn/cpp -Ittnn/cpp/ttnn
TTNN_PYBIND11_INCLUDES = $(TTNN_INCLUDES) $(shell python3-config --includes) -Itt_metal/third_party/pybind11/include

TTNN_LDFLAGS = -ltt_dnn -ldtx -ltensor -ltt_metal -lyaml-cpp $(LDFLAGS)
TTNN_PYBIND11_LDFLAGS = $(TTNN_LDFLAGS) -ltt_lib_csrc -lttnn

TTNN_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast -fno-var-tracking

TTNN_SRC_DIR = ttnn/cpp/ttnn
TTNN_SRCS = $(wildcard $(TTNN_SRC_DIR)/*.cpp) $(wildcard $(TTNN_SRC_DIR)/*/*.cpp)

TTNN_PYBIND11_SRCS = \
    ttnn/cpp/pybind11/__init__.cpp

TTNN_OBJS = $(addprefix $(OBJDIR)/, $(TTNN_SRCS:.cpp=.o))
TTNN_PYBIND11_OBJS = $(addprefix $(OBJDIR)/, $(TTNN_PYBIND11_SRCS:.cpp=.o))

TTNN_DEPS = $(addprefix $(OBJDIR)/, $(TTNN_SRCS:.cpp=.d))
-include $(TTNN_DEPS)

-include $(TTNN_PYBIND11_SRCS:.cpp=.d)

TTNN_LIBS_TO_BUILD = $(TTNN_LIB) \
		     $(TTNN_PYBIND11_LIB) \

ifdef TT_METAL_ENV_IS_DEV
TTNN_LIBS_TO_BUILD += \
	$(TTNN_PYBIND11_LOCAL_SO)
endif

ttnn: $(TTNN_LIBS_TO_BUILD)

$(TTNN_LIB): $(TTNN_OBJS) $(TT_DNN_LIB) $(TENSOR_LIB) $(DTX_LIB) $(TT_METAL_LIB) tt_eager/tt_lib
	@mkdir -p $(LIBDIR)
	$(CXX) $(TTNN_CFLAGS) $(CXXFLAGS) $(SHARED_LIB_FLAGS) -o $@ $(TTNN_OBJS) $(TTNN_LDFLAGS)

$(TTNN_PYBIND11_LIB): $(TTNN_PYBIND11_OBJS) $(TTNN_LIB)
	@mkdir -p $(LIBDIR)
	$(CXX) $(TTNN_CFLAGS) $(CXXFLAGS) $(SHARED_LIB_FLAGS) -o $@ $(TTNN_PYBIND11_OBJS) $(TTNN_LIB) $(TTNN_PYBIND11_LDFLAGS)

$(TTNN_PYBIND11_LOCAL_SO): $(TTNN_PYBIND11_LIB)
	cp -fp $^ $@

$(OBJDIR)/ttnn/cpp/ttnn/%.o: ttnn/cpp/ttnn/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(TTNN_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TTNN_INCLUDES) -MMD -MP -c -o $@ $<

$(OBJDIR)/ttnn/cpp/pybind11/%.o: ttnn/cpp/pybind11/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(TTNN_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TTNN_PYBIND11_INCLUDES) -MMD -MP -c -o $@ $<
