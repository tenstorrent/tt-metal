TT_LIBS_HOME ?= $(TT_METAL_HOME)
EAGER_OUTPUT_DIR = $(OUT)/dist

ttnn:
	echo "Installing editable dev version of tt_eager packages..."
	bash -c "source $(PYTHON_ENV)/bin/activate && pip install -e ."

ttnn/clean:
	rm -rf ttnn/*.egg-info
	rm -rf $(EAGER_OUTPUT_DIR)
