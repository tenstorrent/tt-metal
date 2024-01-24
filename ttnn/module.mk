# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

ttnn/dev_install: python_env/dev
	echo "Installing editable dev version of ttnn package..."
	bash -c "source $(PYTHON_ENV)/bin/activate && pip install -e ttnn"
