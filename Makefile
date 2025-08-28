PY = python
PIP = python -m pip

.PHONY: setup test train eval interp fmt

setup:
	@echo "[setup] Upgrading pip and installing project deps"
	$(PIP) install -U pip wheel setuptools
	# Install GPU PyTorch first if CUDA index is provided via TORCH_INDEX_URL
	if [ -n "$$TORCH_INDEX_URL" ]; then \
		$(PIP) install --index-url $$TORCH_INDEX_URL "torch>=2.1"; \
	fi
	$(PIP) install -e .

test:
	pytest -q

train:
	$(PY) train.py exp=life_smoke

eval:
	$(PY) eval.py exp=life32 ckpt=latest

interp:
	$(PY) interp_run.py exp=life32 action=probes

