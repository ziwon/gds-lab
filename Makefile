BUILD_DIR ?= build
DATASET ?= /data/gds-lab.bin
SIZE_GIB ?= 4
BYTES ?= 1G
CHUNK_BYTES ?= 64M
# Resolved by capability, not by path: scripts/cuda_env.sh rejects an nvcc that
# cannot target this host's GPU. Set CUDA_HOME in the environment to override.
ifndef CUDA_HOME
CUDA_HOME := $(shell . ./scripts/cuda_env.sh >/dev/null 2>&1; printf '%s' "$$CUDA_HOME")
endif

ifneq ($(CUDA_HOME),)
export CUDA_HOME
export PATH := $(CUDA_HOME)/bin:$(PATH)
export LD_LIBRARY_PATH := $(CUDA_HOME)/lib64:$(CUDA_HOME)/lib$(if $(LD_LIBRARY_PATH),:$(LD_LIBRARY_PATH))
endif

.PHONY: configure build inspect dataset bench clean

configure:
	@. ./scripts/cuda_env.sh >/dev/null 2>&1; \
	if ! gdslab_nvcc_targets "$(CUDA_HOME)/bin/nvcc" "$$(gdslab_gpu_arch 2>/dev/null)"; then \
	  echo "CUDA_HOME=$(CUDA_HOME) has no nvcc that can target this GPU." >&2; \
	  echo "Install one with scripts/install_cuda_redist.sh, or set CUDA_HOME to a toolkit" >&2; \
	  echo "whose nvcc lists this GPU in --list-gpu-arch." >&2; \
	  exit 2; \
	fi
	@cache=$(BUILD_DIR)/CMakeCache.txt; \
	if [ -f "$$cache" ] && ! grep -q "^CMAKE_CUDA_COMPILER:[A-Z]*=$(CUDA_HOME)/bin/nvcc$$" "$$cache"; then \
	  echo "CUDA compiler changed since last configure; wiping $(BUILD_DIR)" >&2; \
	  rm -rf $(BUILD_DIR); \
	fi
	cmake -S . -B $(BUILD_DIR) \
	  -DCMAKE_BUILD_TYPE=Release \
	  -DCMAKE_CUDA_COMPILER=$(CUDA_HOME)/bin/nvcc \
	  -DCUDAToolkit_ROOT=$(CUDA_HOME)

build: configure
	cmake --build $(BUILD_DIR) -j

inspect:
	./scripts/inspect.sh $(DATASET)

dataset:
	./scripts/create_dataset.sh $(DATASET) $(SIZE_GIB)

bench: build
	./scripts/run_bench.sh $(DATASET) $(BYTES) $(CHUNK_BYTES)

clean:
	rm -rf $(BUILD_DIR)
