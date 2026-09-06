BUILD_DIR ?= build
DATASET ?= /data/gds-lab.bin
SIZE_GIB ?= 4
BYTES ?= 1G
CHUNK_BYTES ?= 64M

.PHONY: configure build inspect dataset bench clean

# CUDA_HOME from the environment is only a candidate. scripts/cuda_env.sh
# skips an nvcc that cannot target this GPU, so CUDA_HOME=/usr cannot mask
# a capable toolkit. Recipes read $$CUDA_HOME after sourcing, not Make's copy.

configure:
	@set -e; \
	. ./scripts/cuda_env.sh >/dev/null 2>&1; \
	if [ -z "$$CUDA_HOME" ] || ! gdslab_nvcc_targets "$$CUDA_HOME/bin/nvcc" "$$(gdslab_gpu_arch 2>/dev/null)"; then \
	  echo "no nvcc that can target this GPU (CUDA_HOME=$${CUDA_HOME:-unset})." >&2; \
	  echo "Install one with scripts/install_cuda_redist.sh." >&2; \
	  exit 2; \
	fi; \
	cache=$(BUILD_DIR)/CMakeCache.txt; \
	if [ -f "$$cache" ] && ! grep -q "^CMAKE_CUDA_COMPILER:[A-Z]*=$$CUDA_HOME/bin/nvcc$$" "$$cache"; then \
	  echo "CUDA compiler changed since last configure; wiping $(BUILD_DIR)" >&2; \
	  rm -rf $(BUILD_DIR); \
	fi; \
	cmake -S . -B $(BUILD_DIR) \
	  -DCMAKE_BUILD_TYPE=Release \
	  -DCMAKE_CUDA_COMPILER=$$CUDA_HOME/bin/nvcc \
	  -DCUDAToolkit_ROOT=$$CUDA_HOME

build: configure
	@. ./scripts/cuda_env.sh >/dev/null 2>&1; cmake --build $(BUILD_DIR) -j

inspect:
	./scripts/inspect.sh $(DATASET)

dataset:
	./scripts/create_dataset.sh $(DATASET) $(SIZE_GIB)

bench: build
	@. ./scripts/cuda_env.sh >/dev/null 2>&1; ./scripts/run_bench.sh $(DATASET) $(BYTES) $(CHUNK_BYTES)

clean:
	rm -rf $(BUILD_DIR)
