BUILD_DIR ?= build
DATASET ?= /tmp/gds-lab.bin
SIZE_GIB ?= 4
BYTES ?= 1G
CHUNK_BYTES ?= 64M

.PHONY: configure build inspect dataset bench clean

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release

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
