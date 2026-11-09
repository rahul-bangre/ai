BUILD_DIR := build
BIN := $(BUILD_DIR)/qwen_infer
WEIGHTS := weights/qwen2.5-0.5b-instruct.f32.bin
SRCS := $(wildcard src/*.cpp include/*.h include/qwen/*.h)

ifeq ($(wildcard .venv/bin/python),)
PYTHON ?= python3
else
PYTHON ?= .venv/bin/python
endif

.PHONY: help build export run clean rebuild

help:
	@echo "make build     compile qwen_infer"
	@echo "make export    dump HF weights to $(WEIGHTS)"
	@echo "make run       build, export if needed, load weights"
	@echo "make clean     remove build artifacts"
	@echo "make rebuild   clean + build"

build: $(BIN)

$(BIN): CMakeLists.txt $(SRCS)
	cmake -S . -B $(BUILD_DIR)
	cmake --build $(BUILD_DIR) -j

export: $(WEIGHTS)

$(WEIGHTS): scripts/export_weights.py
	$(PYTHON) scripts/export_weights.py --out $@

run: $(BIN) $(WEIGHTS)
	$(BIN) --weights $(WEIGHTS)

clean:
	rm -rf $(BUILD_DIR)

rebuild: clean build
