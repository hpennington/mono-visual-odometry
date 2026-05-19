BUILD_DIR := build
OPEN3D_ROOT := $(HOME)/open3d_install

.PHONY: all release clean

all:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake -DOpen3D_ROOT=$(OPEN3D_ROOT) .. && make

release:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake -DOpen3D_ROOT=$(OPEN3D_ROOT) -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native" .. && make

clean:
	rm -rf $(BUILD_DIR)
