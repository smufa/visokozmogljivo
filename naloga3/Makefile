CPU_SRC = src/cpu/gray_scott_cpu.cpp
CUDA_SRC = src/cuda/gray_scott_cuda.cu
CUDA_SHARED_SRC = src/cuda/gray_scott_shared.cu
CPU_OUT = cpu.out
CUDA_OUT = cuda.out
CUDA_SHARED_OUT = cuda_shared.out

# Add CUDA include path
CUDA_PATH = /usr/local/cuda/include

all: cpu cuda cuda_shared

cpu: $(CPU_SRC)
	g++ -std=c++11 -Iinclude $(CPU_SRC) -O2 -o $(CPU_OUT)

cuda: $(CUDA_SRC)
	nvcc -diag-suppress 550 -O2 -I$(CUDA_PATH) -Iinclude $(CUDA_SRC) -o $(CUDA_OUT)

cuda_shared: $(CUDA_SHARED_SRC) $(CUDA_SRC)
	nvcc -diag-suppress 550 -O2 -I$(CUDA_PATH) -Iinclude $(CUDA_SHARED_SRC) -o $(CUDA_SHARED_OUT)

clean:
	rm -f $(CPU_OUT) $(CUDA_OUT)