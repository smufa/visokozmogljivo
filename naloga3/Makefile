CPU_SRC = src/cpu/gray_scott_cpu.cpp
CUDA_SRC = src/cuda/gray_scott_cuda.cu
CPU_OUT = cpu.out
CUDA_OUT = cuda.out

# Add CUDA include path
CUDA_PATH = /usr/local/cuda/include

all: cpu cuda

cpu: $(CPU_SRC)
	g++ -Iinclude $(CPU_SRC) -O2 -o $(CPU_OUT)

cuda: $(CUDA_SRC)
	nvcc -diag-suppress 550 -O2 -I$(CUDA_PATH) -Iinclude $(CUDA_SRC) -o $(CUDA_OUT)

clean:
	rm -f $(CPU_OUT) $(CUDA_OUT)