# module load CUDA
# make dn

cpu-compile:
	g++ sampleQuential.cpp -O2 -o cpu.out

cpu-local-test-all: cpu-compile
	@mkdir -p result-cpu result-cpu-images
	@for f in $$(ls data); do \
		echo "Processing $$f..."; \
		srun -n1 --reservation=fri ./cpu.out data/$$f result-cpu-images/$${f} > result-cpu/$${f}.out; \
	done
