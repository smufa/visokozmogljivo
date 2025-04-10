# module load CUDA
# make dn

cpu-compile:
	gcc img_hist.cpp -O2 -lm -lgomp -Wl,-rpath,./ -L./ -l:"libfreeimage.so.3" -o img_hist-cpu 

cpu-local: cpu-compile
	./img_hist-cpu images/800x600.jpg	

cpu-local-test-all: cpu-compile
	 @for f in $(shell ls images); do ./img_hist-cpu images/$${f}; done

cpu:
	gcc img_hist.cpp -O2 -lm -lgomp -lOpenCL -Wl,-rpath,./ -L./ -l:"libfreeimage.so.3" -o img_hist-cpu
	srun -n1 -G1 --reservation=fri img_hist-cpu images/800x600.jpg

gpu-compile:	
	gcc dn-07.c -O2 -lm -lgomp -lOpenCL -Wl,-rpath,./ -L./ -l:"libfreeimage.so.3" -o img_hist-gpu

gpu: gpu-compile
	srun -n1 -G1 --reservation=fri img_hist-gpu ./images/800x600.jpg

gpu-all:gpu-compile
	@for f in $(shell ls images); do srun -n1 -G1 --reservation=fri img_hist-gpu images/$${f}; done