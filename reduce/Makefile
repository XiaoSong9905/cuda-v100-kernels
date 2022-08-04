all:
	/usr/local/cuda-11/bin/nvcc -I /usr/local/cuda-11/targets/x86_64-linux/include/ -fopenmp -O3 main.cu -o reduce.out -arch=sm_70 -gencode=arch=compute_70,code=sm_70

clean:
	rm reduce.out