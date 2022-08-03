# config for V-100

all:
	/usr/local/cuda-11/bin/nvcc -I /usr/local/cuda-11/targets/x86_64-linux/include/ -O3 main.cu -o transpose.out -arch=sm_70 -gencode=arch=compute_70,code=sm_70

clean:
	rm transpose.out