#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define CHECK(call)\
{\
    const cudaError_t error_now = call;\
    if(error_now != cudaSuccess) {\
        printf("Error: %s: %d,",__FILE__,__LINE__);\
        printf("code: %d,reason: %s\n",error_now,cudaGetErrorString(error_now));\
    }\
}\



__device__ void print_Arr(const float* arr)
{
    for (int i = 0; i < 8; i++) {
        printf("%f ", arr[i]);
    }
}

__global__ void test(const float* A,const float *B,float *C,const int N){
//    printf("\nhello,world from GPU\n");
//    for(int i = 0; i<N; ++i) {
//        C[i] = A[i] + B[i];
//    }
//    int i = threadIdx.x;
//    C[i] = A[i] + B[i];
//    int j = blockIdx.x;
//    C[j] = A[j] + B[j];
    int k = blockIdx.x *blockDim.x + threadIdx.x;
    if (k<N) C[k] = A[k] + B[k];

}
__global__ void sumMatrixGOU2D(const float* A,const float *B,float *C,const int nx, const int ny) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if(ix<nx && iy <ny) {
        int idex = ix + iy * nx;
        C[idex] = A[idex] + B[idex];
    }

}
__global__ void sumMatrixGOU2D_double(const double * A,const double *B, double *C,const int nx, const int ny) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if(ix<nx && iy <ny) {
        int idex = ix + iy * nx;
        C[idex] = A[idex] + B[idex];
    }

}

void print_arr_host(float *arr) {
    for (int i = 0; i < 8; i++) {
        printf("%f ", arr[i]);
    }
}
void sumArraysOnHost(const float* A,const float *B,float *C,const int N){
    for(int i = 0; i<N; ++i) {
        C[i] = A[i] + B[i];
    }
}

void sumArraysOnHost_double(const double* A,const double *B,double *C,const int N){
    for(int i = 0; i<N; ++i) {
        C[i] = A[i] + B[i];
    }
}
void initialData(float* arr,int size) {
    time_t t;
    srand((unsigned int) time(&t));
    for(int i = 0; i<size; ++i) {
        arr[i] = (float) (rand() & 0XFF)/10.0f;
    }
}

void initialData_double(double* arr,int size) {
    time_t t;
    srand((unsigned int) time(&t));
    for(int i = 0; i<size; ++i) {
        arr[i] = (double) (rand() & 0XFF)/10.0f;
    }
}

void checkResult(float *hostref ,float * gpuref,const int N) {
    double epsilon =1.0E-8;
    int match = 1;
    for(int i=0; i<N; ++i) {
        if(abs(hostref[i] - gpuref[i] >epsilon) ) {
            match = 0;
            printf("arrays do not match!\n");
            printf("host %5.2f gpu%5.2f at current %d\n",hostref[i],gpuref[i],i);
            break;
        }
    }
    if(match) printf("arrays alyways match!\n");
    return;
}
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double) tp.tv_usec *1.e-6);
}

void InitMatrixCPU2D(float*arr,const int nx,const int ny) {
    for(int y=0; y<ny; ++y){
        for(int x=0; x<nx; ++x){
            int idex = x + y * nx;
            arr[idex]=idex+1;
        }
    }
}
int main(int argc,char **argv) {
//    printf("hello,world from CPU\n");
    double iStart,iElaps;
    int dimx = 1;
    int dimy = 1;
    if(argc > 1) dimx = atoi (argv[1]);
    if(argc > 2)  {
        dimx = atoi (argv[1]);
        dimy = atoi (argv[2]);
    }

//    cudaDeviceReset();
//    cudaDeviceSynchronize();
//    auto  error=cudaDeviceSynchronize();
//    printf("%s\n", cudaGetErrorString(error));
    //malloc 第二版本2D
    int nx = 1<<8, ny = 1<<8;
    int nElems = nx * ny;
    size_t nBytes = nElems * sizeof(float);
    float * h_A2 = new float [nElems];
    float * h_B2 = new float [nElems];
    float * h_C2 = new float [nElems];
    float * gpuRef2 = new float [nElems];

    double * h_A3 = new double [nElems];
    double * h_B3 = new double [nElems];
    double * h_C3 = new double [nElems];
    double * gpuRef3 = new double [nElems];
//    InitMatrixCPU2D(h_A2,nx,ny);
//    InitMatrixCPU2D(h_B2,nx,ny);
    initialData(h_A2,nElems);
    initialData(h_B2,nElems);
    initialData_double(h_A3,nElems);
    initialData_double(h_B3,nElems);
    //  计时并运行
    iStart = cpuSecond();
    sumArraysOnHost(h_A2,h_B2,h_C2,nElems);
    iElaps = cpuSecond()-iStart;
    printf("cpu compute 2dsum time :%f\n", iElaps);

    iStart = cpuSecond();
    sumArraysOnHost_double(h_A3,h_B3,h_C3,nElems);
    iElaps = cpuSecond()-iStart;
    printf("cpu compute 2dsum_double time :%f\n", iElaps);

//    printf("h_A2:");
//    print_arr_host(h_A2);
//    printf("\nh_B2:");
//    print_arr_host(h_B2);
//    printf("\nh_C2:");
//    print_arr_host(h_C2);

    //gpu
//    float * d_A2;
//    float * d_B2;
//    float * d_C2;
//    cudaMalloc((float **)&d_A2,nBytes);
//    cudaMalloc((float **)&d_B2,nBytes);
//    cudaMalloc((float **)&d_C2,nBytes);
//
//    cudaMemcpy(d_A2,h_A2,nBytes,cudaMemcpyHostToDevice);
//    cudaMemcpy(d_B2,h_B2,nBytes,cudaMemcpyHostToDevice);

    double * d_A3;
    double * d_B3;
    double * d_C3;
    cudaMalloc((double **)&d_A3,nBytes);
    cudaMalloc((double **)&d_B3,nBytes);
    cudaMalloc((double **)&d_C3,nBytes);
    cudaMemcpy(d_A3,h_A3,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B3,h_B3,nBytes,cudaMemcpyHostToDevice);

    dim3 block2 (dimx, dimy);
    dim3 grid2 ((nx+block2.x-1)/block2.x,(ny+block2.y-1)/block2.y);

//    iStart = cpuSecond();
//    sumMatrixGOU2D<<<grid2,block2>>>(d_A2,d_B2,d_C2,nx,ny);
//    cudaDeviceSynchronize();
//    iElaps = cpuSecond()-iStart;
//    printf("gpu compute 2dsum time:%f when gird(%d,%d) block(%d,%d)\n", iElaps,grid2.x,grid2.y,block2.x,block2.y);

    iStart = cpuSecond();
    sumMatrixGOU2D_double<<<grid2,block2>>>(d_A3,d_B3,d_C3,nx,ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond()-iStart;
    printf("gpu compute 2dsum time:%f when gird(%d,%d) block(%d,%d)\n", iElaps,grid2.x,grid2.y,block2.x,block2.y);
//

//    dim3 block2 (32, 32);
//    dim3 grid2 ((nx+block2.x-1)/block2.x,(ny+block2.y-1)/block2.y);
//
//    iStart = cpuSecond();
//    sumMatrixGOU2D<<<grid2,block2>>>(d_A2,d_B2,d_C2,nx,ny);
//    cudaDeviceSynchronize();
//    iElaps = cpuSecond()-iStart;
//    printf("\ngpu compute 2dsum time:%f when gird(%d,%d) block(%d,%d)\n", iElaps,grid2.x,grid2.y,block2.x,block2.y);
//
//    dim3 block3 (32, 16);
//    dim3 grid3 ((nx+block3.x-1)/block3.x,(ny+block3.y-1)/block3.y);
//    iStart = cpuSecond();
//    sumMatrixGOU2D<<<grid3,block3>>>(d_A2,d_B2,d_C2,nx,ny);
//    cudaDeviceSynchronize();
//    iElaps = cpuSecond()-iStart;
//    printf("\ngpu compute 2dsum time:%f when gird(%d,%d) block(%d,%d)\n", iElaps,grid3.x,grid3.y,block3.x,block3.y);
//
//    dim3 block4 (16, 32);
//    dim3 grid4 ((nx+block4.x-1)/block4.x,(ny+block4.y-1)/block4.y);
//    iStart = cpuSecond();
//    sumMatrixGOU2D<<<grid4,block4>>>(d_A2,d_B2,d_C2,nx,ny);
//    cudaDeviceSynchronize();
//    iElaps = cpuSecond()-iStart;
//    printf("\ngpu compute 2dsum time:%f when gird(%d,%d) block(%d,%d)\n", iElaps,grid4.x,grid4.y,block4.x,block4.y);
//
//    dim3 block5 (16, 16);
//    dim3 grid5 ((nx+block5.x-1)/block5.x,(ny+block5.y-1)/block5.y);
//    iStart = cpuSecond();
//    sumMatrixGOU2D<<<grid5,block5>>>(d_A2,d_B2,d_C2,nx,ny);
//    cudaDeviceSynchronize();
//    iElaps = cpuSecond()-iStart;
//    printf("\ngpu compute 2dsum time:%f when gird(%d,%d) block(%d,%d)\n", iElaps,grid5.x,grid5.y,block5.x,block5.y);

//    cudaMemcpy(gpuRef2,d_C2,nBytes,cudaMemcpyDeviceToHost);
//    checkResult(h_C2,gpuRef2,nElems);
//
//
//
//    // 第一版本1D
////    float * h_A = (float *)malloc(nBytes);
////    float * h_B = (float *)malloc(nBytes);
////    float * h_C = (float *)malloc(nBytes);
////    float * gpuRef = (float *)malloc(nBytes);
//
//    float * h_A = new float[nElems];
//    float * h_B = new float[nElems];
//    float * h_C = new float[nElems];
//    float * gpuRef = new float[nElems];
//
//    if (h_A == NULL || h_B == NULL ||h_C == NULL) {
//        printf("内存分配失败\n");
//        return 1;
//    }
//    // 准备数据
//    iStart = cpuSecond();
//    initialData(h_A,nElems);
//    initialData(h_B,nElems);
//    iElaps = cpuSecond()-iStart;
//    printf("\ninitialdata time:%f\n", iElaps);
//
//    //CPU处理数据
//    iStart = cpuSecond();
//    sumArraysOnHost(h_A2,h_B2,h_C,nElems);
//    iElaps = cpuSecond()-iStart;
//    printf("cpu compute sum time:%f\n", iElaps);
//
////    printf("\nh_A:");
////    print_arr_host(h_A2);
////    printf("\nh_B:");
////    print_arr_host(h_B2);
////    printf("\nh_C:");
////    print_arr_host(h_C);
//
//    //cudamalloc
//    float * d_A;
//    float * d_B;
//    float * d_C;
//    cudaMalloc((float **)&d_A, nBytes);
//    cudaMalloc((float **)&d_B, nBytes);
//    cudaMalloc((float **)&d_C, nBytes);
//
//    //从主机拷贝到设备内存中，注意方向
//    cudaMemcpy(d_A,h_A2,nBytes,cudaMemcpyHostToDevice);
//    cudaMemcpy(d_B,h_B2,nBytes,cudaMemcpyHostToDevice);
//
//    int iLen = 1024;
//    dim3 block (iLen);
//    dim3 grid ((nElems+block.x-1)/block.x);
//    iStart = cpuSecond();
//    //用1一个块中不同线程并行 int i = threadIdx.x; C[i] = A[i] + B[i];
////    test<<<1,nElems>>>(d_A,d_B,d_C,nElems);
////    用不同的块并行计算  int j = blockIdx.x; C[j] = A[j] + B[j];
////    test<<<nElems,1>>>(d_A,d_B,d_C,nElems);
//    test<<<grid,block>>>(d_A,d_B,d_C,nElems);
//    cudaDeviceSynchronize();
//    iElaps = cpuSecond()- iStart;
//
//    printf("gpu compute sum time:%f when gird(%d,%d) block(%d,%d)\n", iElaps,grid.x,grid.y,block.x,block.y);
//    CHECK(cudaMemcpy(gpuRef,d_C,nBytes,cudaMemcpyDeviceToHost));
//
//
////    printf("gpuRef:");
////    print_arr_host(gpuRef);
////    printf("\n");
//    checkResult(h_C,gpuRef,nElems);

    //释放内存
//    delete(h_A);
//    delete(h_B);
//    delete(h_C);
//    delete(gpuRef);
//    cudaFree(d_A);
//    cudaFree(d_B);
//    cudaFree(d_C);

//    delete(h_A2);
//    delete(h_B2);
//    delete(h_C2);
//    delete(gpuRef2);
//    cudaFree(d_A2);
//    cudaFree(d_B2);
//    cudaFree(d_C2);

    delete[](h_A3);
    delete[](h_B3);
    delete[](h_C3);
    delete(gpuRef3);
    cudaFree(d_A3);
    cudaFree(d_B3);
    cudaFree(d_C3);


    return 0;
}