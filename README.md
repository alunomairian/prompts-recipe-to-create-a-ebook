sh
# Código MPI em C

#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    printf("Hello world from processor %d out of %d processors\n", world_rank, world_size);

    MPI_Finalize();
    return 0;
}
Como Compilar e Executar
sh

mpicc -o hello_mpi hello_mpi.c
mpirun -np 4 ./hello_mpi

sh

# Código OpenMP em C

#include <omp.h>
#include <stdio.h>

int main() {
    int n = 1000;
    int a[n], b[n], sum[n];

    // Inicializa os arrays
    for(int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i;
    }

    #pragma omp parallel for
    for(int i = 0; i < n; i++) {
        sum[i] = a[i] + b[i];
    }

    // Exibe a soma de alguns elementos
    printf("Soma dos primeiros 10 elementos:\n");
    for(int i = 0; i < 10; i++) {
        printf("%d ", sum[i]);
    }
    printf("\n");

    return 0;
}
Como Compilar e Executar
sh

gcc -fopenmp -o sum_openmp sum_openmp.c
./sum_openmp

sh

# Código CUDA em C

#include <cuda.h>
#include <stdio.h>

__global__ void matrixMul(int* a, int* b, int* c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += a[row * N + i] * b[i * N + col];
    }
    c[row * N + col] = sum;
}

int main() {
    int N = 2;
    int a[4] = {1, 2, 3, 4};
    int b[4] = {1, 2, 3, 4};
    int c[4] = {0};

    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, N * N * sizeof(int));
    cudaMalloc((void**)&d_b, N * N * sizeof(int));
    cudaMalloc((void**)&d_c, N * N * sizeof(int));

    cudaMemcpy(d_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(2, 2);
    dim3 blocksPerGrid(1, 1);
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaMemcpy(c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Resultado da multiplicação de matrizes:\n");
    for (int i = 0; i < N * N; i++) {
        printf("%d ", c[i]);
        if ((i + 1) % N == 0) printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
Como Compilar e Executar
sh

nvcc -o matrixMul matrixMul.cu
./matrixMul
