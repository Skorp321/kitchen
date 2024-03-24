#include <stdio.h>

__global__ void helloWorld() {
    // Определяем индекс текущего потока
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    printf("Hello, World! I'm thread %d\n", tid);
}

int main() {
    // Устанавливаем количество блоков и количество потоков на блок
    int numBlocks = 2;
    int threadsPerBlock = 4;
    
    // Запускаем ядро
    helloWorld<<<numBlocks, threadsPerBlock>>>();
    
    // Ждем завершения работы GPU
    cudaDeviceSynchronize();
    
    return 0;
}

