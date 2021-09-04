#include <cassert>
#include <cstdlib>
#include <iostream>

void print_matrix(int *m, int y, int x) {
  for(int i = 0; i < y; i++) {
    printf("[");
    for(int j = 0; j < x; j ++)
      printf("%d, ", m[i * x + j]);
    printf("]\n");
  }
  printf("\n");
}

void init_matrix(int *m, int l) {
  for (int i = 0; i < l; i++) {
    m[i] = rand() % 100;
  }
}

__global__ void matmul2d(int *a, int *b, int *c, int m, int n) {
  int i = blockIdx.y * blockDim.y + threadIdx.y; 
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  
  int temp = 0;
  for (int k = 0; k < m; k++) {
    temp += a[i * m + k] * b[k * n + j];
  }
  c[i * n + j] = temp;
}

void verify_result(int *a, int *b, int *c, int l, int m, int n) {
  int temp = 0;
  for(int i = 0; i < l; i++) {
    for(int j = 0; j < n; j++) {
      temp = 0;
      for(int k = 0; k < m; k++) {
        temp += a[i * m + k] * b[k * n + j]; //c[i][j] += a[i][k] * b[k][j]
      }
      printf("%d, %d", c[i * n + j], temp);
      assert(c[i * n + j] == temp); 
    }
  }
}

int main() {
  //a of size l * m and b of size m * n
  //a @ b is of size l * n

  int L = 3;//1 << 9; //512
  int M = 2;//1 << 8; //256
  int N = 3;//1 << 10; //1024

  size_t bytes_a = sizeof(int) * L * M;
  size_t bytes_b = sizeof(int) * M * N;
  size_t bytes_c = sizeof(int) * L * N;

  int *a = new int[L * M];
  int *b = new int[M * N];
  int *c = new int[L * N];

  init_matrix(a, L * M);
  init_matrix(b, M * N);

  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes_a); cudaMalloc(&d_b, bytes_b); cudaMalloc(&d_c, bytes_c);
  cudaMemcpy(d_a, a, bytes_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, bytes_b, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, bytes_c, cudaMemcpyHostToDevice);

  int n_ops = L * N;
  int n_threads = 1; //32
  int n_blocks = (n_ops + n_threads - 1) / n_threads;

  dim3 block_dim(n_threads, n_threads);
  dim3 grid_dim(n_blocks, n_blocks);

  matmul2d<<<grid_dim, block_dim>>>(d_a, d_b, d_c, M, N);
  cudaMemcpy(c, d_c, bytes_c, cudaMemcpyDeviceToHost); 

  print_matrix(a, L, M);
  print_matrix(b, M, N);
  print_matrix(c, L, N);

  verify_result(a, b, c, L, M, N);

  std::cout << "COMPLETED SUCCESSFULLY! \n";

  delete[] a; delete[] b; delete[] c;
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

  return 0;
}


