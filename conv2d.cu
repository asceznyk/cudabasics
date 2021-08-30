#include <cassert>
#include <cstdlib>
#include <iostream>

#define MASK_DIM 7

#define MASK_OFFSET (MASK_DIM / 2)

__constant__ int mask[7 * 7];

__global__ void conv2d(int *matrix, int *result, int N) {
  int y = blockIdx.y + blockDim.y + threadIdx.y; 
  int x = blockIdx.x + blockDim.x + threadIdx.x; 

  int s_y = y - MASK_OFFSET;
  int s_x = x - MASK_OFFSET;

  int temp = 0;
  for (int i = 0; i < MASK_DIM; i++) {
    for (int j = 0; j < MASK_DIM; j++) {
      if((s_y + i) >= 0 && (s_y + i) < N) {
        if((s_x + j) >=0 && (s_x + j) < N) {
          temp += matrix[N * (s_y + i) + (s_x + j)] * mask[MASK_DIM * i + j];
        }
      }
    }
  }

  result[N * y + x] = temp;
}

void init_matrix(int *m, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      m[n * i + j] = rand() % 100;
    }
  }
}

void verify_result(int *matrix, int *result, int *mask, int N) {
  int temp, o_y, o_x;
  
  for(int y = 0; y < N; y++) {
    for(int x = 0; x < N; x++) {

      temp = 0;
      
      for (int i = 0; i < MASK_DIM; i++) {
        o_y = y - MASK_OFFSET + i; 
        for (int j = 0; j < MASK_DIM; j++) {  
          o_x = y - MASK_OFFSET + j;
          if(o_y >= 0 && o_y < N) {
            if(o_x >=0 && o_x < N) {
              temp += matrix[N * o_y + o_x] * mask[MASK_DIM * i + j];
            }
          }
        }
      }

      assert(result[N * y + x] == temp);

    }
  }
}

int main() {
  int N = 1 << 10;

  size_t bytes_n = sizeof(int) * N * N;
  size_t bytes_m = sizeof(int) * MASK_DIM * MASK_DIM;

  int *matrix = new int[N * N];
  init_matrix(matrix, N);

  int *result = new int[N * N]; 

  int *h_mask = new int[MASK_DIM * MASK_DIM];
  init_matrix(h_mask, MASK_DIM);

  int *d_matrix, *d_result;
  cudaMalloc(&d_matrix, bytes_n); cudaMalloc(&d_result, bytes_n);

  cudaMemcpy(d_matrix, matrix, bytes_m, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(mask, h_mask, bytes_m);

  int n_threads = 16;
  int n_blocks = (N + n_threads - 1) / n_threads;

  dim3 block_dim(n_threads, n_threads);
  dim3 grid_dim(n_blocks, n_blocks);

  conv2d<<<grid_dim, block_dim>>>(d_matrix, d_result, N);

  cudaMemcpy(result, d_result, bytes_n, cudaMemcpyDeviceToHost);

  verify_result(matrix, result, h_mask, N);

  std::cout << "COMPLETED SUCCESSFULLY! \n";

  delete[] matrix; delete[] result; delete[] h_mask;
  cudaFree(d_matrix); cudaFree(d_result);

  return 0;
}


