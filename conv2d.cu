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

int main() {
  int N = 1 << 10;

  int bytes_n = sizeof(int) * N * N;
  int bytes_m = sizeof(int) * MASK_DIM * MASK_DIM;

  int matrix = new int[N * N];
  int result = new int[N * N];

  return 0;
}


