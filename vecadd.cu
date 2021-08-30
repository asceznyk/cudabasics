#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

void print_array(std::vector<int> &arr, int m) {
  for (int i = 0; i < m; i++)
    printf("%d, ", arr[i]);
}

void init_array(std::vector<int> &arr, int m) {
  for (int i = 0; i < m; i++)
    arr.push_back(rand() % 100);
}

__global__ void add_array(const int *__restrict a, const int *__restrict b,
                          int *__restrict c, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < N) c[i] = a[i] + b[i];
}

int main() {
  constexpr int N = 1 << 16; //2^16 = 65536 elements
  constexpr size_t bytes = sizeof(int) * N; 

  std::vector<int> a; a.reserve(N);
  std::vector<int> b; b.reserve(N);
  std::vector<int> c; c.reserve(N);

  init_array(a, N); init_array(b, N);

  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes); cudaMalloc(&d_b, bytes); cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

  print_array(a, N);
  printf("\n");
  print_array(b, N);
}

