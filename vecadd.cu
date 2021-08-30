#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

__global__ void add_array(const int *__restrict a, const int *__restrict b,
                          int *__restrict c, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < N) c[i] = a[i] + b[i];
}

void init_array(std::vector<int> &arr, int m) {
  for (int i = 0; i < m; i++)
    arr.push_back(rand() % 100);
}

void verify_result(std::vector<int> &a, std::vector<int> &b, std::vector<int> &c) {
  for (int i = 0; i < a.size(); i++) 
    assert(c[i] == a[i] + b[i]);
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

  cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

  int n_threads = 1 << 10; //2^10 = 1024 threads
  int n_blocks = (N + n_threads - 1) / n_threads;

  add_array<<<n_blocks, n_threads>>>(d_a, d_b, d_c, N);

  cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  verify_result(a, b, c);

  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

  std::cout << "COMPLETED SUCCESSFULLY! \n";

  int out = a[0] + b[0];
  printf("for example a[0] + b[0] = %d and c[0] = %d; \n", out, c[0]);
  
  return 0;
}
