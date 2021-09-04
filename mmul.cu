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
}

void init_matrix(int *m, int l) {
  for (int i = 0; i < l; i++) {
    m[i] = rand() % 100;
  }
}

void matmul(int *a, int *b, int *c, int l, int m, int n) {
  for(int i = 0; i < l; i++) {
    for(int j = 0; j < m; j++) { 
      for(int k = 0; k < n; k++) {
        c[i * n + j] += a[i * m + k] * b[k + j * n]; //c[i][j] += a[i][k] * b[k][j]
      }
    }
  }
}

int main() {
  //a of size l * m and b of size m * n
  //a @ b is of size l * n

  int L = 4; //512
  int M = 2; //256
  int N = 3; //1024

  int *a = new int[L * M];
  int *b = new int[M * N];
  int *c = new int[L * N];

  init_matrix(a, L * M);
  init_matrix(b, M * N);

  matmul(a, b, c, L, M, N);

  print_array(a, L, M);
  print_array(b, M, N);
  print_array(c, L, N);

  return 0;
}


