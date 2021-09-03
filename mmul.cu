#include <cassert>
#include <cstdlib>
#include <iostream>

void print_array(int *m, int l) {
  for(int i = 0; i < l; i ++)
    printf("%d, ", m[i]);
  printf("\n");
}

void init_matrix(int *m, int y, int x) {
  for (int i = 0; i < y; i++) {
    for (int j = 0; j < x; j++) {
      m[i * (y+1) + j] = 1; //rand() % 100;
    }
  }
}

void matmul(int *a, int *b, int *c, int m, int n) {
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < m; j++) { 
      for(int k = 0; k < n; k++) {
        c[i * m + j] += a[i * m + k] * b[k + j * m]; //c[i][j] = a[i][k] * b[k][j]
      }
    }
  }
}

int main() {
  //a of size m * n and b of size n * m
  //a @ b is of size m * m

  int M = 2; //256
  int N = 3; //1024

  int *a = new int[M * N];
  int *b = new int[N * M];
  int *c = new int[M * M];

  init_matrix(a, M, N);
  init_matrix(b, N, M);

  matmul(a, b, c, M, N);

  print_array(a, M * N);
  print_array(b, N * M);
  print_array(c, M * M);

  return 0;
}


