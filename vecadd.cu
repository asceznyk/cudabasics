#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

void print_array(int arr[], int m) {
  for (int i = 0; i < m; i++)
    printf("%d, ", arr[i]);
}

void init_array(int arr[], int m) {
  for (int i = 0; i < m; i++)
    arr[i] = rand() % 100;
}

void add_array(int a[], int b[], int m[]) {
  for (int i = 0; i < m; i ++)
    b[i] = a[i] + b[i];
}

int main() {
  int n = 10;
  int *a = new int[n];
  int *b = new int[n];

  init_array(a, n);
  init_array(b, n);

  print_array(a, n);
  printf("\n");
  print_array(b, n);
  printf("\n");
  add_array(a, b, n);
  print_array(b, n);
}

