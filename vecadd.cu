#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

void print_array(int arr[], int m) {
  for (int i = 0; i < m; i++)
    printf("%d", arr[i]);
}

void init_array(int arr[], int m) {
  for (int i = 0; i < m; i++)
    arr[i] = rand() % 100;
}

int main() {
  int n = 10;
  int *a = new int[n];

  init_array(a, n);
  print_array(a, n);
}

