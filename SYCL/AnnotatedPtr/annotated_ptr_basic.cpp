// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;


int main() {
  queue Q;
  if (Q.is_host()) {
    std::cout << "Skipping test\n";
    return 0;
  }

  auto USMPtr = malloc_shared<int>(1, Q);
  auto AnnotUSMPtr = annotated_ptr(USMPtr, properties{runtime_aligned, restrict, alignment<32>});
  Q.single_task([=]() { *AnnotUSMPtr = 42; }).wait();
  assert(*AnnotUSMPtr == 42 && "Unexpected value read from annotated_ptr");
  free(USMPtr, Q);
  return 0;
}
