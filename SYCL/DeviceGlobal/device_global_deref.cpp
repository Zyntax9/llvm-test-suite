// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -DUSE_SYCL_DEVICE_IMAGE %s -o %t_dev_img_scope.out
// RUN: %CPU_RUN_PLACEHOLDER %t_dev_img_scope.out
// RUN: %GPU_RUN_PLACEHOLDER %t_dev_img_scope.out
// RUN: %ACC_RUN_PLACEHOLDER %t_dev_img_scope.out

#include <CL/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

struct StructWithMember {
  int x;
  int getX() { return x; }
};

struct StructWithDeref {
  StructWithMember y[1];
  StructWithMember *operator->() {
    return y;
  }
};

#ifdef USE_SYCL_DEVICE_IMAGE
device_global<StructWithMember *, decltype(properties{device_image_scope})>
    DeviceGlobalVar1;
device_global<StructWithDeref, decltype(properties{device_image_scope})>
    DeviceGlobalVar2;
#else
device_global<StructWithMember *> DeviceGlobalVar1;
device_global<StructWithDeref> DeviceGlobalVar2;
#endif

int main() {
  queue Q;
  if (Q.is_host()) {
    std::cout << "Skipping test\n";
    return 0;
  }

  StructWithMember *DGMem = malloc_device<StructWithMember>(1, Q);

  Q.single_task([=]() {
    DeviceGlobalVar1 = DGMem;
    DeviceGlobalVar1->x = 1234;
    DeviceGlobalVar2->x = 4321;
  }).wait();

  int Out[2] = {0,0};
  {
    buffer<int, 1> OutBuf{Out, 2};
    Q.submit([&](handler &CGH) {
      auto OutAcc = OutBuf.get_access<access::mode::write>(CGH);
      CGH.single_task([=]() {
        OutAcc[0] = DeviceGlobalVar1->getX();
        OutAcc[1] = DeviceGlobalVar2->getX();
      });
    });
  }
  free(DGMem, Q);

  assert(Out[0] == 1234 && "First value does not match.");
  assert(Out[1] == 4321 && "Second value does not match.");
  return 0;
}
