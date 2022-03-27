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

#ifdef USE_SYCL_DEVICE_IMAGE
device_global<int, decltype(properties{device_image_scope})>
    DeviceGlobalVar1;
device_global<int, decltype(properties{device_image_scope})>
    DeviceGlobalVar2;
#else
device_global<int> DeviceGlobalVar1;
device_global<int> DeviceGlobalVar2;
#endif

int main() {
  queue Q;
  if (Q.is_host()) {
    std::cout << "Skipping test\n";
    return 0;
  }

  int In1 = 42;
  Q.memcpy(DeviceGlobalVar1, &In1).wait();

  int In2 = 1234;
  Q.submit([&](handler &CGH) {
    CGH.memcpy(DeviceGlobalVar2, &In2);
  }).wait();

  int Out[2] = {0, 0};
  {
    buffer<int, 1> OutBuf{Out, 2};
    Q.submit([&](handler &CGH) {
      auto OutAcc = OutBuf.get_access<access::mode::write>(CGH);
      CGH.single_task([=]() {
        OutAcc[0] = DeviceGlobalVar1;
        OutAcc[1] = DeviceGlobalVar2;
      });
    });
  }
  assert(Out[0] == In1 && "First value does not match.");
  assert(Out[1] == In2 && "Second value does not match.");
  return 0;
}
