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

  Q.single_task([=]() {
     DeviceGlobalVar1 = 42;
     DeviceGlobalVar2 = 1234;
   }).wait();

  int Out1 = 0;
  Q.memcpy(&Out1, DeviceGlobalVar1).wait();
  int Out2 = 0;
  Q.submit([&](handler &CGH) { CGH.memcpy(&Out2, DeviceGlobalVar2); }).wait();
  assert(Out1 == 42 && "First read value does not match.");
  assert(Out2 == 1234 && "Second read value does not match.");
  return 0;
}
