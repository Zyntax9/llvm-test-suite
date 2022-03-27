// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -DUSE_DEVICE_IMAGE_SCOPE %s -o %t_dev_img_scope.out
// RUN: %CPU_RUN_PLACEHOLDER %t_dev_img_scope.out
// RUN: %GPU_RUN_PLACEHOLDER %t_dev_img_scope.out
// RUN: %ACC_RUN_PLACEHOLDER %t_dev_img_scope.out

#include <CL/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

#ifdef USE_DEVICE_IMAGE_SCOPE
using props = decltype(properties{device_image_scope});
#else
using props = decltype(properties{});
#endif

device_global<int[4], props> DeviceGlobalVar;

int main() {
  queue Q;
  if (Q.is_host()) {
    std::cout << "Skipping test\n";
    return 0;
  }

  Q.single_task([=]() { DeviceGlobalVar[0] = 42; }).wait();

  int OutVal = 0;
  {
    buffer<int, 1> OutBuf(&OutVal, 1);
    Q.submit([&](handler &CGH) {
      auto OutAcc = OutBuf.get_access<access::mode::write>(CGH);
      CGH.single_task([=]() { OutAcc[0] = DeviceGlobalVar[0]; });
    });
  }
  assert(OutVal == 42 && "Unexpected value read from device_global");
  return 0;
}
