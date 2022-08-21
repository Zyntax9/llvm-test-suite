// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-device-code-split=per_kernel -fsycl-unnamed-lambda %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

template <int, size_t> class KernelFunctorName;
template <int, size_t> class KernelFunctionName;

template <int Dims> struct RangeCreator {};
template <> struct RangeCreator<1> {
  static sycl::range<1> get() { return {1024}; }
};
template <> struct RangeCreator<2> {
  static sycl::range<2> get() { return {1024, 1024}; }
};
template <> struct RangeCreator<3> {
  static sycl::range<3> get() { return {1024, 1024, 1024}; }
};

template <int Dims> struct KernelFunctor {
  int *OutData;

  KernelFunctor(int *Out) : OutData(Out) {}

  void operator()(sycl::item<Dims>) const {
    *OutData = sycl::ext::oneapi::experimental::this_sub_group()
                   .get_group_linear_range();
  }
};

template <typename KernelName, int Dims, size_t SubGroupSize, typename FuncT>
int runTest(sycl::queue &Q, const std::vector<size_t> &SupportedSubGroupSizes,
            int *OutData, FuncT &F) {
  auto Props = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::sub_group_size<SubGroupSize>};

  bool SupportsSize =
      std::find(SupportedSubGroupSizes.begin(), SupportedSubGroupSizes.end(),
                SubGroupSize) != SupportedSubGroupSizes.end();

  // TODO: HIP and CUDA backends do not currently throw for unsupported
  //       sub-group sizes. Remove the early-exit below once they do.
  if (!SupportsSize && (Q.get_backend() == sycl::backend::ext_oneapi_cuda ||
                        Q.get_backend() == sycl::backend::ext_oneapi_hip)) {
    std::cout
        << "Unsupported sub-group size " << SubGroupSize
        << " skipped because backend does not correctly throw an exception."
        << std::endl;
    return 0;
  }

  *OutData = 0;
  try {
    Q.submit([&](sycl::handler &CGH) {
       CGH.parallel_for<KernelName>(RangeCreator<Dims>::get(), Props, F);
     }).wait_and_throw();

    if (!SupportsSize) {
      std::cout << "Unsupported sub-group size " << SubGroupSize
                << " unexpectedly did not throw an exception." << std::endl;
      return 1;
    }

    if (*OutData != SubGroupSize) {
      std::cout << "Read sub-group size " << *OutData
                << " is not the same as the the specified sub-group size "
                << SubGroupSize << "." << std::endl;
      return 1;
    }
  } catch (sycl::exception &E) {

    if (SupportsSize) {
      std::cout
          << "Supported sub-group size " << SubGroupSize
          << " unexpectedly threw an errc::kernel_not_supported exception."
          << std::endl;
      return 1;
    }

    if (E.code() != sycl::errc::kernel_not_supported) {
      std::cout << (SupportsSize ? "S" : "Uns") << "upported sub-group size "
                << SubGroupSize
                << " unexpectedly threw a SYCL exception: " << E.what()
                << std::endl;
      return 1;
    }
  } catch (...) {
    std::cout << "Sub-group size " << SubGroupSize << " ("
              << (SupportsSize ? "" : "un")
              << "supported) threw an unexpected exception." << std::endl;
    return 1;
  }
  return 0;
}

template <int Dims>
int runFunctorTests(sycl::queue &Q, const std::vector<size_t> &SubGroupSizes,
                    int *OutData) {
  int Failures = 0;
  auto KF = KernelFunctor<Dims>{OutData};
  Failures += runTest<KernelFunctorName<Dims, 1>, Dims, 1>(Q, SubGroupSizes,
                                                           OutData, KF);
  Failures += runTest<KernelFunctorName<Dims, 8>, Dims, 8>(Q, SubGroupSizes,
                                                           OutData, KF);
  Failures += runTest<KernelFunctorName<Dims, 16>, Dims, 16>(Q, SubGroupSizes,
                                                             OutData, KF);
  Failures += runTest<KernelFunctorName<Dims, 32>, Dims, 32>(Q, SubGroupSizes,
                                                             OutData, KF);
  Failures += runTest<KernelFunctorName<Dims, 64>, Dims, 64>(Q, SubGroupSizes,
                                                             OutData, KF);
  Failures += runTest<KernelFunctorName<Dims, 128>, Dims, 128>(Q, SubGroupSizes,
                                                               OutData, KF);
  Failures += runTest<KernelFunctorName<Dims, 256>, Dims, 256>(Q, SubGroupSizes,
                                                               OutData, KF);
  Failures += runTest<KernelFunctorName<Dims, 512>, Dims, 512>(Q, SubGroupSizes,
                                                               OutData, KF);
  Failures += runTest<KernelFunctorName<Dims, 1024>, Dims, 1024>(
      Q, SubGroupSizes, OutData, KF);
  return Failures;
}

template <int Dims>
int runFunctionTests(sycl::queue &Q, const std::vector<size_t> &SubGroupSizes,
                     int *OutData) {
  int Failures = 0;
  auto F = [=](sycl::item<Dims>) {
    *OutData = sycl::ext::oneapi::experimental::this_sub_group()
                   .get_group_linear_range();
  };
  Failures += runTest<KernelFunctionName<Dims, 1>, Dims, 1>(Q, SubGroupSizes,
                                                            OutData, F);
  Failures += runTest<KernelFunctionName<Dims, 8>, Dims, 8>(Q, SubGroupSizes,
                                                            OutData, F);
  Failures += runTest<KernelFunctionName<Dims, 16>, Dims, 16>(Q, SubGroupSizes,
                                                              OutData, F);
  Failures += runTest<KernelFunctionName<Dims, 32>, Dims, 32>(Q, SubGroupSizes,
                                                              OutData, F);
  Failures += runTest<KernelFunctionName<Dims, 64>, Dims, 64>(Q, SubGroupSizes,
                                                              OutData, F);
  Failures += runTest<KernelFunctionName<Dims, 128>, Dims, 128>(
      Q, SubGroupSizes, OutData, F);
  Failures += runTest<KernelFunctionName<Dims, 256>, Dims, 256>(
      Q, SubGroupSizes, OutData, F);
  Failures += runTest<KernelFunctionName<Dims, 512>, Dims, 512>(
      Q, SubGroupSizes, OutData, F);
  Failures += runTest<KernelFunctionName<Dims, 1024>, Dims, 1024>(
      Q, SubGroupSizes, OutData, F);
  return Failures;
}

int main() {
  sycl::queue Q;
  std::vector<size_t> SubGroupSizes =
      Q.get_device().get_info<sycl::info::device::sub_group_sizes>();

  int *ReadSubGroupSize = sycl::malloc_shared<int>(1, Q);

  int Failures = 0;

  Failures += runFunctorTests<1>(Q, SubGroupSizes, ReadSubGroupSize);
  Failures += runFunctorTests<2>(Q, SubGroupSizes, ReadSubGroupSize);
  Failures += runFunctorTests<3>(Q, SubGroupSizes, ReadSubGroupSize);
  Failures += runFunctionTests<1>(Q, SubGroupSizes, ReadSubGroupSize);
  Failures += runFunctionTests<2>(Q, SubGroupSizes, ReadSubGroupSize);
  Failures += runFunctionTests<3>(Q, SubGroupSizes, ReadSubGroupSize);

  sycl::free(ReadSubGroupSize, Q);
  return Failures;
}
