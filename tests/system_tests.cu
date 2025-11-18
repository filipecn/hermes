#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <hermes/system/gpu.h>

using namespace hermes;

#ifdef HERMES_DEVICE_ENABLED
TEST_CASE("cuda", "[system]") { print_cuda_devices(); }
#endif
