/// Copyright (c) 2021, FilipeCN.
///
/// The MIT License (MIT)
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to
/// deal in the Software without restriction, including without limitation the
/// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
/// sell copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
/// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
/// IN THE SOFTWARE.
///
///\file random_tests.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-06-28
///
///\brief

#include <catch2/catch.hpp>

#include <hermes/random/rng.h>
#include <hermes/random/noise.h>
#include <hermes/storage/array.h>

using namespace hermes;

#ifdef HERMES_DEVICE_CODE
#include <hermes/common/cuda_utils.h>
HERMES_CUDA_KERNEL(pcg)(int *result) {
  HERMES_CUDA_RETURN_IF_NOT_THREAD_0
  PCGRNG rng(1);
  rng.uniformU32();
  *result = 0;
}
#endif

TEST_CASE("PCG", "[random]") {
  HERMES_CUDA_CODE(
      UnifiedArray<int> results(1);
      HERMES_CUDA_LAUNCH_AND_SYNC((1), pcg_k, results.data())
      REQUIRE(results[0] == 0);
  )
}

TEST_CASE("Perlin", "[noise][random]") {
  SECTION("2d") {
  }
}
