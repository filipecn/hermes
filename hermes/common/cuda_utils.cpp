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
///\file cuda_utils.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-08-07
///
///\brief

#include <hermes/common/cuda_utils.h>

namespace hermes::cuda_utils {

Lock::Lock() {
  int state = 0;
#ifdef HERMES_DEVICE_ENABLED
  cudaMalloc((void **) &mutex, sizeof(int));
  cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
#endif
}

Lock::~Lock() {
#ifdef HERMES_DEVICE_ENABLED
  cudaFree(mutex);
#endif
}

HERMES_DEVICE_FUNCTION void Lock::lock() {
#ifdef HERMES_DEVICE_CODE
  while (atomicCAS(mutex, 0, 1) != 0);
#endif
}

HERMES_DEVICE_FUNCTION void Lock::unlock() {
#ifdef HERMES_DEVICE_CODE
  atomicExch(mutex, 0);
#endif
}

}