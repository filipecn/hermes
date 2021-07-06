/*
 * Copyright (c) 2017 FilipeCN
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#ifndef HERMES_LOG_DEBUG_H
#define HERMES_LOG_DEBUG_H

#include <hermes/common/defs.h>
#include <hermes/logging/logging.h>
#include <cmath>
#include <iostream>
#include <sstream>

#ifndef HERMES_DEBUG
#define HERMES_DEBUG
#endif

#ifndef CHECKS_ENABLED
#define CHECKS_ENABLED
#endif

#ifndef ASSERTIONS_ENABLED
#define ASSERTIONS_ENABLED
#endif

#ifndef INFO_ENABLED
#define INFO_ENABLED
#endif

// *********************************************************************************************************************
//                                                                                                              UTILS
// *********************************************************************************************************************
#ifndef LOG_LOCATION
#define LOG_LOCATION "[" << __FILE__ << "][" << __LINE__ << "]"
#endif
// *********************************************************************************************************************
//                                                                                               COMPILATION WARNINGS
// *********************************************************************************************************************
#ifndef HERMES_UNUSED_VARIABLE
#define HERMES_UNUSED_VARIABLE(x) ((void)x)
#endif

#ifndef HERMES_NOT_IMPLEMENTED
#define HERMES_NOT_IMPLEMENTED \
  hermes::Log::warn("[{}][{}][{}] calling not implemented function.", __FILE__, __LINE__, __FUNCTION__);
#endif
// *********************************************************************************************************************
//                                                                                                         DEBUG MODE
// *********************************************************************************************************************
#ifdef HERMES_DEBUG
#define HERMES_DEBUG_CODE(CODE_CONTENT) {CODE_CONTENT}
#else
#define HERMES_DEBUG_CODE(CODE_CONTENT)
#endif
// *********************************************************************************************************************
//                                                                                                            LOGGING
// *********************************************************************************************************************
#ifdef INFO_ENABLED

#ifndef HERMES_PING
#define HERMES_PING hermes::Log::info("[{}][{}][{}]", __FILE__, __LINE__, __FUNCTION__);
#endif

#ifndef HERMES_LOG
#define HERMES_LOG(A) hermes::Log::info("[{}][{}][{}]: {}", __FILE__, __LINE__, __FUNCTION__, A);
#endif

#ifndef HERMES_LOG_WARNING
#define HERMES_LOG_WARNING(A) hermes::Log::warn("[{}][{}][{}]: {}", __FILE__, __LINE__, __FUNCTION__, A);
#endif

#ifndef HERMES_LOG_ERROR
#define HERMES_LOG_ERROR(A) hermes::Log::error("[{}][{}][{}]: {}", __FILE__, __LINE__, __FUNCTION__, A);
#endif

#ifndef HERMES_LOG_CRITICAL
#define HERMES_LOG_CRITICAL(A) hermes::Log::critical("[{}][{}][{}]: {}", __FILE__, __LINE__, __FUNCTION__, A);
#endif

#ifndef HERMES_LOG_VARIABLE
#define HERMES_LOG_VARIABLE(A) hermes::Log::info("[{}][{}][{}]: {} = {}", __FILE__, __LINE__, __FUNCTION__, #A, A);
#endif

#else

#define HERMES_PING
#define HERMES_LOG
#define HERMES_LOG_VARIABLE

#endif
// *********************************************************************************************************************
//                                                                                                             CHECKS
// *********************************************************************************************************************
#ifdef CHECKS_ENABLED

#define HERMES_CHECK_EXP(expr)                                                                                      \
  if(expr) {}                                                                                                       \
  else {                                                                                                            \
    hermes::Log::warn("[{}][{}][CHECK_EXP FAIL {}]", __FILE__, __LINE__, (#expr));                                  \
  }

#define HERMES_CHECK_EXP_WITH_LOG(expr, M)                                                                          \
  if(expr) {}                                                                                                       \
  else {                                                                                                            \
    hermes::Log::warn("[{}][{}][CHECK_EXP FAIL {}]: {}", __FILE__, __LINE__, (#expr), M);                           \
  }
#else

#define HERMES_CHECK_EXP(expr)
#define HERMES_CHECK_EXP_WITH_LOG(expr, M)

#endif // CHECKS_ENABLED
// *********************************************************************************************************************
//                                                                                                          ASSERTION
// *********************************************************************************************************************
#ifdef ASSERTIONS_ENABLED

//#define debugBreak() asm ("int 3")
#define debugBreak()

#define HERMES_ASSERT(expr)                                                                                         \
  if(expr) {}                                                                                                       \
  else {                                                                                                            \
    hermes::Log::error("[{}][{}][ASSERT FAIL {}]", __FILE__, __LINE__, #expr);                                      \
    debugBreak();                                                                                                   \
  }
#define HERMES_ASSERT_WITH_LOG(expr, M)                                                                             \
  if(expr) {}                                                                                                       \
  else {                                                                                                            \
    hermes::Log::error("[{}][{}][ASSERT FAIL {}]: {}", __FILE__, __LINE__, #expr, M);                               \
    debugBreak();                                                                                                   \
  }
#else

#define HERMES_ASSERT(expr)
#define HERMES_ASSERT_WITH_LOG(expr, M)

#endif // ASSERTIONS_ENABLED
// *********************************************************************************************************************
//                                                                                                          CODE FLOW
// *********************************************************************************************************************
#define HERMES_RETURN_IF(A, R)                                                                                      \
  if (A) {                                                                                                          \
    return R;                                                                                                       \
  }
#define HERMES_RETURN_IF_NOT(A, R)                                                                                  \
  if (!(A)) {                                                                                                       \
    return R;                                                                                                       \
  }
#define HERMES_LOG_AND_RETURN_IF_NOT(A, R, M)                                                                       \
  if (!(A)) {                                                                                                       \
    HERMES_LOG(M)                                                                                                   \
    return R;                                                                                                       \
  }
// *********************************************************************************************************************
//                                                                                                               CUDA
// *********************************************************************************************************************
#ifdef ENABLE_CUDA

#define HERMES_CHECK_CUDA(err) \
  if((err) != cudaSuccess)  { \
      HERMES_LOG_CRITICAL(cudaGetErrorString(err)) \
      cudaDeviceReset();\
      exit(99);\
  }

#define HERMES_CHECK_LAST_CUDA HERMES_CHECK_CUDA(cudaGetLastError())

inline void hermes_print_cuda_devices() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Major compute capability: %d\n", prop.major);
    printf("  Minor compute capability: %d\n", prop.minor);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("  Device can map host memory with "
           "cudaHostAlloc/cudaHostGetDevicePointer: %d\n",
           prop.canMapHostMemory);
    printf("  Clock frequency in kilohertz: %d\n", prop.clockRate);
    printf("  Compute mode (See cudaComputeMode): %d\n", prop.computeMode);
    printf("  Device can concurrently copy memory and execute a kernel: %d\n",
           prop.deviceOverlap);
    printf("  Device is integrated as opposed to discrete: %d\n",
           prop.integrated);
    printf("  Specified whether there is a run time limit on kernels: %d\n",
           prop.kernelExecTimeoutEnabled);
    printf("  Maximum size of each dimension of a grid: %d %d %d\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Maximum size of each dimension of a block: %d %d %d\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Maximum number of threads per block: %d\n",
           prop.maxThreadsPerBlock);
    printf("  Maximum pitch in bytes allowed by memory copies: %zu\n",
           prop.memPitch);
    printf("  Number of multiprocessors on device: %d\n",
           prop.multiProcessorCount);
    printf("  32-bit registers available per block: %d\n", prop.regsPerBlock);
    printf("  Shared memory available per block in bytes: %zu\n",
           prop.sharedMemPerBlock);
    printf("  Alignment requirement for textures: %zu\n", prop.textureAlignment);
    printf("  Constant memory available on device in bytes: %zu\n",
           prop.totalConstMem);
    printf("  Global memory available on device in bytes: %zu\n",
           prop.totalGlobalMem);
    printf("  Warp size in threads: %d\n", prop.warpSize);
  }
}

inline void hermes_print_cuda_memory_usage() {
  size_t free_byte;
  size_t total_byte;
  HERMES_CHECK_CUDA(cudaMemGetInfo(&free_byte, &total_byte));
  auto free_db = (double) free_byte;
  auto total_db = (double) total_byte;
  double used_db = total_db - free_db;
  printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
         used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0,
         total_db / 1024.0 / 1024.0);
}

#define CUDA_MEMORY_USAGE                                                                                           \
  {                                                                                                                 \
    std::cerr << "[INFO][" << __FILE__ << "][" << __LINE__ << "]";                                                  \
    hermes_print_cuda_memory_usage();                                                                                      \
  }
#else

#define HERMES_CHECK_CUDA(err)
#define CUDA_MEMORY_USAGE

#endif

namespace hermes {

inline void printBits(u32 n) {
  for (int i = 31; i >= 0; i--)
    if ((1 << i) & n)
      std::cout << '1';
    else
      std::cout << '0';
}
} // namespace hermes

#endif
