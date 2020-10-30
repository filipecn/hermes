/*
 * Copyright (c) 2019 FilipeCN
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

#ifndef HERMES_STORAGE_TEXTURE_KERNELS_H
#define HERMES_STORAGE_TEXTURE_KERNELS_H

#include <cuda_runtime.h>
#include <hermes/common/cuda_parallel.h>

namespace hermes {

namespace cuda {

void fillTexture(float *data, float value, unsigned int w, unsigned int h);
void fillTexture(cudaPitchedPtr data, float value, unsigned int w,
                 unsigned int h, unsigned int d);
void copyToArray3D(cudaPitchedPtr data, cudaArray *array, cudaExtent size);

} // namespace cuda

} // namespace hermes

#endif
