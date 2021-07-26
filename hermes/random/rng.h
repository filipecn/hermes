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
///\file rng.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-06-28
///
///\brief
///
/// - Code for the PCGRNG was based on pbrt-v3 code:
///
/// pbrt source code is Copyright(c) 1998-2016
///                  Matt Pharr, Greg Humphreys, and Wenzel Jakob.
/// is file is part of pbrt.
/// distribution and use in source and binary forms, with or without
/// dification, are permitted provided that the following conditions are
/// t:
/// Redistributions of source code must retain the above copyright
/// notice, this list of conditions and the following disclaimer.
/// Redistributions in binary form must reproduce the above copyright
/// notice, this list of conditions and the following disclaimer in the
/// documentation and/or other materials provided with the distribution.
/// IS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
/// " AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
/// , THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
/// RTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
/// LDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
/// ECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
/// MITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
/// TA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
/// EORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
/// NCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
///  THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef HERMES_RANDOM_RNG_H
#define HERMES_RANDOM_RNG_H

#include <hermes/common/defs.h>
#include <hermes/geometry/bbox.h>
#include <hermes/numeric/numeric.h>
#include <hermes/numeric/interpolation.h>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                                RNG
// *********************************************************************************************************************
/// \brief Random Number Generator
/// Implements the "Mersenne Twister" by Makoto Matsumoto and Takuji Nishimura.
class RNG {
public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// \param seed
  explicit RNG(u32 seed = 0) { HERMES_UNUSED_VARIABLE(seed); }
  virtual ~RNG() = default;
  // *******************************************************************************************************************
  //                                                                                                         SETTINGS
  // *******************************************************************************************************************
  /// \param seed
  void setSeed(u32 seed) { HERMES_UNUSED_VARIABLE(seed); }
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  /// pseudo-random floating-point number.
  /// \return a float in the range [0, 1)
  virtual float randomFloat() { return 0.f; }
  /// pseudo-random integer number.
  /// \return int in the range [0, 2^32)
  virtual ulong randomInt() { return 0; }
  /// \brief pseudo-random integer number.
  /// \return unsigned int in the range [0, 2^32)
  virtual ulong randomUInt() { return 0; }
};

// *********************************************************************************************************************
//                                                                                 PCG pseudo-random number generator
// *********************************************************************************************************************
/// PCG RNG (O'Neill 2014)
class PCGRNG {
public:
  HERMES_DEVICE_CALLABLE PCGRNG() {}
  HERMES_DEVICE_CALLABLE PCGRNG(u64 sequence_index) { setSequence(sequence_index); }
  HERMES_DEVICE_CALLABLE void setSequence(u64 sequence_index) {
    state = 0u;
    inc = (sequence_index << 1u) | 1u;
    uniformU32();
    state += 0x853c49e6748fea9bULL;
    uniformU32();
  }
  HERMES_DEVICE_CALLABLE u32 uniformU32() {
    u64 old_state = state;
    state = old_state * 0x5851f42d4c957f2dULL + inc;
    u32 xor_shifted = (u32) (((old_state >> 18u) ^ old_state) >> 27u);
    u32 rot = (u32) (old_state >> 59u);
    return (xor_shifted >> rot) | (xor_shifted << ((~rot + 1u) & 31));
  }
  HERMES_DEVICE_CALLABLE u32 uniformU32(u32 b) {
    u32 threshold = (~b + 1u) % b;
    while (true) {
      auto r = uniformU32();
      if (r >= threshold)
        return r % b;
    }
  }
  HERMES_DEVICE_CALLABLE real_t uniformFloat() {
#ifdef HERMES_DEVICE_CODE
    return ::min(Constants::one_minus_epsilon, real_t(uniformU32() * 2.3283064365386963e-10f));
#else
    return std::min(Constants::one_minus_epsilon, real_t(uniformU32() * 2.3283064365386963e-10f));
#endif
  }
private:
  u64 state{0x853c49e6748fea9bULL}, inc{0xda3e39cb94b95bdbULL};
};

// *********************************************************************************************************************
//                                                                                                     HaltonSequence
// *********************************************************************************************************************
/// \brief Random Number Generator
///Implements the "Halton Sequence".
class HaltonSequence : public RNG {
public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// default_color constructor.
  HaltonSequence() : base(2), ind(1) {}
  /// \param b base ( > 1)
  explicit HaltonSequence(uint b) : base(b), ind(1) {}
  // *******************************************************************************************************************
  //                                                                                                         SETTINGS
  // *******************************************************************************************************************
  /// \param b base ( > 1)
  void setBase(uint b) {
    base = b;
    ind = 1;
  }
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  /// pseudo-random floating-point number.
  /// \return a float in the range [0, 1)
  float randomFloat() override {
    float result = 0.f;
    float f = 1.f;
    uint i = ind++;
    while (i > 0) {
      f /= base;
      result += f * (i % base);
      i /= base;
    }
    return result;
  }
  /// \param a lower bound
  /// \param b upper bound
  /// \return random float in the range [a,b)
  float randomFloat(float a, float b) { return interpolation::lerp(randomFloat(), a, b); }

private:
  uint base, ind;
};

// *********************************************************************************************************************
//                                                                                                         RNGSampler
// *********************************************************************************************************************
class RNGSampler {
public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// \param rng random number generator
  explicit RNGSampler(RNG *rngX = new HaltonSequence(3),
                      RNG *rngY = new HaltonSequence(5),
                      RNG *rngZ = new HaltonSequence(7))
      : rngX_(rngX), rngY_(rngY), rngZ_(rngZ) {}
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  /// Samples a 1-dimensional bbox region
  /// \param region sampling domain
  /// \return a random point inside **region**
  float sample(const bbox1 &region) {
    return rngX_->randomFloat() * region.extends() + region.lower;
  }
  /// Samples a 2-dimensional bbox region
  /// \param region sampling domain
  /// \return a random point inside **region**
  point2 sample(const bbox2 &region) {
    return point2(rngX_->randomFloat() * region.size(0) + region.lower[0],
                  rngY_->randomFloat() * region.size(1) + region.lower[1]);
  }
  /// Samples a 3-dimensional bbox region
  /// \param region sampling domain
  /// \return a random point inside **region**
  point3 sample(const bbox3 &region) {
    return point3(rngX_->randomFloat() * region.size(0) + region.lower[0],
                  rngY_->randomFloat() * region.size(1) + region.lower[1],
                  rngZ_->randomFloat() * region.size(2) + region.lower[2]);
  }

private:
  std::shared_ptr<RNG> rngX_;
  std::shared_ptr<RNG> rngY_;
  std::shared_ptr<RNG> rngZ_;
};

} // namespace hermes

#endif // HERMES_RANDOM_RNG_H
