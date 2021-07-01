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
  float randomFloat(float a, float b) { return Interpolation::lerp(randomFloat(), a, b); }

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
