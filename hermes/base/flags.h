/* Copyright (c) 2025, FilipeCN.
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/// \file   flags.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2025-07-25
/// \brief  Support for boolean operations for enums.

#pragma once

#include <hermes/core/types.h>

namespace hermes {

template <typename FlagBitsType> struct FlagTraits {
  static HERMES_CONST_OR_CONSTEXPR bool is_bitmask = false;
};

/// Add boolean operations to enums.
/// \tparam BitType Enum type.
template <typename BitType> class Flags {
public:
  using BitsType = BitType;
  using MaskType = typename std::underlying_type<BitType>::type;

  //                                                              constructors

  HERMES_CONSTEXPR Flags() HERMES_NOEXCEPT : mask_(0) {}

  HERMES_CONSTEXPR Flags(BitType bit) HERMES_NOEXCEPT
      : mask_(static_cast<MaskType>(bit)) {}

  HERMES_CONSTEXPR
  Flags(const Flags<BitType> &rhs) HERMES_NOEXCEPT = default;

  HERMES_CONSTEXPR explicit Flags(MaskType flags) HERMES_NOEXCEPT
      : mask_(flags) {}

  //                                                          logical operator

  HERMES_CONSTEXPR bool operator!() const HERMES_NOEXCEPT { return !mask_; }

  //                                                      relational operators

  HERMES_CONSTEXPR bool
  operator<(const Flags<BitType> &rhs) const HERMES_NOEXCEPT {
    return mask_ < rhs.mask_;
  }

  HERMES_CONSTEXPR bool
  operator<=(const Flags<BitType> &rhs) const HERMES_NOEXCEPT {
    return mask_ <= rhs.mask_;
  }

  HERMES_CONSTEXPR bool
  operator>(const Flags<BitType> &rhs) const HERMES_NOEXCEPT {
    return mask_ > rhs.mask_;
  }

  HERMES_CONSTEXPR bool
  operator>=(const Flags<BitType> &rhs) const HERMES_NOEXCEPT {
    return mask_ >= rhs.mask_;
  }

  HERMES_CONSTEXPR bool
  operator==(const Flags<BitType> &rhs) const HERMES_NOEXCEPT {
    return mask_ == rhs.mask_;
  }

  HERMES_CONSTEXPR bool
  operator!=(const Flags<BitType> &rhs) const HERMES_NOEXCEPT {
    return mask_ != rhs.mask_;
  }

  //                                                         bitwise operators

  HERMES_CONSTEXPR Flags<BitType>
  operator&(const Flags<BitType> &rhs) const HERMES_NOEXCEPT {
    return Flags<BitType>(mask_ & rhs.mask_);
  }

  HERMES_CONSTEXPR Flags<BitType>
  operator|(const Flags<BitType> &rhs) const HERMES_NOEXCEPT {
    return Flags<BitType>(mask_ | rhs.mask_);
  }

  HERMES_CONSTEXPR Flags<BitType>
  operator^(const Flags<BitType> &rhs) const HERMES_NOEXCEPT {
    return Flags<BitType>(mask_ ^ rhs.mask_);
  }

  HERMES_CONSTEXPR Flags<BitType> operator~() const HERMES_NOEXCEPT {
    return Flags<BitType>(mask_ ^ FlagTraits<BitType>::all_flags.mask_);
  }

  //                                                      assignment operators

  Flags<BitType> &
  operator=(const Flags<BitType> &rhs) HERMES_NOEXCEPT = default;

  HERMES_CONSTEXPR Flags<BitType> &
  operator|=(const Flags<BitType> &rhs) HERMES_NOEXCEPT {
    mask_ |= rhs.mask_;
    return *this;
  }

  HERMES_CONSTEXPR Flags<BitType> &
  operator&=(const Flags<BitType> &rhs) HERMES_NOEXCEPT {
    mask_ &= rhs.mask_;
    return *this;
  }

  HERMES_CONSTEXPR Flags<BitType> &
  operator^=(const Flags<BitType> &rhs) HERMES_NOEXCEPT {
    mask_ ^= rhs.mask_;
    return *this;
  }

  //                                                            cast operators

  explicit HERMES_CONSTEXPR operator bool() const HERMES_NOEXCEPT {
    return !!mask_;
  }

  explicit HERMES_CONSTEXPR operator MaskType() const HERMES_NOEXCEPT {
    return mask_;
  }

  HERMES_CONSTEXPR bool contain(BitType bit) const HERMES_NOEXCEPT;

private:
  MaskType mask_;
};

} // namespace hermes

// bitwise operators

template <typename BitType>
HERMES_CONSTEXPR hermes::Flags<BitType>
operator&(BitType bit, const hermes::Flags<BitType> &flags) HERMES_NOEXCEPT {
  return flags.operator&(bit);
}

template <typename BitType>
HERMES_CONSTEXPR hermes::Flags<BitType>
operator|(BitType bit, const hermes::Flags<BitType> &flags) HERMES_NOEXCEPT {
  return flags.operator|(bit);
}

template <typename BitType>
HERMES_CONSTEXPR hermes::Flags<BitType>
operator^(BitType bit, const hermes::Flags<BitType> &flags) HERMES_NOEXCEPT {
  return flags.operator^(bit);
}

// bitwise operators on BitType

template <typename BitType,
          typename std::enable_if<hermes::FlagTraits<BitType>::is_bitmask,
                                  bool>::type = true>
inline HERMES_CONSTEXPR hermes::Flags<BitType>
operator&(BitType lhs, BitType rhs) HERMES_NOEXCEPT {
  return hermes::Flags<BitType>(lhs) & rhs;
}

template <typename BitType,
          typename std::enable_if<hermes::FlagTraits<BitType>::is_bitmask,
                                  bool>::type = true>
inline HERMES_CONSTEXPR hermes::Flags<BitType>
operator|(BitType lhs, BitType rhs) HERMES_NOEXCEPT {
  return hermes::Flags<BitType>(lhs) | rhs;
}

template <typename BitType,
          typename std::enable_if<hermes::FlagTraits<BitType>::is_bitmask,
                                  bool>::type = true>
inline HERMES_CONSTEXPR hermes::Flags<BitType>
operator^(BitType lhs, BitType rhs) HERMES_NOEXCEPT {
  return hermes::Flags<BitType>(lhs) ^ rhs;
}

template <typename BitType,
          typename std::enable_if<hermes::FlagTraits<BitType>::is_bitmask,
                                  bool>::type = true>
inline HERMES_CONSTEXPR hermes::Flags<BitType>
operator~(BitType bit) HERMES_NOEXCEPT {
  return ~(Flags<BitType>(bit));
}

template <typename BitType>
HERMES_CONSTEXPR bool
hermes::Flags<BitType>::contain(BitType bit) const HERMES_NOEXCEPT {
  return (*this & bit) == bit;
}
