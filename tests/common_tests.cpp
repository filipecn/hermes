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
///\file common_tests.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-06-26
///
///\brief

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <hermes/common/size.h>
#include <hermes/common/str.h>
#include <hermes/system/threads.h>

using namespace hermes;

TEST_CASE("size", "[common]") {
  {
    size2 s(10, 2);
    REQUIRE(s.width == 10);
    REQUIRE(s.height == 2);
    REQUIRE(s[0] == s.width);
    REQUIRE(s[1] == s.height);
    REQUIRE(s.total() == 20);
    REQUIRE(s.contains(5, 0));
    REQUIRE(!s.contains(10, 0));
  }
}

void foo() {
  using namespace std::chrono_literals;
  HERMES_PROFILE_FUNCTION();
  hermes::SystemTime::init();
  std::this_thread::sleep_for(100ms);
  for (int j = 0; j < 2; ++j) {
    HERMES_PROFILE_SCOPE("for loop");
    std::this_thread::sleep_for(200us);
  }
}
