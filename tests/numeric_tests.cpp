#include "hermes/core/debug.h"
#include "hermes/io/logger.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <hermes/numeric/interval.h>
#include <hermes/numeric/math.h>
#include <hermes/numeric/matrix.h>

using namespace hermes;

TEST_CASE("Numbers") {
  SECTION("floating point") {
    REQUIRE_THAT(numbers::fract(0.1), Catch::Matchers::WithinRel(0.1, 1e-6));
    REQUIRE_THAT(numbers::fract(10.2), Catch::Matchers::WithinRel(0.2, 1e-6));
    REQUIRE_THAT(numbers::fract(-20.3), Catch::Matchers::WithinRel(-0.3, 1e-5));
  } //
  SECTION("limits") {
    REQUIRE(numeric::limits::lowest<f32>() ==
            std::numeric_limits<f32>::lowest());
    REQUIRE(numeric::limits::lowest<f64>() ==
            std::numeric_limits<f64>::lowest());
    REQUIRE(numeric::limits::greatest<f32>() ==
            std::numeric_limits<f32>::max());
    REQUIRE(numeric::limits::greatest<f64>() ==
            std::numeric_limits<f64>::max());
  } //
  SECTION("functions") {
    REQUIRE(numbers::cmp::min({9, 0, 1, 4, -1}) == -1);
    REQUIRE(numbers::cmp::max({9, 0, 1, 4, -1}) == 9);
  } //
  SECTION("fast exp") {
    REQUIRE(1 == math::fastExp(0));

    // TODO
    // real_t maxErr = 0;
    // RNG rng(6502);
    // for (int i = 0; i < 100; ++i) {
    //  real_t v = interpolation::lerp(rng.randomFloat(), -20.f, 20.f);
    //  real_t f = math::numbers::fastExp(v);
    //  real_t e = std::exp(v);
    //  real_t err = std::abs((f - e) / e);
    //  maxErr = std::max(err, maxErr);
    //  REQUIRE(err < 0.0003f);
    //} //
    SECTION("pow") {
      REQUIRE(math::pow<0>(2.f) == 1 << 0);
      REQUIRE(math::pow<1>(2.f) == 1 << 1);
      REQUIRE(math::pow<2>(2.f) == 1 << 2);
      REQUIRE(math::pow<3>(2.f) == 1 << 3);
      REQUIRE(math::pow<4>(2.f) == 1 << 4);
      REQUIRE(math::pow<5>(2.f) == 1 << 5);
      REQUIRE(math::pow<6>(2.f) == 1 << 6);
      REQUIRE(math::pow<7>(2.f) == 1 << 7);
      REQUIRE(math::pow<8>(2.f) == 1 << 8);
      REQUIRE(math::pow<9>(2.f) == 1 << 9);
      REQUIRE(math::pow<10>(2.f) == 1 << 10);
      REQUIRE(math::pow<11>(2.f) == 1 << 11);
      REQUIRE(math::pow<12>(2.f) == 1 << 12);
      REQUIRE(math::pow<13>(2.f) == 1 << 13);
      REQUIRE(math::pow<14>(2.f) == 1 << 14);
      REQUIRE(math::pow<15>(2.f) == 1 << 15);
      REQUIRE(math::pow<16>(2.f) == 1 << 16);
      REQUIRE(math::pow<17>(2.f) == 1 << 17);
      REQUIRE(math::pow<18>(2.f) == 1 << 18);
      REQUIRE(math::pow<19>(2.f) == 1 << 19);
      REQUIRE(math::pow<20>(2.f) == 1 << 20);
      REQUIRE(math::pow<21>(2.f) == 1 << 21);
      REQUIRE(math::pow<22>(2.f) == 1 << 22);
      REQUIRE(math::pow<23>(2.f) == 1 << 23);
      REQUIRE(math::pow<24>(2.f) == 1 << 24);
      REQUIRE(math::pow<25>(2.f) == 1 << 25);
      REQUIRE(math::pow<26>(2.f) == 1 << 26);
      REQUIRE(math::pow<27>(2.f) == 1 << 27);
      REQUIRE(math::pow<28>(2.f) == 1 << 28);
      REQUIRE(math::pow<29>(2.f) == 1 << 29);
    } //
  }
}

TEST_CASE("Check") { REQUIRE(!numbers::is_nan(3.f)); }

TEST_CASE("numeric") {
  real_t A[2][2] = {{0, 1}, {1, 0}};
  real_t B[2] = {3, 4};
  real_t x0 = 0, x1 = 0;
  REQUIRE(numeric::soveLinearSystem(A, B, &x0, &x1));
  REQUIRE_THAT(x0, Catch::Matchers::WithinRel(4, 1e-6));
  REQUIRE_THAT(x1, Catch::Matchers::WithinRel(3, 1e-6));
}

TEST_CASE("interval") {
  SECTION("sanity") {
    hermes::Interval<f32> a(-1, 1), b(0, 2);
    auto c = a * b;
    HERMES_UNUSED_VARIABLE(c);
    // HERMES_LOG_VARIABLE(c);
    // HERMES_LOG_VARIABLE(c.sqr());
    // HERMES_LOG_VARIABLE(c.sqrt());
  } //
}

TEST_CASE("Matrix", "[numeric]") {
  REQUIRE(sizeof(math::mat4) == sizeof(real_t) * 16);
  REQUIRE(sizeof(math::mat3) == sizeof(real_t) * 9);
  SECTION("Identity") {
    math::mat4 m;
    m.setIdentity();
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c)
        if (r == c)
          REQUIRE_THAT(m[r][c], Catch::Matchers::WithinRel(1, 1e-8));
        else
          REQUIRE_THAT(m[r][c], Catch::Matchers::WithinRel(0, 1e-8));
    REQUIRE(m.isIdentity());
    HERMES_ERROR("{}", hermes::to_string(m));
  }
  SECTION("Multiplication") {
    math::mat4 I;
    I.setIdentity();
    I = I * 2.f;
    math::mat4 a(1, 2, 3, 4,    //
                 5, 6, 7, 8,    //
                 9, 10, 11, 12, //
                 13, 14, 15, 16);
    HERMES_LOG_VARIABLE(I);
    HERMES_LOG_VARIABLE(a);
    HERMES_LOG_VARIABLE(I * a);
    HERMES_LOG_VARIABLE(a * I);
  } //
  SECTION("Sanity") {
    math::mat3 m;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        m[i][j] = i * 10 + j;
    HERMES_ERROR("{}", hermes::to_string(m));
  }
}
