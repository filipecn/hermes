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
    REQUIRE(numbers::lowest<f32>() == std::numeric_limits<f32>::lowest());
    REQUIRE(numbers::lowest<f64>() == std::numeric_limits<f64>::lowest());
    REQUIRE(numbers::greatest<f32>() == std::numeric_limits<f32>::max());
    REQUIRE(numbers::greatest<f64>() == std::numeric_limits<f64>::max());
  } //
  SECTION("functions") {
    REQUIRE(numbers::min({9, 0, 1, 4, -1}) == -1);
    REQUIRE(numbers::max({9, 0, 1, 4, -1}) == 9);
  } //
  SECTION("fast exp") {
    REQUIRE(1 == numbers::fastExp(0));

    // TODO
    // real_t maxErr = 0;
    // RNG rng(6502);
    // for (int i = 0; i < 100; ++i) {
    //  real_t v = interpolation::lerp(rng.randomFloat(), -20.f, 20.f);
    //  real_t f = numbers::fastExp(v);
    //  real_t e = std::exp(v);
    //  real_t err = std::abs((f - e) / e);
    //  maxErr = std::max(err, maxErr);
    //  REQUIRE(err < 0.0003f);
    //} //
    SECTION("pow") {
      REQUIRE(numbers::pow<0>(2.f) == 1 << 0);
      REQUIRE(numbers::pow<1>(2.f) == 1 << 1);
      REQUIRE(numbers::pow<2>(2.f) == 1 << 2);
      REQUIRE(numbers::pow<3>(2.f) == 1 << 3);
      REQUIRE(numbers::pow<4>(2.f) == 1 << 4);
      REQUIRE(numbers::pow<5>(2.f) == 1 << 5);
      REQUIRE(numbers::pow<6>(2.f) == 1 << 6);
      REQUIRE(numbers::pow<7>(2.f) == 1 << 7);
      REQUIRE(numbers::pow<8>(2.f) == 1 << 8);
      REQUIRE(numbers::pow<9>(2.f) == 1 << 9);
      REQUIRE(numbers::pow<10>(2.f) == 1 << 10);
      REQUIRE(numbers::pow<11>(2.f) == 1 << 11);
      REQUIRE(numbers::pow<12>(2.f) == 1 << 12);
      REQUIRE(numbers::pow<13>(2.f) == 1 << 13);
      REQUIRE(numbers::pow<14>(2.f) == 1 << 14);
      REQUIRE(numbers::pow<15>(2.f) == 1 << 15);
      REQUIRE(numbers::pow<16>(2.f) == 1 << 16);
      REQUIRE(numbers::pow<17>(2.f) == 1 << 17);
      REQUIRE(numbers::pow<18>(2.f) == 1 << 18);
      REQUIRE(numbers::pow<19>(2.f) == 1 << 19);
      REQUIRE(numbers::pow<20>(2.f) == 1 << 20);
      REQUIRE(numbers::pow<21>(2.f) == 1 << 21);
      REQUIRE(numbers::pow<22>(2.f) == 1 << 22);
      REQUIRE(numbers::pow<23>(2.f) == 1 << 23);
      REQUIRE(numbers::pow<24>(2.f) == 1 << 24);
      REQUIRE(numbers::pow<25>(2.f) == 1 << 25);
      REQUIRE(numbers::pow<26>(2.f) == 1 << 26);
      REQUIRE(numbers::pow<27>(2.f) == 1 << 27);
      REQUIRE(numbers::pow<28>(2.f) == 1 << 28);
      REQUIRE(numbers::pow<29>(2.f) == 1 << 29);
    } //
  }
}

TEST_CASE("Check") { REQUIRE(!Check::is_nan(3.f)); }

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
    // HERMES_LOG_VARIABLE(c);
    // HERMES_LOG_VARIABLE(c.sqr());
    // HERMES_LOG_VARIABLE(c.sqrt());
  } //
}

TEST_CASE("Matrix", "[numeric]") {
  SECTION("Identity") {
    mat4 m;
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
  SECTION("Sanity") {
    mat3 m;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        m[i][j] = i * 10 + j;
    HERMES_ERROR("{}", hermes::to_string(m));
  }
}
