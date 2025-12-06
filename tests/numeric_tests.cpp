#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <hermes/math/math.h>
#include <hermes/numeric/interpolation.h>
#include <hermes/numeric/interval.h>
#include <hermes/numeric/matrix.h>
#include <hermes/random/rng.h>

using namespace hermes;

TEST_CASE("Numbers") {
  SECTION("f32ing point") {
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

    real_t maxErr = 0;
    random::RNG rng(6502);
    for (int i = 0; i < 100; ++i) {
      real_t v = numeric::lerp(rng.randomFloat(), -20.f, 20.f);
      real_t f = math::fastExp(v);
      real_t e = std::exp(v);
      real_t err = std::abs((f - e) / e);
      maxErr = std::max(err, maxErr);
      REQUIRE(err < 0.0003f);
    } //
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

TEST_CASE("interpolation", "[numeric][interpolation]") {
  SECTION("linear") {
    { // 1D
      f32 dx = 0.01;
      auto f = [](f32 x) -> f32 { return std::cos(x) * std::sin(x); };
      random::HaltonSequence sampler;
      for (int i = 0; i < 1000; ++i) {
        auto p = sampler.randomFloat();
        REQUIRE_THAT(numeric::lerp<f32>(p, f(0), f(dx)),
                     Catch::Matchers::WithinRel(f(p * dx), 1e-3f));
      }
    }
    { // 2D
      auto f = [](f32 x, f32 y) -> f32 { return std::cos(x) * std::sin(y); };
      random::RNGSampler sampler;
      f32 dx = 0.01;
      for (int i = 0; i < 1000; ++i) {
        auto p = sampler.sample(geo::bounds::bbox2::Unit());
        REQUIRE_THAT(numeric::bilerp<f32>(p.x, p.y, f(0.00, 0.00), f(dx, 0.00),
                                          f(dx, dx), f(0.00, dx)),
                     Catch::Matchers::WithinRel(f(p.x * dx, p.y * dx), 1e-3f));
      }
      { // 3D
        // TODO
      }
    }
  }

  SECTION("monotonicCubic") {
    { // 1D test
      f32 dx = 0.01;
      auto f = [](f32 x) -> f32 { return std::cos(x) * std::sin(x); };
      for (f32 s = 0.0; s <= 1.0; s += 0.01) {
        REQUIRE_THAT(numeric::monotonicCubicInterpolate(
                         f(-1 * dx), f(0), f(1 * dx), f(2 * dx), s),
                     Catch::Matchers::WithinRel(f(s * dx), 1e-3f));
      }
    }
    { // 2D test
      f32 dx = 0.01;
      auto f = [](f32 x, f32 y) -> f32 { return std::cos(x) * std::sin(y); };
      f32 v[4][4];
      for (int s = 0; s < 4; s++)
        for (int u = 0; u < 4; u++)
          v[s][u] = f(s * dx, u * dx);
      random::RNGSampler sampler;
      for (int i = 0; i < 1000; ++i) {
        auto p = sampler.sample(geo::bounds::bbox2::Unit());
        REQUIRE_THAT(
            numeric::monotonicCubicInterpolate(v, geo::point2(p.x, p.y)),
            Catch::Matchers::WithinRel(f(dx + p.x * dx, dx + p.y * dx), 1e-3f));
      }
    }
    { // 3D test
      f32 dx = 0.01;
      auto f = [](f32 x, f32 y, f32 z) -> f32 {
        return std::cos(x) * std::sin(y) * std::sin(z);
      };
      f32 v[4][4][4];
      for (int s = 0; s < 4; s++)
        for (int u = 0; u < 4; u++)
          for (int w = 0; w < 4; w++)
            v[s][u][w] = f(s * dx, u * dx, w * dx);
      random::RNGSampler sampler;
      for (int i = 0; i < 1000; ++i) {
        auto p = sampler.sample(geo::bounds::bbox3::Unit());
        REQUIRE_THAT(
            numeric::monotonicCubicInterpolate(v, p),
            Catch::Matchers::WithinRel(
                f(dx + p.x * dx, dx + p.y * dx, dx + p.z * dx), 1e-3f));
      }
    }
  }
}
