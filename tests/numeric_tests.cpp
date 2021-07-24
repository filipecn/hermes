#include <catch2/catch.hpp>
#include <cmath>

#include <hermes/numeric/numeric.h>
#include <hermes/numeric/e_float.h>
#include <hermes/numeric/interval.h>
#include <hermes/numeric/interpolation.h>
#include <hermes/random/rng.h>
#include <hermes/common/cuda_utils.h>

using namespace hermes;

#ifdef HERMES_DEVICE_ENABLED
HERMES_CUDA_KERNEL(testEFloat)(bool *result) {
  HERMES_CUDA_RETURN_IF_NOT_THREAD_0
  EFloat ef;
}
#endif

TEST_CASE("interval") {
  HERMES_NOT_IMPLEMENTED
}

TEST_CASE("EFloat") {
  EFloat ef;
}

TEST_CASE("interpolation", "[numeric][interpolation]") {
  SECTION("linear") {
    { // 1D
      float dx = 0.01;
      auto f = [](float x) -> float { return std::cos(x) * std::sin(x); };
      HaltonSequence sampler;
      for (int i = 0; i < 1000; ++i) {
        auto p = sampler.randomFloat();
        REQUIRE(interpolation::lerp<float>(p, f(0), f(dx)) == Approx(f(p * dx)).margin(1e-6));
      }
    }
    { // 2D
      auto f = [](float x, float y) -> float { return std::cos(x) * std::sin(y); };
      RNGSampler sampler;
      float dx = 0.01;
      for (int i = 0; i < 1000; ++i) {
        auto p = sampler.sample(bbox2::unitBox());
        REQUIRE(interpolation::bilerp<float>(p.x, p.y, f(0.00, 0.00), f(dx, 0.00), f(dx, dx),
                                             f(0.00, dx)) ==
            Approx(f(p.x * dx, p.y * dx)).margin(1e-6));
      }
      { // 3D
        // TODO
      }
    }
  }

  SECTION("monotonicCubic") {
    { // 1D test
      float dx = 0.01;
      auto f = [](float x) -> float { return std::cos(x) * std::sin(x); };
      for (float s = 0.0; s <= 1.0; s += 0.01) {
        REQUIRE(interpolation::monotonicCubicInterpolate(f(-1 * dx), f(0), f(1 * dx),
                                                         f(2 * dx),
                                                         s) == Approx(f(s * dx)).margin(1e-7));
      }
    }
    { // 2D test
      float dx = 0.01;
      auto f = [](float x, float y) -> float { return std::cos(x) * std::sin(y); };
      float v[4][4];
      for (int s = 0; s < 4; s++)
        for (int u = 0; u < 4; u++)
          v[s][u] = f(s * dx, u * dx);
      RNGSampler sampler;
      for (int i = 0; i < 1000; ++i) {
        auto p = sampler.sample(bbox2::unitBox());
        REQUIRE(interpolation::monotonicCubicInterpolate(v, point2(p.x, p.y)) ==
            Approx(f(dx + p.x * dx, dx + p.y * dx)).margin(1e-7));
      }
    }
    { // 3D test
      float dx = 0.01;
      auto f = [](float x, float y, float z) -> float {
        return std::cos(x) * std::sin(y) * std::sin(z);
      };
      float v[4][4][4];
      for (int s = 0; s < 4; s++)
        for (int u = 0; u < 4; u++)
          for (int w = 0; w < 4; w++)
            v[s][u][w] = f(s * dx, u * dx, w * dx);
      RNGSampler sampler;
      for (int i = 0; i < 1000; ++i) {
        auto p = sampler.sample(bbox3::unitBox());
        REQUIRE(interpolation::monotonicCubicInterpolate(v, p) ==
            Approx(f(dx + p.x * dx, dx + p.y * dx, dx + p.z * dx))
                .margin(1e-7));
      }
    }
  }
}

/*
TEST_CASE("Grid1", "[numeric][grid]") {
  SECTION("Sanity") {
    Grid1<float> g(10, 0.1, 1);
    for (auto e : g.accessor())
      e.value = e.index * 10;
    Grid1<float> gg;
    gg = g;
    REQUIRE(g.resolution() == gg.resolution());
    REQUIRE(g.spacing() == gg.spacing());
    REQUIRE(g.origin() == gg.origin());
    for (auto e : g.accessor())
      REQUIRE(e.value == Approx(gg.accessor()[e.index]).margin(1e-8));
    g = 7.3f;
    for (auto e : g.accessor())
      REQUIRE(e.value == Approx(7.3).margin(1e-8));
    Array1<float> a(10);
    a = 1.f;
    g = a;
    for (auto e : g.accessor())
      REQUIRE(e.value == Approx(1).margin(1e-8));
    gg = g;
    for (auto e : gg.accessor())
      REQUIRE(e.value == Approx(1).margin(1e-8));
  }//
  SECTION("GridIterator") {
    Grid1<float> g(10);
    g.setSpacing(0.1);
    for (auto e : g.accessor()) {
      REQUIRE(e.worldPosition() == Approx(e.index * 0.1).margin(1e-7));
      REQUIRE(e.region().size() == Approx(0.1).margin(1e-7));
      e.value = e.index * 10;
    }
    for (auto e : g.accessor())
      REQUIRE(e.value == Approx(e.index * 10).margin(1e-8));
  }//
  SECTION("GridAccessor") {
    float dx = 0.01;
    Grid1<float> g(10, dx);
    auto f = [](real_t x) -> float { return std::sin(x); };
    for (u64 i = 0; i < g.resolution(); ++i) {
      g.accessor()[i] = f(g.accessor().worldPosition(i));
      REQUIRE(g.accessor()[i] ==
          Approx(f(g.accessor().worldPosition(i))).margin(1e-8));
    }
    // cell info
    REQUIRE(g.accessor().cellRegion(4).extends() == Approx(dx).margin(1e-8));
    REQUIRE(g.accessor().cellIndex(5.5 * dx) == 5);
    REQUIRE(g.accessor().cellPosition(8.7 * dx) == Approx(0.7).margin(1e-6));
    SECTION("CLAMP_TO_EDGE + LINEAR INTERPOLATION") {
      auto acc =
          g.accessor(AddressMode::CLAMP_TO_EDGE, interpolationMode::LINEAR);
      REQUIRE(acc[-5] == Approx(f(acc.worldPosition(0))).margin(1e-8));
      REQUIRE(acc[5] == Approx(f(acc.worldPosition(5))).margin(1e-8));
      REQUIRE(acc[11] == Approx(f(acc.worldPosition(9))).margin(1e-8));
      for (u64 j = 0; j < 9; ++j) {
        RNGSampler sampler;
        for (int i = 0; i < 1000; ++i) {
          auto p = sampler.sample(acc.cellRegion(j));
          REQUIRE(acc(p) == Approx(f(p)).margin(1e-5));
        }
      }
    }//
    SECTION("CLAMP_TO_EDGE + MONOTONIC CUBIC INTERPOLATION") {
      auto acc = g.accessor(AddressMode::CLAMP_TO_EDGE,
                            interpolationMode::MONOTONIC_CUBIC);
      for (u64 j = 1; j < 8; ++j) {
        RNGSampler sampler;
        for (int i = 0; i < 1000; ++i) {
          auto p = sampler.sample(acc.cellRegion(j));
          REQUIRE(acc(p) == Approx(f(p)).margin(1e-7));
        }
      }
    }//
    SECTION("BORDER + LINEAR INTERPOLATION") {
      auto acc = g.accessor(AddressMode::BORDER, interpolationMode::LINEAR);
      REQUIRE(acc[-5] == Approx(0).margin(1e-8));
      REQUIRE(acc[11] == Approx(0).margin(1e-8));
      for (u64 j = 0; j < 9; ++j) {
        RNGSampler sampler;
        for (int i = 0; i < 1; ++i) {
          auto p = sampler.sample(acc.cellRegion(j));
          REQUIRE(acc(p) == Approx(f(p)).margin(1e-5));
        }
      }
    }//
    SECTION("BORDER + MONOTONIC CUBIC INTERPOLATION") {
      auto acc =
          g.accessor(AddressMode::BORDER, interpolationMode::MONOTONIC_CUBIC);
      for (u64 j = 1; j < 8; ++j) {
        RNGSampler sampler;
        for (int i = 0; i < 1000; ++i) {
          auto p = sampler.sample(acc.cellRegion(j));
          REQUIRE(acc(p) == Approx(f(p)).margin(1e-7));
        }
      }
    }//
  }//
  SECTION("ConstGridAccessor") {
    float dx = 0.01;
    Grid1<float> g(10, dx);
    auto f = [](real_t x) -> float { return std::sin(x) * std::cos(x); };
    for (u64 i = 0; i < g.resolution(); ++i) {
      g.accessor()[i] = f(g.accessor().worldPosition(i));
      REQUIRE(g.accessor()[i] ==
          Approx(f(g.accessor().worldPosition(i))).margin(1e-8));
    }
    auto cf = [](const Grid1<float> &g, float dx, const std::function<float(real_t)> &f) {
      // cell info
      REQUIRE(g.accessor().cellRegion(4).extends() == Approx(dx).margin(1e-8));
      REQUIRE(g.accessor().cellIndex(5.5 * dx) == 5);
      REQUIRE(g.accessor().cellPosition(3.1 * dx) == Approx(0.1).margin(1e-6));
      REQUIRE(g.accessor().cellPosition(8.7 * dx) == Approx(0.7).margin(1e-6));
      SECTION("CLAMP_TO_EDGE + LINEAR INTERPOLATION") {
      auto acc =
          g.accessor(AddressMode::CLAMP_TO_EDGE, interpolationMode::LINEAR);
      REQUIRE(acc[-5] == Approx(f(acc.worldPosition(0))).margin(1e-8));
      REQUIRE(acc[5] == Approx(f(acc.worldPosition(5))).margin(1e-8));
      REQUIRE(acc[11] == Approx(f(acc.worldPosition(9))).margin(1e-8));
      for (u64 j = 0; j < 9; ++j) {
        RNGSampler sampler;
        for (int i = 0; i < 1000; ++i) {
          auto p = sampler.sample(acc.cellRegion(j));
          REQUIRE(acc(p) == Approx(f(p)).margin(1e-5));
        }
      }
      auto cacc = g.accessor();
      for (const auto &v : cacc) {
      }
    }//
      SECTION("CLAMP_TO_EDGE + MONOTONIC CUBIC INTERPOLATION") {
      auto acc = g.accessor(AddressMode::CLAMP_TO_EDGE,
                            interpolationMode::MONOTONIC_CUBIC);
      for (u64 j = 1; j < 8; ++j) {
        RNGSampler sampler;
        for (int i = 0; i < 1000; ++i) {
          auto p = sampler.sample(acc.cellRegion(j));
          REQUIRE(acc(p) == Approx(f(p)).margin(1e-7));
        }
      }
    }//
      SECTION("BORDER + LINEAR INTERPOLATION") {
      auto acc = g.accessor(AddressMode::BORDER, interpolationMode::LINEAR);
      REQUIRE(acc[-5] == Approx(0).margin(1e-8));
      REQUIRE(acc[11] == Approx(0).margin(1e-8));
      for (u64 j = 0; j < 9; ++j) {
        RNGSampler sampler;
        for (int i = 0; i < 1; ++i) {
          auto p = sampler.sample(acc.cellRegion(j));
          REQUIRE(acc(p) == Approx(f(p)).margin(1e-5));
        }
      }
    }//
      SECTION("BORDER + MONOTONIC CUBIC INTERPOLATION") {
      auto acc =
          g.accessor(AddressMode::BORDER, interpolationMode::MONOTONIC_CUBIC);
      for (u64 j = 1; j < 8; ++j) {
        RNGSampler sampler;
        for (int i = 0; i < 1000; ++i) {
          auto p = sampler.sample(acc.cellRegion(j));
          REQUIRE(acc(p) == Approx(f(p)).margin(1e-7));
        }
      }
    }//
    };
    cf(g, dx, f);
  }//
  SECTION("Linear interpolation") {
    SECTION("constant function") {
      Grid1<float> g(10);
      g = 4.f;
      auto acc = g.accessor(AddressMode::CLAMP_TO_EDGE, interpolationMode::LINEAR);
      REQUIRE(acc(0.5) == Approx(4.f).margin(1e-6));
      REQUIRE(acc(-0.5) == Approx(4.f).margin(1e-6));
      auto acc2 = g.accessor(AddressMode::BORDER, interpolationMode::LINEAR);
      REQUIRE(acc2(0.5) == Approx(4.f).margin(1e-6));
      REQUIRE(acc2(-0.5) == Approx(2.f).margin(1e-6));
      REQUIRE(acc2(9.5) == Approx(2.f).margin(1e-6));
    }
  }
}

TEST_CASE("Grid2", "[numeric][grid]") {
  SECTION("Sanity") {
    Grid2<float> g(size2(10, 10), vec2(0.1, 0.1), point2(1, 2));
    for (auto e : g.accessor())
      e.value = e.index.i * 10 + e.index.j;
    Grid2<float> gg;
    gg = g;
    REQUIRE(g.resolution() == gg.resolution());
    REQUIRE(g.spacing() == gg.spacing());
    REQUIRE(g.origin() == gg.origin());
    for (auto e : g.accessor())
      REQUIRE(e.value == Approx(gg.accessor()[e.index]).margin(1e-8));
    g = 7.3f;
    for (auto e : g.accessor())
      REQUIRE(e.value == Approx(7.3).margin(1e-8));
    Array2<float> a(size2(10, 10));
    a = 1.f;
    g = a;
    for (auto e : g.accessor())
      REQUIRE(e.value == Approx(1).margin(1e-8));
    gg = g;
    for (auto e : gg.accessor())
      REQUIRE(e.value == Approx(1).margin(1e-8));
  }//
  SECTION("Apply") {
    Grid2<f32> grid({100, 100});
    grid.apply([](const hermes::point2 &p) -> f32 { return p.x * 100 + p.y; });
    for (auto g : grid.accessor())
      REQUIRE(g.value == Approx(g.worldPosition().x * 100 + g.worldPosition().y));
    grid = [](const hermes::point2 &p) -> f32 { return p.y * 100 + p.x; };
    for (auto g : grid.accessor())
      REQUIRE(g.value == Approx(g.worldPosition().y * 100 + g.worldPosition().x));
  }//
  SECTION("GridIterator") {
    Grid2<float> g(size2(10, 10));
    g.setSpacing(vec2(0.1, 0.2));
    for (auto e : g.accessor()) {
      REQUIRE(e.worldPosition().x == Approx(e.index.i * 0.1).margin(1e-7));
      REQUIRE(e.worldPosition().y == Approx(e.index.j * 0.2).margin(1e-7));
      REQUIRE(e.region().size(0) == Approx(0.1).margin(1e-7));
      REQUIRE(e.region().size(1) == Approx(0.2).margin(1e-7));
      REQUIRE(e.flatIndex() == e.index.j * g.resolution().width + e.index.i);
      e.value = e.index.i * 10 + e.index.j;
    }
    for (auto e : g.accessor())
      REQUIRE(e.value == Approx(e.index.i * 10 + e.index.j).margin(1e-8));
  }//
  SECTION("GridAccessor") {
    float dx = 0.01;
    Grid2<float> g(size2(10, 10), vec2(dx, dx));
    auto f = [](point2 p) -> float { return std::sin(p.x) * std::cos(p.y); };
    for (auto ij : Index2Range<i32>(g.resolution())) {
      g.accessor()[ij] = f(g.accessor().worldPosition(ij));
      REQUIRE(g.accessor()[ij] ==
          Approx(f(g.accessor().worldPosition(ij))).margin(1e-8));
    }
    // cell info
    REQUIRE(g.accessor().cellRegion(index2(4, 4)).extends().x ==
        Approx(dx).margin(1e-8));
    REQUIRE(g.accessor().cellRegion(index2(4, 4)).extends().y ==
        Approx(dx).margin(1e-8));
    REQUIRE(g.accessor().cellIndex(point2(5.5 * dx, 7.8 * dx)) == index2(5, 7));
    REQUIRE(g.accessor().cellPosition(point2(3.1 * dx, 8.7 * dx)).x ==
        Approx(0.1).margin(1e-6));
    REQUIRE(g.accessor().cellPosition(point2(3.1 * dx, 8.7 * dx)).y ==
        Approx(0.7).margin(1e-6));
    SECTION("CLAMP_TO_EDGE + LINEAR INTERPOLATION") {
      auto acc =
          g.accessor(AddressMode::CLAMP_TO_EDGE, interpolationMode::LINEAR);
      REQUIRE(acc[index2(-5, 5)] ==
          Approx(f(acc.worldPosition(index2(0, 5)))).margin(1e-8));
      REQUIRE(acc[index2(5, -5)] ==
          Approx(f(acc.worldPosition(index2(5, 0)))).margin(1e-8));
      REQUIRE(acc[index2(11, 5)] ==
          Approx(f(acc.worldPosition(index2(9, 5)))).margin(1e-8));
      REQUIRE(acc[index2(5, 11)] ==
          Approx(f(acc.worldPosition(index2(5, 9)))).margin(1e-8));
      for (index2 ij : Index2Range<i32>(size2(9, 9))) {
        RNGSampler sampler;
        for (int i = 0; i < 1000; ++i) {
          auto p = sampler.sample(acc.cellRegion(ij));
          REQUIRE(acc(p) == Approx(f(p)).margin(1e-5));
        }
      }
    }//
    SECTION("CLAMP_TO_EDGE + MONOTONIC CUBIC INTERPOLATION") {
      auto acc = g.accessor(AddressMode::CLAMP_TO_EDGE,
                            interpolationMode::MONOTONIC_CUBIC);
      for (index2 ij : Index2Range<i32>(index2(1, 1), index2(8, 8))) {
        RNGSampler sampler;
        for (int i = 0; i < 1000; ++i) {
          auto p = sampler.sample(acc.cellRegion(ij));
          REQUIRE(acc(p) == Approx(f(p)).margin(1e-7));
        }
      }
    }//
    SECTION("BORDER + LINEAR INTERPOLATION") {
      auto acc = g.accessor(AddressMode::BORDER, interpolationMode::LINEAR);
      REQUIRE(acc[index2(-5, 5)] == Approx(0).margin(1e-8));
      REQUIRE(acc[index2(5, -5)] == Approx(0).margin(1e-8));
      REQUIRE(acc[index2(11, 5)] == Approx(0).margin(1e-8));
      REQUIRE(acc[index2(5, 11)] == Approx(0).margin(1e-8));
      for (index2 ij : Index2Range<i32>(size2(9, 9))) {
        RNGSampler sampler;
        for (int i = 0; i < 1; ++i) {
          auto p = sampler.sample(acc.cellRegion(ij));
          REQUIRE(acc(p) == Approx(f(p)).margin(1e-5));
        }
      }
    }//
    SECTION("BORDER + MONOTONIC CUBIC INTERPOLATION") {
      auto acc =
          g.accessor(AddressMode::BORDER, interpolationMode::MONOTONIC_CUBIC);
      for (index2 ij : Index2Range<i32>(index2(1, 1), index2(8, 8))) {
        RNGSampler sampler;
        for (int i = 0; i < 1000; ++i) {
          auto p = sampler.sample(acc.cellRegion(ij));
          REQUIRE(acc(p) == Approx(f(p)).margin(1e-7));
        }
      }
    }//
  }//
  SECTION("ConstGridAccessor") {
    float dx = 0.01;
    Grid2<float> g(size2(10, 10), vec2(dx, dx));
    auto f = [](point2 p) -> float { return std::sin(p.x) * std::cos(p.y); };
    for (auto ij : Index2Range<i32>(g.resolution())) {
      g.accessor()[ij] = f(g.accessor().worldPosition(ij));
      REQUIRE(g.accessor()[ij] ==
          Approx(f(g.accessor().worldPosition(ij))).margin(1e-8));
    }
    auto cf = [](const Grid2<float> &g, float dx, const std::function<float(point2)> &f) {
      // cell info
      REQUIRE(g.accessor().cellRegion(index2(4, 4)).extends().x ==
          Approx(dx).margin(1e-8));
      REQUIRE(g.accessor().cellRegion(index2(4, 4)).extends().y ==
          Approx(dx).margin(1e-8));
      REQUIRE(g.accessor().cellIndex(point2(5.5 * dx, 7.8 * dx)) == index2(5, 7));
      REQUIRE(g.accessor().cellPosition(point2(3.1 * dx, 8.7 * dx)).x ==
          Approx(0.1).margin(1e-6));
      REQUIRE(g.accessor().cellPosition(point2(3.1 * dx, 8.7 * dx)).y ==
          Approx(0.7).margin(1e-6));
      SECTION("CLAMP_TO_EDGE + LINEAR INTERPOLATION") {
      auto acc =
          g.accessor(AddressMode::CLAMP_TO_EDGE, interpolationMode::LINEAR);
      REQUIRE(acc[index2(-5, 5)] ==
          Approx(f(acc.worldPosition(index2(0, 5)))).margin(1e-8));
      REQUIRE(acc[index2(5, -5)] ==
          Approx(f(acc.worldPosition(index2(5, 0)))).margin(1e-8));
      REQUIRE(acc[index2(11, 5)] ==
          Approx(f(acc.worldPosition(index2(9, 5)))).margin(1e-8));
      REQUIRE(acc[index2(5, 11)] ==
          Approx(f(acc.worldPosition(index2(5, 9)))).margin(1e-8));
      for (index2 ij : Index2Range<i32>(size2(9, 9))) {
        RNGSampler sampler;
        for (int i = 0; i < 1000; ++i) {
          auto p = sampler.sample(acc.cellRegion(ij));
          REQUIRE(acc(p) == Approx(f(p)).margin(1e-5));
        }
      }
      auto cacc = g.accessor();
      for (const auto &v : cacc) {
      }
    }//
      SECTION("CLAMP_TO_EDGE + MONOTONIC CUBIC INTERPOLATION") {
      auto acc = g.accessor(AddressMode::CLAMP_TO_EDGE,
                            interpolationMode::MONOTONIC_CUBIC);
      for (index2 ij : Index2Range<i32>(index2(1, 1), index2(8, 8))) {
        RNGSampler sampler;
        for (int i = 0; i < 1000; ++i) {
          auto p = sampler.sample(acc.cellRegion(ij));
          REQUIRE(acc(p) == Approx(f(p)).margin(1e-7));
        }
      }
    }//
      SECTION("BORDER + LINEAR INTERPOLATION") {
      auto acc = g.accessor(AddressMode::BORDER, interpolationMode::LINEAR);
      REQUIRE(acc[index2(-5, 5)] == Approx(0).margin(1e-8));
      REQUIRE(acc[index2(5, -5)] == Approx(0).margin(1e-8));
      REQUIRE(acc[index2(11, 5)] == Approx(0).margin(1e-8));
      REQUIRE(acc[index2(5, 11)] == Approx(0).margin(1e-8));
      for (index2 ij : Index2Range<i32>(size2(9, 9))) {
        RNGSampler sampler;
        for (int i = 0; i < 1; ++i) {
          auto p = sampler.sample(acc.cellRegion(ij));
          REQUIRE(acc(p) == Approx(f(p)).margin(1e-5));
        }
      }
    }//
      SECTION("BORDER + MONOTONIC CUBIC INTERPOLATION") {
      auto acc =
          g.accessor(AddressMode::BORDER, interpolationMode::MONOTONIC_CUBIC);
      for (index2 ij : Index2Range<i32>(index2(1, 1), index2(8, 8))) {
        RNGSampler sampler;
        for (int i = 0; i < 1000; ++i) {
          auto p = sampler.sample(acc.cellRegion(ij));
          REQUIRE(acc(p) == Approx(f(p)).margin(1e-7));
        }
      }
    }//
    };
    cf(g, dx, f);
  }//
}

TEST_CASE("VectorGrid2", "[numeric][grid]") {
  SECTION("constructors operators") {
    VectorGrid2<float> empty;
    REQUIRE(empty.resolution() == size2(0, 0));
    empty = VectorGrid2<float>(size2(10, 10), vec2(1));
    REQUIRE(empty.resolution() == size2(10, 10));
    auto g = empty;
    REQUIRE(g.resolution() == size2(10, 10));
    auto g2 = VectorGrid2<float>(size2(7, 7), vec2(1));
    REQUIRE(g2.resolution() == size2(7, 7));
    std::vector<VectorGrid2<float>> gs;
    gs.emplace_back(size2(10, 10), vec2(1));
    gs.emplace_back(size2(7, 7), vec2(1));
    std::vector<VectorGrid2<float>> gs2 = gs;
  }//
  SECTION("CELL CENTERED") {
    VectorGrid2<float> vg(size2(10, 10), vec2(1));
    REQUIRE(vg.resolution() == size2(10, 10));
    REQUIRE(vg.u().resolution() == size2(10, 10));
    REQUIRE(vg.v().resolution() == size2(10, 10));
    REQUIRE(vg.origin().x == Approx(0).margin(1e-8));
    REQUIRE(vg.origin().y == Approx(0).margin(1e-8));
    REQUIRE(vg.spacing().x == Approx(1).margin(1e-8));
    REQUIRE(vg.spacing().y == Approx(1).margin(1e-8));
    REQUIRE(vg.u().origin().x == Approx(0).margin(1e-8));
    REQUIRE(vg.u().origin().y == Approx(0).margin(1e-8));
    REQUIRE(vg.v().origin().x == Approx(0).margin(1e-8));
    REQUIRE(vg.v().origin().y == Approx(0).margin(1e-8));
    VectorGrid2<float> vg2;
    vg2 = vg;
    REQUIRE(vg2.resolution() == size2(10, 10));
    REQUIRE(vg2.u().resolution() == size2(10, 10));
    REQUIRE(vg2.v().resolution() == size2(10, 10));
    REQUIRE(vg2.origin().x == Approx(0).margin(1e-8));
    REQUIRE(vg2.origin().y == Approx(0).margin(1e-8));
    REQUIRE(vg2.spacing().x == Approx(1).margin(1e-8));
    REQUIRE(vg2.spacing().y == Approx(1).margin(1e-8));
    REQUIRE(vg2.u().origin().x == Approx(0).margin(1e-8));
    REQUIRE(vg2.u().origin().y == Approx(0).margin(1e-8));
    REQUIRE(vg2.v().origin().x == Approx(0).margin(1e-8));
    REQUIRE(vg2.v().origin().y == Approx(0).margin(1e-8));
    vg.setGridType(VectorGridType::STAGGERED);
    REQUIRE(vg.resolution() == size2(10, 10));
    REQUIRE(vg.u().resolution() == size2(11, 10));
    REQUIRE(vg.v().resolution() == size2(10, 11));
    REQUIRE(vg.origin().x == Approx(0).margin(1e-8));
    REQUIRE(vg.origin().y == Approx(0).margin(1e-8));
    REQUIRE(vg.spacing().x == Approx(1).margin(1e-8));
    REQUIRE(vg.spacing().y == Approx(1).margin(1e-8));
    REQUIRE(vg.u().origin().x == Approx(-0.5).margin(1e-8));
    REQUIRE(vg.u().origin().y == Approx(0).margin(1e-8));
    REQUIRE(vg.v().origin().x == Approx(0).margin(1e-8));
    REQUIRE(vg.v().origin().y == Approx(-0.5).margin(1e-8));
  }//
  SECTION("STAGGERED") {
    VectorGrid2<float> vg(VectorGridType::STAGGERED);
    vg.setResolution(size2(10, 10));
    REQUIRE(vg.resolution() == size2(10, 10));
    REQUIRE(vg.u().resolution() == size2(11, 10));
    REQUIRE(vg.v().resolution() == size2(10, 11));
    REQUIRE(vg.origin().x == Approx(0).margin(1e-8));
    REQUIRE(vg.origin().y == Approx(0).margin(1e-8));
    REQUIRE(vg.spacing().x == Approx(1).margin(1e-8));
    REQUIRE(vg.spacing().y == Approx(1).margin(1e-8));
    REQUIRE(vg.u().origin().x == Approx(-0.5).margin(1e-8));
    REQUIRE(vg.u().origin().y == Approx(0).margin(1e-8));
    REQUIRE(vg.v().origin().x == Approx(0).margin(1e-8));
    REQUIRE(vg.v().origin().y == Approx(-0.5).margin(1e-8));
    VectorGrid2<float> vg2;
    vg2 = vg;
    REQUIRE(vg2.resolution() == size2(10, 10));
    REQUIRE(vg2.u().resolution() == size2(11, 10));
    REQUIRE(vg2.v().resolution() == size2(10, 11));
    REQUIRE(vg2.origin().x == Approx(0).margin(1e-8));
    REQUIRE(vg2.origin().y == Approx(0).margin(1e-8));
    REQUIRE(vg2.spacing().x == Approx(1).margin(1e-8));
    REQUIRE(vg2.spacing().y == Approx(1).margin(1e-8));
    REQUIRE(vg2.u().origin().x == Approx(-0.5).margin(1e-8));
    REQUIRE(vg2.u().origin().y == Approx(0).margin(1e-8));
    REQUIRE(vg2.v().origin().x == Approx(0).margin(1e-8));
    REQUIRE(vg2.v().origin().y == Approx(-0.5).margin(1e-8));
    vg.setGridType(VectorGridType::CELL_CENTERED);
    REQUIRE(vg.resolution() == size2(10, 10));
    REQUIRE(vg.u().resolution() == size2(10, 10));
    REQUIRE(vg.v().resolution() == size2(10, 10));
    REQUIRE(vg.origin().x == Approx(0).margin(1e-8));
    REQUIRE(vg.origin().y == Approx(0).margin(1e-8));
    REQUIRE(vg.spacing().x == Approx(1).margin(1e-8));
    REQUIRE(vg.spacing().y == Approx(1).margin(1e-8));
    REQUIRE(vg.u().origin().x == Approx(0).margin(1e-8));
    REQUIRE(vg.u().origin().y == Approx(0).margin(1e-8));
    REQUIRE(vg.v().origin().x == Approx(0).margin(1e-8));
    REQUIRE(vg.v().origin().y == Approx(0).margin(1e-8));
  }//
  SECTION("VectorGridAccessor") {
    auto f = [](point2 wp) -> float { return sin(wp.x) * cos(wp.y); };
    SECTION("CELL CENTERED") {
      VectorGrid2<float> vg(size2(10), vec2(0.1));
      for (auto e : vg.u().accessor()) {
        REQUIRE(e.worldPosition().x == Approx(0.1 * e.index.i).margin(1e-8));
        REQUIRE(e.worldPosition().y == Approx(0.1 * e.index.j).margin(1e-8));
        e.value = f(e.worldPosition());
      }
      for (auto e : vg.v().accessor()) {
        REQUIRE(e.worldPosition().x == Approx(0.1 * e.index.i).margin(1e-8));
        REQUIRE(e.worldPosition().y == Approx(0.1 * e.index.j).margin(1e-8));
        e.value = f(e.worldPosition());
      }
      auto acc = vg.accessor();
      for (index2 ij : Index2Range<i32>(vg.resolution())) {
        auto v = acc[ij];
        REQUIRE(v.x == Approx(f(acc.worldPosition(ij))));
        REQUIRE(v.y == Approx(f(acc.worldPosition(ij))));
      }
    }//
    SECTION("STAGGERED") {
      VectorGrid2<float> vg(size2(10), vec2(0.1), point2(),
                            VectorGridType::STAGGERED);
      for (auto e : vg.u().accessor()) {
        REQUIRE(e.worldPosition().x ==
            Approx(0.1 * e.index.i - 0.05).margin(1e-8));
        REQUIRE(e.worldPosition().y == Approx(0.1 * e.index.j).margin(1e-8));
        e.value = f(e.worldPosition());
      }
      for (auto e : vg.v().accessor()) {
        REQUIRE(e.worldPosition().x == Approx(0.1 * e.index.i).margin(1e-8));
        REQUIRE(e.worldPosition().y ==
            Approx(0.1 * e.index.j - 0.05).margin(1e-8));
        e.value = f(e.worldPosition());
      }
      auto acc = vg.accessor();
      for (index2 ij : Index2Range<i32>(vg.resolution())) {
        auto v = acc[ij];
        REQUIRE(v.x == Approx((f(acc.u().worldPosition(ij)) +
            f(acc.u().worldPosition(ij.right()))) /
            2));
        REQUIRE(v.y == Approx((f(acc.v().worldPosition(ij)) +
            f(acc.v().worldPosition(ij.up()))) /
            2));
      }
    }
  }//
  SECTION("ConstVectorGridAccessor") {
    auto f = [](point2 wp) -> float { return sin(wp.x) * cos(wp.y); };
    SECTION("CELL CENTERED") {
      VectorGrid2<float> vg(size2(10), vec2(0.1));
      for (auto e : vg.u().accessor()) {
        REQUIRE(e.worldPosition().x == Approx(0.1 * e.index.i).margin(1e-8));
        REQUIRE(e.worldPosition().y == Approx(0.1 * e.index.j).margin(1e-8));
        e.value = f(e.worldPosition());
      }
      for (auto e : vg.v().accessor()) {
        REQUIRE(e.worldPosition().x == Approx(0.1 * e.index.i).margin(1e-8));
        REQUIRE(e.worldPosition().y == Approx(0.1 * e.index.j).margin(1e-8));
        e.value = f(e.worldPosition());
      }
      auto ctest = [f](const VectorGrid2<float> &g) {
        auto acc = g.accessor();
        for (index2 ij : Index2Range<i32>(g.resolution())) {
          auto v = acc[ij];
          REQUIRE(v.x == Approx(f(acc.worldPosition(ij))));
          REQUIRE(v.y == Approx(f(acc.worldPosition(ij))));
        }
      };
      ctest(vg);
    }//
    SECTION("STAGGERED") {
      VectorGrid2<float> vg(size2(10), vec2(0.1), point2(),
                            VectorGridType::STAGGERED);
      for (auto e : vg.u().accessor()) {
        REQUIRE(e.worldPosition().x ==
            Approx(0.1 * e.index.i - 0.05).margin(1e-8));
        REQUIRE(e.worldPosition().y == Approx(0.1 * e.index.j).margin(1e-8));
        e.value = f(e.worldPosition());
      }
      for (auto e : vg.v().accessor()) {
        REQUIRE(e.worldPosition().x == Approx(0.1 * e.index.i).margin(1e-8));
        REQUIRE(e.worldPosition().y ==
            Approx(0.1 * e.index.j - 0.05).margin(1e-8));
        e.value = f(e.worldPosition());
      }
      auto ctest = [f](const VectorGrid2<float> &g) {
        auto acc = g.accessor();
        for (index2 ij : Index2Range<i32>(g.resolution())) {
          auto v = acc[ij];
          REQUIRE(v.x == Approx((f(acc.u().worldPosition(ij)) +
              f(acc.u().worldPosition(ij.right()))) /
              2));
          REQUIRE(v.y == Approx((f(acc.v().worldPosition(ij)) +
              f(acc.v().worldPosition(ij.up()))) /
              2));
        }
      };
      ctest(vg);
    }
  }
}

TEST_CASE("FDMatrix", "[numeric][fdmatrix]") {
  SECTION("2d") {
    SECTION("sanity") {
      // y
      // |
      //  ---x
      //  S S S S  - - - -
      //  S S S S  - - - -
      //  S S S S  - - - -
      //  S S S S  - - - -
      size2 size(4);
      FDMatrix2<f32> A(size);
      REQUIRE(A.grid_size() == size);
      REQUIRE(A.size() == 4u * 4u);
      auto &indices = A.indexData();
      int curIndex = 0;
      for (index2 ij : Index2Range<i32>(size))
        if (ij.j == 0 || ij.j == static_cast<i64>(size.height - 1) ||
            ij.i == 0 || ij.i == static_cast<i64>(size.width - 1))
          indices[ij] = -1;
        else
          indices[ij] = curIndex++;
      for (index2 ij : Index2Range<i32>(index2(1), index2(size).plus(-1, -1))) {
        if (ij.i > 1)
          REQUIRE(A.elementIndex(ij.left()) != -1);
        else if (ij.i == 1)
          REQUIRE(A.elementIndex(ij.left()) == -1);
        if (ij.j > 1)
          REQUIRE(A.elementIndex(ij.down()) != -1);
        else if (ij.j == 1)
          REQUIRE(A.elementIndex(ij.down()) == -1);
        if (ij.i < static_cast<i64>(size.width) - 2)
          REQUIRE(A.elementIndex(ij.right()) != -1);
        else if (ij.i == static_cast<i64>(size.width) - 2)
          REQUIRE(A.elementIndex(ij.right()) == -1);
        if (ij.j < static_cast<i64>(size.height) - 2)
          REQUIRE(A.elementIndex(ij.up()) != -1);
        else if (ij.j == static_cast<i64>(size.height) - 2)
          REQUIRE(A.elementIndex(ij.up()) == -1);
        A(ij, ij) = 6;
        A(ij, ij.right()) = 1;
        A(ij, ij.up()) = 2;
      }
    }SECTION("matrix vector") {
      // 3 4 5 0  0     14
      // 4 3 0 5  1  =  18
      // 5 0 3 4  2     18
      // 0 5 4 3  3     22
      FDMatrix2<f32> A(size2(2));
      int index = 0;
      for (index2 ij : Index2Range<i32>(A.grid_size()))
        A.indexData()[ij] = index++;
      for (index2 ij : Index2Range<i32>(A.grid_size())) {
        A(ij, ij) = 3;
        A(ij, ij.right()) = 4;
        A(ij, ij.up()) = 5;
      }
      Vector<f32> x(A.size(), 0);
      for (u32 i = 0; i < x.size(); i++)
        x[i] = i;
      int idx = 0;
      float ans[4] = {14, 18, 18, 22};
      auto iacc = A.indexData();
      for (index2 ij : Index2Range<i32>(A.grid_size())) {
        REQUIRE(
            ans[idx] ==
                (iacc.stores(ij.left()) ? iacc[ij.left()] : 0) * A(ij, ij.left()) +
                    (iacc[ij.down()] ? iacc[ij.down()] : 0) * A(ij, ij.down()) +
                    (iacc.stores(ij) ? iacc[ij] : 0) * A(ij, ij) +
                    (iacc.stores(ij.right()) ? iacc[ij.right()] : 0) *
                        A(ij, ij.right()) +
                    (iacc.stores(ij.up()) ? iacc[ij.up()] : 0) * A(ij, ij.up()));
        idx++;
      }
      Vector<f32> r = A * x;
      for (u32 i = 0; i < r.size(); i++)
        REQUIRE(r[i] == ans[i]);
    }
  }
}

TEST_CASE("DiffOps") {
  SECTION("2D") {
    SECTION("plane") {
      Grid2<f32> field({100, 100}, {0.1, 0.1});
      field = 10;
      auto grad = DiffOps::grad<f32>(field.constAccessor());
      REQUIRE(grad.resolution() == hermes::size2(100, 100));
      REQUIRE(grad.spacing() == hermes::vec2(0.1, 0.1));
      REQUIRE(grad.origin() == hermes::point2(0, 0));
      auto acc = grad.accessor();
      for (int i = 0; i < 100000; ++i) {
        hermes::point2 wp((rand() % 100) * 0.1, (rand() % 100) * 0.1);
        auto g = acc(wp);
        REQUIRE(hermes::Check::is_zero(g.x));
        REQUIRE(hermes::Check::is_zero(g.y));
      }
    }//
    SECTION("x2 gradient") {
      Grid2<f32> field({100, 100}, {0.1, 0.1});
      field = [](const hermes::point2 &p) -> f32 { return p.x * p.x + p.y * p.y; };
      auto grad = DiffOps::grad(field.constAccessor());
      auto acc = grad.accessor();
      for (int i = 0; i < 100000; ++i) {
        hermes::point2 wp((rand() % 98) * 0.1, (rand() % 98) * 0.1);
        auto g = acc(wp);
        REQUIRE(hermes::Check::is_equal(g.x, 2.f, 0.01f));
        REQUIRE(hermes::Check::is_equal(g.y, 2.f, 0.01f));
      }
    }//
  }//
}*/

/*
///////////////////////////////////////////////////////////////////////////////
////////////////////       INTERPOLATION     //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST_CASE("interpolation", "[numeric][interpolation]") {
  SECTION("monotonicCubic") {
    { // 1D test
      auto f = [](float x) -> float { return cos(x) * sin(x); };
      for (float s = 0.0; s <= 1.0; s += 0.01) {
        REQUIRE(ponos::monotonicCubicinterpolate(f(-0.1), f(0.0), f(0.1), f(0.2), s) ==
            Approx(f(s * 0.1)).margin(0.1 * 0.1));
      }
      for (float s = 0.0; s <= 1.0; s += 0.01)
        REQUIRE(
            ponos::monotonicCubicinterpolate(f(-0.01), f(0.0), f(0.01), f(0.02), s) ==
                Approx(f(s * 0.01)).margin(0.01 * 0.01));
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
////////////////////       GRID              //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

struct map_ipj {
  __device__ void operator()(index2 index, float &value) const {
    value = index.i * 10 + index.j;
  }
};

TEST_CASE("Grid", "[numeric][grid][access]") {
  SECTION("2d") {
    // check paramters constructor
    Grid2<float> g(size2(10, 10), vec2(0.1, 0.1), point2(1, 2));
    REQUIRE(g.spacing().x == Approx(0.1).margin(1e-8));
    REQUIRE(g.spacing().y == Approx(0.1).margin(1e-8));
    REQUIRE(g.origin().x == Approx(1).margin(1e-8));
    REQUIRE(g.origin().y == Approx(2).margin(1e-8));
    // value assign operator
    g = 3.0;
    // check host data method
    ponos::Grid2<float> hg = g.hostData();
    for (auto e : hg.accessor())
      REQUIRE(e.value == Approx(3).margin(1e-8));
    // check map method
    g.map(map_ipj());
    hg = g.hostData();
    for (auto e : hg.accessor())
      REQUIRE(e.value == Approx(e.index.i * 10 + e.index.j).margin(1e-8));
    // default constructor
    Grid2<float> gg;
    // check assign operator
    gg = g;
    REQUIRE(g.resolution() == gg.resolution());
    REQUIRE(g.spacing() == gg.spacing());
    REQUIRE(g.origin() == gg.origin());
    g = 7.3f;
    hg = g.hostData();
    for (auto e : hg.accessor())
      REQUIRE(e.value == Approx(7.3).margin(1e-8));
    // host data assign operator
    Array2<float> a(size2(10, 10));
    a = 1.f;
    g = a;
    hg = g.hostData();
    for (auto e : hg.accessor())
      REQUIRE(e.value == Approx(1).margin(1e-8));
    // check copy constructors
    std::vector<Grid2<f32>> grid_vector;
    grid_vector.emplace_back();
    grid_vector[0].setResolution(size2(10, 10));
    grid_vector.emplace_back();
    grid_vector.emplace_back();
    REQUIRE(grid_vector[0].resolution() == size2(10));
    REQUIRE(grid_vector[1].resolution() == size2(0));
    // check constructor from host grid
    Grid2<float> d_g(hg);
    REQUIRE(d_g.resolution() == size2(hg.resolution()));
    REQUIRE(d_g.spacing() == vec2(hg.spacing()));
    REQUIRE(d_g.origin() == point2(hg.origin()));
  }
}

TEST_CASE("VectorGrid", "[numeric][grid]") {
  SECTION("2d") {
    SECTION("constructors operatiors") {
      VectorGrid2<float> empty;
      REQUIRE(empty.resolution() == size2(0, 0));
      empty = VectorGrid2<float>(size2(10, 10), vec2f(1));
      REQUIRE(empty.resolution() == size2(10, 10));
      VectorGrid2<float> g = empty;
      REQUIRE(g.resolution() == size2(10, 10));
      auto g2 = std::move(VectorGrid2<float>(size2(7, 7), vec2f(1)));
      REQUIRE(g2.resolution() == size2(7, 7));
      std::vector<VectorGrid2<float>> gs;
      gs.emplace_back(size2(10, 10), vec2(1));
      gs.emplace_back(size2(7, 7), vec2(1));
      std::vector<VectorGrid2<float>> gs2 = gs;
    }
    SECTION("CELL CENTERED") {
      VectorGrid2<float> vg(size2(10, 10), vec2(1));
      REQUIRE(vg.resolution() == size2(10, 10));
      REQUIRE(vg.u().resolution() == size2(10, 10));
      REQUIRE(vg.v().resolution() == size2(10, 10));
      REQUIRE(vg.origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg.origin().y == Approx(0).margin(1e-8));
      REQUIRE(vg.spacing().x == Approx(1).margin(1e-8));
      REQUIRE(vg.spacing().y == Approx(1).margin(1e-8));
      REQUIRE(vg.u().origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg.u().origin().y == Approx(0).margin(1e-8));
      REQUIRE(vg.v().origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg.v().origin().y == Approx(0).margin(1e-8));
      VectorGrid2<float> vg2;
      vg2 = vg;
      REQUIRE(vg2.resolution() == size2(10, 10));
      REQUIRE(vg2.u().resolution() == size2(10, 10));
      REQUIRE(vg2.v().resolution() == size2(10, 10));
      REQUIRE(vg2.origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg2.origin().y == Approx(0).margin(1e-8));
      REQUIRE(vg2.spacing().x == Approx(1).margin(1e-8));
      REQUIRE(vg2.spacing().y == Approx(1).margin(1e-8));
      REQUIRE(vg2.u().origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg2.u().origin().y == Approx(0).margin(1e-8));
      REQUIRE(vg2.v().origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg2.v().origin().y == Approx(0).margin(1e-8));
      vg.setGridType(ponos::VectorGridType::STAGGERED);
      REQUIRE(vg.resolution() == size2(10, 10));
      REQUIRE(vg.u().resolution() == size2(11, 10));
      REQUIRE(vg.v().resolution() == size2(10, 11));
      REQUIRE(vg.origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg.origin().y == Approx(0).margin(1e-8));
      REQUIRE(vg.spacing().x == Approx(1).margin(1e-8));
      REQUIRE(vg.spacing().y == Approx(1).margin(1e-8));
      REQUIRE(vg.u().origin().x == Approx(-0.5).margin(1e-8));
      REQUIRE(vg.u().origin().y == Approx(0).margin(1e-8));
      REQUIRE(vg.v().origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg.v().origin().y == Approx(-0.5).margin(1e-8));
    }
    SECTION("STAGGERED") {
      VectorGrid2<float> vg(ponos::VectorGridType::STAGGERED);
      vg.setResolution(size2(10, 10));
      REQUIRE(vg.resolution() == size2(10, 10));
      REQUIRE(vg.u().resolution() == size2(11, 10));
      REQUIRE(vg.v().resolution() == size2(10, 11));
      REQUIRE(vg.origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg.origin().y == Approx(0).margin(1e-8));
      REQUIRE(vg.spacing().x == Approx(1).margin(1e-8));
      REQUIRE(vg.spacing().y == Approx(1).margin(1e-8));
      REQUIRE(vg.u().origin().x == Approx(-0.5).margin(1e-8));
      REQUIRE(vg.u().origin().y == Approx(0).margin(1e-8));
      REQUIRE(vg.v().origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg.v().origin().y == Approx(-0.5).margin(1e-8));
      VectorGrid2<float> vg2;
      vg2 = vg;
      REQUIRE(vg2.resolution() == size2(10, 10));
      REQUIRE(vg2.u().resolution() == size2(11, 10));
      REQUIRE(vg2.v().resolution() == size2(10, 11));
      REQUIRE(vg2.origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg2.origin().y == Approx(0).margin(1e-8));
      REQUIRE(vg2.spacing().x == Approx(1).margin(1e-8));
      REQUIRE(vg2.spacing().y == Approx(1).margin(1e-8));
      REQUIRE(vg2.u().origin().x == Approx(-0.5).margin(1e-8));
      REQUIRE(vg2.u().origin().y == Approx(0).margin(1e-8));
      REQUIRE(vg2.v().origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg2.v().origin().y == Approx(-0.5).margin(1e-8));
      vg.setGridType(ponos::VectorGridType::CELL_CENTERED);
      REQUIRE(vg.resolution() == size2(10, 10));
      REQUIRE(vg.u().resolution() == size2(10, 10));
      REQUIRE(vg.v().resolution() == size2(10, 10));
      REQUIRE(vg.origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg.origin().y == Approx(0).margin(1e-8));
      REQUIRE(vg.spacing().x == Approx(1).margin(1e-8));
      REQUIRE(vg.spacing().y == Approx(1).margin(1e-8));
      REQUIRE(vg.u().origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg.u().origin().y == Approx(0).margin(1e-8));
      REQUIRE(vg.v().origin().x == Approx(0).margin(1e-8));
      REQUIRE(vg.v().origin().y == Approx(0).margin(1e-8));
    }
  }
}

struct map_sincos {
  map_sincos(Grid2Accessor<float> acc) : acc(acc) {}
  __device__ void operator()(index2 index, float &value) const {
    auto wp = acc.worldPosition(index);
    value = sin(wp.x) * cos(wp.y);
  }
  Grid2Accessor<float> acc;
};

TEST_CASE("VectorGridAccessor", "[numeric][grid][accessor]") {
  auto f = [](ponos::point2 wp) -> float { return sin(wp.x) * cos(wp.y); };
  SECTION("2d") {
    SECTION("methods") {
      ponos::VectorGrid2<float> hg(ponos::size2(10), ponos::vec2(0.1),
                                   ponos::point2(),
                                   ponos::VectorGridType::STAGGERED);
      VectorGrid2<float> dg(size2(10), vec2(0.1), point2(),
                            ponos::VectorGridType::STAGGERED);
      auto hacc = hg.accessor();
      auto dacc = dg.accessor();
      for (index2 ij : Index2Range<i32>(dg.resolution())) {
        REQUIRE(hacc.u().worldPosition(ij.ponos()).x ==
            Approx(dacc.u().worldPosition(ij).x).margin(1e-8));
        REQUIRE(hacc.u().worldPosition(ij.ponos()).y ==
            Approx(dacc.u().worldPosition(ij).y).margin(1e-8));
        REQUIRE(hacc.v().worldPosition(ij.ponos()).x ==
            Approx(dacc.v().worldPosition(ij).x).margin(1e-8));
        REQUIRE(hacc.v().worldPosition(ij.ponos()).y ==
            Approx(dacc.v().worldPosition(ij).y).margin(1e-8));
      }
    }
    SECTION("device to host") {
      VectorGrid2<float> dg(size2(10), vec2f(0.01));
      dg.u().map(map_sincos(dg.u().accessor()));
      dg.v().map(map_sincos(dg.v().accessor()));
      auto hg = dg.hostData();
      auto dacc = dg.accessor();
      auto hacc = hg.accessor();
      for (index2 ij : Index2Range<i32>(dg.resolution())) {
        auto v = hacc[ij.ponos()];
        REQUIRE(v.x == Approx(f(hacc.worldPosition(ij.ponos()))));
        REQUIRE(v.y == Approx(f(hacc.worldPosition(ij.ponos()))));
        REQUIRE(hacc.u().worldPosition(ij.ponos()).x ==
            Approx(dacc.u().worldPosition(ij).x).margin(1e-8));
        REQUIRE(hacc.u().worldPosition(ij.ponos()).y ==
            Approx(dacc.u().worldPosition(ij).y).margin(1e-8));
        REQUIRE(hacc.v().worldPosition(ij.ponos()).x ==
            Approx(dacc.v().worldPosition(ij).x).margin(1e-8));
        REQUIRE(hacc.v().worldPosition(ij.ponos()).y ==
            Approx(dacc.v().worldPosition(ij).y).margin(1e-8));
      }
    }
    SECTION("host to device") {
      ponos::VectorGrid2<float> hg(ponos::size2(10), ponos::vec2(0.1),
                                   ponos::point2(),
                                   ponos::VectorGridType::STAGGERED);
      for (auto e : hg.u().accessor())
        e.value = f(e.worldPosition());
      for (auto e : hg.v().accessor())
        e.value = f(e.worldPosition());

      VectorGrid2<float> dg = hg;

      auto hdg = dg.hostData();
      auto hacc = hg.accessor();
      auto dacc = dg.accessor();
      for (index2 ij : Index2Range<i32>(dg.resolution())) {
        auto v = hacc[ij.ponos()];
        REQUIRE(v.x == Approx((f(hacc.u().worldPosition(ij.ponos())) +
            f(hacc.u().worldPosition(ij.right().ponos()))) /
            2));
        REQUIRE(v.y == Approx((f(hacc.v().worldPosition(ij.ponos())) +
            f(hacc.v().worldPosition(ij.up().ponos()))) /
            2));
        REQUIRE(hacc.u().worldPosition(ij.ponos()).x ==
            Approx(dacc.u().worldPosition(ij).x).margin(1e-8));
        REQUIRE(hacc.u().worldPosition(ij.ponos()).y ==
            Approx(dacc.u().worldPosition(ij).y).margin(1e-8));
        REQUIRE(hacc.v().worldPosition(ij.ponos()).x ==
            Approx(dacc.v().worldPosition(ij).x).margin(1e-8));
        REQUIRE(hacc.v().worldPosition(ij.ponos()).y ==
            Approx(dacc.v().worldPosition(ij).y).margin(1e-8));
      }
    }
  }
}

TEST_CASE("FDMatrix", "[numeric][fdmatrix]") {
  SECTION("2d") {
    SECTION("sanity") {
      // y
      // |
      //  ---x
      //  S S S S  - - - -
      //  S S S S  - - - -
      //  S S S S  - - - -
      //  S S S S  - - - -
      ponos::size2 size(4);
      ponos::FDMatrix2<f32> A(size);
      auto &indices = A.indexData();
      int curIndex = 0;
      for (ponos::index2 ij : ponos::Index2Range<i32>(size))
        if (ij.j == 0 || ij.j == static_cast<i64>(size.height - 1) ||
            ij.i == 0 || ij.i == static_cast<i64>(size.width - 1))
          indices[ij] = -1;
        else
          indices[ij] = curIndex++;
      FDMatrix2<f32> d_A = A;
      auto h_A = d_A.hostData();
      for (ponos::index2 ij : ponos::Index2Range<i32>(
          ponos::index2(1), ponos::index2(size).plus(-1, -1))) {
        if (ij.i > 1)
          REQUIRE(h_A.elementIndex(ij.left()) != -1);
        else if (ij.i == 1)
          REQUIRE(h_A.elementIndex(ij.left()) == -1);
        if (ij.j > 1)
          REQUIRE(h_A.elementIndex(ij.down()) != -1);
        else if (ij.j == 1)
          REQUIRE(h_A.elementIndex(ij.down()) == -1);
        if (ij.i < static_cast<i64>(size.width) - 2)
          REQUIRE(h_A.elementIndex(ij.right()) != -1);
        else if (ij.i == static_cast<i64>(size.width) - 2)
          REQUIRE(h_A.elementIndex(ij.right()) == -1);
        if (ij.j < static_cast<i64>(size.height) - 2)
          REQUIRE(h_A.elementIndex(ij.up()) != -1);
        else if (ij.j == static_cast<i64>(size.height) - 2)
          REQUIRE(h_A.elementIndex(ij.up()) == -1);
        h_A(ij, ij) = 6;
        h_A(ij, ij.right()) = 1;
        h_A(ij, ij.up()) = 2;
      }
    }
    SECTION("matrix vector") {
      // 3 4 5 0  0     14
      // 4 3 0 5  1  =  18
      // 5 0 3 4  2     18
      // 0 5 4 3  3     22
      ponos::FDMatrix2<f32> A(ponos::size2(2));
      int index = 0;
      for (ponos::index2 ij : ponos::Index2Range<i32>(A.gridSize()))
        A.indexData()[ij] = index++;
      for (ponos::index2 ij : ponos::Index2Range<i32>(A.gridSize())) {
        A(ij, ij) = 3;
        A(ij, ij.right()) = 4;
        A(ij, ij.up()) = 5;
      }
      ponos::Vector<f32> x(A.size(), 0);
      for (u32 i = 0; i < x.size(); i++)
        x[i] = i;
      float ans[4] = {14, 18, 18, 22};
      FDMatrix2<f32> d_A = A;
      Vector<f32> d_x = x;
      Vector<f32> r = d_A * d_x;
      auto h_r = r.hostData();
      for (u32 i = 0; i < r.size(); i++)
        REQUIRE(h_r[i] == ans[i]);
    }
  }
}
 */
