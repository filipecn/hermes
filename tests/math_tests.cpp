#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <hermes/math/math.h>
#include <hermes/math/space_filling.h>

TEST_CASE("Space Filling", "math") {
  SECTION("morton") {
    REQUIRE(hermes::math::space_filling::mortonEncode({0, 0}) == 0);
    REQUIRE(hermes::math::space_filling::mortonEncode({1, 0}) == 1);
    REQUIRE(hermes::math::space_filling::mortonEncode({0, 1}) == 2);
    REQUIRE(hermes::math::space_filling::mortonEncode({1, 1}) == 3);
    REQUIRE(hermes::math::space_filling::mortonEncode({2, 0}) == 4);
    for (auto z : hermes::math::space_filling::MortonRange(0, 4)) {
      REQUIRE(*z == hermes::math::space_filling::mortonEncode(z.coord2()));
    }
  }
}
