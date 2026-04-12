#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <hermes/math/math.h>
#include <hermes/math/space_filling.h>

TEST_CASE("angles", "[math]") {

  REQUIRE_THAT(hermes::math::wrapDegrees(-720),
               Catch::Matchers::WithinRel(0.0, 1e-6));

  REQUIRE_THAT(hermes::math::wrapDegrees(270 * 7),
               Catch::Matchers::WithinRel(90.0, 1e-6));

  REQUIRE_THAT(hermes::math::wrapDegrees(-270 * 7),
               Catch::Matchers::WithinRel(270.0, 1e-6));

  REQUIRE_THAT(hermes::math::wrapDegrees(-180.0),
               Catch::Matchers::WithinRel(180.0, 1e-6));
}

TEST_CASE("Space Filling", "math") {
  SECTION("onion") {
    SECTION("sanity") {
      hermes::size2 size(1, 1);
      SECTION("all") {
        h_size count = 0;
        for (auto o : hermes::math::space_filling::OnionRange(size)) {
          REQUIRE(o.coord2() == hermes::index2(0, 0));
          REQUIRE(o.index() == count);
          count++;
        }
        REQUIRE(count == size.total());
      }
      SECTION("start") {
        h_size count = 0;
        for (auto o : hermes::math::space_filling::OnionRange(size, 1, 1)) {
          HERMES_UNUSED_VARIABLE(o);
          count++;
        }
        REQUIRE(count == 0);
      }
      SECTION("count") {
        h_size count = 0;
        for (auto o : hermes::math::space_filling::OnionRange(size, 1)) {
          REQUIRE(o.coord2() == hermes::index2(0, 0));
          REQUIRE(o.index() == count);
          count++;
        }
        REQUIRE(count == size.total());
      }
    }
    SECTION("h vector") {
      hermes::size2 size(3, 1);
      //  0  0  1  2
      //     0  1  2
      SECTION("all") {
        h_size count = 0;
        for (auto o : hermes::math::space_filling::OnionRange(size)) {
          REQUIRE(o.coord2() == hermes::index2(count, 0));
          REQUIRE(o.index() == count);
          count++;
        }
        REQUIRE(count == size.total());
      }
      SECTION("start") {
        h_size count = 0;
        for (auto o : hermes::math::space_filling::OnionRange(size, 1, 1)) {
          HERMES_UNUSED_VARIABLE(o);
          count++;
        }
        REQUIRE(count == 0);
      }
      SECTION("count") {
        h_size count = 0;
        for (auto o : hermes::math::space_filling::OnionRange(size, 1)) {
          REQUIRE(o.coord2() == hermes::index2(count, 0));
          REQUIRE(o.index() == count);
          count++;
        }
        REQUIRE(count == size.total());
      }
    }
    SECTION("v vector") {
      hermes::size2 size(1, 3);
      SECTION("all") {
        h_size count = 0;
        for (auto o : hermes::math::space_filling::OnionRange(size)) {
          REQUIRE(o.coord2() == hermes::index2(0, count));
          REQUIRE(o.index() == count);
          count++;
        }
        REQUIRE(count == size.total());
      }
      SECTION("start") {
        h_size count = 0;
        for (auto o : hermes::math::space_filling::OnionRange(size, 1, 1)) {
          HERMES_UNUSED_VARIABLE(o);
          count++;
        }
        REQUIRE(count == 0);
      }
      SECTION("count") {
        h_size count = 0;
        for (auto o : hermes::math::space_filling::OnionRange(size, 1)) {
          REQUIRE(o.coord2() == hermes::index2(0, count));
          REQUIRE(o.index() == count);
          count++;
        }
        REQUIRE(count == size.total());
      }
    }
    SECTION("even square") {
      hermes::size2 size(4, 4);
      //  3  9  8  7  6
      //  2 10 15 14  5
      //  1 11 12 13  4
      //  0  0  1  2  3
      //     0  1  2  3
      std::vector<hermes::index2> ans = {
          {0, 0}, {1, 0}, {2, 0}, {3, 0}, {3, 1}, {3, 2}, {3, 3}, {2, 3},
          {1, 3}, {0, 3}, {0, 2}, {0, 1}, {1, 1}, {2, 1}, {2, 2}, {1, 2},
      };
      SECTION("all") {
        h_size count = 0;
        for (auto o : hermes::math::space_filling::OnionRange(size)) {
          REQUIRE(ans[count] == o.coord2());
          REQUIRE(o.index() == count);
          count++;
        }
        REQUIRE(count == size.total());
      }
      SECTION("start") {
        h_size count = 12;
        for (auto o : hermes::math::space_filling::OnionRange(size, 1, 1)) {
          REQUIRE(ans[count] == o.coord2());
          REQUIRE(o.index() == count);
          count++;
        }
        REQUIRE(count - 12 == 4);
      }
      SECTION("count") {
        h_size count = 0;
        for (auto o : hermes::math::space_filling::OnionRange(size, 1)) {
          REQUIRE(ans[count] == o.coord2());
          REQUIRE(o.index() == count);
          count++;
        }
        REQUIRE(count == 12);
      }
    }
    SECTION("odd square") {
      hermes::size2 size(3, 3);
      std::vector<hermes::index2> ans = {{0, 0}, {1, 0}, {2, 0}, {2, 1}, {2, 2},
                                         {1, 2}, {0, 2}, {0, 1}, {1, 1}};
      //  2  6  5  4
      //  1  7  8  3
      //  0  0  1  2
      //     0  1  2
      SECTION("all") {
        h_size count = 0;
        for (auto o : hermes::math::space_filling::OnionRange(size)) {
          REQUIRE(ans[count] == o.coord2());
          REQUIRE(o.index() == count);
          count++;
        }
        REQUIRE(count == size.total());
      }
      SECTION("start") {
        h_size count = 8;
        for (auto o : hermes::math::space_filling::OnionRange(size, 1, 1)) {
          REQUIRE(ans[count] == o.coord2());
          REQUIRE(o.index() == count);
          count++;
        }
        REQUIRE(count - 8 == 1);
      }
      SECTION("count") {
        h_size count = 0;
        for (auto o : hermes::math::space_filling::OnionRange(size, 1)) {
          REQUIRE(ans[count] == o.coord2());
          REQUIRE(o.index() == count);
          count++;
        }
        REQUIRE(count == 8);
      }
    }
    SECTION("odd,even rectangle") {
      hermes::size2 size(3, 4);
      //  3  7  6  5
      //  2  8 11  4
      //  1  9 10  3
      //  0  0  1  2
      //     0  1  2
      std::vector<hermes::index2> ans = {{0, 0}, {1, 0}, {2, 0}, {2, 1},
                                         {2, 2}, {2, 3}, {1, 3}, {0, 3},
                                         {0, 2}, {0, 1}, {1, 1}, {1, 2}};
      SECTION("all") {
        h_size count = 0;
        for (auto o : hermes::math::space_filling::OnionRange(size)) {
          REQUIRE(ans[count] == o.coord2());
          REQUIRE(o.index() == count);
          count++;
        }
        REQUIRE(count == size.total());
      }
      SECTION("start") {
        h_size count = 10;
        for (auto o : hermes::math::space_filling::OnionRange(size, 1, 1)) {
          REQUIRE(ans[count] == o.coord2());
          REQUIRE(o.index() == count);
          count++;
        }
        REQUIRE(count - 10 == 2);
      }
      SECTION("count") {
        h_size count = 0;
        for (auto o : hermes::math::space_filling::OnionRange(size, 1)) {
          REQUIRE(ans[count] == o.coord2());
          REQUIRE(o.index() == count);
          count++;
        }
        REQUIRE(count == 10);
      }
    }
    SECTION("even,odd rectangle") {
      hermes::size2 size(4, 3);
      //  2  8  7  6  5
      //  1  9 10 11  4
      //  0  0  1  2  3
      //     0  1  2  3
      std::vector<hermes::index2> ans = {{0, 0}, {1, 0}, {2, 0}, {3, 0},
                                         {3, 1}, {3, 2}, {2, 2}, {1, 2},
                                         {0, 2}, {0, 1}, {1, 1}, {2, 1}};
      SECTION("all") {
        h_size count = 0;
        for (auto o : hermes::math::space_filling::OnionRange(size)) {
          REQUIRE(ans[count] == o.coord2());
          REQUIRE(o.index() == count);
          count++;
        }
        REQUIRE(count == size.total());
      }
      SECTION("start") {
        h_size count = 10;
        for (auto o : hermes::math::space_filling::OnionRange(size, 1, 1)) {
          REQUIRE(ans[count] == o.coord2());
          REQUIRE(o.index() == count);
          count++;
        }
        REQUIRE(count - 10 == 2);
      }
      SECTION("count") {
        h_size count = 0;
        for (auto o : hermes::math::space_filling::OnionRange(size, 1)) {
          REQUIRE(ans[count] == o.coord2());
          REQUIRE(o.index() == count);
          count++;
        }
        REQUIRE(count == 10);
      }
    }
  }
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
