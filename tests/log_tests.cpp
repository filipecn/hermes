#include <catch2/catch_test_macros.hpp>

// #include <hermes/logging/memory_dump.h>
#include <hermes/common/debug.h>
// #include <hermes/geometry/point.h>
// #include <hermes/geometry/transform.h>
#include <hermes/log/console_colors.h>

using namespace hermes;

TEST_CASE("debug macros", "[log]") {
  SECTION("flow") {
    {
      auto rif = [](int a, int b) -> bool {
        HERMES_RETURN_VALUE_IF(a == b, true);
        return false;
      };
      auto rnif = [](int a, int b) -> bool {
        HERMES_RETURN_VALUE_IF_NOT(a == b, true);
        return false;
      };
      REQUIRE(!rif(3, 2));
      REQUIRE(rif(3, 3));
      REQUIRE(rnif(3, 2));
      REQUIRE(!rnif(3, 3));
    }
    {
      auto rif = [](int a, int b) -> bool {
        HERMES_RETURN_VALUE_IF(a == b, true);
        return false;
      };
      auto rnif = [](int a, int b) -> bool {
        HERMES_LOG_AND_RETURN_VALUE_IF_NOT(a == b, true, "message");
        return false;
      };
      REQUIRE(!rif(3, 2));
      REQUIRE(rif(3, 3));
      REQUIRE(rnif(3, 2));
      REQUIRE(!rnif(3, 3));
    }
  }
  HERMES_PING
  HERMES_INFO(std::to_string(3).c_str());
  HERMES_INFO("{:.15s}", "long string test");
  HERMES_DEBUG("debug");
  HERMES_TRACE("trace");
  HERMES_INFO("info");
  HERMES_WARN("warning");
  HERMES_ERROR("error");
  HERMES_CRITICAL("critical");
  Log::addOptions(logging_options::abbreviate);
  int a = 0;
  int b = 3;
  int c = 4;
  Log::removeOptions(logging_options::location);
  HERMES_LOG_VARIABLE(a);
  HERMES_LOG_VARIABLES(a, b, c);
  HERMES_CHECK(3 == 3);
  HERMES_CHECK(3 == 2);
  HERMES_CHECK(3 == 3, "message");
  HERMES_CHECK(3 == 2, "message");
  HERMES_ASSERT(3 == 3);
  // HERMES_ASSERT(3 == 2);
  HERMES_ASSERT(3 == 3, "message");
  // HERMES_ASSERT(3 == 2, "message");
  //  C logs
  HERMES_C_LOG("c logging %d", 1);
  HERMES_C_LOG("c logging");
  HERMES_C_ERROR("c logging error %d", 1);
  HERMES_C_ERROR("c logging error");
  Log::addOptions(logging_options::location);
  Log::removeOptions(logging_options::abbreviate);

  auto callLog = []() {
    HERMES_DEBUG("this is a debug");
    HERMES_TRACE("this is a trace");
    HERMES_INFO("this is a info");
    HERMES_WARN("this is a warning");
    HERMES_ERROR("this is an error");
    HERMES_CRITICAL("this is a critical");
  };

  callLog();

  std::stringstream ss;
  hermes::Log::setStream(&ss);
  hermes::Log::setLevel(hermes::Log::Level::warn);

  callLog();

  REQUIRE(ss.str().size());
}

TEST_CASE("Console Colors", "[log]") {

#define PRINT_COLOR_NAME(COLOR) std::cout << (COLOR) << #COLOR << std::endl;

  PRINT_COLOR_NAME(ConsoleColors::default_color);
  PRINT_COLOR_NAME(ConsoleColors::black);
  PRINT_COLOR_NAME(ConsoleColors::red);
  PRINT_COLOR_NAME(ConsoleColors::green);
  PRINT_COLOR_NAME(ConsoleColors::yellow);
  PRINT_COLOR_NAME(ConsoleColors::blue);
  PRINT_COLOR_NAME(ConsoleColors::magenta);
  PRINT_COLOR_NAME(ConsoleColors::cyan);
  PRINT_COLOR_NAME(ConsoleColors::light_gray);
  PRINT_COLOR_NAME(ConsoleColors::dark_gray);
  PRINT_COLOR_NAME(ConsoleColors::light_red);
  PRINT_COLOR_NAME(ConsoleColors::light_green);
  PRINT_COLOR_NAME(ConsoleColors::light_yellow);
  PRINT_COLOR_NAME(ConsoleColors::light_blue);
  PRINT_COLOR_NAME(ConsoleColors::light_magenta);
  PRINT_COLOR_NAME(ConsoleColors::light_cyan);
  PRINT_COLOR_NAME(ConsoleColors::white);
  PRINT_COLOR_NAME(ConsoleColors::background_default_color);
  PRINT_COLOR_NAME(ConsoleColors::background_black);
  PRINT_COLOR_NAME(ConsoleColors::background_red);
  PRINT_COLOR_NAME(ConsoleColors::background_green);
  PRINT_COLOR_NAME(ConsoleColors::background_yellow);
  PRINT_COLOR_NAME(ConsoleColors::background_blue);
  PRINT_COLOR_NAME(ConsoleColors::background_magenta);
  PRINT_COLOR_NAME(ConsoleColors::background_cyan);
  PRINT_COLOR_NAME(ConsoleColors::background_light_gray);
  PRINT_COLOR_NAME(ConsoleColors::background_dark_gray);
  PRINT_COLOR_NAME(ConsoleColors::background_light_red);
  PRINT_COLOR_NAME(ConsoleColors::background_light_green);
  PRINT_COLOR_NAME(ConsoleColors::background_light_yellow);
  PRINT_COLOR_NAME(ConsoleColors::background_light_blue);
  PRINT_COLOR_NAME(ConsoleColors::background_light_magenta);
  PRINT_COLOR_NAME(ConsoleColors::background_light_cyan);
  PRINT_COLOR_NAME(ConsoleColors::background_white);

  std::cout << ConsoleColors::background_default_color;
  for (u8 r = 0; r < 32; ++r) {
    for (u8 i = 0; i < 8; ++i)
      std::cout << ConsoleColors::color(r * 8 + i) << std::to_string(r * 8 + i)
                << " ";
    std::cout << std::endl;
  }

  std::cout << ConsoleColors::black;
  for (u8 r = 0; r < 32; ++r) {
    for (u8 i = 0; i < 8; ++i)
      std::cout << ConsoleColors::background_color(r * 8 + i)
                << std::to_string(r * 8 + i) << " ";
    std::cout << std::endl;
  }
  std::cout << ConsoleColors::reset << "reset\n";
  std::cout << ConsoleColors::combine(ConsoleColors::blink,
                                      ConsoleColors::green)
            << "blink green " << ConsoleColors::reset << std::endl;
#undef PRINT_COLOR_NAME
}

/*
 */
