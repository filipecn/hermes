#include <catch2/catch_test_macros.hpp>

#include <hermes/io/logger.h>
#include <hermes/io/memory_dumper.h>

using namespace hermes;

TEST_CASE("Log", "[io]") {
  HERMES_PING
  HERMES_INFO(std::to_string(3).c_str());
  HERMES_INFO("{:.15s}", "long string test");
  HERMES_DEBUG("debug");
  HERMES_TRACE("trace");
  HERMES_INFO("info");
  HERMES_WARN("warning");
  HERMES_ERROR("error");
  HERMES_CRITICAL("critical");
  Logger::addOptions(logging_option_bits::abbreviate);
  int a = 0;
  int b = 3;
  int c = 4;
  Logger::removeOptions(logging_option_bits::location);
  HERMES_LOG_VARIABLE(a);
  HERMES_LOG_VARIABLES(a, b, c);
  //  C logs
  HERMES_C_LOG("c logging %d", 1);
  HERMES_C_LOG("c logging");
  HERMES_C_ERROR("c logging error %d", 1);
  HERMES_C_ERROR("c logging error");
  Logger::addOptions(logging_option_bits::location);
  Logger::removeOptions(logging_option_bits::abbreviate);

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
  Logger::setStream(&ss);
  Logger::setLevel(Logger::Level::warn);

  callLog();

  REQUIRE(ss.str().size());
  Logger::setStream(&std::cout);
}

TEST_CASE("Console Colors", "[io]") {

#define PRINT_COLOR_NAME(COLOR) std::cout << (COLOR) << #COLOR << std::endl;

  PRINT_COLOR_NAME(colors::console::default_color);
  PRINT_COLOR_NAME(colors::console::black);
  PRINT_COLOR_NAME(colors::console::red);
  PRINT_COLOR_NAME(colors::console::green);
  PRINT_COLOR_NAME(colors::console::yellow);
  PRINT_COLOR_NAME(colors::console::blue);
  PRINT_COLOR_NAME(colors::console::magenta);
  PRINT_COLOR_NAME(colors::console::cyan);
  PRINT_COLOR_NAME(colors::console::light_gray);
  PRINT_COLOR_NAME(colors::console::dark_gray);
  PRINT_COLOR_NAME(colors::console::light_red);
  PRINT_COLOR_NAME(colors::console::light_green);
  PRINT_COLOR_NAME(colors::console::light_yellow);
  PRINT_COLOR_NAME(colors::console::light_blue);
  PRINT_COLOR_NAME(colors::console::light_magenta);
  PRINT_COLOR_NAME(colors::console::light_cyan);
  PRINT_COLOR_NAME(colors::console::white);
  PRINT_COLOR_NAME(colors::console::background_default_color);
  PRINT_COLOR_NAME(colors::console::background_black);
  PRINT_COLOR_NAME(colors::console::background_red);
  PRINT_COLOR_NAME(colors::console::background_green);
  PRINT_COLOR_NAME(colors::console::background_yellow);
  PRINT_COLOR_NAME(colors::console::background_blue);
  PRINT_COLOR_NAME(colors::console::background_magenta);
  PRINT_COLOR_NAME(colors::console::background_cyan);
  PRINT_COLOR_NAME(colors::console::background_light_gray);
  PRINT_COLOR_NAME(colors::console::background_dark_gray);
  PRINT_COLOR_NAME(colors::console::background_light_red);
  PRINT_COLOR_NAME(colors::console::background_light_green);
  PRINT_COLOR_NAME(colors::console::background_light_yellow);
  PRINT_COLOR_NAME(colors::console::background_light_blue);
  PRINT_COLOR_NAME(colors::console::background_light_magenta);
  PRINT_COLOR_NAME(colors::console::background_light_cyan);
  PRINT_COLOR_NAME(colors::console::background_white);

  std::cout << colors::console::background_default_color;
  for (u8 r = 0; r < 32; ++r) {
    for (u8 i = 0; i < 8; ++i)
      std::cout << colors::console::color(r * 8 + i)
                << std::to_string(r * 8 + i) << " ";
    std::cout << std::endl;
  }

  std::cout << colors::console::black;
  for (u8 r = 0; r < 32; ++r) {
    for (u8 i = 0; i < 8; ++i)
      std::cout << colors::console::background_color(r * 8 + i)
                << std::to_string(r * 8 + i) << " ";
    std::cout << std::endl;
  }
  std::cout << colors::console::reset << "reset\n";
  std::cout << colors::console::combine(colors::console::blink,
                                        colors::console::green)
            << "blink green " << colors::console::reset << std::endl;
#undef PRINT_COLOR_NAME
}

TEST_CASE("MemoryDumper", "[log]") {
  SECTION("hex") {
    const char s[] = "abcdefghijklmnopqrstuvxzwy";
    std::cerr << MemoryDumper::dump(s, sizeof(s), 8, {},
                                    memory_dumper_option_bits::colored_output |
                                        memory_dumper_option_bits::show_ascii);
    const u32 v32[] = {1, 2, 3, 4, 5};
    MemoryDumper::dump(v32, 5);
  } //
  SECTION("binary") {
    const u8 v8[] = {1, 2, 3, 4, 5};
    MemoryDumper::dump(v8, 5, 8, {}, memory_dumper_option_bits::binary);
  } //
  SECTION("decimal") {
    const u16 v16[] = {1, 2, 3, 4, 5};
    MemoryDumper::dump(v16, 5, 8, {}, memory_dumper_option_bits::decimal);
  } //
  SECTION("hide zeros") {
    struct Data {
      u32 b;
      u8 a;
      u16 c;
    };
    const Data data[] = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}};
    MemoryDumper::dump(data, 4, 8, {}, memory_dumper_option_bits::hide_zeros);
  } //
  SECTION("row size") {
    u64 v[] = {0, 1, 2, 3, 4, 5, 6, 7};
    MemoryDumper::dump(v, 8, 24);
  } //
  SECTION("hide header and ascii") {
    u16 v[] = {0, 1, 2, 3, 4, 5, 6, 7};
    MemoryDumper::dump(v, 8, 8, {}, memory_dumper_option_bits::hide_header);
  } //
  SECTION("cache align") {
    u64 v[] = {0, 1, 2, 3, 4, 5, 6, 7};
    std::cerr << MemoryDumper::dumpInfo(v, 8);
    MemoryDumper::dump(v, 8, 64, {}, memory_dumper_option_bits::cache_align);
  } //
  SECTION("colored output") {
    SECTION("packed members") {
      struct S {
        u64 b;
        u32 a;
        u8 c;
      };
      S v[5] = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}, {5, 5, 5}};
      MemoryDumper::dump(
          v, 5, 16,
          MemoryDumper::Layout()
              .withOffset(0)
              .withSizeOf<S>(5)
              .withColor(colors::console::combine(colors::console::yellow,
                                                  colors::console::dim))
              .withSubRegion( // a
                  MemoryDumper::Layout()
                      .withOffset(offsetof(S, a))
                      .withSize(sizeof(S::a), 1)
                      .withColor(colors::console::green))
              .withSubRegion( // b
                  MemoryDumper::Layout()
                      .withOffset(offsetof(S, b))
                      .withSize(sizeof(S::b), 1)
                      .withColor(colors::console::red))
              .withSubRegion( // c
                  MemoryDumper::Layout()
                      .withOffset(offsetof(S, c))
                      .withSize(sizeof(S::c), 1)
                      .withColor(colors::console::blue)),
          memory_dumper_option_bits::colored_output |
              memory_dumper_option_bits::cache_align);
    } //
    SECTION("bad alignment") {
      struct S {
        u8 c;
        u64 b;
        u32 a;
      };
      S v[5] = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}, {5, 5, 5}};
      MemoryDumper::dump(
          v, 5, 16,
          MemoryDumper::Layout()
              .withSizeOf<S>(5)
              .withColor(colors::console::combine(colors::console::yellow,
                                                  colors::console::dim))
              .withSubRegion(MemoryDumper::Layout()
                                 .withOffset(offsetof(S, a))
                                 .withSize(sizeof(S::a), 1)
                                 .withColor(colors::console::blue))
              .withSubRegion(MemoryDumper::Layout()
                                 .withOffset(offsetof(S, b))
                                 .withSize(sizeof(S::b), 1)
                                 .withColor(colors::console::green))
              .withSubRegion(MemoryDumper::Layout()
                                 .withOffset(offsetof(S, c))
                                 .withSize(sizeof(S::c), 1)
                                 .withColor(colors::console::red)),
          memory_dumper_option_bits::colored_output |
              memory_dumper_option_bits::cache_align);
    } //
    SECTION("array") {
      int a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
      MemoryDumper::dump(a, 10, 8,
                         MemoryDumper::Layout()
                             .withSizeOf<int>(10)
                             .withColor(colors::console::red)
                             .withType(DataType::I32),
                         memory_dumper_option_bits::type_values);
    } //
    SECTION("member values") {
      struct S {
        f32 a;
        i32 b;
        i16 c;
        //        i8 p[2];
      };
      S v[3] = {{0.1, 10, 1}, {0.2, 20, 2}, {0.3, 30, 3}};
      auto layout = MemoryDumper::Layout()
                        .withSizeOf<S>(3)
                        .withColor(colors::console::red)
                        .withSubRegion(MemoryDumper::Layout()
                                           .withOffset(offsetof(S, a))
                                           .withSize(sizeof(S::a), 1)
                                           .withColor(colors::console::yellow)
                                           .withType(DataType::F32))
                        .withSubRegion(MemoryDumper::Layout()
                                           .withOffset(offsetof(S, b))
                                           .withSize(sizeof(S::b), 1)
                                           .withColor(colors::console::blue)
                                           .withType(DataType::I32))
                        .withSubRegion(MemoryDumper::Layout()
                                           .withOffset(offsetof(S, c))
                                           .withSize(sizeof(S::c), 1)
                                           .withColor(colors::console::green)
                                           .withType(DataType::I16));
      MemoryDumper::dump(v, 3, 9, layout,
                         memory_dumper_option_bits::type_values |
                             memory_dumper_option_bits::colored_output);
    } //
    //  SECTION("hermes member values") {
    //    struct S {
    //      vec3 v;
    //      point2 p;
    //    };
    //    S v[3] = {{{1, 2, 3}, {4, 5}},
    //              {{10, 20, 30}, {40, 50}},
    //              {{100, 200, 300}, {400, 500}}};

    //  auto layout =
    //      MemoryDumper::Layout()
    //          .withSizeOf<S>(3)
    //          .withSubRegion(
    //              vec3::memoryDumpLayout().withColor(ConsoleColors::blue))
    //          .withSubRegion(
    //              point2::memoryDumpLayout().withColor(ConsoleColors::yellow));

    //  MemoryDumper::dump(v, 3, 8, layout,
    //                     memory_dumper_options::type_values |
    //                         memory_dumper_options::colored_output);
    //} //
    // SECTION("region construction") {
    //  MemoryDumper::Layout layout =
    //      MemoryDumper::Layout()
    //          .withSubRegion(
    //              {.field_size_in_bytes = sizeof(point2), .count = 3}, true)
    //          .withSubRegion(MemoryDumper::Layout().withSizeOf<u32>(4),
    //                         true);

    //  REQUIRE(layout.count == 1);
    //  REQUIRE(layout.offset == 0);
    //  REQUIRE(layout.field_size_in_bytes ==
    //          sizeof(point2) * 3 + sizeof(u32) * 4);
    //  REQUIRE(layout.sizeInBytes() ==
    //          layout.field_size_in_bytes * layout.count);

    //  REQUIRE(layout.sub_regions[0].sizeInBytes() == sizeof(point2) * 3);
    //  REQUIRE(layout.sub_regions[0].offset == 0);
    //  REQUIRE(layout.sub_regions[1].sizeInBytes() == sizeof(u32) * 4);
    //  REQUIRE(layout.sub_regions[1].offset ==
    //          layout.sub_regions[0].sizeInBytes());
    //} //
  } //
  // SECTION("transforms") {
  //   Transform t;
  //   MemoryDumper::dump(&t, 1, 16, Transform::memoryDumpLayout(),
  //                      memory_dumper_options::colored_output |
  //                          memory_dumper_options::type_values);
  // } //
}
