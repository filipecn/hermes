#ifndef HERMES_COMMON_TIMER_H
#define HERMES_COMMON_TIMER_H

#include <chrono>
#include <ctime>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                              Timer
// *********************************************************************************************************************
/// \brief Helper class to measure time.
class Timer {
public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// \brief default constructor.
  /// \note Starts the timer.
  Timer() { tick(); }
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  /// \brief  tick
  /// \note mark current time
  void tick() { lastTick = std::chrono::high_resolution_clock::now(); }
  /// \brief  get
  /// \return elapsed time since last call to **tick** in milliseconds
  double tack() {
    auto curTick = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(curTick - lastTick)
        .count();
  }
  /// \brief  get
  /// same as calling **tack** first and then **tick**
  /// \return elapsed time since last call to **tick**
  double tackTick() {
    double elapsed = tack();
    tick();
    return elapsed;
  }

private:
  std::chrono::high_resolution_clock::time_point lastTick;
};

} // hermes namespace

#endif // HERMES_COMMON_TIMER_H
