//
// Created by Erin M Gunn on 1/17/21.
//

#ifndef RADTREE_TIMER_H
#define RADTREE_TIMER_H

#include <chrono>
#include <iostream>
#include <iomanip>

namespace TIMER {
namespace {
    static std::chrono::time_point<std::chrono::steady_clock> time_stamp;
}

inline void START(const std::string& msg) {
    std::cout << "====================\n";
    std::cout << "Benchmark: " << msg << '\n';
    time_stamp = std::chrono::steady_clock::now();
}

inline void START() {
    return START("");
}

inline void STOP() {
    std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - time_stamp;

    std::cout << "Time: " << std::fixed << std::setprecision(4) <<  elapsed.count() << " ms\n";
    std::cout << "--------------------\n";
}

}

#endif //RADTREE_TIMER_H
