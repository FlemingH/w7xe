#pragma once

#include <chrono>

namespace best_rtpc {

class Timer {
public:
    void start() { t0_ = clock::now(); }

    double elapsed_ms() const {
        auto t1 = clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0_).count();
    }

private:
    using clock = std::chrono::high_resolution_clock;
    clock::time_point t0_;
};

}  // namespace best_rtpc
