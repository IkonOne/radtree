//
// Created by Erin M Gunn on 1/17/21.
//

#include "../radtree.h"
#include "../timer.h"

#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main(void) {
    TIMER::START("Generating 32M random float3");
    thrust::host_vector<float3> h_vec(32 << 20);
    std::generate(h_vec.begin(), h_vec.end(), []() -> float3 { return { (float)rand(), (float)rand(), (float)rand() }; });
    TIMER::STOP();

    TIMER::START("Transfering data to device");
    thrust::device_vector<float3> d_vec = h_vec;
    TIMER::STOP();

    TIMER::START("Building radtree");
    radtree::builder b;
    auto rt = b.build(d_vec.begin(), d_vec.end());
    TIMER::STOP();
}