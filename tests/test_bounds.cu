//
// Created by Erin M Gunn on 1/16/21.
//

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <algorithm>
#include "radtree/radtree.h"
#include "../timer.h"

int main(void)
{
    std::cout << "sizeof(float3) = " << sizeof(float3) << " Bytes \n";
    std::cout << "sizeof(float3) * 32M = " << sizeof(float3) * (32 << 20) << " Bytes \n";
    std::cout << sizeof(float3) * 32 << " MB \n";

    TIMER::START("Generating 32M random float3");
    thrust::host_vector<float3> h_vec(32 << 20);
    std::generate(h_vec.begin(), h_vec.end(), []() -> float3 { return { (float)rand(), (float)rand(), (float)rand() }; });
    TIMER::STOP();

    TIMER::START("Transfering data to device");
    thrust::device_vector<float3> d_vec = h_vec;
    TIMER::STOP();

    TIMER::START("Reducing a bounding box containing all of the random float3's");
    auto b = radtree::utils::bounds::from_device(d_vec.begin(), d_vec.end());
    TIMER::STOP();

    std::cout << '\n';
    std::cout << "Bounds\n";
    std::cout << "Min: " << b.min.x << " " << b.min.y << " " << b.min.z << '\n';
    std::cout << "Max: " << b.max.x << " " << b.max.y << " " << b.max.z << '\n';
    std::cout << '\n';

    // transfer data back to host
    TIMER::START("Transfer data to host");
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
    TIMER::STOP();

    return 0;
}
