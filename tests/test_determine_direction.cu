//
// Created by Erin M Gunn on 1/18/21.
//

#include "../radtree.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>

int main(void) {
    thrust::host_vector<uint32_t> h_mcs;
    h_mcs.push_back(0b00001);
    h_mcs.push_back(0b00010);
    h_mcs.push_back(0b00100);
    h_mcs.push_back(0b00101);
    h_mcs.push_back(0b10011);
    h_mcs.push_back(0b11000);
    h_mcs.push_back(0b11001);
    h_mcs.push_back(0b11110);
    thrust::device_vector<uint32_t> d_mcs = h_mcs;

    thrust::host_vector<int> expected;
    expected.push_back(1);
    expected.push_back(-1);
    expected.push_back(1);
    expected.push_back(-1);
    expected.push_back(1);
    expected.push_back(1);
    expected.push_back(-1);
    expected.push_back(-1);

    thrust::device_vector<int> d_actual(expected.size());
    thrust::host_vector<int> actual(expected.size());
    {
        auto mc_it_first = thrust::make_zip_iterator(
            thrust::make_tuple(
                thrust::make_constant_iterator(d_mcs.begin()),
                thrust::make_constant_iterator(d_mcs.size()),
                thrust::make_counting_iterator(0)
            )
        );
        thrust::transform(
            mc_it_first, mc_it_first + d_mcs.size(),
            d_actual.begin(),
            radtree::utils::direction_op()
        );
        thrust::copy(d_actual.begin(), d_actual.end(), actual.begin());
    }

    int result = 0;

    for (int i = 0; i < expected.size(); ++i) {
        std::cout << "Checking: " << i << "\n";
        if (expected[i] != actual[i]) {
            std::cout << "[FAILED] at index: " << i << "\n";
            std::cout << "    Expected: " << expected[i] << "\n";
            std::cout << "    Actual: " << actual[i] << "\n";
        }
    }

    return result;
}