//
// Created by Erin M Gunn on 1/18/21.
//

#include "radtree/radtree.h"
#include "radtree/detail/direction_op.h"

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

    const int N = h_mcs.size();

    thrust::host_vector<int> expected;
    expected.push_back(N-1);
    expected.push_back(1);
    expected.push_back(1);
    expected.push_back(3);
    expected.push_back(3);
    expected.push_back(2);
    expected.push_back(1);

    thrust::device_vector<int> d_direction(N-1);
    {
        auto dir_it = thrust::make_zip_iterator(
            thrust::make_tuple(
                thrust::make_constant_iterator(d_mcs.begin()),
                thrust::make_constant_iterator(N),
                thrust::make_counting_iterator(0)
            )
        );

        thrust::transform(dir_it, dir_it + N-1, d_direction.begin(), radtree::detail::direction_op());
    }

    thrust::device_vector<int> d_max_len(N);
    {
        auto max_len_it = thrust::make_zip_iterator(
            thrust::make_tuple(
                thrust::make_constant_iterator(d_mcs.begin()),
                thrust::make_constant_iterator(N),
                d_direction.begin(),
                thrust::make_counting_iterator(0)
            )
        );

        thrust::transform(max_len_it, max_len_it + N-1, d_max_len.begin(), radtree::detail::max_len_op());
    }

    thrust::device_vector<int> actual(N);
    thrust::device_vector<int> d_actual(N);
    {
        auto len_it = thrust::make_zip_iterator(
            thrust::make_tuple(
                thrust::make_constant_iterator(d_mcs.begin()),
                thrust::make_constant_iterator(N),
                d_direction.begin(),
                d_max_len.begin(),
                thrust::make_counting_iterator(0)
            )
        );

        thrust::transform(len_it, len_it + N-1, d_actual.begin(), radtree::detail::len_op());
        thrust::copy(d_actual.begin(), d_actual.end(), actual.begin());
    }

    for (int i = 0; i < N-1; ++i) {
        std::cout << "Checking: " << i << "\n";
        if (expected[i] != actual[i]) {
            std::cout << "Failed at index: " << i << "\n";
            std::cout << "    Expected: " << expected[i] << "\n";
            std::cout << "    Actual: " << actual[i] << "\n";
        }
    }

    return 0;
}