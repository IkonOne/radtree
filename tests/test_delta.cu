//
// Created by Erin M Gunn on 1/18/21.
//

#include "radtree/radtree.h"
#include "../timer.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>

using namespace radtree::utils;

int expect_eq(int expected, int actual) {
    std::cout << "Checking ( " << expected << " vs " << actual << " )\n";
    if (expected != actual) {
        std::cout << "Longest common prefix is incorrect.\n";
        std::cout << "    Expected: " << expected << "\n";
        std::cout << "    Actual: " << actual << "\n";
        return 1;
    }
    return 0;
}

struct delta_op {
    template <typename TTestTuple, typename TMCTuple>
    __device__
    int operator()(TTestTuple idx_tuple, TMCTuple mc_tuple) {
        return delta(
            thrust::get<0>(mc_tuple), thrust::get<1>(mc_tuple),
            thrust::get<0>(idx_tuple), thrust::get<1>(idx_tuple)
        );
    }
};

int main(void) {
    // sorted morton codes
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

    thrust::host_vector<int> h_expected;
    thrust::host_vector<thrust::pair<int, int>> h_tests;

    h_expected.push_back(-1);
    h_tests.push_back(thrust::make_pair(0, -1));
    h_expected.push_back(-1);
    h_tests.push_back(thrust::make_pair(0, h_mcs.size()));


    h_expected.push_back(30);
    h_tests.push_back(thrust::make_pair(0, 1));
    h_expected.push_back(29);
    h_tests.push_back(thrust::make_pair(1, 2));
    h_expected.push_back(31);
    h_tests.push_back(thrust::make_pair(2, 3));
    h_expected.push_back(27);
    h_tests.push_back(thrust::make_pair(3, 4));
    h_expected.push_back(28);
    h_tests.push_back(thrust::make_pair(4, 5));
    h_expected.push_back(31);
    h_tests.push_back(thrust::make_pair(5, 6));
    h_expected.push_back(29);
    h_tests.push_back(thrust::make_pair(6, 7));

    thrust::device_vector<thrust::pair<int,int>> d_tests = h_tests;

    thrust::device_vector<int> d_actual(d_tests.size());
    thrust::transform(
        d_tests.begin(), d_tests.end(),
        thrust::make_constant_iterator(thrust::make_pair(d_mcs.begin(), d_mcs.size())),
        d_actual.begin(),
        delta_op()
    );

    thrust::host_vector<int> h_actual = d_actual;

    for (int i = 0; i < h_tests.size(); ++i)
        expect_eq(h_expected[i], h_actual[i]);

//    infos[0] = 0;
//    infos[1] = 0xFFFFFFFF;
//    expect_eq(0, delta(infos, size, 0, 1));
}