//
// Created by Erin M Gunn on 1/17/21.
//

#include <cassert>
#include <iostream>
#include <iomanip>

#include "../radtree.h"

int test_mc(const uint32_t mc, const float3 point) {
    if (mc != radtree::utils::encode_morton_code(point)) {
        std::cout << "Encoding Failed\n";
        std::cout << "    Expected: 0x" << std::hex << std::setw(8) << std::setfill('0') << mc << '\n';
        auto actual = radtree::utils::encode_morton_code(point);
        std::cout << "    Actual: 0x" << std::hex << std::setw(8) << std::setfill('0') << actual << '\n';
        return 1;
    }
    return 0;
};

int main(void) {
    int out = 0;

    out += test_mc(0x00000000, make_float3(0.999999f, 0.99999f, 0.99999f));
    out += test_mc(0x00000001, make_float3(1.0f, 0.0f, 0.0f));
    out += test_mc(0x00000002, make_float3(0.0f, 1.0f, 0.0f));
    out += test_mc(0x00000003, make_float3(1.0f, 1.0f, 0.0f));
    out += test_mc(0x00000004, make_float3(0.0f, 0.0f, 1.0f));

    out += test_mc(1 << 29, make_float3(0.0f, 0.0f, 512.0f));
    out += test_mc(0x3FFFFFFF, make_float3(1024.0f, 1024.0f, 1024.0f));

    return out;
}