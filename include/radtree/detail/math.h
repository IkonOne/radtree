#ifndef RADTREE_DETAIL_MATH_H
#define RADTREE_DETAIL_MATH_H

#include <cstdint>

namespace radtree::detail {
   // Assumes values of vec are in [0, 1024]
    __host__ __device__
    uint32_t encode_morton_code(const float3& vec) {

        // The first 10 bits of x are shifted to have two 0's between them
        auto left_shift = [](uint32_t x) {
            if (x == (1 << 10)) --x;
            x = (x | (x << 16)) & 0b00000011000000000000000011111111;
            x = (x | (x <<  8)) & 0b00000011000000001111000000001111;
            x = (x | (x <<  4)) & 0b00000011000011000011000011000011;
            x = (x | (x <<  2)) & 0b00001001001001001001001001001001;
            return x;
        };

        return left_shift(vec.x) | left_shift(vec.y) << 1 | left_shift(vec.z) << 2;
    }

    __host__ __device__
    int ceil_div_by_2(int val) {
        // maybe...
        // if val <= 1
        //     return 0
        // return (val & 1) + val / 2;

        int r = val / 2;
        if (r != 0 && r*2 != val)
            ++r;
        return r;
    }

    template <typename RandomIt>
    __device__
    int delta(RandomIt first, int size, int i, int j) {
        if (j < 0 || size <= j)
            return -1;

        int d = __clz(first[i] ^ first[j]);
        if (d == 32)
            d += __clz(i ^ j);
        return d;
    }
} // radtree::detail

#endif //RADTREE_DETAIL_MATH_H