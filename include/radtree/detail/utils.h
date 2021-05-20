#ifndef RADTREE_DETAIL_UTILS_H
#define RADTREE_DETAIL_UTILS_H

#include "math.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace radtree::utils {
    /**
        *
        * Transform Functors/Operators
        *
        */

    using d_vec_mc_type = thrust::device_vector<uint32_t>;
    using h_vec_mc_type = thrust::host_vector<uint32_t>;
    using d_vec_dir_type = thrust::device_vector<int>;
    using h_vec_dir_type = thrust::host_vector<int>;

    struct max_len_op {
        template <typename TMaxLenZipIt>
        __device__
        int operator()(TMaxLenZipIt it) {
            auto mc_first = thrust::get<0>(it);
            int size = thrust::get<1>(it);
            int dir = thrust::get<2>(it);
            int idx = thrust::get<3>(it);

            const int delta_min = detail::delta(mc_first, size, idx, idx - dir);
            int max_len = 2;
            while (detail::delta(mc_first, size, idx, idx + max_len * dir) > delta_min)
                max_len = max_len << 1;

            return max_len;
        }
    };

    struct len_op {
        template <typename TLenZipIt>
        __device__
        int operator()(TLenZipIt it) {
            auto mc_first = thrust::get<0>(it);
            int size = thrust::get<1>(it);
            int dir = thrust::get<2>(it);
            int max_len = thrust::get<3>(it);
            int idx = thrust::get<4>(it);

            // TODO: this is done twice
            const int delta_min = detail::delta(mc_first, size, idx, idx - dir);

            int len = 0;
            for (int t = max_len>>1; t > 0; t >>= 1) {
                if (detail::delta(mc_first, size, idx, idx + (len + t) * dir) > delta_min)
                    len += t;
            }

            return len;
        }
    };

    struct find_split_op {
        template <typename TSplitZipIt>
        __device__
        int operator()(TSplitZipIt it) {
            auto mc_first = thrust::get<0>(it);
            int size = thrust::get<1>(it);
            int dir = thrust::get<2>(it);
            int max_len = thrust::get<3>(it);
            int len = thrust::get<4>(it);
            int idx = thrust::get<5>(it);

            const int delta_node = detail::delta(mc_first, size, idx, idx + len * dir);
            int s = 0;

            for (int t = detail::ceil_div_by_2(len); t > 0; t = detail::ceil_div_by_2(t)) {
                if (detail::delta(mc_first, size, idx, idx + (s + t) * dir) > delta_node)
                    s += t;
            }

            return idx + (s * dir) + thrust::min(dir, 0);
        }
    };
} // radtree::utils

#endif // RADTREE_DETAIL_UTILS_H