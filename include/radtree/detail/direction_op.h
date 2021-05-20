#ifndef RADTREE_DETAIL_DIRECTION_OP_H
#define RADTREE_DETAIL_DIRECTION_OP_H

#include "radtree/detail/math.h"

#include <thrust/tuple.h>

namespace radtree::detail {
    struct direction_op {
        template <typename TDirectionZip>
        __device__
        int operator()(TDirectionZip it) {
            auto mc_first = thrust::get<0>(it);
            int size = thrust::get<1>(it);
            int idx = thrust::get<2>(it);

            auto diff = delta(mc_first, size, idx, idx+1) - delta(mc_first, size, idx, idx-1);
            return diff > 0 ? 1 : -1;
        }
    };
} // radtree::detail

#endif // RADTREE_DETAIL_DIRECTION_OP_H