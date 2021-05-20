#ifndef RADTREE_DETAIL_H
#define RADTREE_DETAIL_H

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

namespace radtree::detail {

struct bounds {
    bounds() = default;
    ~bounds() = default;
    bounds(const bounds&) = default;

    __host__ __device__
    explicit bounds(const float3& val)
        : min(val), max(val) { }

    float3 min, max;

    // maps the value to [0,1024]
    __device__
    float3 map_to_1024(const float3& val) {
        return {
            (val.x - min.x) / (max.x - min.x) * 1024.0f,
            (val.y - min.y) / (max.y - min.y) * 1024.0f,
            (val.z - min.z) / (max.z - min.z) * 1024.0f
        };
    }

    struct float3_to_bounds_op {
        __device__
        bounds operator()(const float3& f3) {
            return bounds(f3);
        }
    };

    struct union_op {
        __device__
        bounds operator()(const bounds& lhs, const bounds& rhs) {
            bounds b;
            b.min = { thrust::min(lhs.min.x, rhs.min.x), thrust::min(lhs.min.y, rhs.min.y), thrust::min(lhs.min.z, rhs.min.z) };
            b.max = { thrust::max(lhs.max.x, rhs.max.x), thrust::max(lhs.max.y, rhs.max.y), thrust::max(lhs.max.z, rhs.max.z) };
            return b;
        }
    };

    static
    __host__
    bounds from_device(thrust::device_vector<float3>::iterator first, thrust::device_vector<float3>::iterator last) {
        bounds init(*first);
        auto result = thrust::transform_reduce(first, last, float3_to_bounds_op(), init, union_op());
        return result;
    }
};

} // radtree::detail

#endif // RADTREE_DETAIL_H