//
// Created by Erin M Gunn on 1/16/21.
//

#ifndef RADTREE_RADTREE_H
#define RADTREE_RADTREE_H

#include <cstdint>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/unique.h>
#include <npp.h>

namespace radtree {
    namespace utils {
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

        /**
         *
         * Transform Functors/Operators
         *
         */

        using d_vec_mc_type = thrust::device_vector<uint32_t>;
        using h_vec_mc_type = thrust::host_vector<uint32_t>;
        using d_vec_dir_type = thrust::device_vector<int>;
        using h_vec_dir_type = thrust::host_vector<int>;

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

        struct max_len_op {
            template <typename TMaxLenZipIt>
            __device__
            int operator()(TMaxLenZipIt it) {
                auto mc_first = thrust::get<0>(it);
                int size = thrust::get<1>(it);
                int dir = thrust::get<2>(it);
                int idx = thrust::get<3>(it);

                const int delta_min = delta(mc_first, size, idx, idx - dir);
                int max_len = 2;
                while (delta(mc_first, size, idx, idx + max_len * dir) > delta_min)
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
                const int delta_min = delta(mc_first, size, idx, idx - dir);

                int len = 0;
                for (int t = max_len>>1; t > 0; t >>= 1) {
                    if (delta(mc_first, size, idx, idx + (len + t) * dir) > delta_min)
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

                const int delta_node = delta(mc_first, size, idx, idx + len * dir);
                int s = 0;

                for (int t = ceil_div_by_2(len); t > 0; t = ceil_div_by_2(t)) {
                    if (delta(mc_first, size, idx, idx + (s + t) * dir) > delta_node)
                        s += t;
                }

                return idx + (s * dir) + thrust::min(dir, 0);
            }
        };
    } // utils

    namespace { // private
        struct mc_encoder {
            explicit mc_encoder(utils::bounds b)
                : b(b) {}

            utils::bounds b;

            __device__
            uint32_t operator()(const float3& val) {
                return utils::encode_morton_code(b.map_to_1024(val));
            }
        };


    } // private

    class radtree {
        friend class builder;
    public:
        radtree() = default;
        ~radtree() = default;
        radtree(const radtree&) = default;

    private:
        struct inner_node {
            int parent = -1;
            int children[8] = {-1};
            uint32_t morton_code;
        };

        struct leaf {
            int parent = -1;
        };

        utils::bounds b_;
    };

    struct builder {
        __host__
        radtree build(thrust::device_vector<float3>::iterator first, thrust::device_vector<float3>::iterator last) {
            const int N = thrust::distance(first, last);
            radtree rt;

            thrust::device_vector<uint32_t> d_mcs(N);
            {
                rt.b_ = utils::bounds::from_device(first, last);
                thrust::transform(first, last, d_mcs.begin(), mc_encoder(rt.b_));
            }

            thrust::device_vector<int> d_direction(N-1);
            {
                auto dir_it = thrust::make_zip_iterator(
                    thrust::make_tuple(
                        thrust::make_constant_iterator(d_mcs.begin()),
                        thrust::make_constant_iterator(N),
                        thrust::make_counting_iterator(0)
                    )
                );

                thrust::transform(dir_it, dir_it + N-1, d_direction.begin(), utils::direction_op());
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

                thrust::transform(max_len_it, max_len_it + N-1, d_max_len.begin(), utils::max_len_op());
            }

            thrust::device_vector<int> d_len(N);
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

                thrust::transform(len_it, len_it + N-1, d_len.begin(), utils::len_op());
            }

            thrust::device_vector<int> d_split(N);
            {
                auto split_it = thrust::make_zip_iterator(
                    thrust::make_tuple(
                        thrust::make_constant_iterator(d_mcs.begin()),
                        thrust::make_constant_iterator(N),
                        d_direction.begin(),
                        d_max_len.begin(),
                        d_len.begin(),
                        thrust::make_counting_iterator(0)
                    )
                );

                thrust::transform(split_it, split_it + N-1, d_split.begin(), utils::find_split_op());
            }

            return rt;
        }
    };
} // radtree

#endif //RADTREE_RADTREE_H
