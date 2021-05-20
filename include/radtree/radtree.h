//
// Created by Erin M Gunn on 1/16/21.
//

#ifndef RADTREE_RADTREE_H
#define RADTREE_RADTREE_H

#include "radtree/detail/utils.h"
#include "radtree/detail/bounds.h"

#include <cstdint>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/unique.h>
#include <npp.h>

namespace radtree {
    namespace { // private
        struct mc_encoder {
            explicit mc_encoder(detail::bounds b)
                : b(b) {}

            detail::bounds b;

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

        detail::bounds b_;
    };

    struct builder {
        __host__
        radtree build(thrust::device_vector<float3>::iterator first, thrust::device_vector<float3>::iterator last) {
            const int N = thrust::distance(first, last);
            radtree rt;

            thrust::device_vector<uint32_t> d_mcs(N);
            {
                rt.b_ = detail::bounds::from_device(first, last);
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
