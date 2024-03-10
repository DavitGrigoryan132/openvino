// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/util/convert_color_to_nv12_base.hpp"

namespace ov {
namespace reference {

template <typename T>
std::tuple<T, T, T> rgb_pixel_to_yuv(float r_val, float g_val, float b_val) {
    auto y = 0.257f * r_val + 0.504f * g_val + 0.098f * b_val + 16.f;
    auto u = -0.148 * r_val - 0.291f * g_val + 0.439 * b_val + 128.f;
    auto v = 0.439 * r_val - 0.368f * g_val - 0.071f * b_val + 128.f;
    return std::tuple<T, T, T>{y, u, v};
}

template <typename T>
void color_convert_to_nv12(const T* ptr,
                           T* out_y,
                           T* out_uv,
                           std::size_t batch_size,
                           std::size_t image_h,
                           std::size_t image_w,
                           std::size_t stride_y,
                           std::size_t stride_uv,
                           ov::op::util::ConvertColorToNV12Base::ColorConversion type) {
    for (std::size_t batch = 0; batch < batch_size; batch++) {
        const T* image_ptr = ptr + batch * image_w * image_h;
        T* out_y_ptr = out_y + batch * stride_y;
        T* out_uv_ptr = out_uv + batch * stride_uv;
        for (std::size_t h = 0; h < image_h; h += 2) {
            for (std::size_t w = 0; w < image_w; w += 2) {
                double mean_u = 0;
                double mean_v = 0;

                std::array<std::pair<std::size_t, std::size_t>, 4> steps{std::pair<std::size_t, std::size_t>{0, 0},
                                                                         std::pair<std::size_t, std::size_t>{0, 1},
                                                                         std::pair<std::size_t, std::size_t>{1, 0},
                                                                         std::pair<std::size_t, std::size_t>{1, 1}};
                for (auto step : steps) {
                    float r_val = 0;
                    float g_val = 0;
                    float b_val = 0;

                    auto index = ((h + step.first) * image_w + (w + step.second)) * 3;
                    if (type == ov::op::util::ConvertColorToNV12Base::ColorConversion::RGB_TO_NV12_SINGLE_PLANE) {
                        r_val = static_cast<float>(image_ptr[index]);
                        g_val = static_cast<float>(image_ptr[index + 1]);
                        b_val = static_cast<float>(image_ptr[index + 2]);
                    } else if (type ==
                               ov::op::util::ConvertColorToNV12Base::ColorConversion::BGR_TO_NV12_SINGLE_PLANE) {
                        b_val = static_cast<float>(image_ptr[index]);
                        g_val = static_cast<float>(image_ptr[index + 1]);
                        r_val = static_cast<float>(image_ptr[index + 2]);
                    }
                    T y, u, v;
                    std::tie(y, u, v) = rgb_pixel_to_yuv<T>(r_val, g_val, b_val);
                    mean_u += u;
                    mean_v += v;

                    out_y_ptr[(h + step.first) * image_w + w + step.second] = y;
                }
                auto uv_index = (h / 2) * image_w + (w / 2) * w;
                out_uv_ptr[uv_index] = mean_u / 4;
                out_uv_ptr[uv_index + 1] = mean_v / 4;
            }
        }
    }
}

template <ov::element::Type_t T>
inline bool color_convert_to_nv12(const std::shared_ptr<Node>& op,
                                  ov::TensorVector& outputs,
                                  const ov::TensorVector& inputs,
                                  ov::op::util::ConvertColorToNV12Base::ColorConversion type) {
    using ET = typename ov::element_type_traits<T>::value_type;
    static const size_t N_DIM = 0;
    static const size_t H_DIM = 1;
    static const size_t W_DIM = 2;

    const auto& input_tensor = inputs[0];
    auto batch_size = input_tensor.get_shape()[N_DIM];
    auto image_w = input_tensor.get_shape()[W_DIM];
    auto image_h = input_tensor.get_shape()[H_DIM];
    if (type == ov::op::util::ConvertColorToNV12Base::ColorConversion::RGB_TO_NV12_SINGLE_PLANE or
        type == ov::op::util::ConvertColorToNV12Base::ColorConversion::BGR_TO_NV12_SINGLE_PLANE) {
        outputs[0].set_shape({batch_size, image_h * 3 / 2, image_w, 1});
        color_convert_to_nv12(input_tensor.data<ET>(),
                              outputs[0].data<ET>(),
                              outputs[0].data<ET>() + image_h * image_w,
                              batch_size,
                              image_h,
                              image_w,
                              3 / 2 * image_h * image_w,
                              3 / 2 * image_h * image_w,
                              type);
    } else {
        outputs[0].set_shape({batch_size, image_h, image_w, 1});
        outputs[1].set_shape({batch_size, image_h / 2, image_w / 2, 1});
        // TODO implement for two plane
    }
    return true;
}

}  // namespace reference
}  // namespace ov
