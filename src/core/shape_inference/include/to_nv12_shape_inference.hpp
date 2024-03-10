// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dimension_util.hpp"
#include "openvino/op/util/convert_color_to_nv12_base.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const util::ConvertColorToNV12Base* op, const std::vector<TShape>& input_shapes) {
    const auto has_one_input = input_shapes.size() == 1;
    NODE_VALIDATION_CHECK(op, has_one_input);

    const auto& image_shape = input_shapes[0];
    const auto image_rank = image_shape.rank();

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           image_rank.compatible(4),
                           "RGB or BGR input shall have 4 dimensions (N, H, W, C)");

    auto output_shapes = std::vector<TRShape>{image_shape};
    auto& out_shape = output_shapes.front();

    out_shape[1] *= 3;
    out_shape[1] /= 2;
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           !ov::util::dim::is_empty(out_shape[1]),
                           "Image height shall be divisible by 2");

    out_shape[3] = 1;
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           ov::util::dim::is_divisible(out_shape[1], 3),
                           "Output image height must be divisible by 3");
    NODE_SHAPE_INFER_CHECK(op, input_shapes, ov::util::dim::is_divisible(out_shape[2], 2), "Image width must be even");

    return output_shapes;
}
}  // namespace op
}  // namespace ov
