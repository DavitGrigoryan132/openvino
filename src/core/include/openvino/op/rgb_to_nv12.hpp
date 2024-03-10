// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/convert_color_to_nv12_base.hpp"

namespace ov {
namespace op {
namespace v14 {
class OPENVINO_API RGBtoNV12 : public util::ConvertColorToNV12Base {
public:
    OPENVINO_OP("RGBtoNV12", "opset14", util::ConvertColorToNV12Base);

    RGBtoNV12() = default;

    explicit RGBtoNV12(const Output<Node>& arg);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v14
}  // namespace op
}  // namespace ov