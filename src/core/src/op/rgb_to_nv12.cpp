// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/rgb_to_nv12.hpp"

#include "itt.hpp"

ov::op::v14::RGBtoNV12::RGBtoNV12(const ov::Output<ov::Node>& arg)
    : util::ConvertColorToNV12Base(arg, util::ConvertColorToNV12Base::ColorConversion::RGB_TO_NV12_SINGLE_PLANE) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> ov::op::v14::RGBtoNV12::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v14_RGBtoNV12_clone_with_new_inputs);
    OPENVINO_ASSERT(new_args.size() == 1, "RGBtoNV12 shall have one input node");
    return std::make_shared<RGBtoNV12>(new_args.at(0));
}
