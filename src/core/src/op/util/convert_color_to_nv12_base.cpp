// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/convert_color_to_nv12_base.hpp"

#include "itt.hpp"
#include "to_nv12_shape_inference.hpp"

ov::op::util::ConvertColorToNV12Base::ConvertColorToNV12Base(const Output<Node>& arg, ColorConversion format)
    : Op({arg}),
      m_format(format) {}

void ov::op::util::ConvertColorToNV12Base::validate_and_infer_types() {
    OV_OP_SCOPE(v14_Convert_to_NV12_Base_validate_and_infer_types);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);

    auto out_type = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          is_type_supported(out_type),
                          "Input type shall have u8 or floating-point precision, got ",
                          out_type);

    set_output_type(0, out_type, output_shapes.front());
}

bool ov::op::util::ConvertColorToNV12Base::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

bool ov::op::util::ConvertColorToNV12Base::is_type_supported(const ov::element::Type& type) const {
    return type.is_dynamic() || type.is_real() || type == ov::element::u8;
}
