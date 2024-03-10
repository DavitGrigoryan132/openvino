// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace op {
namespace util {
// TODO write documentation
class OPENVINO_API ConvertColorToNV12Base : public Op {
public:
    // TODO write documentation
    enum class ColorConversion : int {
        RGB_TO_NV12_SINGLE_PLANE = 0,
        BGR_TO_NV12_SINGLE_PLANE = 1,
        RGB_TO_NV12_TWO_PLANE = 2,
        BGR_TO_NV12_TWO_PLANE = 3
    };

protected:
    ConvertColorToNV12Base() = default;

    // TODO write documentation
    explicit ConvertColorToNV12Base(const Output<Node>& arg, ColorConversion format);

public:
    OPENVINO_OP("ConvertColorToNV12Base", "util");

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

protected:
    bool is_type_supported(const ov::element::Type& type) const;

    ColorConversion m_format = ColorConversion::RGB_TO_NV12_SINGLE_PLANE;
};
}  // namespace util
}  // namespace op
}  // namespace ov
