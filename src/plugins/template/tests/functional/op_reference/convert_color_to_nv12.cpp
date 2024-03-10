// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/rgb_to_nv12.hpp"

using namespace ov;
using namespace reference_tests;

class ReferenceConvertColorToNV12LayerTest : public testing::Test, public CommonReferenceTest {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        abs_threshold = 2.f;  // allow R, G, B absolute deviation to 2 (of max 255)
        threshold = 1.f;      // Ignore relative comparison (100%)
    }

public:
    template <typename T>
    static std::shared_ptr<Model> CreateFunction(const reference_tests::Tensor& input) {
        const auto in = std::make_shared<op::v0::Parameter>(input.type, input.shape);
        std::shared_ptr<Node> conv;
        conv = std::make_shared<T>(in);
        auto res = std::make_shared<op::v0::Result>(conv);
        return std::make_shared<Model>(ResultVector{res}, ParameterVector{in});
    }

    template <typename T>
    static std::shared_ptr<Model> CreateFunction2(const reference_tests::Tensor& input1,
                                                  const reference_tests::Tensor& input2) {
        const auto in1 = std::make_shared<op::v0::Parameter>(input1.type, input1.shape);
        const auto in2 = std::make_shared<op::v0::Parameter>(input2.type, input2.shape);
        std::shared_ptr<Node> conv;
        conv = std::make_shared<T>(in1, in2);
        auto res = std::make_shared<op::v0::Result>(conv);
        return std::make_shared<Model>(ResultVector{res}, ParameterVector{in1, in2});
    }
};

TEST_F(ReferenceConvertColorToNV12LayerTest, CompareWithHardcodedRefs_r_u8_single_rgb) {
    auto input = std::vector<uint8_t>{0xff, 0, 0, 0xff, 0, 0, 0xff, 0, 0, 0xff, 0, 0};
    auto input_shape = Shape{1, 2, 2, 3};
    auto exp_out = std::vector<uint8_t>{0x51, 0x51, 0x51, 0x51, 0x5a, 0xf0};
    auto out_shape = Shape{1, 3, 2, 1};
    reference_tests::Tensor inp_tensor(input_shape, element::u8, input);
    inputData = {inp_tensor.data};
    function = CreateFunction<op::v14::RGBtoNV12>(inp_tensor);
    reference_tests::Tensor exp_tensor_u8(out_shape, element::u8, exp_out);
    refOutData = {exp_tensor_u8.data};
    Exec();
}

TEST_F(ReferenceConvertColorToNV12LayerTest, CompareWithHardcodedRefs_g_fp32_single_rgb) {
    auto input = std::vector<float>{0, 255.f, 0, 0, 255.f, 0, 0, 255.f, 0, 0, 255.f, 0};
    auto input_shape = Shape{1, 2, 2, 3};
    auto exp_out = std::vector<float>{145.f, 145.f, 145.f, 145.f, 54.f, 34.f};
    auto out_shape = Shape{1, 3, 2, 1};

    reference_tests::Tensor inp_tensor(input_shape, element::f32, input);
    inputData = {inp_tensor.data};

    reference_tests::Tensor exp_tensor(out_shape, element::f32, exp_out);
    refOutData = {exp_tensor.data};

    function = CreateFunction<op::v14::RGBtoNV12>(inp_tensor);

    Exec();
}
