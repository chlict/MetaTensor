#pragma once

#include "Utils.hpp"
#include "Tensor.hpp"
#include "xforms/GenIR.hpp"
#include "xforms/AllocTensor.hpp"
#include "xforms/CodeGen.hpp"
#include <iostream>

struct ExprCompiler {
    // TODO: check AST type
    template <typename AST>
    static constexpr auto compile(AST &&ast) {
        auto ir1 = GenIR().transform(ast);
        auto ir2 = AllocTensor().transform(ir1);
        auto codes = CodeGenPass().transform(ir2);
        return codes;
    }
};