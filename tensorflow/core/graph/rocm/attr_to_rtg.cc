/* 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifdef TENSORFLOW_USE_ROCM
#include "attr_to_rtg.h"
#include "convert_graph.h"

namespace tensorflow {
namespace rtglib {
namespace convert {

void GetProgram(const NameAttrList& function, void ** p_program) {
    auto attr_map = function.attr();
    AttrValue value = attr_map.at("func");
    int size = value.list().func_size();
    rtg::program * program = new rtg::program;
    CHECK(program) << "Fail to create RTG program";
    
    Converter convert(program, nullptr);
    for (int i = 0; i < size; ++i) {
        const NameAttrList& func = value.list().func(i);
        convert.decodeAttr(func);
    }
    std::cout << *program << std::endl;
    *p_program = program;
}

void EvalProgram(void* p_program, std::vector<string>& param_names, Tensor* output)
{
    rtg::program* program = reinterpret_cast<rtg::program*>(p_program);
    std::unordered_map<string, rtg::argument> params;
    int size = param_names.size();
    int count = 0;
    Converter convert(program, nullptr);
    for (auto& ins : GET_INSTS_FROM_PROGRAM(program)) {
        CHECK(convert.starts_with(ins.op.name(), Converter::literal_prefix)) << "Expect literals";
        params[param_names[count]] = ins.lit.get_argument();
        if (++count == size)
            break;
    }
    program->compile(rtg::cpu::cpu_target{});
    rtg::argument arg = program->eval(params);
    const TensorShape dst_shape = output->shape();
    const rtg::shape arg_shape = arg.get_shape();
    TensorShape src_shape;
    convert.getTensorShape(arg_shape, src_shape);
    CHECK(src_shape.IsSameSize(dst_shape));
    memcpy(const_cast<char*> (output->tensor_data().data()),
           arg.cast<char>(), arg_shape.bytes());
}

void GetOutputShape(void * p_program, TensorShape& ret_shape)
{
    rtg::program* program = reinterpret_cast<rtg::program*>(p_program);
    T_RTG_INST_REF ins = program->get_last_instruction();
    rtg::shape shape = ins->result;
    Converter convert(program, nullptr);
    convert.getTensorShape(shape, ret_shape);
}

void AddInput(void * p_program, const Tensor& input)
{
    rtg::program* program = reinterpret_cast<rtg::program*>(p_program);
    Converter convert(program, nullptr);
    TensorProto tensor_proto;
    input.AsProtoTensorContent(&tensor_proto);
    rtg::literal li;
    convert.getLiteralFromTensor(tensor_proto, li, true);
    T_RTG_INST_REF ins = program->get_first_instruction();
    T_RTG_INST_REF new_ins = program->insert_literal(ins, li);
    CHECK(program->get_first_instruction() == new_ins) << "insert error";
}

void GetParamNames(void* p_program,  std::vector<string>& param_names)
{
    rtg::program* program = reinterpret_cast<rtg::program*>(p_program);
    Converter convert(program, nullptr);
    for (auto& ins : GET_INSTS_FROM_PROGRAM(program)) {
        string name = ins.op.name();
        if (convert.starts_with(name, Converter::param_prefix)) {
            name = rtg::any_cast<rtg::builtin::param>(ins.op).parameter;
            param_names.push_back(name);
        } else {
            break;
        }
    }
}

} // namspace convert
} // namespace rtglib
} // namespace tensorflow 

#endif // TENSORFLOW_USE_ROCM
