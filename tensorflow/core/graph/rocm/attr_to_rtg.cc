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

void EvalProgram(void* p_program, std::vector<string>& param_names, Tensor* output, std::vector<const Tensor*>& input_ptrs, bool use_gpu)
{
    rtg::program* program = reinterpret_cast<rtg::program*>(p_program);
    std::unordered_map<string, rtg::argument> params;
    int size = param_names.size();
    Converter convert(program, nullptr);
    if (!use_gpu) {
        for (int i = 0; i < size; ++i) {
            TensorProto tensor_proto;
            input_ptrs[i]->AsProtoTensorContent(&tensor_proto);
            rtg::literal li;
            convert.getLiteralFromTensor(tensor_proto, li, false);
            params[param_names[i]] = li.get_argument();
        }
        program->compile(rtg::cpu::cpu_target{});
    } else {
        for (int i = 0; i < size; ++i) {
            const Tensor* ptr = input_ptrs[i];
            rtg::shape shape = convert.getShape(ptr);
            char* data = const_cast<char*> (ptr->tensor_data().data());
            rtg::argument arg = {shape, data};
            params[param_names[i]] = arg;
        }
        program->compile(rtg::miopen::miopen_target{});
        auto handle = rtg::miopen::make_obj<rtg::miopen::miopen_handle>(&miopenCreate);
        params["handle"] = {rtg::shape::any_type, handle.get()};
    } 
    rtg::argument arg = program->eval(params);
    const TensorShape dst_shape = output->shape();
    const rtg::shape arg_shape = arg.get_shape();
    TensorShape src_shape;
    convert.getTensorShape(arg_shape, src_shape);
    CHECK(src_shape.IsSameSize(dst_shape));
#if 0    
    float* f_ptr = arg.cast<float>();
    size = arg_shape.bytes()/sizeof(float);
#endif
    if (!use_gpu) {
        memcpy(const_cast<char*> (output->tensor_data().data()),
               arg.cast<char>(), arg_shape.bytes());
    } else {

    }
}

void GetOutputShape(void * p_program, TensorShape& ret_shape)
{
    rtg::program* program = reinterpret_cast<rtg::program*>(p_program);
    T_RTG_INST_REF ins = std::prev(program->end());
    rtg::shape shape = ins->result;
    Converter convert(program, nullptr);
    convert.getTensorShape(shape, ret_shape);
}

#if 0    
void AddInput(void * p_program, const Tensor& input)
{
    rtg::program* program = reinterpret_cast<rtg::program*>(p_program);
    Converter convert(program, nullptr);
    TensorProto tensor_proto;
    input.AsProtoTensorContent(&tensor_proto);
    rtg::literal li;
    convert.getLiteralFromTensor(tensor_proto, li, false);
    T_RTG_INST_REF ins = program->begin();
    T_RTG_INST_REF new_ins = program->insert_literal(ins, li);
    CHECK(program->begin() == new_ins) << "insert error";
    std::cout << *program << std::endl;
}
#endif    

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
