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

void EvalProgram(void* p_program, const Tensor& input)
{
    

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
    convert.getLiteralFromTensor(tensor_proto, li);
    
}

} // namspace convert
} // namespace rtglib
} // namespace tensorflow 

#endif // TENSORFLOW_USE_ROCM
