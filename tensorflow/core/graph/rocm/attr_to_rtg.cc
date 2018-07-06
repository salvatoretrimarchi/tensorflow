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

void GetProgram(const NameAttrList& function, void ** p_program, int &bytes) {
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
    bytes = convert.next_offset;
    *p_program = program;
}

void EvalProgram(void* p_program, Tensor* output, std::vector<const Tensor*>& input_ptrs, bool use_gpu, void* scratch_mem_ptr, int size)
{
    rtg::program* program = reinterpret_cast<rtg::program*>(p_program);
    Converter convert(program, nullptr);
    rtg::shape output_shape = convert.getShape(output);
    char* output_ptr = const_cast<char*> (output->tensor_data().data());
    rtg::argument arg;
    int param_cnt = 0;
    std::unordered_map<string, rtg::argument> params;
    char* base_ptr = reinterpret_cast<char*> (scratch_mem_ptr);

    for (auto& ins : GET_INSTS_FROM_PROGRAM(program)) {
        string name = ins.op.name();
        if (convert.starts_with(name, Converter::param_prefix)) {
            name = rtg::any_cast<rtg::builtin::param>(ins.op).parameter;
            const Tensor* ptr = input_ptrs[param_cnt++];
            rtg::shape shape = convert.getShape(ptr);
            char* data = const_cast<char*> (ptr->tensor_data().data());
            rtg::argument arg = {shape, data};
            params[name] = arg;
        } else if (!use_gpu) {
            break;
        } else if (convert.starts_with(name, Converter::literal_prefix)) {
            // place literal in GPU memory
            std::string str = ins.op.name();
#if 1
            rtg::shape shape = ins.lit.get_shape();
            char * lit_ptr = base_ptr + convert.get_offset(shape);
            hipMemcpy(lit_ptr, ins.lit.data(), shape.bytes(), hipMemcpyHostToDevice);
            params[str] = {shape, lit_ptr};
#else            
            rtg::argument arg = rtg::miopen::to_gpu(ins.lit.get_argument());
            params[str] = arg;
#endif            
        }
    }
    if (!use_gpu) {
        program->compile(rtg::cpu::cpu_target{});
        arg = program->eval(params);
    } else  {
        
        auto handle = rtg::miopen::make_obj<rtg::miopen::miopen_handle>(&miopenCreate);

        params["output"] = {output_shape, output_ptr};
        params["handle"] = {rtg::shape::any_type, handle.get()};
        program->compile(rtg::miopen::miopen_target{});
        std::cout << *program << std::endl;
        arg = program->eval(params);
    }
    const TensorShape dst_shape = output->shape();    
    const rtg::shape arg_shape = arg.get_shape();
    TensorShape src_shape;
    convert.getTensorShape(arg_shape, src_shape);
    CHECK(src_shape.IsSameSize(dst_shape));
    if (!use_gpu) {
        memcpy(const_cast<char*> (output->tensor_data().data()),
               arg.cast<char>(), arg_shape.bytes());
    } else {
#if 1
        rtg::argument ret = {arg_shape, output_ptr};
        rtg::argument val = rtg::miopen::from_gpu(ret);
        float* f_ptr = val.cast<float>();
        float ele = f_ptr[0];
#endif                
        
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

} // namspace convert
} // namespace rtglib
} // namespace tensorflow 

#endif // TENSORFLOW_USE_ROCM
