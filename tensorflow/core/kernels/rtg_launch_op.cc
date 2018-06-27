/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/rtg_launch_op.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace gpu = perftools::gputools;

namespace tensorflow {

RTGLaunchOp::RTGLaunchOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  const NameAttrList* func;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("function", &func));
  program = nullptr;
  rtglib::convert::GetProgram(*func, &program);
}

void RTGLaunchOp::Compute(OpKernelContext* ctx) {
  VLOG(1) << "RTGLaunchOp::Compute ";
  
  gpu::Stream* stream =
      ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr;

    // Execute the computation.
    VLOG(2) << "Executing computation.";

    std::vector<string> param_names;
    rtglib::convert::GetParamNames(program, param_names);
    unsigned input_size = ctx->num_inputs();
    OP_REQUIRES(ctx, input_size == param_names.size(),
                errors::InvalidArgument("unmatched parameters and inputs"));

    std::vector<const Tensor*> input_ptrs;
    for (unsigned i = 0; i < input_size; ++i) {
        const Tensor& input = ctx->input(i);
        input_ptrs.push_back(&input);
    }
    OP_REQUIRES(ctx, ctx->num_outputs() == 1,
                errors::InvalidArgument("expect single output"));
    
    TensorShape output_shape;
    rtglib::convert::GetOutputShape(program, output_shape);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    Device* device = static_cast<Device*>(ctx->device());
    bool use_gpu = (device->device_type() == DEVICE_GPU) ? true : false;
    rtglib::convert::EvalProgram(program, param_names, output, input_ptrs, use_gpu);
    ctx->set_output(0, *output);
    
#if 0    
    auto start_time = env->NowMicros();
    auto run_result = executable->Run(arg_ptrs, run_options);
    OP_REQUIRES(ctx, run_result.ok(), run_result.status());
    output = std::move(run_result.ValueOrDie());
    auto elapsed = env->NowMicros() - start_time;
    VLOG(2) << "Elapsed time: " << elapsed << "us";
    int output_num = 0;
    for (int i = 0; i < ctx->num_outputs(); ++i) {
        if (kernel->outputs[i].is_constant) {
            // Output is a constant
            const Tensor& const_tensor = kernel->outputs[i].constant_value;
            const size_t total_bytes = const_tensor.TotalBytes();
            if (stream && total_bytes > 0) {
                // Copy host -> device. (Empty tensors don't have backing buffers.)
                VLOG(1) << "Constant output tensor on device";
                Tensor* output_tensor;
                TF_CHECK_OK(
                            ctx->allocate_output(i, const_tensor.shape(), &output_tensor));
                
                const void* src_ptr = DMAHelper::base(&const_tensor);
                void* dst_ptr = DMAHelper::base(output_tensor);
                gpu::DeviceMemoryBase gpu_dst_ptr(dst_ptr, total_bytes);
                stream->ThenMemcpy(&gpu_dst_ptr, src_ptr, total_bytes);
            } else {
                // No copy required.
                ctx->set_output(i, const_tensor);
            }
        }
    }
#endif
  VLOG(1) << "Done";
}

RTGLaunchOp::~RTGLaunchOp() {
  VLOG(1) << "RTGLaunchOp destroyed";
}

//TODO: back to GPU
REGISTER_KERNEL_BUILDER(Name("RTGLaunchOp").Device(DEVICE_GPU), RTGLaunchOp);
// REGISTER_KERNEL_BUILDER(Name("RTGLaunchOp").Device(DEVICE_CPU), RTGLaunchOp);    

}  // namespace tensorflow

#endif // TENSORFLOW_USE_ROCM
