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
#ifndef TENSORFLOW_RTGLIB_CONVERT_
#define TENSORFLOW_RTGLIB_CONVERT_

#ifndef TENSORFLOW_RTGLIB_COMMON_HEADER_
#include "common_headers.h"
#endif  // TENSORFLOW_RTGLIB_COMMON_HEADER_

#include "rocm/include/rtg/program.hpp"

#define GET_INSTS_FROM_PROGRAM(prog) (prog)->instructions

namespace tensorflow {
namespace rtglib {
namespace convert {

#define MIN_CLUSTER_SIZE 3    

struct Cluster {
     explicit Cluster() { init(); }
     void addInputEdge(const Edge* edge)  { input_edges.push_back(edge);  }
     void addOutputEdge(const Edge* edge) { output_edges.push_back(edge); }
     void addNode(Node* node)             { nodes.push_back(node);        }
     int  getSize()                       { return nodes.size();          }
     std::vector<const Edge*> input_edges;
     std::vector<const Edge*> output_edges;
     std::vector<Node*> nodes;  // sorted in reversed post order.
     void init() {
         input_edges.clear();
         output_edges.clear();
         nodes.clear();
     }
};

typedef rtg::instruction T_RTG_INST;
typedef std::vector<T_RTG_INST*> T_RTG_INST_PTRS; 
typedef const std::vector<std::pair<string, Tensor>> T_INPUT_MAP; 
 
class  Converter;
using OpConverter =
    std::function<tensorflow::Status(Converter&, const tensorflow::NodeDef&, const T_RTG_INST_PTRS&)>;

using AttrEncoder=
    std::function<void(rtg::instruction&, NameAttrList&, Converter&)>;

using AttrDecoder=
    std::function<void(const NameAttrList&, Converter*, string&)>;
    
struct Converter {
    explicit Converter(rtg::program* p, T_INPUT_MAP* map) {
        Init(); program = p; inputs = map;
    }
    bool isCandidate(const Node*);
    bool isRegistered(const Node*);
    void add_instruction(const Node*);
    void add_parameter(const NodeDef&);
    void decodeAttr(const NameAttrList&);
    void getNodeType(const NodeDef&, DataType*);
    rtg::shape getNodeShape(const NodeDef&, DataType* p_dtype = nullptr);
    rtg::shape getAttrShape(const NameAttrList&);
    rtg::shape::type_t getShapeType(const DataType&);
    DataType getType(const rtg::shape::type_t&);
    void getTensorShape(const rtg::shape&, TensorShape&);
    std::unordered_map<string, OpConverter> op_registry_;
    std::unordered_map<string, AttrEncoder> attr_encoder_registry_;
    std::unordered_map<string, AttrDecoder> attr_decoder_registry_;
    void Init() {
        register_op_converters();
        register_attr_encoders();
        register_attr_decoders();
        instructions.clear();
        rtgInsNames.clear();
        rtgInsCnt.clear();
    }
    string lookupEncoder(const string);
    string lookupDecoder(const string);
    void register_op_converters();
    void register_attr_encoders();
    void register_attr_decoders();
    bool starts_with(const string&, const string&);
    string substract_prefix(const string&, const string&);
    std::unordered_map<string, rtg::instruction*> instructions;
    std::unordered_map<rtg::instruction*, string> rtgInsNames;
    std::unordered_map<string, int> rtgInsCnt;
    rtg::program* program;
    T_INPUT_MAP* inputs;
    static const string prefix;
    static const string postfix;
    string device;
};

const string Converter::prefix = "@";
const string Converter::postfix = "@"; 
 
Status AddActivation(Converter&, const NodeDef&, const T_RTG_INST_PTRS&);
Status AddBiasAdd(Converter&, const NodeDef&, const T_RTG_INST_PTRS&);
Status AddConst(Converter&, const NodeDef&, const T_RTG_INST_PTRS&);
Status AddConv2D(Converter&, const NodeDef&, const T_RTG_INST_PTRS&);
Status AddIdentity(Converter&, const NodeDef&, const T_RTG_INST_PTRS&);
Status AddMaxPool(Converter&, const NodeDef&, const T_RTG_INST_PTRS&);
Status AddScale(Converter&, const NodeDef&, const T_RTG_INST_PTRS&);
Status ConvertGraphToRTG(std::unique_ptr<Graph>*, T_INPUT_MAP*);
Status ConvertSubGraphToRTG(std::unique_ptr<Graph>*, Cluster&, T_INPUT_MAP*);
Status BuildLaunchNode(std::unique_ptr<Graph>*, Cluster&,Converter&, string&);
void SetInputAttr(rtg::instruction&, NameAttrList&, Converter&);
void SetNameAttr(rtg::instruction&, NameAttrList&, Converter&); 
void EncodeActivationAttr(rtg::instruction&, NameAttrList&, Converter&); 
void EncodeConstAttr(rtg::instruction&, NameAttrList&, Converter&); 
void EncodeConvolutionAttr(rtg::instruction&, NameAttrList&, Converter&);
void EncodeParamAttr(rtg::instruction&, NameAttrList&, Converter&);
void DecodeActivationAttr(const NameAttrList&, Converter*, string&);
void DecodeConstAttr(const NameAttrList&, Converter*, string&);
void DecodeConvolutionAttr(const NameAttrList&, Converter*, string&);
void DecodeInputAttr(T_RTG_INST_PTRS& inputs, const NameAttrList& func, Converter* convert);
void DecodeParamAttr(const NameAttrList&, Converter*, string&); 
 
} // namspace convert
} // namespace rtglib
} // namespace tensorflow

#endif // TENSORFLOW_RTGLIB_CONVERT_
#endif // TENSORFLOW_USE_ROCM
