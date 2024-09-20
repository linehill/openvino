// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "gemm_kernel_dbk.h"
#include "kernel_selector_utils.h"

// FIXME: set CL_HPP_TARGET_OPENCL_VERSION.
#include <CL/opencl.hpp>
// Temporary include until DBKs have been upstreamed to Khronous OpenCL headers.
#include <CL/cl_exp_defined_builtin_kernels.h>

#include <memory>

namespace kernel_selector {
ParamsKey GemmKernelDBK::GetSupportedKey() const {
    ParamsKey k;

    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);

    // Not yet supported (at least not tried yet) on PoCL.
    // k.EnableInputDataType(Datatype::INT8);
    // k.EnableInputDataType(Datatype::UINT8);
    // k.EnableInputDataType(Datatype::INT32);
    // k.EnableOutputDataType(Datatype::INT8);
    // k.EnableOutputDataType(Datatype::UINT8);
    // k.EnableOutputDataType(Datatype::INT32);

    auto enableLayout = [&](DataLayout l) -> void {
        k.EnableInputLayout(l);
        k.EnableOutputLayout(l);
    };
    enableLayout(DataLayout::bfyx);
    // More layouts are supported but not tried yet.

    // TODO: Investigate if some of these could be enabled.
    // k.EnableBatching();
    // k.EnableDifferentTypes();
    // k.EnableTensorPitches();
    // k.EnableTensorOffset();
    // k.EnableQuantization(QuantizationType::SYMMETRIC);
    // k.EnableIndirectGemm();

    return k;
}

DeviceFeaturesKey GemmKernelDBK::get_required_device_features_key(const Params& params) const {
    // TODO: Add new requirement property for DBK and require it?
    return DeviceFeaturesKey();
}

GemmKernelBase::DispatchData GemmKernelDBK::SetDefault(const gemm_params& params) const {
    return DispatchData();
}

JitConstants GemmKernelDBK::GetJitConstants(const gemm_params& params) const {
    // DBKs are not built from sources.
    return JitConstants({});
}

void GemmKernelDBK::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const gemm_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsPriority GemmKernelDBK::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_1;  // High priority for experimentation.
}

// TODO: move these to an own file.
static cl_tensor_datatype translate_to_opencl(Datatype type) {
    switch (type) {
    default:
        assert(!"Missing/unrecognized Datatype!");
        // Fall-through.
    case Datatype::UNSUPPORTED:
        return CL_TENSOR_DTYPE_UNKNOWN;

    case Datatype::UINT4:
        return CL_TENSOR_DTYPE_UINT4;
    case Datatype::INT4:
        return CL_TENSOR_DTYPE_INT4;
    case Datatype::INT8:
        return CL_TENSOR_DTYPE_INT8;
    case Datatype::UINT8:
        return CL_TENSOR_DTYPE_UINT8;
    case Datatype::INT16:
        return CL_TENSOR_DTYPE_INT16;
    case Datatype::UINT16:
        return CL_TENSOR_DTYPE_UINT16;
    case Datatype::INT32:
        return CL_TENSOR_DTYPE_INT32;
    case Datatype::UINT32:
        return CL_TENSOR_DTYPE_UINT32;
    case Datatype::INT64:
        return CL_TENSOR_DTYPE_INT64;
    case Datatype::F16:
        return CL_TENSOR_DTYPE_FP16;
    case Datatype::F32:
        return CL_TENSOR_DTYPE_FP32;
    }
    assert(!"Unreachable!");
    return CL_TENSOR_DTYPE_UNKNOWN;
}

static cl_tensor_dim translate_to_opencl(const Tensor::Dim& dim) {
    assert(!dim.pad.before && !dim.pad.after && !dim.pad.is_dynamic);
    assert(!dim.is_dynamic);
    return dim.v;
}

static void get_opencl_shape(const DataTensor& tensor, unsigned effective_rank, cl_tensor_shape* shape_out) {
    assert(tensor.GetLayout() == DataLayout::bfyx && "TODO: OV->OpenCL tensor translation for other datalayouts");

    for (unsigned i = 0; i < effective_rank; i++)
        shape_out[i] = translate_to_opencl(tensor.GetDims().at(effective_rank - i - 1));
}

static std::unique_ptr<cl_tensor_layout_blas> translate_to_blas_layout(const DataLayout& layout,
                                                                       unsigned target_rank,
                                                                       const cl_tensor_shape* shape) {
    assert(target_rank);
    auto ocl_layout = std::unique_ptr<cl_tensor_layout_blas>(new cl_tensor_layout_blas);

    switch (layout) {
    default:
        assert(!"Unrecognized or blocked layout!");
        return nullptr;
    case DataLayout::bfyx: {
        cl_tensor_dim dim_offset = 4 - target_rank;
        ocl_layout->leading_dims[0] = 3 - dim_offset;
        ocl_layout->leading_dims[1] = 2 - dim_offset;
        ocl_layout->leading_dims[2] = 1 - dim_offset;
        ocl_layout->leading_dims[3] = 0 - dim_offset;
        break;
    }
    }

    // Fill strides. This assumes that the tensor is densely packed.
    size_t prev_stride = 1;
    for (unsigned i = 0; i < target_rank; i++)
        ocl_layout->leading_strides[i] = prev_stride = prev_stride * shape[ocl_layout->leading_dims[i]];

    return ocl_layout;
}

static std::tuple<cl_tensor_desc, std::unique_ptr<cl_tensor_layout_blas>> translate_for_gemm_dbk(
    const DataTensor& cldnn_tensor) {
    // GemmKernelDBK::Validate() ensures that tensor shape is [1, 1, ..., 1, A, B, C] at most.
    unsigned effective_rank = std::min<unsigned>(cldnn_tensor.Dimentions(), 3u);

    cl_tensor_desc desc;
    desc.rank = effective_rank;
    desc.properties[0] = 0;
    assert(desc.rank < CL_MEM_MAX_TENSOR_RANK);
    get_opencl_shape(cldnn_tensor, effective_rank, desc.shape);
    desc.dtype = translate_to_opencl(cldnn_tensor.GetDType());

    auto layout_uptr = translate_to_blas_layout(cldnn_tensor.GetLayout(), effective_rank, desc.shape);

    desc.layout_type = CL_TENSOR_LAYOUT_BLAS;
    desc.layout = layout_uptr.get();

    return std::make_tuple(desc, std::move(layout_uptr));
}

KernelsData GemmKernelDBK::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        GPU_DEBUG_LOG << "Rejected GemmKernelDBK: invalid/unsupported parameters\n";
        return {};
    }

    const auto& gmm_params = static_cast<const gemm_params&>(params);
    auto entry_point = GetEntryPoint(kernelName, gmm_params.layerID, params);

    auto dispatchData = SetDefault(gmm_params);
    KernelData k_data = KernelData::Default<gemm_params>(params);
    GetUpdateDispatchDataFunc(k_data);

    auto& kernel = k_data.kernels[0];

    cl_tensor_desc lhs_tensor;
    cl_tensor_desc rhs_tensor;
    cl_tensor_desc out_tensor;
    std::unique_ptr<cl_tensor_layout_blas> lhs_layout;
    std::unique_ptr<cl_tensor_layout_blas> rhs_layout;
    std::unique_ptr<cl_tensor_layout_blas> out_layout;

    std::tie(lhs_tensor, lhs_layout) = translate_for_gemm_dbk(gmm_params.inputs[0]);
    std::tie(rhs_tensor, rhs_layout) = translate_for_gemm_dbk(gmm_params.inputs[1]);
    std::tie(out_tensor, out_layout) = translate_for_gemm_dbk(gmm_params.outputs[0]);

    cl_dbk_attributes_exp_matmul matmul_desc =
        {lhs_tensor, rhs_tensor, out_tensor, gmm_params.transpose_input0, gmm_params.transpose_input1, {}};

    // This loads intermediate binaries that needs building.
    if (!params.engineInfo.dbk_query) {
        GPU_DEBUG_LOG << "Rejected gemm-dbk: can't query/create DBKs\n";
        return {};
    }

    void* raw_program = params.engineInfo.dbk_query(POCL_CDBI_DBK_EXP_MATMUL, &matmul_desc, entry_point.c_str());
    if (!raw_program) {
        GPU_DEBUG_LOG << "Rejected gemm-dbk: either unsupported or there was an API error\n";
        return {};
    }
    auto gemm_dbk_program = static_cast<cl_program>(raw_program);

    auto kernel_string = std::make_shared<KernelString>();
    // kernel_string::{str,jit,undefs} members are empty.
    kernel_string->entry_point = entry_point;
#if CL_TARGET_OPENCL_VERSION >= 300
    kernel_string->options += " -cl-std=CL3.0";
#elif CL_TARGET_OPENCL_VERSION >= 200
    kernel_string->options += " -cl-std=CL2.0";
#endif
    kernel_string->has_microkernels = false;
    kernel_string->batch_compilation = false;

    auto deleter = [](void* to_release) -> void {
        assert(to_release);
        clReleaseProgram(static_cast<cl_program>(to_release));
    };
    auto wrapped_dbk = std::shared_ptr<void>(static_cast<void*>(gemm_dbk_program), deleter);
    kernel_string->builtin_kernels.push_back(wrapped_dbk);

    kernel.code.kernelString = kernel_string;

    // DBKs ignore global and local work sizes.
    kernel.params.workGroups.global = std::vector<size_t>(2, 1);
    kernel.params.arguments = GetArgsDesc(static_cast<uint32_t>(gmm_params.inputs.size()), false, false, 0, 1, false);

    std::cerr << "XXX selected gemm-dbk\n";  // DEBUG
    return {k_data};
}

bool GemmKernelDBK::Validate(const Params& params) const {
    if (!Parent::Validate(params))
        return false;
    const auto& gmm_params = static_cast<const gemm_params&>(params);

    if (gmm_params.activations.size() || gmm_params.fused_ops.size()) {
      GPU_DEBUG_LOG << "Rejected gemm-dbk: activations or fused ops" << std::endl;
      return false;
    }

    if (gmm_params.has_dynamic_tensors()) {
      GPU_DEBUG_LOG << "Rejected gemm-dbk: has dynamic tensors" << std::endl;
      return false;
    }

    for (const auto& io_tensors : {gmm_params.inputs, gmm_params.outputs}) {
        for (const auto& tensor : io_tensors) {
            // Stides/pitches are not defined well yet in the tensor/DBK extension. Only accept packed tensors for now.
            if (tensor.PitchesDifferFromLogicalDims()) {
                GPU_DEBUG_LOG << "Rejected gemm-dbk: a non-packed tensor" << std::endl;
                return false;
            }

            for (const auto& dim : tensor.GetDims()) {
                // Padding is not recognized consept in the tensor/DBK extension.
                if (dim.pad.before || dim.pad.after || dim.pad.is_dynamic) {
                    GPU_DEBUG_LOG << "Rejected gemm-dbk: a tensor has padding" << std::endl;
                    return false;
                }
            }

            // Check the layout is supported and can be reduced to three rank tensor (e.g. [1, N, H, W] -> [H, H, W])
            // without rearrangement of the data.
            switch (tensor.GetLayout()) {
            default:
                GPU_DEBUG_LOG << "Rejected gemm-dbk: unsupported datalayout?" << std::endl;
                return false;
            case DataLayout::bfyx:
                // IIUC, the gemm operation associates the f-dim as the batch dimension.
                if (tensor.GetDims().at(3).v != 1)
                    return false;
                break;
            }
        }
    }

    // GEMM DBK is planned but not yet implemented.
    if (gmm_params.alpha != 1.0f && gmm_params.beta != 0.0f) {
        GPU_DEBUG_LOG << "Rejected gemm-dbk: unsupported alpha or beta value" << std::endl;
        return false;
    }

    return true;
}
}  // namespace kernel_selector
