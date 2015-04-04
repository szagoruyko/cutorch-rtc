#include <nvrtc.h>
#include <vector>
#include <iostream>
#include <type_traits>

#include "THC/THC.h"
#include "THC/THCApply.cuh"

const char* instanciate_apply1 = "                                      \n\
#include <header.h>                                                     \n\
struct Op {                                                             \n\
  __device__ __forceinline__ void operator()(float* v) {                \n\
    float& x = *v;                                                      \n\
    %s;                                                                 \n\
  }                                                                     \n\
};                                                                      \n\
extern \"C\" __global__                                                 \n\
void kernel(TensorInfo<%s> a, %s totalElements)                         \n\
{                                                                       \n\
  Op op;                                                                \n\
  THCudaTensor_pointwiseApply1<Op, %s, %d> (a, totalElements, op);      \n\
}                                                                       \n\
";

void NVRTC_CHECK(nvrtcResult result)
{
  if(result != NVRTC_SUCCESS)
    THError(nvrtcGetErrorString(result));
}

void CUDA_CHECK(CUresult result)
{
  if(result != CUDA_SUCCESS)
  {
    const char* errstr;
    cuGetErrorString(result, &errstr);
    THError(errstr);
  }
}

template <typename IndexType>
void THCudaTensor_pointwiseApply1RTC(
    TensorInfo<IndexType> aInfo,
    const char* apply_header,
    const char* op,
    IndexType totalElements,
    dim3 grid, dim3 block,
    int A)
{
  // using c++11 std::is_same here
  const char* type;
  if (std::is_same<IndexType, unsigned long>::value)
    type = "unsigned long";
  else if(std::is_same<IndexType, unsigned int>::value)
    type = "unsigned int";

  char src[2048];
  sprintf(src, instanciate_apply1, op, type, type, type, A);
  const char *headers[] = {apply_header};
  const char *includeNames[] = {"header.h"};

  nvrtcProgram program;
  NVRTC_CHECK(nvrtcCreateProgram(&program, src, NULL, 1, headers, includeNames));
  nvrtcResult result = nvrtcCompileProgram(program, 0, NULL); 
  if(result == NVRTC_ERROR_COMPILATION)
  {
    size_t logsize;
    nvrtcGetProgramLogSize(program, &logsize);
    std::vector<char> log(logsize);
    nvrtcGetProgramLog(program, log.data());
    THError(log.data());
  }
  else
    NVRTC_CHECK(result);

  size_t ptx_size;
  NVRTC_CHECK(nvrtcGetPTXSize(program, &ptx_size));
  std::vector<char> ptx(ptx_size);
  NVRTC_CHECK(nvrtcGetPTX(program, ptx.data()));

  CUmodule module;
  CUfunction func;

  CUDA_CHECK(cuModuleLoadDataEx(&module, ptx.data(), 0, NULL, NULL));
  CUDA_CHECK(cuModuleGetFunction(&func, module, "kernel"));

  void *args[] = {(void*)&aInfo, (void*)&totalElements};
  CUDA_CHECK(cuLaunchKernel(func,
                            grid.x, grid.y, grid.z,
                            block.x, block.y, block.z,
                            0, 0, args, NULL));

  CUDA_CHECK(cuModuleUnload(module));
  NVRTC_CHECK(nvrtcDestroyProgram(&program));
}

extern "C" 
bool THCudaTensor_pointwiseApply1(THCState* state,
                                  THCudaTensor* a,
                                  const char* apply_header,
                                  const char* op_string)
{
  TensorArgType aType = ReadWrite;
  long totalElements = THCudaTensor_nElement(state, a);

  if (THCudaTensor_nDimension(state, a) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (THCudaTensor_nDimension(state, a) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  const dim3 block = getApplyBlock();

  dim3 grid;
  if (!getApplyGrid(state, totalElements, grid)) {
    return false;
  }

  // If tensor args have overlapping indices and are read/write, then
  // we must expand the tensor to a contiguous form first, since
  // otherwise there are conflicting writes. Upon copying back to the
  // non-contiguous form, there will be conflicting writes, but at
  // least with copy, one of the updaters will win atomically. This is
  // a sketchy property of the old system as well (writing into all
  // indices of a tensor with overlapping indices should probably be
  // an error, since it is unclear which one should win), but we will
  // preserve this last-writer-wins (in arbitrary copy order) behavior.
  THCudaTensor* oldA = NULL;

  if (aType == ReadWrite && THC_overlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = THCudaTensor_newContiguous(state, a);
  }

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, A)                                   \
  THCudaTensor_pointwiseApply1RTC(aInfo, apply_header, op_string, (TYPE)totalElements, grid, block, A);
  //THCudaTensor_pointwiseApply1<Op, TYPE, A>                    \
  //  <<<grid, block>>>(aInfo, (TYPE) totalElements, op);

#define HANDLE_A_CASE(TYPE, A)                      \
  {                                                 \
    if (aInfo.isContiguous()) {                     \
      HANDLE_CASE(TYPE, -2);                        \
    } else {                                        \
      switch (A) {                                  \
        case 1:                                     \
          HANDLE_CASE(TYPE, 1);                     \
          break;                                    \
        case 2:                                     \
          HANDLE_CASE(TYPE, 2);                     \
          break;                                    \
        case 3:                                     \
          HANDLE_CASE(TYPE, 3);                     \
          break;                                    \
        default:                                    \
          HANDLE_CASE(TYPE, -1);                    \
          break;                                    \
      }                                             \
    }                                               \
  }

  // Can we use 32-bit integer math in the kernel (the linear ID for the copy
  // and the resulting non-linear offset is all computable using 32-bit math?)
  // We also use unsigned index math in the kernel, as signed div/mod has
  // additional overhead.
  if (THC_canUse32BitIndexMath(state, a)) {
    TensorInfo<unsigned int> aInfo(state, a);

    HANDLE_A_CASE(unsigned int, aInfo.dims);
  } else {
    TensorInfo<unsigned long> aInfo(state, a);

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (aInfo.isContiguous()) {
      THCudaTensor_pointwiseApply1RTC(aInfo, apply_header, op_string, (unsigned long)totalElements, grid, block, -2);
    } else {
      THCudaTensor_pointwiseApply1RTC(aInfo, apply_header, op_string, (unsigned long)totalElements, grid, block, -1);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_A_CASE

  if (oldA) {
    // Ignore overlaps when copying back; if we use THCudaTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    THCudaTensor_copyIgnoringOverlaps(state, oldA, a);
    THCudaTensor_free(state, a);
    a = oldA;
  }

  return true;
}

