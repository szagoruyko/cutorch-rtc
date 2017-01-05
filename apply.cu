#include <vector>
#include <memory>
#include <unordered_map>
#include <iostream>
#include <type_traits>
#include <cuda_fp16.h>

#include "THC/THC.h"
#include "THC/THCApply.cuh"

#include "compile_ptx.h"
#include "apply.h"

struct ApplyHash {
  std::string op;
  std::string index_type;
  std::string ta_type;
  int dim;

  ApplyHash(std::string op, std::string ta_type, std::string index_type, int Adim) :
    op(op),
    ta_type(ta_type),
    index_type(index_type),
    dim(Adim)
  {}

  ApplyHash(std::string op, std::string ta_type, std::string index_type, int Adim, int Bdim) :
    op(op),
    ta_type(ta_type),
    index_type(index_type),
    dim(Adim + 64 * Bdim)
  {}

  ApplyHash(std::string op, std::string ta_type, std::string index_type, int Adim, int Bdim, int Cdim) :
    op(op),
    ta_type(ta_type),
    index_type(index_type),
    dim(Adim + 64 * Bdim + 64 * 64 * Cdim)
  {}

  bool operator == (const ApplyHash& other) const
  {
    return ta_type == other.ta_type &&
           index_type == other.index_type &&
           dim == other.dim &&
           op == other.op;
  }
};

namespace std
{
  template<>
  struct hash<ApplyHash>
  {
    typedef ApplyHash argument_type;
    typedef std::size_t result_type;

    result_type operator()(argument_type const& s) const
    {
      result_type const h1 ( std::hash<std::string>()(s.op));
      result_type const h2 ( std::hash<std::string>()(s.ta_type));
      result_type const h3 ( std::hash<std::string>()(s.index_type));
      result_type const h4 ( std::hash<int>()(s.dim));
      return ((h1 ^ (h2 << 1)) ^ (h3 << 1)) ^ (h4 << 1);
    }
  };
}

typedef std::vector<char> PTX;
typedef std::shared_ptr<PTX> PTXPtr;

typedef std::unordered_map<ApplyHash, std::shared_ptr<PTX>> ApplyCache;

ApplyCache applycache;

template <class T>
const char* getTypeString()
{
  if (std::is_same<T, unsigned long>::value)
    return "unsigned long";
  else if (std::is_same<T, unsigned int>::value)
    return "unsigned int";
  else if (std::is_same<T, double>::value)
    return "double";
  else if (std::is_same<T, float>::value)
    return "float";
  else if (std::is_same<T, half>::value)
    return "half";
  return "";
}

// Example op: 'x = y*2'
const char* instanciate_apply1 = "                                      \n\
#include <%s>                                                     \n\
typedef %s Ta;                                                          \n\
typedef %s IndexType; 							\n\
struct Op {                                                             \n\
    __device__ __forceinline__ void operator()(Ta* __v) {                \n\
    Ta& x = *__v;                                                      \n\
    %s;                                                                 \n\
  }                                                                     \n\
};                                                                      \n\
extern \"C\" __global__                                                 \n\
void kernel(TensorInfo<Ta, IndexType> a, IndexType totalElements)       \n\
{                                                                       \n\
  Op op;                                                                \n\
  kernelPointwiseApply1<Op,Ta,IndexType,%d> (a, totalElements, op);	\n\
}                                                                       \n\
";

template <typename Ta, typename IndexType>
void kernelPointwiseApply1RTC(
    TensorInfo<Ta, IndexType> aInfo,
    const char* op,
    IndexType totalElements,
    dim3 grid, dim3 block,
    int A,
    cudaStream_t stream)
{
  // using c++11 std::is_same here
  const char* index_type = getTypeString<IndexType>();
  const char* ta_type = getTypeString<Ta>();

  char src[2048];
  const char *headers[] = {THCApplyRTC_cuh};
  const char *includeNames[] = {"THCApplyRTC.cuh"};
  sprintf(src, instanciate_apply1, includeNames[0], ta_type, index_type, op, A);

  PTXPtr ptx;
  ApplyHash hash(op, ta_type, index_type, A);
  auto found_hash = applycache.find(hash);
  if(found_hash == applycache.end())
  {
    ptx = PTXPtr(new PTX());
    compilePTX(src, headers, includeNames, *ptx);
    applycache.emplace(hash, ptx);
  }
  else
    ptx = found_hash->second;

  void *args[] = {(void*)&aInfo, (void*)&totalElements};
  launch(ptx->data(), "kernel", args, grid, block, (CUstream)stream);
}

// Example op: 'x = x*y'
const char* instanciate_apply2 = "                                      \n\
#include <%s>                                                           \n\
typedef %s Ta;                                                          \n\
typedef %s Tb;                                                          \n\
typedef %s IndexType; 							\n\
struct Op {                                                             \n\
  __device__ __forceinline__ void operator()(Ta* __a, Tb* __b) {     	\n\
    Ta& x = *__a;                                                      \n\
    Tb& y = *__b;							\n\
    %s;                                                                 \n\
  }                                                                     \n\
};                                                                      \n\
extern \"C\" __global__                                                 \n\
void kernel(TensorInfo<Ta, IndexType> a,                                \n\
            TensorInfo<Tb, IndexType> b,                                \n\
            IndexType totalElements)                                    \n\
{                                                                       \n\
  Op op;                                                                \n\
  kernelPointwiseApply2<Op,Ta,Tb,IndexType,%d,%d>                       \n\
                        (a, b, totalElements, op);                      \n\
}                                                                       \n\
";

template <typename Ta, typename Tb, typename IndexType>
void kernelPointwiseApply2RTC(
    TensorInfo<Ta, IndexType> aInfo,
    TensorInfo<Tb, IndexType> bInfo,
    const char* op,
    IndexType totalElements,
    dim3 grid, dim3 block,
    int A, int B,
    cudaStream_t stream)
{
  // using c++11 std::is_same here
  const char* index_type = getTypeString<IndexType>();
  const char* ta_type = getTypeString<Ta>();
  const char* tb_type = getTypeString<Tb>();

  char src[2048];
  const char *headers[] = {THCApplyRTC_cuh};
  const char *includeNames[] = {"THCApplyRTC.cuh"};
  sprintf(src, instanciate_apply2, includeNames[0], ta_type, tb_type, index_type, op, A, B);

  PTXPtr ptx;
  ApplyHash hash(op, std::string(ta_type)+"_"+tb_type, index_type, A, B);
  auto found_hash = applycache.find(hash);
  if(found_hash == applycache.end())
  {
    ptx = PTXPtr(new PTX());
    compilePTX(src, headers, includeNames, *ptx);
    applycache.emplace(hash, ptx);
  }
  else
    ptx = found_hash->second;

  void *args[] = {(void*)&aInfo, (void*)&bInfo, (void*)&totalElements};
  launch(ptx->data(), "kernel", args, grid, block, (CUstream)stream);
}


// Example op: 'x = y*z'
const char* instanciate_apply3 = "                                      \n\
#include <%s>                                                           \n\
typedef %s Ta;                                                          \n\
typedef %s Tb;                                                          \n\
typedef %s Tc;                                                          \n\
typedef %s IndexType; 							\n\
struct Op {                                                             \n\
  __device__ __forceinline__						\n\
  void operator()(Ta* a, Tb* b, Tc *c) {            			\n\
    Ta& x = *a;                                                      \n\
    Tb& y = *b;							\n\
    Tc& z = *c;							\n\
    %s;                                                                 \n\
  }                                                                     \n\
};                                                                      \n\
extern \"C\" __global__                                                 \n\
void kernel(TensorInfo<Ta, IndexType> a,						\n\
    	    TensorInfo<Tb, IndexType> b,						\n\
    	    TensorInfo<Tc, IndexType> c,						\n\
	    IndexType totalElements)       				\n\
{                                                                       \n\
  Op op;                                                                \n\
  kernelPointwiseApply3<Op,Ta,Tb,Tc,IndexType,%d,%d,%d>                 \n\
  			(a, b, c, totalElements, op);			\n\
}                                                                       \n\
";

template <typename Ta, typename Tb, typename Tc, typename IndexType>
void kernelPointwiseApply3RTC(
    TensorInfo<Ta, IndexType> aInfo,
    TensorInfo<Tb, IndexType> bInfo,
    TensorInfo<Tc, IndexType> cInfo,
    const char* op,
    IndexType totalElements,
    dim3 grid, dim3 block,
    int A, int B, int C,
    cudaStream_t stream)
{
  // using c++11 std::is_same here
  const char* index_type = getTypeString<IndexType>();
  const char* ta_type = getTypeString<Ta>();
  const char* tb_type = getTypeString<Tb>();
  const char* tc_type = getTypeString<Tc>();

  char src[4096];
  const char *headers[] = {THCApplyRTC_cuh};
  const char *includeNames[] = {"THCApplyRTC.cuh"};
  sprintf(src, instanciate_apply3, includeNames[0], ta_type, tb_type, tc_type, index_type, op, A, B, C);

  PTXPtr ptx;
  ApplyHash hash(op, std::string(ta_type)+"_"+tb_type+"_"+tc_type, index_type, A, B, C);
  auto found_hash = applycache.find(hash);
  if(found_hash == applycache.end())
  {
    ptx = PTXPtr(new PTX());
    compilePTX(src, headers, includeNames, *ptx);
    applycache.emplace(hash, ptx);
  }
  else
    ptx = found_hash->second;

  void *args[] = {(void*)&aInfo, (void*)&bInfo, (void*)&cInfo, (void*)&totalElements};
  launch(ptx->data(), "kernel", args, grid, block, (CUstream)stream);
}


template <typename TensorTypeA>
bool THC_pointwiseApply1(THCState* state,
                         TensorTypeA* a,
                         const char* op_string)
{
  TensorArgType aType = ReadWrite;
  auto stream = THCState_getCurrentStream(state);
  if (TensorUtils<TensorTypeA>::getDims(state, a) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (TensorUtils<TensorTypeA>::getDims(state, a) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  const dim3 block = getApplyBlock();

  dim3 grid;
  long totalElements = TensorUtils<TensorTypeA>::getNumElements(state, a);

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
  TensorTypeA* oldA = NULL;

  if (aType == ReadWrite &&
      TensorUtils<TensorTypeA>::overlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = TensorUtils<TensorTypeA>::newContiguous(state, a);
  }

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, A)                                            \
  kernelPointwiseApply1RTC<typename TensorUtils<TensorTypeA>::DataType, TYPE> \
      (aInfo, op_string, (TYPE) totalElements, grid, block, A, stream);

#define HANDLE_A_CASE(TYPE, A)                  \
  {                                             \
    if (aInfo.isContiguous()) {                 \
      HANDLE_CASE(TYPE, -2);                    \
    } else {                                    \
      switch (A) {                              \
        case 1:                                 \
        HANDLE_CASE(TYPE, 1);                   \
        break;                                  \
        case 2:                                 \
        HANDLE_CASE(TYPE, 2);                   \
        break;                                  \
        default:                                \
        HANDLE_CASE(TYPE, -1);                  \
        break;                                  \
      }                                         \
    }                                           \
  }

  // Can we use 32-bit integer math in the kernel (the linear ID for the copy
  // and the resulting non-linear offset is all computable using 32-bit math?)
  // We also use unsigned index math in the kernel, as signed div/mod has
  // additional overhead.
  if (TensorUtils<TensorTypeA>::canUse32BitIndexMath(state, a)) {
    TensorInfo<typename TensorUtils<TensorTypeA>::DataType, unsigned int> aInfo =
      getTensorInfo<TensorTypeA, unsigned int>(state, a);
    aInfo.collapseDims();

    HANDLE_A_CASE(unsigned int, aInfo.dims);
  } else {
    TensorInfo<typename TensorUtils<TensorTypeA>::DataType, unsigned long> aInfo =
      getTensorInfo<TensorTypeA, unsigned long>(state, a);
    aInfo.collapseDims();

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (aInfo.isContiguous()) {
      kernelPointwiseApply1RTC<typename TensorUtils<TensorTypeA>::DataType, unsigned long>
        (aInfo, op_string, totalElements, grid, block, -2, stream);
    } else {
      kernelPointwiseApply1RTC<typename TensorUtils<TensorTypeA>::DataType, unsigned long>
        (aInfo, op_string, totalElements, grid, block, -1, stream);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_A_CASE

  if (oldA) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    TensorUtils<TensorTypeA>::copyIgnoringOverlaps(state, oldA, a);
    TensorUtils<TensorTypeA>::free(state, a);
    a = oldA;
  }

  return true;
}


#define THC_POINTWISE_APPLY1(TYPE) \
  extern "C" \
  bool TH_CONCAT_2(TYPE, _pointwiseApply1)(THCState* state, \
                                    TYPE* a, \
                                    const char* op_string) { \
    return THC_pointwiseApply1<TYPE>(state, a, op_string); \
  }

THC_POINTWISE_APPLY1(THCudaTensor)
THC_POINTWISE_APPLY1(THCudaDoubleTensor)
THC_POINTWISE_APPLY1(THCudaHalfTensor)


template <typename TensorTypeA,
          typename TensorTypeB>
bool THC_pointwiseApply2(THCState* state,
                         TensorTypeA* a,
                         TensorTypeB* b,
                         const char* op_string) {
  TensorArgType aType = ReadWrite;
  TensorArgType bType = ReadWrite;
  auto stream = THCState_getCurrentStream(state);
  long totalElements = TensorUtils<TensorTypeA>::getNumElements(state, a);

  if (totalElements != TensorUtils<TensorTypeB>::getNumElements(state, b)) {
    return false;
  }

  if (TensorUtils<TensorTypeA>::getDims(state, a) > MAX_CUTORCH_DIMS ||
      TensorUtils<TensorTypeB>::getDims(state, b) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (TensorUtils<TensorTypeA>::getDims(state, a) == 0) {
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
  TensorTypeA* oldA = NULL;
  TensorTypeB* oldB = NULL;

  if (aType == ReadWrite &&
      TensorUtils<TensorTypeA>::overlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = TensorUtils<TensorTypeA>::newContiguous(state, a);
  }
  if (bType == ReadWrite &&
      TensorUtils<TensorTypeB>::overlappingIndices(state, b)) {
    // Must perform in contiguous space
    oldB = b;
    b = TensorUtils<TensorTypeB>::newContiguous(state, b);
  }

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, A, B)                                         \
  kernelPointwiseApply2RTC<typename TensorUtils<TensorTypeA>::DataType, \
                           typename TensorUtils<TensorTypeB>::DataType, \
                           TYPE>                                        \
      (aInfo, bInfo, op_string, (TYPE) totalElements, grid, block, A, B, stream);

#define HANDLE_B_CASE(TYPE, A, B)               \
  {                                             \
    if (bInfo.isContiguous()) {                 \
      HANDLE_CASE(TYPE, A, -2);                 \
    } else {                                    \
      switch (B) {                              \
        case 1:                                 \
        HANDLE_CASE(TYPE, A, 1);                \
        break;                                  \
        case 2:                                 \
        HANDLE_CASE(TYPE, A, 2);                \
        break;                                  \
        default:                                \
        HANDLE_CASE(TYPE, A, -1);               \
        break;                                  \
      }                                         \
    }                                           \
  }

#define HANDLE_A_CASE(TYPE, A, B)               \
  {                                             \
    if (aInfo.isContiguous()) {                 \
      HANDLE_B_CASE(TYPE, -2, B);               \
    } else {                                    \
      switch (A) {                              \
        case 1:                                 \
        HANDLE_B_CASE(TYPE, 1, B);              \
        break;                                  \
        case 2:                                 \
        HANDLE_B_CASE(TYPE, 2, B);              \
        break;                                  \
        default:                                \
        HANDLE_B_CASE(TYPE, -1, B);             \
        break;                                  \
      }                                         \
    }                                           \
  }

  if (TensorUtils<TensorTypeA>::canUse32BitIndexMath(state, a) &&
      TensorUtils<TensorTypeB>::canUse32BitIndexMath(state, b)) {
    TensorInfo<typename TensorUtils<TensorTypeA>::DataType, unsigned int> aInfo =
      getTensorInfo<TensorTypeA, unsigned int>(state, a);
    aInfo.collapseDims();

    TensorInfo<typename TensorUtils<TensorTypeB>::DataType, unsigned int> bInfo =
      getTensorInfo<TensorTypeB, unsigned int>(state, b);
    bInfo.collapseDims();

    HANDLE_A_CASE(unsigned int, aInfo.dims, bInfo.dims);
  } else {
    TensorInfo<typename TensorUtils<TensorTypeA>::DataType, unsigned long> aInfo =
      getTensorInfo<TensorTypeA, unsigned long>(state, a);
    aInfo.collapseDims();

    TensorInfo<typename TensorUtils<TensorTypeB>::DataType, unsigned long> bInfo =
      getTensorInfo<TensorTypeB, unsigned long>(state, b);
    bInfo.collapseDims();

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (aInfo.isContiguous() && bInfo.isContiguous()) {
      kernelPointwiseApply2RTC<typename TensorUtils<TensorTypeA>::DataType,
                               typename TensorUtils<TensorTypeB>::DataType,
                               unsigned long>
        (aInfo, bInfo, op_string, totalElements, grid, block, -2, -2, stream);
    } else {
      kernelPointwiseApply2RTC<typename TensorUtils<TensorTypeA>::DataType,
                               typename TensorUtils<TensorTypeB>::DataType,
                               unsigned long>
        (aInfo, bInfo, op_string, (unsigned long) totalElements, grid, block, -1, -1, stream);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_B_CASE
#undef HANDLE_A_CASE

  if (oldA) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    TensorUtils<TensorTypeA>::copyIgnoringOverlaps(state, oldA, a);
    TensorUtils<TensorTypeA>::free(state, a);
    a = oldA;
  }

  if (oldB) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldB contiguous.
    TensorUtils<TensorTypeB>::copyIgnoringOverlaps(state, oldB, b);
    TensorUtils<TensorTypeB>::free(state, b);
    b = oldB;
  }

  return true;
}

#define THC_POINTWISE_APPLY2(TYPE) \
  extern "C"  \
  bool TH_CONCAT_2(TYPE,_pointwiseApply2)(THCState* state, \
                                    TYPE* a, \
                                    TYPE* b, \
                                    const char* op_string) { \
    return THC_pointwiseApply2<TYPE>(state, a, b, op_string); \
  } \

THC_POINTWISE_APPLY2(THCudaTensor)
THC_POINTWISE_APPLY2(THCudaDoubleTensor)
THC_POINTWISE_APPLY2(THCudaHalfTensor)
  

template <typename TensorTypeA,
          typename TensorTypeB,
          typename TensorTypeC>
bool THC_pointwiseApply3(THCState* state,
                         TensorTypeA* a,
                         TensorTypeB* b,
                         TensorTypeC* c,
                         const char* op_string) {
  TensorArgType aType = ReadWrite;
  TensorArgType bType = ReadWrite;
  TensorArgType cType = ReadWrite;
  auto stream = THCState_getCurrentStream(state);
  long totalElements = TensorUtils<TensorTypeA>::getNumElements(state, a);

  if (totalElements != TensorUtils<TensorTypeB>::getNumElements(state, b) ||
      totalElements != TensorUtils<TensorTypeC>::getNumElements(state, c)) {
    return false;
  }

  if (TensorUtils<TensorTypeA>::getDims(state, a) > MAX_CUTORCH_DIMS ||
      TensorUtils<TensorTypeB>::getDims(state, b) > MAX_CUTORCH_DIMS ||
      TensorUtils<TensorTypeC>::getDims(state, c) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (TensorUtils<TensorTypeA>::getDims(state, a) == 0) {
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
  TensorTypeA* oldA = NULL;
  TensorTypeB* oldB = NULL;
  TensorTypeC* oldC = NULL;

  if (aType == ReadWrite &&
      TensorUtils<TensorTypeA>::overlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = TensorUtils<TensorTypeA>::newContiguous(state, a);
  }
  if (bType == ReadWrite &&
      TensorUtils<TensorTypeB>::overlappingIndices(state, b)) {
    // Must perform in contiguous space
    oldB = b;
    b = TensorUtils<TensorTypeB>::newContiguous(state, b);
  }
  if (cType == ReadWrite &&
      TensorUtils<TensorTypeC>::overlappingIndices(state, c)) {
    // Must perform in contiguous space
    oldC = c;
    c = TensorUtils<TensorTypeC>::newContiguous(state, c);
  }

#define HANDLE_CASE(TYPE, A, B, C)                                      \
  kernelPointwiseApply3RTC<typename TensorUtils<TensorTypeA>::DataType, \
                           typename TensorUtils<TensorTypeB>::DataType, \
                           typename TensorUtils<TensorTypeC>::DataType, \
                           TYPE>                                        \
      (aInfo, bInfo, cInfo, op_string, (TYPE) totalElements, grid, block, A, B, C, stream);

#define HANDLE_C_CASE(TYPE, A, B, C)            \
  {                                             \
    if (cInfo.isContiguous()) {                 \
      HANDLE_CASE(TYPE, A, B, -2);              \
    } else {                                    \
      switch (C) {                              \
        case 1:                                 \
        HANDLE_CASE(TYPE, A, B, 1);             \
        break;                                  \
        case 2:                                 \
        HANDLE_CASE(TYPE, A, B, 2);             \
        break;                                  \
        default:                                \
        HANDLE_CASE(TYPE, A, B, -1);            \
        break;                                  \
      }                                         \
    }                                           \
  }

#define HANDLE_B_CASE(TYPE, A, B, C)            \
  {                                             \
    if (bInfo.isContiguous()) {                 \
      HANDLE_C_CASE(TYPE, A, -2, C);            \
    } else {                                    \
      switch (B) {                              \
        case 1:                                 \
        HANDLE_C_CASE(TYPE, A, 1, C);           \
        break;                                  \
        case 2:                                 \
        HANDLE_C_CASE(TYPE, A, 2, C);           \
        break;                                  \
        default:                                \
        HANDLE_C_CASE(TYPE, A, -1, C);          \
        break;                                  \
      }                                         \
    }                                           \
  }

#define HANDLE_A_CASE(TYPE, A, B, C)            \
  {                                             \
    if (aInfo.isContiguous()) {                 \
      HANDLE_B_CASE(TYPE, -2, B, C);            \
    } else {                                    \
      switch (A) {                              \
        case 1:                                 \
        HANDLE_B_CASE(TYPE, 1, B, C);           \
        break;                                  \
        case 2:                                 \
        HANDLE_B_CASE(TYPE, 2, B, C);           \
        break;                                  \
        default:                                \
        HANDLE_B_CASE(TYPE, -1, B, C);          \
        break;                                  \
      }                                         \
    }                                           \
  }

  if (TensorUtils<TensorTypeA>::canUse32BitIndexMath(state, a) &&
      TensorUtils<TensorTypeB>::canUse32BitIndexMath(state, b) &&
      TensorUtils<TensorTypeC>::canUse32BitIndexMath(state, c)) {
    TensorInfo<typename TensorUtils<TensorTypeA>::DataType, unsigned int> aInfo =
      getTensorInfo<TensorTypeA, unsigned int>(state, a);
    aInfo.collapseDims();

    TensorInfo<typename TensorUtils<TensorTypeB>::DataType, unsigned int> bInfo =
      getTensorInfo<TensorTypeB, unsigned int>(state, b);
    bInfo.collapseDims();

    TensorInfo<typename TensorUtils<TensorTypeC>::DataType, unsigned int> cInfo =
      getTensorInfo<TensorTypeC, unsigned int>(state, c);
    cInfo.collapseDims();

    HANDLE_A_CASE(unsigned int, aInfo.dims, bInfo.dims, cInfo.dims);
  } else {
    TensorInfo<typename TensorUtils<TensorTypeA>::DataType, unsigned long> aInfo =
      getTensorInfo<TensorTypeA, unsigned long>(state, a);
    aInfo.collapseDims();

    TensorInfo<typename TensorUtils<TensorTypeB>::DataType, unsigned long> bInfo =
      getTensorInfo<TensorTypeB, unsigned long>(state, b);
    bInfo.collapseDims();

    TensorInfo<typename TensorUtils<TensorTypeC>::DataType, unsigned long> cInfo =
      getTensorInfo<TensorTypeC, unsigned long>(state, c);
    cInfo.collapseDims();

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (aInfo.isContiguous() && bInfo.isContiguous() && cInfo.isContiguous()) {
      kernelPointwiseApply3RTC<typename TensorUtils<TensorTypeA>::DataType,
                               typename TensorUtils<TensorTypeB>::DataType,
                               typename TensorUtils<TensorTypeC>::DataType,
                               unsigned long>
        (aInfo, bInfo, cInfo, op_string, totalElements, grid, block, -2, -2, -2, stream);
    } else {
      kernelPointwiseApply3RTC<typename TensorUtils<TensorTypeA>::DataType,
                               typename TensorUtils<TensorTypeB>::DataType,
                               typename TensorUtils<TensorTypeC>::DataType,
                               unsigned long>
        (aInfo, bInfo, cInfo, op_string, (unsigned long) totalElements, grid, block, -1, -1, -1, stream);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_C_CASE
#undef HANDLE_B_CASE
#undef HANDLE_A_CASE

  if (oldA) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    TensorUtils<TensorTypeA>::copyIgnoringOverlaps(state, oldA, a);
    TensorUtils<TensorTypeA>::free(state, a);
    a = oldA;
  }

  if (oldB) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldB contiguous.
    TensorUtils<TensorTypeB>::copyIgnoringOverlaps(state, oldB, b);
    TensorUtils<TensorTypeB>::free(state, b);
    b = oldB;
  }

  if (oldC) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldC contiguous.
    TensorUtils<TensorTypeC>::copyIgnoringOverlaps(state, oldC, c);
    TensorUtils<TensorTypeC>::free(state, c);
    c = oldC;
  }

  return true;
}


#define THC_POINTWISE_APPLY3(TYPE) \
  extern "C" \
  bool TH_CONCAT_2(TYPE, _pointwiseApply3)(THCState* state, \
                                    TYPE* a, \
                                    TYPE* b, \
                                    TYPE* c, \
                                    const char* op_string) { \
    return THC_pointwiseApply3<TYPE>(state, a, b, c, op_string); \
  }

THC_POINTWISE_APPLY3(THCudaTensor)
THC_POINTWISE_APPLY3(THCudaDoubleTensor)
THC_POINTWISE_APPLY3(THCudaHalfTensor)
