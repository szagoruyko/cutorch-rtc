#include <vector>
#include <memory>
#include <unordered_map>
#include <iostream>
#include <type_traits>
#include <cuda_fp16.h>

#include "THC/THC.h"
#include "THC/THCApply.cuh"

#include "compile_ptx.h"

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
    return op == other.op &&
           ta_type == other.ta_type &&
           index_type == other.index_type &&
           dim == other.dim;
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
#include <header.h>                                                     \n\
typedef %s Ta;                                                          \n\
typedef %s IndexType; 							\n\
struct Op {                                                             \n\
  __device__ __forceinline__ void operator()(Ta* v) {                \n\
    Ta& x = *v;                                                      \n\
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
    const char* apply_header,
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
  sprintf(src, instanciate_apply1, ta_type, index_type, op, A);
  const char *headers[] = {apply_header};
  const char *includeNames[] = {"header.h"};

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

/*
// Example op: 'x = x*y'
const char* instanciate_apply2 = "                                      \n\
#define TYPE %s								\n\
#include <header.h>                                                     \n\
struct Op {                                                             \n\
  __device__ __forceinline__						\n\
  void operator()(float* a, float* b) {      				\n\
    float& x = *a;                                                      \n\
    float& y = *b;							\n\
    %s;                                                                 \n\
  }                                                                     \n\
};                                                                      \n\
extern \"C\" __global__                                                 \n\
void kernel(TensorInfo<TYPE> a, TensorInfo<TYPE> b, TYPE totalElements) \n\
{                                                                       \n\
  Op op;                                                                \n\
  THCudaTensor_pointwiseApply2<Op,TYPE,%d,%d>				\n\
  			(a, b, totalElements, op);			\n\
}                                                                       \n\
";

template <typename IndexType>
void THCudaTensor_pointwiseApply2RTC(
    TensorInfo<IndexType> aInfo,
    TensorInfo<IndexType> bInfo,
    const char* apply_header,
    const char* op,
    IndexType totalElements,
    dim3 grid, dim3 block,
    int A, int B,
    cudaStream_t stream)
{
  // using c++11 std::is_same here
  const char* type;
  if (std::is_same<IndexType, unsigned long>::value)
    type = "unsigned long";
  else if(std::is_same<IndexType, unsigned int>::value)
    type = "unsigned int";

  char src[2048];
  sprintf(src, instanciate_apply2, type, op, A, B);
  const char *headers[] = {apply_header};
  const char *includeNames[] = {"header.h"};

  PTXPtr ptx;
  Apply2Hash hash(op, type, A, B);
  auto found_hash = apply2cache.find(hash);
  if(found_hash == apply2cache.end())
  {
    ptx = PTXPtr(new PTX());
    compilePTX(src, headers, includeNames, *ptx);
    apply2cache.emplace(hash, ptx);
  }
  else
    ptx = found_hash->second;

  void *args[] = {(void*)&aInfo, (void*)&bInfo, (void*)&totalElements};
  launch(ptx->data(), "kernel", args, grid, block, (CUstream)stream);
}


// Example op: 'x = y*z'
const char* instanciate_apply3 = "                                      \n\
#define TYPE %s								\n\
#include <header.h>                                                     \n\
struct Op {                                                             \n\
  __device__ __forceinline__						\n\
  void operator()(float* a, float* b, float *c) {			\n\
    float& x = *a;                                                      \n\
    float& y = *b;							\n\
    float& z = *c;							\n\
    %s;                                                                 \n\
  }                                                                     \n\
};                                                                      \n\
extern \"C\" __global__                                                 \n\
void kernel(TensorInfo<TYPE> a,						\n\
    	    TensorInfo<TYPE> b,						\n\
    	    TensorInfo<TYPE> c,						\n\
	    TYPE totalElements)       					\n\
{                                                                       \n\
  Op op;                                                                \n\
  THCudaTensor_pointwiseApply3<Op,TYPE,%d,%d,%d>			\n\
  			(a, b, c, totalElements, op);			\n\
}                                                                       \n\
";

template <typename IndexType>
void THCudaTensor_pointwiseApply3RTC(
    TensorInfo<IndexType> aInfo,
    TensorInfo<IndexType> bInfo,
    TensorInfo<IndexType> cInfo,
    const char* apply_header,
    const char* op,
    IndexType totalElements,
    dim3 grid, dim3 block,
    int A, int B, int C,
    cudaStream_t stream)
{
  // using c++11 std::is_same here
  const char* type;
  if (std::is_same<IndexType, unsigned long>::value)
    type = "unsigned long";
  else if(std::is_same<IndexType, unsigned int>::value)
    type = "unsigned int";

  char src[2048];
  sprintf(src, instanciate_apply3, type, op, A, B, C);
  const char *headers[] = {apply_header};
  const char *includeNames[] = {"header.h"};

  PTXPtr ptx;
  Apply3Hash hash(op, type, A, B, C);
  auto found_hash = apply3cache.find(hash);
  if(found_hash == apply3cache.end())
  {
    ptx = PTXPtr(new PTX());
    compilePTX(src, headers, includeNames, *ptx);
    apply3cache.emplace(hash, ptx);
  }
  else
    ptx = found_hash->second;

  void *args[] = {(void*)&aInfo, (void*)&bInfo, (void*)&cInfo, (void*)&totalElements};
  launch(ptx->data(), "kernel", args, grid, block, (CUstream)stream);
}

*/

template <typename TensorTypeA>
bool THC_pointwiseApply1(THCState* state,
                         TensorTypeA* a,
                         const char* apply_header,
                         const char* op_string)
{
  TensorArgType aType = ReadWrite;
  cudaStream_t stream = state->currentStream;
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
      (aInfo, apply_header, op_string, (TYPE) totalElements, grid, block, A, stream);

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
        (aInfo, apply_header, op_string, totalElements, grid, block, -2, stream);
    } else {
      kernelPointwiseApply1RTC<typename TensorUtils<TensorTypeA>::DataType, unsigned long>
        (aInfo, apply_header, op_string, totalElements, grid, block, -1, stream);
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

extern "C" 
bool THCudaTensor_pointwiseApply1(THCState* state, THCudaTensor* a, const char* apply_header, const char* op_string)
{
  return THC_pointwiseApply1<THCudaTensor>(state, a, apply_header, op_string);
}

extern "C" 
bool THCudaHalfTensor_pointwiseApply1(THCState* state, THCudaHalfTensor* a, const char* apply_header, const char* op_string)
{
  return THC_pointwiseApply1<THCudaHalfTensor>(state, a, apply_header, op_string);
}

extern "C" 
bool THCudaDoubleTensor_pointwiseApply1(THCState* state, THCudaDoubleTensor* a, const char* apply_header, const char* op_string)
{
  return THC_pointwiseApply1<THCudaDoubleTensor>(state, a, apply_header, op_string);
}

/*
extern "C" 
bool THCudaTensor_pointwiseApply1(THCState* state,
                                  THCudaTensor* a,
                                  const char* apply_header,
                                  const char* op_string)
{
  TensorArgType aType = ReadWrite;
  cudaStream_t stream = state->currentStream;
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
  THCudaTensor_pointwiseApply1RTC(aInfo, apply_header, op_string, (TYPE)totalElements, grid, block, A, stream);

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
      THCudaTensor_pointwiseApply1RTC(aInfo, apply_header, op_string, (unsigned long)totalElements, grid, block, -2, stream);
    } else {
      THCudaTensor_pointwiseApply1RTC(aInfo, apply_header, op_string, (unsigned long)totalElements, grid, block, -1, stream);
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
*/

/*
extern "C"
bool THCudaTensor_pointwiseApply2(THCState* state,
                                  THCudaTensor* a,
                                  THCudaTensor* b,
                                  const char* apply_header,
                                  const char* op_string)
{
  TensorArgType aType = ReadWrite;
  TensorArgType bType = ReadWrite;
  cudaStream_t stream = state->currentStream;

  long totalElements = THCudaTensor_nElement(state, a);

  if (totalElements != THCudaTensor_nElement(state, b)) {
    return false;
  }

  if (THCudaTensor_nDimension(state, a) > MAX_CUTORCH_DIMS ||
      THCudaTensor_nDimension(state, b) > MAX_CUTORCH_DIMS) {
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
  THCudaTensor* oldB = NULL;

  if (aType == ReadWrite && THC_overlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = THCudaTensor_newContiguous(state, a);
  }
  if (bType == ReadWrite && THC_overlappingIndices(state, b)) {
    // Must perform in contiguous space
    oldB = b;
    b = THCudaTensor_newContiguous(state, b);
  }

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, A, B)                                \
  THCudaTensor_pointwiseApply2RTC(aInfo, bInfo, apply_header, op_string, (TYPE)totalElements, grid, block, A, B, stream);

#define HANDLE_B_CASE(TYPE, A, B)                   \
  {                                                 \
    if (bInfo.isContiguous()) {                     \
      HANDLE_CASE(TYPE, A, -2);                     \
    } else {                                        \
      switch (B) {                                  \
        case 1:                                     \
          HANDLE_CASE(TYPE, A, 1);                  \
          break;                                    \
        case 2:                                     \
          HANDLE_CASE(TYPE, A, 2);                  \
          break;                                    \
        case 3:                                     \
          HANDLE_CASE(TYPE, A, 3);                  \
          break;                                    \
        default:                                    \
          HANDLE_CASE(TYPE, A, -1);                 \
          break;                                    \
      }                                             \
    }                                               \
  }

#define HANDLE_A_CASE(TYPE, A, B)                   \
  {                                                 \
    if (aInfo.isContiguous()) {                     \
      HANDLE_B_CASE(TYPE, -2, B);                   \
    } else {                                        \
      switch (A) {                                  \
        case 1:                                     \
          HANDLE_B_CASE(TYPE, 1, B);                \
          break;                                    \
        case 2:                                     \
          HANDLE_B_CASE(TYPE, 2, B);                \
          break;                                    \
        case 3:                                     \
          HANDLE_B_CASE(TYPE, 3, B);                \
          break;                                    \
        default:                                    \
          HANDLE_B_CASE(TYPE, -1, B);               \
          break;                                    \
      }                                             \
    }                                               \
  }

  if (THC_canUse32BitIndexMath(state, a) &&
      THC_canUse32BitIndexMath(state, b)) {
    TensorInfo<unsigned int> aInfo(state, a);
    TensorInfo<unsigned int> bInfo(state, b);

    HANDLE_A_CASE(unsigned int, aInfo.dims, bInfo.dims);
  } else {
    TensorInfo<unsigned long> aInfo(state, a);
    TensorInfo<unsigned long> bInfo(state, b);

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (aInfo.isContiguous() && bInfo.isContiguous()) {
      THCudaTensor_pointwiseApply2RTC(aInfo, bInfo, apply_header, op_string,
	  			(unsigned long)totalElements, grid, block, -2, -2, stream);
    } else {
      THCudaTensor_pointwiseApply2RTC(aInfo, bInfo, apply_header, op_string,
	  			(unsigned long)totalElements, grid, block, -1, -1, stream);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_B_CASE
#undef HANDLE_A_CASE

  if (oldA) {
    // Ignore overlaps when copying back; if we use THCudaTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    THCudaTensor_copyIgnoringOverlaps(state, oldA, a);
    THCudaTensor_free(state, a);
    a = oldA;
  }

  if (oldB) {
    // Ignore overlaps when copying back; if we use THCudaTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldB contiguous.
    THCudaTensor_copyIgnoringOverlaps(state, oldB, b);
    THCudaTensor_free(state, b);
    b = oldB;
  }

  return true;
}
extern "C"
bool THCudaTensor_pointwiseApply3(THCState* state,
                                  THCudaTensor* a,
                                  THCudaTensor* b,
                                  THCudaTensor* c,
                                  const char* apply_header,
                                  const char* op_string)
{
  TensorArgType aType = ReadWrite;
  TensorArgType bType = ReadWrite;
  TensorArgType cType = ReadWrite;
  cudaStream_t stream = state->currentStream;

  long totalElements = THCudaTensor_nElement(state, a);

  if (totalElements != THCudaTensor_nElement(state, b) ||
      totalElements != THCudaTensor_nElement(state, c)) {
    return false;
  }

  if (THCudaTensor_nDimension(state, a) > MAX_CUTORCH_DIMS ||
      THCudaTensor_nDimension(state, b) > MAX_CUTORCH_DIMS ||
      THCudaTensor_nDimension(state, c) > MAX_CUTORCH_DIMS) {
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
  THCudaTensor* oldB = NULL;
  THCudaTensor* oldC = NULL;

  if (aType == ReadWrite && THC_overlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = THCudaTensor_newContiguous(state, a);
  }

  if (bType == ReadWrite && THC_overlappingIndices(state, b)) {
    // Must perform in contiguous space
    oldB = b;
    b = THCudaTensor_newContiguous(state, b);
  }

  if (cType == ReadWrite && THC_overlappingIndices(state, c)) {
    // Must perform in contiguous space
    oldC = c;
    c = THCudaTensor_newContiguous(state, c);
  }

#define HANDLE_CASE(TYPE, A, B, C)                                      \
  THCudaTensor_pointwiseApply3RTC(aInfo, bInfo, cInfo,			\
      apply_header, op_string, (TYPE)totalElements, grid, block,	\
      A, B, C, stream);								\

#define HANDLE_C_CASE(TYPE, A, B, C)             \
  {                                              \
    if (cInfo.isContiguous()) {                  \
      HANDLE_CASE(TYPE, A, B, -2);               \
    } else {                                     \
      switch (C) {                               \
        case 1:                                  \
          HANDLE_CASE(TYPE, A, B, 1);            \
          break;                                 \
        case 2:                                  \
          HANDLE_CASE(TYPE, A, B, 2);            \
          break;                                 \
        case 3:                                  \
          HANDLE_CASE(TYPE, A, B, 3);            \
          break;                                 \
        default:                                 \
          HANDLE_CASE(TYPE, A, B, -1);           \
          break;                                 \
      }                                          \
    }                                            \
  }

#define HANDLE_B_CASE(TYPE, A, B, C)                 \
  {                                                  \
    if (bInfo.isContiguous()) {                      \
      HANDLE_C_CASE(TYPE, A, -2, C);                 \
    } else {                                         \
      switch (B) {                                   \
        case 1:                                      \
          HANDLE_C_CASE(TYPE, A, 1, C);              \
          break;                                     \
        case 2:                                      \
          HANDLE_C_CASE(TYPE, A, 2, C);              \
          break;                                     \
        case 3:                                      \
          HANDLE_C_CASE(TYPE, A, 3, C);              \
          break;                                     \
        default:                                     \
          HANDLE_C_CASE(TYPE, A, -1, C);             \
          break;                                     \
      }                                              \
    }                                                \
  }

#define HANDLE_A_CASE(TYPE, A, B, C)                 \
  {                                                  \
    if (aInfo.isContiguous()) {                      \
      HANDLE_B_CASE(TYPE, -2, B, C);                 \
    } else {                                         \
      switch (A) {                                   \
        case 1:                                      \
          HANDLE_B_CASE(TYPE, 1, B, C);              \
          break;                                     \
        case 2:                                      \
          HANDLE_B_CASE(TYPE, 2, B, C);              \
          break;                                     \
        case 3:                                      \
          HANDLE_B_CASE(TYPE, 3, B, C);              \
          break;                                     \
        default:                                     \
          HANDLE_B_CASE(TYPE, -1, B, C);             \
          break;                                     \
      }                                              \
    }                                                \
  }

  if (THC_canUse32BitIndexMath(state, a) &&
      THC_canUse32BitIndexMath(state, b) &&
      THC_canUse32BitIndexMath(state, c)) {
    TensorInfo<unsigned int> aInfo(state, a);
    TensorInfo<unsigned int> bInfo(state, b);
    TensorInfo<unsigned int> cInfo(state, c);

    HANDLE_A_CASE(unsigned int, aInfo.dims, bInfo.dims, cInfo.dims);
  } else {
    TensorInfo<unsigned long> aInfo(state, a);
    TensorInfo<unsigned long> bInfo(state, b);
    TensorInfo<unsigned long> cInfo(state, c);

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (aInfo.isContiguous() && bInfo.isContiguous() && cInfo.isContiguous()) {
      THCudaTensor_pointwiseApply3RTC(aInfo, bInfo, cInfo,
	  		apply_header, op_string,
	  		(unsigned long)totalElements, grid, block, -2, -2, -2, stream);
    } else {
      THCudaTensor_pointwiseApply3RTC(aInfo, bInfo, cInfo,
	  		apply_header, op_string,
	  		(unsigned long)totalElements, grid, block, -1, -1, -1, stream);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_C_CASE
#undef HANDLE_B_CASE
#undef HANDLE_A_CASE

  if (oldA) {
    // Ignore overlaps when copying back; if we use THCudaTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    THCudaTensor_copyIgnoringOverlaps(state, oldA, a);
    THCudaTensor_free(state, a);
    a = oldA;
  }

  if (oldB) {
    // Ignore overlaps when copying back; if we use THCudaTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldB contiguous.
    THCudaTensor_copyIgnoringOverlaps(state, oldB, b);
    THCudaTensor_free(state, b);
    b = oldB;
  }

  if (oldC) {
    // Ignore overlaps when copying back; if we use THCudaTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldC contiguous.
    THCudaTensor_copyIgnoringOverlaps(state, oldC, c);
    THCudaTensor_free(state, c);
    c = oldC;
  }

  return true;
}
*/
