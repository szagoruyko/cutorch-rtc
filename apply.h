#ifndef THCAPPLYRTC_CUH
#define THCAPPLYRTC_CUH

// c++11 raw string literal
const char THCApplyRTC_cuh[] = R"(
#define MAX_CUTORCH_DIMS 25

template <typename T, typename IndexType>
struct TensorInfo {
  __device__ inline bool isContiguous() const {
    return (dims == 1 && strides[0] == 1);
  }

  T* data;
  IndexType sizes[MAX_CUTORCH_DIMS];
  IndexType strides[MAX_CUTORCH_DIMS];
  int dims;
};

// Translate a linear index for the apply to a T* offset;
// specialized on `Dims` to reduce nvcc compilation time
template <typename T, typename IndexType, int Dims>
struct IndexToOffset {
  static __host__ __device__ IndexType get(
      IndexType linearId,
      const TensorInfo<T, IndexType>& info) {
    IndexType offset = 0;

    // Use static dims
    for (int i = Dims - 1; i >= 0; --i) {
      IndexType curDimIndex = linearId % info.sizes[i];
      IndexType curDimOffset = curDimIndex * info.strides[i];
      offset += curDimOffset;

      if (i > 0) {
        linearId /= info.sizes[i];
      }
    }

    return offset;
  }
};

template <typename T, typename IndexType>
struct IndexToOffset<T, IndexType, -2> {
  static inline __host__ __device__ IndexType
    get(IndexType linearId, const TensorInfo<T, IndexType>& info) {
      return linearId;
    }
};

template <typename T, typename IndexType>
struct IndexToOffset<T, IndexType, -1> {
  static inline __host__ __device__ IndexType get(
      IndexType linearId,
      const TensorInfo<T, IndexType>& info) {

    IndexType offset = 0;

    // Use dynamic dims
    for (int i = info.dims - 1; i >= 0; --i) {
      IndexType curDimIndex = linearId % info.sizes[i];
      IndexType curDimOffset = curDimIndex * info.strides[i];
      offset += curDimOffset;

      linearId /= info.sizes[i];
    }

    return offset;
  }
};

// THCApply.cuh
//
// This file contains pointwise operation functions and kernels that
// work on both contiguous and non-contiguous tensor arguments of
// arbitrary (up to MAX_CUTORCH_DIMS) dimensioned arguments without
// copying or temporary storage.
//

// Threads per block for our apply kernel
// FIXME: use occupancy calculator instead
#define THC_APPLY_THREADS_PER_BLOCK 32 * 16

template <typename Op,
         typename Ta,
         typename IndexType,
         int ADims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
  __global__ void
  kernelPointwiseApply1(TensorInfo<Ta, IndexType> a,
      IndexType totalElements,
      Op op) {
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
        linearIndex < totalElements;
        linearIndex += gridDim.x * blockDim.x) {
      // Convert `linearIndex` into an offset of `a`
      const IndexType aOffset =
        IndexToOffset<Ta, IndexType, ADims>::get(linearIndex, a);

      op(&a.data[aOffset]);
    }
  }

template <typename Op,
         typename Ta, typename Tb,
         typename IndexType,
         int ADims, int BDims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
  __global__ void
  kernelPointwiseApply2(TensorInfo<Ta, IndexType> a,
      TensorInfo<Tb, IndexType> b,
      IndexType totalElements,
      Op op) {
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
        linearIndex < totalElements;
        linearIndex += gridDim.x * blockDim.x) {
      // Convert `linearIndex` into an offset of `a`
      const IndexType aOffset =
        IndexToOffset<Ta, IndexType, ADims>::get(linearIndex, a);

      // Convert `linearIndex` into an offset of `b`
      const IndexType bOffset =
        IndexToOffset<Tb, IndexType, BDims>::get(linearIndex, b);

      op(&a.data[aOffset], &b.data[bOffset]);
    }
  }

template <typename Op,
         typename Ta, typename Tb, typename Tc,
         typename IndexType,
         int ADims, int BDims, int CDims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
  __global__ void
  kernelPointwiseApply3(TensorInfo<Ta, IndexType> a,
      TensorInfo<Tb, IndexType> b,
      TensorInfo<Tc, IndexType> c,
      IndexType totalElements,
      Op op) {
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
        linearIndex < totalElements;
        linearIndex += gridDim.x * blockDim.x) {
      // Convert `linearIndex` into an offset of `a`
      const IndexType aOffset =
        IndexToOffset<Ta, IndexType, ADims>::get(linearIndex, a);

      // Convert `linearIndex` into an offset of `b`
      const IndexType bOffset =
        IndexToOffset<Tb, IndexType, BDims>::get(linearIndex, b);

      // Convert `linearIndex` into an offset of `c`
      const IndexType cOffset =
        IndexToOffset<Tc, IndexType, CDims>::get(linearIndex, c);

      op(&a.data[aOffset], &b.data[bOffset], &c.data[cOffset]);
    }
  }
)";

#endif // THCAPPLYRTC_CUH
