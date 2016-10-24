#include <nvrtc.h>
#include <vector>

#include "compile_ptx.h"

inline void NVRTC_CHECK(nvrtcResult result)
{
  if(result != NVRTC_SUCCESS)
    THError(nvrtcGetErrorString(result));
}

void compilePTX(const char* src,
    		const char* headers[],
		const char* includeNames[],
		std::vector<char>& ptx)
{
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
  ptx.resize(ptx_size);
  NVRTC_CHECK(nvrtcGetPTX(program, ptx.data()));
  NVRTC_CHECK(nvrtcDestroyProgram(&program));
}

inline void CUDA_CHECK(CUresult result)
{
  if(result != CUDA_SUCCESS)
  {
    const char* errstr;
    cuGetErrorString(result, &errstr);
    THError(errstr);
  }
}

void launch(const char* ptx, const char* name, void* args[], dim3 grid, dim3 block, CUstream stream)
{
  CUmodule module;
  CUfunction func;

  CUDA_CHECK(cuModuleLoadData(&module, ptx));
  CUDA_CHECK(cuModuleGetFunction(&func, module, name));

  CUDA_CHECK(cuLaunchKernel(func,
                            grid.x, grid.y, grid.z,
                            block.x, block.y, block.z,
                            0, stream, args, NULL));

  CUDA_CHECK(cuModuleUnload(module));
}

extern "C"
void launchPTX(THCState* state, const char* ptx, const char* name, void* args[], int* grid, int* block)
{
  cudaStream_t stream = THCState_getCurrentStream(state);
  launch(ptx, name, args, dim3(grid[0], grid[1], grid[2]), dim3(block[0], block[1], block[2]), (CUstream)stream);
}

