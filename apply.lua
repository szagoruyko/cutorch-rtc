include 'init.lua'
require 'nvrtc'

local ffi = require 'ffi'
local kernel_source = [[
extern "C" __global__
void kernel(float* a, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float &x = a[i];
  if (i < n)
    LAMBDA;
}
]]

local CUDA_NUM_THREADS = 1024

local ptx_cache = {}

local function get_blocks(N)
  return math.floor((N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
end


function torch.CudaTensor:apply(lambda)
  local kernel = kernel_source:gsub('LAMBDA', lambda)
  local ptx = ptx_cache[lambda] or nvrtc.compileReturnPTX(kernel)

  cutorch.launchPTX(ptx, {self, {'int', self:numel()}}, {CUDA_NUM_THREADS}, {get_blocks(self:numel())})
end
