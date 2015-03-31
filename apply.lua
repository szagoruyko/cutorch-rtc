local kernel_source = [[
extern "C" __global__
void kernel(float* a, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float &x = a[i];
  if (i < n)
    {
]]

local CUDA_NUM_THREADS = 256

local function get_blocks(N)
  return math.floor((N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
end

local ptx_cache = {}


function torch.CudaTensor:apply(lambda)
  assert(self:contiguous(), 'current version of apply only works on contiguous tensors!')
  local kernel = kernel_source..lambda..';}}'
  local ptx
  if not ptx_cache[lambda] then
    ptx = nvrtc.compileReturnPTX(kernel)
    ptx_cache[lambda] = ptx
  else
    ptx = ptx_cache[lambda]
  end

  cutorch.launchPTX(ptx, 'kernel', {self, {'int', self:numel()}}, {CUDA_NUM_THREADS}, {get_blocks(self:numel())})
  return self
end
