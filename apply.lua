include 'ffi.lua'

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
  local ptx
  if not ptx_cache[lambda] then
    local kernel = kernel_source:gsub('LAMBDA', lambda)

    local program = ffi.new'nvrtcProgram[1]'
    nvrtc.errcheck('nvrtcCreateProgram', program, kernel, nil, 0, nil, nil)
    local err = nvrtc.C.nvrtcCompileProgram(program[0], 0, nil)
    if tonumber(err) == 6 then
      local log_size = ffi.new'size_t[1]'
      nvrtc.errcheck('nvrtcGetProgramLogSize', program[0], log_size)
      local log = ffi.new('char[?]', tonumber(log_size[0]))
      nvrtc.errcheck('nvrtcGetProgramLog', program[0], log)
      error(ffi.string(log))
    end

    local ptx_size = ffi.new'size_t[1]'
    nvrtc.errcheck('nvrtcGetPTXSize', program[0], ptx_size)
    ptx = ffi.new('char[?]', tonumber(ptx_size[0]))
    nvrtc.errcheck('nvrtcGetPTX', program[0], ptx)
    ptx_cache[lambda] = ptx
  
    nvrtc.errcheck('nvrtcDestroyProgram', program)
  else
    ptx = ptx_cache[lambda]
  end

  -- done with nvrtc, switch to Driver API
  local context = ffi.new'CUcontext[1]'
  local module = ffi.new'CUmodule[1]'
  local func = ffi.new'CUfunction[1]'

  CU.errcheck('cuCtxGetCurrent', context)
  CU.errcheck('cuModuleLoadDataEx', module, ptx, 0, nil, nil)
  CU.errcheck('cuModuleGetFunction', func, module[0], 'kernel')

  local args = ffi.new'void*[2]'
  args[0] = ffi.new('float*[1]', self:data())
  args[1] = ffi.new('int[1]', self:numel())
  
  CU.errcheck('cuLaunchKernel', func[0],
  		CUDA_NUM_THREADS, 1, 1,
		get_blocks(self:numel()), 1, 1,
		0, nil,
		args, nil)

  CU.errcheck('cuCtxSynchronize')
  CU.errcheck('cuModuleUnload', module[0])
end
