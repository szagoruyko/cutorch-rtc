require 'nvrtc'
CU = {}
include 'ffi.lua'
include 'apply.lua'

local ffi = require 'ffi'
local C = CU.C

local errcheck = function(f, ...)
   local status = C[f](...)
   if status ~= 'CUDA_SUCCESS' then
     --TODO: handle errors properly
     print(f, status)
      --local str = ffi.string(C.CUGetErrorString(status))
      --error('Error in cuda: ' .. str)
   end
end

function cutorch.launchPTX(ptx, kernel_name, arguments, gridDim, blockDim)
  assert(torch.type(gridDim) == 'table' and #gridDim > 0)
  assert(torch.type(blockDim) == 'table' and #blockDim > 0)
  assert(torch.Tensor(blockDim):prod() <= 1024)

  local module = ffi.new'CUmodule[1]'
  local func = ffi.new'CUfunction[1]'

  errcheck('cuModuleLoadDataEx', module, ptx, 0, nil, nil)
  errcheck('cuModuleGetFunction', func, module[0], kernel_name)

  local args = ffi.new('void*[?]', #arguments)
  for i,v in ipairs(arguments) do
    if torch.type(v) == 'torch.CudaTensor' then
      args[i-1] = ffi.new('float*[1]', v:data())
    elseif torch.type(v) == 'table' then
      args[i-1] = ffi.new(v[1]..'[1]', v[2])
    else
      --TODO: add textures
      error('unsupported kernel argument #'..i)
    end
  end

  errcheck('cuLaunchKernel', func[0],
           gridDim[1], gridDim[2] or 1, gridDim[3] or 1,
           blockDim[1], blockDim[2] or 1, blockDim[3] or 1,
           0, nil, args, nil)

  errcheck('cuModuleUnload', module[0])
end
