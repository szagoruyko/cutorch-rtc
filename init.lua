require 'nvrtc'
CU = {}
include 'ffi.lua'

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


function torch.CudaTensor:apply1(lambda)
  assert(type(lambda) == 'string')

  CU.APPLY_C.THCudaTensor_pointwiseApply1(cutorch.getState(),
  		self:cdata(),
		CU.APPLY_INCLUDE, lambda)
  return self
end


function torch.CudaTensor:apply2(b, lambda)
  assert(type(lambda) == 'string')
  assert(torch.type(b) == 'torch.CudaTensor')

  CU.APPLY_C.THCudaTensor_pointwiseApply2(cutorch.getState(),
  		self:cdata(), b:cdata(), 
		CU.APPLY_INCLUDE, lambda)
  return self
end


function torch.CudaTensor:apply3(b, c, lambda)
  assert(type(lambda) == 'string')
  assert(torch.type(b) == 'torch.CudaTensor')
  assert(torch.type(c) == 'torch.CudaTensor')

  CU.APPLY_C.THCudaTensor_pointwiseApply3(cutorch.getState(),
  		self:cdata(), b:cdata(), c:cdata(),
		CU.APPLY_INCLUDE, lambda)
  return self
end
