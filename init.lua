local ffi = require 'ffi'
local cutorch = require 'cutorch'
local C, APPLY_INCLUDE = table.unpack(require 'cutorch-rtc/ffi')

function cutorch.launchPTX(ptx, kernel_name, arguments, gridDim, blockDim)
  assert(torch.type(gridDim) == 'table' and #gridDim > 0)
  assert(torch.type(blockDim) == 'table' and #blockDim > 0)
  assert(torch.Tensor(blockDim):prod() <= 1024)

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

  local grid = ffi.new('int[3]', 1)
  local block = ffi.new('int[3]', 1)
  for i,v in ipairs(gridDim) do grid[i-1] = v end
  for i,v in ipairs(blockDim) do block[i-1] = v end
  C.launchPTX(cutorch.getState(), ptx, kernel_name, args, grid, block)
end


function torch.CudaTensor:apply1(lambda)
  assert(type(lambda) == 'string')

  C.THCudaTensor_pointwiseApply1(cutorch.getState(),
  		self:cdata(),
		APPLY_INCLUDE, lambda)
  return self
end


function torch.CudaTensor:apply2(b, lambda)
  assert(type(lambda) == 'string')
  assert(torch.type(b) == 'torch.CudaTensor')

  C.THCudaTensor_pointwiseApply2(cutorch.getState(),
  		self:cdata(), b:cdata(), 
		APPLY_INCLUDE, lambda)
  return self
end


function torch.CudaTensor:apply3(b, c, lambda)
  assert(type(lambda) == 'string')
  assert(torch.type(b) == 'torch.CudaTensor')
  assert(torch.type(c) == 'torch.CudaTensor')

  C.THCudaTensor_pointwiseApply3(cutorch.getState(),
  		self:cdata(), b:cdata(), c:cdata(),
		APPLY_INCLUDE, lambda)
  return self
end
