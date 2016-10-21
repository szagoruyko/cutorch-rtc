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
    elseif torch.type(v) == 'torch.CudaHalfTensor' then
      args[i-1] = ffi.new('half*[1]', v:data())
    elseif torch.type(v) == 'torch.CudaDoubleTensor' then
      args[i-1] = ffi.new('double*[1]', v:data())
    elseif torch.type(v) == 'torch.CudaIntTensor' then
      args[i-1] = ffi.new('int*[1]', v:data())
    elseif torch.type(v) == 'torch.CudaByteTensor' then
      args[i-1] = ffi.new('uint8*[1]', v:data())
    elseif torch.type(v) == 'torch.CudaCharTensor' then
      args[i-1] = ffi.new('int8*[1]', v:data())
    elseif torch.type(v) == 'table' then
      args[i-1] = ffi.new(v[1]..'[1]', v[2])
    elseif torch.type(v) == 'cdata' then
      args[i-1] = v
    else
      --TODO: add textures
      error('unsupported kernel argument #'..i..': '..torch.type(v))
    end
  end

  local grid = ffi.new('int[3]', 1)
  local block = ffi.new('int[3]', 1)
  for i,v in ipairs(gridDim) do grid[i-1] = v end
  for i,v in ipairs(blockDim) do block[i-1] = v end
  C.launchPTX(cutorch.getState(), ptx, kernel_name, args, grid, block)
end

local types = {
   'CudaTensor',
   'CudaHalfTensor',
   'CudaDoubleTensor',
}

for i,ttype in ipairs(types) do
   torch[ttype].apply1 = function(self, lambda)
     assert(type(lambda) == 'string')

     C['TH'..ttype..'_pointwiseApply1'](cutorch.getState(),
                   self:cdata(),
                   APPLY_INCLUDE,
                   lambda)
     return self
   end

   torch[ttype].apply2 = function(self, b, lambda)
     assert(type(lambda) == 'string')

     C['TH'..ttype..'_pointwiseApply2'](cutorch.getState(),
                   self:cdata(),
                   b:cdata(),
                   APPLY_INCLUDE,
                   lambda)
     return self
   end

   torch[ttype].apply3 = function(self, b, c, lambda)
     assert(type(lambda) == 'string')

     C['TH'..ttype..'_pointwiseApply3'](cutorch.getState(),
                   self:cdata(),
                   b:cdata(),
                   c:cdata(),
                   APPLY_INCLUDE,
                   lambda)
     return self
   end
end

