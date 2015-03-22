require 'cutorch'
local C = dofile 'ffi.lua'
local ffi = require 'ffi'

local errcheck = function(f, ...)
   local status = C[f](...)
   if status ~= 'CUDA_SUCCESS' then
      local str = ffi.string(C.CUGetErrorString(status))
      error('Error in cuda: ' .. str)
   end
end

function cutorch.launchPTX(ptx, arguments, blocks, threads)
  assert(torch.type(blocks) == 'table' and #blocks > 0)
  assert(torch.type(threads) == 'table' and #threads > 0)

  local module = ffi.new'CUmodule[1]'
  local func = ffi.new'CUfunction[1]'

  errcheck('cuModuleLoadDataEx', module, ptx, 0, nil, nil)
  errcheck('cuModuleGetFunction', func, module[0], 'kernel')

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
           threads[1], threads[2] or 1, threads[3] or 1,
           blocks[1], blocks[2] or 1, blocks[3] or 1,
           0, nil, args, nil)

  errcheck('cuModuleUnload', module[0])
end
