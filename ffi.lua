local ffi = require 'ffi'

ffi.cdef[[
void launchPTX(THCState* state, const char* ptx, const char* name, void* args[], int* grid, int* block);
]]

local cdef = [[
bool THCudaTensor_pointwiseApply1(THCState* state,
                                  THCudaTensor* a,
                                  const char* op_string);

bool THCudaTensor_pointwiseApply2(THCState* state,
                                  THCudaTensor* a,
                                  THCudaTensor* b,
                                  const char* op_string);

bool THCudaTensor_pointwiseApply3(THCState* state,
                                  THCudaTensor* a,
                                  THCudaTensor* b,
                                  THCudaTensor* c,
                                  const char* op_string);
]]

for i,v in ipairs{
   {'THCudaTensor', 'THCudaHalfTensor'},
   {'THCudaTensor', 'THCudaTensor'},
   {'THCudaTensor', 'THCudaDoubleTensor'},
} do
   local s = cdef:gsub(unpack(v))
   ffi.cdef(s)
end

return ffi.load(package.searchpath('libcutorchrtc', package.cpath))
