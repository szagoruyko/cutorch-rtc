
function torch.CudaTensor:new_apply(lambda)
  CU.APPLY_C.THCudaTensor_pointwiseApply1(cutorch.getState(), self:cdata(), CU.APPLY_INCLUDE, lambda)
  return self
end

