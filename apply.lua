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
