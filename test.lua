require 'cutorch-rtc'
local nvrtc = require 'nvrtc'

local mytester = torch.Tester()

local rtctest = torch.TestSuite()

local function test_apply(T)
  local a = torch.rand(32):type(T)
  local ref_y = torch.fill(a, 9)
  local ptx_y = a:clone():apply1'x = 9.'

  local ferr = torch.max((ref_y:double() - ptx_y:double()):abs())
  mytester:asserteq(ferr, 0, 'apply1 error')
end

function rtctest.apply1()
   test_apply'torch.CudaTensor'
   test_apply'torch.CudaDoubleTensor'
   -- test_apply'torch.CudaHalfTensor'
end

--[[
function rtctest.noncontig_apply1()
  local a = torch.rand(32,2):cuda():select(2,1)
  local ref_y = torch.mul(a, 9)
  local ptx_y = a:clone():apply1'x = x * 9.'

  local ferr = torch.max((ref_y - ptx_y):abs())
  mytester:asserteq(ferr, 0, 'apply1 error')
end

function rtctest.apply2()
  local a = torch.rand(32):cuda()
  local b = torch.rand(32):cuda()
  local ref_y = torch.cmul(a,b)
  local ptx_y = a:clone():apply2(b, 'x = x*y')

  local ferr = torch.max((ref_y - ptx_y):abs())
  mytester:asserteq(ferr, 0, 'apply2 error')
end

function rtctest.noncontig_apply2()
  local a = torch.rand(32,2):cuda():select(2,1)
  local b = torch.rand(32):cuda()
  local ref_y = torch.cmul(a,b)
  local ptx_y = a:clone():apply2(b, 'x = x*y')

  local ferr = torch.max((ref_y - ptx_y):abs())
  mytester:asserteq(ferr, 0, 'apply2 error')
end

function rtctest.apply3()
  local a = torch.rand(32):cuda()
  local b = torch.rand(32):cuda()
  local c = torch.rand(32):cuda()
  local ref_y = torch.cmul(b,c)
  local ptx_y = a:clone():apply3(b, c, 'x = y*z')

  local ferr = torch.max((ref_y - ptx_y):abs())
  mytester:asserteq(ferr, 0, 'apply3 error')
end

function rtctest.noncontig_apply3()
  local a = torch.rand(32):cuda()
  local b = torch.rand(32):cuda()
  local c = torch.rand(32,2):cuda():select(2,1)
  local ref_y = torch.cmul(b,c)
  local ptx_y = a:clone():apply3(b, c, 'x = y*z')

  local ferr = torch.max((ref_y - ptx_y):abs())
  mytester:asserteq(ferr, 0, 'apply3 error')
end
]]

local function measureExecTime(foo)
  local s = torch.tic()
  foo()
  cutorch.synchronize()
  return torch.toc(s)
end

function rtctest.cachetest()
  local a = torch.rand(32):cuda()
  local foo = function() a:apply1'x = 2' end
  local t1 = measureExecTime(foo)
  local t2 = measureExecTime(foo)
  mytester:assertgt(t1/t2, 10, 'apply1 caching')

  local b = torch.rand(32):cuda()
  local foo = function() a:apply2(b, 'x = 2*y') end
  local t1 = measureExecTime(foo)
  local t2 = measureExecTime(foo)
  mytester:assertgt(t1/t2, 10, 'apply2 caching')

  local c = torch.rand(32):cuda()
  local foo = function() a:apply3(b,c, 'x = y*z') end
  local t1 = measureExecTime(foo)
  local t2 = measureExecTime(foo)
  mytester:assertgt(t1/t2, 10, 'apply3 caching')
end

function rtctest.launchPTXtest()
  local kernel = [[
  extern "C" __global__
  void kernel(float *a, int n)
  {
    int tx = blockIdx.x*blockDim.x + threadIdx.x;
    if(tx < n)
      a[tx] *= 2.f;
  }
  ]]

  local ptx = nvrtc.compileReturnPTX(kernel)
  local a = torch.randn(32):cuda()
  local b = a:clone()
  cutorch.launchPTX(ptx, 'kernel', {a, {'int', a:numel()}}, {1}, {32})

  local err = torch.max((a - b*2):abs())
  mytester:asserteq(err, 0, 'launchPTX err')
end

mytester:add(rtctest)
mytester:run()
