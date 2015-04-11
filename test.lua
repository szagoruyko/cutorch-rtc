require 'cutorch-rtc'

local mytester = torch.Tester()

local rtctest = {}

function rtctest.apply1()
  local a = torch.rand(32):cuda()
  local ref_y = torch.sqrt(a)
  local ptx_y = a:clone():apply1'x = sqrt(x)'

  local ferr = torch.max((ref_y - ptx_y):abs())
  mytester:asserteq(ferr, 0, 'apply1 error')
end

function rtctest.noncontig_apply1()
  local a = torch.rand(32,2):cuda():select(2,1)
  local ref_y = torch.sqrt(a)
  local ptx_y = a:clone():apply1'x = sqrt(x)'

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

mytester:add(rtctest)
mytester:run()
