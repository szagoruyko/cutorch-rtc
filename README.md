# cutorch-rtc

Basic feature list:

 * cutorch.launchPTX function
 * simple apply function

This is a simple provisional version to make use of CUDA 7 runtime compilation features.
Installation:
```
luarocks install https://raw.githubusercontent.com/szagoruyko/cutorch-rtc/master/cutorch-rtc-scm-1.rockspec
```
Then after requiring ```cutorch-rtc``` you will get ```launchPTX``` function, which can run ptx code generated with NVRTC, and ```cutorch.apply``` function:
```lua
require 'cutorch-rtc'
t = torch.randn(8):cuda()
t:apply'x = x < 0 ? 0 : x'
```
That would be a simple ReLU implementation. It is possible to use any CUDA device function inside a kernel. An example is in ```apply.lua```.
