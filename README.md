# cutorch-rtc

Basic feature list:

 * cutorch.launchPTX function
 * apply kernels from cutorch

This package brings CUDA 7 runtime compilation to Torch. Linux or OS X with C++11 compiler required.
Installation:
```
luarocks install https://raw.githubusercontent.com/szagoruyko/cutorch-rtc/master/cutorch-rtc-scm-1.rockspec
```
Then after requiring ```cutorch-rtc``` you will get ```launchPTX``` function, which can run ptx code generated with NVRTC, and ```cutorch.apply``` functions:
```lua
require 'cutorch-rtc'
t = torch.randn(8):cuda()
t:apply1'x = x < 0 ? 0 : x'
```
That would be a simple ReLU implementation.

## Documentation

### cutorch.launchPTX
Runs compiled PTX.
```lua
function cutorch.launchPTX(ptx, kernel_name, arguments, gridDim, blockDim)
```
Arguments:
 * ptx - compiled PTX lua string
 * kernel_name - name of kernel to run from the given PTX
 * arguments - lua table with CudaTensors as inputs and subtables in the form {'int', n} to provide scalar arguments
 * gridDim - size of the grid table, has to have at least one value, others will be filled with ones
 * blockDim - size of block table, again has to have at least one value, others will be ones

PTX can be generated in runtime with https://github.com/szagoruyko/nvrtc.torch

### apply1

Applies provided operator to a tensor:
```lua
function CudaTensor.apply1(self, op)
```
op has to be a lua string assigning a value to variable 'x'. CUDA built-in __device__ functions can be used, see CUDA documentation for more information. Multiline ops supported, has to be separated with ;
Both contiguous and non-contiguous tensors are valid. First call to any apply operation takes about 0.5s, then the compiled code is cached and other calls are fast.

### apply2

Applies provided operator using two tensors:
```lua
function CudaTensor.apply2(self, a, op)
```
op has to use 'x' and 'y' - self and a tensors. Can assign values to both tensors. See apply1 for properties.

### apply3

Applies provided operator using three tensors:
```lua
function CudaTensor.apply3(self, a, b, op)
```
op has to use 'x', 'y' and 'z' - self, a and b tensors. Can assign values to all three tensors. See apply1 for properties.

