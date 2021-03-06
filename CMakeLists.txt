CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")

FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(CUDA 7.0 REQUIRED)

INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/THC")

SET(src-cuda apply.cu compile_ptx.cpp)
FILE(GLOB luasrc *.lua)

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)
IF(APPLE)
  LINK_DIRECTORIES("${Torch_INSTALL_LIB}" ${CUDA_TOOLKIT_ROOT_DIR}/lib)
ELSE()
  LINK_DIRECTORIES("${Torch_INSTALL_LIB}" ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
ENDIF(APPLE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")

ADD_TORCH_PACKAGE(cutorch-rtc "${src}" "${luasrc}" "cutorch RTC")

CUDA_ADD_LIBRARY(cutorchrtc MODULE ${src-cuda})
TARGET_LINK_LIBRARIES(cutorchrtc luaT THC TH nvrtc cuda)

### Torch packages supposes libraries prefix is "lib"
SET_TARGET_PROPERTIES(cutorchrtc PROPERTIES
  PREFIX "lib"
  IMPORT_PREFIX "lib")
   
INSTALL(TARGETS cutorchrtc
    RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
    LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

INSTALL(
  FILES
  ${luasrc}
  DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/cutorchrtc")
