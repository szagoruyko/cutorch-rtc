#ifndef THCRTC_COMPILE_RTC_H
#define THCRTC_COMPILE_RTC_H

#include "THC/THC.h"

void launch(const char* ptx, const char* name, void* args[], dim3 grid, dim3 block, CUstream stream);

void compilePTX(const char* src,
    		const char* headers[],
		const char* includeNames[],
		std::vector<char>& ptx);

#endif // THCRTC_COMPILE_RTC_H
