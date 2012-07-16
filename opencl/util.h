#pragma once

#include "CL/cl.h"

char* file_contents(const char *filename,
                    int *length);

const char* oclErrorString(cl_int error);
