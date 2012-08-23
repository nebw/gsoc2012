/**
 * Copyright (C) 2012 Benjamin Wild
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#pragma once

#define __CL_ENABLE_EXCEPTIONS

#ifdef _MSC_VER
# define _CRT_SECURE_NO_DEPRECATE
#endif // ifdef _MSC_VER

#include <cpplog.hpp>

#include <gnuplot_i.h>

#include <GL/glew.h>
#include <GL/glfw.h>

#include "glm/glm.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <CL/cl.h>
#include <CL/cl.hpp>

#include <clFFT.h>

#include "gtest/internal/gtest-internal.h"

#include <cassert>
#include <ctime>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdio.h>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <boost/chrono.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/format.hpp>
#include <boost/foreach.hpp>
#include <boost/scoped_array.hpp>
#include <boost/filesystem.hpp>
