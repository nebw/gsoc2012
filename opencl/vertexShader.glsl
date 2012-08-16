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

attribute vec2 coord2d;
//varying vec4 f_color;
//uniform vec4 color;
//uniform float offset_x;
//uniform float offset_y;
//uniform float scale;
//#ifdef GLES2
//uniform lowp float sprite;
//#else
//uniform float sprite;
//#endif

void main(void) {
	gl_Position = vec4(coord2d.xy, 0, 1);
  //gl_Position = vec4((coord2d.x + offset_x) * scale, (coord2d.y + offset_y) * scale, 0, 1);
  //f_color = color;
  //f_color = vec4(1, 1, 1, 1);
  //gl_PointSize = max(1.0, sprite);
}
