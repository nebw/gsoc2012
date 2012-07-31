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
