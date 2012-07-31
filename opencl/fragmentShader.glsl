//uniform sampler2D mytexture;
//varying vec4 f_color;
uniform vec4 color;
//uniform float sprite;

void main(void) {
  //if(sprite > 1.0)
  //  gl_FragColor = texture2D(mytexture, gl_PointCoord) * f_color;
  //else
	gl_FragColor = color;
}
