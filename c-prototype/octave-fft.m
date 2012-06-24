function ret = _w(i)
  ret = tanh(1 / (2* 1 * 32)) * exp(-abs(i) / (1 * 32));
end

_input = [0.186775; 0.186775; 0.186775; 0.186775; 0.186775; 0; 0.186775; 0.186775; 0.186775; 0.186775];
n = size(_input)(1);

input = _input;

for i = linspace(1, n, n)
  expected = 0;
  for j = linspace(1, n, n)
    expected += _w(i-j) * input(j);
  end
  expected
end

input = vertcat(_input, zeros((2*n)-n-1,1));

distances = _w(transpose(linspace(0, (2*n)-n-1, n)));
distances = vertcat(flipud(distances), distances(2:end));

input_f = fft(input);
distances_f = fft(distances);

convolution_f = input_f .* distances_f;

convolution = ifft(convolution_f)(n:end)
