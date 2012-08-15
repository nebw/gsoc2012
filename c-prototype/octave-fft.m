function ret = _w(i)
  ret = tanh(1 / (2* 1 * 32)) * exp(-abs(i) / (1 * 32));
end

%# _input = [0.186775; 0.186775; 0.186775; 0.186775; 0.186775; 0; 0.186775; 0.186775; 0.186775; 0.186775];
%# _input = rand(4, 1);
_input = [0.225325; 0.219934; 0.227601; 0.399255; 0.419467; 0.418893; 0.418829; 0.419278; 0.419278; 0.418829; 0.418893; 0.419467; 0.399255; 0.227601; 0.219934; 0.225325]
n = size(_input)(1);

input = vertcat(_input, zeros(n,1))

distances = _w(transpose(linspace(0, n-1, n)));
distances = vertcat(flipud(distances), distances(2:end));
distances = vertcat(distances, 0)

input_f = fft(input);
distances_f = fft(distances);

convolution_f = input_f .* distances_f;

convolution = ifft(convolution_f)
convolution = convolution(n:end-1)

input = _input;

for i = linspace(1, n, n)
  expected = 0;
  for j = linspace(1, n, n)
    expected += _w(i-j) * input(j);
  end
  
  if(abs(expected - convolution) > 1e-7)
    [expected real(convolution(i))]
  end
end