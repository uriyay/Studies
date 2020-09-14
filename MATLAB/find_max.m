A = [23 42 37 18 52];

max_val = -inf;
for i = 1:length(A)
  if A(i) > max_val
    max_val = A(i)
  endif
endfor
disp(max_val);
