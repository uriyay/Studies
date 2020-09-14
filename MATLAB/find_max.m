B = [23 42 37 18 52];

function max_val = get_max(A)
  max_val = -inf;
  for i = 1:length(A)
    if A(i) > max_val
      max_val = A(i)
    endif
  endfor
endfunction

disp(get_max(B));
