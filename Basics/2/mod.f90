module ramp
real, constant :: twopi
contains
attributes(global) &
subroutine buildramp(x, n)
real, device :: x(n)
integer, value :: n
real, shared :: term
if (threadidx%x == 1) term = &
twopi / float(n)
call syncthreads()
i = (blockidx%x-1)*blockdim%x &
+ threadidx%x
if (i <= n) then
x(i) = cos(float(i-1)*term)
end if
return
end subroutine
end module
