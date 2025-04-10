program t1
use cudafor
use mytests
integer, parameter :: n = 100
integer, allocatable, device :: iarr(:)
integer h(n)
istat = cudaSetDevice(0)
allocate(iarr(n))
h = 0; iarr = h
call test1<<<1,n>>> (iarr)
h = iarr
print *,&
"Errors: ", count(h.ne.(/ (i,i=1,n) /))
deallocate(iarr)
end program t1
