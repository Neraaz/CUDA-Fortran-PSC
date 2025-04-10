program p
!@cuf use cudafor
real a(1000)
!@cuf attributes(managed) :: a
real b(1000)
!@cuf real, device :: b_dev(1000)
b = 2.0
!@cuf b_dev = b
!@cuf associate(b=>b_dev)
!$cuf kernel do(1) <<<*,*>>>
do i = 1, 1000
    a(i) = real(i) * b(i)
end do
!@cuf end associate
#ifdef _CUDA
print *,"GPU sum passed? ",sum(a).eq.1000*1001
#else
print *,"CPU sum passed? ",sum(a).eq.1000*1001
#endif
end program
