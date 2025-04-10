program t
interface sort
   subroutine sort_int(array, n) &
        bind(C,name='thrust_int_sort_wrapper')
        integer(4), device, dimension(*) :: array
        integer(4), value :: n
    end subroutine
end interface
integer(4), parameter :: n = 100
integer(4), device :: a_d(n)
integer(4) :: a_h(n)
!$cuf kernel do
do i = 1, n
    a_d(i) = 1 + mod(47*i,n)
end do
call sort(a_d, n)
a_h = a_d
nres  = count(a_h .eq. (/(i,i=1,n)/))
if (nres.eq.n) then
    print *,"test PASSED"
else
    print *,"test FAILED"
endif
end
