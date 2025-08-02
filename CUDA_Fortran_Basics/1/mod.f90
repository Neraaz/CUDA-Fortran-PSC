module mytests
contains
attributes(global)  &
subroutine test1( a )
integer, device :: a(*)
i = threadIdx%x
a(i) = i
return
end subroutine test1
end module mytests
