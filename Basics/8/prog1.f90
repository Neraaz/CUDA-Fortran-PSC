module testfill
    use ffill
    contains
    attributes(global) subroutine Kernel(arr)
        integer, device :: arr(*)
        call fill(arr)
    end subroutine Kernel

    integer function sum(arr)
        integer, device :: arr(:)
        sum = 0
        !$cuf kernel do <<<*,*>>>
        do i = 1, size(arr)
          sum = sum + arr(i)
        end do
    end function sum
end module testfill

program tfill
use testfill
integer, device :: iarr(100)
iarr = 0
call Kernel<<<1,100>>>(iarr)
print *,sum(iarr)==100*101/2
end program tfill
