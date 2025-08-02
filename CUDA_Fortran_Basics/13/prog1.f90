module mrand
    use curand_device
    integer, parameter :: n = 500
    contains
    attributes(global) subroutine randsub(a)
    real, device :: a(n,n,4)
    type(curandStateXORWOW) :: h
    integer(8) :: seed, seq, offset
    j = blockIdx%x; i = threadIdx%x
    seed = 12345_8 + j*n*n + i*2
    seq = 0_8
    offset = 0_8
    call curand_init(seed, seq, offset, h)
    do k = 1, 4
        a(i,j,k) = curand_uniform(h)
    end do
    end subroutine
end module

program t   ! nvfortran t.cuf
use mrand
use cudafor ! recognize maxval, minval, sum w/managed
real, managed :: a(n,n,4)
a = 0.0
call randsub<<<n,n>>>(a)
print *,maxval(a),minval(a),sum(a)/(n*n*4)
end program
