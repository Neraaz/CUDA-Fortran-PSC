program test_zgemm
    use iso_c_binding, only: c_double_complex, c_int
    implicit none

    ! Matrix dimensions
    integer(c_int), parameter :: m = 1024, n = 1024, k = 1024
    integer(c_int), parameter :: lda = m, ldb = k, ldc = m

    ! Complex scalars
    complex(c_double_complex), parameter :: alpha = (1.0d0, 0.0d0)
    complex(c_double_complex), parameter :: beta  = (0.0d0, 0.0d0)

    ! Matrices A (m x k), B (k x n), and C (m x n)
    complex(c_double_complex) :: A(m,k), B(k,n), C(m,n)
    integer :: i, j
    real :: start_time, end_time

    ! Initialize A and B with pattern (i + j*I)
    do j = 1, k
        do i = 1, m
            A(i,j) = cmplx(i, j, kind=kind(1.0d0))
        end do
    end do

    do j = 1, n
        do i = 1, k
            B(i,j) = cmplx(i, j, kind=kind(1.0d0))
        end do
    end do

    ! Initialize C to zero
    C = (0.0d0, 0.0d0)

    ! Start timer
    call cpu_time(start_time)

    call cuda_zgemm_interface(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)

    ! End timer
    call cpu_time(end_time)

    ! Output timing
    print '(A,F10.6,A)', 'Computation time: ', end_time - start_time, ' seconds'

    ! Output first 2x2 block of result for verification
    print *, 'First 2x2 block of Result C:'
    do i = 1, 2
        print *, C(i,1:2)
    end do
end program test_zgemm
