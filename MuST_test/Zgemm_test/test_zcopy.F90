program test_zgemm
    use iso_c_binding, only: c_double_complex, c_int
    implicit none

    ! Matrix dimensions
    interface
        subroutine zcopy(n, zx, incx, zy, incy)
            integer, intent(in) :: n, incx, incy
            complex*16, intent(in) :: zx(*)
            complex*16, intent(out) :: zy(*)
        end subroutine zcopy
    end interface
    integer(c_int), parameter :: m = 25, n = 25, k = 25

    ! Complex scalars
    complex(c_double_complex), parameter :: alpha = (1.0d0, 0.0d0)
    complex(c_double_complex), parameter :: beta  = (0.0d0, 0.0d0)

    ! Matrices A (m x k), B (k x n), and C (m x n)
    complex(c_double_complex) :: A(m,k), B(k,n), C(m,n)
    integer :: i, j, block_row, block_col
    integer :: start_row, start_col
    integer, parameter :: o_block = 100, p_block = 100, q_block = 100
    integer, parameter :: o = m * o_block, p = n * p_block, q = k * q_block
    integer(c_int), parameter :: lda = o, ldb = q, ldc = o
    real :: start_time, end_time
    complex(c_double_complex) :: Big_A(o,q), Big_B(q,p), Big_C(o,p)

    ! Initialize A and B with pattern (i + j*I)
    do j = 1, k
        do i = 1, m
            if (j == i) then
             A(i,j) = cmplx(2.0, 0.0, kind=kind(1.0d0))
            else
             A(i,j) = cmplx(0.0, 0.0, kind=kind(1.0d0))
            endif
        end do
    end do
    ! Initialize big matrix to zero
    Big_A = (0.0d0, 0.0d0)
    
    ! Copy small_mat to EVERY block in the big matrix
    do block_row = 1, o_block
        do block_col = 1, q_block
            ! Compute starting position for current block
            start_row = (block_row - 1) * m + 1
            start_col = (block_col - 1) * k + 1
            
            ! Copy using zcopy (column-wise)
            do j = 1, k
                call zcopy(m, A(1,j), 1, &
                          Big_A(start_row, start_col + j - 1), 1)
            end do
        end do
    end do

    do j = 1, n
        do i = 1, k
            if (j == i) then
             B(i,j) = cmplx(3.0, 0.0, kind=kind(1.0d0))
            else
             B(i,j) = cmplx(0.0, 0.0, kind=kind(1.0d0))
            endif
        end do
    end do
    ! Initialize big matrix to zero
    
    Big_B = (0.0d0, 0.0d0)
    ! Copy small_mat to EVERY block in the big matrix
    do block_row = 1, q_block
        do block_col = 1, p_block
            ! Compute starting position for current block
            start_row = (block_row - 1) * k + 1
            start_col = (block_col - 1) * n + 1
            
            ! Copy using zcopy (column-wise)
            do j = 1, n
                call zcopy(k, B(1,j), 1, &
                          Big_B(start_row, start_col + j - 1), 1)
            end do
        end do
    end do

    ! Initialize Big_C to zero
    Big_C = (0.0d0, 0.0d0)

    ! Start timer
    call cpu_time(start_time)
    call cuda_zgemm_interface(m, n, k, alpha, A, 25, B, 25, beta, C, 25)
    call cuda_zgemm_interface(o, p, q, alpha, Big_A, lda, Big_B, ldb, beta, Big_C, ldc)
    ! End timer
    call cpu_time(end_time)

    ! Output timing
    print '(A,F10.6,A)', 'Computation time: ', end_time - start_time, ' seconds'

    ! Output first 2x2 block of result for verification
    print *, 'First 2x2 block of Result C:'
    do i = 1, 2
        print *, C(i,1:2), shape(C)
    end do
    do i = 1, 2
        print *, Big_C(i,1:2), shape(Big_C)
    end do
    print *, 'First 2x2 block of Result A:'
    do i = 1, 2
        print *, Big_A(i,1:2), shape(Big_A)
    end do
    print *, 'First 2x2 block of Result B:'
    do i = 1, 2
        print *, Big_B(i,1:2), shape(Big_B)
    end do
end program test_zgemm
