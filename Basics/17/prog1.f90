! start the module containing the matmul kernel
module mmul_mod
use cudafor
contains
 ! mmul_kernel computes A*B into C where
 ! A is NxM, B is MxL, C is then NxL
 attributes(global) subroutine mmul_kernel( A, B, C, N, M, L )
  real :: A(N,M), B(M,L), C(N,L)
  integer, value :: N, M, L
  integer :: i, j, kb, k, tx, ty
  ! submatrices stored in shared memory
  real, shared :: Asub(16,16), Bsub(16,16)
  ! the value of C(i,j) being computed
  real :: Cij
  ! Get the thread indices
  tx = threadidx%x
  ty = threadidx%y
  ! This thread computes C(i,j) = sum(A(i,:) * B(:,j))
  i = (blockidx%x-1) * 16 + tx
  j = (blockidx%y-1) * 16 + ty
  Cij = 0.0
  ! Do the k loop in chunks of 16, the block size
  do kb = 1, M, 16
    ! Fill the submatrices
    ! Each of the 16x16 threads in the thread block
    ! loads one element of Asub and Bsub
     Asub(tx,ty) = A(i,kb+ty-1)
    Bsub(tx,ty) = B(kb+tx-1,j)
    ! Wait until all elements are filled
    call syncthreads()
    ! Multiply the two submatrices
    ! Each of the 16x16 threads accumulates the
    ! dot product for its element of C(i,j)
    do k = 1,16
      Cij = Cij + Asub(tx,k) * Bsub(k,ty)
    enddo
    ! Synchronize to make sure all threads are done
    ! reading the submatrices before overwriting them
    ! in the next iteration of the kb loop
    call syncthreads()
  enddo
  ! Each of the 16x16 threads stores its element
  ! to the global C array
  C(i,j) = Cij
  end subroutine mmul_kernel
  subroutine mmul( A, B, C )
    real, dimension(:,:) :: A, B, C
    ! allocatable device arrays
    real, device, allocatable, dimension(:,:) :: Adev,Bdev,Cdev
    ! dim3 variables to define the grid and block shapes
    type(dim3) :: dimGrid, dimBlock
    ! Get the array sizes
    N = size( A, 1 )
    M = size( A, 2 )
    L = size( B, 2 )
    ! Allocate the device arrays
    allocate( Adev(N,M), Bdev(M,L), Cdev(N,L) )
    ! Copy A and B to the device
    Adev = A(1:N,1:M)
    Bdev(:,:) = B(1:M,1:L)
    ! Create the grid and block dimensions
    dimGrid = dim3( N/16, L/16, 1 )
    dimBlock = dim3( 16, 16, 1 )
    call mmul_kernel<<<dimGrid,dimBlock>>>( Adev, Bdev, Cdev, N, M, L)
   ! Copy the results back and free up memory
    C(1:N,1:L) = Cdev
    deallocate( Adev, Bdev, Cdev )
  end subroutine mmul
end module mmul_mod

program main
use mmul_mod
implicit none
real, allocatable :: A1(:,:), B1(:,:), C1(:,:)
integer :: N1, M1, L1
N1 = 1024
M1 = 1024
L1 = 1024
allocate(A1(N1,M1), B1(M1,L1), C1(N1,L1))
A1(:,:) = 2.0
B1(:,:) = 2.0
C1 = 0.0
call mmul(A1,B1,C1)
print *, C1(:,4)
deallocate(A1,B1,C1)
end program main
