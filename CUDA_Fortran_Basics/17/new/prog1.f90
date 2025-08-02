module mmul_mod
use cudafor
contains
  attributes(global) subroutine mmul_kernel(A, B, C, N, M, L)
    real :: A(N,M), B(M,L), C(N,L)
    integer, value :: N, M, L
    integer :: i, j, kb, k, tx, ty
    real, shared :: Asub(16,16), Bsub(16,16)
    real :: Cij
  
    tx = threadidx%x
    ty = threadidx%y
    i = (blockidx%x - 1) * 16 + tx
    j = (blockidx%y - 1) * 16 + ty
  
    Cij = 0.0
    do kb = 1, M, 16
      if (i <= N .and. (kb + ty - 1) <= M) then
        Asub(tx,ty) = A(i, kb + ty - 1)
      else
        Asub(tx,ty) = 0.0  ! Avoid out-of-bounds access
      end if
  
      if (j <= L .and. (kb + tx - 1) <= M) then
        Bsub(tx,ty) = B(kb + tx - 1, j)
      else
        Bsub(tx,ty) = 0.0  ! Avoid out-of-bounds access
      end if
  
      call syncthreads()
  
      if (i <= N .and. j <= L) then
        do k = 1, 16
          Cij = Cij + Asub(tx, k) * Bsub(k, ty)
        end do
      end if
  
      call syncthreads()
    end do
  
    if (i <= N .and. j <= L) then
      C(i, j) = Cij
    end if
  end subroutine mmul_kernel
  
  subroutine mmul(A, B, C)
    real, dimension(:,:) :: A, B, C
    real, device, allocatable, dimension(:,:) :: Adev, Bdev, Cdev
    type(dim3) :: dimGrid, dimBlock
    integer :: N, M, L
  
    N = size(A, 1)
    M = size(A, 2)
    L = size(B, 2)
  
    allocate(Adev(N, M), Bdev(M, L), Cdev(N, L))
  
    Adev = A
    Bdev = B
  
    dimGrid = dim3((N + 15) / 16, (L + 15) / 16, 1)  ! Ensure coverage for non-multiples
    dimBlock = dim3(16, 16, 1)
  
    call mmul_kernel<<<dimGrid, dimBlock>>>(Adev, Bdev, Cdev, N, M, L)
  
    C = Cdev
  
    deallocate(Adev, Bdev, Cdev)
  end subroutine mmul
end module mmul_mod

program main
use mmul_mod
implicit none
real, allocatable :: A1(:,:), B1(:,:), C1(:,:)
integer :: N1, M1, L1
N1 = 1000
M1 = 1000
L1 = 1000
allocate(A1(N1,M1), B1(M1,L1), C1(N1,L1))
A1(:,:) = 2.0
B1(:,:) = 2.0
C1 = 0.0
call mmul(A1,B1,C1)
print *, C1(:,4)
deallocate(A1,B1,C1)
end program main
