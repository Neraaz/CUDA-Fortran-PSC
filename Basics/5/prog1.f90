module mytest
contains
attributes(global) subroutine saxpy(x, y, a, n)
  implicit none
  real, device :: x(n), y(n)
  real, value :: a
  integer, value :: n
  integer :: i
  n = size(x)
  i = blockDim%x * (blockIdx%x - 1) + threadIdx%x
  if (i <= n) then
    y(i) = y(i) + a * x(i)
  end if
end subroutine saxpy
end module mytest

program testSaxpy
  use cudafor
  use mytest
  implicit none
  real, allocatable :: x(:), y(:), z(:)
  real, allocatable, device :: x_d(:), y_d(:)
  real :: A
  integer :: tBlock, grid, j, N, ierr
  !type(dim3) :: grid, tBlock

  N = 100000

  allocate(x(N), y(N), z(N), x_d(N), y_d(N))
  !tBlock = dim3(256,1,1)
  tBlock = 256
  grid = ceiling(real(N)/tBlock)

  !call random_number(x)
  !call random_number(y)
  !call random_number(A)

  x = 2.0
  y = 3.0
  !do j = 1, N
  !  x(j) = 1.0
  !  y(j) = 2.0
  !end do

  A = 2.0
  print *, x(1), y(1), A
  print *, size(x)
  !x = 1.0; y = 2.0; a = 2.0
  x_d = x
  y_d = y
  call saxpy<<<grid, tBlock>>>(x_d, y_d, A, N)
  ierr = cudaDeviceSynchronize()
  z = y_d

  print *, "Size of arrays: ", N
  print *, 'Grid             : ', grid
  print *, 'Threads per block: ', tBlock

  print *, "Constant A:", A
  print *, x(1), z(1)
  deallocate(x,y,z,x_d,y_d)
end program testSaxpy
