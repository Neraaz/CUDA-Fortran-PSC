program testSaxpy
  use cudafor
  implicit none
  real, allocatable :: x(:), y(:)
  real, allocatable, device :: x_d(:), y_d(:)
  real, allocatable :: z(:)
  real :: A
  integer :: i, N, grid
  integer, parameter :: tBlock=256

  N = 100000

  allocate(x(N), y(N), x_d(N), y_d(N))
  grid = (N + tBlock - 1)/tBlock

  x = 2.0
  y = 3.0

  x_d = x
  y_d = y

  A = 2.0
  !$cuf kernel do <<<grid, tBlock>>>
  do i = 1, N
   y_d(i) = y_d(i) + A * x_d(i)
  end do 
  y = y_d

  print *, "Size of arrays: ", N
  print *, "Constant A:", A
  print *, y(1)
  deallocate(x,y,x_d,y_d)
end program testSaxpy
