! Kernel definition
attributes(global) subroutine ksaxpy( n, a, x, y )
    real, dimension(*) :: x,y
    real :: a
    integer :: n, i
    i = (blockidx%x-1) * blockdim%x + threadidx%x
    if( i <= n ) y(i) = a * x(i) + y(i)
end subroutine

program run
  use cudafor
  implicit none
  integer :: n = 1000, ierr
  real :: a = 1.0
  real, dimension(:), allocatable :: x, y
  real, device, dimension(:), allocatable :: x_d, y_d
  allocate(x(n), y(n))
  allocate(x_d(n), y_d(n))
  x = 2.0
  y = 3.0
  ! Debug: Print host arrays
  print *, "Host x(1:5): ", x(1:5)
  print *, "Host y(1:5): ", y(1:5)
  x_d = x
  y_d = y
  ! Debug: Print device arrays (before kernel launch)
  y = y_d
  print *, "Device y(1:5) before kernel: ", y(1:5)
  call ksaxpy<<<(n+63)/64, 64>>>( n, a, x_d, y_d )
  ierr = cudadevicesynchronize()
  if (ierr /= 0) then
      print *, "CUDA synchronization failed with error code: ", ierr
  end if
  y = y_d
  print *, "Device y(1:5) after kernel: ", y(1:5)

  write(*,*), "Final y(1:5): ", y(1:5)
  write(*,*), y
  deallocate(x_d,y_d)
  deallocate(x,y)
end program run
