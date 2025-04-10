! Kernel definition
attributes(global) subroutine saxpy(x, y, a)
    implicit none
    real :: x(:), y(:)
    real, value :: a
    integer :: n, i
    n = size(x)
    i = blockDim%x * (blockIdx%x-1) + threadIdx%x
    if (i <= n) then
        y(i) = y(i) + a*x(i)
    endif
end subroutine saxpy

program run
  use cudafor
  implicit none
  integer, parameter :: N = 10000
  real :: x1(N), y1(N), a
  real, device :: xd(N), yd(N)  ! Fixed-size device arrays
  integer :: istat, i

  ! Initialize host arrays
  do i = 1, N
      x1(i) = 2.0
      y1(i) = 3.0
  end do
  a = 2.0

  print *, "Host x(1:5): ", x1(1:5)
  print *, "Host y(1:5): ", y1(1:5)

  ! Copy to device
  xd = x1
  yd = y1

  print *, "Copied to device"

  ! Kernel launch
  print *, "Launching kernel with N = ", N
  print *, "Grid size: ", (N + 255) / 256
  print *, "Block size: 256"
  call saxpy<<<(N+255)/256, 256>>>(xd, yd, a)
  ! Copy result back to host
  y1 = yd
  print *, "Device y(1:5) after kernel: ", y1(1:5)
end program run

