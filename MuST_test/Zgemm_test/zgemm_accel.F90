subroutine cuda_zgemm_interface(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) &
    bind(C, name="cuda_zgemm_interface_")
    use iso_c_binding
    implicit none
    
    ! Argument declarations
    integer(c_int), intent(in) :: m, n, k, lda, ldb, ldc
    complex(c_double_complex), intent(in) :: alpha, beta
    complex(c_double_complex), intent(in) :: A(*), B(*)
    complex(c_double_complex), intent(inout) :: C(*)
    
#ifdef CUDA
    call cuda_zgemm_c(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
#else
    call zgemm('n', 'n', m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
#endif
end subroutine cuda_zgemm_interface
