subroutine cuda_zgemm_interface(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    use KindParamModule, only : IntKind, RealKind, CmplxKind
    use ErrorHandlerModule, only : ErrorHandler
    implicit none
    
    ! Argument declarations
    integer(kind=IntKind), intent(in) :: m, n, k, lda, ldb, ldc
    complex(kind=CmplxKind), intent(in) :: alpha, beta
    complex(kind=CmplxKind), intent(in) :: A(lda,k), B(ldb,n)
    complex(kind=CmplxKind), intent(inout) :: C(ldc,n)

    ! Local temporary arrays
    integer(kind=IntKind) :: j
    complex (kind=CmplxKind) :: C_temp(m*n)

    if (m > ldc) then
       call ErrorHandler('cuda_zgemm_interface','m > ldc',m,ldc)
    endif
#ifdef CUDA
    call cuda_zgemm_c(m, n, k, alpha, A, lda, B, ldb, beta, C_temp, m)
    do j = 1, n
       call zcopy(m,C_temp((j-1)*m+1),1,C(1,j),1)
    enddo
#else
    call zgemm('n', 'n', m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
#endif
end subroutine cuda_zgemm_interface
