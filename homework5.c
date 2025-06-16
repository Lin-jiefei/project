#include <petscksp.h>
#include <math.h>

int main(int argc, char **argv) {
    const char *help = "This project solve a transient heat equation in a two-dimensional unit square"
	"ρc∂u/∂t − κ∂2u/∂x2 = f on Ω × (0, T )"
	"u = g on Γg × (0, T )"
	"κ∂u/∂xnx = h on Γh × (0, T )"
	"u|t=0 = u0 in Ω."
	"Options:\n"
	"  -n <size>       : Mesh size (default: 100)\n"
	"  -dt <timestep>  : Time step size (default: 0.001)\n"
	"  -max_steps <int> : Maximum time steps (default: 1000)\n"
	"  -tol <value>    : Convergence tolerance (default: 1e-8)\n"
	"  -view_solution  : View final solution\n"
	"  KSP/PC options  : Any standard PETSc options for solvers/preconditioners\n";
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));
    PetscMPIInt rank;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
        // 添加时间测量
    PetscLogDouble total_time_start, total_time_end;
    PetscTime(&total_time_start);
    PetscLogDouble solve_time_total = 0.0;  // 累计求解时间
    

    PetscInt    N = 10000;          // 矩阵默认大小
    PetscReal   tol = 1e-8;        // 收敛容差
    PetscBool   view_exact = PETSC_FALSE; // 是否查看精确解
    PetscReal   dt = 0.001;        // 时间步长
    PetscInt    max_steps = 1000;  // 最大时间步数
    PetscReal   T_final = 1.0;     // 模拟结束时间
    PetscReal   kappa = 1.0;       // 热传导系数
    PetscReal   rho_c = 1.0;       // ρc 乘积 
    // 从命令行获取参数
    PetscOptionsGetInt(NULL, NULL, "-n", &N, NULL);
    PetscOptionsGetReal(NULL, NULL, "-tol", &tol, NULL);
        PetscOptionsGetReal(NULL, NULL, "-dt", &dt, NULL);
    PetscOptionsGetInt(NULL, NULL, "-max_steps", &max_steps, NULL);
    PetscOptionsGetBool(NULL, NULL, "-view_exact", &view_exact, NULL);
    
    if (rank == 0) {
        PetscPrintf(PETSC_COMM_WORLD, "=== project ===\n");
        PetscPrintf(PETSC_COMM_WORLD, "Matrix size: %d, Tolerance: %.1e, Max steps: %d, Time step: %.3f\n", 
                    N, tol, max_steps, (double)dt;
    }
    
    // 1. 创建并装配矩阵 A
    Mat A;
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, total_nodes, total_nodes));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatSetUp(A));
    // 预分配矩阵内存
    PetscInt max_nnz = 5;
    PetscCall(MatMPIAIJSetPreallocation(A, max_nnz, NULL, max_nnz, NULL));
   
    // 获取进程的矩阵部分
    PetscInt Istart, Iend;
    PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));
    
    // 设置矩阵元素（仅处理本地部分）
    for (PetscInt i = Istart; i < Iend; i++) {
        PetscInt    cols[3];
        PetscScalar vals[3];
        PetscInt    ncols = 0;
        
        // 左邻居（如果存在）
        if (i > 0) {
            cols[ncols] = i - 1;
            vals[ncols] = -1.0;
            ncols++;
        }
        
        // 对角线元素
        cols[ncols] = i;
        vals[ncols] = 2.0;  // 主对角元为2
        ncols++;
        
        // 右邻居（如果存在）
        if (i < N - 1) {
            cols[ncols] = i + 1;
            vals[ncols] = -1.0;
            ncols++;
        }
        
        // 设置值
        MatSetValues(A, 1, &i, ncols, cols, vals, INSERT_VALUES);
    }
    
    // 完成矩阵装配
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    
    // 2. 设置初始向量 z0 = [1,0,...,0]^T
    Vec z;
    PetscCall(VecCreate(PETSC_COMM_WORLD, &z));
    PetscCall(VecSetSizes(z, PETSC_DECIDE, N));
    PetscCall(VecSetFromOptions(z));
    PetscCall(VecSet(z, 0.0));  // 初始化为全零
    
    // 仅rank 0进程设置第一个元素为1
    if (rank == 0) {
        PetscInt idx = 0;
        PetscScalar val = 1.0;
        VecSetValues(z, 1, &idx, &val, INSERT_VALUES);
    }
    VecAssemblyBegin(z);
    VecAssemblyEnd(z);
    
    // 可选：查看初始向量
    if (view_exact) {
        PetscPrintf(PETSC_COMM_WORLD, "===== Initial Vector =====\n");
        VecView(z, PETSC_VIEWER_STDOUT_WORLD);
    }
    
    // 3. 创建KSP求解器上下文
    KSP ksp;
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));  // 设置系统矩阵
    
    // 设置初始的容差
    PetscCall(KSPSetTolerances(ksp, 1e-12, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT));
    
    // 允许命令行覆盖所有求解器选项
    PetscCall(KSPSetFromOptions(ksp));
    
    // 4. 逆幂迭代主循环
    PetscInt    k;
    PetscReal   lambda_prev = 0.0; // 上一次迭代的特征值
    PetscReal   lambda_min = 0.0;  // 当前特征值（实际是A^{-1}的最大特征值）
    PetscBool   converged = PETSC_FALSE;
    Vec         y;
    
    PetscCall(VecDuplicate(z, &y));
   
    // 主迭代循环
    for (k = 0; k < max_iter; k++) {
        // 测量求解时间 - 开始
        PetscLogDouble solve_start, solve_end;
        PetscTime(&solve_start);
        
        // 求解线性系统 A y = z
        PetscCall(KSPSolve(ksp, z, y));
        
        // 测量求解时间 - 结束
        PetscTime(&solve_end);
        solve_time_total += (solve_end - solve_start);
        
        // 计算新向量的范数 ||y||
        PetscReal norm_y;
        PetscCall(VecNorm(y, NORM_2, &norm_y));
        
        // 归一化 z_new = y / ||y||
        PetscCall(VecCopy(y, z));
        PetscCall(VecScale(z, 1.0 / norm_y));
        
        // 计算正确的Rayleigh商：λ = (z^{k})^T * y^{k+1}
        PetscReal lambda_inv;
        PetscCall(VecDot(z, y, &lambda_inv));
        lambda_min = PetscRealPart(lambda_inv); // A^{-1}的最大特征值
        
        // 检查收敛（基于特征值的变化）
        if (k > 0) {
            PetscReal diff_eig = PetscAbsReal(lambda_min - lambda_prev) / PetscAbsReal(lambda_min);
            if (diff_eig < tol) {
                converged = PETSC_TRUE;
                break;
            }
        }
        lambda_prev = lambda_min;
        
        // 每10步打印进度（仅root进程）
        if (k % 10 == 0 && rank == 0) {
            PetscReal actual_lambda = 1.0 / lambda_min; // A的最小特征值
            PetscPrintf(PETSC_COMM_WORLD, "Iter %3d: lambda_min = %.6e\n", 
                       k, (double)actual_lambda);
        }
    }
    
    // 结束总时间测量
    PetscTime(&total_time_end);
    double total_elapsed_time = (double)(total_time_end - total_time_start);
    
    // 5. 结果输出
    PetscReal actual_lambda = 1.0 / lambda_min; // 实际的最小特征值
    PetscReal exact_min = 4.0 * PetscSinReal(PETSC_PI/(2*(N+1))) * 
                          PetscSinReal(PETSC_PI/(2*(N+1)));
    
    if (converged) {
        PetscReal error = PetscAbsReal(actual_lambda - exact_min) / exact_min;
        
        if (rank == 0) {
            PetscPrintf(PETSC_COMM_WORLD, "\n===== CONVERGED =====\n");
            PetscPrintf(PETSC_COMM_WORLD, "Iterations:   %d\n", k);
            PetscPrintf(PETSC_COMM_WORLD, "Computed Min Eigenvalue: %.12e\n", (double)actual_lambda);
            PetscPrintf(PETSC_COMM_WORLD, "Exact Min Eigenvalue:    %.12e\n", (double)exact_min);
            PetscPrintf(PETSC_COMM_WORLD, "Relative Error:          %.4e\n", (double)error);
        }
    } else if (rank == 0) {
        PetscPrintf(PETSC_COMM_WORLD, "\nWARNING: Not converged after %d iterations\n", max_iter);
        PetscPrintf(PETSC_COMM_WORLD, "Computed Min Eigenvalue: %.12e\n", (double)actual_lambda);
        PetscPrintf(PETSC_COMM_WORLD, "Exact Min Eigenvalue:    %.12e\n", (double)exact_min);
    }
    
    // 6. 性能分析
    KSPConvergedReason reason;
    KSPGetConvergedReason(ksp, &reason);
    
    PetscInt its;
    KSPGetIterationNumber(ksp, &its);
    
    if (rank == 0) {
        PetscPrintf(PETSC_COMM_WORLD, "\n===== Performance Summary =====\n");
        PetscPrintf(PETSC_COMM_WORLD, "Total Iterations of KSP Solver:  %d\n", its);
        PetscPrintf(PETSC_COMM_WORLD, "Total Inverse Iterations:       %d\n", k);
        PetscPrintf(PETSC_COMM_WORLD, "Total Solve Time (linear solves): %.4f sec\n", (double)solve_time_total);
        PetscPrintf(PETSC_COMM_WORLD, "Total Elapsed Time:              %.4f sec\n", total_elapsed_time);
        
        if (k > 0) {
            double avg_solve_time = solve_time_total * 1000.0 / k;
            PetscPrintf(PETSC_COMM_WORLD, "Avg Time per Linear Solve:       %.4f ms\n", avg_solve_time);
        }
        
        PetscPrintf(PETSC_COMM_WORLD, "Converged Reason:               %s\n", KSPConvergedReasons[reason]);
    }
    
    // 7. 资源清理
    VecDestroy(&z);
    VecDestroy(&y);
    MatDestroy(&A);
    KSPDestroy(&ksp);
    
    PetscFinalize();
    return 0;
}
