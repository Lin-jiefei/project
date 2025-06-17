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
    
     // 设置矩阵元素
    PetscReal coeff = kappa * dt / (h * h);  // 离散化系数
    PetscReal diag_val = rho_c + 4.0 * coeff;  // 主对角线系数
    
    // 设置所有局部节点
    for (PetscInt idx = Istart; idx < Iend; idx++) {
        PetscInt    i = idx / N;  
        PetscInt    j = idx % N; 
        PetscInt    cols[5];
        PetscScalar vals[5];
        PetscInt    ncols = 0;
        
        // 主对角线元素
        cols[ncols] = idx;
        vals[ncols] = diag_val;
        ncols++;
        
        // 相邻节点：左 (i-1, j)
        if (i > 0) {
            cols[ncols] = idx - N;
            vals[ncols] = -coeff;
            ncols++;
        }
        
        // 相邻节点：右 (i+1, j)
        if (i < N - 1) {
            cols[ncols] = idx + N;
            vals[ncols] = -coeff;
            ncols++;
        }
        
        // 相邻节点：下 (i, j-1)
        if (j > 0) {
            cols[ncols] = idx - 1;
            vals[ncols] = -coeff;
            ncols++;
        }
        
        // 相邻节点：上 (i, j+1)
        if (j < N - 1) {
            cols[ncols] = idx + 1;
            vals[ncols] = -coeff;
            ncols++;
        }
     
        // 设置值
        MatSetValues(A, 1, &idx, ncols, cols, vals, INSERT_VALUES);
    }
    
    // 完成矩阵装配
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    
    // 2. 设置初始条件向量   
     // 设置初始温度分布 
    Vec u;
    PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
    PetscCall(VecSetSizes(u, PETSC_DECIDE, total_nodes));
    PetscCall(VecSetFromOptions(u));
    PetscCall(VecSet(u, 0.0));  // 初始化为0
    VecAssemblyBegin(u);
    VecAssemblyEnd(u);
    Vec u_old, rhs; // 创建临时向量存储旧解和右端项
    PetscCall(VecDuplicate(u, &u_old));
    PetscCall(VecDuplicate(u, &rhs));
    

    // 可选：查看初始向量
    if (view_exact) {
        PetscPrintf(PETSC_COMM_WORLD, "===== Initial Vector =====\n");
        VecView(u, PETSC_VIEWER_STDOUT_WORLD);
    }
    
    // 3. 创建KSP求解器上下文
    KSP ksp;
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));  // 设置系统矩阵
    
    // 设置求解器
    PetscCall(KSPSetTolerances(ksp, tol, PETSC_DEFAULT, PETSC_DEFAULT，max_steps));
    
    // 允许命令行覆盖所有求解器选项
    PetscCall(KSPSetFromOptions(ksp));
    
    // 4.时间步循环
    PetscInt    step;
    PetscReal   time = 0.0; 
    PetscBool   converged = PETSC_FALSE;
   for (step = 0; step < max_steps; step++) {
        // 更新当前时间
        time += dt;
        if (time > T_final) break;
        
        // 保存旧解
        PetscCall(VecCopy(u, u_old));
        
        // 组装右端向量 
        PetscCall(VecCopy(u_old, rhs));
        PetscCall(VecScale(rhs, rho_c));
        
        // 测量求解时间 - 开始
        PetscLogDouble solve_start, solve_end;
        PetscTime(&solve_start);
        
        // 求解线性系统 A * u = rhs
        PetscCall(KSPSolve(ksp, rhs, u));
        
        // 测量求解时间 - 结束
        PetscTime(&solve_end);
        solve_time_total += (solve_end - solve_start);
        
        // 计算时间步之间的变化量
        PetscReal norm_diff;
        PetscCall(VecAXPY(u_old, -1.0, u)); // u_old = u - u_old
        PetscCall(VecNorm(u_old, NORM_2, &norm_diff));
        
        // 每10步打印进度
        if (step % 10 == 0 && rank == 0) {
            PetscPrintf(PETSC_COMM_WORLD, "Step %4d: Time = %.4f, Norm(delta_u) = %.4e\n", 
                       step, (double)time, (double)norm_diff);
        }
        
        // 检查收敛
        if (norm_diff < tol) {
            converged = PETSC_TRUE;
            break;
        }
    }
    // 结束总时间测量
    PetscTime(&total_time_end);
    double total_elapsed_time = (double)(total_time_end - total_time_start);
    
  // 5.结果输出 
    if (rank == 0) {
        if (converged) {
            PetscPrintf(PETSC_COMM_WORLD, "\n===== CONVERGED =====\n");
            PetscPrintf(PETSC_COMM_WORLD, "Time steps:     %d\n", step);
            PetscPrintf(PETSC_COMM_WORLD, "Final time:     %.4f\n", (double)time);
            PetscPrintf(PETSC_COMM_WORLD, "Final norm_diff: %.4e\n", (double)norm_diff);
        } else {
            PetscPrintf(PETSC_COMM_WORLD, "\nWARNING: Not converged after %d time steps\n", max_steps);
            PetscPrintf(PETSC_COMM_WORLD, "Final time:     %.4f\n", (double)time);
        }
    }
    
    // 查看最终解
    if (view_exact) {
        PetscPrintf(PETSC_COMM_WORLD, "===== Final Solution =====\n");
        VecView(u, PETSC_VIEWER_STDOUT_WORLD);
    }
    
    
    // 6. 性能分析
    KSPConvergedReason reason;
    KSPGetConvergedReason(ksp, &reason);
    
    PetscInt its;
    KSPGetIterationNumber(ksp, &its);
    
    if (rank == 0) {
        PetscPrintf(PETSC_COMM_WORLD, "\n===== Performance Summary =====\n");
        PetscPrintf(PETSC_COMM_WORLD, "Total Time Steps:           %d\n", step);
        PetscPrintf(PETSC_COMM_WORLD, "Total Iterations of KSP Solver:  %d\n", its);
        PetscPrintf(PETSC_COMM_WORLD, "Total Solve Time (linear solves): %.4f sec\n", (double)solve_time_total);
        PetscPrintf(PETSC_COMM_WORLD, "Total Elapsed Time:              %.4f sec\n", total_elapsed_time);
        
        if (step > 0) {
            double avg_solve_time = solve_time_total * 1000.0 / step;
            PetscPrintf(PETSC_COMM_WORLD, "Avg Time per Linear Solve:       %.4f ms\n", avg_solve_time);
        }
        
        PetscPrintf(PETSC_COMM_WORLD, "Converged Reason:               %s\n", KSPConvergedReasons[reason]);
    }
    
    // 7. 资源清理
    VecDestroy(&u);
    VecDestroy(&u_old);
    VecDestroy(&rhs);
    MatDestroy(&A);
    KSPDestroy(&ksp);
    
    PetscFinalize();
    return 0;
}
