#include <petscksp.h>
#include <math.h>
#include <string.h> 
int main(int argc, char **argv) {
    const char *help = "This project solve a transient heat equation in a two-dimensional unit square"
	"ρc∂u/∂t − κ∂2u/∂x2 = f on Ω × (0, T )"
	"u = g on Γg × (0, T )"
	"κ∂u/∂xnx = h on Γh × (0, T )"
	"u|t=0 = u0 in Ω."
	"  - Finite difference spatial discretization\n"
	"  - Explicit/Implicit Euler time marching schemes\n\n"
	"Options:\n"
	"  -n <size>       : Mesh size (default: 100)\n"
	"  -dt <timestep>  : Time step size (default: 0.001)\n"
	"  -max_steps <int> : Maximum time steps (default: 1000)\n"
	"  -tol <value>    : Convergence tolerance (default: 1e-8)\n"
	"  -view_solution  : View final solution\n"
	"  KSP/PC options  : Any standard PETSc options for solvers/preconditioners\n";
	"  -kappa <value>  : Thermal conductivity (default: 1.0)\n"
	"  -rho_c <value>  : Density * specific heat (default: 1.0)\n"
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
    PetscReal   kappa = 1.0;       // 热传导系数
    PetscReal   rho_c = 1.0;       // ρc 乘积 
    char time_method[20] = "implicit"; // 默认时间方法

    // 从命令行获取参数
    PetscOptionsGetInt(NULL, NULL, "-n", &N, NULL);
    PetscOptionsGetReal(NULL, NULL, "-tol", &tol, NULL);
    PetscOptionsGetReal(NULL, NULL, "-dt", &dt, NULL);
    PetscOptionsGetInt(NULL, NULL, "-max_steps", &max_steps, NULL);
    PetscOptionsGetBool(NULL, NULL, "-view_exact", &view_exact, NULL);
    PetscCall(PetscOptionsGetString(NULL, NULL, "-time_method", time_method, sizeof(time_method), NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-kappa", &kappa, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-rho_c", &rho_c, NULL));
    PetscInt total_nodes = N * N;
    PetscReal h = 1.0 / (N - 1);
    if (rank == 0) {
        PetscPrintf(PETSC_COMM_WORLD, "=== Transient Heat Equation Solver ===\n");
        PetscPrintf(PETSC_COMM_WORLD, "Method: %s Euler, Mesh: %d, dt: %.4f, Steps: %d, Tol: %.1e\n", 
                    time_method, N, (double)dt, max_steps, tol);
        PetscPrintf(PETSC_COMM_WORLD, "Physical params: kappa=%.2f, rho_c=%.2f\n", (double)kappa, (double)rho_c);
    }
	//稳定性分析 
 	if (rank == 0) {
        PetscReal alpha = kappa * dt / (rho_c * h * h);
        PetscPrintf(PETSC_COMM_WORLD, "\n=== Stability Analysis ===\n");
        PetscPrintf(PETSC_COMM_WORLD, "CFL number (α) = %.4f\n", (double)alpha);
        
        if (strcmp(time_method, "explicit") == 0) {
            PetscPrintf(PETSC_COMM_WORLD, "Explicit Euler stability condition: α < 0.25\n");
            if (alpha > 0.25) {
                PetscPrintf(PETSC_COMM_WORLD, "WARNING: CFL condition violated! Simulation may be unstable.\n");
            }
        } else {
            PetscPrintf(PETSC_COMM_WORLD, "Implicit Euler is unconditionally stable\n");
        }
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
     // 显式方法拉普拉斯矩阵   
       Mat K = NULL;
    if (strcmp(time_method, "explicit") == 0) {
        PetscCall(MatCreate(PETSC_COMM_WORLD, &K));
        PetscCall(MatSetSizes(K, PETSC_DECIDE, PETSC_DECIDE, total_nodes, total_nodes));
        PetscCall(MatSetFromOptions(K));
    }
    // 获取进程的矩阵部分
    PetscInt Istart, Iend;
    PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));
    PetscReal coeff;
    PetscReal diag_val;
    
     // 计算隐式方法系数
    if (strcmp(time_method, "implicit") == 0) {
        coeff = kappa * dt / (h * h);
        diag_val = rho_c + 4.0 * coeff;
    } 
    // 计算显式方法系数
    else {
        coeff = -kappa / (h * h);  // 符号变化
        diag_val = 1.0;  // 显式方法不需要包含在系统矩阵中
    }
    // 设置矩阵元素
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
         // 处理显式方法的特殊逻辑
        if (strcmp(time_method, "explicit") == 0) {
            // 显式方法需要填充拉普拉斯矩阵
            if (i > 0) { 
             // 相邻节点：左 (i-1, j)
                cols[ncols] = idx - N; 
                vals[ncols] = coeff;
                ncols++; 
            }
            if (i < N - 1) { 
                // 相邻节点：右 (i+1, j)
                cols[ncols] = idx + N; 
                vals[ncols] = coeff;
                ncols++; 
            }
            if (j > 0) { 
                   // 相邻节点：下 (i, j-1)
                cols[ncols] = idx - 1; 
                vals[ncols] = coeff;
                ncols++; 
            }
            if (j < N - 1) { 
                  // 相邻节点：上 (i, j+1)
                cols[ncols] = idx + 1; 
                vals[ncols] = coeff;
                ncols++; 
            }
            PetscCall(MatSetValues(K, 1, &idx, ncols, cols, vals, INSERT_VALUES));
        } 
        // 隐式方法的常规设置
        else {
            if (i > 0) { 
                cols[ncols] = idx - N; 
                vals[ncols] = -coeff;
                ncols++; 
            }
            if (i < N - 1) { 
                cols[ncols] = idx + N; 
                vals[ncols] = -coeff;
                ncols++; 
            }
            if (j > 0) { 
                cols[ncols] = idx - 1; 
                vals[ncols] = -coeff;
                ncols++; 
            }
            if (j < N - 1) { 
                cols[ncols] = idx + 1; 
                vals[ncols] = -coeff;
                ncols++; 
            }
            PetscCall(MatSetValues(A, 1, &idx, ncols, cols, vals, INSERT_VALUES));
        }
    }
  
      
    
    // 完成矩阵装配
    if (strcmp(time_method, "explicit") == 0) {
        PetscCall(MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY));
    } else {
        PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    }
    
    // 2. 设置初始条件向量   
     // 设置初始温度分布 
    Vec u;
    PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
    PetscCall(VecSetSizes(u, PETSC_DECIDE, total_nodes));
    PetscCall(VecSetFromOptions(u));
    
    PetscCall(VecSet(u, 0.0));
    // 设置热源（中心点）
    PetscInt center_idx = (N/2) * N + N/2;
    PetscScalar src_val = 1.0;
    PetscCall(VecSetValues(u, 1, &center_idx, &src_val, INSERT_VALUES));
    
    PetscCall(VecAssemblyBegin(u));
    PetscCall(VecAssemblyEnd(u));
    
    Vec u_old, rhs; // 创建临时向量存储旧解和右端项
    PetscCall(VecDuplicate(u, &u_old));
    PetscCall(VecDuplicate(u, &rhs));
    
  
    // 3. 创建KSP求解器上下文
      KSP ksp = NULL;
    if (strcmp(time_method, "implicit") == 0) {
        PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
        PetscCall(KSPSetOperators(ksp, A, A)); // 设置系统矩阵
        PetscCall(KSPSetTolerances(ksp, tol, PETSC_DEFAULT, PETSC_DEFAULT, max_steps));   // 设置求解器
        PetscCall(KSPSetFromOptions(ksp));    // 允许命令行覆盖所有求解器选项
    } 
    // 4.时间步循环
    PetscInt    step;
    PetscReal   time = 0.0; 
    PetscReal   norm_diff = 0.0;
    PetscReal   T_final = dt * max_steps;
   for (step = 0; step < max_steps; step++) {
        // 更新当前时间
        time += dt;
        if (time > T_final) break;
        
        // 保存旧解
        PetscCall(VecCopy(u, u_old));
        
       // 显式欧拉法
        if (strcmp(time_method, "explicit") == 0) {
            PetscLogDouble solve_start, solve_end;
            PetscTime(&solve_start);
            
            // 显式更新: u^{n+1} = u^n + (dt/(ρc)) * κ∇²u^n
            PetscCall(MatMult(K, u_old, rhs));
            PetscCall(VecScale(rhs, dt / rho_c));
            PetscCall(VecAXPY(u, 1.0, rhs));
            
            PetscTime(&solve_end);
            solve_time_total += (solve_end - solve_start);
        } 
        // 隐式欧拉法
        else {
            PetscLogDouble solve_start, solve_end;
            PetscTime(&solve_start);
            
            // 右端项: ρc * u_old
            PetscCall(VecCopy(u_old, rhs));
            PetscCall(VecScale(rhs, rho_c));
            
            // 求解: A u^{n+1} = rhs
            PetscCall(KSPSolve(ksp, rhs, u));
            
            PetscTime(&solve_end);
            solve_time_total += (solve_end - solve_start);
        }
        
        // 计算变化量
        PetscCall(VecAXPY(u_old, -1.0, u));
        PetscCall(VecNorm(u_old, NORM_2, &norm_diff));
        
        if (step % 50 == 0 && rank == 0) {
            PetscPrintf(PETSC_COMM_WORLD, "Step %4d: Time=%.4f, ||Δu||=%.2e\n", 
                        step, (double)time, (double)norm_diff);
        }
        
        // 简单收敛检查
        if (norm_diff < tol && step > 10) {
            break;
        }
    }
    // 结束总时间测量
    PetscTime(&total_time_end);
    double total_elapsed_time = (double)(total_time_end - total_time_start);
    
  // 5.结果输出 
    if (rank == 0) {
        PetscPrintf(PETSC_COMM_WORLD, "\n===== Results =====\n");
        PetscPrintf(PETSC_COMM_WORLD, "Total steps: %d, Final time: %.4f\n", step, (double)time);
        PetscPrintf(PETSC_COMM_WORLD, "Final change norm: %.4e\n", (double)norm_diff);
        PetscPrintf(PETSC_COMM_WORLD, "Total solve time: %.4f sec\n", (double)solve_time_total);
        PetscPrintf(PETSC_COMM_WORLD, "Total elapsed time: %.4f sec\n", total_elapsed_time);
    }
    
    // 查看最终解
    if (view_exact) {
        PetscPrintf(PETSC_COMM_WORLD, "===== Final Solution =====\n");
        VecView(u, PETSC_VIEWER_STDOUT_WORLD);
    }
    
    
    
    
    // 6. 资源清理
    VecDestroy(&u);
    VecDestroy(&u_old);
    VecDestroy(&rhs);
    MatDestroy(&A);
      if (K) MatDestroy(&K);
    if (ksp) KSPDestroy(&ksp);
    PetscFinalize();
    return 0;
}
