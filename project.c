#include <petscksp.h>
#include <petscviewerhdf5.h> 
#include <math.h>
#include <string.h> 
#include <assert.h>  
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//添加重启数据结构
typedef struct {
    PetscInt iteration;
    PetscReal time;
} RestartData;
//为MMS添加函数原型
PetscScalar manufactured_solution(PetscReal x, PetscReal y, PetscReal t);
PetscScalar manufactured_source(PetscReal x, PetscReal y, PetscReal t, PetscReal kappa, PetscReal rho_c);

int main(int argc, char **argv) {
	const char *help = 
	"This project solve a transient heat equation in a two-dimensional unit square\n"
	"ρc∂u/∂t − κ∂2u/∂x2 = f on Ω × (0, T )\n"
	"u = g on Γg × (0, T )\n"
	"κ∂u/∂xnx = h on Γh × (0, T )\n"
	"u|t=0 = u0 in Ω.\n"
	"  - Finite difference spatial discretization\n"
	"Options:\n"
	"  -n <size>             : Mesh size (default: 100)\n"
	"  -dt <timestep>        : Time step size (default: 0.001)\n"
	"  -max_steps <int>      : Maximum time steps (default: 1000)\n"
	"  -time_method <str>    : 'implicit' or 'explicit' (default: implicit)\n"
	"  -view_solution        : View final solution\n"
	"  -mms		         : Use Method of Manufactured Solutions for verification (default: PETSC_FALSE)\n"
	"   KSP/PC options       : Any standard PETSc options for solvers/preconditioners\n";
	"  -kappa <value>        : Thermal conductivity (default: 1.0)\n"
	"  -rho_c <value>        : Density * specific heat (default: 1.0)\n"
	"  -enable_restart <int> : Enable HDF5 restart functionality (default: PETSC_FALSE)\n"
	"  -restart_interval     : Restart file save interval (default: 10)\n"
	"  -restart_file         : Filename for restart data (default: \"restart.h5\")\n"; 
	"  -vtk_output           : Enable VTK output for visualization (default: PETSC_FALSE)\n"
	PetscFunctionBeginUser;
	PetscCall(PetscInitialize(&argc, &argv, NULL, help));

	PetscMPIInt rank;
	PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
	// 添加了PETSc事件日志用于精细计时
	PetscLogEvent ASSEMBLY_TIME, SOLVE_TIME;
	PetscCall(PetscLogEventRegister("Matrix Assembly", MAT_CLASSID, &ASSEMBLY_TIME));
	PetscCall(PetscLogEventRegister("Linear Solve",    KSP_CLASSID,  &SOLVE_TIME));
	// 参数定义
	PetscInt    N = 10000;          // 矩阵默认大小
	PetscBool   view_solution = PETSC_FALSE; // 是否查看最终解
	PetscBool   use_mms = PETSC_FALSE; // 是否使用mms
	PetscReal   dt = 0.001;        // 时间步长
	PetscInt    max_steps = 1000;  // 最大时间步数
	PetscReal   kappa = 1.0;       // 热传导系数
	PetscReal   rho_c = 1.0;       // ρc 乘积 
	char time_method[20] = "implicit"; // 默认时间方法
	// 重启与可视化参数
	PetscBool   enable_restart = PETSC_FALSE;
	PetscInt    restart_interval = 10;
	char        restart_load_file[PETSC_MAX_PATH_LEN] = "";
	PetscBool   restart_load_flag = PETSC_FALSE;
	PetscBool   vtk_output     = PETSC_FALSE;

	// 从命令行获取参数
	PetscOptionsGetInt(NULL, NULL, "-n", &N, NULL);
	PetscOptionsGetReal(NULL, NULL, "-dt", &dt, NULL);
	PetscOptionsGetInt(NULL, NULL, "-max_steps", &max_steps, NULL);
	PetscOptionsGetBool(NULL, NULL, "-view_solution", &view_solution, NULL);
	PetscCall(PetscOptionsGetString(NULL, NULL, "-time_method", time_method, sizeof(time_method), NULL));
	PetscCall(PetscOptionsGetReal(NULL, NULL, "-kappa", &kappa, NULL));
	PetscCall(PetscOptionsGetReal(NULL, NULL, "-rho_c", &rho_c, NULL));
	PetscCall(PetscOptionsGetBool(NULL, NULL, "-mms", &use_mms, NULL));
	PetscCall(PetscOptionsGetBool(NULL, NULL, "-enable_restart", &enable_restart, NULL));
	PetscCall(PetscOptionsGetInt(NULL, NULL, "-restart_interval", &restart_interval, NULL));
	PetscCall(PetscOptionsGetString(NULL, NULL, "-restart_load", restart_load_file, sizeof(restart_load_file), &restart_load_flag));
	PetscCall(PetscOptionsGetBool(NULL, NULL, "-vtk_output", &vtk_output, NULL));
	// 参数验证
	assert(kappa > 0.0 && "Heat conductivity must be positive");
	assert(rho_c > 0.0 && "Density * specific heat must be positive");
	assert(N > 1 && "Mesh size must be greater than 1");
	assert(dt > 0.0 && "Time step must be positive");
    
	PetscInt total_nodes = N * N;
	PetscReal h = 1.0 / (N - 1);
	// 初始化 RestartData
	RestartData restart_data = {0, 0.0};
	PetscInt    start_step = 0;
	PetscReal   time = 0.0;
	// 重启加载逻辑
	Vec u;
	PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
	PetscCall(VecSetSizes(u, PETSC_DECIDE, total_nodes));
	PetscCall(VecSetFromOptions(u));
	PetscCall(PetscObjectSetName((PetscObject)u, "temperature"));

	if (restart_load_flag) {
	PetscViewer viewer;
	PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, restart_load_file, FILE_MODE_READ, &viewer));
	PetscCall(VecLoad(u, viewer));
	// 加载元数据
	PetscCall(PetscViewerHDF5PushGroup(viewer, "/restart_data"));
	PetscCall(PetscViewerHDF5ReadAttribute(viewer, "time", "time", PETSC_REAL, &restart_data.time));
	PetscCall(PetscViewerHDF5ReadAttribute(viewer, "iteration", "iteration", PETSC_INT, &restart_data.iteration));
	PetscCall(PetscViewerHDF5PopGroup(viewer));
	PetscCall(PetscViewerDestroy(&viewer));
        
	start_step = restart_data.iteration + 1;
	time = restart_data.time;
	
	if (rank == 0) {
	PetscPrintf(PETSC_COMM_WORLD, "Restarting from file %s at step %d, time %.4f\n", restart_load_file, start_step, (double)time);
		}
	}
	
	if (rank == 0) {
	PetscPrintf(PETSC_COMM_WORLD, "=== 2D Transient Heat Equation Solver ===\n");
	PetscPrintf(PETSC_COMM_WORLD, "Grid: %dx%d (h=%.4f), Time Method: %s, dt: %.4f, Max Steps: %d\n", N, N, (double)h, time_method, (double)dt, max_steps);
	PetscPrintf(PETSC_COMM_WORLD, "Params: kappa=%.2f, rho_c=%.2f, MMS: %s\n", (double)kappa, (double)rho_c, use_mms ? "enabled" : "disabled");
	}
	// 稳定性分析
	if (strcmp(time_method, "explicit") == 0) {
	PetscReal cfl = kappa * dt / (rho_c * h * h);
		if (rank == 0) {
	PetscPrintf(PETSC_COMM_WORLD, "Stability Analysis (Explicit): CFL number α = %.4f\n", (double)cfl);
			if (cfl > 0.25) {
	PetscPrintf(PETSC_COMM_WORLD, "WARNING: Stability condition α <= 0.25 violated! Simulation may be unstable.\n");
			}
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
	
	// 获取进程的矩阵部分
	PetscCall(PetscLogEventBegin(ASSEMBLY_TIME, 0, 0, 0, 0));
	PetscInt Istart, Iend;
	PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));
	
	for (PetscInt idx = Istart; idx < Iend; idx++) {
	PetscInt i = idx / N;
	PetscInt j = idx % N;

	// 边界条件判断
	if (i == 0 || i == N - 1 || j == 0 || j == N - 1) {
	// 狄利克雷边界点: u = g，这里设g=0
	PetscScalar v = 1.0;
	PetscCall(MatSetValues(A, 1, &idx, 1, &idx, &v, INSERT_VALUES));
	} else {
	// 内部点
	PetscInt    cols[5];
	PetscScalar vals[5];
	PetscInt    ncols = 0;
	// 对角线
	cols[ncols] = idx;
	vals[ncols] = (strcmp(time_method, "implicit") == 0) ? (rho_c + 4.0 * kappa * dt / (h * h)) : rho_c;
	ncols++;
            
	PetscScalar off_diag = (strcmp(time_method, "implicit") == 0) ? (-kappa * dt / (h * h)) : (kappa * dt / (h * h));

	// 四个邻居点
	cols[ncols] = idx - N; vals[ncols] = off_diag; ncols++; // (i-1, j)
	cols[ncols] = idx + N; vals[ncols] = off_diag; ncols++; // (i+1, j)
	cols[ncols] = idx - 1; vals[ncols] = off_diag; ncols++; // (i, j-1)
	cols[ncols] = idx + 1; vals[ncols] = off_diag; ncols++; // (i, j+1)
            
	PetscCall(MatSetValues(A, 1, &idx, ncols, cols, vals, INSERT_VALUES));
		}
	}
	PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
	PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
	PetscCall(PetscLogEventEnd(ASSEMBLY_TIME, 0, 0, 0, 0));
	// 2. 设置初始条件向量   
	// 设置初始温度分布 
	Vec u, u_old, rhs;// 创建临时向量存储旧解和右端项
	PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
	PetscCall(VecSetSizes(u, PETSC_DECIDE, total_nodes));
	PetscCall(VecSetFromOptions(u));
	// 重启文件加载
	if (enable_restart && restart_data.iteration > 0) {
	PetscViewer viewer;
	PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, restart_file, FILE_MODE_READ, &viewer));
	PetscCall(VecLoad(u, viewer));
	PetscCall(PetscViewerDestroy(&viewer));
        
	if (rank == 0) {
	PetscPrintf(PETSC_COMM_WORLD, "Restarting from iteration %d, time %.4f\n", 
	restart_data.iteration, restart_data.time);
		}
	}
	else {
	PetscCall(VecSet(u, 0.0));
	//设置热源（中心点）
	PetscInt center_idx = (N/2) * N + N/2;
	PetscScalar src_val = 1.0;
	PetscCall(VecSetValues(u, 1, &center_idx, &src_val, INSERT_VALUES));
	}
	PetscCall(VecAssemblyBegin(u));
	PetscCall(VecAssemblyEnd(u));
 
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
	PetscInt step = restart_data.iteration;
	PetscReal time = restart_data.time;
	PetscReal   norm_diff = 0.0;
	for (step = 0; step < max_steps; step++) {
	// 更新当前时间
	time += dt;
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
	PetscPrintf(PETSC_COMM_WORLD, "Step %4d: Time=%.4f, ||Δu||=%.2e\n", step, (double)time, (double)norm_diff);
	}
	// 定期保存重启文件
	if (enable_restart && (step % restart_interval == 0)) {
	PetscViewer viewer;
	PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, restart_file, FILE_MODE_WRITE, &viewer));
	// 保存元数据
	restart_data.iteration = step;
	restart_data.time = time;
	restart_data.dt = dt;
	restart_data.h = h;
	restart_data.N = N;
	restart_data.kappa = kappa;
	restart_data.rho_c = rho_c;
	
	PetscCall(PetscObjectSetName((PetscObject)&restart_data, "restart_data"));
	PetscCall(PetscViewerHDF5WriteObject(viewer, (PetscObject)&restart_data));   
	// 保存解向量
	PetscCall(VecView(u, viewer));
	PetscCall(PetscViewerDestroy(&viewer));
	if (rank == 0) {
	PetscPrintf(PETSC_COMM_WORLD, "Saved restart file at step %d\n", step);
		}
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
    
	// 防御性资源清理
	if (u) PetscCall(VecDestroy(&u));
	if (u_old) PetscCall(VecDestroy(&u_old));
	if (rhs) PetscCall(VecDestroy(&rhs));
	if (A) PetscCall(MatDestroy(&A));
	if (K) PetscCall(MatDestroy(&K));
	if (ksp) PetscCall(KSPDestroy(&ksp));
	PetscFinalize();
	return 0;
	}
