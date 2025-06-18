#include <petscksp.h>
#include <petscviewerhdf5.h> 
#include <math.h>
#include <string.h> 
#include <assert.h>  
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


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
	"  -kappa <value>        : Thermal conductivity (default: 1.0)\n"
	"  -rho_c <value>        : Density * specific heat (default: 1.0)\n"
	"   Verification & I/O Options:\n"
	"  -mms                    : Use Method of Manufactured Solutions for verification (default: PETSC_FALSE)\n"
	"  -enable_restart         : Enable saving restart files (default: PETSC_FALSE)\n"
	"  -restart_interval <int> : Iteration interval to save restart file (default: 10)\n"
	"  -restart_load <file>    : Load solution from a restart file to resume simulation\n"
	"  -vtk_output             : Enable VTK output for visualization (default: PETSC_FALSE)\n"
	"  -vtk_interval <int>     : Iteration interval to save VTK file (default: 10, same as restart_interval)\n"
	"  -view_solution          : View final solution vector in terminal\n\n"
	"Performance Analysis:\n"
	"  Run with -log_view to see detailed performance metrics, including registered events for assembly and solve times.\n";
	PetscFunctionBeginUser;
	PetscCall(PetscInitialize(&argc, &argv, NULL, help));

	PetscMPIInt rank;
	PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
	// 添加了PETSc事件日志用于精细计时
	PetscLogEvent ASSEMBLY_TIME, SOLVE_TIME;
	PetscCall(PetscLogEventRegister("Matrix Assembly", MAT_CLASSID, &ASSEMBLY_TIME));
	PetscCall(PetscLogEventRegister("Linear Solve",    KSP_CLASSID,  &SOLVE_TIME));
	// 参数定义
	PetscInt    N = 100;          // 矩阵默认大小
	PetscBool   view_solution = PETSC_FALSE; // 是否查看最终解
	PetscBool   use_mms = PETSC_FALSE; // 是否使用mms
	PetscReal   dt = 0.001;        // 时间步长
	PetscInt    max_steps = 1000;  // 最大时间步数
	PetscReal   kappa = 1.0;       // 热传导系数
	PetscReal   rho_c = 1.0;       // ρc 乘积 
	char time_method[20] = "implicit"; // 默认时间方法
	// 重启与可视化参数
	PetscBool   enable_restart = PETSC_FALSE;
	PetscInt    io_interval = 10;
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
	PetscCall(PetscOptionsGetInt(NULL, NULL, "-restart_interval", &io_interval, NULL));
	PetscCall(PetscOptionsGetString(NULL, NULL, "-restart_load", restart_load_file, sizeof(restart_load_file), &restart_load_flag));
	PetscCall(PetscOptionsGetBool(NULL, NULL, "-vtk_output", &vtk_output, NULL));
	PetscCall(PetscOptionsGetInt(NULL, NULL, "-vtk_interval", &io_interval, NULL));     
	// 参数验证
	assert(kappa > 0.0 && "Heat conductivity must be positive");
	assert(rho_c > 0.0 && "Density * specific heat must be positive");
	assert(N > 1 && "Mesh size must be greater than 1");
	assert(dt > 0.0 && "Time step must be positive");
	assert(io_interval > 0 && "I/O interval must be positive");
	
	PetscInt total_nodes = N * N;
	PetscReal h = 1.0 / (PetscReal)(N - 1);

	PetscInt    start_step = 0;
	PetscReal   time = 0.0;
	// 创建解向量
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
	PetscCall(PetscViewerHDF5PushGroup(viewer, "/")); // HDF5 attributes are on the root group by default
	PetscInt  loaded_N;
	PetscCall(PetscViewerHDF5ReadAttribute(viewer, "N", "N", PETSC_INT, NULL, &loaded_N));
	PetscCall(PetscViewerHDF5ReadAttribute(viewer, "dt", "dt", PETSC_REAL, NULL, &dt));
	PetscCall(PetscViewerHDF5ReadAttribute(viewer, "kappa", "kappa", PETSC_REAL, NULL, &kappa));
	PetscCall(PetscViewerHDF5ReadAttribute(viewer, "rho_c", "rho_c", PETSC_REAL, NULL, &rho_c));
	PetscCall(PetscViewerHDF5ReadAttribute(viewer, "time", "time", PETSC_REAL, NULL, &time));
	PetscCall(PetscViewerHDF5ReadAttribute(viewer, "iteration", "iteration", PETSC_INT, NULL, &start_step));
	PetscCall(PetscViewerHDF5PopGroup(viewer));
        
	PetscCall(PetscViewerDestroy(&viewer));

	// 检查加载的网格大小是否与当前设置匹配
	if (loaded_N != N) {
	SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Restart file was saved with N=%d, but current setting is N=%d. Sizes must match.", loaded_N, N);
	}
	start_step++; // Resume from the next step
	if (rank == 0) {
	PetscPrintf(PETSC_COMM_WORLD, "Restarting from file %s. Resuming at step %d, time %.4f\n", restart_load_file, start_step, (double)time);
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
	PetscCall(MatSeqAIJSetPreallocation(A, max_nnz, NULL));
	// 获取进程的矩阵部分
	PetscCall(PetscLogEventBegin(ASSEMBLY_TIME, 0, 0, 0, 0));
	PetscInt Istart, Iend;
	PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));
	PetscBool is_implicit = (strcmp(time_method, "implicit") == 0);
	
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
	// 对角线
	if (is_implicit) {
                // IMPLICIT: A = (ρc*I - Δt*κ*∇²)
                // Diagonal term: ρc + 4*κ*Δt/h²
                // Off-diagonal term: -κ*Δt/h²
                vals[0] = rho_c + 4.0 * kappa * dt / (h * h);
                vals[1] = -kappa * dt / (h * h);
                vals[2] = -kappa * dt / (h * h);
                vals[3] = -kappa * dt / (h * h);
                vals[4] = -kappa * dt / (h * h);
            } else {
                // EXPLICIT: A = κ*∇²
                // Diagonal term: -4*κ/h²
                // Off-diagonal term: κ/h²
                vals[0] = -4.0 * kappa / (h * h);
                vals[1] = kappa / (h * h);
                vals[2] = kappa / (h * h);
                vals[3] = kappa / (h * h);
                vals[4] = kappa / (h * h);
            }
            cols[0] = idx;
            cols[1] = idx - N; // (i-1, j)
            cols[2] = idx + N; // (i+1, j)
            cols[3] = idx - 1; // (i, j-1)
            cols[4] = idx + 1; // (i, j+1)
            PetscCall(MatSetValues(A, 1, &idx, 5, cols, vals, INSERT_VALUES));
        }
    }
	PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
	PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
	PetscCall(PetscLogEventEnd(ASSEMBLY_TIME, 0, 0, 0, 0));
	// 2. 设置初始条件向量   
	// 设置初始温度分布 
	Vec u_old, rhs, f_vec;
	PetscCall(VecDuplicate(u, &u_old));
	PetscCall(VecDuplicate(u, &rhs));
	PetscCall(VecDuplicate(u, &f_vec));
	PetscCall(PetscObjectSetName((PetscObject)f_vec, "forcing_term"));
	// 重启文件加载
	// 设置初始条件 (仅在不从重启文件加载时)
	if (!restart_load_flag) {
	PetscScalar *u_arr;
	PetscCall(VecGetArray(u, &u_arr));
	PetscCall(VecGetOwnershipRange(u, &Istart, &Iend)); // Istart/Iend need to be for the vector
	for (PetscInt idx = Istart; idx < Iend; idx++) {
	PetscInt i = idx / N;
	PetscInt j = idx % N;
	if (use_mms) {
	u_arr[idx - Istart] = manufactured_solution((PetscReal)j * h, (PetscReal)i * h, 0.0);
	} else {
	// 默认初始条件：狄利克雷边界为0，内部为0，除了中心热点
	if (i == 0 || i == N - 1 || j == 0 || j == N - 1) {
	u_arr[idx-Istart] = 0.0;
		} else if (i == N / 2 && j == N / 2) {
			u_arr[idx-Istart] = 1.0;
			} else {
				u_arr[idx-Istart] = 0.0;
			}
		}
	}
	PetscCall(VecRestoreArray(u, &u_arr));
	}
    
	// 3. 创建KSP求解器上下文 (仅用于隐式方法)
	KSP ksp = NULL;
	if (is_implicit) {
	PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
	PetscCall(KSPSetOperators(ksp, A, A));
	PetscCall(KSPSetFromOptions(ksp));
	}
	// 4.时间步循环
	for (PetscInt step = start_step; step < max_steps; step++) {
	// 更新当前时间
	time += dt;
	// 保存旧解
	PetscCall(VecCopy(u, u_old));
	// 计算源项 f(t_n+1)
	PetscScalar *f_arr;
	PetscCall(VecGetArray(f_vec, &f_arr));
	PetscCall(VecGetOwnershipRange(f_vec, &Istart, &Iend));
	for (PetscInt idx = Istart; idx < Iend; idx++) {
	PetscInt i = idx / N;
	PetscInt j = idx % N;
	if (i == 0 || i == N - 1 || j == 0 || j == N - 1) {
	f_arr[idx-Istart] = 0.0; // 边界上的源项设为0，实际值将在RHS中强制设定
	} else if (use_mms) {
	// For implicit: f at t_n+1; for explicit: f at t_n
	// We use t (which is t_n+1) for simplicity in both cases.
	f_arr[idx-Istart] = manufactured_source((PetscReal)j * h, (PetscReal)i * h, time, kappa, rho_c);
	} else {
	f_arr[idx-Istart] = 0.0; // 默认无源项
		}
	}
	PetscCall(VecRestoreArray(f_vec, &f_arr));
	
	// 时间步进逻辑
	if (is_implicit) {
	// 隐式: (ρc*I - Δt*κ*∇²) u_new = ρc*u_old + Δt*f_new
	// rhs = ρc * u_old + dt * f
	PetscCall(VecCopy(u_old, rhs));
	PetscCall(VecScale(rhs, rho_c));
	PetscCall(VecAXPY(rhs, dt, f_vec));
	
	// 强制执行狄利克雷边界条件
	PetscScalar *rhs_arr;
	PetscCall(VecGetArray(rhs, &rhs_arr));
	PetscCall(VecGetOwnershipRange(rhs, &Istart, &Iend));
	for(PetscInt idx = Istart; idx < Iend; idx++) {
	PetscInt i = idx / N;
	PetscInt j = idx % N;
		if(i == 0 || i == N-1 || j == 0 || j == N-1){
			if (use_mms) {
			rhs_arr[idx-Istart] = manufactured_solution((PetscReal)j*h, (PetscReal)i*h, time);
	} else {
		rhs_arr[idx-Istart] = 0.0;
			}
		}
	}
	PetscCall(VecRestoreArray(rhs, &rhs_arr));
	// 求解线性系统 Au = rhs
	PetscCall(PetscLogEventBegin(SOLVE_TIME, 0, 0, 0, 0));
	PetscCall(KSPSolve(ksp, rhs, u));
	PetscCall(PetscLogEventEnd(SOLVE_TIME, 0, 0, 0, 0));

	} else { // Explicit
	// 显式: u_new = u_old + (Δt/ρc) * (A*u_old + f_old)  (其中 A=κ*∇²)
	// rhs = A * u_old
	PetscCall(MatMult(A, u_old, rhs));
	// rhs = A * u_old + f
	PetscCall(VecAXPY(rhs, 1.0, f_vec));
	// u_new = u_old + (dt/rho_c) * rhs
	PetscCall(VecAXPY(u, dt / rho_c, rhs));
	
	// 强制执行狄利克雷边界条件 (在更新之后直接设定)
	PetscScalar *u_arr;
	PetscCall(VecGetArray(u, &u_arr));
	PetscCall(VecGetOwnershipRange(u, &Istart, &Iend));
	for(PetscInt idx = Istart; idx < Iend; idx++) {
	PetscInt i = idx / N;
	PetscInt j = idx % N;
	if(i == 0 || i == N-1 || j == 0 || j == N-1){
	if (use_mms) {
	u_arr[idx-Istart] = manufactured_solution((PetscReal)j*h, (PetscReal)i*h, time);
	} else {
	u_arr[idx-Istart] = 0.0;
			}
		}
	}
	PetscCall(VecRestoreArray(u, &u_arr));
	}

	if (step % 50 == 0) {
	PetscReal norm_u;
	// 集体操作：所有进程都必须执行
	PetscCall(VecNorm(u, NORM_2, &norm_u)); 

	// 单进程操作：只有0号进程执行，以避免重复打印
	if (rank == 0) { 
	PetscPrintf(PETSC_COMM_WORLD, "Step %4d: Time=%.4f, ||u||_2=%.2e\n", step, (double)time, (double)norm_u);
		}
	}
	//统一的I/O逻辑
	if ( (enable_restart || vtk_output) && (step > 0 && step % io_interval == 0) ) {
	// 保存VTK
	if (vtk_output) {
	char filename[PETSC_MAX_PATH_LEN];
	sprintf(filename, "solution_step_%04d.vts", step);
	PetscViewer viewer;
	PetscCall(PetscViewerVTKOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer));
	PetscCall(VecView(u, viewer));
	PetscCall(PetscViewerDestroy(&viewer));
	}
	// 定期保存重启文件
	if (enable_restart) {
	char filename[PETSC_MAX_PATH_LEN];
	sprintf(filename, "restart_step_%04d.h5", step);
	PetscViewer viewer;
	PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer));
	PetscCall(VecView(u, viewer));
	PetscCall(PetscViewerHDF5PushGroup(viewer, "/"));
	PetscCall(PetscViewerHDF5WriteAttribute(viewer, "N", "N", PETSC_INT, &N));
	PetscCall(PetscViewerHDF5WriteAttribute(viewer, "dt", "dt", PETSC_REAL, &dt));
	PetscCall(PetscViewerHDF5WriteAttribute(viewer, "kappa", "kappa", PETSC_REAL, &kappa));
	PetscCall(PetscViewerHDF5WriteAttribute(viewer, "rho_c", "rho_c", PETSC_REAL, &rho_c));
	PetscCall(PetscViewerHDF5WriteAttribute(viewer, "time", "time", PETSC_REAL, &time));
	PetscCall(PetscViewerHDF5WriteAttribute(viewer, "iteration", "iteration", PETSC_INT, &step));
	PetscCall(PetscViewerHDF5PopGroup(viewer));
	// 保存解向量
	PetscCall(PetscViewerDestroy(&viewer));
	if (rank == 0) {
	PetscPrintf(PETSC_COMM_WORLD, "Saved restart file: %s\n", filename);
                }
            }
        }
    }
	
	// --- MODIFICATION ---: 改进MMS误差计算说明
	if (use_mms) {
	Vec u_exact_vec;
	PetscCall(VecDuplicate(u, &u_exact_vec));
	PetscScalar *exact_arr;
	PetscCall(VecGetArray(u_exact_vec, &exact_arr));
	PetscCall(VecGetOwnershipRange(u_exact_vec, &Istart, &Iend));
	for (PetscInt idx = Istart; idx < Iend; idx++) {
	PetscInt i = idx / N;
	PetscInt j = idx % N;
	exact_arr[idx - Istart] = manufactured_solution((PetscReal)j * h, (PetscReal)i * h, time);
	}
	PetscCall(VecRestoreArray(u_exact_vec, &exact_arr));
	PetscReal err_norm;
	PetscCall(VecAXPY(u, -1.0, u_exact_vec)); // u = u_num - u_exact
	PetscCall(VecNorm(u, NORM_INFINITY, &err_norm));
        
	// 5.结果输出 
	if (rank == 0) {
	PetscPrintf(PETSC_COMM_WORLD, "\n===== MMS Verification =====\n");
	PetscPrintf(PETSC_COMM_WORLD, "L-infinity error ||u_num - u_exact||_∞ at T=%.4f is: %.4e\n", (double)time, (double)err_norm);
	PetscPrintf(PETSC_COMM_WORLD, "To determine convergence orders α and β (e ≈ C₁Δx^α + C₂Δt^β),\n");
	PetscPrintf(PETSC_COMM_WORLD, "you need to run this code multiple times with different -n and -dt values and analyze the resulting errors.\n");
        }
        PetscCall(VecDestroy(&u_exact_vec));
        }
	// 查看最终解
	if (view_solution) {
	PetscPrintf(PETSC_COMM_WORLD, "===== Final Solution =====\n");
	PetscCall(VecView(u, PETSC_VIEWER_STDOUT_WORLD));
	}
	// 清理
	PetscCall(VecDestroy(&u));
	PetscCall(VecDestroy(&u_old));
	PetscCall(VecDestroy(&rhs));
	PetscCall(VecDestroy(&f_vec));
	PetscCall(MatDestroy(&A));
	if (ksp) PetscCall(KSPDestroy(&ksp));
	PetscCall(PetscFinalize());
	return 0;
	}

	// MMS相关函数实现 (保持不变)
	PetscScalar manufactured_solution(PetscReal x, PetscReal y, PetscReal t) {
	return sin(M_PI * x) * sin(M_PI * y) * exp(-t);
	}

	PetscScalar manufactured_source(PetscReal x, PetscReal y, PetscReal t, PetscReal kappa, PetscReal rho_c) {
	// f = ρc * ∂u/∂t - κ * (∂²u/∂x² + ∂²u/∂y²)
	PetscScalar u_t = -sin(M_PI * x) * sin(M_PI * y) * exp(-t);
	PetscScalar laplacian_u = -2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y) * exp(-t);
	return rho_c * u_t - kappa * laplacian_u;
	}


	
