#include <petscksp.h>      // KSP/PC：用于线性方程组求解器和预处理器
#include <petscdm.h>         // DM：用于数据管理(Domain Management)的通用接口
#include <petscdmda.h>       // DMDA：用于管理结构化网格(如本项目中的二维矩形网格)的特定DM
#include <petscviewerhdf5.h> // Viewer：用于读写HDF5文件（重启功能需要）
// 标准C库
#include <math.h>          // M_PI, sin(), exp() 等数学函数
#include <string.h>        // strcmp() 用于比较字符串（判断时间方法）
#include <assert.h>        // assert() 用于进行防御性编程，检查关键参数
/*******************************************************************************
 * 宏定义与函数原型
 *******************************************************************************/
// 为不提供M_PI宏的编译器定义圆周率PI，增加代码可移植性
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * 为“制造解方法”(Method of Manufactured Solutions, MMS)声明函数原型。
 * MMS是一种严谨的代码验证技术，通过一个已知的精确解来反推源项和边界条件，
 * 从而可以精确地计算出数值解的误差。
 */
// 精确解函数 u(x, y, t)
	PetscScalar manufactured_solution(PetscReal x, PetscReal y, PetscReal t);
// 根据精确解反推出来的源项 f(x, y, t)
	PetscScalar manufactured_source(PetscReal x, PetscReal y, PetscReal t, PetscReal kappa, PetscReal rho_c);


/*******************************************************************************
 * 主函数 main
 * C程序的入口点。
 *******************************************************************************/
	int main(int argc, char **argv) {
	/*
	* help字符串：
	* 定义了程序的帮助信息。当用户使用 -h 或 --help 运行程序时，
	* PETSc会自动打印这段信息，非常便于用户了解程序功能和可用选项。
	*/
	const char *help =
	"使用DMDA在二维单位正方形上求解瞬态热传导方程。\n"
	"方程: ρc*∂u/∂t - κ*∇²u = f\n\n"
	"选项:\n"
	"  -n <size>             : 网格大小 N x N (默认: 100)\n"
	"  -dt <timestep>          : 时间步长 (默认: 0.001)\n"
	"  -max_steps <int>        : 最大时间步数 (默认: 1000)\n"
	"  -time_method <str>      : 时间推进方法 'implicit' 或 'explicit' (默认: implicit)\n"
	"  -kappa <value>          : 热导率 κ (默认: 1.0)\n"
	"  -rho_c <value>          : 密度×比热 ρc (默认: 1.0)\n\n"
	"验证与输入/输出选项:\n"
	"  -mms                    : 使用制造解方法进行验证 (默认: PETSC_FALSE)\n"
	"  -enable_restart         : 启用HDF5重启文件保存 (默认: PETSC_FALSE)\n"
	"  -restart_load <file>    : 从HDF5重启文件加载并继续计算\n"
	"  -vtk_output             : 启用VTK文件输出以进行可视化 (默认: PETSC_FALSE)\n"
	"  -io_interval <int>      : 保存重启/VTK文件的迭代间隔 (默认: 10)\n"
	"  -view_solution          : 在终端打印最终解向量\n\n";

	/*
	* PETSc 程序初始化
	* 这是所有PETSc程序的强制性起始步骤。
	*/
	PetscFunctionBeginUser; // PETSc函数追踪的开始
	PetscCall(PetscInitialize(&argc, &argv, NULL, help)); // 初始化PETSc环境，MPI，并处理命令行参数

	/*
	* 并行环境设置
	* 获取当前进程在所有进程中的排名（rank）。
	* 在并行程序中，通常只有rank=0的进程负责打印信息到屏幕。
	*/
	PetscMPIInt rank;
	PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    
	/*
	* 性能日志事件注册
	* 注册自定义的计时器，用于精确测量代码中特定部分（如矩阵组装、线性求解）的耗时。
	* 运行程序时加上 -log_view 选项即可查看性能报告。
 	*/
	PetscLogEvent ASSEMBLY_TIME, SOLVE_TIME;
	PetscCall(PetscLogEventRegister("Matrix Assembly", MAT_CLASSID, &ASSEMBLY_TIME));
	PetscCall(PetscLogEventRegister("Linear Solve",    KSP_CLASSID,  &SOLVE_TIME));

	/*
	* 参数变量声明与初始化
	* 定义程序中使用的所有参数，并赋予它们默认值。
	*/
	// 数值与物理参数
	PetscInt    N                = 100;
	PetscReal   dt               = 0.001;
	PetscInt    max_steps        = 1000;
	PetscReal   kappa            = 1.0;
	PetscReal   rho_c            = 1.0;
	char        time_method[20]  = "implicit";
	// 控制与I/O参数
	PetscBool   view_solution    = PETSC_FALSE;
	PetscBool   use_mms          = PETSC_FALSE;
	PetscBool   enable_restart   = PETSC_FALSE;
	PetscInt    io_interval      = 10;
	char        restart_load_file[PETSC_MAX_PATH_LEN] = "";
	PetscBool   restart_load_flag = PETSC_FALSE;
	PetscBool   vtk_output       = PETSC_FALSE;

	/*
	* 从命令行解析参数
	* 使用PETSc提供的函数来读取用户在命令行中指定的参数值，覆盖默认值。
	*/
	PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &N, NULL));
	PetscCall(PetscOptionsGetReal(NULL, NULL, "-dt", &dt, NULL));
	PetscCall(PetscOptionsGetInt(NULL, NULL, "-max_steps", &max_steps, NULL));
	PetscCall(PetscOptionsGetString(NULL, NULL, "-time_method", time_method, sizeof(time_method), NULL));
	PetscCall(PetscOptionsGetReal(NULL, NULL, "-kappa", &kappa, NULL));
	PetscCall(PetscOptionsGetReal(NULL, NULL, "-rho_c", &rho_c, NULL));
	PetscCall(PetscOptionsGetBool(NULL, NULL, "-mms", &use_mms, NULL));
	PetscCall(PetscOptionsGetBool(NULL, NULL, "-enable_restart", &enable_restart, NULL));
	PetscCall(PetscOptionsGetInt(NULL, NULL, "-io_interval", &io_interval, NULL));
 	PetscCall(PetscOptionsGetString(NULL, NULL, "-restart_load", restart_load_file, sizeof(restart_load_file), &restart_load_flag));
	PetscCall(PetscOptionsGetBool(NULL, NULL, "-vtk_output", &vtk_output, NULL));
	PetscCall(PetscOptionsGetBool(NULL, NULL, "-view_solution", &view_solution, NULL));
	/*
	* 参数验证 (防御性编程)
	*/
	assert(kappa > 0.0 && "Heat conductivity must be positive");
	assert(rho_c > 0.0 && "Density * specific heat must be positive");
	assert(N > 1 && "Mesh size must be greater than 1");
	assert(dt > 0.0 && "Time step must be positive");
	assert(io_interval > 0 && "I/O interval must be positive");
	/*
	* 创建 DMDA (Distributed Mesh Data Management) 对象
	* DMDA是PETSc中用于处理结构化网格的核心工具。它存储了所有关于网格的几何信息和
	* 并行划分信息，是进行高效并行计算和可视化的基础。
	*/
	DM da; // 声明一个DM对象
	// 创建一个2D的DMDA
	PetscCall(DMDACreate2d(PETSC_COMM_WORLD,       // MPI通信域
			DM_BOUNDARY_NONE,       // 边界类型(x方向)，这里不通过DM管理边界
			DM_BOUNDARY_NONE,       // 边界类型(y方向)
			DMDA_STENCIL_STAR,      // 模板类型(五点星型模板)
			N, N,                   // 全局网格维度
			PETSC_DECIDE, PETSC_DECIDE, // PETSc自动决定进程如何划分网格
			1,                      // 每个节点的自由度(这里是温度，所以是1)
			1,                      // 模板宽度(五点模板宽度为1)
			NULL, NULL, &da));
	PetscCall(DMSetUp(da)); // 完成DMDA的设置
	// 为DMDA设置坐标信息，这样VTK文件才能知道每个点的物理坐标(x,y)
	PetscCall(DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0));

	/*
	* 基于DMDA创建向量和矩阵
	* 这样做可以确保向量和矩阵的并行布局与DMDA的网格划分完全一致。
	*/
	Vec u; // 声明解向量 u (温度)
	PetscCall(DMCreateGlobalVector(da, &u));
	PetscCall(PetscObjectSetName((PetscObject)u, "temperature")); // 设置在VTK文件中显示的名字
    
	Mat A; // 声明矩阵 A
	PetscCall(DMCreateMatrix(da, &A));

	/*
	* 初始化模拟状态变量
	*/
	PetscInt    start_step = 0;
	PetscReal   time = 0.0;
	PetscReal   h = 1.0 / (PetscReal)(N - 1); // 计算空间步长
    
	// 如果指定了重启文件，则从文件中加载数据 (这部分代码可以无缝对接DMDA)
	if (restart_load_flag) {
	PetscViewer viewer;
	PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, restart_load_file, FILE_MODE_READ, &viewer));
	PetscCall(VecLoad(u, viewer)); // 加载解向量，DMDA确保维度匹配
        
	// 从HDF5文件的属性中加载所有必要的元数据
	PetscCall(PetscViewerHDF5PushGroup(viewer, "/"));
	PetscInt  loaded_N;
	PetscCall(PetscViewerHDF5ReadAttribute(viewer, "N", "N", PETSC_INT, NULL, &loaded_N));
	PetscCall(PetscViewerHDF5ReadAttribute(viewer, "dt", "dt", PETSC_REAL, NULL, &dt));
	PetscCall(PetscViewerHDF5ReadAttribute(viewer, "kappa", "kappa", PETSC_REAL, NULL, &kappa));
	PetscCall(PetscViewerHDF5ReadAttribute(viewer, "rho_c", "rho_c", PETSC_REAL, NULL, &rho_c));
	PetscCall(PetscViewerHDF5ReadAttribute(viewer, "time", "time", PETSC_REAL, NULL, &time));
	PetscCall(PetscViewerHDF5ReadAttribute(viewer, "iteration", "iteration", PETSC_INT, NULL, &start_step));
	PetscCall(PetscViewerHDF5PopGroup(viewer));
        
	PetscCall(PetscViewerDestroy(&viewer));

	// 验证网格大小是否匹配，防止从错误的存档重启
	if (loaded_N != N) {
	SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Restart file was saved with N=%d, but current setting is N=%d. Sizes must match.", loaded_N, N);
	}
        
	start_step++; // 从加载的下一步开始计算
	if (rank == 0) {
	PetscPrintf(PETSC_COMM_WORLD, "成功从文件 %s 重启。将在第 %d 步 (time=%.4f) 继续计算。\n", restart_load_file, start_step, (double)time);
		}
	}

	// 打印模拟的基本信息
	if (rank == 0) {
	PetscPrintf(PETSC_COMM_WORLD, "=== 2D瞬态热传导方程求解器 (DMDA版本) ===\n");
	PetscPrintf(PETSC_COMM_WORLD, "网格: %dx%d (h=%.4f), 时间方法: %s, dt: %.4f, 最大步数: %d\n", N, N, (double)h, time_method, (double)dt, max_steps);
	PetscPrintf(PETSC_COMM_WORLD, "参数: kappa=%.2f, rho_c=%.2f, MMS: %s\n", (double)kappa, (double)rho_c, use_mms ? "enabled" : "disabled");
	}
	//显式格式的稳定性分析
	if (strcmp(time_method, "explicit") == 0) {
	PetscReal cfl = kappa * dt / (rho_c * h * h);
	if (rank == 0) {
	PetscPrintf(PETSC_COMM_WORLD, "稳定性分析 (显式): 扩散数 α = κ*Δt/(ρc*h²) = %.4f\n", (double)cfl);
	if (cfl > 0.25) { // 2D五点格式的稳定性条件
	PetscPrintf(PETSC_COMM_WORLD, "警告: 违反稳定性条件 α <= 0.25！模拟结果可能不稳定。\n");
			}
		}
	}
 	/*
	* 矩阵组装 (Matrix Assembly)
	* 根据选择的时间方法（隐式或显式），填充矩阵A的非零元素。
	*/
	PetscCall(PetscLogEventBegin(ASSEMBLY_TIME, 0, 0, 0, 0)); // 开始计时
	PetscBool is_implicit = (strcmp(time_method, "implicit") == 0);
	DMDALocalInfo info; // 获取本进程负责的局部网格信息
	PetscCall(DMDAGetLocalInfo(da, &info));

	// 遍历本进程负责的所有网格点 (i, j)
	for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
		for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
		MatStencil row_stencil = {0, j, i, 0}; // 定义当前点的二维模板索引
		// 情况1：边界点。使用狄利克雷边界条件，在矩阵中对应行为单位矩阵行。
		if (i == 0 || i == info.mx - 1 || j == 0 || j == info.my - 1) {
		PetscScalar v = 1.0;
		PetscCall(MatSetValuesStencil(A, 1, &row_stencil, 1, &row_stencil, &v, INSERT_VALUES));
		} else { // 情况2：内部点。根据五点差分格式填充。
		MatStencil  cols_stencil[5];
		PetscScalar vals[5];
		PetscInt    ncols = 5;
                
		// 相邻点：(i,j) (i-1,j) (i+1,j) (i,j-1) (i,j+1)
		cols_stencil[0].j = j;     cols_stencil[0].i = i;
		cols_stencil[1].j = j;     cols_stencil[1].i = i - 1;
		cols_stencil[2].j = j;     cols_stencil[2].i = i + 1;
		cols_stencil[3].j = j - 1; cols_stencil[3].i = i;
		cols_stencil[4].j = j + 1; cols_stencil[4].i = i;

		// 根据时间方法填充不同的矩阵值
		if(is_implicit) { // 隐式: A = (ρc*I - Δt*κ*∇²)
		vals[0] = rho_c + 4.0*kappa*dt/(h*h); // 对角元
		vals[1] = -kappa*dt/(h*h);
		vals[2] = -kappa*dt/(h*h);
		vals[3] = -kappa*dt/(h*h);
		vals[4] = -kappa*dt/(h*h);
		} else { // 显式: A = κ*∇²
		vals[0] = -4.0*kappa/(h*h); // 对角元
		vals[1] = kappa/(h*h);
		vals[2] = kappa/(h*h);
		vals[3] = kappa/(h*h);
		vals[4] = kappa/(h*h);
		}
		// 使用基于模板的接口填充矩阵
		PetscCall(MatSetValuesStencil(A, 1, &row_stencil, ncols, cols_stencil, vals, INSERT_VALUES));
			}
		}
	}
	// 所有进程完成局部填充后，进行全局组装
	PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
	PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
	PetscCall(PetscLogEventEnd(ASSEMBLY_TIME, 0, 0, 0, 0)); // 结束计时

	/*
	* 设置初始条件 u(t=0)
	*/
	if (!restart_load_flag) {
	PetscScalar **u_arr;
	// 使用DMDAVecGetArray可以方便地获得一个二维数组的指针
	PetscCall(DMDAVecGetArray(da, u, &u_arr));
	for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
		for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
			if (use_mms) { // MMS初始条件
			u_arr[j][i] = manufactured_solution((PetscReal)i * h, (PetscReal)j * h, 0.0);
			} else { // 默认初始条件：中心热点
			if (i == N / 2 && j == N / 2) u_arr[j][i] = 1.0; else u_arr[j][i] = 0.0;
			}
		}
	}
	PetscCall(DMDAVecRestoreArray(da, u, &u_arr)); // 释放数组指针
	}
    
	/*
	* 创建求解器(KSP)和工作向量
	*/
	KSP ksp = NULL;
	if (is_implicit) { // 线性求解器只在隐式方法中需要
	PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
	PetscCall(KSPSetOperators(ksp, A, A)); // 设置求解的矩阵
	PetscCall(KSPSetFromOptions(ksp));     // 允许用户从命令行配置求解器参数
	}
	// 创建其他工作向量，使用VecDuplicate确保它们有和u完全相同的并行布局
	Vec u_old, rhs, f_vec;
	PetscCall(VecDuplicate(u, &u_old));
	PetscCall(VecDuplicate(u, &rhs));
	PetscCall(VecDuplicate(u, &f_vec));

	/*******************************************************************************
	* 时间步进主循环
	* 这是程序的核心，模拟温度随时间的演化。
	*******************************************************************************/
	for (PetscInt step = start_step; step < max_steps; step++) {
	// 更新时间和保存上一步的解
	time += dt;
	PetscCall(VecCopy(u, u_old));

	// 1. 计算当前时间步的源项 f_vec
	PetscScalar **f_arr;
	PetscCall(DMDAVecGetArray(da, f_vec, &f_arr));
	for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
		for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
			if (i == 0 || i == info.mx - 1 || j == 0 || j == info.my - 1) {
			f_arr[j][i] = 0.0;
			} else if (use_mms) {
			f_arr[j][i] = manufactured_source((PetscReal)i * h, (PetscReal)j * h, time, kappa, rho_c);
		} else {
			f_arr[j][i] = 0.0;
			}
		}
	}
	PetscCall(DMDAVecRestoreArray(da, f_vec, &f_arr));
        
	// 2. 根据时间方法进行求解
	if (is_implicit) {
	// -- 隐式方法 --
	// 构造右手项: rhs = ρc*u_old + Δt*f_new
	PetscCall(VecCopy(u_old, rhs));
	PetscCall(VecScale(rhs, rho_c));
	PetscCall(VecAXPY(rhs, dt, f_vec));
            
	// 在右手项中强制施加狄利克雷边界条件
	PetscScalar **rhs_arr;
	PetscCall(DMDAVecGetArray(da, rhs, &rhs_arr));
	for(PetscInt j = info.ys; j < info.ys + info.ym; j++) {
		for(PetscInt i = info.xs; i < info.xs + info.xm; i++) {
			if(i == 0 || i == info.mx-1 || j == 0 || j == info.my-1){
				if (use_mms) rhs_arr[j][i] = manufactured_solution((PetscReal)i*h, (PetscReal)j*h, time);
				else rhs_arr[j][i] = 0.0;
			}
		}
	}
	PetscCall(DMDAVecRestoreArray(da, rhs, &rhs_arr));

	// 求解线性系统 Au = rhs
	PetscCall(PetscLogEventBegin(SOLVE_TIME, 0, 0, 0, 0));
	PetscCall(KSPSolve(ksp, rhs, u));
	PetscCall(PetscLogEventEnd(SOLVE_TIME, 0, 0, 0, 0));

	} else { // -- 显式方法 --
	// 更新公式: u_new = u_old + (Δt/ρc) * (κ*∇²u_old + f_old)
	// (这里的矩阵A代表 κ*∇²)
	PetscCall(MatMult(A, u_old, rhs));    // rhs = κ*∇² * u_old
	PetscCall(VecAXPY(rhs, 1.0, f_vec));  // rhs = κ*∇² * u_old + f_old
	PetscCall(VecAXPY(u, dt / rho_c, rhs)); // u_new = u_old + (Δt/ρc)*rhs
            
	// 在更新后的解上直接强制施加狄利克雷边界条件
	PetscScalar **u_arr_explicit;
	PetscCall(DMDAVecGetArray(da, u, &u_arr_explicit));
	for(PetscInt j = info.ys; j < info.ys + info.ym; j++) {
		for(PetscInt i = info.xs; i < info.xs + info.xm; i++) {
			if(i == 0 || i == info.mx-1 || j == 0 || j == info.my-1){
			if (use_mms) u_arr_explicit[j][i] = manufactured_solution((PetscReal)i*h, (PetscReal)j*h, time);
			else u_arr_explicit[j][i] = 0.0;
			}
		}
	}
	PetscCall(DMDAVecRestoreArray(da, u, &u_arr_explicit));
	}

	// 3. 定期打印计算状态
	if (step % 50 == 0) {
	PetscReal norm_u;
	PetscCall(VecNorm(u, NORM_2, &norm_u)); // 集体操作：所有进程参与计算
	if (rank == 0) { // 单进程操作：只有0号进程打印结果
	PetscPrintf(PETSC_COMM_WORLD, "Step %5d: Time=%8.4f, ||u||_2 = %9.2e\n", step, (double)time, (double)norm_u);
		}
	}
        
        // 4. 定期保存可视化或重启文件
	if ( (vtk_output || enable_restart) && (step % io_interval == 0 || step == max_steps - 1) ) {
		if (vtk_output) {
		char filename[PETSC_MAX_PATH_LEN];
		sprintf(filename, "solution_step_%04d.vts", step);
		PetscViewer viewer;
		PetscCall(PetscViewerVTKOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer));
		PetscCall(VecView(u, viewer)); // VecView现在可以正确工作，因为它能从u关联的DMDA中获取几何信息
		PetscCall(PetscViewerDestroy(&viewer));
	}
	if (enable_restart) {
		char filename[PETSC_MAX_PATH_LEN];
	sprintf(filename, "restart_step_%04d.h5", step);
	PetscViewer viewer;
	PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer));
	PetscCall(VecView(u, viewer));
	// 将所有元数据作为属性写入HDF5文件
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

	//*在所有时间步结束后，如果使用了-mms选项，则计算数值解与精确解的误差。
	if (use_mms) {
	Vec u_exact_vec;
	PetscCall(VecDuplicate(u, &u_exact_vec));
	PetscScalar *exact_arr;
	PetscCall(VecGetArray(u_exact_vec, &exact_arr));
	PetscCall(VecGetOwnershipRange(u_exact_vec, &Istart, &Iend));
	// 遍历网格，计算每个点的精确解
	for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
		for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
		exact_arr[j][i] = manufactured_solution((PetscReal)i * h, (PetscReal)j * h, time);
		}
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
	/*******************************************************************************
	* 清理与收尾
	* 释放所有由PETSc创建的对象，防止内存泄漏。
	*******************************************************************************/
	PetscCall(VecDestroy(&u));
	PetscCall(VecDestroy(&u_old));
	PetscCall(VecDestroy(&rhs));
	PetscCall(VecDestroy(&f_vec));
	PetscCall(MatDestroy(&A));
	if (ksp) PetscCall(KSPDestroy(&ksp));
	PetscCall(DMDestroy(&da)); // 销毁DMDA对象
    
	// 结束PETSc程序，这是所有PETSc程序的强制性结尾步骤。
	PetscCall(PetscFinalize());
	return 0;
}


/*******************************************************************************
 * MMS 辅助函数
 * 这部分函数提供了MMS方法所需的精确解和源项的数学表达式。
 *******************************************************************************/
	// u(x,y,t) = sin(πx) * sin(πy) * exp(-t)
	PetscScalar manufactured_solution(PetscReal x, PetscReal y, PetscReal t) {
	return sin(M_PI * x) * sin(M_PI * y) * exp(-t);
	}

	// f = ρc*∂u/∂t - κ*∇²u
	PetscScalar manufactured_source(PetscReal x, PetscReal y, PetscReal t, PetscReal kappa, PetscReal rho_c) {
	PetscScalar u_t = -sin(M_PI * x) * sin(M_PI * y) * exp(-t);
	PetscScalar laplacian_u = -2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y) * exp(-t);
	return rho_c * u_t - kappa * laplacian_u;
	}	
