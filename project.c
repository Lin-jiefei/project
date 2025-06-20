#include <petsc.h>
#include <petscksp.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscviewerhdf5.h>
#include <petscerror.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* 制造解函数声明 */
PetscScalar manufactured_solution(PetscReal x, PetscReal y, PetscReal t);
PetscScalar manufactured_source(PetscReal x, PetscReal y, PetscReal t, PetscReal kappa, PetscReal rho_c);

int main(int argc, char **argv) {
    PetscErrorCode ierr;
    const char *help =
        "使用DMDA在二维单位正方形上求解瞬态热传导方程。\n"
        "方程: ρc*∂u/∂t - κ*∇²u = f\n\n"
        "选项:\n"
        "  -n <size>             : 网格大小 N x N (默认: 100)\n"
        "  -dt <timestep>        : 时间步长 (默认: 0.001)\n"
        "  -max_steps <int>      : 最大时间极数 (默认: 1000)\n"
        "  -time_method <str>    : 时间推进方法 'implicit' 或 'explicit' (默认: implicit)\n"
        "  -kappa <value>        : 热导率 κ (默认: 1.0)\n"
        "  -rho_c <value>        : 密度×比热 ρc (默认: 极.0)\n\n"
        "验证与输入/输出选项:\n"
        "  -mms                  : 使用制造解方法进行验证 (默认: PETSC_FALSE)\n"
        "  -enable_restart       : 启用HDF5重启文件保存 (默认: PETSC_FALSE)\n"
        "  -restart_load <file>  : 从HDF5重启文件加载并继续计算\n"
        "  -vtk_output           : 启用VTK文件输出以进行可视化 (默认: PETSC_FALSE)\n"
        "  -io_interval <int>    : 保存重启/VTK文件的迭代间隔 (默认: 10)\n"
        "  -view_solution        : 在终端打印最终解向量\n\n";

    /* 初始化PETSc */
    ierr = PetscInitialize(&argc, &argv, NULL, help);
    if (ierr) return ierr;

    PetscMPIInt rank;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (ierr) return ierr;

    /* 性能日志事件 */
    PetscLogEvent ASSEMBLY_TIME, SOLVE_TIME;
    ierr = PetscLogEventRegister("Matrix Assembly", MAT_CLASSID, &ASSEMBLY_TIME);
    if (ierr) return ierr;
    ierr = PetscLogEventRegister("Linear Solve", KSP_CLASSID, &SOLVE_TIME);
    if (ierr) return ierr;

    /* 参数设置 */
    PetscInt N = 100;
    PetscReal dt = 0.001;
    PetscInt max_steps = 1000;
    PetscReal kappa = 1.0;
    PetscReal rho_c = 1.0;
    char time_method[20] = "implicit";
    PetscBool view_solution = PETSC_FALSE;
    PetscBool use_mms = PETSC_FALSE;
    PetscBool enable_restart = PETSC_FALSE;
    PetscInt io_interval = 10;
    char restart_load_file[PETSC_MAX_PATH_LEN] = "";
    PetscBool restart_load_flag = PETSC_FALSE;
    PetscBool vtk_output = PETSC_FALSE;

    /* 从命令行解析参数 */
    ierr = PetscOptionsGetInt(NULL, NULL, "-n", &N, NULL);
    if (ierr) return ierr;
    ierr = PetscOptionsGetReal(NULL, NULL, "-dt", &dt, NULL);
    if (ierr) return ierr;
    ierr = PetscOptionsGetInt(NULL, NULL, "-max_steps", &max_steps, NULL);
    if (ierr) return ierr;
    ierr = PetscOptionsGetString(NULL, NULL, "-time_method", time_method, sizeof(time_method), NULL);
    if (ierr) return ierr;
    ierr = PetscOptionsGetReal(NULL, NULL, "-kappa", &kappa, NULL);
    if (ierr) return ierr;
    ierr = PetscOptionsGetReal(NULL, NULL, "-rho_c", &rho_c, NULL);
    if (ierr) return ierr;
    ierr = PetscOptionsGetBool(NULL, NULL, "-mms", &use_mms, NULL);
    if (ierr) return ierr;
    ierr = PetscOptionsGetBool(NULL, NULL, "-enable_restart", &enable_restart, NULL);
    if (ierr) return ierr;
    ierr = PetscOptionsGetInt(NULL, NULL, "-io_interval", &io_interval, NULL);
    if (ierr) return ierr;
    ierr = PetscOptionsGetString(NULL, NULL, "-restart_load", restart_load_file, sizeof(restart_load_file), &restart_load_flag);
    if (ierr) return ierr;
    ierr = PetscOptionsGetBool(NULL, NULL, "-vtk_output", &vtk_output, NULL);
    if (ierr) return ierr;
    ierr = PetscOptionsGetBool(NULL, NULL, "-view_solution", &view_solution, NULL);
    if (ierr) return ierr;

    /* 参数验证 - 使用正确的SETERRQ格式 */
    if (kappa <= 0.0) {
        SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Heat conductivity must be positive");
    }
    if (rho_c <= 0.0) {
        SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Density * specific heat must be positive");
    }
    if (N <= 1) {
        SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Mesh size must be greater than 1");
    }
    if (dt <= 0.0) {
        SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Time step must be positive");
    }
    if (io_interval <= 0) {
        SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "I/O interval must be positive");
    }

    /* 创建DMDA对象 */
    DM da;
    ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 
                         DMDA_STENCIL_STAR, N, N, PETSC_DECIDE, PETSC_DECIDE,
                         1, 1, NULL, NULL, &da);
    if (ierr) return ierr;
    ierr = DMSetUp(da);
    if (ierr) return ierr;
    ierr = DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    if (ierr) return ierr;

    /* 创建向量和矩阵 */
    Vec u;
    ierr = DMCreateGlobalVector(da, &u);
    if (ierr) return ierr;
    ierr = PetscObjectSetName((PetscObject)u, "temperature");
    if (ierr) return ierr;
    
    Mat A;
    ierr = DMCreateMatrix(da, &A);
    if (ierr) return ierr;

    /* 初始化状态 */
    PetscInt start_step = 0;
    PetscReal time = 0.0;
    PetscReal h = 1.0 / (PetscReal)(N - 1);

    /* 从重启文件加载 */
    if (restart_load_flag) {
        PetscViewer viewer;
        ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, restart_load_file, FILE_MODE_READ, &viewer);
        if (ierr) return ierr;
        ierr = VecLoad(u, viewer);
        if (ierr) return ierr;
        
        ierr = PetscViewerHDF5PushGroup(viewer, "/");
        if (ierr) return ierr;
        PetscInt loaded_N;
        ierr = PetscViewerHDF5ReadAttribute(viewer, "N", "N", PETSC_INT, NULL, &loaded_N);
        if (ierr) return ierr;
        ierr = PetscViewerHDF5ReadAttribute(viewer, "dt", "dt", PETSC_REAL, NULL, &dt);
        if (ierr) return ierr;
        ierr = PetscViewerHDF5ReadAttribute(viewer, "kappa", "kappa", PETSC_REAL, NULL, &kappa);
        if (ierr) return ierr;
        ierr = PetscViewerHDF5ReadAttribute(viewer, "rho_c", "rho_c", PETSC_REAL, NULL, &rho_c);
        if (ierr) return ierr;
        ierr = PetscViewerHDF5ReadAttribute(viewer, "time", "time", PETSC_REAL, NULL, &time);
        if (ierr) return ierr;
        ierr = PetscViewerHDF5ReadAttribute(viewer, "iteration", "iteration", PETSC_INT, NULL, &start_step);
        if (ierr) return ierr;
        ierr = PetscViewerHDF5PopGroup(viewer);
        if (ierr) return ierr;
        
        ierr = PetscViewerDestroy(&viewer);
        if (ierr) return ierr;

        /* 修复重启文件大小验证 */
        if (loaded_N != N) {
            SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, 
                    "Restart file was saved with N=%d, but current setting is N=%d. Sizes must match.", 
                    loaded_N, N);
        }
        
        start_step++;
        if (rank == 0) {
            PetscPrintf(PETSC_COMM_WORLD, "成功从文件 %s 重启。将在第 %d 步 (time=%.4f) 继续计算。\n", 
                        restart_load_file, start_step, (double)time);
        }
    }

    /* 打印基本信息 */
    if (rank == 0) {
        PetscPrintf(PETSC_COMM_WORLD, "=== 2D瞬态热传导方程求解器 (DMDA版本) ===\n");
        PetscPrintf(PETSC_COMM_WORLD, "网格: %dx%d (h=%.4f), 时间方法: %s, dt: %.4f, 最大步数: %d\n", 
                    N, N, (double)h, time_method, (double)dt, max_steps);
        PetscPrintf(PETSC_COMM_WORLD, "参数: kappa=%.2f, rho_c=%.2f, MMS: %s\n", 
                    (double)kappa, (double)rho_c, use_mms ? "enabled" : "disabled");
    }
    
    /* 显式格式稳定性分析 */
    if (strcmp(time_method, "explicit") == 0) {
        PetscReal cfl = kappa * dt / (rho_c * h * h);
        if (rank == 0) {
            PetscPrintf(PETSC_COMM_WORLD, "稳定性分析 (显式): 扩散数 α = κ*Δt/(ρc*h²) = %.4f\n", (double)cfl);
            if (cfl > 0.25) {
                PetscPrintf(PETSC_COMM_WORLD, "警告: 违反稳定性条件 α <= 0.25！模拟结果可能不稳定。\n");
            }
        }
    }

    /* 矩阵组装 */
    ierr = PetscLogEventBegin(ASSEMBLY_TIME, 0, 0, 0, 0);
    if (ierr) return ierr;
    
    PetscBool is_implicit;
    ierr = PetscStrcmp(time_method, "implicit", &is_implicit);
    if (ierr) return ierr;
    
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da, &info);
    if (ierr) return ierr;

    for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
        for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
            MatStencil row_stencil = {0, j, i, 0};
            
            if (i == 0 || i == info.mx - 1 || j == 0 || j == info.my - 1) {
                PetscScalar v = 1.0;
                ierr = MatSetValuesStencil(A, 1, &row_stencil, 1, &row_stencil, &v, INSERT_VALUES);
                if (ierr) return ierr;
            } else {
                MatStencil cols_stencil[5] = {{0, j, i, 0}, {0, j, i-1, 0}, 
                                              {0, j, i+1, 0}, {0, j-1, i, 0}, 
                                              {0, j+1, i, 0}};
                PetscScalar vals[5];
                PetscInt ncols = 5;
                
                if (is_implicit) {
                    vals[0] = rho_c + 4.0*kappa*dt/(h*h);
                    vals[1] = -kappa*dt/(h*h);
                    vals[2] = -kappa*dt/(h*h);
                    vals[3] = -kappa*dt/(h*h);
                    vals[4] = -kappa*dt/(h*h);
                } else {
                    vals[0] = -4.0*kappa/(h*h);
                    vals[1] = kappa/(h*h);
                    vals[2] = kappa/(h*h);
                    vals[3] = kappa/(h*h);
                    vals[4] = kappa/(h*h);
                }
                
                ierr = MatSetValuesStencil(A, 1, &row_stencil, ncols, cols_stencil, vals, INSERT_VALUES);
                if (ierr) return ierr;
            }
        }
    }
    
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    if (ierr) return ierr;
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    if (ierr) return ierr;
    ierr = PetscLogEventEnd(ASSEMBLY_TIME, 0, 0, 0, 0);
    if (ierr) return ierr;

    /* 设置初始条件 */
    if (!restart_load_flag) {
        PetscScalar **u_arr;
        ierr = DMDAVecGetArray(da, u, &u_arr);
        if (ierr) return ierr;
        
        for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
            for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
                if (use_mms) {
                    u_arr[j][i] = manufactured_solution((PetscReal)i * h, (PetscReal)j * h, 0.0);
                } else {
                    u_arr[j][i] = (i == N/2 && j == N/2) ? 1.0 : 0.0;
                }
            }
        }
        ierr = DMDAVecRestoreArray(da, u, &u_arr);
        if (ierr) return ierr;
    }
    
    /* 创建求解器和工作向量 */
    KSP ksp = NULL;
    if (is_implicit) {
        ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);
        if (ierr) return ierr;
        ierr = KSPSetOperators(ksp, A, A);
        if (ierr) return ierr;
        ierr = KSPSetFromOptions(ksp);
        if (ierr) return ierr;
    }
    
    Vec u_old, rhs, f_vec;
    ierr = VecDuplicate(u, &u_old);
        if (ierr) return ierr;
    ierr = VecDuplicate(u, &rhs);
    if (ierr) return ierr;
    ierr = VecDuplicate(u, &f_vec);
    if (ierr) return ierr;

    /* 时间步进主循环 */
    for (PetscInt step = start_step; step < max_steps; step++) {
        time += dt;
        ierr = VecCopy(u, u_old);
        if (ierr) return ierr;

        /* 计算当前源项 */
        PetscScalar **f_arr;
        ierr = DMDAVecGetArray(da, f_vec, &f_arr);
        if (ierr) return ierr;
        
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
        ierr = DMDAVecRestoreArray(da, f_vec, &f_arr);
        if (ierr) return ierr;
        
        /* 根据时间方法求解 */
        if (is_implicit) {
            /* 构造右手项 */
            ierr = VecCopy(u_old, rhs);
            if (ierr) return ierr;
            ierr = VecScale(rhs, rho_c);
            if (ierr) return ierr;
            ierr = VecAXPY(rhs, dt, f_vec);
            if (ierr) return ierr;
            
            /* 施加边界条件 */
            PetscScalar **rhs_arr;
            ierr = DMDAVecGetArray(da, rhs, &rhs_arr);
            if (ierr) return ierr;
            for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
                for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
                    if (i == 0 || i == info.mx-1 || j == 0 || j == info.my-1) {
                        rhs_arr[j][i] = use_mms ? 
                            manufactured_solution((PetscReal)i*h, (PetscReal)j*h, time) : 0.0;
                    }
                }
            }
            ierr = DMDAVecRestoreArray(da, rhs, &rhs_arr);
            if (ierr) return ierr;
            
            /* 求解线性系统 */
            ierr = PetscLogEventBegin(SOLVE_TIME, 0, 0, 0, 0);
            if (ierr) return ierr;
            ierr = KSPSolve(ksp, rhs, u);
            if (ierr) return ierr;
            ierr = PetscLogEventEnd(SOLVE_TIME, 0, 0, 0, 0);
            if (ierr) return ierr;
        } else {
            /* 显式更新 */
            ierr = MatMult(A, u_old, rhs);
            if (ierr) return ierr;
            ierr = VecAXPY(rhs, 1.0, f_vec);
            if (ierr) return ierr;
            ierr = VecAXPY(u, dt / rho_c, rhs);
            if (ierr) return ierr;
            
            /* 施加边界条件 */
            PetscScalar **u_arr_explicit;
            ierr = DMDAVecGetArray(da, u, &u_arr_explicit);
            if (ierr) return ierr;
            for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
                for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
                    if (i == 0 || i == info.mx-1 || j == 0 || j == info.my-1) {
                        u_arr_explicit[j][i] = use_mms ? 
                            manufactured_solution((PetscReal)i*h, (PetscReal)j*h, time) : 0.0;
                    }
                }
            }
            ierr = DMDAVecRestoreArray(da, u, &u_arr_explicit);
            if (ierr) return ierr;
        }
        
        /* 状态打印 */
        if (step % 50 == 0) {
            PetscReal norm_u;
            ierr = VecNorm(u, NORM_2, &norm_u);
            if (ierr) return ierr;
            if (rank == 0) {
                PetscPrintf(PETSC_COMM_WORLD, "Step %5d: Time=%8.4f, ||u||_2 = %9.2e\n", 
                            step, (double)time, (double)norm_u);
            }
        }
        
        /* 保存文件 */
        if ((vtk_output || enable_restart) && (step % io_interval == 0 || step == max_steps - 1)) {
            if (vtk_output) {
                char filename[PETSC_MAX_PATH_LEN];
                sprintf(filename, "solution_step_%04d.vts", step);
                PetscViewer viewer;
                ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
                if (ierr) return ierr;
                ierr = VecView(u, viewer);
                if (ierr) return ierr;
                ierr = PetscViewerDestroy(&viewer);
                if (ierr) return ierr;
            }
            if (enable_restart) {
                char filename[PETSC_MAX_PATH_LEN];
                sprintf(filename, "restart_step_%04d.h5", step);
                PetscViewer viewer;
                ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
                if (ierr) return ierr;
                ierr = VecView(u, viewer);
                if (ierr) return ierr;
                ierr = PetscViewerHDF5PushGroup(viewer, "/");
                if (ierr) return ierr;
                ierr = PetscViewerHDF5WriteAttribute(viewer, "N", "N", PETSC_INT, &N);
                if (ierr) return ierr;
                ierr = PetscViewerHDF5WriteAttribute(viewer, "dt", "dt", PETSC_REAL, &dt);
                if (ierr) return ierr;
                ierr = PetscViewerHDF5WriteAttribute(viewer, "kappa", "kappa", PETSC_REAL, &kappa);
                if (ierr) return ierr;
                ierr = PetscViewerHDF5WriteAttribute(viewer, "rho_c", "rho_c", PETSC_REAL, &rho_c);
                if (ierr) return ierr;
                ierr = PetscViewerHDF5WriteAttribute(viewer, "time", "time", PETSC_REAL, &time);
                if (ierr) return ierr;
                ierr = PetscViewerHDF5WriteAttribute(viewer, "iteration", "iteration", PETSC_INT, &step);
                if (ierr) return ierr;
                ierr = PetscViewerHDF5PopGroup(viewer);
                if (ierr) return ierr;
                ierr = PetscViewerDestroy(&viewer);
                if (ierr) return ierr;
                if (rank == 0) {
                    PetscPrintf(PETSC_COMM_WORLD, "保存了重启文件: %s\n", filename);
                }
            }
        }
    }

    /* MMS误差计算 */
    if (use_mms) {
        Vec u_exact;
        ierr = VecDuplicate(u, &u_exact);
        if (ierr) return ierr;
        
        PetscScalar **exact_arr;
        ierr = DMDAVecGetArray(da, u_exact, &exact_arr);
        if (ierr) return ierr;
        
        for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
            for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
                exact_arr[j][i] = manufactured_solution((PetscReal)i * h, (PetscReal)j * h, time);
            }
        }
        
        ierr = DMDAVecRestoreArray(da, u_exact, &exact_arr);
        if (ierr) return ierr;
        
        ierr = VecAXPY(u, -1.0, u_exact);
        if (ierr) return ierr;
        
        PetscReal err_norm;
        ierr = VecNorm(u, NORM_INFINITY, &err_norm);
        if (ierr) return ierr;
        
        if (rank == 0) {
            PetscPrintf(PETSC_COMM_WORLD, "\n===== MMS 验证 =====\n");
            PetscPrintf(PETSC_COMM_WORLD, "在时间 T=%.4f 时的 L-infinity 误差 ||u_num - u_exact||_∞ 为: %.4e\n", 
                         (double)time, (double)err_norm);
            PetscPrintf(PETSC_COMM_WORLD, "要确定收敛阶，您需要使用不同的 -n 和 -dt 值多次运行此程序并分析误差变化。\n");
        }
        ierr = VecDestroy(&u_exact);
        if (ierr) return ierr;
    }
    
    /* 查看最终解 */
    if (view_solution) {
        PetscPrintf(PETSC_COMM_WORLD, "===== 最终解 =====\n");
        ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);
        if (ierr) return ierr;
    }
    
    /* 清理资源 */
    ierr = VecDestroy(&u);
    if (ierr) return ierr;
    ierr = VecDestroy(&u_old);
    if (ierr) return ierr;
    ierr = VecDestroy(&rhs);
    if (ierr) return ierr;
    ierr = VecDestroy(&f_vec);
    if (ierr) return ierr;
    ierr = MatDestroy(&A);
    if (ierr) return ierr;
    if (ksp) {
        ierr = KSPDestroy(&ksp);
        if (ierr) return ierr;
    }
    ierr = DMDestroy(&da);
    if (ierr) return ierr;
    
    ierr = PetscFinalize();
    return ierr;
}

/* MMS辅助函数 */
PetscScalar manufactured_solution(PetscReal x, PetscReal y, PetscReal t) {
    return sin(M_PI * x) * sin(M_PI * y) * exp(-t);
}

PetscScalar manufactured_source(PetscReal x, PetscReal y, PetscReal t, PetscReal kappa, PetscReal rho_c) {
    PetscScalar u_t = -sin(M_PI * x) * sin(M_PI * y) * exp(-t);
    PetscScalar laplacian_u = -2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y) * exp(-t);
    return rho_c * u_t - kappa * laplacian_u;
}
