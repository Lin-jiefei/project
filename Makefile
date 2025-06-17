# -*- mode: makefile -*-
# 项目配置
PROJECT_DIR = /home/linjiefei/project
PETSC_DIR = /home/linjiefei/petsc
PETSC_ARCH = arch-linux-c-debug
TARGET = heat_solver
SOURCE = project.c

# PETSc pkg-config 文件位置
petsc.pc := $(PETSC_DIR)/$(PETSC_ARCH)/lib/pkgconfig/petsc.pc
PACKAGES := $(petsc.pc)

# 编译器设置
CC := $(shell pkg-config --variable=ccompiler $(PACKAGES))
CFLAGS := $(shell pkg-config --variable=cflags_extra $(PACKAGES)) \
          $(shell pkg-config --cflags-only-other $(PACKAGES))
CPPFLAGS := $(shell pkg-config --cflags-only-I $(PACKAGES))
LDFLAGS := $(shell pkg-config --libs-only-L --libs-only-other $(PACKAGES))
LDFLAGS += $(patsubst -L%, $(shell pkg-config --variable=ldflag_rpath $(PACKAGES))%, \
           $(shell pkg-config --libs-only-L $(PACKAGES)))
LDLIBS := $(shell pkg-config --libs-only-l $(PACKAGES)) -lm

# 添加防御性编译标志
CFLAGS += -Wall -Wextra -Werror -pedantic -Wformat-security

# 目录和文件路径
SRC_PATH := $(PROJECT_DIR)/$(SOURCE)
OBJ := $(patsubst %.c,%.o,$(SOURCE))

# 构建规则
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)
	@echo "\033[32m编译成功! 可执行文件位置: $(PROJECT_DIR)/$(TARGET)\033[0m"
	@echo "\033[33m运行命令示例: mpirun -n 4 ./$(TARGET) -n 100 -dt 0.001 -time_method implicit\033[0m"

$(OBJ): $(SRC_PATH)
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $(SRC_PATH) -o $@

clean:
	$(RM) $(TARGET) $(OBJ)
	@echo "\033[33m已清理编译文件\033[0m"

# 测试命令
run-test:
	@echo "\033[34m运行基本测试...\033[0m"
	mpirun -n 2 ./$(TARGET) -n 100 -dt 0.001 -time_method explicit
	@echo "\033[34m测试完成!\033[0m"

run-test-implicit:
	@echo "\033[34m运行隐式方法测试...\033[0m"
	mpirun -n 2 ./$(TARGET) -n 200 -dt 0.005 -time_method implicit

restart-test:
	@echo "\033[34m测试重启功能...\033[0m"
	@echo "第一步: 创建初始运行"
	mpirun -n 2 ./$(TARGET) -n 150 -dt 0.002 -enable_restart true -restart_interval 5
	@echo "第二步: 从重启点继续运行"
	mpirun -n 2 ./$(TARGET) -enable_restart true

# 调试输出
print-config:
	@echo "项目目录: $(PROJECT_DIR)"
	@echo "源文件: $(SRC_PATH)"
	@echo "PETSC_DIR: $(PETSC_DIR)"
	@echo "PETSC_ARCH: $(PETSC_ARCH)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "CPPFLAGS: $(CPPFLAGS)"
	@echo "LDFLAGS: $(LDFLAGS)"
	@echo "LDLIBS: $(LDLIBS)"
