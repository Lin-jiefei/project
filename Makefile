# -*- mode: makefile -*-
PETSC_DIR = /home/linjiefei/petsc
PETSC_ARCH = arch-linux-c-debug

# PETSc pkg-config file location
petsc.pc := $(PETSC_DIR)/$(PETSC_ARCH)/lib/pkgconfig/petsc.pc

# Additional libraries that support pkg-config can be added here
PACKAGES := $(petsc.pc)

# Compiler settings using pkg-config
CC := $(shell pkg-config --variable=ccompiler $(PACKAGES))
CFLAGS := $(shell pkg-config --variable=cflags_extra $(PACKAGES)) \
          $(shell pkg-config --cflags-only-other $(PACKAGES))
CPPFLAGS := $(shell pkg-config --cflags-only-I $(PACKAGES))
LDFLAGS := $(shell pkg-config --libs-only-L --libs-only-other $(PACKAGES))
LDFLAGS += $(patsubst -L%, $(shell pkg-config --variable=ldflag_rpath $(PACKAGES))%, \
           $(shell pkg-config --libs-only-L $(PACKAGES)))
LDLIBS := $(shell pkg-config --libs-only-l $(PACKAGES)) -lm

# Target settings (保持原始功能)
TARGET := homework5
SOURCE := homework5.c
OBJ := $(patsubst %.c,%.o,$(SOURCE))

# Rules
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)
	@echo "\033[32m编译成功! 使用以下命令运行:\033[0m"
	@echo "    mpirun -n <进程数> ./$(TARGET) [选项]"

$(OBJ): $(SOURCE)
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	$(RM) $(TARGET) $(OBJ)
	@echo "\033[33m已清理编译文件\033[0m"

run-test:
	mpirun -n 2 ./$(TARGET) -n 1000 -ksp_type cg -pc_type jacobi -tol 1e-8

# Debug print target (保留调试功能)
print:
	@echo CC=$(CC)
	@echo CFLAGS=$(CFLAGS)
	@echo CPPFLAGS=$(CPPFLAGS)
	@echo LDFLAGS=$(LDFLAGS)
	@echo LDLIBS=$(LDLIBS)
