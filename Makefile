# 编译器和编译选项
CC = mpicc
CFLAGS = -O2 -Wall \
    -I/opt/homebrew/Cellar/lapack/3.12.1/include \
    -I/opt/homebrew/Cellar/openblas/0.3.29/include \
    -L/opt/homebrew/Cellar/lapack/3.12.1/lib \
    -L/opt/homebrew/Cellar/openblas/0.3.29/lib \
    -llapacke -llapack -lblas -lopenblas -lm

# 目标文件
TARGETS = Q2 Q3

all: $(TARGETS)

# 编译 Q2
Q2: Q2.c
	$(CC) $(CFLAGS) Q2.c -o Q2

# 编译 Q3
Q3: Q3.c
	$(CC) $(CFLAGS) Q3.c -o Q3

# 运行 Q2（验证 TSQR 正确性）
run_q2: Q2
	mpirun -np 4 ./Q2

# 运行 Q3（测试 scalability 并保存数据）
run_q3: Q3
	mpirun -np 4 ./Q3 > scaling_results.txt

# 清理
clean:
	rm -f $(TARGETS) scaling_results.txt
