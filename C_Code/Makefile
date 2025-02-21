CC = mpicc
CFLAGS = -O2 -Wall \
    -I/opt/homebrew/Cellar/lapack/3.12.1/include \
    -I/opt/homebrew/Cellar/openblas/0.3.29/include \
    -L/opt/homebrew/Cellar/lapack/3.12.1/lib \
    -L/opt/homebrew/Cellar/openblas/0.3.29/lib \
    -llapacke -llapack -lblas -lopenblas -lm

TARGETS = Q2 Q3

all: $(TARGETS)

Q2: Q2.c
	$(CC) $(CFLAGS) Q2.c -o Q2

Q3: Q3.c
	$(CC) $(CFLAGS) Q3.c -o Q3

run_q2: Q2
	mpirun -np 4 ./Q2

run_q3: Q3
	mpirun -np 4 ./Q3 > scaling_results.txt

clean:
	rm -f $(TARGETS) scaling_results.txt
