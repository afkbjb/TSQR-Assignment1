# TSQR Assignment 1

This repository contains the solutions for TSQR Assignment 1, implementing **Tall-Skinny QR (TSQR) decomposition** using **MPI** for parallel computing.

## Directory Structure
```
TSQR-Assignment1/
│── Interpreted_Language/   
│   ├── Q1.ipynb            # Q1: TSQR implementation in Python
│   ├── Q3_plot.ipynb       # Q3: Performance scaling visualization
│   ├── scaling_results.txt  # Execution results from Q3
│── C_Code/                 # C code for Q2 & Q3
│   ├── Q2.c                # TSQR implementation in C with MPI
│   ├── Q3.c                # TSQR execution time 
│   ├── Makefile            # Compilation script
│── README.md              
```

## Running C Code (MPI)
```sh
cd C_Code
make                  # Compile all programs
mpirun -np 4 ./Q2     # Run TSQR with 4 MPI processes
mpirun -np 4 ./Q3     # Run performance test
```

## Running Python Code
```sh
jupyter notebook Interpreted_Language/Q1.ipynb   # Q1 TSQR
jupyter notebook Interpreted_Language/Q3_plot.ipynb  # Q3 scaling plots
```

## Scaling Results
- `scaling_results.txt` contains execution times for different matrix sizes.

- `Q3_plot.ipynb` reads `scaling_results.txt` to generate plots.

  
