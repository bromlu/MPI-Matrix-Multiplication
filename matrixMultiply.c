#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include "matrixIO.h"
#include "mpi.h"

#define ONE_BILLION (double)1000000000.0

typedef struct {
  int *values;
  int rows;
  int columns;
} matrix_t;

typedef struct {
  int ARows;
  int AColumns;
  int BRows;
  int BColumns;
} matrix_info_t;

/* Return the current time. */
double now(void)
{
  struct timespec current_time;
  clock_gettime(CLOCK_REALTIME, &current_time);
  return current_time.tv_sec + (current_time.tv_nsec / ONE_BILLION);
}

/* Print an optional message, usage information, and exit in error.
 */
void
usage(char *prog_name, char *msge)
{
    if (msge && strlen(msge)) {
        fprintf(stderr, "\n%s\n\n", msge);
    }

    fprintf(stderr, "usage: %s [flags]\n", prog_name);
    fprintf(stderr, "  -h                     print help\n");
    fprintf(stderr, "  -a <input file A>      input file for matrix A\n");
    fprintf(stderr, "  -b <input file B>      input file for matrix B\n");
    fprintf(stderr, "  -o <output file>       set output file\n");
    fprintf(stderr, "  -n <number of threads> number of threads to use\n");

    exit(1);
}

/* Multiplies the matrix */ 
int *
multiply(int num_procs, int rank, int Arows, int AColumns, int Brows, int BColumns, int* A, int* B) {
    MPI_Status status;
    int* rtn;

    // Receives the rows in A
    int num_rows = Arows / num_procs;
    int AStart = rank * num_rows;
    int AStop = AStart + num_rows;
    if(rank == num_procs - 1) {
        AStop = Arows;
    }
    int byte_count = (AStop - AStart) * AColumns;
    
    if(rank != 0) {
        A = malloc(byte_count * sizeof(int));
        MPI_Recv(A, byte_count, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }

    // Sets up C
    int CRows = AStop - AStart;
    int CColumns = BColumns;
    int *C = calloc(CRows * CColumns, sizeof(int));

    // Receives columns from B
    int num_columns = BColumns / num_procs;
    int previous_processor = rank;
    int receive_from = 0;
    int BStart = previous_processor * num_columns;
    int BStop = BStart + num_columns;
    if(previous_processor == num_procs - 1) {
        BStop = BColumns;
    }
    byte_count = (BStop - BStart) * Brows;
    if(rank != 0) {
        B = malloc(byte_count * sizeof(int));
        MPI_Recv(B, byte_count, MPI_INT, receive_from, 0, MPI_COMM_WORLD, &status);
    }

    receive_from = rank - 1;
    if(receive_from < 0) {
        receive_from = num_procs - 1;
    }
    int send_to = rank + 1;
    if(send_to == num_procs) {
        send_to = 0;
    }
    int count = 0;

    while(count < num_procs) {
        // Multiply A and B
        for(int r = 0; r < AStop - AStart; r++) {
            for(int c = 0; c < BStop - BStart; c++) {
                for(int i = 0; i < Brows; i++) {
                    C[r * CColumns + c + BStart] += A[r * Brows + i] * B[c * Brows + i];
                }
            } 
        } 

        previous_processor = previous_processor - 1;
        if(previous_processor < 0) {
            previous_processor = num_procs - 1;
        }

        // If done, send C to processor 0, otherwise, pass B on to the next processor and get B from the previous
        count++;
        if(count == num_procs) {
            if(rank != 0) {
                MPI_Send(C, CColumns * CRows, MPI_INT, 0, 0, MPI_COMM_WORLD);
            } else {
                rtn = C;
            }
        } else {
            int bytes_to_send = byte_count;
            BStart = previous_processor * num_columns;
            BStop = BStart + num_columns;
            if(previous_processor == num_procs - 1) {
                BStop = BColumns;
            }
            byte_count = (BStop - BStart) * Brows;
            int* D = malloc(byte_count * sizeof(int));
            MPI_Sendrecv(B, bytes_to_send, MPI_INT, send_to, 0, D, byte_count, MPI_INT, receive_from, 0, MPI_COMM_WORLD, &status); 
            B = D;
        }
    }

    return rtn;
}

/* Splits A into groups of rows and sends each group to a different processor */
int *
split_and_send_A(matrix_t A, int num_procs) {    
    // Split and send groups of rows in A
    int num_rows = A.rows / num_procs;
    int* rtn;
    for (int p = num_procs - 1; p > -1; p--) {
        int start = p * num_rows;
        int stop = start + num_rows;
        if(p == num_procs - 1) {
            stop = A.rows;
        }
        int count = (stop - start) * A.columns;
        int *subArray = malloc(count * sizeof(int));
        for(int row = start; row < stop; row++) {
            for(int column = 0; column < A.columns; column++) {
                subArray[((row - start) * A.columns) + column] = MAT_ELT(A.values, A.columns, row, column);
            }
        }
        if(p == 0) {
            rtn = subArray;
        } else {
            MPI_Send(subArray, count, MPI_INT,
                p, 0, MPI_COMM_WORLD);
            free(subArray);
        }
    }
    return rtn;
}

/* Splits B into groups of columns and sends each group to a different processor */
int *
split_and_send_B(matrix_t B, int num_procs) {
    // Split and send groups of columns in B
    int num_columns = B.columns / num_procs;
    int* rtn;
    for (int p = num_procs - 1; p > -1; p--) {
        int start = p * num_columns;
        int stop = start + num_columns;
        if(p == num_procs - 1) {
            stop = B.columns;
        }
        int count = (stop - start) * B.rows;
        int *subArray = malloc(count * sizeof(int));
        for(int column = start; column < stop; column++) {
            for(int row = 0; row < B.rows; row++) {
                subArray[((column - start) * B.rows) + row] = MAT_ELT(B.values, B.columns, row, column);
            }
        }
        if(p == 0) {
            rtn = subArray;
        } else {
            MPI_Send(subArray, count, MPI_INT,
                p, 0, MPI_COMM_WORLD);
            free(subArray);
        }
    }
    return rtn;
}

/* Gathers rows of C from all other processors and then returns all rows of C */
int* 
gather_and_stitch(int num_procs, int rank, int Arows, int BColumns, int* C) {
    MPI_Status status;
    int num_rows = Arows / num_procs;
    int CColumns = BColumns;

    int *Answer = malloc(Arows * BColumns * sizeof(int));
    int offset = 0;

    for(int p = 0; p < num_procs; p++) {
        int AStart = p * num_rows;
        int AStop = AStart + num_rows;
        if(p == num_procs - 1) {
            AStop = Arows;
        }
        int CRows = AStop - AStart;
        if(p != 0) {
            C = malloc(CRows * CColumns * sizeof(int));
            MPI_Recv(C, CRows * CColumns, MPI_INT, p, 0, MPI_COMM_WORLD, &status);
        }
        memcpy(Answer + offset, C, CRows * CColumns * sizeof(int));
        offset += CRows * CColumns;
        free(C);
    }

    return Answer;
}

int
main(int argc, char **argv)
{
    // Gather arguments
    char *prog_name = argv[0];

    char* input_file_A = NULL;
    char* input_file_B = NULL;
    char* output_file = NULL;

    int character;
    while ((character = getopt(argc, argv, "ha:b:o:")) != -1) {
        switch (character) {
            case 'a':
                input_file_A = optarg;
                break;
            case 'b':
                input_file_B = optarg;
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'h':
            default:
                usage(prog_name, "");
        }
    }

    // Error checking
    if (!input_file_A) {
        usage(prog_name, "No input A file specified");
    }
    if (!input_file_B) {
        usage(prog_name, "No input B file specified");
    }
    if (!output_file) {
        usage(prog_name, "No output file specified");
    }
    if (strcmp(input_file_A, output_file) == 0 || strcmp(input_file_B, output_file) == 0) {
    usage(prog_name, "An input and an output file can't be the same");
    }    

    // Initialize MPI
    int num_procs;
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // If processor 0, load, split, and send out the matrix
    if(rank == 0) {
        fprintf(stderr, "Loading matrix A...");
        matrix_t A;
        A.values = read_matrix(&(A.rows), &(A.columns), input_file_A);
        fprintf(stderr, "Done\n");

        fprintf(stderr, "Loading matrix B...");
        matrix_t B;
        B.values = read_matrix(&(B.rows), &(B.columns), input_file_B);
        fprintf(stderr, "Done\n");

        fprintf(stderr, "Checking dimensions...");
        if(A.columns != B.rows) {
            fprintf(stderr, "Dimensions are not correct, A's columns, %d, should equal B's rows, %d", A.columns, B.rows);
            exit(1);
        }
        fprintf(stderr, "Done\n");

        // Start the timer
        double start_time = now();

        // Send info about the size of A and B
        matrix_info_t *info = malloc(sizeof(matrix_info_t));
        info->ARows = A.rows;
        info->AColumns = A.columns;
        info->BRows = B.rows;
        info->BColumns = B.columns;
        for (int p = 1; p < num_procs; p++) {
            MPI_Send(info, sizeof(matrix_info_t), MPI_BYTE, p, 0, MPI_COMM_WORLD);
        }

        int* Apart = split_and_send_A(A, num_procs);
        int* Bpart = split_and_send_B(B, num_procs);

        free(A.values);
        free(B.values);

        fprintf(stderr, "Multiplying A * B...");
        int* Cpart = multiply(num_procs, rank, info->ARows, info->AColumns, info->BRows, info->BColumns, Apart, Bpart);
        fprintf(stderr, "Done\n");

        // Gather all the results
        int *answer = gather_and_stitch(num_procs, rank, info->ARows, info->BColumns, Cpart);

        // Print time
        printf("    TOOK %5.3f seconds\n", now() - start_time);

        // Write to file
        fprintf(stderr, "Saving output...");
        write_matrix(answer, info->ARows, info->BColumns, output_file);
        fprintf(stderr, "Done\n");

        free(Apart);
        free(Bpart);
        free(answer);
    } else {
        // Receive matrix info
        matrix_info_t *info = malloc(sizeof(matrix_info_t));
        MPI_Status status;
        MPI_Recv(info, sizeof(matrix_info_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);

        // Multiply
        multiply(num_procs, rank, info->ARows, info->AColumns, info->BRows, info->BColumns, NULL, NULL);
        free(info);
    }

    MPI_Finalize();
}
