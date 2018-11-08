matrixChecker:
	gcc -Wall -Werror -o matrixChecker matrixChecker.c matrixIO.c

matrixMultiply:
	mpicc -Wall -Werror -o matrixMultiply matrixMultiply.c matrixIO.c

matrixGenerator:
	gcc -Wall -Werror -o matrixGenerator matrixGenerator.c matrixIO.c

clean:
	rm matrixGenerator matrixMultiply matrixChecker