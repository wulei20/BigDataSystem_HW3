#include "your_reduce.h"
#include <omp.h>

// You may add your functions and variables here

void YOUR_Reduce(const int *sendbuf, int *recvbuf, int count) {
    /*
        Modify the code here.
        Your implementation should have the same result as this MPI_Reduce
        call. However, you MUST NOT use MPI_Reduce (or like) for your hand-in
        version. Instead, you should use MPI_Send and MPI_Recv (or like). See
        the homework instructions for more information.
    */
    
    /*
        You may assume:
        - Data type is always `int` (MPI_INT).
        - Operation is always MPI_SUM.
        - Process to hold final results is always process 0.
        - Number of processes is 2, 4, or 8.
        - Number of elements (`count`) is 1, 16, 256, 4096, 65536, 1048576,
          16777216, or 268435456.
        For other cases, your code is allowed to produce wrong results or even
        crash. It is totally fine.
    */

#define USE_PARALLEL
#ifndef USE_PARALLEL
    int rank, size;
    
    // Get the rank of the process and the size of the communicator
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        // Copy the sendbuf data into recvbuf for process 0
        for (int i = 0; i < count; i++) {
            recvbuf[i] = sendbuf[i];
        }

        // Receive data from other processes and sum it into recvbuf
        int *tempbuf = (int *)malloc(sizeof(int) * count);
        for (int p = 1; p < size; p++) {
            MPI_Recv(tempbuf, count, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < count; i++) {
                recvbuf[i] += tempbuf[i];
            }
        }

        free(tempbuf);
    } else {
        // Other processes send their data to process 0
        MPI_Send(sendbuf, count, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

#else 
    // #define PARALLEL_BY_INDEX
    #ifndef PARALLEL_BY_INDEX
        #define USE_OPENMP
        #ifndef USE_OPENMP
            int rank, size;
            
            // Get the rank of the process and the size of the communicator
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);
            
            // Step 1: Initialize the recvbuf with sendbuf data (local sum)
            for (int i = 0; i < count; i++) {
                recvbuf[i] = sendbuf[i];
            }

            // Step 2: Binary tree reduction process
            int step = 1;
            while (step < size) {
                if (rank % (2 * step) == 0) {
                    // This process will receive from its pair (rank + step)
                    if (rank + step < size) {
                        int *tempbuf = (int *)malloc(sizeof(int) * count);
                        MPI_Recv(tempbuf, count, MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        
                        // Sum the received buffer into recvbuf
                        for (int i = 0; i < count; i++) {
                            recvbuf[i] += tempbuf[i];
                        }
                        free(tempbuf);
                    }
                } else {
                    // This process will send its data to the paired process (rank - step)
                    MPI_Send(recvbuf, count, MPI_INT, rank - step, 0, MPI_COMM_WORLD);
                    break;  // Exit after sending
                }
                step *= 2;
            }
        #else
            int rank, size;
    
            // Get the rank of the process and the size of the communicator
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);
            
            // create a parallel region
            #pragma omp parallel
            {
                int num_threads = omp_get_num_threads();
                int thread_id = omp_get_thread_num();
                int manip_start = thread_id * count / num_threads;
                int manip_end = (thread_id == num_threads - 1) ? count : (thread_id + 1) * count / num_threads;
                int manip_count = manip_end - manip_start;
                
                // Step 1: Initialize the recvbuf with sendbuf data (local sum)
                #pragma omp for
                for (int i = 0; i < count; i++) {
                    recvbuf[i] = sendbuf[i];
                }

                // Step 2: Binary tree reduction process
                int step = 1;
                while (step < size) {
                    if (rank % (2 * step) == 0) {
                        // This process will receive from its pair (rank + step)
                        if (rank + step < size) {
                            int *tempbuf = (int *)malloc(sizeof(int) * manip_count);
                            MPI_Recv(tempbuf, manip_count, MPI_INT, rank + step, thread_id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            
                            // Sum the received buffer into recvbuf
                            for (int i = manip_start; i < manip_end; i++) {
                                recvbuf[i] += tempbuf[i - manip_start];
                            }
                            free(tempbuf);
                        }
                    } else {
                        // This process will send its data to the paired process (rank - step)
                        MPI_Send(recvbuf + manip_start, manip_count, MPI_INT, rank - step, thread_id, MPI_COMM_WORLD);
                        break;  // Exit after sending
                    }
                    step *= 2;
                }
            }
        #endif
    #else
        int rank, size;
        
        // Get the rank of the process and the size of the communicator
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Calculate the number of elements each process will work on
        int chunk_size = count / size;
        int remainder = count % size;

        // Each process calculates the local sum of its portion of the array
        int local_start = rank * chunk_size + (rank < remainder ? rank : remainder);
        int local_count = chunk_size + (rank < remainder ? 1 : 0);
        
        // Initialize the recvbuf to 0
        for (int i = 0; i < count; i++) {
            recvbuf[i] = 0;
        }

        // Calculate the chunk belong to this process and send other chunk to other processes
        for (int i = 0; i < size; i++) {
            int remote_start = i * chunk_size + (i < remainder ? i : remainder);
            int remote_count = chunk_size + (i < remainder ? 1 : 0);

            if (i != rank) {
                MPI_Send(sendbuf + remote_start, remote_count, MPI_INT, i, 0, MPI_COMM_WORLD);
            } else {
                for (int j = 0; j < size; j++) {
                    if (j != rank) {
                        int *tempbuf = (int *)malloc(sizeof(int) * remote_count);
                        MPI_Recv(tempbuf, remote_count, MPI_INT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        for (int k = 0; k < remote_count; k++) {
                            recvbuf[local_start + k] += tempbuf[k];
                        }
                        free(tempbuf);
                    } else {
                        for (int k = 0; k < remote_count; k++) {
                            recvbuf[local_start + k] += sendbuf[remote_start + k];
                        }
                    }
                }
            }
        }

        // Send the result back to process 0
        if (rank != 0) {
            MPI_Send(recvbuf + local_start, local_count, MPI_INT, 0, 0, MPI_COMM_WORLD);
        } else {
            for (int i = 1; i < size; i++) {
                int remote_start = i * chunk_size + (i < remainder ? i : remainder);
                int remote_count = chunk_size + (i < remainder ? 1 : 0);
                MPI_Recv(recvbuf + remote_start, remote_count, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

    #endif
#endif

}
