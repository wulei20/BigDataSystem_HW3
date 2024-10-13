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
        // #define USE_OPENMP
        #ifndef USE_OPENMP
            int rank, size;
            
            // Get the rank of the process and the size of the communicator
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);
            
            // Special case for size = 2
            if (size == 2) {
                if (rank == 0) {
                    MPI_Recv(recvbuf, count, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    for (int i = 0; i < count; i++) {
                        recvbuf[i] += sendbuf[i];
                    }
                } else {
                    // Process 1 sends its data to process 0
                    MPI_Send(sendbuf, count, MPI_INT, 0, 0, MPI_COMM_WORLD);
                }
                return;
            }

            // Binary tree reduction process
            int step = 1;
            int *tempbuf = (int *)malloc(sizeof(int) * count);
            bool init_recv = false;
            while (step < size) {
                if (rank % (2 * step) == 0) {
                    // This process will receive from its pair (rank + step)
                    if (rank + step < size) {
                        MPI_Recv(tempbuf, count, MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        if (!init_recv) {
                            // Initialize the recvbuf with sendbuf data (local sum)
                            for (int i = 0; i < count; i++) {
                                recvbuf[i] = sendbuf[i] + tempbuf[i];
                            }
                            init_recv = true;
                        } else {
                            // Sum the received buffer into recvbuf
                            for (int i = 0; i < count; i++) {
                                recvbuf[i] += tempbuf[i];
                            }
                        }
                    }
                } else {
                    // This process will send its data to the paired process (rank - step)
                    if (!init_recv)
                        MPI_Send(sendbuf, count, MPI_INT, rank - step, 0, MPI_COMM_WORLD);
                    else
                        MPI_Send(recvbuf, count, MPI_INT, rank - step, 0, MPI_COMM_WORLD);
                    break;  // Exit after sending
                }
                step *= 2;
            }
            free(tempbuf);
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

                // Special case for size = 2
                if (size == 2) {
                    if (rank == 0) {
                        MPI_Recv(recvbuf + manip_start, manip_count, MPI_INT, 1, thread_id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        for (int i = manip_start; i < manip_end; i++) {
                            recvbuf[i] += sendbuf[i];
                        }
                    } else {
                        // Process 1 sends its data to process 0
                        MPI_Send(sendbuf + manip_start, manip_count, MPI_INT, 0, thread_id, MPI_COMM_WORLD);
                    }
                } else {
                    // Binary tree reduction process
                    int step = 1;
                    int *tempbuf = (int *)malloc(sizeof(int) * manip_count);
                    bool init_recv = false;
                    while (step < size) {
                        if (rank % (2 * step) == 0) {
                            // This process will receive from its pair (rank + step)
                            if (rank + step < size) {
                                MPI_Recv(tempbuf, manip_count, MPI_INT, rank + step, thread_id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                if (!init_recv) {
                                    // Initialize the recvbuf with sendbuf data (local sum)
                                    for (int i = 0; i < manip_count; i++) {
                                        recvbuf[manip_start + i] = sendbuf[manip_start + i] + tempbuf[i];
                                    }
                                    init_recv = true;
                                } else {
                                    // Sum the received buffer into recvbuf
                                    for (int i = 0; i < manip_count; i++) {
                                        recvbuf[manip_start + i] += tempbuf[i];
                                    }
                                }
                            }
                        } else {
                            // This process will send its data to the paired process (rank - step)
                            if (!init_recv)
                                MPI_Send(sendbuf + manip_start, manip_count, MPI_INT, rank - step, thread_id, MPI_COMM_WORLD);
                            else
                                MPI_Send(recvbuf + manip_start, manip_count, MPI_INT, rank - step, thread_id, MPI_COMM_WORLD);
                            break;  // Exit after sending
                        }
                        step *= 2;
                    }
                    free(tempbuf);
                }
            }
        #endif
    #else
        int rank, size;
        
        // Get the rank of the process and the size of the communicator
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        // Special case for size = 2
        if (size == 2) {
            if (rank == 0) {
                MPI_Recv(recvbuf, count, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int i = 0; i < count; i++) {
                    recvbuf[i] += sendbuf[i];
                }
            } else {
                // Process 1 sends its data to process 0
                MPI_Send(sendbuf, count, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
            return;
        }

        // Calculate the number of elements each process will work on
        int chunk_size = count / size;
        int remainder = count % size;

        // Each process calculates the local sum of its portion of the array
        int local_start = rank * chunk_size + (rank < remainder ? rank : remainder);
        int local_count = chunk_size + (rank < remainder ? 1 : 0);

        int zero_count = chunk_size + (0 < remainder ? 1 : 0);
        int remote_start, remote_count;

        // Calculate the chunk belong to this process and send other chunk to other processes
        int *tempbuf = (int *)malloc(sizeof(int) * (chunk_size + 1));
        bool init_recv = false;
        if (rank != 0) {
            for (int i = 1; i < size; i++) {
                remote_start = i * chunk_size + (i < remainder ? i : remainder);
                remote_count = chunk_size + (i < remainder ? 1 : 0);

                if (i != rank) {
                    MPI_Send(sendbuf + remote_start, remote_count, MPI_INT, i, 0, MPI_COMM_WORLD);
                } else {
                    for (int j = 1; j < size; j++) {
                        if (j == rank) continue;
                        MPI_Recv(tempbuf, remote_count, MPI_INT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        if (!init_recv) {
                            for (int k = 0; k < local_count; k++) {
                                recvbuf[local_start + k] = sendbuf[local_start + k] + tempbuf[k];
                            }
                            init_recv = true;
                        } else {
                            for (int k = 0; k < local_count; k++) {
                                recvbuf[local_start + k] += tempbuf[k];
                            }
                        }
                    }
                }
            }
        }

        // Send the result back to process 0
        if (rank != 0) {
            MPI_Send(sendbuf, zero_count, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(recvbuf + local_start, local_count, MPI_INT, 0, 0, MPI_COMM_WORLD);
        } else {
            MPI_Recv(tempbuf, zero_count, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < zero_count; i++) {
                recvbuf[i] = tempbuf[i] + sendbuf[i];
            }
            for (int i = 2; i < size; i++) {
                MPI_Recv(tempbuf, zero_count, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int j = 0; j < zero_count; j++) {
                    recvbuf[j] += tempbuf[j];
                }
            }
            for (int i = 1; i < size; i++) {
                remote_start = i * chunk_size + (i < remainder ? i : remainder);
                remote_count = chunk_size + (i < remainder ? 1 : 0);
                MPI_Recv(tempbuf, remote_count, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int j = 0; j < remote_count; j++) {
                    recvbuf[remote_start + j] = tempbuf[j] + sendbuf[remote_start + j];
                }
            }
        }
        free(tempbuf);

    #endif
#endif

}
