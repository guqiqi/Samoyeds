// #include <stdio>
#include <random>
#include <iostream>
#include <cuda_fp16.h>
//#include <half/half.hpp>

#define SPTC_M 4
#define SPTC_N 2
#define METADATA_SIZE 2
#define BITS_PER_BYTE 8  // 1 Byte = 8 Bits
#define NUM_OF_META_PER_UINT static_cast<int>(sizeof(uint) * BITS_PER_BYTE / METADATA_SIZE)

void create_indices_ordered(uint *A_indices,
                            int A_rows,
                            int A_cols,
                            int vector_length,
                            int structure_N,
                            int structure_M) {
    int indices_rows = A_rows / structure_M * structure_N;
    int indices_cols = A_cols / vector_length;

    for (int iter_cols = 0; iter_cols < indices_cols; iter_cols++) {
        for (int iter_rows = 0; iter_rows < indices_rows; iter_rows += structure_M) {
            for (int i = 0; i < structure_M; i++) {
                A_indices[(iter_rows + i) * indices_cols + iter_cols] = 1;
            }
        }
    }
}

void create_indices_specific_pattern(uint *A_indices,
                            int A_rows,
                            int A_cols,
                            int vector_length,
                            int structure_N,
                            int structure_M) {
    int indices_rows = A_rows / structure_M * structure_N;
    int indices_cols = A_cols / vector_length;

    for (int iter_cols = 0; iter_cols < indices_cols; iter_cols++) {
        for (int iter_rows = 0; iter_rows < indices_rows; iter_rows += structure_M) {
            for (int i = 0; i < structure_M; i++) {
                A_indices[(iter_rows + i) * indices_cols + iter_cols] = (iter_rows + i) % 2;
            }
        }
    }
    // printf("A_indices: rows: %d, cols: %d, A_rows: %d, A_cols: %d, N: %d, M: %d.\n", indices_rows, indices_cols, A_rows, A_cols, structure_N, structure_M);
    // for (int iter_rows = 0; iter_rows < indices_rows; iter_rows++) {
    //     for (int iter_cols = 0; iter_cols < indices_cols; iter_cols++) {
    //         printf("%u ", A_indices[iter_rows * indices_cols + iter_cols]);
    //     }
    //     printf("\n");
    // }
}

void create_indices_randomly(uint *A_indices,
                             int A_rows,
                             int A_cols,
                             int vector_length,
                             int structure_N,
                             int structure_M) {
    int indices_rows = A_rows / structure_M * structure_N;
    int indices_cols = A_cols / vector_length;

    std::vector <uint> temp;
    temp.reserve(structure_M);
    for (int i = 0; i < structure_M; i++) {
        temp.push_back(i);
    }

    for (int iter_cols = 0; iter_cols < indices_cols; iter_cols++) {
        for (int iter_rows = 0; iter_rows < indices_rows; iter_rows += structure_N) {
            random_shuffle(temp.begin(), temp.end());
            std::sort(temp.begin(), temp.begin() + structure_N);
            for (int i = 0; i < structure_N; i++) {
                A_indices[(iter_rows + i) * indices_cols + iter_cols] = temp[i];
            }
        }
    }
}

void print_indices(uint *A_indices,
                   int A_rows,
                   int A_cols,
                   int vector_length,
                   int structure_N,
                   int structure_M) {
    int indices_rows = A_rows / structure_M * structure_N;
    int indices_cols = A_cols / vector_length;
    printf("A_indices:\n");
    for (int iter_rows = 0; iter_rows < indices_rows; iter_rows++) {
        for (int iter_cols = 0; iter_cols < indices_cols; iter_cols++) {
            printf("%u ", A_indices[iter_rows * indices_cols + iter_cols]);
        }
        printf("\n");
    }
}


void create_metadata_ordered(uint *A_metadata,
                             int A_rows,
                             int A_cols,
                             int vector_length,
                             int structure_N,
                             int structure_M) {
    int metadata_rows = A_rows / structure_M * structure_N;
    int metadata_cols = A_cols / SPTC_M * SPTC_N / NUM_OF_META_PER_UINT;

    for (int iter_rows = 0; iter_rows < metadata_rows; ++iter_rows) {
        for (int iter_cols = 0; iter_cols < metadata_cols; ++iter_cols) {
            uint temp_meta = 0;
            for (int i = 0; i < NUM_OF_META_PER_UINT; i += SPTC_N) {
                for (int j = 0; j < SPTC_N; ++j) {
                    temp_meta = temp_meta | j << ((i + j) * METADATA_SIZE);
                }
            }
            A_metadata[iter_rows * metadata_cols + iter_cols] = temp_meta;
        }
    }
}

void create_metadata_randomly(uint *A_metadata,
                              int A_rows,
                              int A_cols,
                              int vector_length,
                              int structure_N,
                              int structure_M) {
    int metadata_rows = A_rows / structure_M * structure_N;
    int metadata_cols = A_cols / SPTC_M * SPTC_N / NUM_OF_META_PER_UINT;

    std::vector <uint> temp;
    temp.reserve(SPTC_M);
    for (int i = 0; i < SPTC_M; i++) {
        temp.push_back(i);
    }

    for (int iter_rows = 0; iter_rows < metadata_rows; ++iter_rows) {
        for (int iter_cols = 0; iter_cols < metadata_cols; ++iter_cols) {
            uint temp_meta = 0;
            for (int i = 0; i < NUM_OF_META_PER_UINT; i += SPTC_N) {
                random_shuffle(temp.begin(), temp.end());
                std::sort(temp.begin(), temp.begin() + SPTC_N);
                for (int j = 0; j < SPTC_N; ++j) {
                    temp_meta = temp_meta | temp[j] << ((i + j) * METADATA_SIZE);
                }
            }
            A_metadata[iter_rows * metadata_cols + iter_cols] = temp_meta;
        }
    }
}

void print_metadata(uint *A_metadata,
                    int A_rows,
                    int A_cols,
                    int vector_length,
                    int structure_N,
                    int structure_M) {
    int metadata_rows = A_rows / structure_M * structure_N;
    int metadata_cols = A_cols / SPTC_M * SPTC_N / NUM_OF_META_PER_UINT;
    printf("A_metadata:\n");
    for (int iter_rows = 0; iter_rows < metadata_rows; iter_rows++) {
        for (int iter_cols = 0; iter_cols < metadata_cols; iter_cols++) {
            printf("%x ", A_metadata[iter_rows * metadata_cols + iter_cols]);
        }
        printf("\n");
    }
}

void create_value(half *A_data,
                  half *A_value,
                  uint *A_indices,
                  uint *A_metadata,
                  int A_rows,
                  int A_cols,
                  int vector_length,
                  int structure_N,
                  int structure_M) {
    int indices_rows = A_rows / structure_M * structure_N;
    int indices_cols = A_cols / vector_length;
    int metadata_rows = A_rows / structure_M * structure_N;
    int metadata_cols = A_cols / SPTC_M * SPTC_N / NUM_OF_META_PER_UINT;
    int value_rows = A_rows / structure_M * structure_N;
    int value_cols = A_cols / SPTC_M * SPTC_N;

    for (int iter_row = 0; iter_row < value_rows; ++iter_row) {
        for (int iter_col = 0; iter_col < value_cols; iter_col += SPTC_N) {
            int indices_row_idx = iter_row;
            int indices_col_idx = iter_col / (vector_length / SPTC_M * SPTC_N);
            int selected_index = A_indices[indices_row_idx * indices_cols + indices_col_idx];
            int row_offset = iter_row / structure_N * structure_M + selected_index;

            int metadata_row_idx = iter_row;
            int metadata_col_idx = iter_col / NUM_OF_META_PER_UINT;
            int metadata_inside_idx =
                    iter_col % NUM_OF_META_PER_UINT; //每个uint包含NUM_OF_META_PER_UINT(16)个metadata，计算需要取到第几位
            int mask = ((0b1 << (SPTC_N * METADATA_SIZE)) - 1) << (metadata_inside_idx * METADATA_SIZE);
            uint selected_metadata = (A_metadata[metadata_row_idx * metadata_cols + metadata_col_idx] & mask)
                    >> (metadata_inside_idx * METADATA_SIZE);
            int selected_metadata_entry[SPTC_N];
            int temp = static_cast<int>(selected_metadata);
            int divider = 0b1 << METADATA_SIZE;
            for (int i = 0; i < SPTC_N; ++i) {
                selected_metadata_entry[i] = temp % divider;
                temp /= divider;
            }
            for (int i = 0; i < SPTC_N; ++i) {
                int des = iter_row * value_cols + iter_col + i;
                int index = row_offset * A_cols + iter_col / SPTC_N * SPTC_M + selected_metadata_entry[i];
                A_value[iter_row * value_cols + iter_col + i] =
                        A_data[index];
            }
        }
    }
}

void print_value(half *A_data,
                half *A_value,
                uint *A_indices,
                uint *A_metadata,
                int A_rows,
                int A_cols,
                int vector_length,
                int structure_N,
                int structure_M) {
   int value_rows = A_rows / structure_M * structure_N;
   int value_cols = A_cols / SPTC_M * SPTC_N;
   printf("condense value:\n");
   for (int iter_row = 0; iter_row < value_rows; ++iter_row) {
       for (int iter_col = 0; iter_col < value_cols; ++iter_col) {
           printf("%d ", static_cast<int>(A_data[iter_row * value_cols + iter_col]));
       }
       printf("\n");
   }
}

void create_pruned_value(half *A_pruned_value,
                         half *A_value,
                         uint *A_indices,
                         uint *A_metadata,
                         int A_rows,
                         int A_cols,
                         int vector_length,
                         int structure_N,
                         int structure_M) {
    int indices_rows = A_rows / structure_M * structure_N;
    int indices_cols = A_cols / vector_length;
    int metadata_rows = A_rows / structure_M * structure_N;
    int metadata_cols = A_cols / SPTC_M * SPTC_N / NUM_OF_META_PER_UINT;
    int value_rows = A_rows / structure_M * structure_N;
    int value_cols = A_cols / SPTC_M * SPTC_N;
    int pruned_rows = A_rows;
    int pruned_cols = A_cols;

    for (int iter_row = 0; iter_row < value_rows; ++iter_row) {
        for (int iter_col = 0; iter_col < value_cols; iter_col += structure_N) {
            A_pruned_value[iter_row * pruned_cols + iter_col] = 0;
        }
    }

    for (int iter_row = 0; iter_row < value_rows; ++iter_row) {
        for (int iter_col = 0; iter_col < value_cols; iter_col += SPTC_N) {
            int indices_row_idx = iter_row;
            int indices_col_idx = iter_col / (vector_length / SPTC_M * SPTC_N);
            int selected_index = A_indices[indices_row_idx * indices_cols + indices_col_idx];
            int row_offset = iter_row / structure_N * structure_M + selected_index;

            int metadata_row_idx = iter_row;
            int metadata_col_idx = iter_col / NUM_OF_META_PER_UINT;
            int metadata_inside_idx =
                    iter_col % NUM_OF_META_PER_UINT; //每个uint包含NUM_OF_META_PER_UINT(16)个metadata，计算需要取到第几位
            int mask = ((0b1 << (SPTC_N * METADATA_SIZE)) - 1) << (metadata_inside_idx * METADATA_SIZE);
            uint selected_metadata = (A_metadata[metadata_row_idx * metadata_cols + metadata_col_idx] & mask)
                    >> (metadata_inside_idx * METADATA_SIZE);
            int selected_metadata_entry[SPTC_N];
            int temp = static_cast<int>(selected_metadata);
            int divider = 0b1 << METADATA_SIZE;
            for (int i = 0; i < SPTC_N; ++i) {
                selected_metadata_entry[i] = temp % divider;
                temp /= divider;
            }
            for (int i = 0; i < SPTC_N; ++i) {
                int index = row_offset * pruned_cols + iter_col / SPTC_N * SPTC_M + selected_metadata_entry[i];
                A_pruned_value[index] = A_value[iter_row * value_cols + iter_col + i];
            }
        }
    }
}

// 返回值需要送到kernel中，column_major set as true
void transpose_indices(uint *A_indices,
                       int A_rows,
                       int A_cols,
                       int vector_length,
                       int structure_N,
                       int structure_M,
                       bool column_major) {
    int indices_rows, indices_cols;
    if (column_major) {
        indices_rows = A_rows / structure_M * structure_N;
        indices_cols = A_cols / vector_length;
    } else {
        indices_rows = A_cols / vector_length;
        indices_cols = A_rows / structure_M * structure_N;
    }
    std::vector <uint> temp;
    temp.resize(indices_rows * indices_cols);
    for (int iter_cols = 0; iter_cols < indices_cols; iter_cols++) {
        for (int iter_rows = 0; iter_rows < indices_rows; iter_rows++) {
            temp[iter_cols * indices_rows + iter_rows] = A_indices[iter_rows * indices_cols + iter_cols];
        }
    }
    for (int iter_cols = 0; iter_cols < indices_cols; iter_cols++) {
        for (int iter_rows = 0; iter_rows < indices_rows; iter_rows++) {
            A_indices[iter_rows * indices_cols + iter_cols] = temp[iter_rows * indices_cols + iter_cols];
        }
    }
}

void sparsifier_c_impl(half *A_data,
                       half *A_value,
                       uint *A_indices,
                       uint *A_metadata,
                       int A_rows,
                       int A_cols,
                       int vector_length,
                       int structure_N,
                       int structure_M) {
    // create_indices_ordered(A_indices, A_rows, A_cols, vector_length, structure_N, structure_M);
    create_indices_randomly(A_indices, A_rows, A_cols, vector_length, structure_N, structure_M);

    // create_metadata_ordered(A_metadata, A_rows, A_cols, vector_length, structure_N, structure_M);
    create_metadata_randomly(A_metadata, A_rows, A_cols, vector_length, structure_N, structure_M);

    create_value(A_data, A_value, A_indices, A_metadata, A_rows, A_cols, vector_length, structure_N, structure_M);

    transpose_indices(A_indices, A_rows, A_cols, vector_length, structure_N, structure_M, true);
}

void pruned_A_c_impl(half *A_pruned_value,
                     half *A_value,
                     uint *A_indices,
                     uint *A_metadata,
                     int A_rows,
                     int A_cols,
                     int vector_length,
                     int structure_N,
                     int structure_M) {
    transpose_indices(A_indices, A_rows, A_cols, vector_length, structure_N, structure_M, false);
    create_pruned_value(A_pruned_value, A_value, A_indices, A_metadata, A_rows, A_cols, vector_length, structure_N,
                        structure_M);
    transpose_indices(A_indices, A_rows, A_cols, vector_length, structure_N, structure_M, true);
}