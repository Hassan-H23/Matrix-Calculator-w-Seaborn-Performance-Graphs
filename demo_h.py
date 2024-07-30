import random
from fractions import Fraction
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import timeit
import copy
import numpy as np

det_pos = 0
det_neg = 0
det_zero = 0
invertible_matrix_count = 0
non_invertible_matrix_count = 0
ref_time = []
rref_time = []
inverse_time = []
ref_eros1_count_list = []
ref_eros2_count_list = []
ref_eros3_count_list = []
rref_eros1_count_list = []
rref_eros2_count_list = []
rref_eros3_count_list = []


# ---------------------------------------------------------------------------------------------------------
# Function to visualize matrices using heatmap
def visualize_matrix(input_matrix, title="Matrix"):
    matrix = copy.deepcopy(input_matrix)
    df = pd.DataFrame(matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt=".1f", cmap="viridis")
    plt.title(title)
    plt.show()

# Function to visualize matrix element distributions
def visualize_element_distribution(input_matrix, title="Element Distribution"):
    matrix = copy.deepcopy(input_matrix)
    elements = [element for row in matrix for element in row]
    df = pd.DataFrame({"Elements": elements})
    sns.histplot(df["Elements"], kde=True)
    plt.title(title)
    plt.show()


# Function to highlight pivot elements during REF and RREF
def visualize_pivot_elements(matrix, pivots, title="Pivot Elements"):
    df = pd.DataFrame(matrix)
    for pivot in pivots:
        df.iloc[pivot] = df.iloc[pivot].apply(lambda x: f"**{x}**")
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=df, fmt="", cmap="viridis")
    plt.title(title)
    plt.show()


# Function to visualize the structure of matrices using clustermap
def visualize_clustermap(matrix, title="Clustermap"):
    df = pd.DataFrame(matrix)
    sns.clustermap(df, annot=True, cmap="viridis")
    plt.title(title)
    plt.show()


# Function to compare matrices before and after EROs
def compare_ero(matrix_before, matrix_after, title="Matrix Comparison"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(pd.DataFrame(matrix_before), annot=True, fmt=".1f", cmap="viridis", ax=axes[0])
    sns.heatmap(pd.DataFrame(matrix_after), annot=True, fmt=".1f", cmap="viridis", ax=axes[1])
    axes[0].set_title("Before")
    axes[1].set_title("After")
    plt.suptitle(title)
    plt.show()


# Function to show correlations between columns of a matrix
def visualize_correlations(matrix, title="Correlation Matrix"):
    df = pd.DataFrame(matrix)
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title(title)
    plt.show()

#----------------------REF & RREF Visualization Function-------------------------
def visualize_eros(type1_eros_counts,title,title2):
    # Create a list of matrix indices (assuming indices start from 1)
    matrix_indices = list(range(1, len(type1_eros_counts) + 1))

    # Plotting with Seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=matrix_indices, y=type1_eros_counts, palette="viridis")
    plt.title(title)
    plt.xlabel('Matrix Index')
    plt.ylabel(title2)
    plt.show()



def rref_visualize_type1_eros(type1_eros_counts):
    # Create a list of matrix indices (assuming indices start from 1)
    matrix_indices = list(range(1, len(type1_eros_counts) + 1))

    # Plotting with Seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=matrix_indices, y=type1_eros_counts, palette="viridis")
    plt.title('Number of Type 1 "Eros" in RREF Function For each Matrix')
    plt.xlabel('Matrix Index')
    plt.ylabel('Number of Type 1 "Eros"')
    plt.show()


def rref_visualize_type2_eros(type2_eros_counts):
    # Create a list of matrix indices (assuming indices start from 1)
    matrix_indices = list(range(1, len(type2_eros_counts) + 1))

    # Plotting with Seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=matrix_indices, y=type2_eros_counts, palette="viridis")
    plt.title('Number of Type 2 "Eros" in RREF Function For each Matrix')
    plt.xlabel('Matrix Index')
    plt.ylabel('Number of Type 2 "Eros"')
    plt.show()


def rref_visualize_type3_eros(type3_eros_counts):
    # Create a list of matrix indices (assuming indices start from 1)
    matrix_indices = list(range(1, len(type3_eros_counts) + 1))

    # Plotting with Seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=matrix_indices, y=type3_eros_counts, palette="viridis")
    plt.title('Number of Type 3 "Eros" in RREF Function For each Matrix')
    plt.xlabel('Matrix Index')
    plt.ylabel('Number of Type 3 "Eros"')
    plt.show()


#-------------------------------REF Visualization---------------------------
def ref_visualize_type3_eros(type3_eros_counts):
    # Create a list of matrix indices (assuming indices start from 1)
    matrix_indices = list(range(1, len(type3_eros_counts) + 1))

    # Plotting with Seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=matrix_indices, y=type3_eros_counts, palette="viridis")
    plt.title('Number of Type 3"Eros" in REF Function For each Matrix')
    plt.xlabel('Matrix Index')
    plt.ylabel('Number of Type 3 "Eros"')
    plt.show()

#-------------------------------TIME Visualization---------------------------
def visualize_computation_time_scatter(time_taken, title):
    # Create a list of matrix indices (assuming indices start from 1)
    matrix_indices = list(range(1, len(time_taken) + 1))

    # Plotting with Seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=matrix_indices, y=time_taken, s=100, color='blue', alpha=0.8)
    plt.title(title)
    plt.xlabel('Matrix Index')
    plt.ylabel('Time Taken (seconds)')
    plt.xticks(matrix_indices)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------------------------------------------------------------------

def upper_triangular(input_matrix):
    # Iterate over each column except the last one
    matrix = copy.deepcopy(input_matrix)
    for col in range(4):
        # Find a row with a non-zero entry in the current column below the diagonal
        for row in range(col + 1, 4):
            if matrix[row][col] != 0:
                # Perform row operation to eliminate the entry below the diagonal
                # Calculate the multiplier to make the element below the diagonal zero
                multiplier = matrix[row][col] / matrix[col][col]
                for k in range(col, 4):
                    matrix[row][k] -= multiplier * matrix[col][k]

    return matrix


# Swapping two rows in the matrix
def type1_ERO(input_matrix, row1: int, row2: int):
    # Create a copy of input_matrix to avoid pass by reference issue
    new_matrix = [row[:] for row in input_matrix]
    new_matrix[row1], new_matrix[row2] = new_matrix[row2], new_matrix[row1]
    return new_matrix


def type2_ERO(input_matrix, row: int):
    new_matrix = [row[:] for row in input_matrix]
    pivot = new_matrix[row][row]
    if pivot != 0:
        scale_factor = 1 / pivot
        for i in range(4):
            new_matrix[row][i] *= scale_factor
    return new_matrix


def type3_ERO(input_matrix, row1_index: int, row2_index: int, multiplier: float):
    new_matrix = [row[:] for row in input_matrix]
    for i in range(4):
        new_matrix[row2_index][i] -= multiplier * new_matrix[row1_index][i]
    return new_matrix


# Function to print the matrix
def print_matrix(input_matrix):
    print(f"----------------------------------------------")
    for i in range(4):
        print("|", end='')
        for j in range(4):
            if isinstance(input_matrix[i][j], float):
                if input_matrix[i][j].is_integer():
                    print('    ', int(input_matrix[i][j]), end='    ')
                else:
                    print('    ', Fraction(input_matrix[i][j]).limit_denominator(), end='    ')
            else:
                print('    ', input_matrix[i][j], end='    ')
        print("|")
        print('\n')
    print("----------------------------------------------")


# Function that turns the matrix into RREF
def rref(input_matrix):
    matrix = copy.deepcopy(input_matrix)
    ero_count_1 = 0  # Initialize ERO count
    ero_count_2 = 0  # Initialize ERO count
    ero_count_3 = 0  # Initialize ERO count

    for col in range(4):
        pivot_row = None
        for row in range(col, 4):
            if matrix[row][col] != 0:
                pivot_row = row
                break

        if pivot_row is None:
            continue

        if pivot_row != col:
            matrix = type1_ERO(matrix, pivot_row, col)
            ero_count_1 += 1

        matrix = type2_ERO(matrix, col)
        ero_count_2 += 1

        for row in range(4):
            if row != col and matrix[row][col] != 0:
                multiplier = matrix[row][col]
                matrix = type3_ERO(matrix, col, row, multiplier)
                ero_count_3 += 1
    eros = []
    eros.append(ero_count_1)
    eros.append(ero_count_2)
    eros.append(ero_count_3)
    return matrix, eros


def ref(input_matrix):
    matrix = copy.deepcopy(input_matrix)
    ero_count_1 = 0
    ero_count_3 = 0
    for col in range(4):
        pivot_row = None
        for row in range(col, 4):
            if matrix[row][col] != 0:
                pivot_row = row
                break

        if pivot_row is None:
            continue

        if pivot_row != col:
            matrix = type1_ERO(matrix, pivot_row, col)
            ero_count_1 += 1  # Increment ERO count for Type 1 ERO

        for row in range(col + 1, 4):
            if matrix[row][col] != 0:
                multiplier = matrix[row][col] / matrix[col][col]
                matrix = type3_ERO(matrix, col, row, multiplier)
                ero_count_3 += 1  # Increment ERO count for Type 3 ERO

    eros = []
    eros.append(ero_count_1)
    eros.append(ero_count_3)
    return matrix, eros


def inverse(input_matrix):
    matrix = copy.deepcopy(input_matrix)
    augmented_matrix = [row + identity_row for row, identity_row in
                        zip(matrix, [[float(i == j) for i in range(4)] for j in range(4)])]

    for i in range(4):
        # Find the pivot row
        max_row = max(range(i, 4), key=lambda r: abs(augmented_matrix[r][i]))
        augmented_matrix[i], augmented_matrix[max_row] = augmented_matrix[max_row], augmented_matrix[i]

        pivot = augmented_matrix[i][i]
        augmented_matrix[i] = [element / pivot for element in augmented_matrix[i]]

        for j in range(4):
            if j != i:
                factor = augmented_matrix[j][i]
                augmented_matrix[j] = [augmented_matrix[j][k] - factor * augmented_matrix[i][k] for k in range(2 * 4)]

    # Extract the inverse matrix from the augmented matrix
    inverse_matrix = [row[4:] for row in augmented_matrix]

    return inverse_matrix


def determinant(input_matrix):
    # check if the matrix has a row of 0's or a col of 0's if yes then det(matrix) = 0
    # print("Matrix inside of det function")
    # print_matrix(input_matrix)
    matrix = copy.deepcopy(input_matrix)
    det = 1
    for row in matrix:
        if all(element == 0 for element in row):
            # print("this triggered")
            return 0

    for col in range(len(matrix[0])):
        if all(matrix[row][col] == 0 for row in range(len(matrix))):
            # print("this triggered")
            return 0

    # print("matrix in det function")
    # print_matrix(input_matrix)
    upper_matrix = upper_triangular(matrix)
    # print("Upper triangular matrix")
    # print_matrix(input_matrix)
    for row in range(4):
        det = det * upper_matrix[row][row]
        # print(det)
    return det


def generate_random_matrix():
    matrix_rand = []
    zero_row_probability = 0.2  # Probability of having a row filled with zeros
    zero_col_probability = 0.1  # Probability of having a column filled with zeros

    for i in range(4):
        row = []
        if random.random() < zero_row_probability:
            row = [0] * 4  # Fill the entire row with zeros
        else:
            for j in range(4):
                if random.random() < zero_col_probability:
                    row.append(0)  # Fill specific column with zero
                else:
                    num = random.randint(-20, 20)
                    row.append(num)
        matrix_rand.append(row)

    return matrix_rand
def show_all_matrices(matrix_list,title):
    # Determine the grid size for subplots
    n = len(matrix_list)
    cols = int(np.ceil(np.sqrt(n)))  # Number of columns
    rows = int(np.ceil(n / cols))  # Number of rows
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    fig.suptitle(title)

    # Flatten the axes array for easier iteration
    axes = axes.flatten()
    # Plot each matrix as a heatmap
    for i, matrix in enumerate(matrix_list):
        sns.heatmap(ax=axes[i], data=matrix, annot=True, fmt=".1f", cmap="viridis")
        axes[i].set_title(f'Matrix {i + 1}')
    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

# ------------Test----------------
try:

    matrix_list = []
    ref_matrix_list = []
    rref_matrix_list = []
    inverse_matrix_list = []
    for i in range(10):
        matrix = generate_random_matrix()
        print(f"Matrix {i + 1}")
        print_matrix(matrix)
       # visualize_matrix(matrix, "Original Matrix")
        matrix_list.append(matrix)


        #visualize_element_distribution(matrix, "Original Matrix Element Distribution")
        start = timeit.default_timer()

        rref_matrix, rref_eros = rref(matrix.copy())

        rref_time_temp = timeit.default_timer() - start

        rref_matrix_list.append(rref_matrix)
        rref_time.append(rref_time_temp)
        rref_eros1_count_list.append(rref_eros[0])
        rref_eros2_count_list.append(rref_eros[1])
        rref_eros3_count_list.append(rref_eros[2])
        start = timeit.default_timer()

        ref_matrix, ref_eros = ref(matrix.copy())

        ref_time_temp = timeit.default_timer() - start
        ref_matrix_list.append(ref_matrix)
        ref_time.append(ref_time_temp)
        ref_eros1_count_list.append(ref_eros[0])
        ref_eros3_count_list.append(ref_eros[1])
        print("Matrix in REF")
        print_matrix(ref_matrix)
       # visualize_matrix(ref_matrix, "Matrix in REF")
        print("Matrix in RREF")

        print_matrix(rref_matrix)

       # visualize_matrix(rref_matrix, "Matrix in RREF")

        det_matrix = round(determinant(matrix.copy()), 1)
        if det_matrix > 0:
            det_pos += 1
        if det_matrix < 0:
            det_neg += 1
        if det_matrix != 0:
            invertible_matrix_count += 1
            print(f"Determinant of Matrix : {det_matrix}")
            print("----------------------------------------------")
            print("Inverse of Matrix")
            start = timeit.default_timer()

            inverse_matrix = inverse(matrix)

            inverse_time_temp = timeit.default_timer() - start
            inverse_matrix_list.append(inverse_matrix)
            inverse_time.append(inverse_time_temp)
            print_matrix(inverse_matrix)
            #visualize_matrix(inverse_matrix, "Inverse Matrix")
        else:
            non_invertible_matrix_count += 1
            det_zero += 1
           # visualize_correlations(matrix, "Original Matrix Correlation")
            inverse_time.append(0.0)

    show_all_matrices(matrix_list, "Original Matrices")
    show_all_matrices(ref_matrix_list, "REF Matrices")
    show_all_matrices(rref_matrix_list, "RREF Matrices")
    show_all_matrices(inverse_matrix_list, "Invertible Matrices Only Shown")


    print("------------------------Invertible vs. Non-Invertible------------------------")
    print(f"Invertible Matrix Count : {invertible_matrix_count}")
    print(f"Non Invertible Matrix Count : {non_invertible_matrix_count}")
    print("--------------------------------REF ERO's Count--------------------------------")
    print(f"Type 1 ERO's Count : {ref_eros1_count_list}")
    print(f"Type 3 ERO's Count : {ref_eros3_count_list}")
    visualize_eros(ref_eros1_count_list, 'Number of Type 1 "Eros" in REF Function For each Matrix', "Number of Type 1 Eros" )
    visualize_eros(ref_eros3_count_list,'Number of Type 3 "Eros" in RREF Function For each Matrix', "Number of Type 3 Eros")
    print("--------------------------------RREF ERO's Count--------------------------------")
    print(f"Type 1 ERO's Count : {rref_eros1_count_list}")
    print(f"Type 2 ERO's Count : {rref_eros2_count_list}")
    print(f"Type 3 ERO's Count : {rref_eros3_count_list}")
    visualize_eros(rref_eros1_count_list, 'Number of Type 1 "Eros" in RREF Function For each Matrix', "Number of Type 1 Eros" )
    visualize_eros(rref_eros2_count_list,'Number of Type 2 "Eros" in RREF Function For each Matrix' ,"Number of Type 2 Eros")
    visualize_eros(rref_eros3_count_list,'Number of Type 3 "Eros" in RREF Function For each Matrix', "Number of Type 3 Eros")
    print("--------------------------------Time--------------------------------")
    print(f"Time Taken by REF Function : {ref_time}")
    print(f"Time Taken by RREF Function : {rref_time}")
    print(f"Time Taken by Inverse Function : {inverse_time}")
    visualize_computation_time_scatter(ref_time,"Computation Time for REF Function")
    visualize_computation_time_scatter(rref_time, "Computation Time for RREF Function")
    visualize_computation_time_scatter(inverse_time, "Computation Time for Inverse Function")
    print("--------------------------------Det Count--------------------------------")
    print(f"Positive Determinant: {det_pos}")
    print(f"Negative Determinant: {det_neg}")
    print(f"Zero Determinant: {det_zero}")





except FutureWarning:
    print("")