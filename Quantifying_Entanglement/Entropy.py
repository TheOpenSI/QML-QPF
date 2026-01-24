import numpy as np

# --- Vectorized Helper Functions ---

def rho_qubit0_vectorized(theta1_arr, theta2_arr):
    """
    Computes the reduced density matrix for qubit 0 for an array of (theta1, theta2) pairs.
    theta1_arr, theta2_arr are 1D NumPy arrays of shape (N,).
    Returns an (N, 2, 2) array of density matrices.
    """
    # Ensure inputs are NumPy arrays for vectorized operations
    theta1_arr = np.asarray(theta1_arr)
    theta2_arr = np.asarray(theta2_arr)

    A = np.cos(theta1_arr / 2) * np.cos(theta2_arr / 2)
    B = np.cos(theta1_arr / 2) * np.sin(theta2_arr / 2)
    C = np.sin(theta1_arr / 2) * np.cos(theta2_arr / 2)
    D = np.sin(theta1_arr / 2) * np.sin(theta2_arr / 2)

    # Calculate elements for all N matrices
    rho0_00 = A*A + B*B
    rho0_01 = A*D + B*C
    rho0_11 = D*D + C*C

    # Stack the elements and transpose to get (N, 2, 2) shape
    rho0_matrices = np.array([
        [rho0_00, rho0_01],
        [rho0_01, rho0_11]
    ], dtype=float).transpose(2, 0, 1)

    return rho0_matrices

def rho_qubit1_vectorized(theta1_arr, theta2_arr):
    """
    Computes the reduced density matrix for qubit 1 for an array of (theta1, theta2) pairs.
    theta1_arr, theta2_arr are 1D NumPy arrays of shape (N,).
    Returns an (N, 2, 2) array of density matrices.
    """
    # Ensure inputs are NumPy arrays for vectorized operations
    theta1_arr = np.asarray(theta1_arr)
    theta2_arr = np.asarray(theta2_arr)

    A = np.cos(theta1_arr / 2) * np.cos(theta2_arr / 2)
    B = np.cos(theta1_arr / 2) * np.sin(theta2_arr / 2)
    C = np.sin(theta1_arr / 2) * np.cos(theta2_arr / 2)
    D = np.sin(theta1_arr / 2) * np.sin(theta2_arr / 2)

    # Calculate elements for all N matrices
    rho1_00 = A*A + D*D
    rho1_01 = A*B + D*C
    rho1_11 = B*B + C*C

    # Stack the elements and transpose to get (N, 2, 2) shape
    rho1_matrices = np.array([
        [rho1_00, rho1_01],
        [rho1_01, rho1_11]
    ], dtype=float).transpose(2, 0, 1)

    return rho1_matrices

def von_neumann_entropy_vectorized(rho_matrices):
    """
    Computes the Von Neumann entropy for a stack of density matrices.
    rho_matrices is an (N, 2, 2) array of density matrices.
    Returns a 1D NumPy array of entropies of shape (N,).
    """
    # Compute eigenvalues for all N matrices simultaneously
    # eigvalsh returns (N, 2) for (N, 2, 2) input
    eigvals = np.linalg.eigvalsh(rho_matrices)

    # Avoid log(0) by replacing very small eigenvalues with a tiny positive number
    # This applies to all elements in the eigvals array.
    eigvals[eigvals < 1e-12] = 1e-12

    # Calculate -p * log2(p) for all eigenvalues
    # Then sum along the last axis (axis=1) to get N entropy values
    entropies = -np.sum(eigvals * np.log2(eigvals), axis=1)
    return entropies

# --- Main Entropy Calculation Function (Vectorized) ---

def Total_entropy_vectorized(input_array, qubit):
    """
    Computes the entropy for each pair in the quantum input using vectorized operations.
    input_array: an array of (N, 2), where column 0 = qubit 0 data, column 1 = qubit 1 data.
                 Expected to be a NumPy array.
    qubit: 0 or 1.
           0 => compute reduced density matrix for qubit 0.
           1 => compute reduced density matrix for qubit 1.
    Return: 1D array of entropies of each pair in the input (shape N,).
    """
    # Ensure input_array is a NumPy array for consistent slicing
    input_array = np.asarray(input_array)

    theta1_arr = input_array[:, 0]
    theta2_arr = input_array[:, 1]

    if qubit == 0:
        rho_matrices = rho_qubit0_vectorized(theta1_arr, theta2_arr)
    elif qubit == 1:
        rho_matrices = rho_qubit1_vectorized(theta1_arr, theta2_arr)
    else:
        raise ValueError("Qubit must be 0 or 1")

    # Compute Von Neumann entropy for all matrices at once
    entropies = von_neumann_entropy_vectorized(rho_matrices)
    return entropies

# --- Function for ProcessPoolExecutor ---

def compute_entropy_for_pair_batch(inputs_tuple):
    """
    Computes entropy metrics for a single batch of input systems.
    This function will be executed by ProcessPoolExecutor workers.
    """
    
    # print("Im inside compute_entropy_for_pair_batch")
    # print("len of task inputs0 is ", len(inputs_tuple[0]) )
    
    input1_batch, input2_batch = inputs_tuple

    # print input batch to see
    # print(input1_batch.shape)
    # print("Im here")

    # Call the vectorized Total_entropy function
    S1_rho_0_batch = Total_entropy_vectorized(input1_batch, qubit=0)
    S1_rho_1_batch = Total_entropy_vectorized(input1_batch, qubit=1)
    S2_rho_0_batch = Total_entropy_vectorized(input2_batch, qubit=0)
    S2_rho_1_batch = Total_entropy_vectorized(input2_batch, qubit=1)

    # Calculate averages (these will be single scalar values per batch)
    Average_S1_rho_0 = np.average(S1_rho_0_batch)
    Average_S1_rho_1 = np.average(S1_rho_1_batch)
    Average_S2_rho_0 = np.average(S2_rho_0_batch)
    Average_S2_rho_1 = np.average(S2_rho_1_batch)

    # Calculate maximums (these will be single scalar values per batch)
    Max_S1_rho_0 = np.max(S1_rho_0_batch)
    Max_S1_rho_1 = np.max(S1_rho_1_batch)
    Max_S2_rho_0 = np.max(S2_rho_0_batch)
    Max_S2_rho_1 = np.max(S2_rho_1_batch)

    
    # Return all results as a tuple of tuples
    return (
        (Average_S1_rho_0, Average_S1_rho_1),
        (Average_S2_rho_0, Average_S2_rho_1),
        (Max_S1_rho_0, Max_S1_rho_1),
        (Max_S2_rho_0, Max_S2_rho_1),
        (S1_rho_0_batch),
        (S1_rho_1_batch),
        (S2_rho_0_batch),
        (S2_rho_1_batch)
    )

