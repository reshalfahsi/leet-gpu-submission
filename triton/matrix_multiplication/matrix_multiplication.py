import triton
import triton.language as tl


# This is the core GPU kernel function written in Triton
@triton.jit
def matrix_multiplication_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_an,
    stride_bn, stride_bk,
    stride_cm, stride_ck,
    # Meta-parameters for block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # We can make these pointers type-safe here (optional but good practice)
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    # -----------------------------------------------------------
    # 1. Indices
    #
    # Map the program ID (block ID) to the block's starting row/column
    # in the C matrix.
    pid_m = tl.program_id(axis=0) # Program ID for the M dimension
    pid_k = tl.program_id(axis=1) # Program ID for the K dimension

    # Calculate the starting row and column indices for the current block in C
    m = pid_m * BLOCK_SIZE_M
    k = pid_k * BLOCK_SIZE_K

    # Initialize the accumulator for the block of C
    # This will store the sum of matrix products for this block
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    # -----------------------------------------------------------
    # 2. Iterate over the N dimension
    #
    # To compute C[m:m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K], we need to
    # multiply A[m:m+BLOCK_SIZE_M, n:n+BLOCK_SIZE_N] with
    # B[n:n+BLOCK_SIZE_N, k:k+BLOCK_SIZE_K] and sum them up,
    # where n goes from 0 to N in steps of BLOCK_SIZE_N.
    #
    # The loop runs N / BLOCK_SIZE_N iterations.
    num_n_blocks = tl.cdiv(N, BLOCK_SIZE_N) # Number of blocks along the N dimension

    for pid_n in range(num_n_blocks):
        # Calculate the starting column index for the current block in A
        # and the starting row index for the current block in B
        n = pid_n * BLOCK_SIZE_N

        # -----------------------------------------------------------
        # 3. Load tiles from A and B
        #
        # Generate offsets for loading the current tiles from A and B.
        # We need BLOCK_SIZE_M x BLOCK_SIZE_N from A and
        # BLOCK_SIZE_N x BLOCK_SIZE_K from B.

        # Offsets for the M and N dimensions within the block for A
        offs_am = m + tl.arange(0, BLOCK_SIZE_M)
        offs_an = n + tl.arange(0, BLOCK_SIZE_N)
        # Combine offsets to get the full pointer offsets for A
        # Use broadcasting [:, None] and [None, :] to create a 2D grid of pointers
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_an[None, :] * stride_an)

        # Create a mask for A to handle boundary conditions (if the matrix dimensions are not
        # exact multiples of the block sizes)
        mask_a = (offs_am[:, None] < M) & (offs_an[None, :] < N)

        # Load the tile from A, treating out-of-bounds elements as zero
        a_tile = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # Offsets for the N and K dimensions within the block for B
        offs_bn = n + tl.arange(0, BLOCK_SIZE_N)
        offs_bk = k + tl.arange(0, BLOCK_SIZE_K)
        # Combine offsets to get the full pointer offsets for B
        b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk)

        # Create a mask for B
        mask_b = (offs_bn[:, None] < N) & (offs_bk[None, :] < K)

        # Load the tile from B
        b_tile = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # -----------------------------------------------------------
        # 4. Perform matrix multiplication on the loaded tiles
        #
        # Accumulate the result in 'acc'.
        # Triton's tl.dot performs the matrix multiplication of two tiles.
        acc += tl.dot(a_tile, b_tile, input_precision="ieee")

    # -----------------------------------------------------------
    # 5. Store the accumulated result tile into the output matrix C
    #
    # Generate offsets for storing the C tile.
    offs_cm = m + tl.arange(0, BLOCK_SIZE_M)
    offs_ck = k + tl.arange(0, BLOCK_SIZE_K)
    # Combine offsets for C
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_ck[None, :] * stride_ck)

    # Create a mask for C to handle boundary conditions
    mask_c = (offs_cm[:, None] < M) & (offs_ck[None, :] < K)

    # Store the accumulated result 'acc' into the C matrix
    tl.store(c_ptrs, acc, mask=mask_c)


# This is the wrapper function that launches the kernel
def solve(a_ptr: int, b_ptr: int, c_ptr: int, M: int, N: int, K: int):
    # Assuming row-major layout, strides are simply the dimensions
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1

    # Define block sizes. These are meta-parameters that can be tuned
    # for performance on a specific GPU.
    # Typical values are powers of 2.
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 32 # This dimension is iterated over in the kernel
    BLOCK_SIZE_K = 64

    # The grid represents the number of blocks to launch.
    # We need one block for each BLOCK_SIZE_M x BLOCK_SIZE_K tile in C.
    # The grid is (ceil(M / BLOCK_SIZE_M), ceil(K / BLOCK_SIZE_K)).
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(K, BLOCK_SIZE_K))

    # Launch the kernel
    matrix_multiplication_kernel[grid](
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_an,
        stride_bn, stride_bk,
        stride_cm, stride_ck,
        BLOCK_SIZE_M=BLOCK_SIZE_M, # Pass meta-parameters to the kernel
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
