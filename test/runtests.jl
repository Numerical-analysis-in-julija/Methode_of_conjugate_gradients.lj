using Test
using Methode_of_conjugate_gradients

# Test 1: ScatteredArray multiplication
A = Methode_of_conjugate_gradients.ScatteredArray([3.0 0.0; 0.0 2.0], [1 0; 0 2])
x = [1.0, 2.0]
@test A * x == [3.0, 4.0]

# Test 2: Conjugate Gradient Method
A_dense = [4.0 1.0; 1.0 3.0]
b = [1.0; 1.0]
x0 = [0.0; 0.0]
A_scattered = Methode_of_conjugate_gradients.ScatteredArray([4.0 1.0; 1.0 3.0], [1 2; 1 2])
x_sol, num_iter, residuals = Methode_of_conjugate_gradients.conj_grad(A_scattered, b, x0=x0)
@test x_sol ≈ [0.18181818181818182, 0.2727272727272727] atol=1e-6

# Test 3: Conjugate Gradient Method with more iterations
A_dense_large = [4.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 2.0]
b_large = [1.0; 1.0; 1.0]
x0_large = [0.0; 0.0; 0.0]
A_large = Methode_of_conjugate_gradients.ScatteredArray([4.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 2.0], [1 2 0; 1 2 3; 0 2 3])
x_sol_large, num_iter_large, residuals_large = Methode_of_conjugate_gradients.conj_grad(A_large, b_large, x0=x0_large)
@test x_sol_large ≈ [0.22222222222222227, 0.11111111111111109, 0.44444444444444453] atol=1e-6