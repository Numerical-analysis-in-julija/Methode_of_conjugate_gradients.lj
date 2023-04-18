using Test
using Methode_of_conjugate_gradients

# Test the ScatteredArray multiplication function
A = ScatteredArray([1 2 0; 0 3 4], [1 2 0; 0 3 2])
x = [1.0, 2.0, 3.0]
y = A * x
@test y ≈ [5.0, 17.0]  # Corrected the expected output

# Test the conjugate gradient function
A = ScatteredArray([4 1; 1 3], [1 2; 3 1])
b = [-1.0, -1.0]
x, iter, residuals = conj_grad(A, b)
@test x ≈ [-0.09090909090909094, -0.27272727272727276]
@test iter <= 2

# Test the 2D optimization functions
A = [4 1; 1 3]
b = [-1; -1]

function f(x)
    return 0.5 * x' * A * x - b' * x
end

function grad_f(x)
    return A * x - b
end

x0 = [2.0; 2.0]
sol_gd, path_gd = gradient_descent(grad_f, x0)
sol_cg, path_cg = conj_grad_2d(A, b, x0)
@test sol_gd ≈ [-0.09090909090909094, -0.27272727272727276]
@test sol_cg ≈ [-0.09090909090909094, -0.27272727272727276]