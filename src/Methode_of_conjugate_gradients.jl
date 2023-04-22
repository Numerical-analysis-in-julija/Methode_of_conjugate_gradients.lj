module Methode_of_conjugate_gradients

using Plots
using LinearAlgebra

""" 
The ScatteredArray struct is used to represent a sparse matrix.
it takes two arguments:
    V: a matrix of values
    I: a matrix of indices
"""
struct ScatteredArray
    V::Matrix{Float64}
    I::Matrix{Int}
end

# Multiplication function for ScatteredArray
"""
this is a method definition in Julia for the 
element-wise multiplication between a ScatteredArray and a Vector.
"""
function Base.:*(A::ScatteredArray, x::Vector{Float64})
    result = zeros(size(A.I, 1))

    for row in 1:size(A.I, 1)
        for col in 1:size(A.I, 2)
            i = A.I[row, col]
            if i != 0 && i <= length(x)
                result[row] += A.V[row, col] * x[i]
            end
        end
    end

    return result
end


# Get the size of ScatteredArray
Base.size(A::ScatteredArray, dim::Integer) = size(A.I, dim)

function create_scattered_system_matrix(adj_matrix::Matrix{Int}, st::Vector{Float64})
    n = size(adj_matrix, 1)
    A = zeros(Float64, n, n)

    for i in 1:n
        A[i, i] = st[i] * length(findall(adj_matrix[i, :] .!= 0)) + 1
        for j in 1:n
            if adj_matrix[i, j] == 1
                A[i, j] = -1
            end
        end
    end
    
    # Add a small diagonal perturbation
    A = A + 1e-6 * Matrix{Float64}(LinearAlgebra.I, n, n)

    V = zeros(Float64, n, n)
    I = zeros(Int, n, n)

    for i in 1:n
        k = 1
        for j in 1:n
            if A[i, j] != 0
                V[i, k] = A[i, j]
                I[i, k] = j
                k += 1
            end
        end
    end

    return ScatteredArray(V, I)
end

""" Methode_of_conjugate_gradients.conj_grad works with ScatteredArray
    and Vector{Float64} as input.
    It returns the solution of the linear system Ax = b
    and the number of iterations.
"""
function conj_grad(A::ScatteredArray, b::Vector{Float64}; x0=nothing, tol=1e-6, max_iter=1000)
    if x0 === nothing
        x = zeros(Float64, size(A, 1))
    else
        x = x0
    end

    r = b - A * x
    p = copy(r)
    rsold = dot(r, r)

    residuals = Float64[]
    
    i = 1

    for i in 1:max_iter
        Ap = A * p
        alpha = rsold / dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = dot(r, r)
        push!(residuals, sqrt(rsnew))

        if sqrt(rsnew) < tol
            break
        end

        p = r + (rsnew / rsold) * p
        rsold = rsnew
    end

    return x,i , residuals
end

# Visualisation of the convergence of the conjugate gradient method

A = [4 1; 1 3]
b = [-1; -1]

function f(x)
    return 0.5 * x' * A * x - b' * x
end

function grad_f(x)
    return A * x - b
end

function gradient_descent(grad_f, x0, max_iter=1000, tol=1e-6, lr=0.1)
    x = x0
    path = [x0]

    for i in 1:max_iter
        g = grad_f(x)
        x = x - lr * g
        push!(path, x)

        if norm(grad_f(x)) < tol
            break
        end
    end

    return x, path
end

function conj_grad_2d(A, b, x0, max_iter=1000, tol=1e-6)
    x = x0
    r = b - A * x
    p = copy(r)

    path = [x0]

    for i in 1:max_iter
        alpha = dot(r, r) / dot(p, A * p)
        x = x + alpha * p
        push!(path, x)

        r_new = r - alpha * A * p

        if norm(r_new) < tol
            break
        end

        beta = dot(r_new, r_new) / dot(r, r)
        p = r_new + beta * p
        r = r_new
    end

    return x, path
end

x0 = [2.0; 2.0]
sol_gd, path_gd = gradient_descent(grad_f, x0)
sol_cg, path_cg = conj_grad_2d(A, b, x0)

x = -1:0.1:3
y = -1:0.1:3
contour_plot = Plots.contour(x, y, (x, y) -> f([x; y]), title="Gradient Descent vs Conjugate Gradient", xlabel="x", ylabel="y", legend=:topleft, color=:black, linewidth=0.5)

x_coords_gd = [p[1] for p in path_gd]
y_coords_gd = [p[2] for p in path_gd]
plot!(contour_plot, x_coords_gd, y_coords_gd, marker=:circle, color=:green, lw=1.5, markersize=4, label="Gradient Descent")

x_coords_cg = [p[1] for p in path_cg]
y_coords_cg = [p[2] for p in path_cg]
plot!(contour_plot, x_coords_cg, y_coords_cg, marker=:circle, color=:red, lw=1.5, markersize=4, label="Conjugate Gradient")

plot!(contour_plot)

export ScatteredArray, conj_grad, conj_grad_2d, gradient_descent, create_scattered_system_matrix

end # module Methode_of_conjugate_gradients