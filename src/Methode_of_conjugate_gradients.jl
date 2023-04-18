module Methode_of_conjugate_gradients

using LinearAlgebra
using Plots
using PlotlyJS

struct ScatteredArray
    V::Matrix{Float64}
    I::Matrix{Int}
end

# Multiplication function for ScatteredArray
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

    residuals = Float64[]  # Added this line to initialize an array for residuals

    for i in 1:max_iter
        Ap = A * p
        alpha = rsold / dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = dot(r, r)
        push!(residuals, sqrt(rsnew))  # Added this line to store the residuals

        if sqrt(rsnew) < tol
            break
        end

        p = r + (rsnew / rsold) * p
        rsold = rsnew
    end

    return x, i, residuals  # Added 'residuals' to the return values
end

x_center = 1
y_center = 1

function f(x, y)
    return (1.5 * (x - x_center)^2) + (0.5 * (y - y_center)^2)
end

function grad_f(x, y)
    return [3 * (x - x_center), (y - y_center)]
end

function gradient_descent(grad_f, x0, max_iter=1000, tol=1e-6, lr=0.1)
    x = x0
    path = [x0]

    for i in 1:max_iter
        x = x - lr * grad_f(x...)
        push!(path, x)

        if norm(grad_f(x...)) < tol
            break
        end
    end

    return x, path
end

function conj_grad_2d(grad_f, x0, max_iter=1000, tol=1e-6, eps=1e-4)
    x = x0
    r = -grad_f(x...)
    p = copy(r)

    path = [x0]

    for i in 1:max_iter
        Ap = grad_f((x + dot(r, r) / (dot(p, grad_f(x...)) + eps) * p)...)
        alpha = dot(r, r) / (dot(p, Ap) + eps)
        x = x + alpha * p
        push!(path, x)

        r_new = r - alpha * Ap

        if norm(r_new) < tol
            break
        end

        beta = dot(r_new, r_new) / dot(r, r)
        p = r_new + beta * p
        r = r_new
    end

    return x, path
end

x0 = [3.0, 3.0]
sol_gd, path_gd = gradient_descent(grad_f, x0)
sol_cg, path_cg = conj_grad_2d(grad_f, x0)

x = -2:0.1:4
y = -2:0.1:4
Plots.contour(x, y, f, title="Gradient Descent vs Conjugate Gradient", xlabel="x", ylabel="y", legend=:topleft)

x_coords_gd = [p[1] for p in path_gd]
y_coords_gd = [p[2] for p in path_gd]
plot!(x_coords_gd, y_coords_gd, marker=:circle, color=:red, lw=1.5, markersize=4, label="Gradient Descent")

x_coords_cg = [p[1] for p in path_cg]
y_coords_cg = [p[2] for p in path_cg]
plot!(x_coords_cg, y_coords_cg, marker=:circle, color=:blue, lw=1.5, markersize=4, label="Conjugate Gradient")

plot!()

function f_moving_center(x, y, z)
    return 0.5 * ((x - 1)^2 + (y + 1)^2 + (z - 1)^2)
end

function grad_f_moving_center(x, y, z)
    return [x - 1, y + 1, z - 1]
end

function conj_grad_3d(grad_f, x0, max_iter=1000, tol=1e-6)
    x = x0
    r = -grad_f(x...)
    p = copy(r)

    path = [x0]

    for i in 1:max_iter
        Ap = grad_f(p...)
        alpha = dot(r, r) / dot(p, Ap)
        x = x + alpha * p
        push!(path, x)

        r_new = r - alpha * Ap

        if norm(r_new) < tol
            break
        end

        beta = dot(r_new, r_new) / dot(r, r)
        p = r_new + beta * p
        r = r_new
    end

    return x, path
end

# Test data
x0 = [1.0, 1.0, 1.0]
sol, path = conj_grad_3d(grad_f_moving_center, x0)

# Create a 3D scatter plot of the function
num_points = 20
x_points = range(-2, 2, length=num_points)
y_points = range(-2, 2, length=num_points)
z_points = range(-2, 2, length=num_points)

x_coords = [xi for xi in x_points for yi in y_points for zi in z_points]
y_coords = [yi for xi in x_points for yi in y_points for zi in z_points]
z_coords = [zi for xi in x_points for yi in y_points for zi in z_points]
f_values = [f_moving_center(xi, yi, zi) for xi in x_coords for yi in y_coords for zi in z_coords]

p = plotlyjs()
scatter3d(x_coords, y_coords, z_coords, f_values, mode=:markers, marker_size=3, opacity=0.3, colorbar_title="f(x, y, z)")

# Plot the path
x_path_coords = [p[1] for p in path]
y_path_coords = [p[2] for p in path]
z_path_coords = [p[3] for p in path]
scatter3d!(x_path_coords, y_path_coords, z_path_coords, mode=:markers, marker_size=6, color=:red, legend=false)

export ScatteredArray, conj_grad

end # module Methode_of_conjugate_gradients
