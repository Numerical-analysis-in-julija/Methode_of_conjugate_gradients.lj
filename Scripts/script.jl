using Methode_of_conjugate_gradients

# Example graph adjacency matrix (you need to replace this with your own adjacency matrix)
adj_matrix = [
    0 1 0 0;
    1 0 1 1;
    0 1 0 1;
    0 1 1 0
]

# Example stiffness (you need to replace this with your own stiffness values)
st = [1.0, 1.0, 1.0, 1.0]

A = create_scattered_system_matrix(adj_matrix, st)
b = [1.0, 2.0, 3.0, 4.0]

# Initial approximation and tolerance
x0 = ones(length(st))
tol = 1e-6

x, it, residuals = conj_grad(A, b, x0=x0, tol=tol)

println("Solution: ", x)
println("Number of iterations: ", it)

#generate image of the graph 
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