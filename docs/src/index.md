# Methode_of_conjugate_gradients.jl

This is the documentation for the repository of the method of Conjugate Gradients.jl. The documentation is divided into two parts. The first part provides a mathematical explanation of Conjugate Gradients and Gradient Descent. The second part provides a detailed explanation of the code.

## The mathematical explanation
### Graph Embedding using Conjugate Gradient Method

In the context of graph embedding, the conjugate gradient method can be applied to find an equilibrium position for the nodes in a plane or space using a physical method. The idea is to treat the graph as a system of masses connected by springs, where the goal is to find the equilibrium positions of the nodes that minimize the total potential energy of the system. The potential energy function can be defined based on the distance between neighboring nodes.

The equation given for each coordinate (xi, yi, zi) of the vertices of the graph in space is:

$$
\begin{aligned}
    -st(i) x_i + \sum_{j \in N(i)} x_j - st(i) y_i + \sum_{j \in N(i)} y_j &= 0, \\
    -st(i) y_i + \sum_{j \in N(i)} y_j - st(i) z_i + \sum_{j \in N(i)} z_j &= 0, \\
    -st(i) z_i + \sum_{j \in N(i)} z_j - st(i) x_i + \sum_{j \in N(i)} x_j &= 0.
\end{aligned}
$$

Here, st(i) represents the stage of the i-th node, and N(i) is the set of indices of neighboring nodes. This equation ensures that the total force acting on each node is zero in equilibrium. If some nodes are fixed, the others will occupy an equilibrium position between the fixed nodes.

To solve this system of equations using the conjugate gradient method, we can first rewrite the equations in matrix form Ax = b, where A is a sparse matrix representing the graph structure, and x and b are vectors containing the coordinates of the vertices and the right-hand side of the equation, respectively. The conjugate gradient method can then be used to solve this linear system iteratively, finding the equilibrium positions of the nodes in the graph.

The example provided in the code section demonstrates how to create a sparse matrix representation of the graph embedding problem, and then use the conj_grad function to solve it. 

### Conjugate gradient method
The conjugate gradient method is an iterative algorithm used to solve linear systems of equations, particularly for symmetric and positive-definite matrices. It converges faster than other iterative methods like gradient descent, especially when dealing with large and sparse matrices.
### The Gradient descent method
Gradient descent is an optimization algorithm that tries to minimize a given objective function by iteratively moving in the direction of the steepest descent, as defined by the negative of the gradient. In this code, gradient descent is applied to a 2D quadratic function for visualization purposes.
## The code explaind

1. This code imports three Julia packages: LinearAlgebra, Plots.

    ```
    using LinearAlgebra
    using Plots
    ```
2. The code defines a ScatteredArray structure with two matrices and provides custom multiplication and size functions for it. This data structure is specifically designed to represent sparse matrices efficiently.

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
    
3. The code defines a function to create a ScatteredArray from an adjacency matrix and a vector of strengths. This function is essential for creating the sparse matrix representation of the graph embedding problem.

        function create_scattered_system_matrix(adj_matrix::Matrix{Int},st::Vector{Float64})
            n = size(adj_matrix, 1)
            A = zeros(Float64, n, n)

            for i in 1:n
                A[i, i] = st[i] * sum(adj_matrix[i, :]) - 1
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
        

4. This code defines a conj_grad function that solves a linear system Ax = b using the conjugate gradient method

        function conj_grad(A::ScatteredArray, b::Vector{Float64})
        n = length(b)
        x = zeros(Float64, n)
        r = b - A * x
        p = r
        rsold = r' * r
        max_iter = 1000

            for i = 1:max_iter
                Ap = A * p
                alpha = rsold / (p' * Ap)
                x = x + alpha * p
                r = r - alpha * Ap
                rsnew = r' * rs

                if sqrt(rsnew) < 1e-10
                    break
                end

                p = r + (rsnew / rsold) * p
                rsold = rsnew
            end
        return x, i
        end

The code demonstrates how to use the conj_grad function to solve the graph embedding problem using a physical method. The example is a simple graph with a few nodes and edges. The adjacency matrix and the strengths vector are given.

    # Example graph
    adj_matrix = [
        0 1 0 0 1;
        1 0 1 0 1;
        0 1 0 1 0;
        0 0 1 0 1;
        1 1 0 1 0
    ]

    st = [1.0, 1.0, 1.0, 1.0, 1.0]

    A = create_scattered_system_matrix(adj_matrix, st)
    b = [0.0, 0.0, 0.0, 0.0, 0.0]

    x, iterations = conj_grad(A, b)

    println("Solution: ", x)
    println("Number of iterations: ", iterations)

The output shows the solution vector x and the number of iterations required to achieve convergence.

To summarize, the code provides a complete implementation of the Conjugate Gradient method for solving sparse linear systems with ScatteredArray data type. This is particularly useful for the task of graph embedding using the physical method, as demonstrated in the provided example.

5. The code provided below solves a quadratic optimization problem using both the gradient descent and the conjugate gradient methods. The problem is defined as minimizing the function f(x) = 0.5 * x' * A * x - b' * x, where A and b are given. The contour plot shows the convergence of both methods. The gradient descent method converges to a local minimum, while the conjugate gradient method converges to the global minimum. The code also shows how to use the Plots package to create a contour plot.

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
    
