# Methode of conjugate gradients homework 2 

[![Documenter](https://github.com/lovc21/Methode_of_conjugate_gradients.lj/actions/workflows/Documenter.yml/badge.svg)](https://github.com/lovc21/Methode_of_conjugate_gradients.lj/actions/workflows/Documenter.yml)
[![Runtests](https://github.com/lovc21/Methode_of_conjugate_gradients.lj/actions/workflows/Runtests.yml/badge.svg)](https://github.com/lovc21/Methode_of_conjugate_gradients.lj/actions/workflows/Runtests.yml)

In this repository, you can find the code for Homework 2 of the Numerical Mathematics course. The code is written in Julia, and the main implementation can be found in the file `src/Methode_of_conjugate_gradients.jl`. The code is tested using the file `test/runtests.jl`, and it is documented using the file `docs/make.jl`.

To run the code, it is necessary to have Julia installed on your computer. Once downloaded, the file `src/Methode_of_conjugate_gradients.jl` can be run to see the conjugate gradient method in action and compare it to the standard gradient descent function.

For further understanding, you can refer to the documentation by opening the file `docs/index.html` in your web browser. This page provides additional information about the code, including the mathematical theory behind the conjugate gradient method and comparisons to other methods like gradient descent. This documentation can help deepen your understanding of the code and the underlying mathematics.

## The code explaind

1. This code imports three Julia packages: LinearAlgebra, Plots, and PlotlyJS.

    ```julia
        using LinearAlgebra
        using Plots
        using PlotlyJS
    ```
2. The code defines a ScatteredArray structure with two matrices and provides custom multiplication and size functions for it.

    ```julia
            
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
    ```
3. This code defines a conj_grad function that solves a linear system Ax = b using the conjugate gradient method, given a ScatteredArray A and Vector b. It returns the solution, number of iterations, and residuals.

    ```julia
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
    ```
4. This code provides two optimization functions, gradient descent and conjugate gradient, for a 2D quadratic function.

gradient_descent and conj_grad_2d take a starting point, maximum number of iterations, tolerance, and learning rate/small value for epsilon as arguments. They return the optimized x value and the path taken during optimization as a list.
```julia
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
            Ap = grad_f(p...)
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
```

    