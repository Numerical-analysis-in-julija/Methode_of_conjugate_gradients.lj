# Methode_of_conjugate_gradients.jl

This is the documentation for the repository of the method of Conjugate Gradients.jl. The documentation is divided into two parts. The first part provides a mathematical explanation of Conjugate Gradients and Gradient Descent. The second part provides a detailed explanation of the code.


## The mathematical explanation
Here will 
### Conjugate gradient method

### The Gradient descent method

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

```
