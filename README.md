# Methode of conjugate gradients homework 2 

[![Documenter](https://github.com/lovc21/Methode_of_conjugate_gradients.lj/actions/workflows/Documenter.yml/badge.svg)](https://github.com/lovc21/Methode_of_conjugate_gradients.lj/actions/workflows/Documenter.yml)
[![Runtests](https://github.com/lovc21/Methode_of_conjugate_gradients.lj/actions/workflows/Runtests.yml/badge.svg)](https://github.com/lovc21/Methode_of_conjugate_gradients.lj/actions/workflows/Runtests.yml)

In this repository, you can find the code for Homework 2 of the Numerical Mathematics course. The code is written in Julia, and the main implementation can be found in the file `src/Methode_of_conjugate_gradients.jl`. The code is tested using the file `test/runtests.jl`, and it is documented using the file `docs/make.jl`.

To run the code, you need to have Julia installed on your computer. Download the code and run the `file src/Methode_of_conjugate_gradients.jl`. The file demonstrates how the conjugate gradient method works and compares it to the standard gradient descent function. To find more documentation on the code, go to the file docs/index.html and open it in your web browser. The page contains more information about the conjugate gradient method, the mathematics behind it, and comparisons between methods such as the gradient descent function.

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



    