module Methode_of_conjugate_gradients

using LinearAlgebra
using Plots

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

    return x, i, residuals
end

export ScatteredArray, conj_grad

end # module Methode_of_conjugate_gradients
