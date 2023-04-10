using Test

# Test 1: ScatteredArray multiplication
function test_scattered_array_multiplication()
    A = ScatteredArray([1.0 0.0; 0.0 2.0], [1 0; 0 2])
    x = [1.0, 2.0]
    expected_result = [1.0, 4.0]
    @test A * x == expected_result
end

# Test 2: conj_grad with a simple linear system
function test_conj_grad()
    A = ScatteredArray([2.0 1.0; 1.0 3.0], [1 2; 1 2])
    b = [3.0, 4.0]
    x, _, _ = conj_grad(A, b)
    expected_result = [1.0, 1.0]
    @test isapprox(x, expected_result, atol=1e-6)
end

# Test 3: conj_grad_2d with a simple 2D optimization problem
function test_conj_grad_2d()
    x0 = [2.0, 2.0]
    sol, _ = conj_grad_2d(grad_f, x0)
    expected_result = [0.0, 0.0]
    @test isapprox(sol, expected_result, atol=1e-6)
end
