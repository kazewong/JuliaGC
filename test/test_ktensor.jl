using JuliaGC
using Test

function test_group_actions(operators::Vector{Matrix{Int32}})::Bool
    for parity in [0,1]
        if !vector_dot_vector(Int8(parity), operators)
            print("vector_dot_vector failed, parity = $parity \n")
            return false
        else 
            print("vector_dot_vector passed, parity = $parity \n")
        end

        if !tensor_times_tensor(Int8(parity), operators)
            print("tensor_times_tensor failed, parity = $parity \n")
            return false
        else 
            print("tensor_times_tensor passed, parity = $parity \n")
        end
        if !vectors_dot_tensor(Int8(parity), operators)
            print("vectors_dot_tensor failed, parity = $parity \n")
            return false
        else 
            print("vectors_dot_tensor passed, parity = $parity \n")
        end
    end

    return true
end

function vector_dot_vector(parity::Int8, operators::Vector{Matrix{Int32}})::Bool
    D = size(operators[1],1)
    a = Ktensor(randn(D), parity=parity)
    b = Ktensor(randn(D), parity=parity)
    result = true
    a = [a for i in 1:length(operators)]
    b = [b for i in 1:length(operators)]
    dots = contract.((a.*operators).*(b.*operators),1,2)
    for i in 2:length(dots)
        if (!isapprox(dots[1].data, dots[i].data))
            result = false
            print("vector_dot_vector failed: $(dots[1]) != $(dots[i])")
            return result
        end
    end
    return result
end

function tensor_times_tensor(parity::Int8, operators::Vector{Matrix{Int32}})::Bool
    D = size(operators[1],1)
    a = Ktensor(randn(D,D), parity=parity)
    b = Ktensor(randn(D,D), parity=parity)
    result = true
    a = [a for i in 1:length(operators)]
    b = [b for i in 1:length(operators)]
    dots = contract.(contract.((a.*operators).*(b.*operators),2,3),1,2)
    for i in 2:length(dots)
        if (!isapprox(dots[1].data, dots[i].data))
            result = false
            print("vector_dot_vector failed: $(dots[1]) != $(dots[i])")
            return result
        end
    end
    return result
end

function vectors_dot_tensor(parity::Int8, operators::Vector{Matrix{Int32}})::Bool
    D = size(operators[1],1)
    a = Ktensor(randn(D), parity=parity)
    b = Ktensor(randn(D), parity=0)
    c = Ktensor(randn(D,D), parity=parity)
    result = true
    a = [a for i in 1:length(operators)]
    b = [b for i in 1:length(operators)]
    c = [c for i in 1:length(operators)]
    dots = contract.(contract.((a.*operators).*(c.*operators).*(b.*operators),2,3),1,2)
    for i in 2:length(dots)
        if (!isapprox(dots[1].data, dots[i].data))
            result = false
            print("vector_dot_vector failed: $(dots[1]) != $(dots[i])")
            return result
        end
    end
    return result
end

@test test_group_actions(make_all_operators(2))
@test test_group_actions(make_all_operators(3))
