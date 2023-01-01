using JuliaGC
using Test

function test_group_actions(operators::Vector{Matrix{Int32}})::Bool
    D = size(operators[1],1)
    for parity in [0,1]
        kt1 = Ktensor(randn(D), parity=parity)
        kt2 = Ktensor(randn(D), parity=parity)
        if !vector_dot_vector(kt1, kt2, operators)
            print("vector_dot_vector failed, parity = $parity")
            return false
        end
        if !tensor_times_tensor(kt1, kt2)
            print("tensor_times_tensor failed, parity = $parity")
            return false
        end
        if !vectors_dot_tensor(kt1, kt2)
            print("vectors_dot_tensor failed, parity = $parity")
            return false
        end
    end

    return true
end

function vector_dot_vector(a::Ktensor, b::Ktensor, operators::Vector{Matrix{Int32}})::Bool
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

function tensor_times_tensor(a::Ktensor, b::Ktensor)::Bool
    return true
end

function vectors_dot_tensor(a::Ktensor, b::Ktensor)::Bool
    return true
end

@test test_group_actions(make_all_operators(2))
