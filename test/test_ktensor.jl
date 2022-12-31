using JuliaGC
using Test

function test_group_actions(operators::Vector{Matrix{Int32}})::Bool
    D = size(operators[1],1)
    for parity in [0,1]
        kt1 = Ktensor(randn(D), parity=parity)
        kt2 = Ktensor(randn(D), parity=parity)
    end

end

function vector_dot_vector(a::Ktensor, b::Ktensor)::Bool
end

function tensor_times_tensor(a::Ktensor, b::Ktensor)::Bool
end

function vectors_dot_tensor(a::Ktensor, b::Ktensor)::Bool
end

@test test_group_actions(make_all_operators(2))
