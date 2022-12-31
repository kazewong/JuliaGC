using JuliaGC
using Test
using LinearAlgebra

function test_group(operators::Vector{Matrix{Int32}})::Bool
    D = size(operators[1],1)
    for i in operators
        for j in operators
            if (!(i*j in operators))
                print("Group is not closed under multiplication\n")
                print("The two elements that bug are ", i, " and ", j)
                return false
            end
        end
    end
    print("Group closed under multiplication\n")

    for i in operators
        if (!isapprox(i*i', Matrix(I,D,D)))
            print("Some group element is not orthogonal\n")
            print("The one that bugs is ", i)
            return false
        end
    end
    print("Group is orthogonal\n")

    return true
end


@test test_group(make_all_operators(2))
@test test_group(make_all_operators(3))