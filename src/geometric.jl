module geometric

using LinearAlgebra

function make_generator(dims::Int)::Vector{Matrix{Int32}}
    eye = Matrix{Int32}(I, dims, dims)
    eye[1,1] = -1
    generators = Vector{Matrix{Int32}}()
    push!(generators, eye)
    for i in 1:dims
        for j in i+1:dims
            gg = Matrix{Int32}(I, dims, dims)
            gg[i,i] = 0
            gg[j,j] = 0
            gg[i,j] = -1
            gg[j,i] = 1
            push!(generators, gg)
        end
    end
    return generators
end

function make_operators(operators::Vector{Matrix{Int32}}, generators)::Vector{Matrix{Int32}}
    for operator in operators
        for generator in generators
            new_operator = generator * operator
            push!(operators, new_operator)
            operators = unique(operators)
        end
    end
    return operators
end

function make_all_operators(dims::Int)::Vector{Matrix{Int32}}
    generators = make_generator(dims)
    operators = Vector{Matrix{Int32}}([Matrix(I,dims,dims)])
    len = 0
    while length(operators) != len
        len = length(operators)
        operators = make_operators(operators, generators)
    end
    return operators
end

# Add testing the group

end