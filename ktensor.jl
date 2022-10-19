import Base: +,*
using Tullio
using Combinatorics

struct ktensor
    data :: Array # data of the tensor, [k, ndims]
    order ::Int8 # order of the tensor, k
    parity :: Int8 # parity of the tensor, p
    dimension :: Int8 # dimension of the space where the tensor lives, d
end

function +(a::ktensor, b::ktensor)::ktensor
    if a.order != b.order
        error("Orders of the tensors are not equal")
    end
    if a.parity != b.parity
        error("Parities of the tensors are not equal")
    end
    return ktensor(a.data .+ b.data, a.order, a.parity, a.dimension)
end

function *(a::ktensor, b::ktensor)::ktensor
    if a.order == 0 || b.order == 0
        return ktensor(a.data*b.data, a.order, a.parity, a.dimension)
    end
    return ktensor(a.data .* b.data, a.order, a.parity, a.dimension)
end

function *(a::ktensor, b::Float64)::ktensor
    return ktensor(a.data*b, a.order, a.parity, a.dimension)
end

function norm(a::ktensor)::Float64
    return sqrt(sum(a.data.^2))
end

# Need to implement group element multiplication

function group_mul(a::ktensor, element)::ktensor
end

function contraction(a::ktensor, axis1::Int, axis2::Int)::ktensor
    ex = :(a.data[])
    for i in 1:a.order
        if i == axis1 || i == axis2
            ex.args = vcat(ex.args,:i)
        else
            ex.args = vcat(ex.args,:(:))
        end
    end
    result = eval(:(@tullio S[i] := $ex))
    return ktensor(result, a.order-2, a.parity, a.dimension)
end

function levicivita_multiplication(a::ktensor, indices::Array{Int})::ktensor
    ex = :(a.data[]*levicivita([]))
    outputex = :(S[])
    for i in 1:a.order
        if i in indices
            ex.args[2].args = vcat(ex.args[2].args, Meta.parse("i"*string(i)))
            ex.args[3].args[2].args = vcat(ex.args[3].args[2].args,Meta.parse("i"*string(i)))
            outputex.args = vcat(outputex.args, Meta.parse("i"*string(i)))
        else
            ex.args[2].args = vcat(ex.args[2].args, Meta.parse("k"*string(i)))
            ex.args[3].args[2].args = vcat(ex.args[3].args[2].args, Meta.parse(string(i)))
            outputex.args = vcat(outputex.args, Meta.parse("k"*string(i)))
        end
    end
    result = eval(:(@tullio $outputex := $ex))
    return ktensor(result, a.order,  mod(a.parity + 1,2) , a.dimension)    
end

# TODO(Implement test of the functionality here)