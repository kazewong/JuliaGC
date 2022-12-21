
using Tullio
using Combinatorics
using LinearAlgebra

struct ktensor{O,P,D}
    data::Array{Float64,O} # data of the tensor, [k, ndims]
    order::Val{O} # order of the tensor, k
    parity::Val{P} # parity of the tensor, p
    dimension::Val{D} # dimension of the space where the tensor lives, d

    function ktensor{O,P,D}(data::AbstractArray{Float64,O}) where {O,P,D}
        return new{O,P,D}(data, Val(O), Val(P), Val(D))
    end

    function ktensor(data::AbstractArray{Float64,_O}; parity::Int) where {_O}
        dimension = size(data,1)
        return new{_O,parity,dimension}(data, Val(_O), Val(parity), Val(dimension))
    end
end

@inline order(::ktensor{O}) where {O} = O
@inline parity(::ktensor{O,P}) where {O,P} = P
@inline dimension(::ktensor{O,P,D}) where {O,P,D} = D

function ktensor_like(a::ktensor{O,P,D}, data)::ktensor{O,P,D} where {O,P,D}
    return ktensor{O,P,D}(data)
end

function Base.:+(a::K, b::Real)::K where {K<:ktensor}
    return ktensor_like(a, a.data .+ b)
end

function Base.:+(a::K, b::ktensor)::K where {K<:ktensor}
    order(a) != order(b) && error("Orders of the tensors are not equal")
    parity(a) != parity(b) && error("Parities of the tensors are not equal")
    return ktensor_like(a, a.data + b.data)
end

function Base.:*(a::K, b::Real)::K where {K<:ktensor}
    return ktensor_like(a, a.data .* b)
end

function Base.:*(a::K, b::ktensor)::K where {K<:ktensor}
    if order(a) == 1
        return ktensor_like(a, a.data * b.data)
    end
    return ktensor_like(a, a.data .* b.data)
end

function Base.:*(a::K, b::Float64)::K where {K<:ktensor}
    return ktensor_like(a, a.data * b)
end

norm(a::ktensor)::Float64 = sqrt(sum((x,) -> x^2, a.data))

@generated function _tr_dims(x::AbstractArray{T,N}; dims) where {T,N}
    # val_dims is a tuple of Val(i), Val(j), etc.
    dims = collect(val.parameters[1] for val in dims.parameters)
    indices = [(j in dims) ? :(i) : :(:) for j in 1:N]
    x_part = :(x[$(indices...)])
    summation = if :(:) in indices
        :(out .+= $x_part)
    else
        :(out += $x_part)
    end
    quote
        i = first(axes(x, 1))
        out = zero($x_part)
        for i in axes(x, 1)
            $summation
        end
        out
    end
end
function LinearAlgebra.tr(x::AbstractArray{T,N}; dims) where {T,N}
    possible_dims = ntuple(i -> Val(i), N)
    selected_dims = Tuple(collect(possible_dims[dim] for dim in dims))
    return _tr_dims(x; dims=selected_dims)
end

function contract(a::ktensor{O,P,D}, axis1::Int, axis2::Int) where {O,P,D}
    return ktensor{O-2,P,D}(
        LinearAlgebra.tr(a.data; dims=(axis1, axis2))
    )
end

function levicivita_multiplication(a::ktensor{O,P,D}, indices::Tuple) where {O,P,D}
# This version should work but it is pretty slow
    ex = :(a.data[] * levicivita([]))
    outputex = :(S[])
    for i in 1:order(a)
        if i in indices
            ex.args[2].args = vcat(ex.args[2].args, Meta.parse("i" * string(i)))
            ex.args[3].args[2].args = vcat(
                ex.args[3].args[2].args, Meta.parse("i" * string(i))
            )
            outputex.args = vcat(outputex.args, Meta.parse("i" * string(i)))
        else
            ex.args[2].args = vcat(ex.args[2].args, Meta.parse("k" * string(i)))
            ex.args[3].args[2].args = vcat(ex.args[3].args[2].args, Meta.parse(string(i)))
            outputex.args = vcat(outputex.args, Meta.parse("k" * string(i)))
        end
    end
    result = eval(:(@tullio $outputex := $ex))
    return ktensor{O, mod(P+1,2), D}(result)
end
# TODO(Implement test of the functionality here)