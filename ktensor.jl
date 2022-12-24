
using Tullio
using Combinatorics
using LinearAlgebra

struct ktensor
    data::AbstractArray{Float64} # data of the tensor, [k, ndims]
    order::Int8 # order of the tensor, k
    parity::Int8 # parity of the tensor, p
    dimension::Int8 # dimension of the space where the tensor lives, d
    
    function ktensor(data::AbstractArray{Float64}, order::T, parity::T, dimension::T) where {T<:Integer}
        return new(data, order, parity%2, dimension)
    end

    function ktensor(data::AbstractArray{Float64}; parity::T) where {T<:Integer}
        dimension = size(data,1)
        order = ndims(data)
        return new(data, order, parity%2, dimension)
    end
end

function Base.:+(a::K, b::Real)::K where {K<:ktensor}
    return ktensor(a.data .+ b, parity=a.parity)
end

function Base.:+(a::K, b::ktensor)::K where {K<:ktensor}
    a.order != b.order && error("Orders of the tensors are not equal")
    a.parity != b.parity && error("Parities of the tensors are not equal")
    return ktensor(a.data + b.data, parity=a.parity)
end

function Base.:*(a::K, b::Real)::K where {K<:ktensor}
    return ktensor(a.data .* b, parity=a.parity)
end

function Base.:*(a::K, b::K) where {K<:ktensor} # Fix  outer product
    if a.order == 0 || b.order == 0
        return ktensor(a.data .* b.data, parity = a.parity + b.parity)
    end
    return ktensor(a.data .* b.data, parity = a.parity + b.parity)
end

# Times group element here

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

function contract(a::ktensor, axis1::T, axis2::T) where{T<:Integer}
    return ktensor(
        LinearAlgebra.tr(a.data; dims=(axis1, axis2)), Int8(a.order-2), a.parity, a.dimension
    )
end

# function levicivita_multiplication(a::ktensor{O,P,D}, indices::Tuple) where {O,P,D}
# # This version should work but it is pretty slow
#     ex = :(a.data[] * levicivita([]))
#     outputex = :(S[])
#     for i in 1:order(a)
#         if i in indices
#             ex.args[2].args = vcat(ex.args[2].args, Meta.parse("i" * string(i)))
#             ex.args[3].args[2].args = vcat(
#                 ex.args[3].args[2].args, Meta.parse("i" * string(i))
#             )
#             outputex.args = vcat(outputex.args, Meta.parse("i" * string(i)))
#         else
#             ex.args[2].args = vcat(ex.args[2].args, Meta.parse("k" * string(i)))
#             ex.args[3].args[2].args = vcat(ex.args[3].args[2].args, Meta.parse(string(i)))
#             outputex.args = vcat(outputex.args, Meta.parse("k" * string(i)))
#         end
#     end
#     result = eval(:(@tullio $outputex := $ex))
#     return ktensor{O, mod(P+1,2), D}(result)
# end
# # TODO(Implement test of the functionality here)