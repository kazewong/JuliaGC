module images

include("ktensor.jl")

abstract type AbstractImage end

struct Image <: AbstractImage
    data :: AbstractArray{ktensor} # data of the tensor
    order :: Int8
    parity :: Int8
    dimension :: Int8
    size :: Int8 # size of each dimension of the image, should change to a tuple later

    function Image(order:: Int, parity:: Int, dimension:: Int, size:: Int)
        shape = (size^dimension)
        for i in 1:order
            shape = (shape..., dimension)
        end
        data = zeros(Float64,shape)
        parity = parity % 2
        ktensors = map(x->ktensor(x; parity=parity),collect(eachslice(data,dims=1)))
        return new(ktensors, order, parity, dimension, size)
    end 

    function Image(data::AbstractArray{K}, order::T, parity::T, dimension::T, size::T) where {K<:ktensor,T}
        return new(data, order, parity % 2, dimension, size)
    end
end

# Basic functionalities

function make_pixel(a::T)::AbstractArray where{T<:AbstractImage}
    # Julia counts from 1, so we don't need to make the key, idiot.
    range = 1:a.size
    pixel = collect(Base.product(ntuple(i->range, a.dimension)...))
    return pixel
end

function image_like(a::T, data::AbstractArray{K})::T where {T<:AbstractImage,K<:ktensor}
    return Image(data, a.order, a.parity, a.dimension, a.size)
end

function get_index(a::T, indices::Tuple)::ktensor where {T<:AbstractImage}
    index = 1
    for i in 1:length(indices)
        index += (indices[i]-1)*a.size^(i-1)
    end
    return a.data[index]
end


function set_index(a::T, indices::Tuple, kt::ktensor) where {T<:AbstractImage}
    a.dimension != kt.dimension && error("Dimensions of the tensor does not match the image")
    a.order != kt.order && error("Orders of the tensor does not match the image")
    a.parity != kt.parity && error("Parities of the tensor does not match the image")
    output = copy(a.data)
    index = 1
    for i in 1:length(indices)
        index += (indices[i]-1)*a.size^(i-1)
    end
    output[index] = kt
    return image_like(a, output)
end


# Addition

function Base.:+(a::T, b::ST)::T where {T<:AbstractImage,ST<:Real}
    return image_like(a, a.data .+ b)
end

function Base.:+(a::T, b::T)::T where{T<:AbstractImage}
    a.order != b.order && error("Orders of the tensors are not equal")
    a.parity != b.parity && error("Parities of the tensors are not equal")
    a.dimension != b.dimension && error("Dimensions of the tensors are not equal")
    a.size != b.size && error("Sizes of the tensors are not equal")
    return image_like(a, a.data .+ b.data)
end

# Multiplication

function Base.:*(a::T, b::ST)::T where {T<:AbstractImage, ST<:Real}
    return image_like(a, a.data .* b)
end

function Base.:*(a::T, b::T)::T where {T<:AbstractImage}
    a.dimension != b.dimension && error("Dimensions of the tensors are not equal")
    a.size != b.size && error("Sizes of the tensors are not equal")
    return Image(a.data .* b.data, a.order + b.order, a.parity+b.parity, a.dimension, a.size)
end

# Unpack

function unpack(a::T)::Array{ktensor} where{T<:AbstractImage}
    shape = Tuple(repeat([a.size], a.dimension))
    return reshape(a.data, shape)
end

# Contract

function contract(a::T, axis1::INT, axis2::INT) :: T where{T<:AbstractImage, INT<:Integer}
    return Image(contract.(a.data, axis1, axis2), INT(a.order-2), INT(a.parity), INT(a.dimension), INT(a.size))
end

# Levi civita Contract

function levicivita_contraction(a::T, indices::Tuple) :: T where{T<:AbstractImage}
    return Image(levicivita_contraction.(a.data, repeat([indices],size(a.data,1))), a.order, Int8(a.parity+1), a.dimension, a.size)
end

# Normalize

function normalize(a::T) :: T where{T<:AbstractImage}
    max_norm = maximum(norm.(a.data))
    return a .* (1/max_norm)
end

end