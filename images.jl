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
        return new(data, order, parity, dimension, size)
    end
end


function image_like(a::T, data::AbstractArray{K})::T where {T<:AbstractImage,K<:ktensor}
    return Image(data, a.order, a.parity, a.dimension, a.size)
end

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


function Base.:*(a::T, b::ST)::T where {T<:AbstractImage, ST<:Real}
    return image_like(a, a.data .* b)
end

function Base.:*(a::T, b::T)::T where {T<:AbstractImage}
    a.dimension != b.dimension && error("Dimensions of the tensors are not equal")
    a.size != b.size && error("Sizes of the tensors are not equal")
    return image_like(a, a.data .* b.data)
end

# Unpack

# Convolve

# Contract

function contract(a::T, axis1::INT, axis2::INT) :: T where{T<:AbstractImage, INT<:Integer}
    return Image(contract.(a.data, axis1, axis2), T(a.order-2), T(a.parity), T(a.dimension), T(a.size))
end

# Levi civita Contract

# Normalize
