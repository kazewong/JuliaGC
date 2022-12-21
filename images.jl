using ReusePatterns
include("ktensor.jl")

struct Image
    data :: AbstractArray{ktensor} # data of the tensor
    order :: Int8
    parity :: Int8
    dimension :: Int8
    size :: Int8

    function Image(order:: Int, parity:: Int, dimension:: Int, size:: Int)
        data = zeros(Float64,(size^dimension, dimension, dimension))
        parity = parity % 2
        ktensors = map(x->ktensor(x; parity=parity),collect(eachslice(data,dims=1)))
        return new(ktensors, order, parity, dimension, size)
    end 

    function Image(data::AbstractArray{K}, order::T, parity::T, dimension::T, size::T) where {K<:ktensor,T}
        return new(data, order, parity, dimension, size)
    end
end


function image_like(a::Image, data::AbstractArray{K})::Image where {K<:ktensor}
    return Image(data, a.order, a.parity, a.dimension, a.size)
end

function Base.:+(a::Image, b::Image)::Image
    a.order != b.order && error("Orders of the tensors are not equal")
    a.parity != b.parity && error("Parities of the tensors are not equal")
    a.dimension != b.dimension && error("Dimensions of the tensors are not equal")
    a.size != b.size && error("Sizes of the tensors are not equal")
    return image_like(a, a.data .+ b.data)
end

function Base.:+(a::Image, b::T)::Image where {T}
    return image_like(a, a.data .+ b)
end

function Base.:*(a::Image, b::Image)::Image
    a.dimension != b.dimension && error("Dimensions of the tensors are not equal")
    a.size != b.size && error("Sizes of the tensors are not equal")
    return image_like(a, a.data .* b.data)
end

function Base.:*(a::Image, b::T)::Image where {T}
    return image_like(a, a.data .* b)
end

# Convolve

