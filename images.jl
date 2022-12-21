using ReusePatterns
include("ktensor.jl")

struct Image
    data :: Array{ktensor} # data of the tensor
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
end


function image_like(a::Image, data)::Image
    return Image(data, a.order, a.parity, a.dimension, a.size)
end

function Base.:+(a::Image, b::Image)::Image
    a.order != b.order && error("Orders of the tensors are not equal")
    a.parity != b.parity && error("Parities of the tensors are not equal")
    a.dimension != b.dimension && error("Dimensions of the tensors are not equal")
    a.size != b.size && error("Sizes of the tensors are not equal")
    return Image(a.data + b.data, a.order, a.parity, a.dimension, a.size)
end

function Base.:*(a::Image, b::Image)::Image
    a.dimension != b.dimension && error("Dimensions of the tensors are not equal")
    a.size != b.size && error("Sizes of the tensors are not equal")
    return Image(a.data * b.data, a.order + b.order, a.parity + b.parity, a.dimension, a.size)
end