include("ktensor.jl")
include("images.jl")

struct Filter <: AbstractImage
    data :: AbstractArray{ktensor} # data of the tensor
    order :: Int8
    parity :: Int8
    dimension :: Int8
    size :: Int8

    function Filter(order:: Int, parity:: Int, dimension:: Int, size:: Int)
        shape = (size^dimension)
        for i in 1:order
            shape = (shape..., dimension)
        end
        data = zeros(Float64,shape)
        parity = parity % 2
        ktensors = map(x->ktensor(x; parity=parity),collect(eachslice(data,dims=1)))
        return new(ktensors, order, parity, dimension, size)
    end

    function Filter(data::AbstractArray{ktensor}, order::T, parity::T, ndim::T, size::T) where {T<:Integer}
        return new(data, order, parity, ndim, size)
    end
end


# Convolve
function convolve(a::T, b::Filter) where{T<:AbstractImage}
    
end

# Bigness

function bigness(a::Filter)
    pixels = make_pixel(a)
    pixels = collect.(reshape(pixels,length(pixels)))
    numerator, denominator = 0., 0.
    for i in 1:length(pixels)
        numerator += sqrt(sum((((pixels[i].-ceil(a.size/2)) .* norm(a.data[i])).^2)))
        denominator += norm(a.data[i])
    end
    return numerator / denominator
end

# Rectify


Base.convert(::Type{Filter}, a::Image) = Filter(a.data, a.order, a.parity, a.dimension, a.size)
Base.convert(::Type{Image}, a::Filter) = Image(a.data, a.order, a.parity, a.dimension, a.size)