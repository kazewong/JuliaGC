include("ktensor.jl")
include("images.jl")

struct Filter <: AbstractImage
    data :: AbstractArray{ktensor} # data of the tensor
    order :: Int8
    parity :: Int8
    dimension :: Int8
    size :: Int8

    function Filter(order:: Int, parity:: Int, dimension:: Int, size:: Int)
        data = zeros(Float64,(size^dimension, dimension, dimension))
        ktensors = map(x->ktensor(x; parity=parity),collect(eachslice(data,dims=1)))
        return new(ktensors, order, parity, dimension, size)
    end

    function Filter(data::AbstractArray{ktensor}, order::T, parity::T) where {T<:Integer}
        return new(data, order, parity, ndims(data) ,size(data,1))
    end
end


function make_pixel_and_keys(m::Int) :: Tuple(Array)
    range = -m:1:m
end


Base.convert(::Type{Filter}, a::Image) = Filter(a.data, a.order, a.parity)
Base.convert(::Type{Image}, a::Filter) = Image(a.data, a.order, a.parity, a.dimension, a.size)