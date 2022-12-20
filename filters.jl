include("ktensor.jl")

struct filter
    data :: Array{ktensor} # data of the tensor
    order :: Int8
    parity :: Int8
    dimension :: Int8
    size :: Int8
    
end

function make_pixel_and_keys(m::Int) :: Tuple(Array)
    range = -m:1:m
end

function make_filter(data::Array{ktensor}, order::Int8, parity::Int8)::filter
    return filter(data, order, parity)
end
