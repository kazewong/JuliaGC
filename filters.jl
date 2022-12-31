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

function make_pixel_filter(a::Filter)
    return broadcast(.-, make_pixel(a), Int(ceil(a.size/2)))
end

function flatten_index(indices::Tuple, size::T) where{T<:Integer}
    index = 1
    for i in 1:length(indices)
        if indices[i] < 1 || indices[i] > size
            index += missing
        end
        index += (indices[i]-1)*size^(i-1)
    end
    return index
end

# Convolutions related functions

function Base.CartesianIndex(a::Missing)::Missing
    return missing
end

function get_convolution_indices(a::T, b::Filter) where {T<:AbstractImage}
    # For a fixed shape image and filter, this can be precomputed to avoid repeat computation during convolution
    image_index = make_pixel(a)
    image_index = reshape(image_index,(length(image_index),1))
    filter_index = make_pixel_filter(b)
    filter_index = reshape(filter_index,(1,length(filter_index)))
    index_matrix = broadcast(.+, image_index, filter_index)
    index_matrix = flatten_index.(index_matrix, a.size)
    row_matrix = repeat(reshape(collect(1:size(index_matrix,2)),1, size(index_matrix,2)),outer=size(index_matrix,1))
    result = map((x,y)->(x===missing||y===missing) ? missing : (x,y), index_matrix, row_matrix)
    return result
end

function convolve(a::T, b::Filter, index_matrix::Matrix{Union{Missing, Tuple{Int64, Int64}}}) where{T<:AbstractImage}
    data_matrix = reshape(a.data, (length(a.data), 1)) .* reshape(b.data, (1, length(b.data)))
    result = copy(data_matrix)
    for i in 1:size(data_matrix,1)
        result[i] = sum(data_matrix[CartesianIndex.(collect(skipmissing(index_matrix[i,:])))])
    end
    return result
end

# Bigness

function bigness(a::Filter)
    pixels = make_pixel(a)
    pixels = collect.(reshape(pixels,length(pixels)))
    numerator, denominator = 0., 0.
    for i in 1:length(pixels)
        numerator += sqrt(sum((((pixels[i].-Int(ceil(a.size/2))) .* norm(a.data[i])).^2)))
        denominator += norm(a.data[i])
    end
    return numerator / denominator
end

# Rectify

function rectify(a::Filter)
    if a.order == 0
        return a
    elseif a.order == 1
        if a.parity %2 ==0            
        elseif a.dimension == 2
        end
        return
    end
end

function get_unique_invariant_filters(size::Int8, order::Int8, parity::Int8, dimension::Int8, operators::Matrix{Int8})

end

Base.convert(::Type{Filter}, a::Image) = Filter(a.data, a.order, a.parity, a.dimension, a.size)
Base.convert(::Type{Image}, a::Filter) = Image(a.data, a.order, a.parity, a.dimension, a.size)