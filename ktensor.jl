struct ktensor
    data :: Array{Float64} # data of the tensor
    order ::Int8 # order of the tensor, k
    parity :: Int8 # parity of the tensor, p
end

function +(a::ktensor, b::ktensor)::ktensor
    if a.order != b.order
        error("Orders of the tensors are not equal")
    end
    if a.parity != b.parity
        error("Parities of the tensors are not equal")
    end
    return ktensor(a.data .+ b.data, a.order, a.parity)
end

function *(a::ktensor, b::ktensor)::ktensor
    if a.order == 0 || b.order == 0
        return ktensor(a.data*b.data, a.order, a.parity)
    end
    return ktensor(a.data .* b.data, a.order, a.parity)
end

function *(a::ktensor, b::Float64)::ktensor
    return ktensor(a.data*b, a.order, a.parity)
end

function norm(a::ktensor)::Float64
    return sqrt(sum(a.data.^2))
end

