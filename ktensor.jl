struct ktensor
    order ::Int # order of the tensor, k
    parity :: Int # parity of the tensor, p
    data :: Array{Float64} # data of the tensor
end

function +(a::ktensor, b::ktensor)::ktensor
    if a.order != b.order
        error("Orders of the tensors are not equal")
    end
    if a.parity != b.parity
        error("Parities of the tensors are not equal")
    end
    return ktensor(a.order, a.parity, a.data .+ b.data)
end

function *(a::ktensor, b::ktensor)::ktensor
    if a.order == 0 || b.order == 0
        return ktensor(a.order, a.parity, a.data*b.data)
    end
    return ktensor(a.order, a.parity, a.data .* b.data)
end