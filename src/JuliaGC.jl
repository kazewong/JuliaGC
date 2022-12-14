module JuliaGC
using LinearAlgebra
using Combinatorics

include("geometric.jl")
include("ktensor.jl")
include("images.jl")
include("filters.jl")

export Ktensor, make_all_operators, make_generator, make_operators, contract
export Image, make_pixel, image_like, get_index, set_index
export Filter

end