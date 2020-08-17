
module IonImage

using SpecialPolynomials, Distributions, Random, Images, StatsBase, Plots
using SpecialPolynomials.FastLegendre
using BenchmarkTools
using JLD2


struct ThetaDistribution <: ContinuousUnivariateDistribution
    β::Float32
    order::Int8
    ThetaDistribution(β, order) = new(Float32(β), Int8(order))
end

function pdf(d::ThetaDistribution, x::Float32)
    y = FastLegendre.fastlegendre.(d.order, cos.(x))
    return y
end

function sample_theta(β::Float32, order::Integer, x::Array{Float32,1}, n::Integer)::Array{Float32,1}
    p = 1 .+ β .* pdf.(ThetaDistribution(β, order), x)
    # normalize probabilities
    p ./= sum(p)
    distribution = Weights(p)
    return sample(x, distribution, n)
end

function pol2cart(θ::Array{Float32,1}, r::Array{Float32,1}, center::Float32)
    x = floor.(Int, r .* cos.(θ) .+ center)
    y = floor.(Int, r .* sin.(θ) .+ center)
    return x, y
end


function bin_image!(image::Array{UInt16, 2}, x::Array{Int,1}, y::Array{Int,1})
    for (x_i, y_i) in zip(x, y)
        try
            image[x_i, y_i] += 1
        catch
            nothing
        end
    end
end


function generate_ion_image(β::Float32, σ::Float32, dim::Integer, nions::Integer)::Array{UInt16, 2}
    image = zeros(UInt16, (dim, dim))
    image_center = dim // 2
    ring_center = rand(Uniform(0, image_center * 0.91))
    # generate random velocities/distances from center
    R = convert.(Float32, rand(Normal(ring_center, σ), nions))
    angle_bins = collect(range(Float32(-2. * pi), Float32(2. * pi), length=3000))
    # Generate angle samples
    θ = sample_theta(β, 2, angle_bins, nions)
    x, y = pol2cart(θ, R, Float32(image_center))
    bin_image!(image, x, y)
    return image
end


function generate_dataset(nimages::Integer, dim::Integer)
    β_values = convert(Array{Float32,1}, rand(Uniform(-1., 2.), nimages))
    σ_values = convert(Array{Float32,1}, rand(Uniform(2., 50.), nimages))
    nions = rand(Uniform(4., 6.), nimages)
    jldopen("images.jld2", "w") do file
        file["images"] = zeros(UInt16, (nimages, dim, dim))
        Threads.@threads for i = 1:nimages
            file["images"][i,:,:] = generate_ion_image(β_values[i], σ_values[i], dim, convert(Integer, floor(10^nions[i])))
        end
    end
end

end
