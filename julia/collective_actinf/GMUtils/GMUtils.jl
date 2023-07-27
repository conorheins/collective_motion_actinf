module GMUtils

using LinearAlgebra, BlockDiagonals

function generate_precision_matrices(;ns_φ::Int64 = 2, ndo_φ::Int64 = 2, ns_x::Int64 = 2, ndo_x::Int64 = 2,
    σ_z::Float32 = 1f0, ls_z::Float32 = 0.01f0, σ_ω::Float32 = 1f0, ls_ω::Float32 = 0.01f0,
    s_z::Float32 = 1f0, s_ω::Float32 = 1f0)::Tuple{Array{Float32,2},Array{Float32,2}}

    spatial_correlations_z = compute_spatial_precisions(ns_φ, ls_z, σ_z)[1]
    temporal_correlations_z = compute_temporal_precisions(ndo_φ, s_z)[1]

    𝚷_z::Array{Float32,2} = kron(temporal_correlations_z, spatial_correlations_z)

    spatial_correlations_ω = compute_spatial_precisions(ns_x, ls_ω, σ_ω)[1]
    temporal_correlations_ω = compute_temporal_precisions(ndo_x, s_ω)[1]

    𝚷_ω::Array{Float32,2} = kron(temporal_correlations_ω, spatial_correlations_ω)

    return 𝚷_z, 𝚷_ω

end
function compute_spatial_precisions(n::Int64, ls::Float32, σ::Float32)::Tuple{Array{Float32,2},Array{Float32,2}}
    """
    This function uses the length scale parameter `ls` to parameterise a "spatial" covariance
    matrix using the so-called squared exponential kernel (a common kernel for Gaussian processes).
    Calculated according to the following equation:
    V(x_a, x_b) = σ² exp( - ||x_a - x_b||² / 2*ls²)
    The inverse of this spatial covariance matrix is then computed to return the precision matrix `R`
    """

    if ls == 0f0
        V, R = σ .* diagm(ones(Float32,n)), (1f0/σ) .* diagm(ones(Float32,n))
    else
        V = σ .* exp.((-(collect(Float32,1:n) .- collect(Float32,1:n)').^2) ./ ((2f0*ls)^2))
        # and precision - R
        R = inv(V)
    end

    # clean up these weird 'negative 0f0' values
    V[V.==0f0] .= 0f0
    R[R.==0f0] .= 0f0

    return R, V

end

function compute_temporal_precisions(n::Int64,s::Float32;form::String="Gaussian")::Tuple{Array{Float32,2}, Array{Float32,2}}
    """
    Based on spm_DEM_R by Karl Friston. Returns the precision of the temporal derivatives of a Gaussian process
    Arguments:
    ---------
        -`n` - Int64 integer that determines the truncation order (how many generalised coordinates of motion)
        -`s` - Float32 integer that determines the temporal smoothness - s.d. of kernel (bins)
        -`form` - String with optional settings of 'Gaussian' or '1/f' [default: 'Gaussian']
    Returns:
    ---------
        -`R` - 2-D Float32 Array = E*V*E: precision of n derivatives
        -`V` - 2-D Float32 Array = V: covariance of n derivatives
    """

    if form == "Gaussian"

        k::Array{Int64,1} = collect(0:(n - 1))
        r::Array{Float32,1} = zeros(Float32, 1+2*k[end])
        x::Float32 = sqrt(2f0) * s
        r[1 .+ 2*k] .= cumprod(1 .- 2*k)./(x.^(2*k))

    elseif form == "1/f"
        k = collect(0:(n - 1))
        r = zeros(Float32, 1+2*k[end])
        x = 8f0*s^2
        r[1 .+ 2*k] .= (-1f0).^k.*gamma.(2*k .+ 1f0)./(x.^(2*k))

    end

    # create covariance matrix V in generalised coordinates
    V::Array{Float32,2} = zeros(n,n)
    for i = 1:n
        V[i,:] = r[(collect(1:n) .+ i .- 1)]
        r = -r
    end

    # and precision - R
    R::Array{Float32,2} = inv(V)

    # clean up these weird 'negative 0f0' values
    V[V.==0f0] .= 0f0
    R[R.==0f0] .= 0f0

    return R, V

end

function generate_default_gm_params(;dt = 0.01f0, ns_φ = 4, ndo_φ =2, ns_x = 4, ndo_x = 3, σ_z = 1f0, σ_ω = 1f0, ls_z=0f0, ls_ω = 0f0,
                                    s_z = 1f0, s_ω = 1f0, α_g = 10f0, b = 3.5f0, α_f = 0.5f0, η_f = 1f0, num_iter = 1, κ_μ=10.0f0, κ_a = 0.2f0,
                                    λ = 0.9f0, vectorize_f = false)

    # create dictionary of precision parameters
    precision_params = Dict(:ns_φ => ns_φ, :ndo_φ => ndo_φ, :ns_x => ns_x,
                            :ndo_x => ndo_x, :σ_z => σ_z, :ls_z => ls_z, :σ_ω => σ_ω,
                            :ls_ω => ls_ω, :s_z => s_z, :s_ω => s_ω)

    𝚷_z, 𝚷_ω = generate_precision_matrices(;precision_params...)

    # sensory mapping function g and process ('flow') function f

    if !vectorize_f
        # parameters for the sensory and flow functions
        α_f = α_f .* ones(Float32, ns_x) # linear scaling in Langevin flow function
    end

    η_f = repeat([η_f], ns_x) .* ones(Float32, ns_x) # priors on the dynamics in linear flow function

    # generate the functions, given the parameters defined above
    g(x) = x ./ (1f0 .+ exp.(α_g .* (x .- b)));

    function ∂g∂x(x)
        output = 1f0 ./ (1f0 .+ exp.(α_g .* (x .- b))) .- (α_g .* x .* exp.(α_g .* (x .- b))) ./ (exp.(α_g .* (x .- b)) .+ 1f0).^2;

        if typeof(output) == Float32
            if abs(output) < 1.0f-8 || isnan(output)
                output = 0f0
            end
        else
            output[ (abs.(output) .< 1.0f-8) .| isnan.(output) ] .= 0f0; # prevent numerical underflow
        end

        return output
    end

    # vector version
    # f(x) = -α_f .* (x .- η_f);
    # ∂f∂x(x) = -α_f

    # matrix version
    if α_f isa Number
        A = diagm(α_f .* ones(Float32, ns_x))
    elseif α_f isa Matrix
        A = α_f
    end

    f(x) = -A * (x .- η_f);
    ∂f∂x(x) = -A
    ########################################

    # learning rate, other parameters
    # original κ_μ is in 'μ_updates per second'
    κ_μ = dt * κ_μ./num_iter # learning rate (scaled relative to the number of iterations, and the integration window of the dynamics)

    ########################################

    gm_params = Dict(:ns_φ => ns_φ, :ndo_φ => ndo_φ, :ns_x => ns_x, :ndo_x => ndo_x, :𝚷_z => 𝚷_z, :𝚷_ω => 𝚷_ω,
                    :sensory_func => g, :sensory_func_prime => ∂g∂x, :flow_func => f, :flow_func_prime => ∂f∂x,
                    :num_iter => num_iter, :κ_μ => κ_μ, :κ_a => κ_a, :λ => λ)

    return gm_params
end

function generate_default_gm_params_linear_g(;dt = 0.01f0, ns_φ = 4, ndo_φ =2, ns_x = 4, ndo_x = 3, σ_z = 1f0, σ_ω = 1f0, ls_z=0f0, ls_ω = 0f0,
                                    s_z = 1f0, s_ω = 1f0, α_f = 0.5f0, η_f = 1f0, σ_z_tdist = 1f0, σ_ω_tdist = 1f0, s_z_tdist = 1f0, s_ω_tdist = 1f0,
                                    α_tdist = 0.5f0, η_tdist = 0f0, num_iter = 1, κ_μ=10.0f0, κ_a = 0.2f0, λ = 0.9f0, β_scalar = 1f0)

    # create dictionary of precision parameters
    precision_params = Dict(:ns_φ => ns_φ, :ndo_φ => ndo_φ, :ns_x => ns_x,
                            :ndo_x => ndo_x, :σ_z => σ_z, :ls_z => ls_z, :σ_ω => σ_ω,
                            :ls_ω => ls_ω, :s_z => s_z, :s_ω => s_ω)

    𝚷_z, 𝚷_ω = generate_precision_matrices(;precision_params...)

    # create dictionary of precision parameters for tdist observation and mean
    precision_params_tdist = Dict(:ns_φ => 1, :ndo_φ => ndo_φ, :ns_x => 1,
                            :ndo_x => ndo_x, :σ_z => σ_z_tdist, :ls_z => 0f0, :σ_ω => σ_ω_tdist,
                            :ls_ω => 0f0, :s_z => s_z_tdist, :s_ω => s_ω_tdist)

    𝚷_z_tdist, 𝚷_ω_tdist = generate_precision_matrices(;precision_params_tdist...)

    # sensory mapping function g and process ('flow') function f

    η_f = repeat([η_f], ns_x) .* ones(Float32, ns_x) # priors on the mean vector in linear flow function

    ## NEW NEW VERSION
    η_f_general = vcat([η_f, zeros(Float32, ns_x * (ndo_x - 1))]...) # priors on the mean vectors across generalised orders

    g_transform = hcat([Matrix(1f0I, (ns_x*ndo_φ), (ns_x*ndo_φ)), zeros(Float32, (ns_x*ndo_φ), ns_x*(ndo_x - ndo_φ))]...)
    ∇g::Matrix{Float32} = Matrix(g_transform')

    function g(x)
        return g_transform*x
    end

    # STANDARD VERSION
    # function g(x)
    #     return x
    # end
    #
    # function ∂g∂x(x)
    #     return fill(1f0, size(x))
    # end

    # matrix version
    if α_f isa Number
        A = diagm(α_f .* ones(Float32, ns_x))
    elseif α_f isa Matrix
        A = α_f
    end

    # this is what you use for the gradients ∇f
    generalised_A::Matrix{Float32} = Matrix(BlockDiagonal(repeat([A], ndo_x)))

    # only this version seems to work for predictions, where you remove stuff at the highest generalised order
    # generalised_A_missing::Matrix{Float32} = Matrix(BlockDiagonal( cat(repeat([A], ndo_x-1), [zeros(Float32, ns_x, ns_x)], dims = 1)))
    # tilde_f(x) = -generalised_A_missing * (x .- η_f_general)

    tilde_f(x) = -generalised_A * (x .- η_f_general)
    ∇f::Matrix{Float32} = Matrix(-generalised_A')

    D_shift::Matrix{Float32} = diagm(ns_x => ones(Float32,ndo_x*ns_x- ns_x));
    D_T::Matrix{Float32} = Matrix(D_shift')

    # f(x) = -A * (x .- η_f);
    # ∂f∂x(x) = -A

    f_tdist(tdist) = -α_tdist .* (tdist .- η_tdist)
    ∂f_∂tdist(tdist) = -α_tdist
    ########################################

    # learning rate, other parameters
    # original κ_μ is in 'μ_updates per second'
    κ_μ = dt * κ_μ./num_iter # learning rate (scaled relative to the number of iterations, and the integration window of the dynamics)

    ########################################
    #
    # gm_params = Dict(:ns_φ => ns_φ, :ndo_φ => ndo_φ, :ns_x => ns_x, :ndo_x => ndo_x, :𝚷_z => 𝚷_z, :𝚷_ω => 𝚷_ω,
    #                 :sensory_func => g, :sensory_func_prime => ∂g∂x, :flow_func => f, :flow_func_prime => ∂f∂x,
    #                 :D_shift =>D_shift, :D_T => D_T, :𝚷_z_tdist => 𝚷_z_tdist, :𝚷_ω_tdist => 𝚷_ω_tdist, :flow_func_tdist => f_tdist, :flow_func_tdistprime => ∂f_∂tdist,
    #                 :num_iter => num_iter, :κ_μ => κ_μ, :κ_a => κ_a, :λ => λ, :β_scalar => β_scalar)

    gm_params = Dict(:ns_φ => ns_φ, :ndo_φ => ndo_φ, :ns_x => ns_x, :ndo_x => ndo_x, :𝚷_z => 𝚷_z, :𝚷_ω => 𝚷_ω,
                    :sensory_func => g, :∇g => ∇g, :tilde_f => tilde_f, :∇f => ∇f, :D_shift =>D_shift, :D_T => D_T,
                    :𝚷_z_tdist => 𝚷_z_tdist, :𝚷_ω_tdist => 𝚷_ω_tdist, :flow_func_tdist => f_tdist, :flow_func_tdistprime => ∂f_∂tdist,
                    :num_iter => num_iter, :κ_μ => κ_μ, :κ_a => κ_a, :λ => λ, :β_scalar => β_scalar)

    return gm_params
end

end
