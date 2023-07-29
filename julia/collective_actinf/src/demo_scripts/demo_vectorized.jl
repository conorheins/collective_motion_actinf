# %% Imports

include("../collective_actinf/GeoUtils/GeoUtils.jl")
include("../collective_actinf/SimUtils/SimUtils.jl")
include("../collective_actinf/GMUtils/GMUtils.jl")
include("../collective_actinf/AnalysisUtils/AnalysisUtils.jl")

using JLD, NPZ
using MAT
using Statistics
using StatsBase

using Plots
using LinearAlgebra

# %% global constants that define the simulation
D = 2 # spatial dimension
T = 50; # time of simulation in seconds
dt = 0.01f0; # integration window in seconds

sector_angles = [120f0, 60f0, 0f0, 360f0 - 60f0, 360f0 - 120f0] # four sectors

ns_φ = length(sector_angles) - 1
ns_x = length(sector_angles) - 1


α_f = 0.5f0 # main diagonal strenght of α matrix 
β = 0.0f0 # off-diagonal coupling strength

α_matrix = diagm(0=>α_f .* ones(Float32, ns_x), -1=> -β*α_f .* ones(Float32, ns_x-1),
        1=> -β*α_f .* ones(Float32, ns_x-1))

logπ_z = -0.397f0
logπ_ω = -0.5f0
s_z = 0.8673f0
s_ω = 1f0

num_x, num_y = 6, 12 # num points in horizontal direction, num points in vertical direction
hex_coordinates = GeoUtils.generate_hexagonal_grid(2, num_x, num_y, hex_radius = 0.75f0)

r_x_length = 0.25f0 .* (maximum(hex_coordinates[1,:]) .- minimum(hex_coordinates[1,:]))
r_y_length = 0.5f0 .* (maximum(hex_coordinates[2,:]) .- minimum(hex_coordinates[2,:]))

init_r = GeoUtils.filter_with_ellipse(hex_coordinates,rx = r_x_length, ry = r_y_length)
N = size(init_r,2)

gm_params = GMUtils.generate_default_gm_params_linear_g(dt=dt, α_f = α_matrix, σ_z = exp(-logπ_z), σ_ω = exp(-logπ_ω), s_z = s_z, s_ω = s_ω, ns_φ = ns_φ, ns_x = ns_x)
gp_params = SimUtils.generate_default_gp_params_linear_g(N, T, D, sector_angles = sector_angles, ns_φ = ns_φ, z_gp = 0.01f0, z_action = 0.04f0)

mean_vel, var_vel = [0f0, 1f0], [0.01f0, 0.01f0]
init_v = GeoUtils.initialize_velocities(D, N, mean_vel, var_vel, sampling_fn = randn)

init_r = cat(init_r, zeros(Float32, D, N, gp_params[:T_sim] - 1), dims = 3)
init_v = cat(init_v, zeros(Float32, D, N, gp_params[:T_sim] - 1), dims = 3)

results_dict = SimUtils.run_simulation_new(init_r, init_v, gm_params, gp_params)
r, v = results_dict[:r], results_dict[:v]

# %% Visualize the output

keep_indices = 1:N
time_indices = 1500:3000
x_range = (minimum(r[1,keep_indices,time_indices])-1, maximum(r[1,keep_indices,time_indices])+1)
y_range = (minimum(r[2,keep_indices,time_indices])-1, maximum(r[2,keep_indices,time_indices])+1)

# plot the trajectories of each agent over time
p = plot(r[1,1,time_indices], r[2,1,time_indices], xlims=x_range, ylims=y_range, label="", dpi=325)
for agent_id in keep_indices[2:end]
    plot!(p, r[1,agent_id,time_indices], r[2,agent_id,time_indices], xlims=x_range, ylims=y_range, label="", dpi=325)
end

display(p)
println("Press Enter to exit...")
readline()

# %% If you're running this in a console or IDE, can try using this animation code to display in an embedded plotting window

# time_indices = 1500:10:3000
# x_range = (minimum(r[1,keep_indices,time_indices])-1, maximum(r[1,keep_indices,time_indices])+1)
# y_range = (minimum(r[2,keep_indices,time_indices])-1, maximum(r[2,keep_indices,time_indices])+1)

# anim = @animate for t in time_indices
#     scatter(r[1,keep_indices,t],r[2,keep_indices,t],label="",xlims = x_range, ylims = y_range)
# end
# gif(anim, "test.gif", fps=40)