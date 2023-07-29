module PlotHelpers

using Plots
using ColorSchemes
using JLD

function custom_heatmap(array, color_symbol::Symbol)

    heatmap(array, yflip=true, c=color_symbol)

end

function animate_histograms_over_time(mean_hist_per_condition::Array{Float32,4}, gif_name::String, desired_fps::Int64, color_symbol::Symbol; color_range = nothing)

	"""
    Based on fast implementation of dynamic subplots as described here:
    https://stackoverflow.com/questions/55794068/animating-subplots-using-plots-jl-efficiently
    Where z-series is simply written-over, rather than plots being entirely overwritten (via e.g. heatmap!() or plot!())
    """

	gr()

	if color_range != nothing
		clim_min, clim_max = color_range
	else
    	clim_min, clim_max = minimum(mean_hist_per_condition), maximum(mean_hist_per_condition)
	end

	num_configs = size(mean_hist_per_condition,4)

    # l = @layout [a b c; d e f; g h i] # assumes 9 subplots

	subplot_array = Array{Plots.Plot{Plots.GRBackend},1}(undef, num_configs)
	for config_i = 1:num_configs
		subplot_array[config_i] = heatmap(mean_hist_per_condition[:,:,1,config_i], yflip = true, c = color_symbol, clims=(clim_min,clim_max), colorbar = false, xticks=false,yticks=false)
	end

	# plots the first timestep of the animation
	# full_plot = plot(subplot_array...,layout=l)
	full_plot = plot(subplot_array...,layout=num_configs)


	# animates the remaining timesteps of the animation
    anim = @animate for t = 2:size(mean_hist_per_condition,3)
		for config_i = 1:num_configs
	        full_plot[config_i][1][:z] = mean_hist_per_condition[:,:,t,config_i]
		end
    end

    gif(anim,gif_name,fps=desired_fps)

end

function show_perturbation_trial(condition_folder, gif_name; fps = 10, jld_name_prefix = "init_", init_index = nothing, real_index = nothing, pre_stim_time = 25)

	jld_files = filter(x->startswith(x, jld_name_prefix), readdir(condition_folder))
	# jld_files = filter(x->endswith(x, ".jld"), readdir(condition_folder))

	if init_index == nothing
		init_index = rand(1:length(jld_files))
	elseif init_index > length(jld_files)
		warn("Initialization index greater than the number of initializations available...defaulting to the last initialization...")
		init_index = length(jld_files)
	end

	pre_perturb_r, post_perturb_r = load(joinpath(condition_folder, string(jld_name_prefix,init_index,".jld")), "r_hist", "r_all")

	if real_index == nothing
		real_index = rand(1:size(post_perturb_r,4))
	elseif real_index > size(post_perturb_r,4)
		warn("Realization index greater than the number of realizations available...defaulting to the last realization...")
		real_index = size(post_perturb_r,4)
	end

	r = cat([pre_perturb_r[:,:,(end-pre_stim_time):end], post_perturb_r[:,:,:,real_index]]...,dims=3)

	time_indices = 1:size(r,3)

	x_range = (minimum(r[1,:,:])-1, maximum(r[1,:,:])+1)
	y_range = (minimum(r[2,:,:])-1, maximum(r[2,:,:])+1)

	anim = @animate for t in time_indices
	    scatter(r[1,:,t],r[2,:,t],label="",xlims = x_range, ylims = y_range)
	end

	gif(anim, gif_name, fps=fps)

end


end
