module NoiseUtils

export NoiseStruct

Base.@kwdef mutable struct NoiseStruct

    num_samples::Int = 10000000
    noise_pool::Array{Float32,1} = randn(Float32, num_samples)
    current_index::Int = 1

    function NoiseStruct(num_samples, noise_pool, current_index)
        new(num_samples, noise_pool, current_index)
    end

end

function reset_pool(noise_struct::NoiseStruct)
    """
    Resets the noise pool of the noise structure and puts the current_index back to 1
    """

    noise_struct.noise_pool = randn(Float32,noise_struct.num_samples)
    noise_struct.current_index = 1

end

function get_samples(noise_struct::NoiseStruct, number_of_samples::Int)::Array{Float32,1}
    """
    This functions extracts a prescribed number of samples from noise_struct and augments the
    indices for the next sampling. Will also reset if the desired index is greater than
    the number of available samples.
    """

    if (noise_struct.current_index + number_of_samples - 1) > noise_struct.num_samples
        reset_pool(noise_struct)
    end

    samples = noise_struct.noise_pool[noise_struct.current_index:(noise_struct.current_index+number_of_samples-1)]
    noise_struct.current_index += number_of_samples

    return samples

end

end
