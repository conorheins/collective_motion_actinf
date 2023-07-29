# Julia set up instructions

Start by [downloading and installing Julia](https://julialang.org/download/).

## Set up / installation

If you're within the `julia` sub-directory of the root repository, do the following:

1.  start up julia by typing `julia` into the command line
2.  then navigate down to `collective_actinf` by switching to `shell` mode (via `;`)
3.  switch to `pkg` mode using `]` and then activate the package:

```
>>> julia
shell> cd collective_actinf
] 
pkg > activate .
```

## Running the code 

Once done, you can then run scripts from the command line (e.g., the demos in `demo_scripts`) commands such as the following:

```
julia --project=collective_actinf collective_actinf/src/demo_scripts/demo_decision.jl
```



