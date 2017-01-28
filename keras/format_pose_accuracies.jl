#!/usr/bin/env julia

# Simple script to format the .npz accuracy files produced by regressors, etc.
# My first-ever Julia script :)

using NPZ

name_map = Dict(
    "vae_acc_mean_K" => "VAE (mean-of-K)",
    "random_acc_best_K" => "Random (best-of-K)",
    "ext_acc" => "Extend",
    "vae_acc_best_K" => "VAE (best-of-K)",
    "random_acc_mean_K" => "Random (mean-of-K)"
)

function main()
    if length(ARGS) != 1
        progname = basename(Base.source_path())
        @printf(STDERR, "USAGE: %s <path to .npz file>\n", progname)
        exit(1)
    end

    npz_path = ARGS[1]
    vars = npzread(npz_path)
    relkeys = sort(filter(k -> contains(k, "acc"), collect(keys(vars))))
    data_cols = [vars[k] for k in relkeys]
    K = vars["K"]
    epoch = vars["epoch"]
    @printf("Measurements taken with K=%d, epoch %d\n", K, epoch)
    @printf("| ")
    for key in relkeys
        if in(key, keys(name_map))
            real_key = name_map[key]
        else
            real_key = key
        end
        @printf("|=%s", real_key)
    end
    @printf("|\n")
    nrows = length(data_cols[1])
    for row in 1:nrows
        col_vals = map(col -> col[row], data_cols)
        @printf("|%d", row)
        for col in col_vals
            @printf("|%.2f", col)
        end
        @printf("|\n")
    end
end

main()
