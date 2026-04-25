# =============================================================================
# Siegel Theta Function Benchmark: naive vs S_{(2,2)} decomposition
# Julia version (benchmark reference for paper comparison)
#
# Usage:
#   julia theta_bench.jl
#
# Output:
#   results_julia.json  (same format as Python version)
#
# Notes:
#   - First run per function is discarded (JIT warmup)
#   - N_cut=1,2 / g=2..13 by default (edit CONFIG below)
#   - Single-threaded (matching Python benchmark conditions)
# =============================================================================

using LinearAlgebra
using Dates
using JSON

# -------------------------------- CONFIG -------------------------------------
const G_MAX_NAIVE = 13   # naive is slow; stop here
const G_MAX_S22   = 13
const N_CUTS      = [1, 2]
const N_WARMUP    = 3    # JIT warmup iterations (discarded)
const N_BENCH     = 5    # timed iterations (take minimum)
# -----------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Core: naive theta sum  θ(z|Ω)  with truncation N_cut
# Ω : g×g complex symmetric matrix (positive definite imaginary part)
# z : g-vector (complex)
# --------------------------------------------------------------------------
function theta_naive(z::Vector{ComplexF64}, Omega::Matrix{ComplexF64}, N_cut::Int)
    g = length(z)
    result = ComplexF64(0.0)
    # iterate over n ∈ {-N_cut..N_cut}^g
    rng = -N_cut:N_cut
    for n_tuple in Iterators.product(fill(rng, g)...)
        n = collect(Float64, n_tuple)
        # exp(πi (nᵀ Ω n + 2 nᵀ z))
        phase = π * im * (dot(n, Omega * n) + 2.0 * dot(n, z))
        result += exp(phase)
    end
    return result
end

# --------------------------------------------------------------------------
# S_{(2,2)} decomposition:
#   split g into g1=floor(g/2), g2=g-g1
#   assume Ω is block-diagonal (off-diag block = 0)
#   θ(z|Ω) = θ(z1|Ω1) * θ(z2|Ω2)
# --------------------------------------------------------------------------
function theta_s22(z::Vector{ComplexF64}, Omega::Matrix{ComplexF64}, N_cut::Int)
    g  = length(z)
    g1 = div(g, 2)
    g2 = g - g1

    z1     = z[1:g1]
    z2     = z[g1+1:end]
    Omega1 = Omega[1:g1,    1:g1]
    Omega2 = Omega[g1+1:end, g1+1:end]

    return theta_naive(z1, Omega1, N_cut) * theta_naive(z2, Omega2, N_cut)
end

# --------------------------------------------------------------------------
# Generate a random valid period matrix (positive definite imaginary part)
# --------------------------------------------------------------------------
function make_omega(g::Int; seed::Int=42)
    rng = MersenneTwister(seed)
    # real part: random symmetric
    A = randn(rng, g, g)
    Re = (A + A') / 2
    # imag part: random positive definite
    B = randn(rng, g, g)
    Im_part = B * B' / g + I * 1.0   # ensure positive definite
    return ComplexF64.(Re) + im * ComplexF64.(Im_part)
end

function make_z(g::Int; seed::Int=42)
    rng = MersenneTwister(seed + 1000)
    return randn(rng, ComplexF64, g)
end

# --------------------------------------------------------------------------
# Timing helper: run f() N_BENCH times after N_WARMUP warmups
# returns minimum elapsed seconds
# --------------------------------------------------------------------------
function bench(f::Function)
    # warmup (JIT)
    for _ in 1:N_WARMUP
        f()
    end
    # timed runs
    times = Float64[]
    for _ in 1:N_BENCH
        t0 = time_ns()
        f()
        t1 = time_ns()
        push!(times, (t1 - t0) * 1e-9)
    end
    return minimum(times)
end

# --------------------------------------------------------------------------
# Main benchmark loop
# --------------------------------------------------------------------------
function run_benchmark()
    results = Dict{String, Any}(
        "language"    => "Julia",
        "julia_version" => string(VERSION),
        "date"        => string(today()),
        "n_warmup"    => N_WARMUP,
        "n_bench"     => N_BENCH,
        "note"        => "minimum of $(N_BENCH) runs after $(N_WARMUP) JIT warmups; single-threaded",
        "data"        => []
    )

    for N_cut in N_CUTS
        println("\n=== N_cut = $N_cut ===")
        println(@sprintf("%-4s  %-14s  %-14s  %-10s", "g", "naive (s)", "s22 (s)", "ratio"))
        println("-" ^ 48)

        for g in 2:max(G_MAX_NAIVE, G_MAX_S22)
            Omega = make_omega(g)
            z     = make_z(g)

            row = Dict{String,Any}(
                "g"      => g,
                "N_cut"  => N_cut,
            )

            # --- naive ---
            if g <= G_MAX_NAIVE
                t_naive = bench(() -> theta_naive(z, Omega, N_cut))
                row["naive_sec"] = t_naive
            else
                row["naive_sec"] = nothing
            end

            # --- s22 ---
            if g <= G_MAX_S22
                t_s22 = bench(() -> theta_s22(z, Omega, N_cut))
                row["s22_sec"] = t_s22
            else
                row["s22_sec"] = nothing
            end

            # print
            naive_str = haskey(row, "naive_sec") && row["naive_sec"] !== nothing ?
                        @sprintf("%.6f", row["naive_sec"]) : "---"
            s22_str   = haskey(row, "s22_sec") && row["s22_sec"] !== nothing ?
                        @sprintf("%.6f", row["s22_sec"]) : "---"
            ratio_str = (row["naive_sec"] !== nothing && row["s22_sec"] !== nothing) ?
                        @sprintf("%.0f×", row["naive_sec"] / row["s22_sec"]) : "---"

            println(@sprintf("g=%-2d  %-14s  %-14s  %s", g, naive_str, s22_str, ratio_str))

            push!(results["data"], row)
        end
    end

    # save JSON
    open("results_julia.json", "w") do f
        JSON.print(f, results, 2)
    end
    println("\nSaved: results_julia.json")
end

# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
using Printf
using Random

run_benchmark()