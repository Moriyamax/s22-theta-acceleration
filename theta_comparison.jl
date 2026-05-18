"""
theta_comparison.jl  ― S22上（厳密ブロック対角）専用ベンチマーク

naive vs s22 を同一Ω・z で評価し、相対誤差と速度を比較する。
S22外（近似精度）の測定は本スクリプトの対象外。

【並列化】
  theta_naive: チャンク分割 + Threads.@spawn
  theta_s22:   2ブロックを Threads.@spawn で並列実行

【使い方】
  julia --threads auto theta_comparison.jl
"""

using LinearAlgebra, Random, Dates, Printf

# ───────── パス ─────────
const SCRIPT_DIR  = @__DIR__
const CONFIG_PATH = joinpath(SCRIPT_DIR, "config_theta.txt")
const LOG_PATH    = joinpath(SCRIPT_DIR, "theta_log.txt")
const RESULT_PATH = joinpath(SCRIPT_DIR, "theta_results.json")

# ───────── ロック ─────────
const LOG_LOCK = ReentrantLock()

# ───────── ログ ─────────
function log_msg(msg::String)
    line = "[$(Dates.format(now(),"yyyy-mm-dd HH:MM:SS"))] $msg"
    println(line); flush(stdout)
    lock(LOG_LOCK) do
        open(LOG_PATH,"a") do f; println(f,line); end
    end
end

# ───────── 設定 ─────────
struct Cfg
    g_list::Vector{Int}
    N_cut_list::Vector{Int}
    seed::Int
    resume::Bool
    report_every::Int
end

function load_cfg()::Cfg
    defaults = Cfg([2,3,4,5,6,7,8,9,10,11,12,13,14],[1,2],42,true,1_000_000)
    if !isfile(CONFIG_PATH)
        open(CONFIG_PATH,"w") do f
            println(f,"# theta_comparison.jl config")
            println(f,"g_list = 2,4,6,8")
            println(f,"N_cut_list = 1")
            println(f,"seed = 42")
            println(f,"resume = true")
            println(f,"report_every = 10000000")
        end
        log_msg("設定ファイル生成: $CONFIG_PATH")
        return defaults
    end
    kv = Dict{String,String}()
    open(CONFIG_PATH,"r") do f
        for raw in eachline(f)
            s = strip(raw)
            (isempty(s)||startswith(s,"#")) && continue
            i = findfirst('=',s); i===nothing && continue
            kv[strip(s[1:i-1])] = strip(s[i+1:end])
        end
    end
    d = defaults
    Cfg(
        haskey(kv,"g_list")       ? parse.(Int,split(kv["g_list"],","))     : d.g_list,
        haskey(kv,"N_cut_list")   ? parse.(Int,split(kv["N_cut_list"],",")) : d.N_cut_list,
        haskey(kv,"seed")         ? parse(Int,kv["seed"])                   : d.seed,
        haskey(kv,"resume")       ? lowercase(kv["resume"])=="true"         : d.resume,
        haskey(kv,"report_every") ? parse(Int,kv["report_every"])           : d.report_every,
    )
end

# ───────── JSON保存（atomic write） ─────────
function save_results(rs::Vector{Dict})
    tmp = RESULT_PATH * ".tmp"
    open(tmp,"w") do f
        println(f,"[")
        for (i,r) in enumerate(rs)
            g_v    = r["g"]
            nc_v   = r["N_cut"]
            nre_v  = r["val_naive_re"]
            nim_v  = r["val_naive_im"]
            sre_v  = r["val_s22_re"]
            sim_v  = r["val_s22_im"]
            err_v  = r["rel_err"]
            tn_v   = r["t_naive"]
            ts_v   = r["t_s22"]
            sp_v   = get(r,"speedup",Inf)
            ts_str = string(r["timestamp"])
            println(f,"  {")
            println(f,"    \"g\": $g_v,")
            println(f,"    \"N_cut\": $nc_v,")
            println(f,"    \"val_naive_re\": $nre_v,")
            println(f,"    \"val_naive_im\": $nim_v,")
            println(f,"    \"val_s22_re\": $sre_v,")
            println(f,"    \"val_s22_im\": $sim_v,")
            println(f,"    \"rel_err\": $err_v,")
            println(f,"    \"t_naive\": $tn_v,")
            println(f,"    \"t_s22\": $ts_v,")
            println(f,"    \"speedup\": $sp_v,")
            print(f,  "    \"timestamp\": \"$ts_str\"")
            print(f, "\n  }")
            i < length(rs) ? println(f,",") : println(f)
        end
        println(f,"]")
    end
    mv(tmp, RESULT_PATH; force=true)
end

function load_results()::Vector{Dict}
    isfile(RESULT_PATH) || return Dict[]
    rs = Dict[]; cur = nothing
    open(RESULT_PATH,"r") do f
        for raw in eachline(f)
            s = strip(raw)
            s == "{" && (cur = Dict{String,Any}(); continue)
            (s=="}"||s=="},") && cur!==nothing && (push!(rs,cur); cur=nothing; continue)
            cur === nothing && continue
            q1 = findfirst('"', s);  q1===nothing && continue
            q2 = findnext('"', s, q1+1); q2===nothing && continue
            col = findnext(':', s, q2+1); col===nothing && continue
            k = s[q1+1:q2-1]
            v = strip(s[col+1:end])
            endswith(v,",") && (v = strip(v[1:end-1]))
            if startswith(v,'"')
                cur[k] = v[2:end-1]
            elseif v=="true";  cur[k]=true
            elseif v=="false"; cur[k]=false
            else
                fv = (v=="Inf" ? Inf : v=="-Inf" ? -Inf : v=="NaN" ? NaN :
                      tryparse(Float64,v))
                fv !== nothing && (cur[k]=fv)
            end
        end
    end
    for r in rs
        r["g"]     isa Float64 && (r["g"]     = Int(r["g"]))
        r["N_cut"] isa Float64 && (r["N_cut"] = Int(r["N_cut"]))
    end
    return rs
end

done_key(r::Dict) = (get(r,"g",nothing), get(r,"N_cut",nothing))
already_done(rs,g,N) = any(r->done_key(r)==(g,N), rs)

# ───────── Theta 関数 ─────────
function theta_naive(z::Vector{ComplexF64}, Omega::Matrix{ComplexF64}, N_cut::Int;
                     report_every::Int=0, label::String="")::ComplexF64
    g = length(z)
    b = 2*N_cut + 1
    total = b^g
    nt = max(1, Threads.nthreads())
    counter = Threads.Atomic{Int}(0)
    tasks = map(0:nt-1) do ci
        i_start = ci * (total ÷ nt) + min(ci, total % nt)
        i_end   = i_start + (total ÷ nt) - 1 + (ci < total % nt ? 1 : 0)
        Threads.@spawn begin
            local_sum = zero(ComplexF64)
            nv  = Vector{Float64}(undef, g)
            Omn = Vector{ComplexF64}(undef, g)
            for idx in i_start:i_end
                tmp = idx
                for k in 1:g
                    nv[k] = Float64(tmp % b - N_cut)
                    tmp ÷= b
                end
                mul!(Omn, Omega, nv)
                qc = dot(nv, Omn)
                lc = dot(nv, z)
                local_sum += exp(im*π*qc + 2im*π*lc)
                if report_every > 0
                    cnt = Threads.atomic_add!(counter, 1) + 1
                    if cnt % report_every == 0
                        pct = 100.0*cnt/total
                        lbl2 = isempty(label) ? "" : " [$label]"
                        log_msg(@sprintf("    [theta_naive%s] 格子点 %d / %d (%.2f%%) 完了",
                            lbl2, cnt, total, pct))
                    end
                end
            end
            local_sum
        end
    end
    return sum(fetch(t)::ComplexF64 for t in tasks)
end

function theta_s22(z::Vector{ComplexF64}, Omega::Matrix{ComplexF64}, N_cut::Int;
                   report_every::Int=0, label::String="")::ComplexF64
    g  = length(z); g1 = g÷2
    l1 = isempty(label) ? "B1" : "$label/B1"
    l2 = isempty(label) ? "B2" : "$label/B2"
    t1 = Threads.@spawn theta_naive(z[1:g1],     Omega[1:g1,1:g1],         N_cut; report_every, label=l1)
    t2 = Threads.@spawn theta_naive(z[g1+1:end], Omega[g1+1:end,g1+1:end], N_cut; report_every, label=l2)
    return (fetch(t1)::ComplexF64) * (fetch(t2)::ComplexF64)
end

# ───────── Ω生成（厳密S(2,2)ブロック対角のみ） ─────────
function pos_block(n::Int, rng)
    A = randn(rng, Float64, n, n) .* 0.3
    return im .* (A*A' .+ Matrix(1.5I, n, n))
end

function omega_on(g::Int, rng)
    g1 = g÷2
    Ω  = zeros(ComplexF64, g, g)
    Ω[1:g1,       1:g1]       = pos_block(g1,   rng)
    Ω[g1+1:end,   g1+1:end]   = pos_block(g-g1, rng)
    return (Ω + transpose(Ω)) ./ 2
end

function make_z(g::Int, rng)
    complex.(randn(rng, Float64, g) .* 0.5,
             randn(rng, Float64, g) .* 0.1)
end

function make_data(g::Int, seed::Int)
    z  = make_z(g, MersenneTwister(seed+g))
    Ω  = omega_on(g, MersenneTwister(seed+g+10000))
    return z, Ω
end

# ───────── 比較実行 ─────────
function run_one(g, N, Ω, z; report_every=0)::Dict
    b=2N+1; pts=b^g
    log_msg(@sprintf("  → naive 開始: g=%d N=%d 格子点=%d スレッド=%d",
        g, N, pts, Threads.nthreads()))
    t0=time()
    vn=theta_naive(z, Ω, N; report_every, label="naive/g$g/N$N")
    tn=time()-t0
    log_msg(@sprintf("  ← naive 完了: %.4fs  val=%.6f%+.6fi", tn, real(vn), imag(vn)))

    if abs(vn) < 10.0
        log_msg(@sprintf("  ⚠ 警告: |θ|=%.4f < 10（ゼロ点近傍）rel_errが大きくなる可能性あり。別seedでの再測定を推奨。",
            abs(vn)))
    end

    log_msg(@sprintf("  → s22  開始: g=%d N=%d", g, N))
    t0=time()
    vs=theta_s22(z, Ω, N; report_every, label="s22/g$g/N$N")
    ts=time()-t0
    log_msg(@sprintf("  ← s22  完了: %.4fs  val=%.6f%+.6fi", ts, real(vs), imag(vs)))

    re = abs(vn) < 1e-300 ? NaN : abs(vn-vs) / abs(vn)
    sp = ts > 1e-12 ? tn/ts : Inf
    Dict("g"=>g, "N_cut"=>N,
         "val_naive_re"=>real(vn), "val_naive_im"=>imag(vn),
         "val_s22_re"  =>real(vs), "val_s22_im"  =>imag(vs),
         "rel_err"=>re, "t_naive"=>tn, "t_s22"=>ts, "speedup"=>sp,
         "timestamp"=>Dates.format(now(),"yyyy-mm-ddTHH:MM:SS"))
end

function show_result(r::Dict)
    @printf "    g=%d N=%d\n" r["g"] r["N_cut"]
    @printf "      naive: %.6f%+.6fi (%.4fs)\n" r["val_naive_re"] r["val_naive_im"] r["t_naive"]
    @printf "      s22  : %.6f%+.6fi (%.4fs)\n" r["val_s22_re"]   r["val_s22_im"]   r["t_s22"]
    sp = get(r,"speedup",NaN)
    @printf "      rel_err=%.2e  speedup=%.1fx\n" r["rel_err"] sp
    flush(stdout)
end

# ───────── サマリー ─────────
function summary(rs::Vector{Dict})
    log_msg("="^70)
    log_msg("サマリー")
    log_msg(@sprintf("%3s %2s %12s %10s %10s %8s","g","N","rel_err","naive(s)","s22(s)","倍率"))
    log_msg("-"^70)
    for r in sort(rs, by=r->(get(r,"g",0), get(r,"N_cut",0)))
        sp = get(r,"speedup",NaN)
        log_msg(@sprintf("%3d %2d %12.2e %10.4f %10.4f %7.1fx",
            r["g"], r["N_cut"], r["rel_err"], r["t_naive"], r["t_s22"], sp))
    end
    log_msg("結果: $RESULT_PATH  ログ: $LOG_PATH")
end

# ───────── main（逐次実行） ─────────
function main()
    cfg = load_cfg()
    rs::Vector{Dict} = cfg.resume ? load_results() : Dict[]

    log_msg("="^70)
    log_msg("theta_comparison.jl  naive vs s22  [S22上専用]")
    log_msg("Julia $(VERSION)  スレッド=$(Threads.nthreads())")
    log_msg("g=$(cfg.g_list)  N_cut=$(cfg.N_cut_list)  seed=$(cfg.seed)")
    log_msg("report_every=$(cfg.report_every)  resume=$(cfg.resume)  既完了=$(length(rs))件")
    log_msg("="^70)

    cases = [(g,N) for g in cfg.g_list for N in cfg.N_cut_list]
    total = length(cases)

    for (n,(g,N)) in enumerate(cases)
        if cfg.resume && already_done(rs,g,N)
            log_msg(@sprintf("  スキップ [%d/%d] g=%d N=%d", n, total, g, N))
            i = findfirst(r->done_key(r)==(g,N), rs)
            i !== nothing && show_result(rs[i])
            continue
        end

        log_msg(@sprintf("\n─── [%d/%d] (%.1f%%) g=%d N=%d ───",
            n, total, 100.0*n/total, g, N))
        z, Ω = make_data(g, cfg.seed)
        r = run_one(g, N, Ω, z; report_every=cfg.report_every)
        show_result(r)
        push!(rs, r)
        save_results(rs)
        log_msg(@sprintf("  保存: g=%d N=%d rel_err=%.2e speedup=%.1fx",
            g, N, r["rel_err"], get(r,"speedup",NaN)))
    end

    summary(rs)
    log_msg("完了。")
end

main()
