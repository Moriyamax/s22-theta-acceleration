"""
theta_comparison.jl  ― 修正版 (バグ修正: A1/A2/B2/D2/D3/E1)

修正内容:
  [A1] S22外データ汚染: resume=false での再実行を推奨。
       load_results の Inf/NaN 対応も兼ねて D2 も修正。
  [A2] atomic write: 一時ファイル経由で全データ消失を防止。
  [B2] 3重並列化を解消: main ループを逐次化、theta_naive/theta_s22 の内部並列は維持。
  [D2] Inf/NaN/負数 のパース対応を load_results に追加。
  [D3] omega_off が Ωon をベースにするよう修正（delta=0 → S22上と一致）。
  [D4] |θ| < 10 のとき警告ログを追加。
  [E1] デフォルト値の不整合を修正（seed=42 に統一）。

【使い方】
  julia --threads auto theta_comparison_fixed.jl
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
    delta::Float64
    seed::Int
    resume::Bool
    report_every::Int
end

function load_cfg()::Cfg
    # [E1修正] デフォルト値を統一（seed=42）
    defaults = Cfg([2,3,4,5,6,7,8,9,10,11,12,13,14],[1,2],0.5,42,true,1_000_000)
    if !isfile(CONFIG_PATH)
        open(CONFIG_PATH,"w") do f
            println(f,"# theta_comparison_fixed.jl config")
            println(f,"g_list = 2,4,6,8")
            println(f,"N_cut_list = 1")
            println(f,"delta = 0.5")
            println(f,"seed = 42")   # [E1修正] defaults と統一
            println(f,"resume = true")
            println(f,"report_every = 10000000")
        end
        log_msg("設定ファイル生成: $CONFIG_PATH")
        # [E1修正] config を生成後は読み直す（return defaults はしない）
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
        haskey(kv,"delta")        ? parse(Float64,kv["delta"])              : d.delta,
        haskey(kv,"seed")         ? parse(Int,kv["seed"])                   : d.seed,
        haskey(kv,"resume")       ? lowercase(kv["resume"])=="true"         : d.resume,
        haskey(kv,"report_every") ? parse(Int,kv["report_every"])           : d.report_every,
    )
end

# ───────── JSON保存 [A2修正: atomic write] ─────────
function save_results(rs::Vector{Dict})
    tmp = RESULT_PATH * ".tmp"
    open(tmp,"w") do f
        println(f,"[")
        for (i,r) in enumerate(rs)
            println(f,"  {")
            println(f,"    \"g\": $(r["g"]),")
            println(f,"    \"N_cut\": $(r["N_cut"]),")
            lbl = replace(replace(r["label"],"\\\"=>"\\\\"),"\"=>"\\\"")
            println(f,"    \"label\": \"$lbl\",")
            println(f,"    \"val_naive_re\": $(r["val_naive_re"]),")
            println(f,"    \"val_naive_im\": $(r["val_naive_im"]),")
            println(f,"    \"val_s22_re\": $(r["val_s22_re"]),")
            println(f,"    \"val_s22_im\": $(r["val_s22_im"]),")
            # [D2修正] Inf/NaN を文字列として保存してパース時に対応
            println(f,"    \"rel_err\": $(r["rel_err"]),")
            println(f,"    \"t_naive\": $(r["t_naive"]),")
            println(f,"    \"t_s22\": $(r["t_s22"]),")
            println(f,"    \"speedup\": $(r["speedup"]),")
            print(f,  "    \"timestamp\": \"$(r["timestamp"])\"")
            print(f, "\n  }")
            i < length(rs) ? println(f,",") : println(f)
        end
        println(f,"]")
    end
    mv(tmp, RESULT_PATH; force=true)  # [A2修正] atomic rename
end

# [D2修正] Inf/NaN 対応パーサー
function parse_float_robust(v::String)::Union{Float64,Nothing}
    v == "Inf"  && return Inf
    v == "-Inf" && return -Inf
    v == "NaN"  && return NaN
    return tryparse(Float64, v)
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
                # 末尾の " だけ除去（先頭の " も除去）
                cur[k] = v[2:end-1]
            elseif v=="true";  cur[k]=true
            elseif v=="false"; cur[k]=false
            else
                fv = parse_float_robust(v)  # [D2修正]
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

done_key(r::Dict) = (get(r,"g",nothing), get(r,"N_cut",nothing), get(r,"label",""))
already_done(rs,g,N,lbl) = any(r->done_key(r)==(g,N,lbl), rs)

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
            Omn = Vector{ComplexF64}(undef, g)  # [C2修正] 事前確保
            for idx in i_start:i_end
                tmp = idx
                for k in 1:g
                    nv[k] = Float64(tmp % b - N_cut)
                    tmp ÷= b
                end
                mul!(Omn, Omega, nv)             # [C2修正] アロケーションなし
                qc = dot(nv, Omn)
                lc = dot(nv, z)
                local_sum += exp(im*π*qc + 2im*π*lc)
                if report_every > 0
                    cnt = Threads.atomic_add!(counter, 1) + 1
                    if cnt % report_every == 0
                        pct = 100.0*cnt/total
                        lbl2 = isempty(label) ? "" : " [$label]"
                        log_msg(@sprintf("    [theta_naive%s] %d / %d (%.2f%%)", lbl2, cnt, total, pct))
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

# ───────── テストデータ生成 ─────────
function pos_block(n::Int, rng)
    A = randn(rng,Float64,n,n).*0.3
    return im.*(A*A' .+ Matrix(1.5I,n,n))
end

function omega_on(g::Int, rng)
    g1=g÷2; Ω=zeros(ComplexF64,g,g)
    Ω[1:g1,1:g1]         = pos_block(g1,rng)
    Ω[g1+1:end,g1+1:end] = pos_block(g-g1,rng)
    return (Ω+Ω')./2
end

# [D3修正] Ωon をベースにして摂動を加える。delta=0 なら Ωon と一致。
function omega_off(Ωon::Matrix{ComplexF64}, g::Int, delta::Float64, rng)
    g1=g÷2; g2=g-g1
    Ω = copy(Ωon)                                    # [D3修正] Ωon をコピー
    p = im.*randn(rng,Float64,g1,g2).*0.3.*delta
    Ω[1:g1,g1+1:end]=p; Ω[g1+1:end,1:g1]=conj.(p')
    return (Ω+Ω')./2
end

function make_z(g::Int, rng)
    complex.(randn(rng,Float64,g).*0.5, randn(rng,Float64,g).*0.1)
end

function make_data(g::Int, delta::Float64, seed::Int)
    z   = make_z(g, MersenneTwister(seed+g))
    Ωon = omega_on(g, MersenneTwister(seed+g+10000))
    Ωoff = omega_off(Ωon, g, delta, MersenneTwister(seed+g+20000))  # [D3修正]
    return z, Ωon, Ωoff
end

# ───────── 比較実行 ─────────
function run_one(g,N,Ω,z,label; report_every=0)::Dict
    b=2N+1; pts=b^g
    log_msg(@sprintf("  → naive 開始: g=%d N=%d [%s] 格子点=%d スレッド=%d",
        g,N,label,pts,Threads.nthreads()))
    t0=time(); vn=theta_naive(z,Ω,N; report_every, label="naive/g$g/N$N"); tn=time()-t0
    log_msg(@sprintf("  ← naive 完了: %.4fs  val=%.6f%+.6fi", tn, real(vn),imag(vn)))

    # [D4修正] |θ| が小さい場合に警告
    if abs(vn) < 10.0
        log_msg(@sprintf("  ⚠ 警告: |θ|=%.4f < 10（ゼロ点近傍）。rel_errが大きくなる可能性あり。別seedでの再測定を推奨。", abs(vn)))
    end

    log_msg(@sprintf("  → s22  開始: g=%d N=%d [%s]", g,N,label))
    t0=time(); vs=theta_s22(z,Ω,N; report_every, label="s22/g$g/N$N"); ts=time()-t0
    log_msg(@sprintf("  ← s22  完了: %.4fs  val=%.6f%+.6fi", ts, real(vs),imag(vs)))

    re = abs(vn)<1e-300 ? NaN : abs(vn-vs)/abs(vn)
    sp = ts>1e-12 ? tn/ts : Inf
    Dict("g"=>g,"N_cut"=>N,"label"=>label,
         "val_naive_re"=>real(vn),"val_naive_im"=>imag(vn),
         "val_s22_re"  =>real(vs),"val_s22_im"  =>imag(vs),
         "rel_err"=>re,"t_naive"=>tn,"t_s22"=>ts,"speedup"=>sp,
         "timestamp"=>Dates.format(now(),"yyyy-mm-ddTHH:MM:SS"))
end

function show_result(r::Dict)
    @printf "    g=%d N=%d [%s]\n" r["g"] r["N_cut"] r["label"]
    @printf "      naive: %.6f%+.6fi (%.4fs)\n" r["val_naive_re"] r["val_naive_im"] r["t_naive"]
    @printf "      s22  : %.6f%+.6fi (%.4fs)\n" r["val_s22_re"]   r["val_s22_im"]   r["t_s22"]
    sp = get(r, "speedup", NaN)
    @printf "      rel_err=%.2e  speedup=%.1fx\n" r["rel_err"] sp
    flush(stdout)
end

# ───────── サマリー ─────────
function summary(rs::Vector{Dict})
    log_msg("="^70)
    log_msg("サマリー")
    log_msg(@sprintf("%3s %2s %-20s %12s %10s %10s %8s","g","N","label","rel_err","naive(s)","s22(s)","倍率"))
    log_msg("-"^70)
    for r in sort(rs, by=r->(get(r,"g",0),get(r,"N_cut",0),get(r,"label","")))
        lbl = occursin("S22上",r["label"]) ? "S22上" : "S22外"
        sp = get(r, "speedup", NaN)
        log_msg(@sprintf("%3d %2d %-20s %12.2e %10.4f %10.4f %7.1fx",
            r["g"],r["N_cut"],lbl,r["rel_err"],r["t_naive"],r["t_s22"],sp))
    end
    log_msg("結果: $RESULT_PATH  ログ: $LOG_PATH")
end

# ───────── main [B2修正: 逐次実行に変更] ─────────
function main()
    cfg = load_cfg()
    rs::Vector{Dict} = cfg.resume ? load_results() : Dict[]

    log_msg("="^70)
    log_msg("theta_comparison_fixed.jl  naive vs s22")
    log_msg("Julia $(VERSION)  スレッド=$(Threads.nthreads())")
    log_msg("g=$(cfg.g_list)  N_cut=$(cfg.N_cut_list)  delta=$(cfg.delta)  seed=$(cfg.seed)")
    log_msg("report_every=$(cfg.report_every)  resume=$(cfg.resume)  既完了=$(length(rs))件")
    log_msg("="^70)

    LON  = "S22上 (ε=0厳密)"
    LOFF = @sprintf("S22外 (delta=%.1f)", cfg.delta)

    cases = [(g,N,lbl)
             for g   in cfg.g_list
             for N   in cfg.N_cut_list
             for lbl in [LON, LOFF]]
    total = length(cases)

    # [B2修正] 逐次実行（外ループは逐次、theta_naive/s22 内部の並列は維持）
    for (n,(g,N,lbl)) in enumerate(cases)
        if cfg.resume && already_done(rs,g,N,lbl)
            log_msg(@sprintf("  スキップ [%d/%d] g=%d N=%d [%s]",n,total,g,N,lbl))
            i = findfirst(r->done_key(r)==(g,N,lbl), rs)
            i !== nothing && show_result(rs[i])
            continue
        end

        log_msg(@sprintf("\n─── [%d/%d] (%.1f%%) g=%d N=%d [%s] ───",
            n,total,100.0*n/total,g,N,lbl))
        z,Ωon,Ωoff = make_data(g,cfg.delta,cfg.seed)
        Ω = occursin("S22上",lbl) ? Ωon : Ωoff
        r = run_one(g,N,Ω,z,lbl; report_every=cfg.report_every)
        show_result(r)
        push!(rs,r)
        save_results(rs)  # [A2修正] atomic write で保護済み
        log_msg(@sprintf("  保存: g=%d N=%d [%s] rel_err=%.2e speedup=%.1fx",
            g,N,lbl,r["rel_err"],get(r,"speedup",NaN)))
    end

    summary(rs)
    log_msg("完了。")
end

main()
