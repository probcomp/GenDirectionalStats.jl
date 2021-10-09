import Distributions
using SpecialFunctions: besseli, logfactorial, loggamma

log_besseli_term(order, m, x) = -logfactorial(m) - loggamma(m + order + 1) + (2*m + order)*log(x/2)

isunitvec(v) = isapprox(sum(x^2 for x in v), 1.)

function logsumexp(arr)
    maxarr = maximum(arr)
    return maxarr + log(sum(exp.(arr .- maxarr)))
end

function _log_besseli(order, x; n_terms=100)
    log_terms = Vector{Float64}(undef, n_terms)
    for m in 0:n_terms-1
        log_terms[m+1] = log_besseli_term(order, m, x)
    end
    return logsumexp(log_terms)
end

const max_lookup_input = 50000
const log_besseli_0_lookup = Vector{Float64}(undef, max_lookup_input) # order 0 (used by plane rotation)
const log_besseli_1_lookup = Vector{Float64}(undef, max_lookup_input) # order 1 (used by 3D rotation)

function log_besseli(order, x; n_terms=Int(floor(x/2)) + 20)
    if x > 500.0 && (order == 0 || order == 1)
        if x > max_lookup_input
            error("log besseli not available for x > $max_loopup_input, got: x = $x")
        end
        floorx = Int(floor(x))
        if order == 0
            if !isassigned(log_besseli_0_lookup, floorx)
                log_besseli_0_lookup[floorx] = _log_besseli(0, floorx; n_terms=n_terms)
            end
            return log_besseli_0_lookup[floorx]
        elseif order == 1
            if !isassigned(log_besseli_1_lookup, floorx)
                log_besseli_1_lookup[floorx] = _log_besseli(1, floorx; n_terms=n_terms)
            end
            return log_besseli_1_lookup[floorx]
        else
            @assert false
        end
    else
        return log(besseli(order, x))
    end
end

function Distributions._vmflck(p, κ)
    T = typeof(κ)
    hp = T(p/2)
    q = hp - 1
    return q * log(κ) - hp * log2π - log_besseli(q, κ)
end
