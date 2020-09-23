function _logsumexp(a, b)
    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m))
end
