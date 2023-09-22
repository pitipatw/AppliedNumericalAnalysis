using LinearAlgebra
N = 4
d = 5
X = rand(d,N)

a = rand(d)
α = rand(1)[1]

out1 = X*X'*a + α*a # α is a scalar
out2 = (X*X' + α*I)*a # α is a scalar

