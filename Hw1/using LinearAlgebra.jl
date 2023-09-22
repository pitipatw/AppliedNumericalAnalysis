using LinearAlgebra
k = 4
d = 3
N = 10

c = rand(1:k,N)
A = rand(1:5,k,d)
X = rand(1:5,d,N)

C = zeros(k,N)
for i in axes(C,1)
    for j in axes(C,2)
        if i == c[j]
            C[i,j] = 1
        end

    end
end
println("#####")
println(c)
println("&&&&")
for i in axes(C,1)
    println(C[i,:])
end

println("&&&&")
AX = A*X
for i in axes(AX,1)
    println(AX[i,:])
end

println("&&&&")
for i in axes(C,1)
    println(C[i,:])
end

println("&&&&")
CtAX = C'*A*X
for i in axes(CtAX,1)
    println(CtAX[i,:])
end
