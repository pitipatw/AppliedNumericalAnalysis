println("""
1a. A projection matrix to the null space.
    Null space of AT is the solution vectors x where A^{T}x = 0.
    Maybe 

    An arbitrary vector v (m by 1)
    if x is null space of a
    P0v = cx

    multiply by A{T}
    A^{T}P0v = cA^{T}x = 0 
    A^{T}P0v = 0 
    
    I know that for matrix, ABC = 0 does not imply A =0 or B= 0 or C = 0,
        but it's very temping to substitute QQT = I ,and let the P0 term become 0, is this just a special case, or something? 

        I know that A = QR is unique, therefore, P0 always = 0 ?
        So

    ( Null space AT)
    PB = cx
    ATPB = cATx 
    ATPB = 0 
    **ATP perpendicular to B?

    (RTQT)(1 - QQT)

    RTQT - RTQT QQT

    ...




1b. Review Cholesky Fac.
        gram smidth algorithm.

1c. What is the power iteration? 
    What is the domain eigenvalue of A = uvT.

2. Is this a translation matrix? 
    RTR = I means what? 

    Rotational+translational matrixes can be written in the form Rx + b

3. move nearly rigidly -> in numerical precision? 
 a. Is R another function? or just another matrix?
    If it's just another matrix, why R(qi)
    or the second term is multiplication, then

    since f(q) = Rq + b and f(p) = R(p) + b 

    Therefore, f(qi) - f(p)  = those 2 vectors -> R(qi-p).
 b. This is basically minimizing the distance between points of new points and the original point.
    We can work on the given problem into a normal equation, and use LU decomposition or something to solve it.

 c. Work on i element (subset of the whole matrix problem)



4.  
 a. 
    p is the matrix of the original point
    p' is matrix of the new points.
    Ri is matrix that transforms p to p'
    wij is the weight,
        RtR constraint for what? 

    Objective function : 
        Finding pi' (deformed) and R (the function) that minimize the different the distance betweenthe original pi and pj to p'i and p'just
        Subjected to,
            p'i is pi'bar  >Why do we do this? , why cant it be embeded right away? 

            RTR = I, this is probably due to the requirements at the begining. 
    b. we are assuming symmetry. Take the gradient respect to p',
        Turn the equation into the normal equation first, then take the derivative, should be straight forward.

    c. SOlving p? normal step for the normal equation. 
    d. This is solving p, from the previous part, but put into an n x n matrix.
    
    e. implement to code. 

5. Rrecord the steps for getting A (LU factorization) and post multiplying the result? 
        The code alternating opimization,
""")