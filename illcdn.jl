"""
 Functions that pertains to ill-conditionned linear programming example
"""

#-------------------------------------------------------------------------------
"""
 Usage:
 x=initill(n)
 Sets the global constant matrices and vectors of the dimension <n>
 and returns the standard starting point <x>
 for the Ill-conditioned Linear Programming problem.
"""

function initill(n)

    global matrA, vectB, vectC

    x = zeros(n)
    matrA = zeros(n,n)
    vectB = zeros(n)
    vectC = zeros(n)

    for i in 1:n
        x[i]=0
        vectB[i]=0;
        for j in 1:n
            matrA[i,j]=1/(i+j)
            vectB[i]=vectB[i]+1/(i+j)
        end
        vectC[i]=-1/(i+1)-vectB[i]
    end

    return x
end

#-------------------------------------------------------------------------------
"""
 Usage:
 f=illclinf(x)
 Returns value <f> of the exact penalty function
 at a point <x> for the Ill-conditioned Linear Programing problem.
"""

function illclinf(x)

    global matrA, vectB, vectC

    n=size(x,1)

    f= vectC'*x + 2*n*maximum([0;matrA*x-vectB;-x])

    return f
end
