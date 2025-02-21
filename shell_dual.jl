"""
 Usage:
 x=initdual()
 Sets the global constant matrices and vectors
 and returns the standard starting point <x>
 for Shell Dual Problem
"""

function initdual()

    global Amatr, Bvec, Cmatr, Dvec, Evec

    Amatr= [-16 2 0 1  0;
         0 -2 0 .4 2;
        -3.5 0 2 0 0;
         0 -2 0 -4 -1;
         0 -9 -2 1 -2.8;
         2  0 -4 0  0;
        -1 -1 -1 -1 -1;
        -1 -2 -3 -2 -1;
         1  2  3  4  5;
         1  1  1  1  1]

   Bvec = [-40, -2, -.25, -4, -4, -1 ,-40, -60, 5, 1]

   Cmatr = [ 30 -20 -10  32 -10;
       -20  39  -6 -31  32;
       -10  -6  10  -6 -10;
        32 -31  -6  39 -20;
       -10  32 -10 -20  30]

    Dvec = [ 4, 8, 10, 6, 2]

    Evec = [ -15, -27, -36, -18, -12]

    x= [1.e-4,1.e-4,1.e-4,1.e-4,1.e-4,1.e-4,1.e-4,1.e-4,1.e-4,1.e-4,1.e-4,60,1.e-4,1.e-4,1.e-4]

    return x
end

"""
 Usage:
 f=dsobjf(x)
 Calculates the objective function value <f>
 at a point <x> for Shell Dual Problem
"""
function dsobjf(x)

    global Amatr, Bvec, Cmatr, Dvec, Evec

    f = 2*Dvec'*x[1:5].^3 + (Cmatr*x[1:5])'*x[1:5] - Bvec'*x[6:15]

    return f
end

"""
 Usage:
 g=dsobjg(x)
 Calculates the gradient <g> of the objective function
 at a point <x> for Shell Dual Problem
"""

function dsobjg(x)
    global Amatr, Bvec, Cmatr, Dvec, Evec

    g=zeros(size(x,1))
    g[1:5]=6*Dvec.*x[1:5].^2 + 2*Cmatr*x[1:5]
    g[6:15]= - Bvec
    return g
end

"""
 Usage:
 f=dscntf(x)
 Calculates the maximal residual <f> for the set of constraints
 at a point <x> for Shell Dual Problem
"""

function dscntf(x)
    global Amatr, Bvec, Cmatr, Dvec, Evec

    f=maximum([Amatr'*x[6:15] - 2*Cmatr*x[1:5] - 3*Dvec.*x[1:5].^2 - Evec; -x])
    return f
end

"""
 Usage:
 g=dscntg(x)
 Calculates the gradient <g> of the constraint with the maximal
 residual at a point <x> for Shell Dual Problem
"""

function dscntg(x)
    global Amatr, Bvec, Cmatr, Dvec, Evec
    g=zeros(size(x,1))
    f,k =findmax([Amatr'*x[6:15] - 2*Cmatr*x[1:5] - 3*Dvec.*x[1:5].^2 - Evec; -x])

    if f>0
        if k>5
            g[k-5]=-1
        else
            g[6:15]=Amatr[:,k]
            g[1:5]=-2*Cmatr[:,k]
            g[k]=g[k]-6*Dvec[k]*x[k]
        end
    end

    return g
end
