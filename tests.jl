#######################################################################
# Tests problems for the SOLVOPT algorithm developed by               #
# Alexei Kuntsevich Franz Kappel, 1997.                               #
#                                                                     #
# This script was adapted to Julia language by                        #
# Etienne Briand, 2024. All errors are mine, please report them at    #
# etiennebriand1@gmail[dot]com                                        #
#                                                                     #
# Computation time is order of magnitude faster than the              #
# Matlab version of the code.                                         #
#######################################################################

# Load packages
using BenchmarkTools

# Load necessary Julia scripts
include("SOLVOPT.jl")
include("shor.jl")
include("illcdn.jl")
include("shell_dual.jl")

#-------------------------------------------------------------------------------
"""
 1. Unconstrained Problems
"""

# 1.1 Optimize the function 'shor' without supplying a function that computes
# the gradient.

# shor's global minimum is: 22.6002
# and is reached at x = [1.1246, 0.9794, 1.4777, 0.9206, 1.1242]

x = [-1.;1.;-1.;1.;-1.] #define a starting point

# Optimize
#@benchmark xopt, fopt = solvopt(x,:shorf)  #for complete evaluation
@time xopt, fopt = solvopt(x,:shorf)

#-------------------------------------------------------------------------------
# 1.2 Optimize the function 'shor' using a function that computes the gradient.

x = [-1.;1.;-1.;1.;-1.] #define a starting point

# Optimize
#@benchmark xopt, fopt = solvopt(x,:shorf,grad=:shorg)  #for complete evaluation
@time xopt, fopt = solvopt(x,:shorf,grad=:shorg)
#-------------------------------------------------------------------------------
"""
 2. Ill-conditioned Linear Programming Problems
"""

n = 15
x = initill(n)   #initiate global variables

@time xopt,fopt=solvopt(x,:illclinf)

#optimum at -20.04
#-------------------------------------------------------------------------------
"""
 3. Shell Dual Problem
"""

x = initdual()
f = dsobjf(x)
g = dsobjg(x)
dscntf(x)
dscntg(x)

# This point is one of many optima
#x = [0.300312599798172, 0.333304872799738,0.400174460225146, 0.428074299213949, 0.224100244274714,
#      -3.12747905944624e-09, 2.75198862961992e-08, 5.17051787552501, 1.01334109644349e-08,
#      3.06183405968730, 11.8365155748161, -8.05128638756654e-09, 1.94720668327622e-10, 0.103854646599024,
#      1.43255939885435e-07]
# but the optimal solution is unique at f = 32.3487

xopt,fopt = solvopt(x,:dsobjf,grad=:dsobjg,func=:dscntf,gradc=:dscntg)
