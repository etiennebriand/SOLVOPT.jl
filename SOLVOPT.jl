#=
This file contains the SOLVOPT alogorithm for constrained optimization
and the auxiliary function apprgrdn (gradient approximation)
=#

using LinearAlgebra
#-------------------------------------------------------------------------------

function solvopt(x::Vector,fun;grad=[],options=[],func=[],gradc=[])

    """Usage:

     x,f,options = solvopt(x,fun,grad,options,func,gradc)
     The function SOLVOPT performs a modified version of Shor's r-algorithm in
     order to find a local minimum resp. maximum of a nonlinear function
     defined on the n-dimensional Euclidean space

     or

     a solution of a nonlinear constrained problem:
     min { f(x): g(x) (<)= 0, g(x) in R(m), x in R(n) }

     Arguments:
     -x       is the n-vector of the coordinates of the starting point

     -fun     is a Symbol referring to Julia file which computes the value
              of the objective function <fun> at a point x,
              synopsis: f=fun(x)

    -grad     is a Symbol referring to Julia file which computes the gradient
              vector of the function <fun> at a point x,
              synopsis: g=grad(x)

    -func     is a Symbol referring to Julia file which computes the MAXIMAL
              RESIDUAL(!) for a set of constraints at a point x,
              synopsis: fc=func(x)

    -gradc    is a Symbol referring to Julia file which computes the gradient
              vector for the maximal residual consyraint at a point x,
              synopsis: gc=gradc(x)

    -options  is a vector of optional parameters:

       -options(1)= H, where sign(H)=-1 resp. sign(H)=+1 means minimize
             resp. maximize <fun> (valid only for unconstrained problem)
             and H itself is a factor for the initial trial step size
             (options(1)=-1 by default),

       -options(2)= relative error for the argument in terms of the
             infinity-norm (1.e-4 by default),

       -options(3)= relative error for the function value (1.e-6 by default),

       -options(4)= limit for the number of iterations (15000 by default),

       -options(5)= control of the display of intermediate results and error
             resp. warning messages (default value is 0, i.e., no intermediate
             output but error and warning messages, see more in the manual),

       -options(6)= admissible maximal residual for a set of constraints
             (options(6)=1e-8 by default, see more in the manual),

       -options(7)= the coefficient of space dilation (2.5 by default),

       -options(8)= lower bound for the stepsize used for the difference
             approximation of gradients (1e-12 by default, see more in the manual).
       (* ... changes should be done with care)

     Returned values:
      -x       is the optimizer,

      -f       is the optimum function value,

      -options returns the values of the counters

         -options(9),  the number of iterations, if positive,
             or an abnormal stop code, if negative (see more in the manual),

         -options(10), the number of objective

         -options(11), the number of gradient evaluations,

         -options(12), the number of constraint function evaluations,

         -options(13), the number of constraint gradient evaluations.
    """

    #ARGUMENTS PASSED ----{
    if grad == []
        app = 1             #No user-supplied gradients
    else
        app = 0             #Exact gradients are supplied
    end

    if func == []
        constr = 0           #Unconstrained problem
        appconstr = 0
    else
        constr = 1           #Constrained problem
        if gradc == []
            appconstr = 1     #No user-supplied gradients for constraints
            t = 3
        else
            appconstr = 0     #Exact gradients of constraints are supplied
        end
    end
    #----}

    #OPTIONS ----{
    doptions = [-1, 1.e-4, 1.e-6, 15000, 0, 1.e-8, 2.5, 1e-11] #default options
    nop = size(doptions,1)

    if options == []
        options = doptions
    else
        for ii = 1:nop
            if options[ii] != 0
                doptions[ii] = options[ii]  #replace default options by user specified options
            end
        end
        options = doptions
    end

    options = [options; 0; 0; 0; 0; 0]

    #Check values
    options[[2; 3; 6; 7; 8]] = abs.([options[2:3]; options[6:8]])
    options[2:3] = max(options[2:3], [1.e-12, 1.e-12])
    options[2] = max(options[8] * 1.e2, options[2])
    options[2:3] = min(options[2:3], [1, 1])
    options[6] = max(options[6], 1e-12)
    options[7] = max(options[7], 1.5)
    options[8] = max(options[8], 1e-11)
    #----}

    #define some strings: ----{
    errmes = "SolvOpt error:"
    wrnmes = "SolvOpt warning:"
    error31 = "Function value does not exist (NaN is returned)."
    error32 = "Function equals infinity at the point."
    error40 = "<grad> returns an improper matrix. Check the dimension."
    error41 = "Gradient does not exist (NaN is returned by <grad>)."
    error42 = "Gradient equals infinity at the starting point."
    error43 = "Gradient equals zero at the starting point."
    error51 = "<func> returns NaN at the point."
    error52 = "<func> returns infinite value at the point."
    error60 = "<gradc> returns an improper vector. Check the dimension"
    error61 = "<gradc> returns NaN at the point."
    error62 = "<gradc> returns infinite vector at the point."
    error63 = "<gradc> returns zero vector at an infeasible point."
    error5  = "Function is unbounded."
    error6  = "Choose another starting point."
    warn1   = "Gradient is zero at the point, but stopping criteria are not fulfilled."
    warn20  = "Normal re-setting of a transformation matrix."
    warn21  = "Re-setting due to the use of a new penalty coefficient."
    warn4   = "Iterations limit exceeded."
    warn31  = "The function is flat in certain directions."
    warn32  = "Trying to recover by shifting insensitive variables."
    warn09  = "Re-run from recorded point."
    warn08  = "Ravine with a flat bottom is detected."
    termwarn0 = "SolvOpt: Normal termination."
    termwarn1 = "SolvOpt: Termination warning:"
    appwarn = "The above warning may be reasoned by inaccurate gradient approximation"
    endwarn = [
        "Premature stop is possible. Try to re-run the routine from the obtained point."
        "Result may not provide the optimum. The function apparently has many extremum points."
        "Result may be inaccurate in the coordinates. The function is flat at the optimum."
        "Result may be inaccurate in a function value. The function is extremely steep at the optimum."
        ]
    # ----}

    #STARTING POINT ----{
        n = size(x,1)
    #----}

    #WORKING CONSTANTS AND COUNTERS ----{
    epsnorm = 1.e-15
    epsnorm2 = 1.e-30                   #epsilon & epsilon^2

    if constr == 1
        h1 = -1                         #NLP: restricted to minimization
        cnteps = options[6]             #Max. admissible residual
    else
        h1 = sign(options[1])           #Minimize resp. maximize a function
    end

    k = 0                               #Iteration counter
    wdef = 1 / options[7] - 1           #Default space transf. coeff.

    #Gamma control ---{
    ajb = 1 + 0.1 / n^2                  #Base I
    ajp = 20
    ajpp = ajp                           #Start value for the power
    ajs = 1.15                           #Base II
    knorms = 0
    gnorms = zeros(1, 10)                #Gradient norms stored
    #---}

    #Display control ---{
    if options[5] <= 0
        dispdata = 0
        if options[5] == -1
            dispwarn = 0
        else
            dispwarn = 1
        end
    else
        dispdata = round(options[5])
        dispwarn = 1
    end

    ld = dispdata
    #---}

    #Stepsize control ---{
    dq = 5.1                           #Step divider (at f_{i+1}>gamma*f_{i})
    du20 = 2
    du10 = 1.5
    du03 = 1.05                        #Step multipliers (at certain steps made)
    kstore = 3
    nsteps = zeros(1, kstore)

    if app == 1
        des = 6.3                      #Desired number of steps per 1-D search
    else
        des = 3.3
    end

    mxtc = 3                           #Number of trial cycles (steep wall detect)
    #---}

    termx = 0
    limxterm = 50;                    #Counter and limit for x-criterion
    ddx = max(1e-11, options[8])      #stepsize for gradient approximation
    low_bound = -1 + 1e-4             #Lower bound cosine used to detect a ravine
    ZeroGrad = n * 1.e-16             #Lower bound for a gradient norm
    nzero = 0                         #Zero-gradient events counter
    lowxbound = max(options[2], 1e-3) #Lower bound for values of variables taking into account
    lowfbound = options[3]^2          #Lower bound for function values to be considered as making difference
    krerun = 0                        #Re-run events counter
    detfr = options[3] * 100          #relative error for f/f_{record}
    detxr = options[2] * 10           #relative error for norm(x)/norm(x_{record})
    warnno = 0                        #the number of warn.mess. to end with
    kflat = 0                         #counter for points of flatness
    stepvanish = 0                    #counter for vanished steps
    stopf = 0
    kd=0   #NOTE: i added this
    #----}  End of setting constants
    #----}  End of the preamble

    #COMPUTE THE FUNCTION  ( FIRST TIME ) ----{
    f = eval(fun)(x)
    options[10] = options[10] + 1

    if isnan(f)==true
        if dispwarn == true
            show(errmes*error31*error6);
        end
        options[9]=-3
        return
    elseif abs(f)==Inf
        if dispwarn==true
            show(errmes*error32*error6);
        end
        options[9]=-3;
        return
    end

    xrec = x
    frec = f     #record point and function value

    #Constrained problem
    if constr == 1
        fp = f
        kless = 0

        fc = eval(func)(x)

        if isnan(fc) == true
            if dispwarn == true
                show(errmes*error51*error6)
            end
            options[9]=-5
            return
        elseif abs(fc)==Inf
            if dispwarn == true
                show(errmes*error52*error6);
            end
            options[9]=-5;
            return
        end

        options[12] = options[12] + 1
        PenCoef = 1                              #first rough approximation

        if fc <= cnteps
            FP = 1
            fc = 0                              #feasible point
        else
            FP = 0                              #infeasible point
        end

        f = f + PenCoef * fc

    end
    #----}

    #COMPUTE THE GRADIENT ( FIRST TIME ) ----{
    if app == 1
        deltax = h1 * ddx * ones(size(x))
        if constr == 1
            g = apprgrdn(x, fp, fun, deltax, 1)
        else
            g = apprgrdn(x, f, fun, deltax, 1)
        end
        options[10] = options[10] + n
    else
        g = eval(grad)(x)
        options[11] = options[11] + 1
    end

    ng = norm(g)

    if isnan(ng)==true
        if dispwarn == true
            show(errmes*error41*error6);
        end
        options[9]=-4
        return
    elseif ng==Inf
        if dispwarn == true
            show(errmes*error42*error6);
        end
        options[9]=-4
        return
    elseif ng<ZeroGrad
        if dispwarn == true
            show(errmes*error43*error6);
        end
        options[9]=-4
        return
    end

    if constr == 1
        if FP!=1
            if appconstr == 1
                deltax = sign.(x)
                idx = findall(x -> x == 0, deltax)
                deltax[idx] = ones(size(idx))
                deltax = ddx * deltax

                gc = apprgrdn(x, fc, func, deltax, 0)
                options[12] = options[12] + n
            else
                gc = eval(gradc)(x)
                options[13] = options[13] + 1
            end

            ngc = norm(gc)

            if isnan(ngc)==true
                if dispwarn == true
                    show(errmes*error61*error6)
                end
                options[9]=-6
                return x,f
            elseif ngc==Inf
                if dispwarn == true
                    show(errmes*error62*error6)
                end
                options[9]=-6
                return x,f
            elseif ngc<ZeroGrad
                if dispwarn == true
                    show(errmes*error63)
                end
                options[9]=-6
                return x,f
            end
        g = g + PenCoef * gc
        ng = norm(g)
        end
    end

    grec = g
    nng = ng
    #----}

    #INITIAL STEPSIZE
    h = h1 * sqrt(options[2]) * maximum(abs.(x))       #smallest possible stepsize
    if abs(options[1]) != 1
        h = h1 * maximum(abs.([options[1], h]))        #user-supplied stepsize
    else
        h = h1 * max(1 / log(ng + 1.1), abs.(h))       #calculated stepsize
    end

    #RESETTING LOOP ----{
while true
    kcheck = 0                       #Set checkpoint counter.
    kg = 0                           #stepsizes stored
    kj = 0                           #ravine jump counter
    B = I(n)                         #re-set transf. matrix to identity
    fst = f
    g1 = g
    dx = 0
    #----}

    #MAIN ITERATIONS ----{
    while true
        k = k+ 1
        kcheck = kcheck + 1
        laststep = dx

        #ADJUST GAMMA --{
        gamma = 1 + maximum([ajb^((ajp - kcheck) * n), 2 * options[3]])
        gamma = minimum([gamma, ajs^maximum([1, log10(nng + 1)])])
        #--}

        gt = (g' * B)'
        w = wdef

        #JUMPING OVER A RAVINE ----{
        if (gt' / norm(gt)) * (g1 / norm(g1)) < low_bound
            #dot((gt / norm(gt)), (g1' / norm(g1))') < low_bound

            if kj == 2
                xx = x
            end

            if kj == 0
                kd = 4
            end

            kj = kj + 1
            w = -.9
            h = h * 2             #use large coef. of space dilation

            if kj > 2 * kd      #NOTE: kd can be undefined
                kd = kd + 1
                warnno = 1
                if any(abs.(x - xx) < epsnorm * abs.(x)) #flat bottom is detected
                    if dispwarn == true
                        show(wrnmes * warn08)
                    end
                end
            end
        else
            kj = 0
        end
        #----}

        #DILATION ----{
        z = gt - g1
        nrmz = norm(z)

        if nrmz > epsnorm * norm(gt)
            z = z / nrmz
            g1 = gt + w * (z * gt') * z
            B = B + w * (B * z) * z'
        else
            z = zeros(n)
            nrmz = 0
            g1 = gt
        end
        d1 = norm(g1)
        g0 = (g1 / d1)' * B'
        #----}

        #RESETTING ----{
        if kcheck > 1
            idx = findall(x -> x > ZeroGrad, abs.(g))
            numelem = size(idx, 1)

            if numelem > 0
                grbnd = epsnorm * numelem^2
                if all(abs.(g1[idx]) <= abs.(g[idx]) * grbnd) || nrmz == 0
                    if dispwarn == true
                        show(wrnmes * warn20)
                    end
                    if abs(fst - f) < abs(f) * 0.01
                        ajp = ajp - 10 * n
                    else
                        ajp = ajpp
                    end
                    h = h1 * dx / 3
                    k = k - 1
                    break          #TODO
                end
            end
        end
        #----}

        #STORE THE CURRENT VALUES AND SET THE COUNTERS FOR 1-D SEARCH
        xopt = x
        fopt = f
        k1 = 0
        k2 = 0
        ksm = 0
        kc = 0
        knan = 0
        hp = h

        if constr == 1
            Reset = 0
        end

        #1-D SEARCH ----{
        while true
            x1 = x
            f1 = f
            if constr == 1
                FP1 = FP
                fp1 = fp
            end
            x = x + (hp * g0)'

            #FUNCTION VALUE
            f = eval(fun)(x)
            options[10] = options[10] + 1

            if h1 * f == Inf
                if dispwarn == true
                    show(errmes * error5)
                end
                options[9] = -7
                return x, f
            end

            if constr==1
                fp=f
                fc=eval(func)(x)
                options[12]=options[12]+1

                if isnan(fc) == true
                    if dispwarn==true
                        show(errmes*error51*error6)
                    end
                    options[9]=-5
                    return x,f

                elseif abs(fc)==Inf
                    if dispwarn==true
                        show(errmes*error52*error6)
                    end
                    options[9]=-5
                    return x,f
                end

                if fc<=cnteps
                    FP=1
                    fc=0
                else
                    FP=0
                    fp_rate=(fp-fp1)

                     if fp_rate < -epsnorm
                          if FP1!=1
                               PenCoefNew=-15*fp_rate/norm(x-x1)

                               if PenCoefNew > 1.2*PenCoef
                                  PenCoef=PenCoefNew
                                  Reset=1
                                  kless=0
                                  f=f+PenCoef*fc
                                  break
                               end
                          end
                     end
                  end
                  f=f+PenCoef*fc;
              end

            if abs(f) == Inf || isnan(f) == true
                if dispwarn == true
                    show(wrnmes)
                    if isnan(f) == true
                        show(error31)
                    else
                        show(error32)
                    end
                end

                if ksm == true || kc >= mxtc
                    options[9] = -3
                    return
                else
                    k2 = k2 + 1
                    k1 = 0
                    hp = hp / dq
                    x = x1
                    f = f1
                    knan = 1
                    if constr == 1
                        FP = FP1
                        fp = fp1
                    end
                end

    #STEP SIZE IS ZERO TO THE EXTENT OF EPSNORM
elseif all(abs.(x - x1) < abs.(x) * epsnorm)
                stepvanish = stepvanish + 1
                if stepvanish >= 5
                    options[9] = -14
                    if dispwarn == true
                        show(termwarn1 * endwarn[4, :][1])
                    end
                    return x,f
                else
                    x = x1
                    f = f1
                    hp = hp * 10
                    ksm = 1

                    if constr == 1
                        FP = FP1
                        fp = fp1
                    end
                end

    #USE SMALLER STEP
            elseif h1 * f < h1 * gamma^sign(f1) * f1
                if ksm == 1
                    break
                end

                k2 = k2 + 1
                k1 = 0
                hp = hp / dq
                x = x1
                f = f1

                if constr == 1
                    FP = FP1
                    fp = fp1
                end

                if kc >= mxtc
                    break
                end

    #1-D OPTIMIZER IS LEFT BEHIND
            else
                if h1 * f <= h1 * f1
                    break
                end

    #USE LARGER STEP
                k1 = k1 + 1

                if k2 > 0
                    kc = kc + 1
                end

                k2 = 0

                if k1 >= 20
                    hp = du20 * hp
                elseif k1 >= 10
                    hp = du10 * hp
                elseif k1 >= 3
                    hp = du03 * hp
                end
            end
        end
        #----}  End of 1-D search

        #ADJUST THE TRIAL STEP SIZE ----{
        dx = norm(xopt - x)

        if kg < kstore
            kg = kg + 1
        end

        if kg >= 2
            nsteps[2:kg] = nsteps[1:kg-1]
        end

        nsteps[1] = dx / (abs(h) * norm(g0))

        kk = sum(nsteps[1:kg] .* [kg:-1:1;])/sum([kg:-1:1;])

        if kk > des
            if kg == 1
                h = h * (kk - des + 1)
            else
                h = h * sqrt(kk - des + 1)
            end

        elseif kk < des
            h = h * sqrt(kk / des)
        end

        stepvanish = stepvanish + ksm
        #----}

        #COMPUTE THE GRADIENT ----{
        if app == 1
            deltax = sign.(g0)
            idx = findall(x -> x == 0, deltax)
            deltax[idx] = ones(size(idx))
            deltax = h1 * ddx * deltax

            if constr == 1
                g = apprgrdn(x, fp, fun, deltax, 1)
            else
                g = apprgrdn(x, f, fun, deltax, 1)
            end
            options[10] = options[10] + n
        else
            g = eval(grad)(x)
            options[11] = options[11] + 1
        end

        ng = norm(g)

        if isnan(ng) == true
            if dispwarn == true
                show(errmes * error41)
            end
            options[9] = -4
            return
        elseif ng == Inf
            if dispwarn == true
                show(errmes * error42)
            end
            options[9] = -4
            return
        elseif ng < ZeroGrad
            if dispwarn == treu
                show(wrnmes * warn1)
            end
            ng = ZeroGrad
        end
        #----}

        #Constraints:
        if constr == 1
            if FP != 1
                if ng < 0.01 * PenCoef
                    kless = kless + 1
                    if kless >= 20
                        PenCoef = PenCoef / 10
                        Reset = 1
                        kless = 0
                    end
                else
                    kless = 0
                end

            if appconstr == 1
                deltax = sign.(x)
                idx = findall(x -> x == 0, deltax)
                deltax[idx] = ones(size(idx))
                deltax = ddx * deltax

                gc = apprgrdn(x, fc, func, deltax, 0)

                options[12] = options[12] + n
            else
                gc = eval(gradc)(x)
                options[13] = options[13] + 1
            end

            ngc = norm(gc)

            if isnan(ngc) == true
                if dispwarn == true
                    show(errmes * error61)
                end
                options[9] = -6
                return x, f
            elseif ngc == Inf
                if dispwarn == true
                    show(errmes * error62)
                end
                options[9] = -6
                return x, f
            elseif ngc < ZeroGrad && appconstr != 1
                if dispwarn == true
                    show(errmes * error63)
                end
                options[9] = -6
                return x, f
            end

            g = g + PenCoef * gc
            ng = norm(g)

            if Reset == true
                if dispwarn == true
                    show(wrnmes * warn21)
                end
                h = h1 * dx / 3
                k = k - 1
                nng = ng
                break
            end
        end
    end
    #----}

        if h1 * f > h1 * frec
            frec = f
            xrec = x
            grec = g
        end

        if ng > ZeroGrad
            if knorms < 10
                knorms = knorms + 1
            end
            if knorms >= 2
                gnorms[2:knorms] = gnorms[1:knorms-1]
            end
            gnorms[1] = ng
            nng = (prod(gnorms[1:knorms]))^(1 / knorms)
        end

        #DISPLAY THE CURRENT VALUES ----{
        if k == ld
            show("Iter: $k Function: $f Step Value: $dx Gradient Norm: $ng")
            ld = k + dispdata
        end
        #----}

        #CHECK THE STOPPING CRITERIA ----{
        termflag = 1

        if constr == true
            if FP!=1
                termflag = 0
            end
        end

        if kcheck <= 5
            termflag = 0
        end

        if knan == 1
            termflag = 0
        end

        if kc >= mxtc
            termflag = 0
        end

        #ARGUMENT
        if termflag == 1
            idx = findall(x -> x >= lowxbound, abs.(x))

            if isempty(idx) || all( abs.( xopt[idx] .- x[idx]) <= options[2] * abs.(x[idx]) )
                termx = termx + 1

          #FUNCTION
                if abs(f - frec) > detfr * abs(f) && abs(f - fopt) <= options[3] * abs(f) && krerun <= 3 && constr != 1

                    if any(abs.(xrec[idx] - x[idx]) > detxr * abs.(x[idx]))

                        if dispwarn == 1
                            show(wrnmes * warn09)
                        end

                        x = xrec
                        f = frec
                        g = grec
                        ng = norm(g)
                        krerun = krerun + 1
                        h = h1 * maximum([dx, detxr * norm(x)]) / krerun
                        warnno = 2
                        break         #TODO
                    else
                        h = h * 10
                    end

               elseif  abs(f-frec)> options[3]*abs(f) && norm(x-xrec)<options[2]*norm(x) && constr==1


               elseif abs(f - fopt) <= options[3] * abs(f) || abs(f) <= lowfbound || (abs(f - fopt) <= options[3] && termx >= limxterm)
                    if stopf == true
                        if dx <= laststep
                            if warnno == 1 && ng < sqrt(options[3])
                                warnno = 0
                            end

                            if app != 1
                                if any(abs.(g) .<= epsnorm2)
                                    warnno = 3
                                end
                            end

                            if warnno != 0
                                options[9] = -warnno - 10

                                if dispwarn == true
                                    show(termwarn1 * endwarn[warnno, :][1])
                                    if app == true
                                        show(appwarn)
                                    end
                                end
                            else
                                options[9] = k

                                if dispwarn == true
                                    show(termwarn0)
                                end
                            end

                            return x,f
                        end

                    else
                        stopf = 1
                    end

                elseif dx < 1.e-12 * max(norm(x), 1) && termx >= limxterm
                    options[9] = -14
                    if dispwarn == true
                        show(termwarn1 * endwarn[4, :][1])
                        if app == true
                            show(appwarn)
                        end
                    end
                    x = xrec
                    f = frec
                    return x,f
                else
                    stopf = 0
                end
            end
        end

        #ITERATIONS LIMIT
        if k == options[4]
            options[9] = -9
            if dispwarn == true
                show(wrnmes * warn4)
            end
            return x, f
        end
        #----}

        #ZERO GRADIENT ----{
        if constr == 1
            if ng <= ZeroGrad
                if dispwarn == true
                    show(termwarn1 * warn1)
                end
                options[9] = -8
                return x, f
            end
        else
            if ng <= ZeroGrad
                nzero = nzero + 1
                if dispwarn == true
                    show(wrnmes * warn1)
                end
                if nzero >= 3
                    options[9] = -8
                    return x, f
                end
                g0 = -h * g0 / 2

                for i = 1:10
                    x = x + g0
                    f = eval(fun)(x)
                    options[10] = options[10] + 1
                    if abs(f) == Inf
                        if dispwarn == true
                            show(errmes * error32)
                        end
                        options[9] = -3
                        return x, f
                    elseif isnan(f) == true
                        if dispwarn == true
                            show(errmes * error32)
                        end
                        options[9] = -3
                        return x, f
                    end
                    if app == true
                        deltax = sign(g0)
                        idx = find(deltax == 0)
                        deltax[idx] = ones(size(idx))
                        deltax = h1 * ddx * deltax

                        g = apprgrdn(x, f, fun, deltax, 1)
                        options[10] = options[10] + n
                    else
                        g = feval(grad)(x)
                        options[11] = options[11] + 1
                    end

                    ng = norm(g)

                    if ng == Inf
                        if dispwarn == true
                            show(errmes * error42)
                        end
                        options[9] = -4
                        return x, f
                    elseif isnan(ng) == true
                        if dispwarn == true
                            show(errmes * error41)
                        end
                        options[9] = -4
                        return x, f
                    end
                    if ng > ZeroGrad
                    break                  #TODO
                    end
                end
                if ng <= ZeroGrad
                    if dispwarn == true
                        show(termwarn1 * warn1)
                    end
                    options[9] = -8
                    return x, f
                end
                h = h1 * dx
                break               #TODO
            end
        end
        #----}

        #FUNCTION IS FLAT AT THE POINT ----{
        if constr != 1 && abs(f - fopt) < abs(fopt) * options[3] && kcheck > 5 &&  ng < 1

            idx = findall(x -> x <= epsnorm2, abs.(g))
            ni = size(idx, 1)

            if ni >= 1 && ni <= n / 2 && kflat <= 3
                kflat = kflat + 1

                if dispwarn == true
                    show(wrnmes * warn31)
                end
                warnno = 1
                x1 = x
                fm = f
                for j in idx
                    y = x[j]
                    f2 = fm

                    if y == 0
                        x1[j] = 1
                    elseif abs(y) < 1
                        x1[j] = sign(y)
                    else
                        x1[j] = y
                    end

                    #TODO check here, what counts the i????
                    for i = 1:20
                        x1[j] = x1[j] / 1.15
                        f1 = eval(fun)(x1)
                        options[10] = options[10] + 1

                        if abs(f1) != Inf && isnan(f1) != 1
                            if h1 * f1 > h1 * fm
                                y = x1[j]
                                fm = f1
                            elseif h1 * f2 > h1 * f1
                                break                    #TODO
                            elseif f2 == f1
                                x1[j] = x1[j] / 1.5
                            end
                            f2 = f1
                        end
                    end
                    x1[j] = y
                end

                if h1 * fm > h1 * f
                    if app == true
                        deltax = h1 * ddx * ones(size(deltax))

                        gt = apprgrdn(x1, fm, fun, deltax, 1)
                        options[10] = options[10] + n
                    else

                        gt = eval(grad)(x1)
                        options[11] = options[11] + 1
                    end

                    ngt = norm(gt)
                    if ~isnan(ngt) & ngt > epsnorm2
                        if dispwarn == true
                            show(warn32)
                        end
                        options[3] = options[3] / 5
                        x = x1
                        g = gt
                        ng = ngt
                        f = fm
                        h = h1 * dx / 3
                        break                   #TODO
                    end
                end
            end
        end
        #----}

        end  #iterations
     end #restart

    return x, f #,options
end





#-------------------------------------------------------------------------------

function apprgrdn(x, f, fun, deltax, obj)
    """ Usage:
        g = apprgrdn(x,f,fun,deltax,obj)
        Function apprgrdn performs the finite difference approximation
        of the gradient <g> at a point <x>.

        <f> is the calculated function value at a point <x>

        <fun> is the name of the function, which calculates function values

        <deltax> is a vector of the relative stepsizes

        <obj> is the flag indicating whether the gradient of the objective
        function (1) or the constraint function (0) is to be calculated.
    """

    #n = maximum(size(x))
    n  = size(x,1)
    g = zeros(n)
    y = zeros(n)
    ee = ones(size(x))
    di = abs.(x)

    idx = zeros(n)
    for i = 1:n
        if di[i] < 5e-15
            #idx[i] = 1, #di[i] = 5e-15 * ee[idx]
            di[i] = 5e-15 * ee[i]
        end
    end
    di = deltax .* di

    if obj == 1
        idx = findall(x -> abs(x)< 2e-10,di)
        #idx = abs.(di) .< 2e-10
        di[idx] = 2e-10 * sign.(di[idx])
    else
        idx = findall(x -> abs(x)< 5e-15,di)
        #idx = abs.(di) .< 5e-15
        di[idx] = 5e-15 * sign.(di[idx])
    end

    for i = 1:n
        y[i] = x[i]
    end

    for i = 1:n
        y[i] = x[i] + di[i]
        fi = eval(fun)(y)

        if obj == 1
            if fi == f
                for j = 1:3
                    di[i] = di[i] * 10
                    y[i] = x[i] + di[i]
                    fi = eval(fun)(y)
                    if fi != f
                        break
                    end
                end
            end
        end

        g[i] = (fi - f) / di[i]

        if obj == 1
            if any(x -> x == i, idx)
                y[i] = x[i] - di[i]    #x[i] should not be changed
                fi = eval(fun)(y)
                g[i] = 0.5 * (g[i] + (f - fi) / di[i])
            end
        end


        y[i] = x[i]

    end

    return g
end
