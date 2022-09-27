#include <limits>
#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>

const int IMPLEMENTED_TYPE[] = {2, 4, 5}; // Change this when you implement new symfunc type!
const double epsilon = std::numeric_limits<double>::epsilon();

static inline unsigned int pair_hash(const int i, const int j) {
    auto a = std::minmax(i, j);
    return 65536*a.first + a.second;
}

static inline double pow_int2(const double &x, const double n) {
    double res,tmp;

    if (x < epsilon) return 0.0;
    int nn = abs(n);
    //int nn = (n > 0) ? n : -n;
    tmp = x;

    for (res = 1.0; nn != 0; nn >>= 1, tmp *= tmp)
        if (nn & 1) res *= tmp;

    return (n > 0) ? res : 1.0/res;
}

//deprecated code
static inline double pow_int(const double &x, const double n) {
    double res,tmp;

    if (x == 0.0) return 0.0; // FIXME: abs(x) < epsilon
    int nn = (n > 0) ? n : -n;
    tmp = x;

    for (res = 1.0; nn != 0; nn >>= 1, tmp *= tmp)
        if (nn & 1) res *= tmp;

    return (n > 0) ? res : 1.0/res;
}

static inline double linear(double x, double &deriv) {
	deriv = 1;
	return x;
}

static inline double sigm(double x, double &deriv) {
    double expl = 1./(1.+exp(-x));
    deriv = expl*(1-expl);
    return expl;
}

static inline double tanh(double x, double &deriv) {
    double expl = 2./(1.+exp(-2.*x))-1;
    deriv = 1.-expl*expl;
    return expl;
}

static inline double relu(double x, double &deriv) {

    if (x > 0) {
        deriv = 1.;
        return x;
    } else {
        deriv = 0.;
        return 0;
    }
}

static inline double selu(double x, double &deriv) {
    double alpha = 1.6732632423543772848170429916717;
    double scale = 1.0507009873554804934193349852946;
    if (x > 0) {
        deriv = scale;
        return deriv*x;
    } else {
        deriv = scale*alpha*exp(x);
        return deriv - scale*alpha;
   }
}

static inline double swish(double x, double &deriv) {
    double expl = 1./(1.+exp(-x));
    double dexpl = expl*(1-expl);
    deriv =  expl + x*dexpl;
    return x*expl;
}
static inline void cutf2_poly_noslot(const double dist, const double cutd, double& f, double& df) {
    double xval = dist/cutd;
    double x_minus_1 = xval-1.0;
    double x2 = xval*xval;
    f = x2*xval*(xval*(15.0-6.0*xval)-10.0)+1.0;
    df = -30.0*(x_minus_1*x_minus_1)*x2;
}

static inline void cutf2_noslot(const double dist, const double cutd, double& f, double& df) {
    double cos, sin;
    static const double H_PI = -M_PI*0.5;
    sincos(M_PI*dist/cutd, &sin, &cos);
    f = 0.5 * (1 + cos);
    df = H_PI * sin / cutd;
}

static inline void cutf2_poly(const double dist, const double cutd, double& f, double& df, int slot) {
    static double f_[3], df_[3], dist_[3], cutd_[3];
    if (dist_[slot] == dist && cutd_[slot] == cutd) {
        f = f_[slot];
        df = df_[slot];
        return;
    }
    double frac = dist / cutd;
    if (frac >= 1.0) {
        f = 0.0;
        df = 0.0;
    } else {
        double xval = dist/cutd;
        double x_minus_1 = xval-1.0;
        double x2 = xval*xval;
        f = x2*xval*(xval*(15.0-6.0*xval)-10.0)+1.0;
        df = -30.0*(x_minus_1*x_minus_1)*x2;
    }
    dist_[slot] = dist;
    cutd_[slot] = cutd;
    f_[slot] = f;
    df_[slot] = df;
}


//calculate cutoff function
static inline void cutf2_nocheck(const double dist, const double cutd, double& f, double& df, int slot) {
    static double f_[3], df_[3], dist_[3], cutd_[3];
    //if (dist_[slot] == dist && cutd_[slot] == cutd) {  -> after first cal, rest of symfuc has save value(we don't have to check distance)
    //if cutd is homogeneous to some group, these check can be removed too
    if (dist_[slot] == dist && cutd_[slot] == cutd) {
        f = f_[slot];
        df = df_[slot];
    } else {
        double cos, sin;
        sincos(M_PI*dist/cutd, &sin, &cos);
        f = 0.5 * (1 + cos);
        df = -0.5 * M_PI * sin / cutd;

        dist_[slot] = dist;
        cutd_[slot] = cutd;
        f_[slot] = f;
        df_[slot] = df;
    }
}

//double checking cutoff distance. if(frac >= 1.0) can be removed
//deprecated
static inline void cutf2(const double dist, const double cutd, double& f, double& df, int slot) {
    static double f_[3], df_[3], dist_[3], cutd_[3];
    if (dist_[slot] == dist && cutd_[slot] == cutd) {
        f = f_[slot];
        df = df_[slot];
        return;
    }
    double frac = dist / cutd;
    if (frac >= 1.0) {
        f = 0.0;
        df = 0.0;
    } else {
        double cos, sin;
        sincos(M_PI*frac, &sin, &cos);
        f = 0.5 * (1 + cos); // fc value 1/2 * cos (pi * Rij/Rc) + 1/2
        df = -0.5 * M_PI * sin / cutd; // df/dR = -pi/2 / Rc * sin ( pi * Rij/Rc )
    }
    dist_[slot] = dist;
    cutd_[slot] = cutd;
    f_[slot] = f;
    df_[slot] = df;
}

//G2 for simd
static inline double G2_no_precal(const double Rij, const double fc, const double dfdR, const double* par, double &deriv) {
    // par[0] = R_c
    // par[1] = eta
    // par[2] = R_s
    // precal[0] = fc
    // precal[1] = df/dR
    double tmp = Rij-par[2]; //Rij - Rs
    double expl = exp(-par[1]*tmp*tmp); // e^( eta * (Rs-Rij)^2 )
    deriv = expl*(-2*par[1]*tmp*fc + dfdR); // expl * { dexp(~)/dR * fc + df/dR } = expl * {2 * eta * (Rij - Rs) * fc + df/dR}
    return expl*fc; // G2 value itself (component of symmetry function)
}

static inline double G2(const double Rij, const double* precal, const double* par, double &deriv) {
    // par[0] = R_c
    // par[1] = eta
    // par[2] = R_s
    // precal[0] = fc
    // precal[1] = df/dR
    double tmp = Rij-par[2]; //Rij - Rs
    double expl = exp(-par[1]*tmp*tmp); // e^( eta * (Rs-Rij)^2 )
    deriv = expl*(-2*par[1]*tmp*precal[0] + precal[1]); // expl * { dexp(~)/dR * fc + df/dR } = expl * {2 * eta * (Rij - Rs) * fc + df/dR}
    return expl*precal[0]; // G2 value itself (component of symmetry function)
}

static inline double G4(const double Rij, const double Rik, const double Rjk, const double powtwo, \
          const double* precal_cutf, const double* precal_ang, const double* par, double *deriv, const bool powint) {

    double expl = exp(-par[1]*precal_ang[0]) * powtwo;
    // cos(theta) + 1
    double cosv = 1 + par[3]*precal_ang[1];
    //double powcos = powint ? pow_int2(fabs(cosv), par[2]-1) : pow(fabs(cosv), fabs(par[2]-1));
    // why zeta - 1?? removed fabs(abs version float) cosv is always > 0
    
    // (1+cos(theta)) ^ (zeta-1)  instead of zeta, to get derivative easier
    double powcos = powint ? pow_int2(cosv, par[2]-1) : pow(cosv, par[2]-1);
    //common factor
    double expl_powcos = expl*powcos;
    //zeta * lambda
    double par2_par3 = par[2]*par[3];

    //common factor exp, cos^zeta,f(Rjk), f(Rik)*{ dexp term               + df/dRij term    + dcosv/dRij term              }
    deriv[0] = expl_powcos*precal_cutf[2]*precal_cutf[4]* ((-2*par[1]*Rij*precal_cutf[0] + precal_cutf[1])*cosv + par2_par3*precal_cutf[0]*precal_ang[2]); // wrt Rij

    deriv[1] = expl_powcos*precal_cutf[0]*precal_cutf[4] * \
               ((-2*par[1]*Rik*precal_cutf[2] + precal_cutf[3])*cosv + \
               par2_par3*precal_cutf[2]*precal_ang[3]); // ik

    deriv[2] = expl_powcos*precal_cutf[0]*precal_cutf[2] * \
               ((-2*par[1]*Rjk*precal_cutf[4] + precal_cutf[5])*cosv + \
               par2_par3*precal_cutf[4]*precal_ang[4]); // jk

    //G4 value itself
    return expl_powcos*cosv*precal_cutf[0] * precal_cutf[2] * precal_cutf[4];
}

//Old G4
static inline double G4(const double Rij, const double Rik, const double Rjk, const double powtwo, \
          const double* precal, const double* par, double *deriv, const bool powint) {
    // par[0] = R_c
    // par[1] = eta
    // par[2] = zeta
    // par[3] = lambda
    
    // precal[0, 2, 4] = fc of ij ik jk
    // precal[1, 3, 5] = df/dR of ij ik jk
    // precal[6] = Rij + Rik + Rjk
    // precal[7] = cos(theta ijk)
    // precal[8] = dcos(theta ijk)/Rij , [9] : wrt Rik , [10] : wrt Rjk
    
    // common exp factor
    double expl = exp(-par[1]*precal[6]) * powtwo;
    // cos(theta) + 1
    double cosv = 1 + par[3]*precal[7];
    //double powcos = powint ? pow_int2(fabs(cosv), par[2]-1) : pow(fabs(cosv), fabs(par[2]-1));
    // why zeta - 1?? removed fabs(abs version float) cosv is always > 0
    
    // (1+cos(theta)) ^ (zeta-1)  instead of zeta, to get derivative easier
    double powcos = powint ? pow_int2(cosv, par[2]-1) : pow(cosv, par[2]-1);
    //common factor
    double expl_powcos = expl*powcos;
    //zeta * lambda
    double par2_par3 = par[2]*par[3];

    //common factor exp, cos^zeta,f(Rjk), f(Rik)*{ dexp term               + df/dRij term    + dcosv/dRij term              }
    deriv[0] = expl_powcos*precal[2]*precal[4]*((-2*par[1]*Rij*precal[0] + precal[1])*cosv + par2_par3*precal[0]*precal[8]); // wrt Rij

    deriv[1] = expl_powcos*precal[0]*precal[4] * \
               ((-2*par[1]*Rik*precal[2] + precal[3])*cosv + \
               par2_par3*precal[2]*precal[9]); // ik

    deriv[2] = expl_powcos*precal[0]*precal[2] * \
               ((-2*par[1]*Rjk*precal[4] + precal[5])*cosv + \
               par2_par3*precal[4]*precal[10]); // jk

    //G4 value itself
    return expl_powcos*cosv*precal[0] * precal[2] * precal[4];
}

static inline double G5(const double Rij, const double Rik, const double powtwo, \
          const double* precal_cutf, const double* precal_ang, const double* par, double *deriv, const bool powint) {
    // par[0] = R_c
    // par[1] = eta
    // par[2] = zeta
    // par[3] = lambda
    double expl = exp(-par[1]*precal_ang[5]) * powtwo;
    double cosv = 1 + par[3]*precal_ang[1];
    //double powcos = powint ? pow_int2(fabs(cosv), par[2]-1) : pow(fabs(cosv), fabs(par[2]-1));
    double powcos = powint ? pow_int2(cosv, par[2]-1) : pow(cosv, fabs(par[2]-1));

    double expl_powcos = expl * powcos;
    double par2_par3 = par[2]*par[3];

    deriv[0] = expl_powcos*precal_cutf[2] * \
               ((-2*par[1]*Rij*precal_cutf[0] + precal_cutf[1])*cosv + \
               par2_par3*precal_cutf[0]*precal_ang[2]); // ij

    deriv[1] = expl_powcos*precal_cutf[0] * \
               ((-2*par[1]*Rik*precal_cutf[2] + precal_cutf[3])*cosv + \
               par2_par3*precal_cutf[2]*precal_ang[3]); // ik

    deriv[2] = expl*powcos*precal_cutf[0]*precal_cutf[2] * \
               par2_par3*precal_ang[4]; // jk

    return expl_powcos*cosv*precal_cutf[0] * precal_cutf[2];
}

//Old G5
static inline double G5(const double Rij, const double Rik, const double powtwo, \
          const double* precal, const double* par, double *deriv, const bool powint) {
    // par[0] = R_c
    // par[1] = eta
    // par[2] = zeta
    // par[3] = lambda
    double expl = exp(-par[1]*precal[11]) * powtwo;
    double cosv = 1 + par[3]*precal[7];
    //double powcos = powint ? pow_int2(fabs(cosv), par[2]-1) : pow(fabs(cosv), fabs(par[2]-1));
    double powcos = powint ? pow_int2(cosv, par[2]-1) : pow(cosv, fabs(par[2]-1));

    double expl_powcos = expl * powcos;
    double par2_par3 = par[2]*par[3];

    deriv[0] = expl_powcos*precal[2] * \
               ((-2*par[1]*Rij*precal[0] + precal[1])*cosv + \
               par2_par3*precal[0]*precal[8]); // ij
    deriv[1] = expl_powcos*precal[0] * \
               ((-2*par[1]*Rik*precal[2] + precal[3])*cosv + \
               par2_par3*precal[2]*precal[9]); // ik
    deriv[2] = expl*powcos*precal[0]*precal[2] * \
               par2_par3*precal[10]; // jk

    return expl_powcos*cosv*precal[0] * precal[2];
}

static inline double cutf(double frac) {
    // frac = dist / cutoff_dist
    if (frac >= 1.0) {
        return 0;
    } else {
        return 0.5 * (1 + cos(M_PI*frac));
    }
}

static inline double dcutf(double dist, double cutd) {
    if (dist/cutd >= 1.0) {
        return 0;
    } else {
        return -0.5 * M_PI * sin(M_PI*dist/cutd) / cutd;
    }
}
