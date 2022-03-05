#ifndef MOD_WEIB_SS_H
#define MOD_WEIB_SS_H

#include <RcppArmadillo.h>

struct ModWeibSuffStats {

    double sum_xx;
    double sum_xz;
    double N;

    ModWeibSuffStats() {
        sum_xx = 0.0;
        sum_xz = 0.0;
        N = 0.0;
    }

    void Reset() {
        sum_xx = 0.;
        sum_xz = 0.;
        N = 0.;
    }

    void Increment(double x, double psi, double y) {
        sum_xx += pow(x * psi, 2.0);
        sum_xy += y * x * psi;
        N += 1;
    }

};

#endif