#ifndef SDSOLVER_H
#define SDSOLVER_H

#ifdef __cplusplus
#ifdef SWIG
#include "Python.h"
#endif
#include "log.h"
#include <string>
#include <cmath>
#endif

class SDNuclide {

private:
  /** Standard alpha value */
  double _alpha;

  /** Self scattering probability */
  double _p0;

  /** 1 - Self scattering probability */
  double _1_p0;

  /** Scattering probability of current group to next */
  double _p1;

  /** Scattering probability of max down scattering group */
  double _pm;

  /** Max down scattering group */
  int _max_g;

  /** Name of the nuclide */
  char* _name;

  /** Whether has resonance */
  bool _has_res;

  /** Whether has resonance fission */
  bool _has_res_fis;

  /** Atomic weight ratio */
  double _awr;

  /** Number of hyper-fine energy group */
  int _nfg;

  /** Number of broad energy group */
  int _nbg;

  /** Number of cases */
  int _n_case;

  /** Hyper-fine energy group total cross sections */
  double* _hf_tot;

  /** Hyper-fine energy group scatter cross sections */
  double* _hf_sca;

  /** Hyper-fine energy group fission cross sections */
  double* _hf_fis;

  /** Number densities */
  double* _num_dens;

  /** Potential cross sections of broad energy groups */
  double _potential;

  /** Multi-group total cross sections */
  double** _mg_tot;

  /** Multi-group scatter cross sections */
  double** _mg_sca;

  /** Multi-group fission cross sections */
  double** _mg_fis;

public:
  SDNuclide(const char* name="");
  virtual ~SDNuclide();

  void setName(const char* name);
  void setHFTotal(double* xs, int nfg);
  void setHFScatter(double* xs, int nfg);
  void setHFFissiion(double* xs, int nfg);
  void setAwr(double awr);
  void setPotential(double potential);
  void setNumDens(double* num_dens, int n_case);
  void setNBG(int nbg);
  void computeParameters(double dmu);
  double getMgTotal(int ibg, int icase);
  double getMgScatter(int ibg, int icase);
  double getMgFission(int ibg, int icase);
  void setMgTotal(double** xs);
  void setMgScatter(double** xs);
  void setMgFission(double** xs);

  int getNCase() const;
  int getNFG() const;
  double getPotential() const;
  double getNumDens(int icase) const;
  bool hasRes() const;
  bool hasResFis() const;
  double getHFTotal(int ifg) const;
  double getHFScatter(int ifg) const;
  double getHFFission(int ifg) const;
  double getP0() const;
  double get1P0() const;
  double getP1() const;
  double getPm() const;
  int getMaxG() const;
};


class SDSolver {
private:
  /** Number of nuclides */
  int _n_nuc;

  /** Number of cases */
  int _n_case;

  /** Max energy */
  double _emax;

  /** Min energy */
  double _emin;

  /** Number of hyper-fine energy group */
  int _nfg;

  /** Number of broad energy group */
  int _nbg;

  /** Lethargy width of hyper-fine energy group */
  double _dmu;

  /** Broad energy group boundaries */
  double* _e_broad;

  /** Fine group number in broad group */
  double* _e_f_b;

  /** Hyper-fine flux */
  double** _flux;

  /** Nuclides */
  SDNuclide* _nuclides;

  void _computeParameters();
  void _computeFluxInternal(int icase);

public:
  SDSolver();
  virtual ~SDSolver();

  void setErgGrpBnd(double* e_broad, int n_bnd);
  void setNumNuclide(int n_nuc);
  void setEnergyBnd(double emin, double emax);
  void computeFlux();
  void computeMgXs();
  SDNuclide* getNuclide(int id);
};


inline int SDNuclide::getNCase() const {
  return _n_case;
}


inline int SDNuclide::getNFG() const {
  return _nfg;
}


inline double SDNuclide::getPotential() const {
  return _potential;
}


inline double SDNuclide::getNumDens(int icase) const {
  return _num_dens[icase];
}


inline bool SDNuclide::hasRes() const {
  return _has_res;
}


inline bool SDNuclide::hasResFis() const {
  return _has_res_fis;
}


inline double SDNuclide::getHFTotal(int ifg) const {
  return _hf_tot[ifg];
}


inline double SDNuclide::getHFScatter(int ifg) const {
  return _hf_sca[ifg];
}


inline double SDNuclide::getHFFission(int ifg) const {
  return _hf_fis[ifg];
}


inline double SDNuclide::getP0() const {
  return _p0;
}


inline double SDNuclide::get1P0() const {
  return _1_p0;
}


inline double SDNuclide::getP1() const {
  return _p1;
}


inline double SDNuclide::getPm() const {
  return _pm;
}


inline int SDNuclide::getMaxG() const {
  return _max_g;
}

#endif
