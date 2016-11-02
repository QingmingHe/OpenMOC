#include "SDSolver.h"

SDNuclide::SDNuclide(const char* name) {
  _alpha = 0.0;
  _p0 = 0.0;
  _1_p0 = 0.0;
  _p1 = 0.0;
  _pm = 0.0;
  _max_g = 0;
  _name = NULL;
  setName(name);
  _has_res = false;
  _has_res_fis = false;
  _awr = 0.0;
  _nfg = 0;
  _nbg = 0;
  _n_case = 0;
  _hf_tot = NULL;
  _hf_sca = NULL;
  _hf_fis = NULL;
  _num_dens = NULL;
  _mg_tot = NULL;
  _mg_sca = NULL;
  _mg_fis = NULL;
}


SDNuclide::~SDNuclide() {
  if (_name != NULL)
    delete[] _name;

  if (_hf_tot != NULL)
    delete[] _hf_tot;

  if (_hf_sca != NULL)
    delete[] _hf_sca;

  if (_hf_fis != NULL)
    delete[] _hf_fis;

  if (_num_dens != NULL)
    delete[] _num_dens;

  if (_mg_tot != NULL) {
    for (int i=0; i < _nbg; i++)
      delete[] _mg_tot[i];
    delete[] _mg_tot;
  }

  if (_mg_sca != NULL) {
    for (int i=0; i < _nbg; i++)
      delete[] _mg_sca[i];
    delete[] _mg_sca;
  }

  if (_mg_fis != NULL) {
    for (int i=0; i < _nbg; i++)
      delete[] _mg_fis[i];
    delete[] _mg_fis;
  }
}


void SDNuclide::setName(const char* name) {
  int length = strlen(name);

  if (_name != NULL)
    delete[] _name;

  /* Initialize a character array for the Nuclide's name */
  _name = new char[length+1];

  /* Copy the input character array Nuclide name to the class attribute name */
  for (int i=0; i <= length; i++)
    _name[i] = name[i];
}


void SDNuclide::setHFTotal(double* xs, int nfg) {
  /* Check number of hyper-fine energy group */
  if (_nfg == 0)
    _nfg = nfg;
  else if (_nfg != nfg)
    log_printf(ERROR, "Number of fine group incompatible");

  /* This nuclide is resonant nuclide if has hyper-fine xs */
  _has_res = true;

  /* Copy total cross sections */
  _hf_tot = new double[nfg];
  for (int i=0; i < nfg; i++)
    _hf_tot[i] = xs[i];
}


void SDNuclide::setHFScatter(double* xs, int nfg) {
  /* Check number of hyper-fine energy group */
  if (_nfg == 0)
    _nfg = nfg;
  else if (_nfg != nfg)
    log_printf(ERROR, "Number of fine group incompatible");

  /* This nuclide is resonant nuclide if has hyper-fine xs */
  _has_res = true;

  /* Copy scatter cross sections */
  _hf_sca = new double[nfg];
  for (int i=0; i < nfg; i++)
    _hf_sca[i] = xs[i];
}


void SDNuclide::setHFFissiion(double* xs, int nfg) {
  /* Check number of hyper-fine energy group */
  if (_nfg == 0)
    _nfg = nfg;
  else if (_nfg != nfg)
    log_printf(ERROR, "Number of fine group incompatible");

  /* This nuclide is resonant nuclide if has hyper-fine xs */
  _has_res = true;
  _has_res_fis = true;

  /* Copy fission cross sections */
  _hf_fis = new double[nfg];
  for (int i=0; i < nfg; i++)
    _hf_fis[i] = xs[i];
}


void SDNuclide::setAwr(double awr) {
  _awr = awr;
}


void SDNuclide::setPotential(double potential) {
  _potential = potential;
}


void SDNuclide::setNumDens(double* num_dens, int n_case) {
  /* Copy Number density and case number */
  _n_case = n_case;
  if (_num_dens != NULL)
    delete[] _num_dens;
  _num_dens = new double[n_case];
  for (int i=0; i < n_case; i++)
    _num_dens[i] = num_dens[i];
}


void SDNuclide::setNBG(int nbg) {
  _nbg = nbg;
}


void SDNuclide::computeParameters(double dmu) {
  /* Compute alpha value */
  _alpha = (_awr - 1.0) * (_awr - 1.0) / (_awr + 1.0) / (_awr + 1.0);

  /* Compute max down scatter group */
  _max_g = std::nearbyint(2.0 * std::log((_awr + 1.0) / std::abs(_awr - 1.0))
                          / dmu);

  /** Compute self scattering probability */
  _p0 = (std::exp(dmu) - 1.0 - dmu) / dmu / (1.0 - _alpha);
  _1_p0 = 1.0 - _p0;

  /* Compute scattering probability of current group to next */
  _p1 = std::pow(1.0 - std::exp(-dmu), 2) / dmu / (1.0 - _alpha);

  /** Compute scattering probability of max down scattering group */
  _pm = _p1 * std::exp(dmu * (1.0 - _max_g));
}


double SDNuclide::getMgTotal(int ibg, int icase) {
  return _mg_tot[ibg][icase];
}


double SDNuclide::getMgScatter(int ibg, int icase) {
  return _mg_sca[ibg][icase];
}


double SDNuclide::getMgFission(int ibg, int icase) {
  if (_has_res_fis)
    return _mg_fis[ibg][icase];
  else
    return 0.0;
}


void SDNuclide::setMgTotal(double** xs) {
  if (_mg_tot != NULL) {
    for (int i=0; i < _nbg; i++)
      delete[] _mg_tot[i];
    delete[] _mg_tot;
  }
  _mg_tot = xs;
}


void SDNuclide::setMgScatter(double** xs) {
  if (_mg_sca != NULL) {
    for (int i=0; i < _nbg; i++)
      delete[] _mg_sca[i];
    delete[] _mg_sca;
  }
  _mg_sca = xs;
}


void SDNuclide::setMgFission(double** xs) {
  if (_mg_fis != NULL) {
    for (int i=0; i < _nbg; i++)
      delete[] _mg_fis[i];
    delete[] _mg_fis;
  }
  _mg_fis = xs;
}


SDSolver::SDSolver() {
  _n_nuc = 0;
  _n_case = 0;
  _emax = -1.0;
  _emin = -1.0;
  _nfg = 0;
  _nbg = 0;
  _dmu = 0.0;
  _e_broad = NULL;
  _e_f_b = NULL;
  _flux = NULL;
  _mg_flx = NULL;
  _nuclides = NULL;
}


SDSolver::~SDSolver() {
  if (_e_broad != NULL)
    delete[] _e_broad;

  if (_e_f_b != NULL)
    delete[] _e_f_b;

  if (_flux != NULL) {
    for (int i=0; i < _n_case; i++)
      delete[] _flux[i];
    delete[] _flux;
  }

  if (_mg_flx != NULL) {
    for (int i = 0; i < _nbg; i++)
      delete[] _mg_flx[i];
    delete[] _mg_flx;
  }

  if (_nuclides != NULL)
    delete[] _nuclides;
}


void SDSolver::setErgGrpBnd(double* e_broad, int n_bnd) {
  /* Number of energy group is number of boundary - 1 */
  _nbg = n_bnd - 1;

  /* Copy broad energy group boundaries */
  if (_e_broad != NULL)
    delete[] _e_broad;
  _e_broad = new double[n_bnd];
  for (int i=0; i < n_bnd; i++)
    _e_broad[i] = e_broad[i];
}


void SDSolver::setNumNuclide(int n_nuc) {
  /* Get number of nuclide */
  _n_nuc = n_nuc;

  /* Allocate memory for nuclides */
  _nuclides = new SDNuclide[n_nuc];
}


void SDSolver::setSolErgBnd(double emin, double emax) {
  _emin = emin;
  _emax = emax;
}


void SDSolver::_computeParameters() {
  int inuc;
  int ifg;
  int ibg;
  double e;
  double u;

  /* Check number of cases */
  _n_case = _nuclides[0].getNCase();
  for (inuc=0; inuc < _n_nuc; inuc++)
    if (_n_case != _nuclides[inuc].getNCase())
      log_printf(ERROR, "number of cases incompatible");

  /* Check number of fine groups */
  for (inuc=0; inuc < _n_nuc; inuc++)
    if (_nuclides[inuc].getNFG() != 0) {
      if (_nfg == 0)
        _nfg = _nuclides[inuc].getNFG();
      else if (_nfg != _nuclides[inuc].getNFG())
        log_printf(ERROR, "number of fine group incompatible");
    }

  /* Get energy boundary for calculation */
  if (_emin < 0.0 && _emax < 0.0) {
    _emin = _e_broad[_nbg];
    _emax = _e_broad[0];
  }
  else {
    if (_emin > _e_broad[_nbg])
      log_printf(ERROR, "emin bigger than energy group boundary");
    if (_emax < _e_broad[0])
      log_printf(ERROR, "emax smaller than energy group boundary");
  }

  /* Compute lethargy width of fine group */
  _dmu = std::log(_emax / _emin) / _nfg;

  /* Compute index of broad energy group boundaries in fine group */
  if (_e_f_b != NULL)
    delete[] _e_f_b;
  _e_f_b = new double[_nbg+1];
  _e_f_b[_nbg] = 0;
  u = _dmu;
  e = _emax * std::exp(-u);
  ibg = 0;
  for (ifg = 0; ifg < _nfg && ibg < _nbg+1; ifg++) {
    if (e < _e_broad[ibg]) {
      _e_f_b[ibg] = ifg;
      ibg += 1;
    }
    u += _dmu;
    e = _emax * std::exp(-u);
  }
  if (_e_f_b[_nbg] == 0)
    _e_f_b[_nbg] = _nfg;

  /* Compute parameters for each nuclide */
  for (inuc=0; inuc < _n_nuc; inuc++) {
    _nuclides[inuc].setNBG(_nbg);
    _nuclides[inuc].computeParameters(_dmu);
  }
}


void SDSolver::_computeFluxInternal(int icase) {
  /* Nuclide index */
  int inuc = 0;
  /* Fine group index */
  int ifg = 0;
  /* Max scattering group index */
  int ifgm = 0;
  /* Flux vector for this case */
  double* flux = _flux[icase];
  /* Scattering source */
  double source = 0.0;
  /* Second term scattering source */
  double s2 = 0.0;
  /* Scattering source of last fine group */
  double last_source = 0.0;
  /* Removal cross section */
  double xs_rem = 0.0;
  /* Nuclide pointer */
  SDNuclide* nuc = NULL;
  /* Exponent of dmu */
  double exp_dmu = std::exp(-_dmu);

  /* Calculate the scattering source for the first fine group */
  source = 0.0;
  for (inuc=0; inuc < _n_nuc; inuc++)
    source += _nuclides[inuc].getPotential() *
      _nuclides[inuc].getNumDens(icase);
  last_source = source;

  /** Compute removal cross section */
  xs_rem = 0.0;
  for (inuc=0; inuc < _n_nuc; inuc++) {
    nuc = &_nuclides[inuc];
    if (nuc->hasRes())
      xs_rem += (nuc->getHFTotal(0) - nuc->getHFScatter(0) * nuc->getP0())
        * nuc->getNumDens(icase);
    else
      xs_rem += nuc->get1P0() * nuc->getPotential() * nuc->getNumDens(icase);
  }

  /* Flux for first fine group */
  flux[0] = source / xs_rem;

  /* Flux for other fine groups */
  for (ifg=1; ifg < _nfg; ifg++) {
    s2 = 0.0;
    xs_rem = 0.0;
    /* The third term of scattering source */
    source = last_source * exp_dmu;
    for (inuc=0; inuc < _n_nuc; inuc++) {
      nuc = &_nuclides[inuc];
      ifgm = ifg - nuc->getMaxG() - 1;
      /* The first term of scattering source and removal cross section */
      if (nuc->hasRes()) {
        xs_rem += (nuc->getHFTotal(ifg) - nuc->getHFScatter(ifg) * nuc->getP0())
          * nuc->getNumDens(icase);
        source += nuc->getP1() * nuc->getHFScatter(ifg-1) *
          nuc->getNumDens(icase) * flux[ifg-1];
      }
      else {
        xs_rem += nuc->get1P0() * nuc->getPotential() * nuc->getNumDens(icase);
        source += nuc->getP1() * nuc->getPotential() * nuc->getNumDens(icase)
          * flux[ifg-1];
      }
      /* The second term of scattering source */
      if (ifgm < 0)
        s2 += nuc->getPm() * nuc->getPotential() * nuc->getNumDens(icase) * _dmu;
      else if (nuc->hasRes())
        s2 += nuc->getPm() * nuc->getHFScatter(ifgm) * nuc->getNumDens(icase)
          * flux[ifgm];
      else
        s2 += nuc->getPm() * nuc->getPotential() * nuc->getNumDens(icase)
          * flux[ifgm];
    }
    /* Accumulate the second term of scattering source */
    source -= exp_dmu * s2;
    /* Compute the flux of current fine group */
    flux[ifg] = source / xs_rem;
    /* Bank the scattering source */
    last_source = source;
  }
}


void SDSolver::computeFlux() {
  /* Compute some parameters for slowing down calculation */
  _computeParameters();

  /* Allocate memory for flux */
  _flux = new double*[_n_case];
  for (int i=0; i < _n_case; i++)
    _flux[i] = new double[_nfg];

  /* Compute flux for each case */
  for (int i=0; i < _n_case; i++)
    _computeFluxInternal(i);
}


void SDSolver::computeMgXs() {
  int ibg;
  int ifg;
  int inuc;
  int icase;
  double* flux;
  double** mg_tot;
  double** mg_sca;
  double** mg_fis;
  SDNuclide* nuc;

  /* Allocate memory for multi-group flux */
  _mg_flx = new double*[_nbg];
  for (ibg = 0; ibg < _nbg; ibg++)
    _mg_flx[ibg] = new double[_n_case];

  for (inuc = 0; inuc < _n_nuc; inuc++) {
    nuc = &_nuclides[inuc];

    if (nuc->hasRes()) {
      /* Allocate memory for multi-group cross sections */
      mg_tot = new double*[_nbg];
      for (ibg = 0; ibg < _nbg; ibg++)
        mg_tot[ibg] = new double[_n_case];
      mg_sca = new double*[_nbg];
      for (ibg = 0; ibg < _nbg; ibg++)
        mg_sca[ibg] = new double[_n_case];
      if (nuc->hasResFis()) {
        mg_fis = new double*[_nbg];
        for (ibg = 0; ibg < _nbg; ibg++)
          mg_fis[ibg] = new double[_n_case];
      }

      /* Condense hyper-fine energy groups to broad energy groups */
      for (ibg = 0; ibg < _nbg; ibg++) {
        for (icase = 0; icase < _n_case; icase++) {
          mg_tot[ibg][icase] = 0.0;
          mg_sca[ibg][icase] = 0.0;
          if (nuc->hasResFis())
            mg_fis[ibg][icase] = 0.0;
          flux = _flux[icase];
          _mg_flx[ibg][icase] = 0.0;
          for (ifg = _e_f_b[ibg]; ifg < _e_f_b[ibg+1]; ifg++) {
            _mg_flx[ibg][icase] += flux[ifg];
            mg_tot[ibg][icase] += nuc->getHFTotal(ifg) * flux[ifg];
            mg_sca[ibg][icase] += nuc->getHFScatter(ifg) * flux[ifg];
            if (nuc->hasResFis())
              mg_fis[ibg][icase] += nuc->getHFFission(ifg) * flux[ifg];
          }
          mg_tot[ibg][icase] /= _mg_flx[ibg][icase];
          mg_sca[ibg][icase] /= _mg_flx[ibg][icase];
          if (nuc->hasResFis())
            mg_fis[ibg][icase] /= _mg_flx[ibg][icase];
        }
      }

      /* Set multi-group cross sections */
      nuc->setMgTotal(mg_tot);
      nuc->setMgScatter(mg_sca);
      if (nuc->hasResFis())
        nuc->setMgFission(mg_fis);
    }
  }
}


double SDSolver::getMgFlux(int ibg, int icase) {
  return _mg_flx[ibg][icase];
}


SDNuclide* SDSolver::getNuclide(int id) {
  if (_nuclides == NULL)
    log_printf(ERROR, "call setNumNuclide before calling getNuclide");

  if (id < 0)
    log_printf(ERROR, "id should > 0");

  if (id > _n_nuc - 1)
    log_printf(ERROR, "id exceeds number of nuclide");

  return &_nuclides[id];
}
