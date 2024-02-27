import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from natsort import natsorted

chi = lambda f, fr, df, a, phi : a * np.exp(1j * phi) / (fr ** 2 - f * (f - 1j * df))

def chi_fit(x, fr, df, a, phi):
  N = len(x)
  x_reel = x[N//2:]
  x_imag = x[:N//2]
  y_reel = np.real(chi(x_reel,fr,df,a,phi))
  y_imag = np.imag(chi(x_imag,fr,df,a,phi))
  return np.hstack([y_reel,y_imag])

tableauResonance = np.zeros(7) # Freq . FMR . dFMR . A . FMR2 . dFMR2 . A2

for nomChemin, nomDossiers, nomFichiers in os.walk("fichier-repris"):
  for nomFichier in natsorted(nomFichiers):
    if (nomFichier[0] == '.'):
      continue
    champ, _ = os.path.splitext(nomFichier)
    champ = int(champ)

    df = pd.read_csv(os.path.join(nomChemin,nomFichier))
    f = df["Freq (Hz)"] * 1e-9  # Fréquences en GHz
    masque = (f > .1) * (f < 8.4)   # On enlève les effets sur les bords
    yR = df["Re(chi2)"] * masque
    yI = df["Im(chi2)"] * masque

    try:
      pics, proprietes = find_peaks(gaussian_filter1d(-yI,sigma=.1), distance=10, prominence=1e-5)
      indicesTries = np.argsort(proprietes["prominences"])
      pics = pics[indicesTries]
      pics = pics[-2:]
      pics = np.sort(pics)
    except Exception as e:
      print(f"Une erreur s'est produite : {e}")
      pics = np.argmin(yI)
      proprietes = None
    if (len(pics) == 0):
      pics = np.array([np.argmin(yI)])
      proprietes = None
    
    result = np.array([champ])
    for pic in pics:
      masque = np.abs(f - f[pic]) < 1
      p0 = (f[pic],1,f[pic] ** 2 * yI[pic],0)
      try:
        popt, pcov = curve_fit(chi_fit,np.hstack([f[masque],f[masque]]),np.hstack([np.nan_to_num(yR[masque]),np.nan_to_num(yI[masque])]),\
                            p0=p0, bounds=([f[pic]-1,-4,-np.inf,-np.pi],[f[pic]+1,4,np.inf,np.pi]), maxfev=1000)
      except Exception as e:
        print(f"Une erreur s'est produite : {e}")
        popt = p0
        pcov = None
      
      result = np.hstack([result,popt[0],popt[1],popt[2]])
    
    if (len(result) < 7):
      # print(result, len(result))
      result = np.pad(result,(0,7-len(result)))
    tableauResonance = np.vstack([tableauResonance,result])

print(tableauResonance)

df_resonance = pd.DataFrame({
  "Champ (T)" : tableauResonance[:,0] * 1e-4,
  "FMR1 (GHz)" : tableauResonance[:,1],
  "dF1 (GHz)" : tableauResonance[:,2],
  "Intensite1" : tableauResonance[:,3],
  "FMR2 (GHz)" : tableauResonance[:,4],
  "dF2 (GHz)" : tableauResonance[:,5],
  "Intensite2" : tableauResonance[:,6]
})

df_resonance.to_csv("fit-fig/resultat.csv",index=None)