import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("nomFichier",type=str)
args = parser.parse_args()

chi = lambda f, fr, df, a, phi : a * np.exp(1j * phi) / (fr ** 2 - f * (f - 1j * df))

def chi_fit(x, fr, df, a, phi):
  N = len(x)
  x_reel = x[N//2:]
  x_imag = x[:N//2]
  y_reel = np.real(chi(x_reel,fr,df,a,phi))
  y_imag = np.imag(chi(x_imag,fr,df,a,phi))
  return np.hstack([y_reel,y_imag])

tableauResonance = np.array([]) # (fr df a phi = popt) ×2

df = pd.read_csv(args.nomFichier)
f = df["Freq (Hz)"] * 1e-9  # Fréquences en GHz
masque = (f > .1) * (f < 8.4)   # On enlève les effets sur les bords
yRs = np.array([df["Re(chi1)"] * masque,df["Re(chi2)"] * masque,df["Re(chi3)"] * masque])
yIs = np.array([df["Im(chi1)"] * masque,df["Im(chi2)"] * masque,df["Im(chi3)"] * masque])

result = np.array([])
ax_nb = 0
fig, axs = plt.subplots(ncols=3)

for yR, yI in zip(yRs,yIs):
  champ, _ = os.path.splitext(os.path.split(args.nomFichier)[1])
  champ = int(champ)

  print(f"{yR[200]} devrait être égal à {(df[f'Re(chi{ax_nb+1})'])[200]}")

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

  axs[ax_nb].plot(f, yR, color='k', linestyle='-')
  axs[ax_nb].plot(f, gaussian_filter1d(yI,sigma=10.0), color='k', linestyle = '--')
  axs[ax_nb].scatter(f[pics], yI[pics])
  ax_nb += 1
  
  result = np.array([])
  for pic in pics:
    masque = np.abs(f - f[pic]) < .5
    p0 = (f[pic],1,f[pic] ** 2 * yI[pic],0)
    try:
      popt, pcov = curve_fit(chi_fit,np.hstack([f[masque],f[masque]]),np.hstack([np.nan_to_num(yR[masque]),np.nan_to_num(yI[masque])]),\
                          p0=p0, bounds=([f[pic]-1,-2,-np.inf,-np.pi],[f[pic]+1,2,np.inf,np.pi]))
    except Exception as e:
      print(f"Une erreur s'est produite : {e}")
      popt = p0
      pcov = None
    
    result = np.hstack([result,*popt])
  
  if (len(result) < 8):
    # print(result, len(result))
    result = np.pad(result,(0,8-len(result)))
  if (ax_nb == 1):
    tableauResonance = result
  else:
    tableauResonance = np.vstack([tableauResonance,result])
  print(tableauResonance)

plt.show()

fig, axs = plt.subplots(2,3,sharex=True)

axs[0][0].plot(f,yRs[0])
axs[0][0].plot(f,np.real(chi(f,*tableauResonance[0][4:])))
axs[0][0].plot(f,np.real(chi(f,*tableauResonance[0][:4])))

axs[1][0].plot(f,yIs[0])
axs[1][0].plot(f,np.imag(chi(f,*tableauResonance[0][4:])))
axs[1][0].plot(f,np.imag(chi(f,*tableauResonance[0][:4])))

axs[0][1].plot(f,yRs[1])
axs[0][1].plot(f,np.real(chi(f,*tableauResonance[1][4:])))
axs[0][1].plot(f,np.real(chi(f,*tableauResonance[1][:4])))

axs[1][1].plot(f,yIs[1])
axs[1][1].plot(f,np.imag(chi(f,*tableauResonance[1][4:])))
axs[1][1].plot(f,np.imag(chi(f,*tableauResonance[1][:4])))

axs[0][2].plot(f,yRs[2])
axs[0][2].plot(f,np.real(chi(f,*tableauResonance[2][4:])))
axs[0][2].plot(f,np.real(chi(f,*tableauResonance[2][:4])))

axs[1][2].plot(f,yIs[2])
axs[1][2].plot(f,np.imag(chi(f,*tableauResonance[2][4:])))
axs[1][2].plot(f,np.imag(chi(f,*tableauResonance[2][:4])))

for i in range(3):
  axs[0][i].tick_params(axis='x',direction='in')
  axs[0][i].set_title(f"Chi {i+1}")
  axs[1][i].set_xlabel("Champ Appliqué (mT)")
for i in range(1,3):
  for j in range(2):
    axs[j][i].tick_params(axis='y',direction='in')

axs[0][0].set_ylabel("Re(χ) (u.a.)")
axs[1][0].set_ylabel("Im(χ) (u.a.)")

fig.subplots_adjust(hspace=0.0)

plt.savefig(f"fit-fig/fit-{champ}.png")