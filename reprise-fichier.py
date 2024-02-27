import os
import pandas as pd
import numpy as np
import argparse

ltot = 20e-3 # m
c = 299792458 # m/s

parser = argparse.ArgumentParser(description="Reprend les fichiers de sortie du programme BB-FMR de Serge Vincent\
                                 et en fait un fichier .csv un peu plus lisible.",prog="RepriseBBFMR",usage="python3 reprise.py -l longueur_echantillon -c fichier_cpw -r fichier_reference dossier",\
                                epilog="Fin de l'aide.")

parser.add_argument('-r', '--champReference', type=str)
parser.add_argument('-l', '--ls', type=float)
parser.add_argument('-c', '--cpw', type=str)
parser.add_argument('dossier')
args = parser.parse_args()

df_cpw = pd.read_csv(args.cpw, header=None, delimiter='\t')
s12c = df_cpw[3] + 1j * df_cpw[7]
gammao = - (np.log(np.abs(s12c)) + 1j * np.unwrap(np.angle(s12c))) / ltot

df_champReference = pd.read_csv(args.champReference, header=None, delimiter='\t')
freq = df_champReference[1]
s11r = df_champReference[2] + 1j * df_champReference[6]
s12r = df_champReference[3] + 1j * df_champReference[7]
s21r = df_champReference[4] + 1j * df_champReference[8]
s22r = df_champReference[5] + 1j * df_champReference[9]

s1122 = np.sqrt( np.abs(s11r) * np.abs(s22r) ) * np.exp(1j * np.unwrap( (np.angle(s11r) + np.angle(s22r))) / 2)
s12b = s12r * np.exp(gammao * (ltot-args.ls))
s1122b = s1122 * np.exp(gammao * (ltot-args.ls))

k = (1 + s1122b * s1122b - s12b * s12b) / (2 * s1122b)
k2m1 = k * k - 1
Gamma_p = k + np.sqrt(np.abs(k2m1)) * np.exp(1j * np.unwrap(np.angle(k2m1)) / 2)
Gamma_m = k - np.sqrt(np.abs(k2m1)) * np.exp(1j * np.unwrap(np.angle(k2m1)) / 2)
if (np.max(np.abs(Gamma_p))) < 1 :
  Gamma = Gamma_p
else :
  Gamma = Gamma_m
p = (s1122b + s12b + Gamma) / (1 - (s1122b + s12b)*Gamma)

gammaR = - np.log(np.abs(p)) / args.ls
gammaI = - np.unwrap(np.angle(p)) / args.ls

gamma = gammaR + 1j * gammaI
gammaFs = (1j * c) / (2 * np.pi * freq)

mu_1_r = (gamma * (1 + Gamma)) / (gammaFs * (1 - Gamma))
mu_2_r = (gamma / gammaFs) * (gamma / gammaFs)
mu_3_r = ((1 + Gamma) / (1 - Gamma)) * ((1 + Gamma) / (1 - Gamma))

for nomChemin, nomDossier, nomFichiers in os.walk(args.dossier):
  
  for nomFichier in nomFichiers:

    df = pd.read_csv(os.path.join(nomChemin,nomFichier), header=None, delimiter='\t')
    champ = int(np.mean(df[0].values))

    freq = df[1].values
    s11h = df[2] + 1j * df[6]
    s12h = df[3] + 1j * df[7]
    s21h = df[4] + 1j * df[8]
    s22h = df[5] + 1j * df[9]

    u = 1j * (np.log(np.abs(s12h)) + 1j * np.unwrap(np.angle(s12h)) - np.log(np.abs(s12r)) - 1j * np.unwrap(np.angle(s12h))) / (np.log(np.abs(s12r)) + 1j * np.unwrap(np.angle(s12r)))

    s1122 = np.sqrt( np.abs(s11h) * np.abs(s22h) ) * np.exp(1j * np.unwrap( (np.angle(s11h) + np.angle(s22h))) / 2)
    s12b = s12r * np.exp(gammao * (ltot-args.ls))
    s1122b = s1122 * np.exp(gammao * (ltot-args.ls))

    k = (1 + s1122b * s1122b - s12b * s12b) / (2 * s1122b)
    k2m1 = k * k - 1
    Gamma_p = k + np.sqrt(np.abs(k2m1)) * np.exp(1j * np.unwrap(np.angle(k2m1)) / 2)
    Gamma_m = k - np.sqrt(np.abs(k2m1)) * np.exp(1j * np.unwrap(np.angle(k2m1)) / 2)
    if (np.max(np.abs(Gamma_p))) < 1 :
      Gamma = Gamma_p
    else :
      Gamma = Gamma_m
    p = (s1122b + s12b + Gamma) / (1 - (s1122b + s12b)*Gamma)

    gammaR = - np.log(np.abs(p)) / args.ls
    gammaI = - np.unwrap(np.angle(p)) / args.ls

    gamma = gammaR + 1j * gammaI
    gammaFs = (1j * c) / (2 * np.pi * freq)

    chi_1 = (gamma * (1 + Gamma)) / (gammaFs * (1 - Gamma)) - mu_1_r
    chi_2 = (gamma / gammaFs) * (gamma / gammaFs) - mu_2_r
    chi_3 = ((1 + Gamma) / (1 - Gamma)) * ((1 + Gamma) / (1 - Gamma)) - mu_3_r

    df_repris = pd.DataFrame({
      "Freq (Hz)" : freq * 1e3,
      "Re(S12)" : np.real(s12h),
      "Im(S12)" : np.imag(s12h),
      "Re(S12/ref)" : np.real(s12h/s12r),
      "Im(S12/ref)" : np.imag(s12h/s12r),
      "Re(U)" : np.real(u),
      "Im(U)" : np.imag(u),
      "Re(chi1)" : np.real(chi_1),
      "Im(chi1)" : np.imag(chi_1),
      "Re(chi2)" : np.real(chi_2),
      "Im(chi2)" : np.imag(chi_2),
      "Re(chi3)" : np.real(chi_3),
      "Im(chi3)" : np.imag(chi_3)
    })

    doss, fich = os.path.split(nomFichier)
    fi, ch = os.path.splitext(fich)

    df_repris.to_csv("fichier-repris/"+str(champ)+".csv",index=None)
