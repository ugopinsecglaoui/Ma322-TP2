# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 07:51:03 2021

@author: Ugo PINSEC GLAOUI
@class : 3PSC2
"""
import numpy as np
from math import  *
import matplotlib.pyplot as plt
from scipy.integrate import odeint

"""
_______________________________________________________________
TP2 : Résolution numérique des équations différentielles 
_______________________________________________________________

"""

"4.1 Resolution de l'équation linéarisée : "


g = 9.81
L = 1

w0 = np.sqrt(g/L)
print ("Pulsation w0 : ",w0)
f = w0/(2*pi)
print ("Fréquence f : ",f)
T = 1/f
print ("Période T : ",T)
h = 0.04
print("Le pas : ",h)

print("-----------------------------------------------")

A = np.pi/12
phi = 2*np.pi

#-----------------------------------------------
t = np.linspace(0,4,101)

def teta (t) :
    teta = A*np.cos(np.sqrt(g/L)*t+phi)
    return (teta)

def tetap (t) :
    tetap = -A*w0*np.sin(np.sqrt(g/L)*t+phi)
    return (tetap)
#-----------------------------------------------

plt.plot(t,teta(t),label="Teta (t)",color='r')
plt.title("Représentation de Teta(t)")
plt.legend()
plt.show()


"4.2 Equation exacte : "

#---------------------------------------------
## 4.2.2 : Résolution numérique de l'équation exacte, par la méthode d'Euler
#---------------------------------------------

#___ 1 ___ :
#-----------------------------------------------
#Définir
Y0 = np.array([np.pi/12,0])
 
Y = np.zeros((len(t),2))
Y[0,:] = Y0
for i in range(1,len(t)):
    Y[i,0]=teta(t[i])  
    Y[i,1]=(tetap(t[i]))
#-----------------------------------------------    
def pendule (Y,t) :
    Yp =np.array([Y[1],-(w0**2)*Y[0]])
    return (Yp)

#-----------------------------------------------
#___ 2 ___ :

def Euler(f,Y0,h,N):
    Ye = np.zeros((N,2))
    Ye[0,:]=Y0
    for k in range(N-1):
        Ye[k+1,:]=Ye[k,:] + h*f(Ye[k,:],0)
    return (Ye)


Ye = Euler(pendule,Y0,h,len(t))



#---------------------------------------------
#___ 3 ___ :
""" 
Cette méthode donne des valeurs de theta qui divergent en l'infini comme nous 
le montre la figure. En effet, nous remarquons un déphasage progressif du mouvement calculé par la 
méthode d'euler par rapport au mouvement initial.
De plus, son amplitude augmente progressivement.
Ces résultats diffèrent du signal initiale à cause d'une erreur causée par la 
méthode d'Euler explicite.
En effet, cette méthode facile à implémenter donne très souvent des résultats 
qui divergent du signal initial étudié.

"""

#---------------------------------------------
## 4.2.3 : Méthode de Runge Kutta d'ordre 2 & 4
#---------------------------------------------

def RungeKutta2(f,Y0,h,N):
    Yrk = np.zeros((N,2))
    Yrk[0,:]=Y0
    k1=np.zeros((N,2))
    k2=np.zeros((N,2))

    for i in range(N-1):
        k1[i,:] = f(Yrk[i,:],0)
        k2[i,:] = f(Yrk[i,:]+(h*k1[i,:])/2,h/2)
        Yrk[i+1,:]=Yrk[i,:] + (h/6)*(k1[i,:]+k2[i,:])
    print(Yrk[0,:])

    return (Yrk)


def RungeKutta4(f,Y0,h,N):
    Yrk = np.zeros((N,2))
    Yrk[0,:]=Y0
    k1=np.zeros((N,2))
    k2=np.zeros((N,2))
    k3=np.zeros((N,2))
    k4=np.zeros((N,2))
    for i in range(N-1):
        k1[i,:] = f(Yrk[i,:],0)
        k2[i,:] = f(Yrk[i,:]+(h*k1[i,:])/2,h/2)
        k3[i,:] = f(Yrk[i,:]+(h*k2[i,:])/2,h/2)
        k4[i,:] = f(Yrk[i,:]+(h*k3[i,:]),h)
        Yrk[i+1,:]=Yrk[i,:] + (h/6)*(k1[i,:]+k2[i,:]+k3[i,:]+k4[i,:])
    print(Yrk[0,:])

    return (Yrk)

Yrk2 = RungeKutta2(pendule,Y0,h,len(t))
Yrk4 = RungeKutta4(pendule,Y0,h,len(t))

plt.plot(t,teta(t),label="Teta (t)")
plt.plot(t,Yrk2[:,0],label="RungeKutta ordre 2(t)")
plt.plot(t,Yrk4[:,0],label="RungeKutta ordre 4 (t)")
plt.legend()
plt.show()


#---------------------------------------------
## 4.2.4 : Résolution numérique de l'équation exacte, avec solveur odeint
#---------------------------------------------


Yode = odeint(pendule,Y0,t)
plt.plot(t,teta(t),label="Teta (t)")
plt.plot(t,Ye[:,0],label="Euler (t)")
plt.plot(t,Yrk4[:,0],label="RungeKutta ordre 4(t)")
plt.plot(t,Yode[:,0],label="Yode (t)")

plt.legend()
plt.show()

# ------------------ Portrait de phase ------------------
plt.plot(Ye[:,0],Ye[:,1], label="Portrait de phase Euler") 
plt.legend()
plt.show()



