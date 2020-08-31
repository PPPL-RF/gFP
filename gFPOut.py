import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#import gFP data from serial and find current efficiency
#8/31/2020


CoulLog = 15.
Zb = 1.
Za = 1.
nb = 1.53e27#1e20
vb = 8.8e6
va=vb*5.0
ma = 9.1093837015e-31
mb = 9.1093837015e-31
pi = 3.1415926535897932
ee = 1.602176634e-19;
gama = 4.0*pi*pow(Za,4.0)*pow(ee,4.0)/pow(ma,2.0)


w1=0.3
w2=0.5
#D=1.5e-3
def D(x):
    fac=5
    return ((np.tanh((x-w1)*2*np.pi*fac)+1)*(np.tanh(-1*(x-w2)*2*np.pi*fac)+1))/3.e-3

x_points=500
y_points=250


#import data from csv and sort by vx then vy
type=[('f',float),('v',float),('theta',float),('vx',float),('vy',float),]
gfp = np.genfromtxt("gFPOut.txt", delimiter=' ', skip_header=1, dtype=float) #f v theta vx vy
gfp=gfp.T

#split gfp into individual arrays for clarity
va=5
vb=1
f=gfp[0]
v=gfp[1]*va
theta=gfp[2]
v_par=np.round(gfp[3]*va,2) #vx
v_perp=np.round(gfp[4]*va,2) #vy
dx=1/x_points*va
dy=1/y_points*va

int_f=np.sum(f*2*np.pi*v_perp*dx*dy)

f=f/int_f
nb=1

J= np.sum(2*np.pi*f*v_perp*v_par*dx*dy)*ee
#print(J)
w_squared=((w1*va)**2+(w2*va)**2)/2/vb/vb
#print(w_squared)

f=f.reshape(x_points,y_points)
D_w=D(v_par/va)
D_w=D_w.reshape(x_points,y_points)
print(D_w.shape)

dfdv=np.gradient(f,axis=0)/dx
dfdv2=np.gradient(D_w*dfdv,axis=0)/dx


v_pars=v_perp.reshape(x_points,y_points)
fig,ax=plt.subplots()
im=ax.imshow(dfdv2, origin='lower')
#plt.show()

#v_perp=v_perp.reshape(200,100)
#v_par=v_par.reshape(200,100)
#v=v.reshape(200,100)



#sec_d_f=np.zeros(v_perp.size)

#for el in range(v_perp.size):
    #val_1=f[el]
    #if (v[el]/va > 0.95) & (v_perp[el] == 0):
       # val_2 = 1
        #val_0 = 1
        
        
    #elif (v[el]/va > 0.95) & (v_perp[el] < 0):
        #val_2 = np.sum(np.where((v_perp==v_perp[el]+0.1*va) & (v_par==v_par[el]),f,0))
        #val_0 = 1
        

    #elif (v[el]/va > 0.95) & (v_perp[el] > 0):
        #val_2 = 1
        #val_0 = np.sum(np.where((v_perp==v_perp[el]-0.1*va) & (v_par==v_par[el]),f,0)) 
        
    #else:
        #val_2 = np.sum(np.where((v_perp==v_perp[el]+0.1*va) & (v_par==v_par[el]),f,0))
        #val_0 = np.sum(np.where((v_perp==v_perp[el]-0.1*va) & (v_par==v_par[el]),f,0))
        

    #sec_d_f[el]=(val_2+val_0-2*val_1)/dx**2

P=0
dfdv2=dfdv2.flatten()
f=f.flatten()



for el in range(v_perp.size):
    #if (v_par[el]/va > w1) & (v_par[el]/va < w2):
    P+=mb*v[el]**2/2*dfdv2[el]*2*np.pi*v_perp[el]*dx*dy
eps=8.854e-12
 
jNorm=ee*nb*vb
pNormIn=nb*CoulLog*ee**4/(mb*mb*vb**3)/eps**2
pNorm=pNormIn*nb*mb*vb**2

#pNorm=1

JNew=J/jNorm
PNew=P/pNorm
#print(int_f)
print("w1 and w2",w1,w2)
print("Jnew=",JNew)
#print("PNew=",PNew)
#print("PNorm=",pNorm)
#print("JNorm=",jNorm)
#print(J)
#print(P)
#print(J/P)
#print("w_squared",w_squared)
#print(J/P/w_squared/vb)
print("Normalized J/P=",JNew/PNew/1e9)
print("f_max=",np.max(f))


