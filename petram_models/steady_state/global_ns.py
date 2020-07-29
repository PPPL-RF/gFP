import numpy as np
from numpy import sin, cos, sqrt, pi, arctan2
from scipy.special import erf
from petram.helper.variables import variable

def pow(x, c):
    return x**c

CoulLog = 15.
Zb = 1.
Za = 1.
nb = 1.0e20
vb = 8.8e6 
va=vb*3.0
ma = 9.1093837015e-31
mb = 9.1093837015e-31
ee = 1.602176634e-19
em = 2.718281828459045
gamma = 4.0*pi*pow(Za,4.0)*pow(ee,4.0)/pow(ma,2.0)

def v_theta(x, y):
    y = y if y > 0 else 0.0

    theta = arctan2(y, x)
    vel = sqrt(x**2 + y**2)
    vel = vel if vel != 0 else 1e-12

    return x*va, y*va, vel*va, theta

def calc_A(vel):
    A = (((Zb/Za)**2)*CoulLog*ma/mb*nb*
         (-vel/vb*sqrt(2.0/pi)*exp(-pow(vel,2.0)/2.0/pow(vb,2.0))+erf(vel/sqrt(2.0)/vb)))
    return A

def calc_dA(vel):
    dA = pow(Zb/Za,2.0)*CoulLog*ma/mb*nb*pow(vel,2.0)/pow(vb,3.0)*sqrt(2.0/pi)*exp(-pow(vel,2.0)/2.0/pow(vb,2.0))
    return dA

@variable.float()
def dA_term(x, y):
    x, y, vel, theta = v_theta(x, y)
    dA = calc_dA(vel)
    base = dA/vel*sin(theta)
    return gamma*base

@variable.array(complex=False, shape=(2,))
def A_term(x, y):
    x, y, vel, theta = v_theta(x, y)
    A = calc_A(vel)
    x = A*sin(theta)*cos(theta)/vel
    y = A*sin(theta)*sin(theta)/vel
    vec = array([x, y])
    return vec/va*gamma


@variable.array(complex=False, shape=(2,2))
def BF_term(x, y):
    x, y, vel, theta = v_theta(x, y)

    A = calc_A(vel)

    B = (vb**2)*mb/vel/ma*A
    FWoSin = (pow(Zb/Za,2.0)*CoulLog*nb/2.0/vel*(vb/vel*sqrt(2.0/pi)*exp(-pow(vel,2.0)/2.0/pow(vb,2.0))
              +(1.0-pow(vb,2.0)/pow(vel,2.0))*erf(vel/sqrt(2.0)/vb)))
    F = FWoSin*sin(theta);

    xx = B*sin(theta)*cos(theta)*cos(theta)/vel+vel*F*sin(theta)*sin(theta)
    xy = B*sin(theta)*sin(theta)*cos(theta)/vel-vel*F*cos(theta)*sin(theta)
    yx = B*sin(theta)*sin(theta)*cos(theta)/vel-vel*F*cos(theta)*sin(theta)
    yy = B*sin(theta)*sin(theta)*sin(theta)/vel+vel*F*cos(theta)*cos(theta)

    mat = array([[xx, yx], [xy, yy]])
    return -mat/va/va*gamma





