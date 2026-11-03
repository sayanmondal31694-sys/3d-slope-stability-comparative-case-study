import numpy as np
def fs_circles(H, beta_deg, c, phi_deg, gamma, ru, kh, xc, yc, R, ns=40):
    """Vectorised Bishop simplified for arrays of circles. Slope: toe (0,0), crest (L,H), flat both sides. Returns FoS array (inf if invalid)."""
    b=np.radians(beta_deg); t=np.tan(b); L=H/t; tp=np.tan(np.radians(phi_deg))
    xc=np.asarray(xc,float); yc=np.asarray(yc,float); R=np.asarray(R,float)
    cand=[]
    # segment y=0, x<=0
    d=R**2-yc**2; ok=d>0; s=np.sqrt(np.where(ok,d,0))
    for sg in (-1,1):
        x=xc+sg*s; cand.append(np.where(ok&(x<=0),x,np.nan))
    # segment y=H, x>=L  (needs H<yc for lower arc)
    d=R**2-(yc-H)**2; ok=(d>0)&(yc>H); s=np.sqrt(np.where(ok,d,0))
    for sg in (-1,1):
        x=xc+sg*s; cand.append(np.where(ok&(x>=L),x,np.nan))
    # slope line y=t x, 0<=x<=L
    A=1+t**2; B=-2*xc-2*t*yc; C=xc**2+yc**2-R**2
    disc=B**2-4*A*C; ok=disc>0; sq=np.sqrt(np.where(ok,disc,0))
    for sg in (-1,1):
        x=(-B+sg*sq)/(2*A); y=t*x
        cand.append(np.where(ok&(x>=0)&(x<=L)&(y<yc),x,np.nan))
    cand=np.array(cand)
    with np.errstate(invalid='ignore'):
        xa=np.nanmin(cand,axis=0); xb=np.nanmax(cand,axis=0)
    valid=np.isfinite(xa)&np.isfinite(xb)&((xb-xa)>0.1*H)
    xa=np.where(valid,xa,0.0); xb=np.where(valid,xb,1.0)
    f=(np.arange(ns)+0.5)/ns
    xm=xa[:,None]+(xb-xa)[:,None]*f[None,:]; dx=((xb-xa)/ns)[:,None]
    inside=R[:,None]**2-(xm-xc[:,None])**2
    ym=yc[:,None]-np.sqrt(np.maximum(inside,1e-9))
    g=np.where(xm<0,0.0,np.where(xm>L,H,xm*t))
    h=g-ym
    bad=(h<=1e-6).any(axis=1)
    sina=(xm-xc[:,None])/R[:,None]; cosa=np.sqrt(np.maximum(1-sina**2,1e-9))
    W=gamma*h*dx
    Mdr=(W*sina).sum(1)*R + kh*(W*(yc[:,None]-0.5*(g+ym))).sum(1)
    Wsum=W.sum(1); weak=Mdr<0.02*Wsum*R   # reject degenerate near-zero driving-moment circles
    F=np.ones(len(xc))
    for _ in range(50):
        ma=cosa+sina*tp/F[:,None]
        Fn=((c*dx+W*(1-ru)*tp)/ma).sum(1)*R
        with np.errstate(divide='ignore',invalid='ignore'):
            Fnew=np.where(Mdr>0,Fn/Mdr,np.inf)
        if np.nanmax(np.abs(np.where(np.isfinite(Fnew),Fnew-F,0)))<1e-5: F=Fnew;break
        F=np.where(np.isfinite(Fnew),Fnew,1e9)
    F=np.where(valid&~bad&~weak&(Mdr>0)&np.isfinite(F)&(F>0.02),F,np.inf)
    return F
def min_fs(H,beta,c,phi,gamma,ru=0.0,kh=0.0):
    b=np.radians(beta); L=H/np.tan(b)
    xcs=np.linspace(-1.5*H, L+1.0*H, 90); ycs=np.linspace(0.2*H, 3.0*H, 70)
    XC,YC=np.meshgrid(xcs,ycs); XC=XC.ravel(); YC=YC.ravel()
    best=np.inf
    # radii: from circles that pass just below toe to deep ones
    dtoe=np.sqrt(XC**2+YC**2)
    for k in np.linspace(0.6,1.0,1):  # placeholder
        pass
    for mult in np.concatenate([np.linspace(0.55,1.0,12),np.linspace(1.0,1.9,30)]):
        # R = distance to a point at depth (mult-based) below/above toe along vertical: use R = sqrt(XC^2+(YC+dd)^2), dd from -0.5H..H
        pass
    Rs=[]
    for dd in np.linspace(-0.4*H,1.0*H,36):    # circle passes through (0,-dd) : depth below toe (dd>0) or above toe on face (dd<0)
        Rs.append(np.sqrt(XC**2+(YC+dd)**2))
    # also circles through the crest-region points to catch face circles
    for dd in np.linspace(0.0,0.9*H,10):
        Rs.append(np.sqrt((XC-L*0.5)**2+(YC-0.5*H*0+dd*0-0.5*H)**2)*1.0)
    best=np.inf; bp=None
    for R in Rs:
        F=fs_circles(H,beta,c,phi,gamma,ru,kh,XC,YC,R)
        i=np.argmin(F)
        if F[i]<best: best=F[i]; bp=(XC[i],YC[i],R[i])
    # local refinement (Nelder-Mead)
    from scipy.optimize import minimize
    def obj(p):
        v=fs_circles(H,beta,c,phi,gamma,ru,kh,[p[0]],[p[1]],[p[2]])[0]
        return min(v,50.0)
    if bp is not None:
        r=minimize(obj,bp,method='Nelder-Mead',options={'xatol':0.05,'fatol':1e-4,'maxiter':400})
        best=min(best,r.fun)
    return best
if __name__=="__main__":
    import time; t0=time.time()
    print('cohesionless dry phi30 b26.57:',round(min_fs(50,26.57,0,30,20,0),3),'(expect ~1.156)')
    print('ru=0.2 same:',round(min_fs(50,26.57,0,30,20,0.2),3),'(infinite-slope 0.867)')
    print('phi=0 c/gH=0.1 b60 (Taylor Ns=0.191 -> 0.524):',round(min_fs(50,60,100,0,20,0),3))
    print('phi=0 c/gH=0.1 b45 (Taylor Ns=0.170 -> 0.588):',round(min_fs(50,45,100,0,20,0),3))
    print('Bishop-Morgenstern c/gH=.05 phi=30 cotb=2 ru=0 (chart m~1.9):',round(min_fs(50,26.57,50,30,20,0),3))
    print('time',round(time.time()-t0,1))
