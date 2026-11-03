import sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
from slope_check import min_fs
from concurrent.futures import ProcessPoolExecutor
cases = {
 'Switzerland':(140,38,25.5,0.20,0.0),
 'Zambia':(35,27,23.0,0.35,0.0),
 'India':(25,24,21.5,0.40,0.0),
 'Kazakhstan':(25,34,24.5,0.10,0.0),
 'Mongolia static':(180,42,26.5,0.0,0.0),
 'Mongolia kh=0.12':(180,42,26.5,0.0,0.12),
 'Zambia drained Ru=0.10':(35,27,23.0,0.10,0.0),
 'India drained Ru=0.10':(25,24,21.5,0.10,0.0),
}
def maxang(a):
    name,H,target=a
    c,phi,g,ru,kh=cases[name]
    lo,hi=8.0,80.0
    if min_fs(H,lo,c,phi,g,ru,kh)<target: return (name,H,target,'<8')
    if min_fs(H,hi,c,phi,g,ru,kh)>=target: return (name,H,target,'>80')
    for _ in range(7):
        mid=(lo+hi)/2
        if min_fs(H,mid,c,phi,g,ru,kh)>=target: lo=mid
        else: hi=mid
    return (name,H,target,round(lo,1))
jobs=[(n,H,1.3) for n in cases for H in (15,150)]
jobs+=[('Mongolia kh=0.12',15,1.1),('Mongolia kh=0.12',150,1.1)]
if __name__=="__main__":
    with ProcessPoolExecutor() as ex:
        for r in ex.map(maxang,jobs): print(r,flush=True)
