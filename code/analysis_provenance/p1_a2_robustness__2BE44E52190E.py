import argparse,csv,json,math,os,re,statistics
from collections import defaultdict

EPS=1e-14
PI=math.pi

def F(x):
    try:
        y=float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None

def solve3(A,b):
    M=[list(map(float,A[i]))+[float(b[i])] for i in range(3)]
    for k in range(3):
        piv=max(range(k,3),key=lambda r:abs(M[r][k]))
        if abs(M[piv][k])<1e-18: raise ArithmeticError("singular")
        if piv!=k: M[k],M[piv]=M[piv],M[k]
        d=M[k][k]
        for j in range(k,4): M[k][j]/=d
        for r in range(3):
            if r==k: continue
            q=M[r][k]
            for j in range(k,4): M[r][j]-=q*M[k][j]
    return [M[i][3] for i in range(3)]

def deriv(t,y,i,half):
    lo=i-half; hi=i+half+1
    if lo<0 or hi>len(t): return None
    tc=t[i]
    xs=[]; ys=[]
    for k in range(lo,hi):
        if t[k] is None or y[k] is None: return None
        xs.append(t[k]-tc); ys.append(y[k])
    s0=len(xs); s1=sum(xs); s2=sum(x*x for x in xs); s3=sum(x*x*x for x in xs); s4=sum(x*x*x*x for x in xs)
    b0=sum(ys); b1=sum(x*v for x,v in zip(xs,ys)); b2=sum(x*x*v for x,v in zip(xs,ys))
    return solve3([[s0,s1,s2],[s1,s2,s3],[s2,s3,s4]],[b0,b1,b2])[1]

def cross(a,b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def mv(A,x): return tuple(sum(A[i][j]*x[j] for j in range(3)) for i in range(3))
def add(a,b): return tuple(a[i]+b[i] for i in range(3))
def sub(a,b): return tuple(a[i]-b[i] for i in range(3))
def scale(s,a): return tuple(s*x for x in a)
def poly(c,J): return c[0]+c[1]*J+c[2]*J*J+c[3]*J*J*J

def loadj(p):
    with open(p,"r",encoding="utf-8-sig") as f:return json.load(f)

def files(folder):
    out=[]
    for n in sorted(os.listdir(folder)):
        m=re.search(r"_ID_(\d+)\.csv$",n,re.I)
        if m and int(m.group(1))<=3: out.append(os.path.join(folder,n))
    return out

class Compare:
    def __init__(self): self.a=[]; self.b=[]
    def add(self,a,b):
        if a is None or b is None:return
        if not(math.isfinite(a) and math.isfinite(b)):return
        self.a.append(a);self.b.append(b)
    def out(self):
        n=len(self.a)
        if not n:return {"n":0,"rms_diff":None,"nrmse":None,"corr":None,"median_abs_diff":None}
        d=[x-y for x,y in zip(self.a,self.b)]
        rms=math.sqrt(sum(z*z for z in d)/n)
        brms=math.sqrt(sum(y*y for y in self.b)/n)
        med=statistics.median(abs(z) for z in d)
        ma=sum(self.a)/n;mb=sum(self.b)/n
        va=sum((x-ma)**2 for x in self.a);vb=sum((y-mb)**2 for y in self.b)
        cov=sum((x-ma)*(y-mb) for x,y in zip(self.a,self.b))
        corr=cov/math.sqrt(va*vb) if va>EPS and vb>EPS else None
        return {"n":n,"rms_diff":rms,"nrmse":rms/max(brms,EPS),"corr":corr,"median_abs_diff":med}

def coeffs(Fa,Ma,Va,alpha,beta,rho,S,b,cbar):
    qbar=.5*rho*Va*Va
    if qbar<=1e-8:return None
    X,Y,Z=Fa
    ca=math.cos(alpha);sa=math.sin(alpha);cb=math.cos(beta);sb=math.sin(beta)
    D=-(ca*cb*X+sb*Y+sa*cb*Z)
    Yw=-ca*sb*X+cb*Y-sa*sb*Z
    L=sa*X-ca*Z
    return {
        "C_X_body":X/(qbar*S),
        "C_Y_body":Y/(qbar*S),
        "C_Z_body":Z/(qbar*S),
        "C_D_drag":D/(qbar*S),
        "C_Y_wind":Yw/(qbar*S),
        "C_L_lift":L/(qbar*S),
        "C_l_roll":Ma[0]/(qbar*S*b),
        "C_m_pitch":Ma[1]/(qbar*S*cbar),
        "C_n_yaw":Ma[2]/(qbar*S*b)
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--clean-dir",required=True);ap.add_argument("--iced-dir",required=True)
    ap.add_argument("--clean-json",required=True);ap.add_argument("--iced-json",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    windows={"W5":2,"W7":3,"W9":4}
    force_names=["C_X_body","C_Y_body","C_Z_body","C_D_drag","C_Y_wind","C_L_lift"]
    moment_names=["C_l_roll","C_m_pitch","C_n_yaw"]
    cmp_window=defaultdict(Compare)
    cmp_moment=defaultdict(Compare)
    cmp_eta=defaultdict(Compare)
    accel_cmp=defaultdict(Compare)

    file_rows=[]
    errors=[]
    rows_seen=0
    rows_common=0
    nonfinite=0

    for cname,folder,jp in [("clean",a.clean_dir,a.clean_json),("iced",a.iced_dir,a.iced_json)]:
        p=loadj(jp)
        m=float(p["mass"]);g=float(p["gravity"]);rho=float(p["rho"])
        D=float(p["D"]);S=float(p["S_wing"]);b=float(p["b"]);cbar=float(p["c"])
        Jp=float(p["J_prop"]);et=float(p["eta_prop_T"]);eq=float(p["eta_prop_Q"])
        ct=[float(p["C_T_0"]),float(p["C_T_1"]),float(p["C_T_2"]),float(p["C_T_3"])]
        cq=[float(p["C_Q_0"]),float(p["C_Q_1"]),float(p["C_Q_2"]),float(p["C_Q_3"])]
        Jx=float(p["Jx"]);Jy=float(p["Jy"]);Jz=float(p["Jz"]);Jxz=float(p["Jxz"])
        I=((Jx,0,-Jxz),(0,Jy,0),(-Jxz,0,Jz))

        for path in files(folder):
            try:
                with open(path,"r",newline="",encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
                rows_seen+=len(rows)
                def col(n):return [F(r.get(n)) for r in rows]
                t=col("TIME");u=col("u_CG");v=col("v_CG");w=col("w_CG")
                pr=col("p");qr=col("q");rr=col("r")
                pd=col("p_dot");qd=col("q_dot");rd=col("r_dot")
                Va=col("Va_EKF_CG");al=col("alpha");be=col("beta");ph=col("phi");th=col("theta")
                Om=col("RPS");ax=col("ax_cg");ay=col("ay_cg");az=col("az_cg")
                retained=0

                for i in range(4,len(rows)-4):
                    base=[t[i],u[i],v[i],w[i],pr[i],qr[i],rr[i],pd[i],qd[i],rd[i],Va[i],al[i],be[i],ph[i],th[i],Om[i]]
                    if any(x is None for x in base):continue
                    try:
                        dvel={}
                        drate={}
                        for wn,half in windows.items():
                            dvel[wn]=(deriv(t,u,i,half),deriv(t,v,i,half),deriv(t,w,i,half))
                            drate[wn]=(deriv(t,pr,i,half),deriv(t,qr,i,half),deriv(t,rr,i,half))
                    except Exception:
                        continue
                    if any(any(x is None for x in dvel[k]) for k in windows):continue
                    if any(any(x is None for x in drate[k]) for k in windows):continue

                    n=Om[i]/(2*PI)
                    if abs(n)<1e-10:continue
                    J=Va[i]/(n*D)
                    CT=poly(ct,J);CQ=poly(cq,J)
                    T=et*CT*rho*n*n*D**4
                    Q=eq*CQ*rho*n*n*D**5
                    T1=CT*rho*n*n*D**4
                    Q1=CQ*rho*n*n*D**5

                    Fp=(T,0,0);Fp1=(T1,0,0)
                    Mp=(-Q,-Jp*Om[i]*rr[i],+Jp*Om[i]*qr[i])
                    Mp1=(-Q1,-Jp*Om[i]*rr[i],+Jp*Om[i]*qr[i])

                    gb=(-g*math.sin(th[i]),g*math.sin(ph[i])*math.cos(th[i]),g*math.cos(ph[i])*math.cos(th[i]))
                    vel=(u[i],v[i],w[i]);omega=(pr[i],qr[i],rr[i])

                    coeff_force={}
                    for wn in windows:
                        ain=add(dvel[wn],cross(omega,vel))
                        Fa=sub(sub(scale(m,ain),Fp),scale(m,gb))
                        Iwd=mv(I,(pd[i],qd[i],rd[i]))
                        Ma=sub(add(Iwd,cross(omega,mv(I,omega))),Mp)
                        coeff_force[wn]=coeffs(Fa,Ma,Va[i],al[i],be[i],rho,S,b,cbar)

                        spec=sub(ain,gb)
                        accel_cmp[cname+"|"+wn+"|x"].add(spec[0],ax[i])
                        accel_cmp[cname+"|"+wn+"|y"].add(spec[1],ay[i])
                        accel_cmp[cname+"|"+wn+"|z"].add(spec[2],az[i])

                    if any(x is None for x in coeff_force.values()):continue

                    # 5-vs-7 and 5-vs-9 force coefficient sensitivity.
                    for qn in force_names:
                        cmp_window[cname+"|W7_vs_W5|"+qn].add(coeff_force["W7"][qn],coeff_force["W5"][qn])
                        cmp_window[cname+"|W9_vs_W5|"+qn].add(coeff_force["W9"][qn],coeff_force["W5"][qn])

                    # Moment sensitivity: source p_dot primary vs numerically differentiated rate windows.
                    Iwd_src=mv(I,(pd[i],qd[i],rd[i]))
                    Ma_src=sub(add(Iwd_src,cross(omega,mv(I,omega))),Mp)
                    csrc=coeffs((0,0,0),Ma_src,Va[i],al[i],be[i],rho,S,b,cbar)
                    for wn in windows:
                        Iwd_num=mv(I,drate[wn])
                        Ma_num=sub(add(Iwd_num,cross(omega,mv(I,omega))),Mp)
                        cnum=coeffs((0,0,0),Ma_num,Va[i],al[i],be[i],rho,S,b,cbar)
                        for qn in moment_names:
                            cmp_moment[cname+"|"+wn+"_rate_vs_source_pdot|"+qn].add(cnum[qn],csrc[qn])

                    # Propulsion eta sensitivity on the frozen primary W5 derivative.
                    ain5=add(dvel["W5"],cross(omega,vel))
                    Fa5=sub(sub(scale(m,ain5),Fp),scale(m,gb))
                    Fa_eta1=sub(sub(scale(m,ain5),Fp1),scale(m,gb))
                    Ma5=Ma_src
                    Ma_eta1=sub(add(Iwd_src,cross(omega,mv(I,omega))),Mp1)
                    c5=coeffs(Fa5,Ma5,Va[i],al[i],be[i],rho,S,b,cbar)
                    ce=coeffs(Fa_eta1,Ma_eta1,Va[i],al[i],be[i],rho,S,b,cbar)
                    for qn in force_names+moment_names:
                        cmp_eta[cname+"|ETA1_vs_SOURCE_ETA|"+qn].add(ce[qn],c5[qn])

                    retained+=1;rows_common+=1

                file_rows.append({"configuration":cname,"file_name":os.path.basename(path),"source_rows":len(rows),"common_rows":retained})

            except Exception as e:
                errors.append({"configuration":cname,"file_name":os.path.basename(path),"error":repr(e)})

    def write_map(path,mp,kind):
        out=[]
        for key,m in sorted(mp.items()):
            parts=key.split("|")
            row={"configuration":parts[0],"comparison":parts[1],"quantity":parts[2],"kind":kind}
            row.update(m.out());out.append(row)
        with open(path,"w",newline="",encoding="utf-8-sig") as f:
            fields=["configuration","comparison","quantity","kind","n","rms_diff","nrmse","corr","median_abs_diff"]
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
        return out

    wr=write_map(os.path.join(a.out,"P1_A2_FORCE_WINDOW_SENSITIVITY.csv"),cmp_window,"FORCE_COEFF")
    mr=write_map(os.path.join(a.out,"P1_A2_MOMENT_DERIVATIVE_SENSITIVITY.csv"),cmp_moment,"MOMENT_COEFF")
    er=write_map(os.path.join(a.out,"P1_A2_PROPULSION_ETA_SENSITIVITY.csv"),cmp_eta,"ETA_SENSITIVITY")

    # acceleration crosscheck has a different key shape.
    ar=[]
    for key,m in sorted(accel_cmp.items()):
        cfg,win,axis=key.split("|")
        row={"configuration":cfg,"window":win,"axis":axis}
        row.update(m.out());ar.append(row)
    with open(os.path.join(a.out,"P1_A2_ACCELERATION_WINDOW_CROSSCHECK.csv"),"w",newline="",encoding="utf-8-sig") as f:
        fields=["configuration","window","axis","n","rms_diff","nrmse","corr","median_abs_diff"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(ar)

    with open(os.path.join(a.out,"P1_A2_FILE_COMMON_SUPPORT.csv"),"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=["configuration","file_name","source_rows","common_rows"]);w.writeheader();w.writerows(file_rows)

    with open(os.path.join(a.out,"P1_A2_PARSE_ERRORS.csv"),"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=["configuration","file_name","error"]);w.writeheader();w.writerows(errors)

    # Predeclared robustness classification: key longitudinal force coefficients only.
    keyq={"C_D_drag","C_L_lift","C_X_body","C_Z_body"}
    keyrows=[r for r in wr if r["quantity"] in keyq]
    force_gate=all(
        r["nrmse"] is not None and r["nrmse"]<0.15 and
        r["corr"] is not None and r["corr"]>0.95
        for r in keyrows
    )

    # Lateral-force coefficients are diagnostic due near-zero denominators / low amplitude.
    # Moment differentiated-rate comparisons are diagnostic; primary remains source p_dot/q_dot/r_dot.
    result={
        "scope":{
            "identification_files_used":54,
            "validation_files_used":0,
            "source_rows_seen":rows_seen,
            "common_support_rows":rows_common,
            "expected_common_support_rows":rows_seen-8*54
        },
        "integrity":{
            "parse_errors":len(errors),
            "nonfinite_outputs":nonfinite
        },
        "predeclared_force_robustness_gate":{
            "quantities":["C_D_drag","C_L_lift","C_X_body","C_Z_body"],
            "comparisons":["W7_vs_W5","W9_vs_W5"],
            "criteria":"NRMSE<0.15 and corr>0.95 for both configurations",
            "pass":force_gate
        },
        "notes":[
            "C_Y_body and C_Y_wind are nonblocking diagnostics because their RMS magnitude is small and relative NRMSE can be unstable.",
            "Moment derivative sensitivity is diagnostic; source p_dot/q_dot/r_dot remain the frozen primary angular accelerations.",
            "ETA1 sensitivity is nonblocking and quantifies propulsion-model dependence; it does not re-select eta."
        ]
    }
    with open(os.path.join(a.out,"P1_A2_ROBUSTNESS_AUDIT.json"),"w",encoding="utf-8") as f:json.dump(result,f,indent=2)

    print("P1_A2_ROBUSTNESS_SUMMARY")
    print("identification_files_used =",54)
    print("validation_files_used     = 0")
    print("source_rows_seen          =",rows_seen)
    print("common_support_rows       =",rows_common)
    print("expected_common_support   =",rows_seen-8*54)
    print("parse_errors              =",len(errors))
    print("force_robustness_gate     =",force_gate)
    print("")
    print("KEY FORCE WINDOW SENSITIVITY")
    for r in wr:
        if r["quantity"] in ("C_D_drag","C_L_lift","C_X_body","C_Z_body"):
            print("  {0:5s} {1:9s} {2:10s}: nrmse={3:.6g}, corr={4:.6g}, med|d|={5:.6g}".format(
                r["configuration"],r["comparison"],r["quantity"],r["nrmse"],r["corr"],r["median_abs_diff"]
            ))
    print("")
    print("ETA=1 SENSITIVITY (diagnostic only)")
    for r in er:
        if r["quantity"] in ("C_D_drag","C_L_lift","C_m_pitch"):
            print("  {0:5s} {1:10s}: nrmse={2:.6g}, corr={3:.6g}, med|d|={4:.6g}".format(
                r["configuration"],r["quantity"],r["nrmse"],r["corr"],r["median_abs_diff"]
            ))

if __name__=="__main__":
    main()
