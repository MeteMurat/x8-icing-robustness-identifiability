import argparse, csv, json, math, os, re, statistics
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
        piv=max(range(k,3), key=lambda r: abs(M[r][k]))
        if abs(M[piv][k]) < 1e-18:
            raise ArithmeticError("singular 3x3 normal matrix")
        if piv!=k:
            M[k],M[piv]=M[piv],M[k]
        d=M[k][k]
        for j in range(k,4):
            M[k][j]/=d
        for r in range(3):
            if r==k: continue
            q=M[r][k]
            for j in range(k,4):
                M[r][j]-=q*M[k][j]
    return [M[i][3] for i in range(3)]

def local_quad_derivative(t,y,i,half=2):
    lo=i-half; hi=i+half+1
    if lo<0 or hi>len(t): return None
    tc=t[i]
    ts=[]; ys=[]
    for k in range(lo,hi):
        if t[k] is None or y[k] is None: return None
        ts.append(t[k]-tc); ys.append(y[k])
    s0=len(ts)
    s1=sum(ts)
    s2=sum(x*x for x in ts)
    s3=sum(x*x*x for x in ts)
    s4=sum(x*x*x*x for x in ts)
    b0=sum(ys)
    b1=sum(x*v for x,v in zip(ts,ys))
    b2=sum(x*x*v for x,v in zip(ts,ys))
    a=solve3([[s0,s1,s2],[s1,s2,s3],[s2,s3,s4]],[b0,b1,b2])
    return a[1]

def cross(a,b):
    return (
        a[1]*b[2]-a[2]*b[1],
        a[2]*b[0]-a[0]*b[2],
        a[0]*b[1]-a[1]*b[0],
    )

def matvec(A,x):
    return tuple(sum(A[i][j]*x[j] for j in range(3)) for i in range(3))

def add(a,b): return tuple(a[i]+b[i] for i in range(3))
def sub(a,b): return tuple(a[i]-b[i] for i in range(3))
def scale(s,a): return tuple(s*x for x in a)

def poly(c,J):
    return c[0]+c[1]*J+c[2]*J*J+c[3]*J*J*J

def load_params(path):
    with open(path,"r",encoding="utf-8-sig") as f:
        return json.load(f)

def id_files(folder):
    out=[]
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(".csv"): continue
        m=re.search(r"_ID_(\d+)\.csv$",name,re.I)
        if m and int(m.group(1))>=4:
            out.append(os.path.join(folder,name))
    return out

def qtile(vals,p):
    xs=sorted(x for x in vals if x is not None and math.isfinite(x))
    if not xs: return None
    if len(xs)==1: return xs[0]
    pos=(len(xs)-1)*p
    lo=int(math.floor(pos)); hi=int(math.ceil(pos))
    if lo==hi: return xs[lo]
    w=pos-lo
    return xs[lo]*(1-w)+xs[hi]*w

class PairMetric:
    def __init__(self):
        self.p=[]; self.t=[]
    def add(self,p,t):
        if p is None or t is None: return
        if not (math.isfinite(p) and math.isfinite(t)): return
        self.p.append(p); self.t.append(t)
    def out(self):
        n=len(self.p)
        if not n: return {"n":0,"rmse":None,"bias":None,"nrmse":None,"corr":None}
        err=[a-b for a,b in zip(self.p,self.t)]
        rmse=math.sqrt(sum(e*e for e in err)/n)
        bias=sum(err)/n
        trms=math.sqrt(sum(x*x for x in self.t)/n)
        nrmse=rmse/max(trms,EPS)
        mp=sum(self.p)/n; mt=sum(self.t)/n
        vp=sum((x-mp)**2 for x in self.p); vt=sum((x-mt)**2 for x in self.t)
        cov=sum((x-mp)*(y-mt) for x,y in zip(self.p,self.t))
        corr=cov/math.sqrt(vp*vt) if vp>EPS and vt>EPS else None
        return {"n":n,"rmse":rmse,"bias":bias,"nrmse":nrmse,"corr":corr}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--clean-dir",required=True)
    ap.add_argument("--iced-dir",required=True)
    ap.add_argument("--clean-json",required=True)
    ap.add_argument("--iced-json",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    configs=[
        ("clean",a.clean_dir,load_params(a.clean_json)),
        ("iced",a.iced_dir,load_params(a.iced_json))
    ]

    sample_fields=[
        "configuration","file_name","sample_index","TIME",
        "Va","alpha","beta","phi","theta","psi","p","q","r",
        "du_dt","dv_dt","dw_dt","p_dot","q_dot","r_dot",
        "Omega_prop_rad_s","n_prop_rev_s","J_advance","C_T","C_Q","T_prop_N","Q_drag_Nm",
        "Fprop_X","Fprop_Y","Fprop_Z","Mprop_L","Mprop_M","Mprop_N",
        "g_body_X","g_body_Y","g_body_Z",
        "a_inertial_body_X","a_inertial_body_Y","a_inertial_body_Z",
        "Faero_X","Faero_Y","Faero_Z",
        "Maero_L","Maero_M","Maero_N",
        "qbar",
        "C_X_body","C_Y_body","C_Z_body","C_D_drag","C_Y_wind","C_L_lift",
        "C_l_roll","C_m_pitch","C_n_yaw",
        "src_CX","src_CY","src_CZ","src_CD","src_CS","src_CL","src_Cl","src_Cm","src_Cn",
        "src_prop_thrust_bodyX","src_prop_torque_drag","src_prop_torque_bodyX","src_prop_torque_bodyY","src_prop_torque_bodyZ",
        "ax_cg","ay_cg","az_cg",
        "p_dot_from_rate","q_dot_from_rate","r_dot_from_rate"
    ]

    file_rows=[]
    coeff_by_cfg=defaultdict(lambda: defaultdict(list))
    ref_metrics=defaultdict(PairMetric)
    prop_metrics=defaultdict(PairMetric)
    accel_metrics=defaultdict(PairMetric)
    deriv_metrics=defaultdict(PairMetric)
    parse_errors=[]
    total_rows_seen=0
    total_retained=0
    nonfinite_outputs=0
    qbar_invalid=0
    validation_used=0

    sample_path=os.path.join(a.out,"P1_A1_RECONSTRUCTED_IDENTIFICATION_SAMPLES.csv")
    sf=open(sample_path,"w",newline="",encoding="utf-8-sig")
    sw=csv.DictWriter(sf,fieldnames=sample_fields)
    sw.writeheader()

    try:
        for cname,folder,pms in configs:
            m=float(pms["mass"]); g=float(pms["gravity"]); rho=float(pms["rho"])
            D=float(pms["D"]); S=float(pms["S_wing"]); b=float(pms["b"]); cbar=float(pms["c"])
            Jp=float(pms["J_prop"])
            et=float(pms["eta_prop_T"]); eq=float(pms["eta_prop_Q"])
            ct=[float(pms["C_T_0"]),float(pms["C_T_1"]),float(pms["C_T_2"]),float(pms["C_T_3"])]
            cq=[float(pms["C_Q_0"]),float(pms["C_Q_1"]),float(pms["C_Q_2"]),float(pms["C_Q_3"])]
            Jx=float(pms["Jx"]); Jy=float(pms["Jy"]); Jz=float(pms["Jz"]); Jxz=float(pms["Jxz"])
            I=((Jx,0.0,-Jxz),(0.0,Jy,0.0),(-Jxz,0.0,Jz))

            for path in id_files(folder):
                try:
                    with open(path,"r",newline="",encoding="utf-8-sig") as f:
                        rows=list(csv.DictReader(f))
                    total_rows_seen += len(rows)

                    def col(name):
                        return [F(r.get(name)) for r in rows]

                    t=col("TIME")
                    u=col("u_CG"); v=col("v_CG"); w=col("w_CG")
                    pr=col("p"); qr=col("q"); rr=col("r")
                    pd=col("p_dot"); qd=col("q_dot"); rd=col("r_dot")
                    Va=col("Va_EKF_CG"); alpha=col("alpha"); beta=col("beta")
                    phi=col("phi"); theta=col("theta"); psi=col("psi")
                    Om=col("RPS")
                    axcg=col("ax_cg"); aycg=col("ay_cg"); azcg=col("az_cg")

                    retained=0
                    file_coeff=defaultdict(list)

                    for i in range(2,len(rows)-2):
                        needed=[t[i],u[i],v[i],w[i],pr[i],qr[i],rr[i],pd[i],qd[i],rd[i],
                                Va[i],alpha[i],beta[i],phi[i],theta[i],psi[i],Om[i]]
                        if any(x is None for x in needed):
                            continue

                        try:
                            du=local_quad_derivative(t,u,i,2)
                            dv=local_quad_derivative(t,v,i,2)
                            dw=local_quad_derivative(t,w,i,2)
                            pd_num=local_quad_derivative(t,pr,i,2)
                            qd_num=local_quad_derivative(t,qr,i,2)
                            rd_num=local_quad_derivative(t,rr,i,2)
                        except Exception:
                            continue

                        if None in (du,dv,dw,pd_num,qd_num,rd_num):
                            continue

                        omega=(pr[i],qr[i],rr[i])
                        vel=(u[i],v[i],w[i])
                        vdot=(du,dv,dw)
                        ain=add(vdot,cross(omega,vel))

                        gb=(
                            -g*math.sin(theta[i]),
                            g*math.sin(phi[i])*math.cos(theta[i]),
                            g*math.cos(phi[i])*math.cos(theta[i])
                        )

                        n=Om[i]/(2.0*PI)
                        if abs(n) < 1e-10:
                            continue
                        Jadv=Va[i]/(n*D)
                        CT=poly(ct,Jadv); CQ=poly(cq,Jadv)
                        T=et*CT*rho*n*n*(D**4)
                        Q=eq*CQ*rho*n*n*(D**5)

                        Fprop=(T,0.0,0.0)
                        Mprop=(-Q,-Jp*Om[i]*rr[i],+Jp*Om[i]*qr[i])

                        Fa=sub(sub(scale(m,ain),Fprop),scale(m,gb))

                        omega_dot=(pd[i],qd[i],rd[i])
                        Iwd=matvec(I,omega_dot)
                        Iw=matvec(I,omega)
                        Ma=sub(add(Iwd,cross(omega,Iw)),Mprop)

                        qbar=0.5*rho*Va[i]*Va[i]
                        if not math.isfinite(qbar) or qbar <= 1e-8:
                            qbar_invalid += 1
                            continue

                        CX=Fa[0]/(qbar*S)
                        CYb=Fa[1]/(qbar*S)
                        CZ=Fa[2]/(qbar*S)

                        ca=math.cos(alpha[i]); sa=math.sin(alpha[i])
                        cb=math.cos(beta[i]); sb=math.sin(beta[i])

                        Drag=-(ca*cb*Fa[0] + sb*Fa[1] + sa*cb*Fa[2])
                        Ywind=-ca*sb*Fa[0] + cb*Fa[1] - sa*sb*Fa[2]
                        Lift=sa*Fa[0] - ca*Fa[2]

                        CD=Drag/(qbar*S)
                        CYw=Ywind/(qbar*S)
                        CL=Lift/(qbar*S)

                        Cl=Ma[0]/(qbar*S*b)
                        Cm=Ma[1]/(qbar*S*cbar)
                        Cn=Ma[2]/(qbar*S*b)

                        outs=[CX,CYb,CZ,CD,CYw,CL,Cl,Cm,Cn,T,Q,*Fa,*Ma]
                        if not all(math.isfinite(x) for x in outs):
                            nonfinite_outputs += 1
                            continue

                        out={
                            "configuration":cname,
                            "file_name":os.path.basename(path),
                            "sample_index":i,
                            "TIME":t[i],
                            "Va":Va[i],"alpha":alpha[i],"beta":beta[i],
                            "phi":phi[i],"theta":theta[i],"psi":psi[i],
                            "p":pr[i],"q":qr[i],"r":rr[i],
                            "du_dt":du,"dv_dt":dv,"dw_dt":dw,
                            "p_dot":pd[i],"q_dot":qd[i],"r_dot":rd[i],
                            "Omega_prop_rad_s":Om[i],"n_prop_rev_s":n,"J_advance":Jadv,
                            "C_T":CT,"C_Q":CQ,"T_prop_N":T,"Q_drag_Nm":Q,
                            "Fprop_X":Fprop[0],"Fprop_Y":Fprop[1],"Fprop_Z":Fprop[2],
                            "Mprop_L":Mprop[0],"Mprop_M":Mprop[1],"Mprop_N":Mprop[2],
                            "g_body_X":gb[0],"g_body_Y":gb[1],"g_body_Z":gb[2],
                            "a_inertial_body_X":ain[0],"a_inertial_body_Y":ain[1],"a_inertial_body_Z":ain[2],
                            "Faero_X":Fa[0],"Faero_Y":Fa[1],"Faero_Z":Fa[2],
                            "Maero_L":Ma[0],"Maero_M":Ma[1],"Maero_N":Ma[2],
                            "qbar":qbar,
                            "C_X_body":CX,"C_Y_body":CYb,"C_Z_body":CZ,
                            "C_D_drag":CD,"C_Y_wind":CYw,"C_L_lift":CL,
                            "C_l_roll":Cl,"C_m_pitch":Cm,"C_n_yaw":Cn,
                            "src_CX":F(rows[i].get("CX")),"src_CY":F(rows[i].get("CY")),"src_CZ":F(rows[i].get("CZ")),
                            "src_CD":F(rows[i].get("CD")),"src_CS":F(rows[i].get("CS")),"src_CL":F(rows[i].get("CL")),
                            "src_Cl":F(rows[i].get("Cl")),"src_Cm":F(rows[i].get("Cm")),"src_Cn":F(rows[i].get("Cn")),
                            "src_prop_thrust_bodyX":F(rows[i].get("prop_thrust_bodyX")),
                            "src_prop_torque_drag":F(rows[i].get("prop_torque_drag")),
                            "src_prop_torque_bodyX":F(rows[i].get("prop_torque_bodyX")),
                            "src_prop_torque_bodyY":F(rows[i].get("prop_torque_bodyY")),
                            "src_prop_torque_bodyZ":F(rows[i].get("prop_torque_bodyZ")),
                            "ax_cg":axcg[i],"ay_cg":aycg[i],"az_cg":azcg[i],
                            "p_dot_from_rate":pd_num,"q_dot_from_rate":qd_num,"r_dot_from_rate":rd_num
                        }
                        sw.writerow(out)
                        retained += 1
                        total_retained += 1

                        coeff_map={
                            "C_X_body":CX,"C_Y_body":CYb,"C_Z_body":CZ,
                            "C_D_drag":CD,"C_Y_wind":CYw,"C_L_lift":CL,
                            "C_l_roll":Cl,"C_m_pitch":Cm,"C_n_yaw":Cn
                        }
                        for k,val in coeff_map.items():
                            coeff_by_cfg[cname][k].append(val)
                            file_coeff[k].append(val)

                        # Reference diagnostics only; never used to alter the reconstruction.
                        ref_metrics[cname+"|C_X_body"].add(CX,F(rows[i].get("CX")))
                        ref_metrics[cname+"|C_Y_body"].add(CYb,F(rows[i].get("CY")))
                        ref_metrics[cname+"|C_Z_body"].add(CZ,F(rows[i].get("CZ")))
                        ref_metrics[cname+"|C_D_drag"].add(CD,F(rows[i].get("CD")))
                        ref_metrics[cname+"|C_Y_wind"].add(CYw,F(rows[i].get("CS")))
                        ref_metrics[cname+"|C_L_lift"].add(CL,F(rows[i].get("CL")))
                        ref_metrics[cname+"|C_l_roll"].add(Cl,F(rows[i].get("Cl")))
                        ref_metrics[cname+"|C_m_pitch"].add(Cm,F(rows[i].get("Cm")))
                        ref_metrics[cname+"|C_n_yaw"].add(Cn,F(rows[i].get("Cn")))

                        prop_metrics[cname+"|T"].add(T,F(rows[i].get("prop_thrust_bodyX")))
                        prop_metrics[cname+"|Q_drag"].add(Q,F(rows[i].get("prop_torque_drag")))
                        prop_metrics[cname+"|Mx"].add(Mprop[0],F(rows[i].get("prop_torque_bodyX")))
                        prop_metrics[cname+"|My"].add(Mprop[1],F(rows[i].get("prop_torque_bodyY")))
                        prop_metrics[cname+"|Mz"].add(Mprop[2],F(rows[i].get("prop_torque_bodyZ")))

                        # Acceleration semantics diagnostics only.
                        accel_metrics[cname+"|ain_x"].add(ain[0],axcg[i])
                        accel_metrics[cname+"|ain_y"].add(ain[1],aycg[i])
                        accel_metrics[cname+"|ain_z"].add(ain[2],azcg[i])
                        spec=sub(ain,gb)
                        accel_metrics[cname+"|specific_x"].add(spec[0],axcg[i])
                        accel_metrics[cname+"|specific_y"].add(spec[1],aycg[i])
                        accel_metrics[cname+"|specific_z"].add(spec[2],azcg[i])

                        deriv_metrics[cname+"|p_dot"].add(pd_num,pd[i])
                        deriv_metrics[cname+"|q_dot"].add(qd_num,qd[i])
                        deriv_metrics[cname+"|r_dot"].add(rd_num,rd[i])

                    fs={"configuration":cname,"file_name":os.path.basename(path),
                        "source_rows":len(rows),"retained_rows":retained}
                    for k,vals in sorted(file_coeff.items()):
                        fs[k+"_median"]=statistics.median(vals) if vals else None
                        fs[k+"_p05"]=qtile(vals,0.05)
                        fs[k+"_p95"]=qtile(vals,0.95)
                    file_rows.append(fs)

                except Exception as e:
                    parse_errors.append({"configuration":cname,"file_name":os.path.basename(path),"error":repr(e)})
    finally:
        sf.close()

    # File summary.
    file_fields=["configuration","file_name","source_rows","retained_rows"]
    coeff_names=["C_X_body","C_Y_body","C_Z_body","C_D_drag","C_Y_wind","C_L_lift",
                 "C_l_roll","C_m_pitch","C_n_yaw"]
    for k in coeff_names:
        file_fields += [k+"_median",k+"_p05",k+"_p95"]
    with open(os.path.join(a.out,"P1_A1_FILE_SUMMARY.csv"),"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=file_fields)
        w.writeheader()
        w.writerows(file_rows)

    # Overall distribution summary.
    dist=[]
    for cname in ("clean","iced"):
        for k in coeff_names:
            vals=coeff_by_cfg[cname][k]
            dist.append({
                "configuration":cname,"quantity":k,"n":len(vals),
                "median":statistics.median(vals) if vals else None,
                "p05":qtile(vals,0.05),"p95":qtile(vals,0.95),
                "mean":sum(vals)/len(vals) if vals else None,
                "rms":math.sqrt(sum(x*x for x in vals)/len(vals)) if vals else None
            })
    with open(os.path.join(a.out,"P1_A1_COEFFICIENT_DISTRIBUTIONS.csv"),"w",newline="",encoding="utf-8-sig") as f:
        fields=["configuration","quantity","n","median","p05","p95","mean","rms"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(dist)

    # Reference diagnostics.
    def write_metric_map(name,mp):
        rows=[]
        for key,m in sorted(mp.items()):
            cfg,q=key.split("|",1)
            rows.append({"configuration":cfg,"quantity":q,**m.out()})
        with open(os.path.join(a.out,name),"w",newline="",encoding="utf-8-sig") as f:
            fields=["configuration","quantity","n","rmse","bias","nrmse","corr"]
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
        return rows

    ref_rows=write_metric_map("P1_A1_SOURCE_COEFFICIENT_REFERENCE_DIAGNOSTICS.csv",ref_metrics)
    prop_rows=write_metric_map("P1_A1_PROPULSION_IDENTITY_DIAGNOSTICS.csv",prop_metrics)
    accel_rows=write_metric_map("P1_A1_ACCELERATION_CROSSCHECK_DIAGNOSTICS.csv",accel_metrics)
    deriv_rows=write_metric_map("P1_A1_ANGULAR_DERIVATIVE_CROSSCHECK.csv",deriv_metrics)

    with open(os.path.join(a.out,"P1_A1_PARSE_ERRORS.csv"),"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=["configuration","file_name","error"]); w.writeheader(); w.writerows(parse_errors)

    prop_max=max((r["nrmse"] for r in prop_rows if r["nrmse"] is not None),default=None)

    result={
        "scope":{
            "identification_files_used":54,
            "validation_files_used":validation_used,
            "source_rows_seen":total_rows_seen,
            "retained_rows":total_retained,
            "expected_retained_rows_if_all_files_valid":total_rows_seen-4*54
        },
        "integrity":{
            "parse_errors":len(parse_errors),
            "nonfinite_outputs":nonfinite_outputs,
            "qbar_invalid_rows":qbar_invalid,
            "propulsion_reference_max_nrmse":prop_max
        },
        "method":{
            "translational_derivative":"5-point local quadratic least-squares on actual TIME",
            "rotational_acceleration_primary":"source p_dot/q_dot/r_dot",
            "propulsion":"P1-A0R3 corrected semantics",
            "source_aero_coefficients_role":"reference diagnostic only"
        }
    }
    with open(os.path.join(a.out,"P1_A1_RECONSTRUCTION_AUDIT.json"),"w",encoding="utf-8") as f:
        json.dump(result,f,indent=2)

    print("P1_A1_RECONSTRUCTION_SUMMARY")
    print("identification_files_used =",result["scope"]["identification_files_used"])
    print("validation_files_used     =",result["scope"]["validation_files_used"])
    print("source_rows_seen          =",total_rows_seen)
    print("retained_rows             =",total_retained)
    print("expected_retained_rows    =",result["scope"]["expected_retained_rows_if_all_files_valid"])
    print("parse_errors              =",len(parse_errors))
    print("nonfinite_outputs         =",nonfinite_outputs)
    print("qbar_invalid_rows         =",qbar_invalid)
    print("prop_ref_max_nrmse        =",prop_max)
    print("")
    print("PRELIMINARY IDENTIFICATION-DATA COEFFICIENT DISTRIBUTIONS")
    for row in dist:
        if row["quantity"] in ("C_D_drag","C_L_lift","C_m_pitch"):
            print("  {0:5s} {1:10s}: median={2:.6g}, p05={3:.6g}, p95={4:.6g}, n={5}".format(
                row["configuration"],row["quantity"],row["median"],row["p05"],row["p95"],row["n"]
            ))
    print("")
    print("NOTE: coefficient distributions are pilot diagnostics only; no clean-vs-iced scientific inference is authorized at P1-A1.")

if __name__=="__main__":
    main()
