import argparse, csv, json, math, os, re, statistics
from collections import defaultdict

EPS=1e-12

def f(x):
    try:
        y=float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None

class Metric:
    def __init__(self):
        self.n=0; self.se=0.0; self.maxabs=0.0; self.target2=0.0
    def add(self,p,t):
        if p is None or t is None: return
        if not (math.isfinite(p) and math.isfinite(t)): return
        e=p-t
        self.n += 1
        self.se += e*e
        self.maxabs=max(self.maxabs,abs(e))
        self.target2 += t*t
    def out(self):
        rmse=math.sqrt(self.se/self.n) if self.n else None
        rms_t=math.sqrt(self.target2/self.n) if self.n else None
        nrmse=(rmse/max(rms_t,EPS)) if rmse is not None else None
        return {"n":self.n,"rmse":rmse,"max_abs":self.maxabs if self.n else None,"target_rms":rms_t,"nrmse":nrmse}

def poly(c0,c1,c2,c3,J):
    return c0+c1*J+c2*J*J+c3*J*J*J

def load_params(path):
    with open(path,"r",encoding="utf-8-sig") as fh:
        return json.load(fh)

def train_files(folder):
    out=[]
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(".csv"): continue
        m=re.search(r"_ID_(\d+)\.csv$",name,re.I)
        if m and int(m.group(1))<=3:
            out.append(os.path.join(folder,name))
    return out

def candidate_scales(p,kind):
    if kind=="T":
        return {
            "BASE":1.0,
            "ETA_PROP":float(p["eta_prop_T"]),
            "ETA_SYS_REF":float(p["eta_sys_ref_T"]),
            "ETA_SYS_APPARENT":float(p["eta_sys_apparent_T"]),
            "ETA_PROP_X_SYS_REF":float(p["eta_prop_T"])*float(p["eta_sys_ref_T"]),
            "ETA_PROP_X_SYS_APPARENT":float(p["eta_prop_T"])*float(p["eta_sys_apparent_T"]),
        }
    return {
        "BASE":1.0,
        "ETA_PROP":float(p["eta_prop_Q"]),
        "ETA_SYS_REF":float(p["eta_sys_ref_Q"]),
        "ETA_SYS_APPARENT":float(p["eta_sys_apparent_Q"]),
        "ETA_PROP_X_SYS_REF":float(p["eta_prop_Q"])*float(p["eta_sys_ref_Q"]),
        "ETA_PROP_X_SYS_APPARENT":float(p["eta_prop_Q"])*float(p["eta_sys_apparent_Q"]),
    }

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

    air={
        "Va_from_relative_velocity":Metric(),
        "alpha_plus":Metric(),
        "alpha_minus":Metric(),
        "beta_plus":Metric(),
        "beta_minus":Metric(),
    }
    per_config={}
    total_rows=0
    used_files=[]

    for cname,folder,p in configs:
        M={
            "J_Va":Metric(),
            "J_u":Metric(),
            "thrust":{},
            "qdrag":{},
            "bodyx_plus_drag":Metric(),
            "bodyx_minus_drag":Metric(),
            "gyro_plus":Metric(),
            "gyro_minus":Metric(),
        }

        for sign in (+1,-1):
            prefix="PLUS" if sign>0 else "MINUS"
            for nm,sc in candidate_scales(p,"T").items():
                M["thrust"][prefix+"_"+nm]=Metric()
            for nm,sc in candidate_scales(p,"Q").items():
                M["qdrag"][prefix+"_"+nm]=Metric()

        D=float(p["D"])
        rho=float(p["rho"])
        Jprop=float(p["J_prop"])
        ct=[float(p["C_T_0"]),float(p["C_T_1"]),float(p["C_T_2"]),float(p["C_T_3"])]
        cq=[float(p["C_Q_0"]),float(p["C_Q_1"]),float(p["C_Q_2"]),float(p["C_Q_3"])]

        for path in train_files(folder):
            used_files.append(path)
            with open(path,"r",newline="",encoding="utf-8-sig") as fh:
                rdr=csv.DictReader(fh)
                for row in rdr:
                    total_rows += 1
                    u=f(row.get("u_r_CG")); v=f(row.get("v_r_CG")); w=f(row.get("w_r_CG"))
                    Va=f(row.get("Va_EKF_CG")); alpha=f(row.get("alpha")); beta=f(row.get("beta"))
                    n=f(row.get("RPS")); Jsrc=f(row.get("J"))
                    p_rate=f(row.get("p")); q_rate=f(row.get("q")); r_rate=f(row.get("r"))

                    if None not in (u,v,w,Va):
                        Vpred=math.sqrt(u*u+v*v+w*w)
                        air["Va_from_relative_velocity"].add(Vpred,Va)

                        if alpha is not None:
                            aa=math.atan2(w,u)
                            air["alpha_plus"].add(aa,alpha)
                            air["alpha_minus"].add(-aa,alpha)

                        if beta is not None and Vpred>EPS:
                            arg=max(-1.0,min(1.0,v/Vpred))
                            bb=math.asin(arg)
                            air["beta_plus"].add(bb,beta)
                            air["beta_minus"].add(-bb,beta)

                    if n is None or abs(n)<1e-6:
                        continue

                    if Jsrc is not None and Va is not None:
                        M["J_Va"].add(Va/(n*D),Jsrc)
                    if Jsrc is not None and u is not None:
                        M["J_u"].add(u/(n*D),Jsrc)

                    # Use source J where available to audit the exact coefficient semantics.
                    if Jsrc is None:
                        continue

                    CT=poly(ct[0],ct[1],ct[2],ct[3],Jsrc)
                    CQ=poly(cq[0],cq[1],cq[2],cq[3],Jsrc)
                    baseT=CT*rho*n*n*(D**4)
                    baseQ=CQ*rho*n*n*(D**5)

                    Tsrc=f(row.get("prop_thrust_bodyX"))
                    Qsrc=f(row.get("prop_torque_drag"))
                    bodyX=f(row.get("prop_torque_bodyX"))
                    bodyY=f(row.get("prop_torque_bodyY"))
                    bodyZ=f(row.get("prop_torque_bodyZ"))

                    for sgn in (+1,-1):
                        prefix="PLUS" if sgn>0 else "MINUS"
                        for nm,sc in candidate_scales(p,"T").items():
                            M["thrust"][prefix+"_"+nm].add(sgn*sc*baseT,Tsrc)
                        for nm,sc in candidate_scales(p,"Q").items():
                            M["qdrag"][prefix+"_"+nm].add(sgn*sc*baseQ,Qsrc)

                    if Qsrc is not None and bodyX is not None:
                        M["bodyx_plus_drag"].add(Qsrc,bodyX)
                        M["bodyx_minus_drag"].add(-Qsrc,bodyX)

                    if None not in (bodyY,bodyZ,q_rate,r_rate):
                        omega=2.0*math.pi*n
                        h=Jprop*omega

                        # Candidate +: [My,Mz]=[+h*r,-h*q]
                        # Candidate -: [My,Mz]=[-h*r,+h*q]
                        M["gyro_plus"].add(+h*r_rate,bodyY)
                        M["gyro_plus"].add(-h*q_rate,bodyZ)
                        M["gyro_minus"].add(-h*r_rate,bodyY)
                        M["gyro_minus"].add(+h*q_rate,bodyZ)

        def best(d):
            out=[]
            for k,m in d.items():
                mm=m.out()
                out.append((float("inf") if mm["nrmse"] is None else mm["nrmse"],k,mm))
            out.sort(key=lambda z:z[0])
            return {"name":out[0][1],"metric":out[0][2],"ranking":[{"name":k,"metric":mm} for _,k,mm in out]}

        per_config[cname]={
            "J_Va":M["J_Va"].out(),
            "J_u":M["J_u"].out(),
            "best_J":"VA_OVER_ND" if (M["J_Va"].out()["nrmse"] or 1e99) <= (M["J_u"].out()["nrmse"] or 1e99) else "UR_OVER_ND",
            "thrust_best":best(M["thrust"]),
            "qdrag_best":best(M["qdrag"]),
            "bodyx_plus_drag":M["bodyx_plus_drag"].out(),
            "bodyx_minus_drag":M["bodyx_minus_drag"].out(),
            "bodyx_best":"+QDRAG" if (M["bodyx_plus_drag"].out()["nrmse"] or 1e99) <= (M["bodyx_minus_drag"].out()["nrmse"] or 1e99) else "-QDRAG",
            "gyro_plus":M["gyro_plus"].out(),
            "gyro_minus":M["gyro_minus"].out(),
            "gyro_best":"PLUS_HR_MINUS_HQ" if (M["gyro_plus"].out()["nrmse"] or 1e99) <= (M["gyro_minus"].out()["nrmse"] or 1e99) else "MINUS_HR_PLUS_HQ",
            "params":{
                "D":D,"rho":rho,"J_prop":Jprop,
                "eta_prop_T":float(p["eta_prop_T"]),
                "eta_prop_Q":float(p["eta_prop_Q"])
            }
        }

    air_out={k:v.out() for k,v in air.items()}
    air_out["alpha_best"]="PLUS_ATAN2_W_U" if (air_out["alpha_plus"]["rmse"] or 1e99) <= (air_out["alpha_minus"]["rmse"] or 1e99) else "MINUS_ATAN2_W_U"
    air_out["beta_best"]="PLUS_ASIN_V_OVER_VA" if (air_out["beta_plus"]["rmse"] or 1e99) <= (air_out["beta_minus"]["rmse"] or 1e99) else "MINUS_ASIN_V_OVER_VA"

    # Inertia tensor/scalar consistency.
    inertia={}
    for cname,folder,p in configs:
        I=p["I_cg"]
        checks={
            "Jx_vs_I00":abs(float(p["Jx"])-float(I[0][0])),
            "Jy_vs_I11":abs(float(p["Jy"])-float(I[1][1])),
            "Jz_vs_I22":abs(float(p["Jz"])-float(I[2][2])),
            "Jxz_vs_minus_I02":abs(float(p["Jxz"])+float(I[0][2])),
            "Jxz_vs_minus_I20":abs(float(p["Jxz"])+float(I[2][0])),
        }
        inertia[cname]={
            "checks":checks,
            "max_abs_difference":max(checks.values()),
            "canonical_tensor":[
                [float(p["Jx"]),0.0,-float(p["Jxz"])],
                [0.0,float(p["Jy"]),0.0],
                [-float(p["Jxz"]),0.0,float(p["Jz"])]
            ]
        }

    result={
        "scope":{
            "files_used":len(used_files),
            "rows_seen":total_rows,
            "validation_files_used":0,
            "selection_scope":"ID_1_TO_3_ONLY"
        },
        "airdata":air_out,
        "propulsion":per_config,
        "inertia":inertia,
    }

    with open(os.path.join(a.out,"P1_A0_SEMANTIC_DIAGNOSTICS.json"),"w",encoding="utf-8") as fh:
        json.dump(result,fh,indent=2)

    rows=[]
    for cname,x in per_config.items():
        rows.append({"Configuration":cname,"Test":"J_Va","Candidate":"VA_OVER_ND",**x["J_Va"]})
        rows.append({"Configuration":cname,"Test":"J_u","Candidate":"UR_OVER_ND",**x["J_u"]})
        for item in x["thrust_best"]["ranking"]:
            rows.append({"Configuration":cname,"Test":"THRUST","Candidate":item["name"],**item["metric"]})
        for item in x["qdrag_best"]["ranking"]:
            rows.append({"Configuration":cname,"Test":"QDRAG","Candidate":item["name"],**item["metric"]})
        rows.append({"Configuration":cname,"Test":"BODYX","Candidate":"+QDRAG",**x["bodyx_plus_drag"]})
        rows.append({"Configuration":cname,"Test":"BODYX","Candidate":"-QDRAG",**x["bodyx_minus_drag"]})
        rows.append({"Configuration":cname,"Test":"GYRO_YZ","Candidate":"PLUS_HR_MINUS_HQ",**x["gyro_plus"]})
        rows.append({"Configuration":cname,"Test":"GYRO_YZ","Candidate":"MINUS_HR_PLUS_HQ",**x["gyro_minus"]})

    with open(os.path.join(a.out,"P1_A0_SEMANTIC_CANDIDATE_RANKING.csv"),"w",newline="",encoding="utf-8-sig") as fh:
        fields=["Configuration","Test","Candidate","n","rmse","max_abs","target_rms","nrmse"]
        w=csv.DictWriter(fh,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("P1_A0_SEMANTIC_AUDIT_SUMMARY")
    print("files_used            =",len(used_files))
    print("validation_files_used = 0")
    print("rows_seen             =",total_rows)
    print("")
    print("Airdata identities")
    print("  Va nrmse      =",air_out["Va_from_relative_velocity"]["nrmse"])
    print("  alpha best    =",air_out["alpha_best"],"rmse=",min(air_out["alpha_plus"]["rmse"],air_out["alpha_minus"]["rmse"]))
    print("  beta best     =",air_out["beta_best"],"rmse=",min(air_out["beta_plus"]["rmse"],air_out["beta_minus"]["rmse"]))
    for cname,x in per_config.items():
        print("")
        print(cname.upper())
        print("  J best        =",x["best_J"],"Va_nrmse=",x["J_Va"]["nrmse"],"u_nrmse=",x["J_u"]["nrmse"])
        print("  thrust best   =",x["thrust_best"]["name"],"nrmse=",x["thrust_best"]["metric"]["nrmse"])
        print("  qdrag best    =",x["qdrag_best"]["name"],"nrmse=",x["qdrag_best"]["metric"]["nrmse"])
        print("  bodyX best    =",x["bodyx_best"])
        print("  gyro best     =",x["gyro_best"])
        print("  inertia maxΔ  =",inertia[cname]["max_abs_difference"])

if __name__=="__main__":
    main()
