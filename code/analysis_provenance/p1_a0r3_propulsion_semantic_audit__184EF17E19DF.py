import argparse,csv,json,math,os,re

EPS=1e-14

def F(x):
    try:
        y=float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None

class M:
    def __init__(self):
        self.n=0; self.se=0.0; self.ta=0.0; self.ma=0.0
    def add(self,p,t):
        if p is None or t is None: return
        e=p-t
        self.n+=1; self.se+=e*e; self.ta+=t*t; self.ma=max(self.ma,abs(e))
    def out(self):
        if not self.n: return {"n":0,"rmse":None,"target_rms":None,"nrmse":None,"max_abs":None}
        rm=(self.se/self.n)**0.5
        tr=(self.ta/self.n)**0.5
        return {"n":self.n,"rmse":rm,"target_rms":tr,"nrmse":rm/max(tr,EPS),"max_abs":self.ma}

def poly(c,J):
    return c[0]+c[1]*J+c[2]*J*J+c[3]*J*J*J

def params(path):
    with open(path,"r",encoding="utf-8-sig") as f: return json.load(f)

def files(folder):
    out=[]
    for n in sorted(os.listdir(folder)):
        m=re.search(r"_ID_(\d+)\.csv$",n,re.I)
        if m and int(m.group(1))<=3: out.append(os.path.join(folder,n))
    return out

def best(metrics):
    vals=[]
    for k,v in metrics.items():
        o=v.out()
        score=float("inf") if o["nrmse"] is None else o["nrmse"]
        vals.append((score,k,o))
    vals.sort(key=lambda x:x[0])
    return vals[0], vals

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--clean-dir",required=True)
    ap.add_argument("--iced-dir",required=True)
    ap.add_argument("--clean-json",required=True)
    ap.add_argument("--iced-json",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    result={"scope":{"validation_files_used":0},"configs":{}}
    total_files=0
    total_rows=0

    for cname,folder,jpath in [("clean",a.clean_dir,a.clean_json),("iced",a.iced_dir,a.iced_json)]:
        p=params(jpath)
        D=float(p["D"]); rho=float(p["rho"]); Jp=float(p["J_prop"])
        ct=[float(p["C_T_0"]),float(p["C_T_1"]),float(p["C_T_2"]),float(p["C_T_3"])]
        cq=[float(p["C_Q_0"]),float(p["C_Q_1"]),float(p["C_Q_2"]),float(p["C_Q_3"])]
        et=float(p["eta_prop_T"]); eq=float(p["eta_prop_Q"])

        met={
            "J_if_RPS_is_rev_s":M(),
            "J_if_RPS_is_rad_s":M(),
            "T_correct_rad_s_eta_prop":M(),
            "T_correct_rad_s_no_eta":M(),
            "Q_correct_rad_s_eta_prop":M(),
            "Q_correct_rad_s_no_eta":M(),
            "bodyX_plus_Q":M(),
            "bodyX_minus_Q":M(),
            "gyro_minus_plus_rad_s":M(),
            "gyro_plus_minus_rad_s":M(),
        }

        cfg_files=files(folder); total_files+=len(cfg_files)
        for path in cfg_files:
            with open(path,"r",newline="",encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    total_rows+=1
                    Va=F(row.get("Va_EKF_CG")); Om=F(row.get("RPS")); Js=F(row.get("J"))
                    qs=F(row.get("q")); rs=F(row.get("r"))
                    Tsrc=F(row.get("prop_thrust_bodyX"))
                    Qsrc=F(row.get("prop_torque_drag"))
                    Xsrc=F(row.get("prop_torque_bodyX"))
                    Ysrc=F(row.get("prop_torque_bodyY"))
                    Zsrc=F(row.get("prop_torque_bodyZ"))

                    if None not in (Va,Om,Js) and abs(Om)>EPS:
                        J_rps=Va/(Om*D)
                        J_rad=(2.0*math.pi*Va)/(Om*D)
                        met["J_if_RPS_is_rev_s"].add(J_rps,Js)
                        met["J_if_RPS_is_rad_s"].add(J_rad,Js)

                    if None in (Om,Js) or abs(Om)<=EPS: continue

                    n=Om/(2.0*math.pi)
                    CT=poly(ct,Js); CQ=poly(cq,Js)
                    T0=CT*rho*n*n*(D**4)
                    Q0=CQ*rho*n*n*(D**5)
                    T=et*T0
                    Q=eq*Q0

                    met["T_correct_rad_s_eta_prop"].add(T,Tsrc)
                    met["T_correct_rad_s_no_eta"].add(T0,Tsrc)
                    met["Q_correct_rad_s_eta_prop"].add(Q,Qsrc)
                    met["Q_correct_rad_s_no_eta"].add(Q0,Qsrc)

                    met["bodyX_plus_Q"].add(Qsrc,Xsrc)
                    met["bodyX_minus_Q"].add(-Qsrc,Xsrc)

                    if None not in (qs,rs,Ysrc,Zsrc):
                        h=Jp*Om
                        # Candidate A: My=-h*r, Mz=+h*q
                        met["gyro_minus_plus_rad_s"].add(-h*rs,Ysrc)
                        met["gyro_minus_plus_rad_s"].add(+h*qs,Zsrc)
                        # Candidate B: My=+h*r, Mz=-h*q
                        met["gyro_plus_minus_rad_s"].add(+h*rs,Ysrc)
                        met["gyro_plus_minus_rad_s"].add(-h*qs,Zsrc)

        cfg={k:v.out() for k,v in met.items()}
        cfg["best_J"]="RAD_S" if cfg["J_if_RPS_is_rad_s"]["nrmse"] <= cfg["J_if_RPS_is_rev_s"]["nrmse"] else "REV_S"
        cfg["best_T"]="ETA_PROP" if cfg["T_correct_rad_s_eta_prop"]["nrmse"] <= cfg["T_correct_rad_s_no_eta"]["nrmse"] else "NO_ETA"
        cfg["best_Q"]="ETA_PROP" if cfg["Q_correct_rad_s_eta_prop"]["nrmse"] <= cfg["Q_correct_rad_s_no_eta"]["nrmse"] else "NO_ETA"
        cfg["best_bodyX"]="-Q" if cfg["bodyX_minus_Q"]["nrmse"] <= cfg["bodyX_plus_Q"]["nrmse"] else "+Q"
        cfg["best_gyro"]="MINUS_HR_PLUS_HQ" if cfg["gyro_minus_plus_rad_s"]["nrmse"] <= cfg["gyro_plus_minus_rad_s"]["nrmse"] else "PLUS_HR_MINUS_HQ"
        result["configs"][cname]=cfg

    result["scope"]["identification_files_used"]=total_files
    result["scope"]["rows_seen"]=total_rows

    with open(os.path.join(a.out,"P1_A0R3_CORRECTED_SEMANTIC_DIAGNOSTICS.json"),"w",encoding="utf-8") as f:
        json.dump(result,f,indent=2)

    rows=[]
    for cname,cfg in result["configs"].items():
        for key,val in cfg.items():
            if isinstance(val,dict) and "nrmse" in val:
                rows.append({"Configuration":cname,"Candidate":key,**val})
    with open(os.path.join(a.out,"P1_A0R3_CORRECTED_CANDIDATE_METRICS.csv"),"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=["Configuration","Candidate","n","rmse","target_rms","nrmse","max_abs"])
        w.writeheader(); w.writerows(rows)

    print("P1_A0R2_CORRECTED_SEMANTIC_SUMMARY")
    print("identification_files_used =",total_files)
    print("validation_files_used     = 0")
    print("rows_seen                 =",total_rows)
    for cname,cfg in result["configs"].items():
        print("")
        print(cname.upper())
        print("  J best       =",cfg["best_J"],
              "rad_nrmse=",cfg["J_if_RPS_is_rad_s"]["nrmse"],
              "rev_nrmse=",cfg["J_if_RPS_is_rev_s"]["nrmse"])
        print("  T best       =",cfg["best_T"],
              "eta_nrmse=",cfg["T_correct_rad_s_eta_prop"]["nrmse"])
        print("  Q best       =",cfg["best_Q"],
              "eta_nrmse=",cfg["Q_correct_rad_s_eta_prop"]["nrmse"])
        print("  bodyX best   =",cfg["best_bodyX"],
              "minus_nrmse=",cfg["bodyX_minus_Q"]["nrmse"])
        print("  gyro best    =",cfg["best_gyro"],
              "minusplus_nrmse=",cfg["gyro_minus_plus_rad_s"]["nrmse"])

if __name__=="__main__":
    main()
