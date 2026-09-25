import argparse,csv,json,math,os,statistics
from collections import defaultdict

try:
    from scipy.stats import t as student_t
    HAVE_SCIPY=True
except Exception:
    HAVE_SCIPY=False

EPS=1e-14

def F(x):
    try:
        y=float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None

def load_csv(path):
    with open(path,"r",newline="",encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def tcrit95(df):
    if HAVE_SCIPY:
        return float(student_t.ppf(0.975,df))
    fallback={11:2.200985,13:2.160369}
    return fallback.get(df,1.96)

def jk(point,loo):
    vals=[float(x) for x in loo if x is not None and math.isfinite(float(x))]
    G=len(vals)
    if G<2:return {"G":G,"se":None,"ci_low":None,"ci_high":None,"loo_min":None,"loo_max":None}
    m=sum(vals)/G
    se=math.sqrt((G-1.0)/G*sum((x-m)**2 for x in vals))
    tc=tcrit95(G-1)
    return {"G":G,"se":se,"ci_low":point-tc*se,"ci_high":point+tc*se,
            "loo_min":min(vals),"loo_max":max(vals)}

def sign(x):
    if abs(x)<=EPS:return 0
    return 1 if x>0 else -1

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--b1-est",required=True)
    ap.add_argument("--b1-loo",required=True)
    ap.add_argument("--b2-effects",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    est=load_csv(a.b1_est)
    loo=load_csv(a.b1_loo)
    eff=load_csv(a.b2_effects)

    E={}
    for r in est:
        E[(r["axis"],r["outcome"],r["predictor"])]={
            "clean":float(r["clean_estimate"]),
            "iced":float(r["iced_estimate"]),
            "delta":float(r["delta_iced_minus_clean"])
        }

    C={}
    for r in eff:
        C[(r["axis"],r["outcome"],r["predictor"])]={
            "classification":r["classification"],
            "bh":float(r["bh_fdr_p"]),
            "holm":float(r["holm_p"]),
            "sensitivity_stable":str(r["sensitivity_stable"]).lower()=="true"
        }

    L=defaultdict(dict)
    for r in loo:
        key=(r["axis"],r["outcome"])
        cluster=r["left_out_cluster"]
        L[key][cluster]=r

    # Required frozen B2-qualified effects.
    k_cm=("lon","C_m_pitch","alpha")
    k_clde=("lon","C_L_lift","delta_e")
    k_clda=("lat","C_l_roll","delta_a")
    k_cnp=("lat","C_n_yaw","p_hat")
    k_cyp=("lat","C_Y_wind","p_hat")

    required=[k_cm,k_clde,k_clda,k_cnp,k_cyp]
    for k in required:
        if k not in E or k not in C:
            raise RuntimeError("Missing required derivative/effect row: %r"%(k,))

    consequences=[]

    # 1) Pitch restoring derivative magnitude change: actual stability derivative.
    cmc=E[k_cm]["clean"]; cmi=E[k_cm]["iced"]
    restore_ratio=abs(cmi)/abs(cmc)
    restore_reduction=1.0-restore_ratio
    lrows=[]
    for cluster,r in L[("lon","C_m_pitch")].items():
        cc=float(r["clean_alpha"]);ii=float(r["iced_alpha"])
        if abs(cc)>EPS:lrows.append(1.0-abs(ii)/abs(cc))
    j=jk(restore_reduction,lrows)
    consequences.append({
        "metric":"PITCH_RESTORING_SLOPE_MAGNITUDE_REDUCTION",
        "source_derivative":"C_m_alpha",
        "evidence_class":C[k_cm]["classification"],
        "clean_value":abs(cmc),"iced_value":abs(cmi),
        "consequence_value":restore_reduction,
        "units":"fraction",
        "jackknife_G":j["G"],"jackknife_se":j["se"],
        "ci95_low":j["ci_low"],"ci95_high":j["ci_high"],
        "loo_min":j["loo_min"],"loo_max":j["loo_max"],
        "authorized_wording":"weaker longitudinal pitch-restoring aerodynamic slope over the frozen identification envelope",
        "prohibited_wording":"full static-margin or short-period eigenvalue change without additional model closure"
    })

    # 2) Secondary static-margin proxy -Cm_alpha/CL_alpha. Not a full static margin.
    kla=("lon","C_L_lift","alpha")
    clac=E[kla]["clean"]; clai=E[kla]["iced"]
    proxy_c=-cmc/clac
    proxy_i=-cmi/clai
    proxy_reduction=1.0-proxy_i/proxy_c
    p_loo=[]
    cmloo=L[("lon","C_m_pitch")]
    clloo=L[("lon","C_L_lift")]
    common=sorted(set(cmloo).intersection(clloo))
    valid_proxy=0
    for cluster in common:
        cc=float(cmloo[cluster]["clean_alpha"]);ci=float(cmloo[cluster]["iced_alpha"])
        lc=float(clloo[cluster]["clean_alpha"]);li=float(clloo[cluster]["iced_alpha"])
        if abs(lc)<=EPS or abs(li)<=EPS:continue
        pc=-cc/lc;pi=-ci/li
        if pc>0 and pi>0:
            p_loo.append(1.0-pi/pc);valid_proxy+=1
    jp=jk(proxy_reduction,p_loo)
    proxy_authorized=(proxy_c>0 and proxy_i>0 and valid_proxy==len(common))
    consequences.append({
        "metric":"APPARENT_STATIC_MARGIN_PROXY_REDUCTION",
        "source_derivative":"-C_m_alpha/C_L_alpha",
        "evidence_class":"SECONDARY_PROXY_NOT_PRIMARY_CLAIM",
        "clean_value":proxy_c,"iced_value":proxy_i,
        "consequence_value":proxy_reduction,
        "units":"fraction",
        "jackknife_G":jp["G"],"jackknife_se":jp["se"],
        "ci95_low":jp["ci_low"],"ci95_high":jp["ci_high"],
        "loo_min":jp["loo_min"],"loo_max":jp["loo_max"],
        "authorized_wording":"apparent derivative-ratio proxy only" if proxy_authorized else "NOT AUTHORIZED",
        "prohibited_wording":"geometric static margin, neutral-point shift, or certified stability-margin change"
    })

    # 3) Longitudinal mixed-command to lift coupling: sign reversal, not physical elevator effectiveness.
    dec=E[k_clde]["clean"]; dei=E[k_clde]["iced"]
    signrev=(sign(dec)!=0 and sign(dei)!=0 and sign(dec)!=sign(dei))
    sr=[]
    dloo=L[("lon","C_L_lift")]
    for cluster,r in dloo.items():
        c=float(r["clean_delta_e"]);i=float(r["iced_delta_e"])
        sr.append(1.0 if (sign(c)!=0 and sign(i)!=0 and sign(c)!=sign(i)) else 0.0)
    sr_fraction=sum(sr)/len(sr) if sr else None
    consequences.append({
        "metric":"LONGITUDINAL_MIXED_COMMAND_LIFT_COUPLING_SIGN_REVERSAL",
        "source_derivative":"C_L_delta_e",
        "evidence_class":C[k_clde]["classification"],
        "clean_value":dec,"iced_value":dei,
        "consequence_value":1.0 if signrev else 0.0,
        "units":"binary",
        "jackknife_G":len(sr),"jackknife_se":None,
        "ci95_low":None,"ci95_high":None,
        "loo_min":min(sr) if sr else None,"loo_max":max(sr) if sr else None,
        "authorized_wording":"effective mixed-command-to-lift slope changes sign within the frozen identification model; LOPO sign-reversal fraction=%.3f"%sr_fraction,
        "prohibited_wording":"physical elevator/elevon aerodynamic-control reversal without actuator-angle/mixer calibration"
    })

    # 4) Effective roll-command coupling magnitude change.
    rac=E[k_clda]["clean"]; rai=E[k_clda]["iced"]
    roll_change=abs(rai)/abs(rac)-1.0
    rloo=[]
    for cluster,r in L[("lat","C_l_roll")].items():
        c=float(r["clean_delta_a"]);i=float(r["iced_delta_a"])
        if abs(c)>EPS:rloo.append(abs(i)/abs(c)-1.0)
    jr=jk(roll_change,rloo)
    consequences.append({
        "metric":"ROLL_MIXED_COMMAND_COUPLING_MAGNITUDE_CHANGE",
        "source_derivative":"C_l_delta_a",
        "evidence_class":C[k_clda]["classification"],
        "clean_value":abs(rac),"iced_value":abs(rai),
        "consequence_value":roll_change,
        "units":"fraction",
        "jackknife_G":jr["G"],"jackknife_se":jr["se"],
        "ci95_low":jr["ci_low"],"ci95_high":jr["ci_high"],
        "loo_min":jr["loo_min"],"loo_max":jr["loo_max"],
        "authorized_wording":"effective mixed-command roll-coupling magnitude change",
        "prohibited_wording":"physical aileron/elevon control-effectiveness change without actuator-angle/mixer calibration"
    })

    # 5) Roll-rate to yawing-moment coupling magnitude change.
    npc=E[k_cnp]["clean"]; npi=E[k_cnp]["iced"]
    np_change=abs(npi)/abs(npc)-1.0
    nloo=[]
    for cluster,r in L[("lat","C_n_yaw")].items():
        c=float(r["clean_p_hat"]);i=float(r["iced_p_hat"])
        if abs(c)>EPS:nloo.append(abs(i)/abs(c)-1.0)
    jn=jk(np_change,nloo)
    consequences.append({
        "metric":"ROLL_RATE_TO_YAW_MOMENT_COUPLING_MAGNITUDE_CHANGE",
        "source_derivative":"C_n_p",
        "evidence_class":C[k_cnp]["classification"],
        "clean_value":abs(npc),"iced_value":abs(npi),
        "consequence_value":np_change,
        "units":"fraction",
        "jackknife_G":jn["G"],"jackknife_se":jn["se"],
        "ci95_low":jn["ci_low"],"ci95_high":jn["ci_high"],
        "loo_min":jn["loo_min"],"loo_max":jn["loo_max"],
        "authorized_wording":"stronger roll-rate/yawing-moment aerodynamic cross-coupling over the frozen identification envelope",
        "prohibited_wording":"Dutch-roll damping/eigenvalue change without a closed lateral state matrix"
    })

    # Explicitly excluded model-sensitive candidate.
    exclusions=[{
        "derivative":"C_Y_p",
        "classification":C[k_cyp]["classification"],
        "reason":"primary CI signal did not survive B2 multiplicity/sensitivity qualification",
        "allowed_use":"report as model-sensitive exploratory signal only",
        "forbidden_use":"stability/control consequence mapping"
    }]

    # Integrity / authority hierarchy.
    allowed_classes={"STRONG_HOLM_ROBUST","FDR_ROBUST"}
    mapped_effects=[
        ("C_m_alpha",C[k_cm]["classification"]),
        ("C_L_delta_e",C[k_clde]["classification"]),
        ("C_l_delta_a",C[k_clda]["classification"]),
        ("C_n_p",C[k_cnp]["classification"])
    ]
    authority_pass=all(cls in allowed_classes for name,cls in mapped_effects)

    finite_pass=True
    for r in consequences:
        if r["metric"]=="LONGITUDINAL_MIXED_COMMAND_LIFT_COUPLING_SIGN_REVERSAL":
            continue
        for k in ("clean_value","iced_value","consequence_value"):
            if r[k] is None or not math.isfinite(float(r[k])):finite_pass=False

    global_pass=authority_pass and finite_pass

    with open(os.path.join(a.out,"P1_A4A_R1_CONSEQUENCE_METRICS.csv"),"w",newline="",encoding="utf-8-sig") as f:
        fields=list(consequences[0].keys())
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(consequences)

    with open(os.path.join(a.out,"P1_A4A_R1_EXCLUDED_EFFECTS.csv"),"w",newline="",encoding="utf-8-sig") as f:
        fields=["derivative","classification","reason","allowed_use","forbidden_use"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(exclusions)

    audit={
        "scope":{"validation_files_used":0,"identification_only":True},
        "authority_pass":authority_pass,
        "finite_metrics_pass":finite_pass,
        "mapped_effects":mapped_effects,
        "static_margin_proxy":{
            "clean":proxy_c,"iced":proxy_i,"fractional_reduction":proxy_reduction,
            "loo_all_positive":proxy_authorized,
            "primary_claim_authorized":False
        },
        "mixed_command_lift_sign_reversal":{
            "full_fit":signrev,
            "loo_sign_reversal_fraction":sr_fraction,
            "physical_surface_reversal_claim_authorized":False
        },
        "full_state_matrix_or_eigenvalue_claim_authorized":False,
        "global_pass":global_pass
    }

    with open(os.path.join(a.out,"P1_A4A_R1_AUDIT.json"),"w",encoding="utf-8") as f:
        json.dump(audit,f,indent=2)

    print("P1_A4A_CONSEQUENCE_MAPPING_SUMMARY")
    print("validation_files_used                   = 0")
    print("authority_pass                          =",authority_pass)
    print("finite_metrics_pass                     =",finite_pass)
    print("full_state_matrix/eigenvalue authorized = False")
    print("global_mapping_gate                     =",global_pass)
    print("")
    for r in consequences:
        val=r["consequence_value"]
        if r["units"]=="fraction":
            print("  {0}: {1:+.3%}, 95%CI={2}".format(
                r["metric"],val,
                "NA" if r["ci95_low"] is None else "[{:+.3%},{:+.3%}]".format(r["ci95_low"],r["ci95_high"])
            ))
        else:
            print("  {0}: {1}; {2}".format(r["metric"],val,r["authorized_wording"]))
    print("")
    print("Excluded from consequence mapping:")
    for r in exclusions:
        print("  {0}: {1}".format(r["derivative"],r["classification"]))

if __name__=="__main__":
    main()
