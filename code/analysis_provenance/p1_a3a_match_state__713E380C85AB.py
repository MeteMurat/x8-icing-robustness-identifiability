import argparse,csv,json,math,os,re,statistics

EPS=1e-12

def f(x):
    try:
        y=float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None

def read_csv(path):
    with open(path,"r",newline="",encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))

def sample_support(path):
    support={}
    with open(path,"r",newline="",encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            cfg=r["configuration"]; fn=r["file_name"]; idx=int(r["sample_index"])
            support.setdefault((cfg,fn),set()).add(idx)
    return support

def mean(xs):
    return sum(xs)/len(xs) if xs else None

def sd(xs):
    if len(xs)<2:return 0.0
    m=mean(xs)
    return math.sqrt(sum((x-m)**2 for x in xs)/(len(xs)-1))

def robust_scale(xs):
    xs=sorted(xs)
    n=len(xs)
    if n<2:return 1.0
    def q(p):
        pos=(n-1)*p
        lo=int(math.floor(pos));hi=int(math.ceil(pos))
        if lo==hi:return xs[lo]
        w=pos-lo
        return xs[lo]*(1-w)+xs[hi]*w
    s=(q(.75)-q(.25))/1.349
    if s>1e-10:return s
    s2=sd(xs)
    return s2 if s2>1e-10 else 1.0

def active_feature_vector(row,axis,b,c):
    Va=f(row.get("Va_EKF_CG"))
    alpha=f(row.get("alpha"))
    beta=f(row.get("beta"))
    p=f(row.get("p")); q=f(row.get("q")); r=f(row.get("r"))
    elevator=f(row.get("elevator"))
    aileron=f(row.get("aileron"))
    throttle=f(row.get("throttle"))
    if Va is None or Va<=1e-6:return None

    if axis=="lon":
        vals={
            "Va":Va,
            "alpha":alpha,
            "q_hat":None if q is None else q*c/(2*Va),
            "elevator":elevator,
            "throttle":throttle
        }
    else:
        vals={
            "Va":Va,
            "alpha":alpha,
            "beta":beta,
            "p_hat":None if p is None else p*b/(2*Va),
            "r_hat":None if r is None else r*b/(2*Va),
            "aileron":aileron,
            "throttle":throttle
        }
    if any(v is None or not math.isfinite(v) for v in vals.values()):
        return None
    return vals

def max_abs_smd(matches,C,I,features,scales):
    vals=[]
    detail={}
    for feat in features:
        ca=[C[ci][feat] for ci,ii,d in matches]
        ia=[I[ii][feat] for ci,ii,d in matches]
        s=scales[feat]
        smd=abs(mean(ca)-mean(ia))/max(s,EPS)
        detail[feat]=smd
        vals.append(smd)
    return max(vals) if vals else None,detail

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--clean-dir",required=True)
    ap.add_argument("--iced-dir",required=True)
    ap.add_argument("--pilot-samples",required=True)
    ap.add_argument("--clean-json",required=True)
    ap.add_argument("--iced-json",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    with open(a.clean_json,"r",encoding="utf-8-sig") as fh:
        p=json.load(fh)
    b=float(p["b"]);c=float(p["c"])

    support=sample_support(a.pilot_samples)

    clean_names={}
    for fn in os.listdir(a.clean_dir):
        m=re.search(r"_ID_(\d+)\.csv$",fn,re.I)
        if m and int(m.group(1))<=3:
            clean_names[fn]=os.path.join(a.clean_dir,fn)
    iced_names={}
    for fn in os.listdir(a.iced_dir):
        m=re.search(r"_ID_(\d+)\.csv$",fn,re.I)
        if m and int(m.group(1))<=3:
            iced_names[fn]=os.path.join(a.iced_dir,fn)

    names=sorted(set(clean_names).intersection(iced_names))

    pair_summary=[]
    match_rows=[]
    feature_rows=[]
    errors=[]

    # Frozen pre-outcome matching parameters.
    RMS_CALIPER=1.0
    MAX_COMPONENT_Z=2.0
    MIN_MATCHES=25
    MIN_COVERAGE=0.15
    MAX_POST_SMD=0.25

    for fn in names:
        try:
            axis="lon" if fn.lower().startswith("lon_") else "lat"
            cr=read_csv(clean_names[fn]); ir=read_csv(iced_names[fn])
            cs=support.get(("clean",fn),set())
            iset=support.get(("iced",fn),set())

            C=[];I=[]
            for idx,row in enumerate(cr):
                if idx not in cs:continue
                v=active_feature_vector(row,axis,b,c)
                if v is not None:
                    C.append((idx,v))
            for idx,row in enumerate(ir):
                if idx not in iset:continue
                v=active_feature_vector(row,axis,b,c)
                if v is not None:
                    I.append((idx,v))

            if not C or not I:
                raise RuntimeError("no finite supported rows")

            features=list(C[0][1].keys())
            # Drop zero-variance features pairwise; this is outcome-blind and deterministic.
            active=[]
            centers={}
            scales={}
            for feat in features:
                xs=[v[feat] for _,v in C]+[v[feat] for _,v in I]
                s=robust_scale(xs)
                if s>1e-10:
                    active.append(feat)
                    centers[feat]=statistics.median(xs)
                    scales[feat]=s

            Cstd=[]
            for idx,v in C:
                Cstd.append((idx,{k:(v[k]-centers[k])/scales[k] for k in active},v))
            Istd=[]
            for idx,v in I:
                Istd.append((idx,{k:(v[k]-centers[k])/scales[k] for k in active},v))

            candidates=[]
            for ci,(cidx,cz,craw) in enumerate(Cstd):
                for ii,(iidx,iz,iraw) in enumerate(Istd):
                    diffs=[cz[k]-iz[k] for k in active]
                    maxz=max(abs(x) for x in diffs)
                    if maxz>MAX_COMPONENT_Z:continue
                    rms=math.sqrt(sum(x*x for x in diffs)/len(diffs))
                    if rms<=RMS_CALIPER:
                        candidates.append((rms,maxz,ci,ii))

            candidates.sort(key=lambda x:(x[0],x[1],x[2],x[3]))
            usedC=set();usedI=set();matches=[]
            for rms,maxz,ci,ii in candidates:
                if ci in usedC or ii in usedI:continue
                usedC.add(ci);usedI.add(ii)
                matches.append((ci,ii,rms,maxz))

            mtrip=[(ci,ii,rms) for ci,ii,rms,maxz in matches]
            Cmap=[x[2] for x in Cstd]
            Imap=[x[2] for x in Istd]

            if matches:
                mxsmd,smds=max_abs_smd(mtrip,Cmap,Imap,active,scales)
            else:
                mxsmd=None;smds={k:None for k in active}

            coverage=len(matches)/min(len(Cstd),len(Istd))
            usable=(len(matches)>=MIN_MATCHES and coverage>=MIN_COVERAGE and mxsmd is not None and mxsmd<=MAX_POST_SMD)

            pair_summary.append({
                "file_name":fn,
                "axis":axis,
                "clean_eligible_rows":len(Cstd),
                "iced_eligible_rows":len(Istd),
                "candidate_edges":len(candidates),
                "matched_rows":len(matches),
                "coverage_min_side":coverage,
                "active_feature_count":len(active),
                "max_post_match_abs_smd":mxsmd,
                "usable_pair":usable
            })

            for feat in active:
                feature_rows.append({
                    "file_name":fn,
                    "axis":axis,
                    "feature":feat,
                    "scale":scales[feat],
                    "post_match_abs_smd":smds.get(feat),
                    "used_in_matching":True
                })

            for rank,(ci,ii,rms,maxz) in enumerate(matches,1):
                cidx,Cz,Craw=Cstd[ci]
                iidx,Iz,Iraw=Istd[ii]
                out={
                    "file_name":fn,
                    "axis":axis,
                    "match_rank":rank,
                    "clean_sample_index":cidx,
                    "iced_sample_index":iidx,
                    "standardized_rms_distance":rms,
                    "max_component_abs_z":maxz
                }
                for feat in active:
                    out["clean_"+feat]=Craw[feat]
                    out["iced_"+feat]=Iraw[feat]
                    out["delta_"+feat]=Iraw[feat]-Craw[feat]
                match_rows.append(out)

        except Exception as e:
            errors.append({"file_name":fn,"error":repr(e)})

    # Global outcome-blind gate.
    usable=[r for r in pair_summary if r["usable_pair"]]
    overall_frac=len(usable)/len(pair_summary) if pair_summary else 0.0

    axes={}
    for axis in ("lon","lat"):
        rows=[r for r in pair_summary if r["axis"]==axis]
        u=[r for r in rows if r["usable_pair"]]
        axes[axis]={
            "pairs":len(rows),
            "usable_pairs":len(u),
            "usable_fraction":len(u)/len(rows) if rows else 0.0
        }

    cov=[r["coverage_min_side"] for r in usable]
    median_cov=statistics.median(cov) if cov else 0.0
    smds=[r["max_post_match_abs_smd"] for r in usable if r["max_post_match_abs_smd"] is not None]
    median_max_smd=statistics.median(smds) if smds else None

    gate=(
        len(pair_summary)==27 and
        len(errors)==0 and
        overall_frac>=0.80 and
        axes["lon"]["usable_fraction"]>=0.70 and
        axes["lat"]["usable_fraction"]>=0.70 and
        median_cov>=0.25 and
        median_max_smd is not None and median_max_smd<=0.20
    )

    with open(os.path.join(a.out,"P1_A3A_PAIR_OVERLAP_SUMMARY.csv"),"w",newline="",encoding="utf-8-sig") as fh:
        fields=["file_name","axis","clean_eligible_rows","iced_eligible_rows","candidate_edges","matched_rows",
                "coverage_min_side","active_feature_count","max_post_match_abs_smd","usable_pair"]
        w=csv.DictWriter(fh,fieldnames=fields);w.writeheader();w.writerows(pair_summary)

    all_fields=set()
    for r in match_rows:all_fields.update(r.keys())
    base=["file_name","axis","match_rank","clean_sample_index","iced_sample_index","standardized_rms_distance","max_component_abs_z"]
    extras=sorted(k for k in all_fields if k not in base)
    with open(os.path.join(a.out,"P1_A3A_MATCHING_MAP.csv"),"w",newline="",encoding="utf-8-sig") as fh:
        w=csv.DictWriter(fh,fieldnames=base+extras);w.writeheader();w.writerows(match_rows)

    with open(os.path.join(a.out,"P1_A3A_FEATURE_BALANCE.csv"),"w",newline="",encoding="utf-8-sig") as fh:
        fields=["file_name","axis","feature","scale","post_match_abs_smd","used_in_matching"]
        w=csv.DictWriter(fh,fieldnames=fields);w.writeheader();w.writerows(feature_rows)

    with open(os.path.join(a.out,"P1_A3A_PARSE_ERRORS.csv"),"w",newline="",encoding="utf-8-sig") as fh:
        w=csv.DictWriter(fh,fieldnames=["file_name","error"]);w.writeheader();w.writerows(errors)

    result={
        "scope":{
            "exact_training_pairs":len(pair_summary),
            "validation_files_used":0,
            "outcome_columns_used_in_matching":0
        },
        "frozen_matching_contract":{
            "longitudinal_features":["Va","alpha","q_hat","elevator","throttle"],
            "lateral_features":["Va","alpha","beta","p_hat","r_hat","aileron","throttle"],
            "standardization":"within-file-pair pooled robust scale; zero-variance features dropped",
            "algorithm":"greedy one-to-one minimum standardized-distance matching without replacement",
            "rms_caliper":RMS_CALIPER,
            "max_component_abs_z":MAX_COMPONENT_Z,
            "pair_usable_min_matches":MIN_MATCHES,
            "pair_usable_min_coverage":MIN_COVERAGE,
            "pair_usable_max_abs_smd":MAX_POST_SMD
        },
        "global_gate":{
            "minimum_overall_usable_fraction":0.80,
            "minimum_axis_usable_fraction":0.70,
            "minimum_median_coverage":0.25,
            "maximum_median_pair_max_abs_smd":0.20,
            "overall_usable_pairs":len(usable),
            "overall_usable_fraction":overall_frac,
            "longitudinal":axes["lon"],
            "lateral":axes["lat"],
            "median_coverage_usable_pairs":median_cov,
            "median_pair_max_abs_smd":median_max_smd,
            "pass":gate
        },
        "errors":len(errors)
    }

    with open(os.path.join(a.out,"P1_A3A_OVERLAP_AUDIT.json"),"w",encoding="utf-8") as fh:
        json.dump(result,fh,indent=2)

    print("P1_A3A_OVERLAP_SUMMARY")
    print("exact_training_pairs       =",len(pair_summary))
    print("validation_files_used      = 0")
    print("outcome_columns_used       = 0")
    print("usable_pairs               =",len(usable),"/",len(pair_summary))
    print("overall_usable_fraction    =",overall_frac)
    print("lon usable                 =",axes["lon"]["usable_pairs"],"/",axes["lon"]["pairs"])
    print("lat usable                 =",axes["lat"]["usable_pairs"],"/",axes["lat"]["pairs"])
    print("median_coverage             =",median_cov)
    print("median_pair_max_abs_smd     =",median_max_smd)
    print("parse_errors                =",len(errors))
    print("global_overlap_gate         =",gate)
    print("")
    print("PAIR SUMMARY")
    for r in pair_summary:
        print("  {0:28s} {1:3s} matches={2:4d} cov={3:.3f} maxSMD={4:.3f} usable={5}".format(
            r["file_name"][:28],r["axis"],r["matched_rows"],r["coverage_min_side"],
            r["max_post_match_abs_smd"] if r["max_post_match_abs_smd"] is not None else float("nan"),
            r["usable_pair"]
        ))

if __name__=="__main__":
    main()
