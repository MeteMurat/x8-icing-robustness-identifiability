import argparse,csv,json,os,re

def load_json(path):
    with open(path,"r",encoding="utf-8-sig") as f:
        return json.load(f)

def first_header(path):
    # Header only. No numeric data rows are read.
    with open(path,"r",newline="",encoding="utf-8-sig") as f:
        line=f.readline()
    return next(csv.reader([line]))

def matches(keys,patterns):
    out=[]
    for k in keys:
        lk=k.lower()
        if any(re.search(p,lk) for p in patterns):
            out.append(k)
    return sorted(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--clean-dir",required=True)
    ap.add_argument("--iced-dir",required=True)
    ap.add_argument("--clean-json",required=True)
    ap.add_argument("--iced-json",required=True)
    ap.add_argument("--readme",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    files=[]
    for cfg,folder in (("clean",a.clean_dir),("iced",a.iced_dir)):
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(".csv"):
                files.append((cfg,name,os.path.join(folder,name)))

    headers={}
    header_rows=[]
    all_sets=[]

    candidates=[
        "battery_power",
        "RPS",
        "Va_EKF_CG",
        "alpha",
        "beta",
        "elevator",
        "aileron"
    ]

    for cfg,name,path in files:
        h=first_header(path)
        hs=set(h)
        headers[(cfg,name)]=h
        all_sets.append(hs)

        row={
            "configuration":cfg,
            "file_name":name,
            "n_headers":len(h)
        }

        for c in candidates:
            row["has_"+c]=c in hs

        # Candidate electrical channels are discovered, not assumed.
        row["voltage_like"]=";".join(sorted(
            x for x in h
            if re.search(r"volt|v_batt|battery_v|vbat",x,re.I)
        ))

        row["current_like"]=";".join(sorted(
            x for x in h
            if re.search(r"curr|amp|i_batt|battery_i|ibat",x,re.I)
        ))

        row["power_like"]=";".join(sorted(
            x for x in h
            if re.search(r"power|watt",x,re.I)
        ))

        row["torque_like"]=";".join(sorted(
            x for x in h
            if re.search(r"torque|prop_q|q_prop",x,re.I)
        ))

        row["thrust_like"]=";".join(sorted(
            x for x in h
            if re.search(r"thrust",x,re.I)
        ))

        header_rows.append(row)

    common=set.intersection(*all_sets) if all_sets else set()
    union=set.union(*all_sets) if all_sets else set()

    clean=load_json(a.clean_json)
    iced=load_json(a.iced_json)
    pkeys=sorted(set(clean.keys()).union(iced.keys()))

    param_groups={
        "propulsion_efficiency_like":matches(
            pkeys,[r"eta_prop",r"prop.*eff",r"eff.*prop"]
        ),
        "motor_efficiency_like":matches(
            pkeys,[r"eta_motor",r"motor.*eff",r"eff.*motor"]
        ),
        "esc_efficiency_like":matches(
            pkeys,[r"eta_esc",r"esc.*eff",r"eff.*esc"]
        ),
        "battery_capacity_or_energy_like":matches(
            pkeys,[r"capacity",r"battery.*energy",r"energy.*battery",r"wh\b",r"ah\b"]
        ),
        "motor_or_electrical_model_like":matches(
            pkeys,[r"motor",r"esc",r"electr",r"voltage",r"current",r"battery"]
        ),
        "propeller_torque_model_like":matches(
            pkeys,[r"c_q",r"torque",r"eta_prop_q"]
        ),
        "propeller_thrust_model_like":matches(
            pkeys,[r"c_t",r"thrust",r"eta_prop_t"]
        )
    }

    with open(a.readme,"r",encoding="utf-8-sig",errors="replace") as f:
        readme_lines=f.readlines()

    pat=re.compile(
        r"battery_power|battery|electrical|electric|voltage|current|"
        r"\bRPS\b|propeller|propulsion|thrust|torque|power|capacity|energy|"
        r"endurance|range",
        re.I
    )

    excerpts=[]
    for i,line in enumerate(readme_lines,1):
        if pat.search(line):
            excerpts.append({
                "line_number":i,
                "text":line.rstrip("\r\n")
            })

    battery_all=all("battery_power" in s for s in all_sets)
    rps_all=all("RPS" in s for s in all_sets)

    voltage_names=sorted(
        x for x in union
        if re.search(r"volt|v_batt|battery_v|vbat",x,re.I)
    )

    current_names=sorted(
        x for x in union
        if re.search(r"curr|amp|i_batt|battery_i|ibat",x,re.I)
    )

    power_names=sorted(
        x for x in union
        if re.search(r"power|watt",x,re.I)
    )

    torque_names=sorted(
        x for x in union
        if re.search(r"torque|prop_q|q_prop",x,re.I)
    )

    direct_battery_candidate=bool(battery_all)

    direct_shaft_power_identifiable=bool(
        rps_all and len(torque_names)>0
    )

    motor_eta_available=bool(param_groups["motor_efficiency_like"])
    esc_eta_available=bool(param_groups["esc_efficiency_like"])
    battery_energy_available=bool(param_groups["battery_capacity_or_energy_like"])

    # Scope classification is intentionally conservative.
    if direct_battery_candidate:
        electrical_scope="EMPIRICAL_BATTERY_POWER_CHANNEL_CANDIDATE_REQUIRES_SEMANTIC_FREEZE"
    else:
        electrical_scope="NO_COMMON_BATTERY_POWER_CHANNEL"

    if direct_shaft_power_identifiable:
        shaft_scope="CANDIDATE_DIRECT_OR_SOURCE_DERIVED_REQUIRES_SEMANTIC_AUDIT"
    elif param_groups["propeller_torque_model_like"] and rps_all:
        shaft_scope="MODEL_DEPENDENT_PROP_TORQUE_TIMES_OMEGA_ONLY"
    else:
        shaft_scope="NOT_DIRECTLY_IDENTIFIABLE"

    if battery_energy_available:
        endurance_scope="STILL_NOT_AUTHORIZED_REQUIRES_STEADY_POWER_AND_USABLE_ENERGY_SEMANTICS"
    else:
        endurance_scope="NOT_IDENTIFIABLE_FROM_CURRENT_METADATA_NO_USABLE_BATTERY_ENERGY_MODEL"

    range_scope="NOT_AUTHORIZED_REQUIRES_VALIDATED_STEADY_MISSION_POWER_SPEED_AND_ENERGY_MODEL"

    audit={
        "files":{
            "csv_count":len(files),
            "all_headers_identical":all(s==all_sets[0] for s in all_sets) if all_sets else False,
            "common_header_count":len(common),
            "union_header_count":len(union)
        },

        "channel_availability":{
            "battery_power_present_all_csv":battery_all,
            "RPS_present_all_csv":rps_all,
            "voltage_like_channels":voltage_names,
            "current_like_channels":current_names,
            "power_like_channels":power_names,
            "torque_like_channels":torque_names
        },

        "parameter_key_groups":param_groups,

        "identifiability":{
            "aerodynamic_drag_power":"AUTHORIZED_CLOSED_PARENT_P1_A6",
            "reference_state_aerodynamic_efficiency":"AUTHORIZED_CLOSED_PARENT_P1_A7",
            "empirical_battery_power":electrical_scope,
            "shaft_power":shaft_scope,
            "motor_efficiency_parameters_available":motor_eta_available,
            "esc_efficiency_parameters_available":esc_eta_available,
            "battery_capacity_or_energy_parameters_available":battery_energy_available,
            "endurance":endurance_scope,
            "range":range_scope
        },

        "rules":{
            "battery_power_not_assumed_to_equal_aerodynamic_drag_power":True,
            "battery_power_not_assumed_to_equal_shaft_power":True,
            "propeller_eta_not_assumed_to_equal_motor_or_ESC_efficiency":True,
            "endurance_not_computable_from_reference_LD_alone":True,
            "range_not_computable_from_reference_LD_alone":True,
            "numeric_flight_rows_read":0
        }
    }

    with open(
        os.path.join(a.out,"P1_A8A_METADATA_IDENTIFIABILITY_AUDIT.json"),
        "w",encoding="utf-8"
    ) as f:
        json.dump(audit,f,indent=2)

    fields=list(header_rows[0].keys())
    with open(
        os.path.join(a.out,"P1_A8A_HEADER_AVAILABILITY.csv"),
        "w",newline="",encoding="utf-8-sig"
    ) as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(header_rows)

    with open(
        os.path.join(a.out,"P1_A8A_README_POWER_EXCERPTS.csv"),
        "w",newline="",encoding="utf-8-sig"
    ) as f:
        w=csv.DictWriter(f,fieldnames=["line_number","text"])
        w.writeheader()
        w.writerows(excerpts)

    pg_rows=[]
    for group,keys in param_groups.items():
        if keys:
            for key in keys:
                pg_rows.append({"group":group,"parameter_key":key})
        else:
            pg_rows.append({"group":group,"parameter_key":""})

    with open(
        os.path.join(a.out,"P1_A8A_PARAMETER_KEY_AUDIT.csv"),
        "w",newline="",encoding="utf-8-sig"
    ) as f:
        w=csv.DictWriter(f,fieldnames=["group","parameter_key"])
        w.writeheader()
        w.writerows(pg_rows)

    print("P1_A8A_MISSION_PERFORMANCE_IDENTIFIABILITY_SUMMARY")
    print("csv_count                                   =",len(files))
    print("battery_power_present_all_csv               =",battery_all)
    print("RPS_present_all_csv                         =",rps_all)
    print("voltage_like_channels                       =",voltage_names)
    print("current_like_channels                       =",current_names)
    print("power_like_channels                         =",power_names)
    print("torque_like_channels                        =",torque_names)
    print("motor_efficiency_parameters_available       =",motor_eta_available)
    print("esc_efficiency_parameters_available         =",esc_eta_available)
    print("battery_capacity_or_energy_params_available =",battery_energy_available)
    print("")
    print("IDENTIFIABILITY CLASSIFICATION")
    print("  aerodynamic drag power   = AUTHORIZED_CLOSED_PARENT_P1_A6")
    print("  empirical battery power  =",electrical_scope)
    print("  shaft power              =",shaft_scope)
    print("  endurance                =",endurance_scope)
    print("  range                    =",range_scope)
    print("")
    print("numeric_flight_rows_read =",0)

if __name__=="__main__":
    main()
