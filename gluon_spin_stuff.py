#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020 / Philipp Scior 2021
#
# Calculate proton QPDF with A2A method
#
import gpt as g
import os
from gpt_qpdf_utils import proton_qpdf_measurement

# configure
# root_output = "/p/project/chbi21/gpt_test/DA"
root_output ="."

# 420, 500, 580
groups = {
    "booster_batch_0": {
        "confs": [
            "420",
            "1960",
            "2000",
        ],
        #"evec_fmt": "/p/scratch/gm2dwf/evecs/96I/%s/lanczos.output",
	    "evec_fmt": "/p/project/chbi21/gpt_test/96I/lanczos.output",
        "conf_fmt": "/p/project/chbi21/gpt_test/96I/ckpoint_lat.%s",
    },

}
parameters = {
    "pf" : [1,1,0,0],
    "q" : [0,1,0,0],
    "zmax" : 4,
    "t_insert" : 4,
    "width" : 2.2,
    "boost_in" : [0,0,0],
    "save_propagators" : True
}


jobs = {
    "booster_exact_0": {
        "exact": 1,
        "sloppy": 0,
        "low": 0,
    },  
    "booster_sloppy_0": {
        "exact": 0,
        "sloppy": 10,
        "low": 0,
    }, 
}


jobs_per_run = g.default.get_int("--gpt_jobs", 1)



# find jobs for this run
def get_job(only_on_conf=None):
    # statistics
    n = 0
    for group in groups:
        for job in jobs:
            for conf in groups[group]["confs"]:
                n += 1

    jid = -1
    # for job in jobs:
    #    for group in groups:
    #        for conf in groups[group]["confs"]:
    for group in groups:
        for conf in groups[group]["confs"]:
            for job in jobs:
                jid += 1
                if only_on_conf is not None and only_on_conf != conf:
                    continue
                root_job = f"{root_output}/{conf}/{job}"
                if not os.path.exists(root_job):
                    os.makedirs(root_job)
                    return group, job, conf, jid, n

    return None


if g.rank() == 0:
    first_job = get_job()
    run_jobs = str(
        list(
            filter(
                lambda x: x is not None,
                [first_job] + [get_job(first_job[2]) for i in range(1, jobs_per_run)],
            )
        )
    ).encode("utf-8")
else:
    run_jobs = bytes()
run_jobs = eval(g.broadcast(0, run_jobs).decode("utf-8"))

# every node now knows what to do



# configuration needs to be the same for all jobs, so load eigenvectors and configuration
conf = run_jobs[0][2]
group = run_jobs[0][0]


##### small dummy used for testing
grid = g.grid([8,8,8,8], g.double)
rng = g.random("seed text")
U = g.qcd.gauge.random(grid, rng)

# loading gauge configuration
# U = g.load(groups[group]["conf_fmt"] % conf)
g.message("finished loading gauge config")


# do gauge fixing

U_prime, trafo = g.gauge_fix(U, maxiter=500)
del trafo

L = U[0].grid.fdimensions

Measurement = proton_qpdf_measurement(parameters)


# show available memory
g.mem_report(details=False)
g.message(
    """
================================================================================
       Gluon Helicity run on booster ;  this run will attempt:
================================================================================
"""
)
# per job
for group, job, conf, jid, n in run_jobs:
    g.message(
        f"""

    Job {jid} / {n} :  configuration {conf}, job tag {job}

"""
    )

    job_seed = job.split("_correlated")[0]
    rng = g.random(f"QPDF-ensemble-{conf}-{job_seed}")

    source_positions_sloppy = [
        [rng.uniform_int(min=0, max=L[i] - 1) for i in range(4)]
        for j in range(jobs[job]["sloppy"])
    ]
    source_positions_exact = [
        [rng.uniform_int(min=0, max=L[i] - 1) for i in range(4)]
        for j in range(jobs[job]["exact"])
    ]



    g.message(f" positions_sloppy = {source_positions_sloppy}")
    g.message(f" positions_exact = {source_positions_exact}")

    root_job = f"{root_output}/{conf}/{job}"

    Measurement.set_output_facilites(f"{root_job}/correlators",f"{root_job}/propagators")

    Ax = U_prime[0] - g.adj(U_prime[0]) + g.cshift(U_prime[0],0,-1) + g.adj(g.cshift(U_prime[0],0,-1))
    Ax -= g.identity(Ax) * g.trace(Ax) / 3

    Ay = U_prime[1] - g.adj(U_prime[1]) + g.cshift(U_prime[1],1,-1) + g.adj(g.cshift(U_prime[1],1,-1))
    Ay -= g.identity(Ay) * g.trace(Ay) / 3

    Az = U_prime[2] - g.adj(U_prime[2]) + g.cshift(U_prime[2],2,-1) + g.adj(g.cshift(U_prime[2],2,-1))
    Az -= g.identity(Az) * g.trace(Az) / 3

    At = U_prime[3] - g.adj(U_prime[3]) + g.cshift(U_prime[3],3,-1) + g.adj(g.cshift(U_prime[3],3,-1))
    At -= g.identity(At) * g.trace(At) / 3


    #O321 is the naming convention from Minkowski space. In the Euklidean space version of Grid t=3,
    # so Mikowski O321 = Euklidean O210!
    O321 = g.slice(g.trace(g.field_strength(U_prime, 2, 1)* Ax - g.field_strength(U_prime, 2, 0)* Ay ) , 3)
    O021 = g.slice(g.trace(g.field_strength(U_prime, 3, 1)* Ax - g.field_strength(U_prime, 3, 0)* Ay ) , 3)
    B_dot_A = O321 + g.slice(g.trace(g.field_strength(U_prime, 1, 0) * Az) , 3)


    #do output