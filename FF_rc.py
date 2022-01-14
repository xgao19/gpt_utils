#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020 / Philipp Scior 2021
#
# Calculate pion DA with A2A method
#
import gpt as g
import sys, os
import numpy as np

# configure
# root_output = "/p/project/chbi21/gpt_test/DA"
root_output = "/home/scior/gpt_cpu/tests/qcd"

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

zmax = 6
Gamma = g.gamma["T"]
t_insert = 4
mom = [0,0,0,0]
width = 1.0
pos_boost = [0,0,0]
neg_boost = [0,0,0]


jobs = {
    "booster_exact_0": {
        "exact": 1,
        "sloppy": 0,
        "low": 0,
        "all_time_slices": True,
    },  # 1270 seconds + 660 to load ev
    "booster_sloppy_0": {
        "exact": 0,
        "sloppy": 8,
        "low": 0,
        "all_time_slices": True,
    },  # 2652 seconds + 580 to load ev
    "booster_sloppy_1": {"exact": 0, "sloppy": 32, "low": 0, "all_time_slices": True},
    "booster_sloppy_2": {"exact": 0, "sloppy": 32, "low": 0, "all_time_slices": True},
    "booster_sloppy_3": {"exact": 0, "sloppy": 32, "low": 0, "all_time_slices": True},
    "booster_sloppy_4": {"exact": 0, "sloppy": 32, "low": 0, "all_time_slices": True},
    "booster_sloppy_5": {"exact": 0, "sloppy": 32, "low": 0, "all_time_slices": True},
    "booster_sloppy_6": {"exact": 0, "sloppy": 32, "low": 0, "all_time_slices": True},
    "booster_sloppy_7": {"exact": 0, "sloppy": 32, "low": 0, "all_time_slices": True},
    "booster_low_0": {
        "exact": 0,
        "sloppy": 0,
        "low": 150,
        "all_time_slices": True,
    },  # 2100 seconds + 600 to load ev
    "booster_low_1": {"exact": 0, "sloppy": 0, "low": 600, "all_time_slices": True},
    "booster_low_2": {"exact": 0, "sloppy": 0, "low": 600, "all_time_slices": True},
    "booster_low_3": {"exact": 0, "sloppy": 0, "low": 600, "all_time_slices": True},
    "booster_low_4": {"exact": 0, "sloppy": 0, "low": 600, "all_time_slices": True},
    "booster_low_5": {"exact": 0, "sloppy": 0, "low": 600, "all_time_slices": True},
    "booster_low_6": {"exact": 0, "sloppy": 0, "low": 600, "all_time_slices": True},
    "booster_low_7": {"exact": 0, "sloppy": 0, "low": 600, "all_time_slices": True},
    "booster_exact_0_correlated": {
        "exact": 1,
        "sloppy": 0,
        "low": 0,
        "all_time_slices": False,
    },  # 1270 seconds + 660 to load ev
    "booster_sloppy_0_correlated": {
        "exact": 0,
        "sloppy": 8,
        "low": 0,
        "all_time_slices": False,
    },  # 2652 seconds + 580 to load ev
    "booster_sloppy_1_correlated": {
        "exact": 0,
        "sloppy": 32,
        "low": 0,
        "all_time_slices": False,
    },
    "booster_sloppy_2_correlated": {
        "exact": 0,
        "sloppy": 32,
        "low": 0,
        "all_time_slices": False,
    },
    "booster_low_0_correlated": {
        "exact": 0,
        "sloppy": 0,
        "low": 150,
        "all_time_slices": False,
    },  # 2100 seconds + 600 to load ev
    "booster_low_1_correlated": {
        "exact": 0,
        "sloppy": 0,
        "low": 600,
        "all_time_slices": False,
    },
    "booster_low_2_correlated": {
        "exact": 0,
        "sloppy": 0,
        "low": 600,
        "all_time_slices": False,
    },
}

# At 32 jobs we break even with eigenvector generation

simultaneous_low_positions = 2

jobs_per_run = g.default.get_int("--gpt_jobs", 1)

source_time_slices = 2

save_propagators = False

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
# g.message("finished loading gauge config")


# do gauge fixing

U_prime, trafo = g.gauge_fix(U, maxiter=500)
del U_prime
#trafo = rng.element(g.lattice(U[0]))

L = U[0].grid.fdimensions

l_exact = g.qcd.fermion.mobius(
    U,
    {
        "mass": 0.00054,
        "M5": 1.8,
        "b": 1.5,
        "c": 0.5,
        "Ls": 12,
        "boundary_phases": [1.0, 1.0, 1.0, -1.0],},
)

g.message("after Mobius def")

l_sloppy = l_exact.converted(g.single)

###These are all Dummy inverters so I can do tests without deflation!
light_innerL_inverter = g.algorithms.inverter.preconditioned(g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd), g.algorithms.inverter.cg(eps = 1e-8, maxiter = 200))
light_innerH_inverter = g.algorithms.inverter.preconditioned(g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd), g.algorithms.inverter.cg(eps = 1e-8, maxiter = 300))
## not sure if we are going to include measurements on low modes only?
# light_low_inverter = g.algorithms.inverter.preconditioned(g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd), g.algorithms.inverter.cg(eps = 1e-8, maxiter = 200))

#############Here is the stuff for the inverters used in the production run!!

# eig = g.load(groups[group]["evec_fmt"], grids=l_sloppy.F_grid_eo)
# ## pin coarse eigenvectors to GPU memory
# pin = g.pin(eig[1], g.accelerator)


# light_innerL_inverter = g.algorithms.inverter.preconditioned(
#     g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd),
#     g.algorithms.inverter.sequence(
#         g.algorithms.inverter.coarse_deflate(
#             eig[1],
#             eig[0],
#             eig[2],
#             block=400,
#             fine_block=4,
#             linear_combination_block=32,
#         ),
#         g.algorithms.inverter.split(
#             g.algorithms.inverter.cg({"eps": 1e-8, "maxiter": 200}),
#             mpi_split=g.default.get_ivec("--mpi_split", None, 4),
#         ),
#     ),
# )

# light_innerH_inverter = g.algorithms.inverter.preconditioned(
#     g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd),
#     g.algorithms.inverter.sequence(
#         g.algorithms.inverter.coarse_deflate(
#             eig[1],
#             eig[0],
#             eig[2],
#             block=400,
#             fine_block=4,
#             linear_combination_block=32,
#         ),
#         g.algorithms.inverter.split(
#             g.algorithms.inverter.cg({"eps": 1e-8, "maxiter": 300}),
#             mpi_split=g.default.get_ivec("--mpi_split", None, 4),
#         ),
#     ),
# )

# light_low_inverter = g.algorithms.inverter.preconditioned(
#     g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd),
#     g.algorithms.inverter.coarse_deflate(
#         eig[1], eig[0], eig[2], block=400, linear_combination_block=32, fine_block=4
#     ),
# )

light_exact_inverter = g.algorithms.inverter.defect_correcting(
    g.algorithms.inverter.mixed_precision(light_innerH_inverter, g.single, g.double),
    eps=1e-8,
    maxiter=10,
)

light_sloppy_inverter = g.algorithms.inverter.defect_correcting(
    g.algorithms.inverter.mixed_precision(light_innerL_inverter, g.single, g.double),
    eps=1e-8,
    maxiter=2,
)


############### inverter definitions finished

# prop_l_low = l_sloppy.propagator(light_low_inverter)
prop_l_sloppy = l_exact.propagator(light_sloppy_inverter).grouped(6)
prop_l_exact = l_exact.propagator(light_exact_inverter).grouped(6)


# show available memory
g.mem_report(details=False)
g.message(
    """
================================================================================
       FF run on booster ;  this run will attempt:
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
    rng = g.random(f"DA-ensemble-{conf}-{job_seed}")

    # source_positions_low = [
    #     [rng.uniform_int(min=0, max=L[i] - 1) for i in range(4)]
    #     for j in range(jobs[job]["low"])
    # ]
    source_positions_sloppy = [
        [rng.uniform_int(min=0, max=L[i] - 1) for i in range(4)]
        for j in range(jobs[job]["sloppy"])
    ]
    source_positions_exact = [
        [rng.uniform_int(min=0, max=L[i] - 1) for i in range(4)]
        for j in range(jobs[job]["exact"])
    ]

    all_time_slices = jobs[job]["all_time_slices"]
    use_source_time_slices = source_time_slices
    if not all_time_slices:
        use_source_time_slices = 1

    # g.message(f" positions_low = {source_positions_low}")
    g.message(f" positions_sloppy = {source_positions_sloppy}")
    g.message(f" positions_exact = {source_positions_exact}")
    # g.message(f" all_time_slices = {all_time_slices}")

    root_job = f"{root_output}/{conf}/{job}"
    output = g.gpt_io.writer(f"{root_job}/propagators")
    output_correlator = g.corr_io.writer(f"{root_job}/correlators.dat")

    # make momentum operator
    p = -2 * np.pi * np.array(mom) / L
    P = g.exp_ixp(p)

    G_op = g.gamma[5] * g.adj(P) 
    #the g.adj(P) is due to the fact that the Fourier factor is changed by daggering when using gamma5 hermiticity
    

    # create Wilson line
    def create_WL(z):
        W = []
        W.append(g.qcd.gauge.unit(U[2].grid, Nd=1)[0])
        for dz in range(0, z):
            W.append(g.eval(g.cshift(U[2], 2, dz) * W[dz-1]))
                
        return W


    def contract_2pt(pos, prop_f, prop_b, tag):
        prop_tag = "%s/%s" % (tag, str(pos))
        g.message("Begin sink smearing")
        tmp_trafo = g.convert(trafo, prop_f.grid.precision)

        prop_f = g.create.smear.boosted_smearing(tmp_trafo, prop_f, w=width, boost=pos_boost)
        prop_f = g.create.smear.boosted_smearing(tmp_trafo, prop_b, w=width, boost=neg_boost)
        g.message("Sink smearing completed")

        corr = g.slice(
            g.trace(g.adj(P) * prop_f * P * g.adj(prop_b) ), 3
        ) 

        corr_tag = "%s/2pt" % (prop_tag)
        output_correlator.write(corr_tag, corr)
        g.message("2pt Correlator %s\n" % corr_tag, corr)

    def contract_3pt(pos, prop_f, prop_b, tag):
        threept_tag = "%s/%s" % (tag, str(pos))


        corr = g.slice(
            g.trace(g.adj(prop_b)*Gamma*g.adj(P)*prop_f), 3
        )

        corr_tag = "%s/3pt" % (threept_tag)
        output_correlator.write(corr_tag, corr)
        g.message("3pt Correlator %s\n" % corr_tag, corr)


    def create_src_3pt(pos):
        
        srcD = g.mspincolor(l_exact.U_grid)
        srcD[:] = 0

        g.create.point(srcD, pos)

        return srcD

    def create_bw_seq(src, exact_flag):

        tmp_trafo = g.convert(trafo, src.grid.precision)

        
        if(exact_flag):
            prop_sp = g.eval(prop_l_exact * g.adj(P) * src)
        else:
            prop_sp = g.eval(prop_l_sloppy * g.adj(P) * src)
        #the adj(P) is due to the fact that the Fourier factor is changed by daggering when using gamma5 hermiticity
        prop_sp = g.create.smear.boosted_smearing(tmp_trafo, prop_sp, w=width, boost=neg_boost)

        # sequential solve through t=insertion_time
        t_op = t_insert
        src_seq = g.lattice(src)
        src_seq[:] = 0
        src_seq[:, :, :, t_op] = prop_sp[:, :, :, t_op]

        # create seq prop using gamma5 hermiticity
        dst_seq = g.lattice(src_seq)
        src_seq @= G_op * src_seq

        del prop_sp

        dst_seq = g.create.smear.boosted_smearing(tmp_trafo, src_seq, w=width, boost=neg_boost)
        if(exact_flag):
            dst_seq @= prop_l_exact * src_seq
        else:
            dst_seq @= prop_l_sloppy * src_seq

        dst_seq = g.create.smear.boosted_smearing(tmp_trafo, dst_seq, w=width, boost=neg_boost)

        dst_seq @= g.gamma[5] * dst_seq

        return dst_seq


    def create_src_2pt(pos):
        
        srcD = g.mspincolor(l_exact.U_grid)
        srcD[:] = 0
        
        srcDp = g.mspincolor(l_exact.U_grid)
        srcDp[:] = 0

        srcDm = g.mspincolor(l_exact.U_grid)
        srcDm[:] = 0

        g.create.point(srcD, pos)


        srcDm = g.create.smear.boosted_smearing(trafo, srcD, w=width, boost=neg_boost)
        srcDp = g.create.smear.boosted_smearing(trafo, srcD, w=width, boost=pos_boost)

        del srcD

        return srcDp, srcDm


    # g.message("Starting Wilson loops")
    # W = create_WL(zmax)

    # exact positions
    for pos in source_positions_exact:

        g.message("STARTING EXACT MEASUREMENTS")

        g.message("Starting 2pt function")

        g.message("Generatring boosted src's")
        srcDp, srcDm = create_src_2pt(pos)  

        g.message("Starting prop exact")
        prop_exact_f = g.eval(prop_l_exact * srcDp)
        g.message("forward prop done")
        prop_exact_b = g.eval(prop_l_exact * srcDm)
        g.message("backward prop done")

        g.message("Starting pion contraction (includes sink smearing)")
        contract_2pt(pos, prop_exact_f, prop_exact_b, "exact")
        g.message("pion contraction done")

        del prop_exact_b

        g.message("Starting 3pt function")

        g.message("Generatring point src")
        srcD = create_src_3pt(pos)  

        g.message("Starting bw seq propagator")

        prop_exact_b = create_bw_seq(srcD, True)

        g.message("Starting 3pt contractions")
        contract_3pt(pos, prop_exact_f, prop_exact_b, "exact")
        g.message("3pt contractions done")
       

        del prop_exact_f
        del prop_exact_b
 


        g.message("STARTING SLOPPY MEASUREMENTS")

        g.message("Starting 2pt function")

        # g.message("Generatring boosted src's")
        # srcDp, srcDm = create_src_2pt(pos)  

        g.message("Starting prop sloppy")
        prop_sloppy_f = g.eval(prop_l_sloppy * srcDp)
        g.message("forward prop done")
        prop_sloppy_b = g.eval(prop_l_sloppy * srcDm)
        g.message("backward prop done")

        del srcDp
        del srcDm

        g.message("Starting pion contraction (includes sink smearing)")
        contract_2pt(pos, prop_sloppy_f, prop_sloppy_b, "sloppy")
        g.message("pion contraction done")

        del prop_sloppy_b

        g.message("Starting 3pt function")

        # g.message("Generatring point src")
        # srcD = create_src_3pt(pos)  

        g.message("Starting bw seq propagators")

        prop_sloppy_b = create_bw_seq(srcD, False)

        del srcD

        g.message("Starting 3pt contractions")
        contract_3pt(pos, prop_sloppy_f, prop_sloppy_b, "sloppy")
        g.message("3pt contractions done")

        del prop_sloppy_f
        del prop_sloppy_b
     
    g.message("exact positions done")

    # sloppy positions
    for pos in source_positions_sloppy:

        g.message("STARTING SLOPPY MEASUREMENTS")

        g.message("Starting 2pt function")

        g.message("Generating boosted src's")
        srcDp, srcDm = create_src_2pt(pos)  

        g.message("Starting prop sloppy")
        prop_sloppy_f = g.eval(prop_l_sloppy * srcDp)
        g.message("forward prop done")
        prop_sloppy_b = g.eval(prop_l_sloppy * srcDm)
        g.message("backward prop done")

        del srcDp
        del srcDm

        g.message("Starting pion contraction (includes sink smearing)")
        contract_2pt(pos, prop_sloppy_f, prop_sloppy_b, "sloppy")
        g.message("pion contraction done")

        del prop_sloppy_b

        g.message("Starting 3pt function")

        g.message("Generatring point src")
        srcD = create_src_3pt(pos)  

        g.message("Starting bw seq propagators")

        prop_sloppy_b = create_bw_seq(srcD, False)

        del srcD

        g.message("Starting 3pt contractions")
        contract_3pt(pos, prop_sloppy_f, prop_sloppy_b, "sloppy")
        g.message("3pt contractions done")

        del prop_sloppy_f
        del prop_sloppy_b
    
    g.message("sloppy positions done")
        
#del pin

