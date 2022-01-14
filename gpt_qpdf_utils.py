import gpt as g
import sys, os
import numpy as np

#ordered list of gamma matrix identifiers, needed for the tag in the correlator output
my_gammas = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]

#make the inverters needed for the 96I lattices
def make_96I_inverter(U, evec_file):

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

    l_sloppy = l_exact.converted(g.single)

    eig = g.load(evec_file, grids=l_sloppy.F_grid_eo)
    # ## pin coarse eigenvectors to GPU memory
    pin = g.pin(eig[1], g.accelerator)


    light_innerL_inverter = g.algorithms.inverter.preconditioned(
        g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd),
        g.algorithms.inverter.sequence(
            g.algorithms.inverter.coarse_deflate(
                eig[1],
                eig[0],
                eig[2],
                block=400,
                fine_block=4,
                linear_combination_block=32,
            ),
            g.algorithms.inverter.split(
                g.algorithms.inverter.cg({"eps": 1e-8, "maxiter": 200}),
                mpi_split=g.default.get_ivec("--mpi_split", None, 4),
            ),
        ),
    )

    light_innerH_inverter = g.algorithms.inverter.preconditioned(
        g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd),
        g.algorithms.inverter.sequence(
            g.algorithms.inverter.coarse_deflate(
                eig[1],
                eig[0],
                eig[2],
                block=400,
                fine_block=4,
                linear_combination_block=32,
            ),
            g.algorithms.inverter.split(
                g.algorithms.inverter.cg({"eps": 1e-8, "maxiter": 300}),
                mpi_split=g.default.get_ivec("--mpi_split", None, 4),
            ),
        ),
    )

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


    ############### final inverter definitions
    prop_l_sloppy = l_exact.propagator(light_sloppy_inverter).grouped(6)
    prop_l_exact = l_exact.propagator(light_exact_inverter).grouped(6)

    return prop_l_exact, prop_l_sloppy, pin

def make_debugging_inverter(U):

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

    l_sloppy = l_exact.converted(g.single)

    light_innerL_inverter = g.algorithms.inverter.preconditioned(g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd), g.algorithms.inverter.cg(eps = 1e-8, maxiter = 200))
    light_innerH_inverter = g.algorithms.inverter.preconditioned(g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd), g.algorithms.inverter.cg(eps = 1e-8, maxiter = 300))

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

    return prop_l_exact, prop_l_sloppy




############### inverter definitions finished

# prop_l_low = l_sloppy.propagator(light_low_inverter)
prop_l_sloppy = l_exact.propagator(light_sloppy_inverter).grouped(6)
prop_l_exact = l_exact.propagator(light_exact_inverter).grouped(6)


############## make list of complex phases for momentum proj.
def make_mom_phases(grid, L, plist):    
    one = g.complex(grid)
    p = [-2 * np.pi * np.array([0, 0, pz, 0]) / L for pz in plist]
    P = g.exp_ixp(p)
    mom = [g.eval(pp*one) for pp in P]
    return mom

# create Wilson lines
def create_WL(U,z):
    W = []
    W.append(g.qcd.gauge.unit(U[2].grid)[0])
    for dz in range(0, z):
        W.append(g.eval(g.cshift(U[2], 2, dz) * W[dz-1]))
            
    return W

# Creating list of W*prop_b for all z
def constr_backw_prop_for_DA(prop_b, W, zmax):
    g.message("Creating list of W*prop_b for all z")
    prop_list = [prop_b,]

    for z in range(1,zmax):
        prop_list.append(g.eval(g.adj(g.cshift(prop_b,2,z))*W[z]))
    
    return prop_list

# Function that essentially defines our version of DA
def contract_DA(prop_f, prop_b, phases, tag):

    # create and save correlators
    corr = g.slice_tr1(prop_b,prop_f,phases, 3)

    # corr = g.slice(
    #      g.trace(g.adj(prop_b) * W * g.gamma["Z"] * P * prop_f), 3)

    g.message("Starting IO")       
    for z, corr_p in enumerate(corr):
        corr_tag = "%s/DA/z%s" % (tag, str(z))
        for i, corr_mu in enumerate(corr_p):
            out_tag = f"{corr_tag}/p{plist[i]}"
            for j, corr_t in enumerate(corr_mu):
                out_tag = f"{out_tag}/{my_gammas[j]}"
                output_correlator.write(out_tag, corr_t)
                #g.message("Correlator %s\n" % out_tag, corr_t)


#function that does the contractions for the smeared-smeared pion 2pt function
def contract_2pt(prop_f, prop_b, phases, width, pos_boost, neg_boost, trafo, tag):
    #prop_tag = "%s/%s" % (tag, str(pos))
    g.message("Begin sink smearing")
    tmp_trafo = g.convert(trafo, prop_f.grid.precision)

    prop_f = g.create.smear.boosted_smearing(tmp_trafo, prop_f, w=width, boost=pos_boost)
    prop_b = g.create.smear.boosted_smearing(tmp_trafo, prop_b, w=width, boost=neg_boost)
    g.message("Sink smearing completed")

    # corr = g.slice(
        # g.trace( P *prop_f * g.adj(prop_b) ), 3
    # ) 

    corr = g.slice_tr1(prop_f,g.adj(prop_b),phases, 3)
    #print(np.shape(corr))
    #do correlator output
    corr_tag = "%s/2pt" % (tag)
    corr_p = corr[0]
    #print(np.shape(corr_p))
    for i, corr_mu in enumerate(corr_p):
        out_tag = f"{corr_tag}/p{plist[i]}"
        for j, corr_t in enumerate(corr_mu):
            out_tag = f"{out_tag}/{my_gammas[j]}"
            output_correlator.write(out_tag, corr_t)
            #g.message("Correlator %s\n" % out_tag, corr_t)

#function that creates boosted, smeared src.
def create_src_2pt(pos, width, pos_boost, neg_boost, trafo, grid):
    
    srcD = g.mspincolor(grid)
    srcD[:] = 0
    
    srcDp = g.mspincolor(grid)
    srcDp[:] = 0

    srcDm = g.mspincolor(grid)
    srcDm[:] = 0

    g.create.point(srcD, pos)


    srcDm = g.create.smear.boosted_smearing(trafo, srcD, w=width, boost=neg_boost)
    srcDp = g.create.smear.boosted_smearing(trafo, srcD, w=width, boost=pos_boost)

    del srcD

    return srcDp, srcDm