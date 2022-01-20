from cmath import phase
from math import gamma
import gpt as g
import numpy as np


#ordered list of gamma matrix identifiers, needed for the tag in the correlator output
my_gammas = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]

class pion_measurement:
    def __init__(self, parameters):
        self.zmax = parameters["zmax"]
        self.pzmin = parameters["pzmin"]
        self.pzmax = parameters["pzmax"]
        self.plist = range(self.pzmin,self.pzmax)
        self.width = parameters["width"]
        self.pos_boost = parameters["pos_boost"]
        self.neg_boost = parameters["neg_boost"]
        self.save_propagators = parameters["save_propagators"]

    def set_output_facilites(self, corr_file, prop_file):
        self.output_correlator = g.corr_io.writer(corr_file)
        
        if(self.save_propagators):
            self.output = g.gpt_io.writer(prop_file)

    def propagator_output(self, tag, prop_f, prop_b):

        g.message("Saving forward propagator")
        prop_f_tag = "%s/%s" % (tag, str(self.pos_boost)) 
        self.output.write({prop_f_tag: prop_f})
        self.output.flush()
        g.message("Saving backward propagator")
        prop_b_tag = "%s/%s" % (tag, str(self.neg_boost))
        self.output.write({prop_b_tag: prop_b})
        self.output.flush()
        g.message("Propagator IO done")


    #make the inverters needed for the 96I lattices
    def make_96I_inverter(self, U, evec_file):

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

    def make_debugging_inverter(self, U):

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

        prop_l_sloppy = l_exact.propagator(light_sloppy_inverter).grouped(6)
        prop_l_exact = l_exact.propagator(light_exact_inverter).grouped(6)
        return prop_l_exact, prop_l_sloppy


    ############## make list of complex phases for momentum proj.
    def make_mom_phases(self, grid):    
        one = g.complex(grid)
        p = [-2 * np.pi * np.array([0, 0, pz, 0]) / grid.fdimensions for pz in self.plist]
        P = g.exp_ixp(p)
        mom = [g.eval(pp*one) for pp in P]
        return mom

    # create Wilson lines
    def create_WL(self, U):
        W = []
        W.append(g.qcd.gauge.unit(U[2].grid)[0])
        for dz in range(0, self.zmax):
            W.append(g.eval(g.cshift(U[2], 2, dz) * W[dz-1]))
                
        return W


    #function that does the contractions for the smeared-smeared pion 2pt function
    def contract_2pt(self, prop_f, prop_b, phases, trafo, tag):

        g.message("Begin sink smearing")
        tmp_trafo = g.convert(trafo, prop_f.grid.precision)

        prop_f = g.create.smear.boosted_smearing(tmp_trafo, prop_f, w=self.width, boost=self.pos_boost)
        prop_b = g.create.smear.boosted_smearing(tmp_trafo, prop_b, w=self.width, boost=self.neg_boost)
        g.message("Sink smearing completed")

        # corr = g.slice(
            # g.trace( P *prop_f * g.adj(prop_b) ), 3
        # ) 

        corr = g.slice_tr1(prop_f,g.adj(prop_b),phases, 3)

        #do correlator output
        corr_tag = "%s/2pt" % (tag)
        corr_p = corr[0]
        for i, corr_mu in enumerate(corr_p):
            out_tag = f"{corr_tag}/p{self.plist[i]}"
            for j, corr_t in enumerate(corr_mu):
                out_tag = f"{out_tag}/{my_gammas[j]}"
                self.output_correlator.write(out_tag, corr_t)
                #g.message("Correlator %s\n" % out_tag, corr_t)

    #function that creates boosted, smeared src.
    def create_src_2pt(self, pos, trafo, grid):
        
        srcD = g.mspincolor(grid)
        srcD[:] = 0
        
        g.create.point(srcD, pos)


        srcDm = g.create.smear.boosted_smearing(trafo, srcD, w=self.width, boost=self.neg_boost)
        srcDp = g.create.smear.boosted_smearing(trafo, srcD, w=self.width, boost=self.pos_boost)

        del srcD

        return srcDp, srcDm



class pion_DA_measurement(pion_measurement):

    # Creating list of W*prop_b for all z
    def constr_backw_prop_for_DA(self, prop_b, W):
        g.message("Creating list of W*prop_b for all z")
        prop_list = [prop_b,]

        for z in range(1,self.zmax):
            prop_list.append(g.eval(g.adj(g.cshift(prop_b,2,z))*W[z]))
        
        return prop_list

    # Function that essentially defines our version of DA
    def contract_DA(self, prop_f, prop_b, phases, tag):

        # create and save correlators
        corr = g.slice_tr1(prop_b,prop_f,phases, 3)

        # corr = g.slice(
        #      g.trace(g.adj(prop_b) * W * g.gamma["Z"] * P * prop_f), 3)

        g.message("Starting IO")       
        for z, corr_p in enumerate(corr):
            corr_tag = "%s/DA/z%s" % (tag, str(z))
            for i, corr_mu in enumerate(corr_p):
                out_tag = f"{corr_tag}/p{self.plist[i]}"
                for j, corr_t in enumerate(corr_mu):
                    out_tag = f"{out_tag}/{my_gammas[j]}"
                    self.output_correlator.write(out_tag, corr_t)
                    #g.message("Correlator %s\n" % out_tag, corr_t)

class pion_ff_measurement(pion_measurement):

    #this still needs work!
    def contract_FF(self, prop_f, prop_b, phases, tag):

        #This should work, as both prop are not lists! 
        #Maybe the in the IO I need a corr[0] in the first enumerate(corr)
        corr = g.slice_tr1(prop_b, prop_f, phases, 3)

        g.message("Starting IO")
        for i, corr_mu in enumerate(corr):
            out_tag = f"{tag}/FF/q{self.plist[i]}"
            for j, corr_t in enumerate(corr_mu):
                out_tag = f"{out_tag}/{my_gammas[j]}"
                self.output_correlator.write(out_tag, corr_t)



    def create_bw_seq(self, inverter, prop, trafo, t_insert, pz):

        tmp_trafo = g.convert(trafo, prop.grid.precision)

        sp_prop = g.create.smear.boosted_smearing(tmp_trafo, prop, w=self.width, boost=self.neg_boost)

        del prop

        p = 2.0 * np.pi * np.array(pz) / prop.grid.fdimensions
        P = g.exp_ixp(p)

        # sequential solve through t=insertion_time
        t_op = t_insert
        src_seq = g.lattice(sp_prop)
        src_seq[:] = 0
        src_seq[:, :, :, t_op] = sp_prop[:, :, :, t_op]

        del sp_prop

        src_seq @=  g.gamma["5"]* src_seq
        #multiply with complex conjugate phase because it's a backwards prop.
        src_seq @= g.adj(P)* src_seq 


        #does overwriting on the fly work?
        src_seq = g.create.smear.boosted_smearing(tmp_trafo, src_seq, w=self.width, boost=self.pos_boost)

        dst_seq = g.lattice(src_seq)
        dst_seq @= inverter * src_seq

        dst_seq = g.create.smear.boosted_smearing(tmp_trafo, g.eval(dst_seq), w=self.width, boost=self.neg_boost)

        #This is now in principle B_zx but with the complex conj phase and a missing factor of gamma5 

        return (g.adj(dst_seq)*g.gamma["5"])

class pion_qpdf_measurement(pion_measurement):

    #this still needs work!
    def contract_QPDF(self, prop_f, prop_b, tag):

        #This contraction function still needs to be written, as of now
        #a list as 2nd argument does not work
        corr = g.slice_tr2(prop_b, prop_f, 3)

        g.message("Starting IO")       
        for z, corr_mu in enumerate(corr):
            corr_tag = "%s/QPDF/z%s" % (tag, str(z))
            out_tag = f"{corr_tag}/{self.pz}"
            for j, corr_t in enumerate(corr_mu):
                out_tag = f"{out_tag}/{my_gammas[j]}"
                self.output_correlator.write(out_tag, corr_t)

    def create_fw_prop_QPDF(self, prop_f, W):
        g.message("Creating list of W*prop_f for all z")
        prop_list = [prop_f,]

        for z in range(1,self.zmax):
            prop_list.append(g.eval(W[z]*g.cshift(prop_f,2,z)))
        
        return prop_list      

    def create_bw_seq(self, inverter, prop, trafo, t_insert, pz, isFF):

        tmp_trafo = g.convert(trafo, prop.grid.precision)

        sp_prop = g.create.smear.boosted_smearing(tmp_trafo, prop, w=self.width, boost=self.pos_boost)
        
        del prop

        p = 2.0 * np.pi * np.array(pz) / prop.grid.fdimensions
        P = g.exp_ixp(p)

        # sequential solve through t=insertion_time
        t_op = t_insert
        src_seq = g.lattice(sp_prop)
        src_seq[:] = 0
        src_seq[:, :, :, t_op] = sp_prop[:, :, :, t_op]

        del sp_prop

        src_seq @=  g.gamma["5"]* src_seq
        #multiply with complex conjugate phase because it's a backwards prop.
        src_seq @= g.adj(P)* src_seq 


        #does overwriting on the fly work?
        src_seq = g.create.smear.boosted_smearing(tmp_trafo, src_seq, w=self.width, boost=self.neg_boost)

        dst_seq = g.lattice(src_seq)
        dst_seq @= inverter * src_seq

        dst_seq = g.create.smear.boosted_smearing(tmp_trafo, g.eval(dst_seq), w=self.width, boost=self.neg_boost)

        #This is now in principle B_zx but with the complex conj phase and a missing factor of gamma5 

        return (g.adj(dst_seq)*g.gamma["5"])