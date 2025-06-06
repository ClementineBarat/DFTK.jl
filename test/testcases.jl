@testmodule TestCases begin
using DFTK
using Unitful
using UnitfulAtomic
using LinearAlgebra: Diagonal, diagm
using PseudoPotentialData

gth_lda_large = PseudoFamily("cp2k.nc.sr.lda.v0_1.largecore.gth")
gth_lda_semi  = PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth")
pd_lda_family = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")

silicon = (;
    lattice = [0.0  5.131570667152971 5.131570667152971;
               5.131570667152971 0.0 5.131570667152971;
               5.131570667152971 5.131570667152971  0.0],
    atnum = 14,
    mass = 28.085u"u",
    n_electrons = 8,
    temperature = 0.0,
    psp_gth = gth_lda_semi[:Si],
    psp_upf = pd_lda_family[:Si],
    positions = [ones(3)/8, -ones(3)/8],      # in fractional coordinates
    kgrid = ExplicitKpoints([[   0,   0, 0],  # kcoords in fractional coordinates
                             [ 1/3,   0, 0],
                             [ 1/3, 1/3, 0],
                             [-1/3, 1/3, 0]],
                            [1/27, 8/27, 6/27, 12/27]),
)
silicon = merge(silicon,
                (; atoms=fill(ElementPsp(silicon.atnum, load_psp(silicon.psp_gth)), 2)))

magnesium = (;
    lattice = [-3.0179389205999998 -3.0179389205999998 0.0000000000000000;
               -5.2272235447000002 5.2272235447000002 0.0000000000000000;
               0.0000000000000000 0.0000000000000000 -9.7736219469000005],
    atnum = 12,
    mass = 24.305u"u",
    n_electrons = 4,
    psp_gth = gth_lda_large[:Mg],
    psp_upf = pd_lda_family[:Mg],
    positions = [[2/3, 1/3, 1/4], [1/3, 2/3, 3/4]],
    kgrid = ExplicitKpoints([[0,   0,   0],
                             [1/3, 0,   0],
                             [1/3, 1/3, 0],
                             [0,   0,   1/3],
                             [1/3, 0,   1/3],
                             [1/3, 1/3, 1/3]],
                            [1/27, 6/27, 2/27, 2/27, 12/27, 4/27]),
    temperature = 0.01,
)
magnesium = merge(magnesium,
                  (; atoms=fill(ElementPsp(magnesium.atnum, load_psp(magnesium.psp_gth)), 2)))


aluminium = (;
    lattice = Matrix(Diagonal([4 * 7.6324708938577865, 7.6324708938577865,
                               7.6324708938577865])),
    atnum = 13,
    mass = 39.9481u"u",
    n_electrons = 12,
    psp_gth = gth_lda_semi[:Al],
    psp_upf = pd_lda_family[:Al],
    positions = [[0, 0, 0], [0, 1/2, 1/2], [1/8, 0, 1/2], [1/8, 1/2, 0]],
    temperature = 0.0009500431544769484,
)
aluminium = merge(aluminium,
                  (; atoms=fill(ElementPsp(aluminium.atnum, load_psp(aluminium.psp_gth)), 4)))


aluminium_primitive = (;
    lattice = [5.39697192863632 2.69848596431816 2.69848596431816;
               0.00000000000000 4.67391479368660 1.55797159787754;
               0.00000000000000 0.00000000000000 4.40660912710674],
    atnum = 13,
    mass = 39.9481u"u",
    n_electrons = 3,
    psp_gth = gth_lda_semi[:Al],
    psp_upf = pd_lda_family[:Al],
    positions = [zeros(3)],
    temperature = 0.0009500431544769484,
)
aluminium_primitive = merge(aluminium_primitive,
                            (; atoms=fill(ElementPsp(aluminium_primitive.atnum,
                                                     load_psp(aluminium_primitive.psp_gth)), 1)))


platinum_hcp = (;
    lattice = [10.00000000000000 0.00000000000000 0.00000000000000;
               5.00000000000000 8.66025403784439 0.00000000000000;
               0.00000000000000 0.00000000000000 16.3300000000000],
    atnum = 78,
    mass = 195.0849u"u",
    n_electrons = 36,
    psp_gth = gth_lda_semi[:Pt],
    psp_upf = pd_lda_family[:Pt],
    positions = [zeros(3), ones(3) / 3],
    temperature = 0.0009500431544769484,
)
platinum_hcp = merge(platinum_hcp,
                     (; atoms=fill(ElementPsp(platinum_hcp.atnum,
                                              load_psp(platinum_hcp.psp_gth)), 2)))

iron_bcc = (;
    lattice = 2.71176 .* [[-1 1 1]; [1 -1  1]; [1 1 -1]],
    atnum = 26,
    mass = 55.8452u"u",
    n_electrons = 8,
    psp_gth = gth_lda_large[:Fe],
    psp_upf = pd_lda_family[:Fe],
    positions = [zeros(3)],
    temperature = 0.01,
)
iron_bcc = merge(iron_bcc, (; atoms=[ElementPsp(iron_bcc.atnum, load_psp(iron_bcc.psp_gth))]))

o2molecule = (;
    lattice = diagm([6.5, 6.5, 9.0]),
    atnum = 8,
    mass = 15.999u"u",
    n_electrons = 12,
    psp_gth = gth_lda_semi[:O],
    psp_upf = pd_lda_family[:O],
    positions = 0.1155 * [[0, 0, 1], [0, 0, -1]],
    temperature = 0.02,
)
o2molecule = merge(o2molecule,
                   (; atoms=fill(ElementPsp(o2molecule.atnum,
                                            load_psp(o2molecule.psp_gth)), 2)))

all_testcases = (; silicon, magnesium, aluminium, aluminium_primitive, platinum_hcp,
                 iron_bcc, o2molecule)
end
