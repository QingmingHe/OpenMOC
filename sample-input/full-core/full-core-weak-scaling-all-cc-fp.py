import numpy
from openmoc import *
import openmoc.log as log
import openmoc.materialize as materialize
import openmoc.process as process


###############################################################################
#######################   Main Simulation Parameters   ########################
###############################################################################

num_threads = numpy.linspace(1,12,12)
compiler = 'all'
precision = 'all'
num_azim = 8

setOutputDirectory('Full-Core-Weak-Scaling')

log.setLogLevel('NORMAL')

log.py_printf('TITLE', 'Weak Scaling a Mock Full Core PWR...')


###############################################################################
###########################   Creating Materials   ############################
###############################################################################

log.py_printf('NORMAL', 'Importing materials data from HDF5...')

materials = materialize.materialize('../c5g7-materials.hdf5')

uo2_id = materials['UO2'].getId()
mox43_id = materials['MOX-4.3%'].getId()
mox7_id = materials['MOX-7%'].getId()
mox87_id = materials['MOX-8.7%'].getId()
guide_tube_id = materials['Guide Tube'].getId()
fiss_id = materials['Fission Chamber'].getId()
water_id = materials['Water'].getId()


###############################################################################
###########################   Creating Surfaces   #############################
###############################################################################

log.py_printf('NORMAL', 'Creating surfaces...')

circles = []
planes = []
planes.append(XPlane(x=-224.91))
planes.append(XPlane(x=224.91))
planes.append(YPlane(y=-224.91))
planes.append(YPlane(y=224.91))
circles.append(Circle(x=0., y=0., radius=0.54))
circles.append(Circle(x=0., y=0., radius=0.58))
circles.append(Circle(x=0., y=0., radius=0.62))
planes[0].setBoundaryType(VACUUM)
planes[1].setBoundaryType(VACUUM)
planes[2].setBoundaryType(VACUUM)
planes[3].setBoundaryType(VACUUM)


###############################################################################
#############################   Creating Cells   ##############################
###############################################################################

log.py_printf('NORMAL', 'Creating cells...')

cells = []

# UO2 pin cells
cells.append(CellBasic(universe=1, material=uo2_id, rings=3, sectors=8))
cells.append(CellBasic(universe=1, material=water_id, sectors=8))
cells.append(CellBasic(universe=1, material=water_id, sectors=8))
cells.append(CellBasic(universe=1, material=water_id, sectors=8))
cells[0].addSurface(-1, circles[0])
cells[1].addSurface(+1, circles[0])
cells[1].addSurface(-1, circles[1])
cells[2].addSurface(+1, circles[1])
cells[2].addSurface(-1, circles[2])
cells[3].addSurface(+1, circles[2])


# 4.3% MOX pin cells
cells.append(CellBasic(universe=2, material=mox43_id, rings=3, sectors=8))
cells.append(CellBasic(universe=2, material=water_id, sectors=8))
cells.append(CellBasic(universe=2, material=water_id, sectors=8))
cells.append(CellBasic(universe=2, material=water_id, sectors=8))
cells[4].addSurface(-1, circles[0])
cells[5].addSurface(+1, circles[0])
cells[5].addSurface(-1, circles[1])
cells[6].addSurface(+1, circles[1])
cells[6].addSurface(-1, circles[2])
cells[7].addSurface(+1, circles[2])


# 7% MOX pin cells
cells.append(CellBasic(universe=3, material=mox7_id, rings=3, sectors=8))
cells.append(CellBasic(universe=3, material=water_id, sectors=8))
cells.append(CellBasic(universe=3, material=water_id, sectors=8))
cells.append(CellBasic(universe=3, material=water_id, sectors=8))
cells[8].addSurface(-1, circles[0])
cells[9].addSurface(+1, circles[0])
cells[9].addSurface(-1, circles[1])
cells[10].addSurface(+1, circles[1])
cells[10].addSurface(-1, circles[2])
cells[11].addSurface(+1, circles[2])


# 8.7% MOX pin cells
cells.append(CellBasic(universe=4, material=mox87_id, rings=3, sectors=8))
cells.append(CellBasic(universe=4, material=water_id, sectors=8))
cells.append(CellBasic(universe=4, material=water_id, sectors=8))
cells.append(CellBasic(universe=4, material=water_id, sectors=8))
cells[12].addSurface(-1, circles[0])
cells[13].addSurface(+1, circles[0])
cells[13].addSurface(-1, circles[1])
cells[14].addSurface(+1, circles[1])
cells[14].addSurface(-1, circles[2])
cells[15].addSurface(+1, circles[2])

# Fission chamber pin cells
cells.append(CellBasic(universe=5, material=fiss_id, rings=3, sectors=8))
cells.append(CellBasic(universe=5, material=water_id, sectors=8))
cells.append(CellBasic(universe=5, material=water_id, sectors=8))
cells.append(CellBasic(universe=5, material=water_id, sectors=8))
cells[16].addSurface(-1, circles[0])
cells[17].addSurface(+1, circles[0])
cells[17].addSurface(-1, circles[1])
cells[18].addSurface(+1, circles[1])
cells[18].addSurface(-1, circles[2])
cells[19].addSurface(+1, circles[2])

# Guide tube pin cells
cells.append(CellBasic(universe=6, material=guide_tube_id, rings=3, sectors=8))
cells.append(CellBasic(universe=6, material=water_id, sectors=8))
cells.append(CellBasic(universe=6, material=water_id, sectors=8))
cells.append(CellBasic(universe=6, material=water_id, sectors=8))
cells[20].addSurface(-1, circles[0])
cells[21].addSurface(+1, circles[0])
cells[21].addSurface(-1, circles[1])
cells[22].addSurface(+1, circles[1])
cells[22].addSurface(-1, circles[2])
cells[23].addSurface(+1, circles[2])

# Moderator cell
cells.append(CellBasic(universe=7, material=water_id))

# Top left, bottom right lattice
cells.append(CellFill(universe=10, universe_fill=20))

# Top right, bottom left lattice
cells.append(CellFill(universe=11, universe_fill=21))

# Moderator lattice - very finely spaced
cells.append(CellFill(universe=12, universe_fill=22))

# Moderator lattice - semi-finely spaced
cells.append(CellFill(universe=13, universe_fill=23))

#Moderator lattice - coarsely spaced
cells.append(CellFill(universe=14, universe_fill=24))

# Full geometry
cells.append(CellFill(universe=0, universe_fill=30))
cells[-1].addSurface(+1, planes[0])
cells[-1].addSurface(-1, planes[1])
cells[-1].addSurface(+1, planes[2])
cells[-1].addSurface(-1, planes[3])


###############################################################################
###########################   Creating Lattices   #############################
###############################################################################

log.py_printf('NORMAL', 'Creating lattices...')

lattices = []

# Top left, bottom right 17 x 17 assemblies
lattices.append(Lattice(id=20, width_x=1.26, width_y=1.26))
lattices[-1].setLatticeCells(
    [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1, 1, 1, 1],
     [1, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 6, 1, 1, 6, 1, 1, 5, 1, 1, 6, 1, 1, 6, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 1, 1, 1],
     [1, 1, 1, 1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])


# Top right, bottom left 17 x 17 assemblies 
lattices.append(Lattice(id=21, width_x=1.26, width_y=1.26))
lattices[-1].setLatticeCells(
    [[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
     [2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2],
     [2, 3, 3, 3, 3, 6, 3, 3, 6, 3, 3, 6, 3, 3, 3, 3, 2],
     [2, 3, 3, 6, 3, 4, 4, 4, 4, 4, 4, 4, 3, 6, 3, 3, 2],
     [2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 2],
     [2, 3, 6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 3, 2],
     [2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 2],
     [2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 2],
     [2, 3, 6, 4, 4, 6, 4, 4, 5, 4, 4, 6, 4, 4, 6, 3, 2],
     [2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 2],
     [2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 2],
     [2, 3, 6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 3, 2],
     [2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 2],
     [2, 3, 3, 6, 3, 4, 4, 4, 4, 4, 4, 4, 3, 6, 3, 3, 2],
     [2, 3, 3, 3, 3, 6, 3, 3, 6, 3, 3, 6, 3, 3, 3, 3, 2],
     [2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2],
     [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]])


# Sliced up water cells - very finely spaced
lattices.append(Lattice(id=22, width_x=0.252, width_y=0.252))
lattices[-1].setLatticeCells(
    [[7, 7, 7, 7, 7],
     [7, 7, 7, 7, 7],
     [7, 7, 7, 7, 7],
     [7, 7, 7, 7, 7],
     [7, 7, 7, 7, 7]])


# Semi-Finely spaced water lattice (0.063 cm x 0.063 cm)
lattices.append(Lattice(id=23, width_x=1.26, width_y=1.26))
lattices[-1].setLatticeCells(
    [[12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]])


# Coarsely spaced water lattice (1.26 cm x 1.26 cm)
lattices.append(Lattice(id=24, width_x=1.26, width_y=1.26))
lattices[-1].setLatticeCells(
        [[7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
         [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
         [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
         [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
         [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
         [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
         [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
         [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
         [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
         [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
         [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
         [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
         [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
         [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
         [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
         [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
         [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]])


# 21 x 21 core to represent two bundles and water
lattices.append(Lattice(id=30, width_x=21.42, width_y=21.42))
lattices[-1].setLatticeCells(
            [[13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13],
             [13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 13, 13],
             [13, 13, 13, 13, 14, 14, 14, 10, 11, 10, 11, 10, 11, 10, 14, 14, 13, 13, 13, 13, 13],
             [13, 13, 13, 14, 14, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 14, 13, 13, 13, 13],
             [13, 13, 14, 14, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 14, 14, 13, 13],
             [13, 13, 14, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 14, 13, 13],
             [13, 14, 14, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 14, 13, 13],
             [13, 14, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 14, 13],
             [13, 14, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 14, 13],
             [13, 14, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 14, 13],
             [13, 14, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 14, 13],
             [13, 14, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 14, 13],
             [13, 14, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 14, 13],
             [13, 14, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 14, 13],
             [13, 14, 14, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 14, 13, 13],
             [13, 13, 14, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 14, 13, 13],
             [13, 13, 14, 14, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 14, 13, 13, 13],
             [13, 13, 13, 14, 14, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 14, 13, 13, 13, 13],
             [13, 13, 13, 13, 14, 14, 14, 10, 11, 10, 11, 10, 11, 10, 14, 14, 13, 13, 13, 13, 13],
             [13, 13, 14, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 13, 13],
             [13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13]])


###############################################################################
##########################   Creating the Geometry   ##########################
###############################################################################

log.py_printf('NORMAL', 'Creating geometry...')

geometry = Geometry()

for material in materials.values(): geometry.addMaterial(material)
for cell in cells: geometry.addCell(cell)
for lattice in lattices: geometry.addLattice(lattice)

geometry.initializeFlatSourceRegions()


###############################################################################
###############################   Weak Scaling   ##############################
###############################################################################

log.py_printf('NORMAL', 'Running a weak scaling study...')

process.weakScalingStudy(geometry, num_azim=num_azim, precision=precision, 
                           compiler=compiler, num_threads=num_threads)

log.py_printf('TITLE', 'Finished')