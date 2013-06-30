from openmoc import *
import openmoc.log as log
import openmoc.plotter as plotter
import openmoc.materialize as materialize
import openmoc.mic as mic


###############################################################################
#######################   Main Simulation Parameters   ########################
###############################################################################

num_threads = 4
track_spacing = 0.1
num_azim = 16
tolerance = 1E-3
max_iters = 1000
gridsize = 500

log.setLogLevel('NORMAL')


###############################################################################
###########################   Creating Materials   ############################
###############################################################################

log.py_printf('NORMAL', 'Importing materials data from HDF5...')

materials = materialize.materialize('../c5g7-materials.hdf5')

uo2_id = materials['UO2'].getId()
water_id = materials['Water'].getId()


###############################################################################
###########################   Creating Surfaces   #############################
###############################################################################

log.py_printf('NORMAL', 'Creating surfaces...')

circle = Circle(x=0.0, y=0.0, radius=1.0)
left = XPlane(x=-2.0)
right = XPlane(x=2.0)
top = YPlane(y=2.0)
bottom = YPlane(y=-2.0)

left.setBoundaryType(REFLECTIVE)
right.setBoundaryType(REFLECTIVE)
top.setBoundaryType(REFLECTIVE)
bottom.setBoundaryType(REFLECTIVE)


###############################################################################
#############################   Creating Cells   ##############################
###############################################################################

log.py_printf('NORMAL', 'Creating cells...')

cells = []
cells.append(CellBasic(universe=1, material=uo2_id))
cells.append(CellBasic(universe=1, material=water_id))
cells.append(CellFill(universe=0, universe_fill=2))

cells[0].addSurface(halfspace=-1, surface=circle)
cells[1].addSurface(halfspace=+1, surface=circle)
cells[2].addSurface(halfspace=+1, surface=left)
cells[2].addSurface(halfspace=-1, surface=right)
cells[2].addSurface(halfspace=+1, surface=bottom)
cells[2].addSurface(halfspace=-1, surface=top)


###############################################################################
###########################   Creating Lattices   #############################
###############################################################################

log.py_printf('NORMAL', 'Creating simple pin cell lattice...')

lattice = Lattice(id=2, width_x=4.0, width_y=4.0)
lattice.setLatticeCells([[1]])


###############################################################################
##########################   Creating the Geometry   ##########################
###############################################################################

log.py_printf('NORMAL', 'Creating geometry...')

Timer.startTimer()

geometry = Geometry()
for material in materials.values(): geometry.addMaterial(material)
for cell in cells: geometry.addCell(cell)
geometry.addLattice(lattice)

geometry.initializeFlatSourceRegions()

Timer.stopTimer()
Timer.recordSplit('Iniitilializing the geometry')
Timer.resetTimer()


###############################################################################
########################   Creating the TrackGenerator   ######################
###############################################################################

log.py_printf('NORMAL', 'Initializing the track generator...')

Timer.startTimer()

track_generator = TrackGenerator(geometry, num_azim, track_spacing)
track_generator.generateTracks()

Timer.stopTimer()
Timer.recordSplit('Ray tracing across the geometry')
Timer.resetTimer()


###############################################################################
###########################   Running a Simulation   ##########################
###############################################################################

Timer.startTimer()

solver = mic.MICSolver(geometry, track_generator)
solver.setNumThreads(num_threads)
solver.setSourceConvergenceThreshold(tolerance)
solver.convergeSource(max_iters)

Timer.stopTimer()
Timer.recordSplit('Converging the source with %d MIC threads' % (num_threads))
Timer.resetTimer()


###############################################################################
############################   Generating Plots   #############################
###############################################################################

log.py_printf('NORMAL', 'Plotting data...')

Timer.startTimer()

#plotter.plotTracks(track_generator)
#plotter.plotSegments(track_generator)
#plotter.plotMaterials(geometry, gridsize)
#plotter.plotCells(geometry, gridsize)
#plotter.plotFlatSourceRegions(geometry, gridsize)
#plotter.plotFluxes(geometry, solver, energy_groups=[1,2,3,4,5,6,7])

Timer.stopTimer()
Timer.recordSplit('Generating visualizations')
Timer.resetTimer()
Timer.printSplits()

log.py_printf('TITLE', 'Finished')
