# This example shows how to extend/modify the plots generated

from emva1288.process import Emva1288
from emva1288.camera.dataset_generator import DatasetGenerator
from emva1288.process.plotting import Emva1288Plot, EVMA1288plots


# Create a plot
class PRNUmage(Emva1288Plot):
    name = 'PRNU Image'

    def plot(self, test):
        self.ax.imshow(test.spatial['avg'],
                       aspect='auto',
                       gid='%d:data' % test.id)


# Get a results object
dataset_generator = DatasetGenerator(width=100,
                                     height=50,
                                     bit_depth=8,
                                     dark_current_ref=30)
fname = dataset_generator.descriptor_path
e = Emva1288(fname)

######
# To plot only the PRNUImage
e.plot(PRNUmage)

#####
# To plot all plus PRNUImage
plots = EVMA1288plots + [PRNUmage]
e.plot(*plots)
