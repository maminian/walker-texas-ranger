from matplotlib import pyplot
import cmocean
import numpy as np
import utils


level = utils.load_level('level1.csv')
start = utils.get_start(level)
finish = utils.get_finish(level)


#
# initialize transition matrix
# this will be normalized after boundaries are
# excluded from the state.
#
transition = np.ones( list(level.shape)+[4], dtype=float)

for i in range(transition.shape[0]):
    for j in range(transition.shape[1]):
        uldr = utils.get_neighbors([i,j])
        for k,pair in enumerate(uldr):
            if not level[pair[0],pair[1]]:
                transition[i,j,k] = 0.
        #
        transition[i,j,:] /= np.sum(transition[i,j,:])
#

###

pos = np.array(start)

maxit = 1000
path = []
actions = []
path.append( pos )
completed = False
for k in range(maxit):
    action = utils.select_action(transition[pos[0],pos[1],:])

    pos = utils.get_neighbors(pos)[action]
    path.append( pos )
    actions.append( action )

    if all( pos==finish ):
        completed = True
        break
    #
    #_ = input()
#

path = np.array(path)

fig,ax = utils.vis_level(level)

ax.plot(path[:,0], path[:,1], c='r', lw=1)

fig.show()
