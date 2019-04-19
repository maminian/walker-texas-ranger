from matplotlib import pyplot
import cmocean
import numpy as np
import utils


level,transition,start,finish = utils.initialize('level2.csv')

###

pathlens = []

for j in range(100):
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

    rv = utils.reward(path, completed, maxit)
    transition = utils.update_transitions(transition, path, actions, rv)

    pathlens.append( len(path) )
#


fig,ax = utils.vis_level(level)

ax.plot(path[:,0], path[:,1], c='r', lw=1)

fig.show()
