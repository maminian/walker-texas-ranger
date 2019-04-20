from matplotlib import pyplot
import numpy as np
import utils

level,transition,start,finish = utils.initialize('level1.csv')

fig,ax = utils.vis_level(level)

pyplot.ion()

###

pathlens = []

for j in range(100):
    # todo: throw this all into a function.
    pos = np.array(start)

    maxit = 10000
    path = []
    actions = []
    path.append( pos )
    completed = False
    for k in range(maxit):
        action = utils.select_action(transition[pos[0],pos[1],:])

        pos = utils.get_neighbors(pos,level)[action]
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

    if j%1==0:
        utils.vis_path(ax,path)
        utils.vis_transition(ax,transition)

        for entry in ax.texts:
            entry.remove()

        bbox=dict(facecolor='k', edgecolor='black', alpha=0.3)
        ax.text(0,0,'iteration %i'%(j+1), fontsize=14, va='center', ha='left', color='w',bbox=bbox)

        # save frames for animation.
        # fig.savefig('iter_%s'%str(j).zfill(4), dpi=120, bbox_inches='tight')
        pyplot.pause(0.02)
    #
#

fig.show()
