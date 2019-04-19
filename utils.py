import numpy as np

def initialize(fname):

    import numpy as np

    level = load_level(fname)
    start = get_start(level)
    finish = get_finish(level)


    #
    # initialize transition matrix
    # this will be normalized after boundaries are
    # excluded from the state.
    #
    transition = np.ones( list(level.shape)+[4], dtype=float)

    for i in range(transition.shape[0]):
        for j in range(transition.shape[1]):
            uldr = get_neighbors([i,j])
            for k,pair in enumerate(uldr):
                if not level[pair[0],pair[1]]:
                    transition[i,j,k] = 0.
            #
            transition[i,j,:] /= np.sum(transition[i,j,:])
    #
    transition[np.where(np.isnan(transition))]

    return level,transition,start,finish
#

def load_level(fname):
    import numpy as np
    import csv
    with open(fname,'r') as f:
        csvr = csv.reader(f)
        level = list(csvr)
        if level[0][-1]=='':
            level = np.array([row[:-1] for row in level], dtype=int)
        else:
            level = np.array(level, dtype=int)
    #
    return level
#

def get_neighbors(tup):
    import numpy as np
    tup = np.array(tup, dtype=int)
    uldr = np.array([[0,1],[0,-1],[-1,0],[1,0]]) + tup
    uldr = np.mod(uldr,9)
    return uldr
#

def get_start(levelin):
    import numpy as np
    loc = np.where(levelin==2)
    return np.array([loc[0][0], loc[1][0]])
#

def get_finish(levelin):
    import numpy as np
    loc = np.where(levelin==3)
    return np.array([loc[0][0], loc[1][0]])
#

def select_action(tr):
    #
    # Sampling from a pmf defined by tr.
    #
    import numpy as np
    rn = np.random.rand()
    cmf = np.cumsum(tr)
    return np.where(rn < cmf)[0][0]
#

# Update the transitions based on
# (a) did we reach the finish? first pass - no reward if false.
# (b) how many iterations did it take? Transition updates more
# strongly for faster finishes.

def reward(path,flag,maxiter):
    import numpy as np
    return int(flag)*np.exp(-len(path)/maxiter)
#

def update_transitions(trs,path,actions,rv):
    '''
    Update the transition matrix based on the
    reward value, expected to be a value between 0 and 1.
    '''
    import numpy as np

    # need some kind of diffusive-like reassignment of
    # the transition probabilities.
    kappa = 0.5

    if rv==0:
        return trs
    else:
        # current choice is to save the position
        for p,a in zip(path[:-1],actions[:-1]):
            others = np.setdiff1d(np.arange(4), a)
            pot = sum(trs[p[0],p[1],others])

            trs[p[0],p[1],others] *= (1-kappa)
            trs[p[0],p[1],a] += kappa*pot
        #
        return trs
    #
#

def vis_level(level):
    from matplotlib import pyplot
    try:
        import cmocean
        mycm = cmocean.cm.algae_r
    except:
        mycm = pyplot.cm.inferno_r
    #

    fig,ax = pyplot.subplots(1,1)
    ax.imshow(level.T, cmap=mycm)

    return fig,ax
#

directions = np.array([[0,1],[0,-1],[-1,0],[1,0]])
def vis_transition(myax,trs):
    import numpy as np
    scaling = 0.5

    for i in range(trs.shape[0]):
        for j in range(trs.shape[1]):
            tr = trs[i,j,:]
            for k in range(4):
                vec = scaling*tr[k]*directions[k]
                # myax.arrow(i,j,vec[0],vec[1], c='r',)
                drawarrow(myax, i, j, vec[0], vec[1])
    #
    return
#

def drawarrow(myax,x,y,dx,dy):
    myax.arrow(x,y,dx,dy, facecolor='k', edgecolor='w', width=0.05, edgewidth=0.05)
    return
#