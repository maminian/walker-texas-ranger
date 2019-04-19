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
            udlr = get_neighbors([i,j], level)
            for k,pair in enumerate(udlr):
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
            level = np.array([row[:-1] for row in level])
        else:
            level = np.array(level)
    #

    # assume missing entries are meant to be 1.
    level[level==''] = '1'
    level = np.array(level, dtype=int)

    return level
#

def get_neighbors(tup,level):
    import numpy as np

    nX,nY = level.shape

    tup = np.array(tup, dtype=int)
    udlr = np.array([[0,1],[0,-1],[-1,0],[1,0]]) + tup
    udlr[:2] = np.mod(udlr[:2], nY)
    udlr[2:] = np.mod(udlr[2:], nX)
    return udlr
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
    kappa = 0.2

    if rv==0:
        return trs
    else:
        # current choice is to save the position
        for p,a in zip(path,actions):
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

    fig,ax = pyplot.subplots(1,1, figsize=(6,6))
    ax.imshow(level.T, cmap=mycm)
    fig.tight_layout()

    return fig,ax
#

directions = np.array([[0,1],[0,-1],[-1,0],[1,0]])
def vis_transition(myax,trs):
    import numpy as np
    scaling = 0.6

    for i in range(trs.shape[0]):
        for j in range(trs.shape[1]):
            tr = trs[i,j,:]
            for k in range(4):
                if tr[k]==0:
                    continue
                vec = scaling*tr[k]*directions[k]
                drawarrow(myax, i, j, vec[0], vec[1])
    #
    return
#

def drawarrow(myax,x,y,dx,dy):
    myax.arrow(x,y,dx,dy, facecolor='k', edgecolor='w', width=0.05, linewidth=0.4)
    return
#

def vis_path(ax,mypath, remove_old=True):
    import numpy as np
    from matplotlib import pyplot

    if remove_old:
        for entry in ax.lines:
            entry.remove()
    #

    mypath = np.array(mypath)
    ax.plot(mypath[:,0], mypath[:,1],lw=3, alpha=0.7, c=pyplot.cm.tab10(1))
    return
#

if __name__=="__main__":
    import numpy as np
    from matplotlib import pyplot

    level,transition,start,finish = initialize('level3.csv')

    fig,ax = vis_level(level)
    vis_transition(ax,transition)

    fig.show()
    pyplot.ion()
#
