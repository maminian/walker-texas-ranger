# stray global stuff
import numpy as np
directions = np.array([[0,1],[0,-1],[-1,0],[1,0]])

def initialize(fname):
    '''
    Loads a level by filename.
    This should be a simple CSV file denoted by
        0: boundary (impassable)
        1: terrain (passable)
        2: starting location
        3: target location
    Empty entries in the csv file will be considered "1"
    for convenience.

    Inputs:
        fname: string; name of CSV file.
    Outputs:
        level: numpy array of integers which encode the level.
        transition: numpy array of shape (level.shape[0], level.shape[1], 4)
            which encodes transition probabilities at every location.
        start: numpy array size 2 of starting position
        finish: numpy array size 2 of final position
    '''
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
    '''
    Raw level-loading script. Handles various annoyances
    that prevents us from a simpler load function.

    It's recommended to just use initialize() instead.

    Inputs:
        fname: string; name of csv file
    Outputs:
        level: numpy array associated with the csv file,
            where empty entries are imputed with 1 (passable terrain).
    '''
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
    '''
    Get the coordinates of neighbors to the
    current position "tup". This is done
    with modular arithmetic to simplify code
    later.

    Inputs:
        tup: numpy integer array size 2 of the current coordinate.
        level: The numpy array encoding the level. Used only
            to determine the overall shape.
    Outputs:
        udlr: numpy array of shape (4,2) giving the coordinates
            of neighbors to "tup" in the order of "up, down, left, right",
            with a doubly periodic domain.
    '''
    import numpy as np

    nX,nY = level.shape

    tup = np.array(tup, dtype=int)
    udlr = np.array([[0,1],[0,-1],[-1,0],[1,0]]) + tup
    udlr[:2] = np.mod(udlr[:2], nY)
    udlr[2:] = np.mod(udlr[2:], nX)
    return udlr
#

def get_start(levelin):
    '''
    Gets the location of the start.
    Essentially np.where(level==2).

    Inputs:
        levelin: numpy array encoding the level
    Outputs:
        coord: numpy integer array shape 2 encoding the start position
    '''
    import numpy as np
    loc = np.where(levelin==2)
    return np.array([loc[0][0], loc[1][0]])
#

def get_finish(levelin):
    '''
    Gets the location of the finish.
    Essentially np.where(level==3).

    Inputs:
        levelin: numpy array encoding the level
    Outputs:
        coord: numpy integer array shape 2 encoding the finish position
    '''
    import numpy as np
    loc = np.where(levelin==3)
    return np.array([loc[0][0], loc[1][0]])
#

def select_action(tr):
    '''
    Samples from a probability mass function described by "tr".
    In principle "tr" can be an arbitrary list of floats.
    For this project it's a 4-vector encoding probabilities of
    moving up, down, left, right.

    Inputs:
        tr: numpy float array of probabilities, summing to one.
    Outputs:
        idx: the randomly selected index of the pmf (which
            indicates the direction to move).
    '''
    import numpy as np
    rn = np.random.rand()
    cmf = np.cumsum(tr)
    return np.where(rn < cmf)[0][0]
#

def reward(path,flag,maxiter):
    '''
    The reward function.

    The reward currently implemented is based on
        (a) Did we reach the finish?

            In this implementationn, no reward is given if flag==False;
            i.e. the walker has no kind of global information (or
            even heuristic global information) indicating "how close"
            they are to the finish if they didn't reach it.

        (b) How many iterations did it take?

            Transition updates more strongly for faster finishes.
            Currently this is a normalized factor based on maximum
            number of iterations the walker will bother with.
            This also doesn't account for the smallest possible
            path possible (so it's possible levels with very long
            shortest paths relative to maxiter will receive small rewards).

    Inputs:
        path : The list of the entire trajectory taken.
        flag : boolean; whether or not the walker reached the finish.
        maxiter : integer; the maximum number of iterations the walker
            would have attempted.
    Outputs:
        rv : float; the reward. By design this is in [0,1].
    '''
    import numpy as np
    return int(flag)*np.exp(-len(path)/maxiter)
#

def update_transitions(trs,path,actions,rv, kappa=0.1):
    '''
    Update the transition matrix based on the
    reward value, expected to be a value between 0 and 1.

    The update to the transition is a re-apportioning of
    a constant factor "kappa" of the unused actions at each location
    in the path to the "successful" action. In principle this
    means that non-optimal things can be learned, and
    local trajectories off the optimal can be "ingrained"
    and never optimized if they aren't addressed in the
    first few iterations.

    Inputs:
        trs: global transition array (is this the Q table?)
        path: list of positions the walker took
        actions: the corresponding action took at each entry of path.
        rv: the reward which was decided upon, likely by the reward()
            function above.
        kappa: the factor which controls how strongly the pmf is
            adjusted towards a "successful" action. Optional.
            Default: 0.1. Must be in (0,1).
    Outputs:
        trs: The updated transition array.
    '''
    import numpy as np

    if rv==0:
        return trs
    else:
        for p,a in zip(path,actions):
            # re-apportion the transition probabilities
            # according to a geometric factor kappa.
            others = np.setdiff1d(np.arange(4), a)
            pot = sum(trs[p[0],p[1],others])

            trs[p[0],p[1],others] *= (1-kappa)
            trs[p[0],p[1],a] += kappa*pot
        #
        return trs
    #
#

#################################################
#
# Visualization tools below
#

def vis_level(level, show=True):
    '''
    Visualization of the level itself, without
    any additional bells or whistles.

    Inputs:
        level: numpy array encoding the level.
        show: whether to fig.show() within this function. (Default: True)
    Outputs:
        fig,ax: a pyplot figure/axis pair which
            can be used for plotting additional information.
    '''
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

    if show: fig.show()

    return fig,ax
#

def vis_transition(myax, trs, remove_old=True):
    '''
    Visualize the transitions by drawing arrows proportional
    to the probabilities for each direction.

    Inputs:
        myax: axis to draw transitions on
        trs: transition array
        remove_old: boolean; whether to remove all artists
            on the axis (presumed to be previously drawn arrows)
            Optional. Default: True

    Outputs:
        None
    '''
    import numpy as np
    scaling = 0.6

    for i in range(trs.shape[0]):
        for j in range(trs.shape[1]):
            tr = trs[i,j,:]
            for k in range(4):
                if tr[k]==0:
                    # don't draw length-zero vectors
                    continue
                # note: directions is defined at the top of this file.
                vec = scaling*tr[k]*directions[k]
                drawarrow(myax, i, j, vec[0], vec[1])
    #
    return
#

def drawarrow(myax,x,y,dx,dy):
    '''
    Helper function for vis_transition(). Draws arrows with chosen properties.
    '''
    myax.arrow(x,y,dx,dy, facecolor='k', edgecolor='w', width=0.05, linewidth=0.4)
    return
#

def vis_path(ax, mypath, remove_old=True):
    '''
    Visualize the path taken by a walker on the desired
    pyplot axis. Arbitrary choices in plot style are hard-coded for now.

    Inputs:
        ax: pyplot axis to draw the path on.
        mypath: the ordered list of coordinates the walker visited.
        remove_old: boolean; whether to remove all artists
            on the axis (presumed to be previously drawn arrows)
            Optional. Default: True

    Outputs:
        None
    '''
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
