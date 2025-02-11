import torch


def memo(f):
    # Peter Norvig's
    """Memorize the return value for each call to f(args).
    Then when called again with same args, we can just look it up."""
    cache = {}

    def _f(*args):
        try:
            return cache[args]
        except KeyError:
            cache[args] = result = f(*args)
            return result
        except TypeError:
            # some element of args can't be a dict key
            return f(*args)

    _f.cache = cache
    return _f


@memo
def bspline_basis(c, n, degree):
    """ bspline basis function
        c        = number of control points.
        n        = number of points on the curve.
        degree   = curve degree
    """
    # Create knot vector and a range of samples on the curve
    kv = torch.tensor([0] * degree + torch.arange(c - degree + 1).tolist() +
                      [c - degree] * degree, dtype=torch.int)  # knot vector
    u = torch.linspace(0, c - degree, n, dtype=torch.float)  # samples range

    # Cox - DeBoor recursive function to calculate basis
    def coxDeBoor(k, d):
        # Test for end conditions
        if (d == 0):
            return ((u - kv[k] >= 0) & (u - kv[k + 1] < 0)).type(torch.float)

        denom1 = kv[k + d] - kv[k]
        term1 = 0
        if denom1 > 0:
            term1 = ((u - kv[k]) / denom1) * coxDeBoor(k, d - 1)

        denom2 = kv[k + d + 1] - kv[k + 1]
        term2 = 0
        if denom2 > 0:
            term2 = ((-(u - kv[k + d + 1]) / denom2) * coxDeBoor(k + 1, d - 1))

        return term1 + term2

    # Compute basis for each point
    b = torch.stack([coxDeBoor(k, degree) for k in range(c)], axis=1)

    b[n - 1][-1] = 1

    return b


def surf_bspline(cm, n_pts, degree=(1, 1)):
    """ Generate a 2D surface from a control matrix


    :param cm     = 2D matrix Control point Matrix
    :param n_pts  = (tuple), number of points on the curve.
    :param degree = (tuple), degree of the spline in each direction
    :return:
    """

    p, q = cm.shape

    b_p = bspline_basis(p, n_pts[0], degree[0])
    b_q = bspline_basis(q, n_pts[1], degree[1])

    Q_i = (b_p @ cm).T
    surf = (b_q @ Q_i)

    return surf


def surf_bspline_3D(cm, n_pts, degree=(1, 1, 1)):
    """ Generate a 3D surface from a control matrix

    :param cm     = 3D matrix Control point Matrix
    :param n_pts  = (tuple), number of points on the curve.
    :param degree = (tuple), degree of the spline in each direction
    :return: 3D surface of shape (n_pts[0],n_pts[1],n_pts[2])

    test:
    .. code-block:: python

        P_im = torch.rand((5,6,7),dtype=torch.float)
        img = surf_bspline_3D(P_im,(100,200,300))

    """

    p, q, r = cm.shape

    b_p = bspline_basis(p, n_pts[0], degree[0])
    b_q = bspline_basis(q, n_pts[1], degree[1])
    b_r = bspline_basis(r, n_pts[2], degree[2])

    Q_i = torch.einsum('ij,jkl->ikl', b_p, cm)
    Q_ij = torch.einsum('ij,jkl->ikl', b_q, Q_i.transpose(0, 1)).transpose(0, 1)
    surf_3d = torch.einsum('ij,jkl->ikl', b_r, Q_ij.transpose(0, 2)).transpose(0, 2)

    return surf_3d


def field2D_bspline(cms, n_pts, degree=(1, 1), dim_stack=0):
    """ Generate 2D fields from a 2D control matrix

    :param cms: shape = (2,p,q) Control matricies
    :param n_pts: (tuple) grid dimension
    :param degree: (tuple), degree of the spline in each direction
    :return: vector field of shape (2,n_pts[0],n_pts[1])
    """

    field_x = surf_bspline(cms[0], n_pts, degree)
    field_y = surf_bspline(cms[1], n_pts, degree)

    return torch.stack((field_x, field_y), dim=dim_stack)


def field3D_bspline(cms, n_pts, degree=(1, 1, 1), dim_stack=0):
    """ Generate 3D fields from a 3D control matix

    :param cms: shape = (3,p,q,r) Control matricies
    :param n_pts: (tuple) grid dimension
    :param degree: (tuple), degree of the spline in each direction
    :param dim_stack: (int) dimension to stack the field
    :return: vector field of shape (3,n_pts[0],n_pts[1],n_pts[2]) if dim_stack = 0
            vector field of shape (n_pts[0],n_pts[1],n_pts[2],3) if dim_stack = -1
    """

    field_x = surf_bspline_3D(cms[0], n_pts, degree)
    field_y = surf_bspline_3D(cms[1], n_pts, degree)
    field_z = surf_bspline_3D(cms[2], n_pts, degree)

    return torch.stack((field_x, field_y, field_z), dim=dim_stack)


# ===== CMS EXAMPLES +==================
# some interesting field control matrices
def getCMS_turn():
    cms = torch.tensor([  # control matrices
        [[0, 0, 0, 0, 0],
         [0, 1, 0, -1, 0],
         [0, 0, 0, 0, 0],
         [0, -0.25, 0, .25, 0],
         [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0],
         [0, -.25, 0, .25, 0],  # [0, .1, .5, .75, 0],
         [0, 0, 0, 0, 0],  # [0, .2, .75, 1, 0],
         [0, 1, 0, -1, 0],  # [0, .1, .5, .75, 0],
         [0, 0, 0, 0, 0]]
    ])
    return cms


def getCMS_allcombinaision():
    cms = torch.tensor([  # control matrices
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, +1, 0, -1, 0, -1, 0, -1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, +1, 0, -1, 0, +1, 0, +1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, +1, 0, +1, 0, -1, 0, +1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, -1, 0, -1, 0, -1, 0, +1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         ],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, +1, 0, +1, 0, -1, 0, +1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],  # [0, .2, .75, 1, 0],
         [0, -1, 0, -1, 0, -1, 0, +1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, +1, 0, -1, 0, -1, 0, -1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, +1, 0, -1, 0, +1, 0, +1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    ], dtype=torch.float)
    return cms


"""
import matplotlib.pyplot as plt

P_x = torch.randint(-1,4,size= (4+1,),dtype=torch.float)#[-1, 2, 0, -1]
P_y = torch.randint(-1,4,size= (4+1,),dtype=torch.float)#[2, 1, 2, -1]
cv = torch.stack((P_x,P_y)).T

n = 100
degree = 2  # Curve degree


b = bspline_basis(len(cv), n, degree)
#print(b)
points_basis = b @ cv    # 1D b-spline
# print(np.allclose(points_basis, points_scipy))


fig, ax = plt.subplots()
xx = torch.linspace(0, 3, 50)

ax.plot(points_basis[:,0],points_basis[:,1],'r--')
ax.plot(P_x,P_y,'o-')

i = 0
for x,y in zip(P_x,P_y):
    plt.annotate('$P_{:}$'.format(i),
                 (x,y))
    i += 1

ax.grid(True)
ax.legend(loc='best')
plt.show()

# ----------------
plt.figure()
plt.plot(b)
plt.title('basis functions')
plt.show()
"""
# %%
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

p,q = 5,7
cm = torch.ones((p,q))
cm[1:-1,1:-1] = torch.rand((p-2,q-2))

n_pts = (20,30) # = np.random.random((6,8))*2-1on peu mettre des tres grandes valeurs ici et l'execution est
                # encore très rapide, mais à éviter pour le plot
degree = (2,3)
surf = surf_bspline(cm,n_pts,degree)


x,y = np.meshgrid(np.linspace(0,1,p),np.linspace(0,1,q))
xx,yy = np.meshgrid(np.linspace(0,1,n_pts[0]),np.linspace(0,1,n_pts[1]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.plot_wireframe(x,y,cm)
ax.plot(x.flatten(),y.flatten(),cm.flatten().data.numpy(),'ro')

ax.plot_wireframe(xx,yy,surf.data.numpy(),color = 'gray',linewidth=0.5)
# points_2D = surf_bspline(cm,(100,120),(2,2))
plt.show()
"""
