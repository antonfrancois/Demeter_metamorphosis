import torch

from .toolbox import update_progress



class GradientDescent(torch.optim.Optimizer):

    def __init__(self,loss,x_0,lr = 1e-6,gamma=None):
        """

        :param loss: already intialised loss function to perform the gradient descent
        """
        self.loss = loss
        self.x = x_0
        if lr == 'auto':
            self.auto = True
            self.gamma = .5 if gamma is None else gamma
        else:
            self.auto = False
            self.gamma = lr # step multiplicator

    def step(self):
        """
        One gradient descent step,

        :return: the loss value at this step.
        """
        loss_val = self.loss(self.x)
        loss_val.backward(retain_graph=False)

        if self.auto:
            self.backtracking_search(loss_val)

        self.x.data = self.x.data - self.gamma*self.x.grad.data
        self.x.grad.zero_()
        return loss_val


    def __call__(self,x_0=None,n_iter = 200):
        """

        :param x_0: initialisation of the gradient descent
        :param lr: step of the gradient
        :param n_iter: number of iteration
        :return:
        """
        if not x_0 is None:
            self.x = x_0

        loss_stock = torch.zeros(n_iter)

        for i in range(n_iter):
            loss_stock[i] = self.step()

            update_progress((i+1)/n_iter)

        return self.x, loss_stock

    def backtracking_search(self,loss_val):
        new_gamma = 2*self.gamma

        norm_grad = torch.sqrt((self.x.grad.data**2).sum())
        # it = 0
        # print(" \nENTER Backtrack")
        # print(self.loss(self.x - new_gamma*self.x.grad.data) > (loss_val - 0.1*new_gamma*norm_grad).item())
        while (self.loss(self.x - new_gamma*self.x.grad.data) > (loss_val - 0.1*new_gamma*norm_grad)).item():
#             # print('gamma :',new_gamma)
            # print(self.loss(self.x.data) -self.loss(self.x.data-new_gamma*self.x.grad.data))
            new_gamma *= .5
            # it += 1
        # print('it :', it)
        # print('f(x - g) ',self.loss(self.x - new_gamma*self.x.grad.data).item(),
        #       'f(x) - g',loss_val - 0.1*new_gamma*norm_grad.item())
        # print('bc_s : new_gamma',new_gamma,'norm_grad',norm_grad)
        # print("OUT  Backtrack\n")
        self.gamma = new_gamma


"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n_pts = (200,300)
xx,yy = torch.meshgrid(torch.linspace(-2,2,n_pts[0]),torch.linspace(-6,6,n_pts[1]))

def func(X):
    xx,yy = X
    return torch.sqrt(torch.abs(xx))*torch.atan(10*xx**2)+torch.cos(yy)*2*yy

F = func((xx,yy))

# on va maintenant chercher le minimum de surf
x =torch.tensor((1.5,6),dtype=torch.float,requires_grad=True)

gd = GradientDescent(func,x,lr='auto')

n_iter = 100
loss_val_stock = torch.zeros(n_iter)

x_stock = torch.zeros((2,n_iter))

for t in range(n_iter):
    x_stock[:,t] = gd.x.data
    loss_val_stock[t] = gd.step()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.plot_wireframe(x,y,cm)
ax.plot(x_stock[0,:].detach().numpy(),
        x_stock[1,:].detach().numpy(),
        loss_val_stock.detach().numpy(),
        'ro--')

ax.plot_wireframe(xx,yy,F.data.numpy(),color = 'gray',linewidth=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
# points_2D = surf_bspline(cm,(100,120),(2,2))
plt.show()
# """
