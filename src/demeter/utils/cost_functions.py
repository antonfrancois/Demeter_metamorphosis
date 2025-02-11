import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from . import torchbox as tb


class SumSquaredDifference:
    """ Compute the sum squared difference of two images """

    def __init__(self, target, cancer_seg=None,cancer_on_target=True):
        # self.source =source
        self.target = target
        # print('posterior init : source '+ str(source.shape))
        self.field_size = target.shape[-2:]
        # self.field_size = (source.shape[3],source.shape[2])
        self.cancer_on_target = cancer_on_target
        self.cancer_seg = cancer_seg  #mask # TODO : verifier que masks size = [N,0,H,W]

    def __call__(self,source_deform=None):
        return self.function(source_deform)

    def function(self,source_deform):
        """

        :param field_diff: tensor (H,W,2)
        :return:
        """
        self.source_deform = source_deform
        # print('poterior function : source_deform '+str(self.source_deform.shape))
        if self.cancer_seg is None:
            return .5*((self.target - self.source_deform)**2).sum()
        elif self.cancer_on_target:
            return .5*(self.cancer_seg* (self.target - self.source_deform)**2).sum()
        else:
            print("WARNING : la formule est fausse, corrige ça Anton du futur")
            return .5*((self.target - self.cancer_seg * self.source_deform)**2).sum()


    def imgShow(self,axes = None):
        if axes is None:
            fig,axes = plt.subplots(1,2)
        axes[0].imshow(tb.imCmp(self.source,self.target))
        axes[0].set_title('before')

        axes[1].imshow(tb.imCmp(self.source_deform,self.target))
        axes[1].set_title('after registration')

        # plt.show()
        return axes

class Combine_ssd_CFM:

    def __init__(self,ssd_target,cfm_target,seg,cancer_on_target = True):
        self.B_ssd,self.B_cfm = ssd_target.shape[0],cfm_target.shape[0]
        self.ssd = SumSquaredDifference(ssd_target)
        self.cfm = SumSquaredDifference(cfm_target,seg,cancer_on_target=cancer_on_target)

    def __call__(self,img_ssd,img_cfm):
        if img_ssd.shape[0] != self.B_ssd:
            raise ValueError(f"The ssd comparison was initialized with batch size {self.B_ssd} "
                             f"and img_ssd is shape : {img_ssd}")
        if img_cfm.shape[0] != self.B_cfm:
            raise ValueError(f"The cfm comparison was initialized with batch size {self.B_cfm} "
                             f"and img_cfm is shape : {img_cfm}")

        return self.ssd(img_ssd)/self.B_ssd + self.cfm(img_cfm)/self.B_cfm



#    ========= MUTUAL INFORMATION =========
# Differentiable approximation to the mutual information (MI) metric.
# Implementation in PyTorch
#
# This is an adaptation of the implementation found at:
#  https://github.com/KrakenLeaf/pytorch_differentiable_mutual_info

# Note: This code snippet was taken from the discussion found at:
#               https://discuss.pytorch.org/t/differentiable-torch-histc/25865/2
# By Tony-Y
class SoftHistogram1D(nn.Module):
    '''
    Differentiable 1D histogram calculation (supported via pytorch's autograd)
    inupt:
    x     - N x D array, where N is the batch size and D is the length of each data series
    bins  - Number of bins for the histogram
    min   - Scalar min value to be included in the histogram
    max   - Scalar max value to be included in the histogram
    sigma - Scalar smoothing factor fir the bin approximation via sigmoid functions.
          Larger values correspond to sharper edges, and thus yield a more accurate approximation
    output:
    N x bins array, where each row is a histogram
    '''

    def __init__(self, bins=50, min=0, max=1, sigma=10):
        super(SoftHistogram1D, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)  # Bin centers
        self.centers = nn.Parameter(self.centers, requires_grad=False)  # Wrap for allow for cuda support

    def forward(self, x):
        # Replicate x and for each row remove center
        x = torch.unsqueeze(x, 1) - torch.unsqueeze(self.centers, 1)

        # Bin approximation using a sigmoid function
        x = torch.sigmoid(self.sigma * (x + self.delta / 2)) - torch.sigmoid(self.sigma * (x - self.delta / 2))

        # Sum along the non-batch dimensions
        x = x.sum(dim=-1)
        # x = x / x.sum(dim=-1).unsqueeze(1)  # normalization
        return x


# Note: This is an extension to the 2D case of the previous code snippet
class SoftHistogram2D(nn.Module):
    """
    Differentiable 1D histogram calculation (supported via pytorch's autograd)
    Parameters:
    -------------
    x, y  - N x D array, where N is the batch size and D is the length of each data series
         (i.e. vectorized image or vectorized 3D volume)
    bins  - Number of bins for the histogram
    min   - Scalar min value to be included in the histogram
    max   - Scalar max value to be included in the histogram
    sigma - Scalar smoothing factor fir the bin approximation via sigmoid functions.
          Larger values correspond to sharper edges, and thus yield a more accurate approximation
    Returns:
    -------------
        N x bins array, where each row is a histogram
    """

    def __init__(self, bins=50, min=0, max=1, sigma=10):
        super(SoftHistogram2D, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)  # Bin centers
        self.centers = nn.Parameter(self.centers, requires_grad=False)  # Wrap for allow for cuda support

    def forward(self, x, y):
        assert x.size() == y.size(), "(SoftHistogram2D) x and y sizes do not match"

        # Replicate x and for each row remove center
        x = torch.unsqueeze(x, 1) - torch.unsqueeze(self.centers, 1)
        y = torch.unsqueeze(y, 1) - torch.unsqueeze(self.centers, 1)

        # Bin approximation using a sigmoid function (can be sigma_x and sigma_y respectively - same for delta)
        x = torch.sigmoid(self.sigma * (x + self.delta / 2)) - torch.sigmoid(self.sigma * (x - self.delta / 2))
        y = torch.sigmoid(self.sigma * (y + self.delta / 2)) - torch.sigmoid(self.sigma * (y - self.delta / 2))

        # Batched matrix multiplication - this way we sum jointly
        z = torch.matmul(x, y.permute((0, 2, 1)))
        return z


class Mutual_Information(nn.Module):
    r"""
    This class is a pytorch implementation of the mutual information (MI) calculation between two images.
    This is an approximation, as the images' histograms rely on differentiable approximations of rectangular windows.

    .. math::
            I(X, Y) = H(X) + H(Y) - H(X, Y) = \sum(\sum(p(X, Y) * log(p(Y, Y)/(p(X) * p(Y)))))

    where $H(X) = -\sum(p(x) * log(p(x)))$ is the entropy
    """

    def __init__(self, bins=50, min=0, max=1, sigma=10, reduction='sum'):
        super(Mutual_Information, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.reduction = reduction

        # 2D joint histogram
        self.hist2d = SoftHistogram2D(bins, min, max, sigma)

        # Epsilon - to avoid log(0)
        self.eps = torch.tensor(0.00000001, dtype=torch.float32, requires_grad=False)

    def independence_plot(self, joinedLaw_xy,marginal_x,marginal_y):
        fig, ax = plt.subplots(1, 1)
        ax.imshow(((joinedLaw_xy)/(marginal_x*marginal_y))[0].detach().numpy(), cmap='Spectral')
        plt.show()

    def forward(self, im1, im2, plot=False):
        """
        Forward implementation of a differentiable MI estimator for batched images
        :param im1: N x ... tensor, where N is the batch size
                    ... dimensions can take any form, i.e. 2D images or 3D volumes.
        :param im2: N x ... tensor, where N is the batch size
        :return: N x 1 vector - the approximate MI values between the batched im1 and im2
        """

        # Check for valid inputs
        assert im1.size() == im2.size(), "(MI_pytorch) Inputs should have the same dimensions."

        batch_size = im1.size()[0]

        # Flatten tensors
        # im1_flat = im1.view(im1.size()[0], -1)
        # im2_flat = im2.view(im2.size()[0], -1)
        im1_flat = im1.reshape((im1.shape[0],-1))
        im2_flat = im2.reshape((im2.shape[0],-1))


        # Calculate joint histogram
        hgram = self.hist2d(im1_flat, im2_flat)

        # Convert to a joint distribution
        # Pxy = torch.distributions.Categorical(probs=hgram).probs
        p_xy = torch.div(hgram, torch.sum(hgram.view(hgram.size()[0], -1)))

        # Calculate the marginal distributions
        marginal_y = torch.sum(p_xy, dim=1).unsqueeze(1)
        marginal_x = torch.sum(p_xy, dim=2).unsqueeze(1)

        # Use the KL divergence distance to calculate the MI
        joinedLaw_xy = torch.matmul(marginal_x.permute((0, 2, 1)), marginal_y)

        # Reshape to batch_size X all_the_rest
        p_xy = p_xy.reshape(batch_size, -1)
        joinedLaw_xy = joinedLaw_xy.reshape(batch_size, -1)

        if plot:
            self.independence_plot(joinedLaw_xy, marginal_x, marginal_y)

        # Calculate mutual information - this is an approximation due to the histogram calculation and eps,
        # but it can handle batches
        if batch_size == 1:
            # No need for eps approximation in the case of a single batch
            nzs = p_xy > 0  # Calculate based on the non-zero values only
            mut_info = torch.matmul(
                p_xy[nzs],
                torch.log(p_xy[nzs]) - torch.log(joinedLaw_xy[nzs])
            )  # MI calculation
        else:
            # For arbitrary batch size > 1
            mut_info = torch.sum(p_xy * (torch.log(p_xy + self.eps) - torch.log(joinedLaw_xy + self.eps)), dim=1)

        # Reduction
        if self.reduction == 'sum':
            mut_info = torch.sum(mut_info)
        elif self.reduction == 'batchmean':
            mut_info = torch.sum(mut_info)
            mut_info = mut_info / float(batch_size)

        return mut_info


#    ========= END MUTUAL INFORMATION =========