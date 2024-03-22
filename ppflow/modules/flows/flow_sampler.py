import os
os.environ["GEOMSTATS_BACKEND"] = "pytorch"
os.environ["GEOMSTATS_DEVICE"] = "cuda"
import torch 
from scipy.spatial.transform import Rotation
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from einops import rearrange
from functorch import vmap
from ..common.so3 import * 
from ..common.so2 import *
from ..common.layers import clampped_one_hot

from geomstats._backend import _backend_config as _config
### IMPORTANT!
# torch.set_default_tensor_type("torch.cuda.FloatTensor")
torch.set_default_dtype(torch.float32)
_config.DEFAULT_DTYPE = torch.cuda.FloatTensor 

def riemannian_gradient(f, R):
    coefficients = torch.zeros(list(R.shape[:-2])+[3], requires_grad=True).to(R.device)
    R_delta  = expmap(R, R @ hat(coefficients))
    grad_coefficients = torch.autograd.grad(f(R_delta).sum(), coefficients, )[0]
    return R @ hat(grad_coefficients)

class SO3FlowSampler(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.manifold = SpecialOrthogonal(n=3, point_type="matrix")
        self.basic_sampler = SO3ConditionalFlowMatcher(manifold=self.manifold)
    
    def sample_field(self, x1, t, mask=None):
        '''
        x1: [B, ..., 3, 3] rotation matrix
        t: [B,] time
        '''
        x0 = torch.tensor(Rotation.random(x1.size(0)).as_matrix()).to(x1)
        
        if mask is not None:
            mask = mask.reshape(-1, *([1] * (x0.dim() - 1)))
            x0 = torch.where(mask, x0, x1)
        
        t, ut, xt = self.basic_sampler.sample_location_and_conditional_flow(x0, x1, time_der=True, t=t)
        if mask is not None:
            xt = torch.where(mask, xt, x1)
            ut = torch.where(mask, ut, torch.zeros_like(ut))
        return ut, xt
    
    def inference(self, xt, vx_t, dt, mask=None):
        dt = dt.reshape(-1, *([1] * (xt.dim() - 1)))
        x_new = expmap(xt, vx_t * dt)
        x_new = self.manifold.rotation_vector_from_matrix(x_new)
        x_new = self.manifold.matrix_from_rotation_vector(x_new)
        if mask is not None:
            x_new = torch.where(mask, x_new, xt)
        return x_new

class R3FlowSampler(nn.Module):
    def __init__(self, sigma=0.01) -> None:
        super().__init__()
        self.sigma = sigma 

    def inference(self, xt, vx_t, dt, mask=None):
        dt = dt.reshape(-1, *([1] * (xt.dim() - 1)))
        x_new = xt + vx_t * dt
        if mask is not None:
            x_new = torch.where(mask, x_new, xt)
        return x_new

    def sample_field(self, x1, t, mask=None):
        '''
        x1: [B, ..., 3] translation vector
        t: [B,] time
        '''
        x0 = torch.randn_like(x1).to(x1)
        
        if mask is not None:
            mask = mask.reshape(-1, *([1] * (x0.dim() - 1)))
            x0 = torch.where(mask, x0, x1)
        
        t, ut, xt = self.sample_location_and_conditional_flow(x0, x1, t=t)
        if mask is not None:
            xt = torch.where(mask, xt, x1)
            ut = torch.where(mask, ut, torch.zeros_like(ut))

        return ut, xt
    
    def sample_location_and_conditional_flow(self, x0, x1, t=None):
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0).to(x0.device)

        xt = self.sample_conditional_xt(x0, x1, t, sigma=self.sigma)
        ut = self.compute_conditional_vector_field(x0, x1)
        return t, ut, xt


    def sample_conditional_xt(self, x0, x1, t, sigma):
        """
        Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        xt : Tensor, shape (bs, *dim)

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        t = t.reshape(-1, *([1] * (x0.dim() - 1)))
        mu_t = t * x1 + (1 - t) * x0
        epsilon = torch.randn_like(x0)
        return mu_t + sigma * epsilon
    
    def compute_conditional_vector_field(self, x0, x1):
        """
        Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        return x1 - x0

class TorusFlowSampler(nn.Module):
    def __init__(self, sigma=0.0) -> None:
        super().__init__()
        self.sigma = sigma 

    def inference(self, xt, vx_t, dt, mask=None):
        dt = dt.reshape(-1, *([1] * (xt.dim() - 1)))
        x_new = xt + vx_t * dt
        if mask is not None:
            x_new = torch.where(mask, x_new, xt)
        return regularize(x_new)
    

    def sample_field(self, x1, t, mask=None):
        '''
        x1: [B, ..., 3] translation vector
        t: [B,] time
        '''
        x0 = torch.randn_like(x1).to(x1)
        x0 = regularize(x0)
        
        if mask is not None:
            # mask = mask.reshape(-1, *([1] * (x0.dim() - 1)))
            x0 = torch.where(mask, x0, x1)
        
        t, ut, xt = self.sample_location_and_conditional_flow(x0, x1, t=t)
        if mask is not None:
            xt = torch.where(mask, xt, x1)
            ut = torch.where(mask, ut, torch.zeros_like(ut))

        return ut, xt
    
    def sample_location_and_conditional_flow(self, x0, x1, t=None):
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0).to(x0.device)

        xt = self.sample_conditional_xt(x0, x1, t, sigma=self.sigma)
        ut = self.compute_conditional_vector_field(x0, x1)
        return t, ut, xt


    def sample_conditional_xt(self, x0, x1, t, sigma):
        """
        Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        xt : Tensor, shape (bs, *dim)

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        t = t.reshape(-1, *([1] * (x0.dim() - 1)))
        geodesics, x1 = geodesic_so2(x0, x1)
        mu_t = t * x1 + (1 - t) * x0
        epsilon = torch.randn_like(x0)
        return mu_t + sigma * epsilon
    
    def compute_conditional_vector_field(self, x0, x1):
        """
        Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        return x1 - x0

    
class TypeFlowSampler(nn.Module):
    def __init__(self, sigma=0.0, num_classes=20) -> None:
        super().__init__()
        self.sigma = sigma 
        self.num_classes = num_classes

    def inference(self, xt, ct, vc_t, dt, mask):
        dt = dt.reshape(-1, *([1] * (ct.dim() - 1)))

        c_new = ct + vc_t * dt
        
        x_new = self._sample(c_new.clamp(min=0, max=1))
        if mask is not None:
            c_new = torch.where(mask.unsqueeze(-1), c_new, ct)
            x_new = torch.where(mask, x_new, xt)

        return x_new, c_new

    @staticmethod
    def _sample(c):
        """
        Args:
            c:    (N, L, K).
        Returns:
            x:    (N, L).
        """
        N, L, K = c.size()
        c = c.view(N*L, K) + 1e-8
        x = torch.multinomial(c, 1).view(N, L)
        return x

    def sample_field(self, x1, t, mask=None):
        '''
        x1: [B, ..., 3] translation vector
        t: [B,] time
        '''
        N, L = x1.shape
        K = self.num_classes
        c1 = clampped_one_hot(x1, num_classes=K).float()
        c0 = torch.ones_like(c1).to(c1) / K
        
        if mask is not None:
            mask = mask.reshape(N, L, *([1] * (c0.dim() - 2)))
            c0 = torch.where(mask, c0, c1)
        
        t, ut, ct = self.sample_location_and_conditional_flow(c0, c1, t=t)
        if mask is not None:
            ct = torch.where(mask, ct, c1)
            ut = torch.where(mask, ut, torch.zeros_like(ut))

        xt = self._sample(ct)
        return ut, xt, ct
    
    def sample_location_and_conditional_flow(self, c0, c1, t=None):
        if t is None:
            t = torch.rand(c0.shape[0]).type_as(c0).to(c0.device)

        ct = self.sample_conditional_ct(c0, c1, t, sigma=self.sigma)
        ut = self.compute_conditional_vector_field(c0, c1)
        return t, ut, ct

    def posterior(self, ct, uc_t, t):
        """
        Returns:
            theta:  Posterior probability at (t=1) step, (N, L, K).
        """
        K = self.num_classes
        t = t.reshape(-1, *([1] * (ct.dim() - 1)))
        if ct.dim() == 3:
            ct = ct   # When x_t is probability distribution.
        else:
            ct = clampped_one_hot(ct, num_classes=K).float() # (N, L, K)

        theta = ct + uc_t * (1-t)   # (N, L, K)
        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)
        return theta


    def sample_conditional_ct(self, c0, c1, t, sigma):
        """
        Draw a sample from the probability path N(t * c1 + (1 - t) * c0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        c0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        c1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        ct : Tensor, shape (bs, *dim)

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        t = t.reshape(-1, *([1] * (c0.dim() - 1)))
        mu_t = t * c1 + (1 - t) * c0
        epsilon = torch.randn_like(c0)
        return mu_t + sigma * epsilon
    
    def compute_conditional_vector_field(self, c0, c1):
        """
        Compute the conditional vector field ut(c1|c0) = c1 - c0, see Eq.(15) [1].

        Parameters
        ----------
        c0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        c1 : Tensor, shape (bs, *dim)
            represents the target minibatch

        Returns
        -------
        ut : conditional vector field ut(c1|c0) = c1 - c0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        return c1 - c0

class SO3ConditionalFlowMatcher:
    """
    Class to compute the FoldFlow-base method. It is the parent class for the 
    FoldFlow-OT and FoldFlow-SFM methods. For sake of readibility, the doc of 
    most function only describes the function purpose. The documentation 
    describing all inputs can be found in the 
    sample_location_and_conditional_flow class.
    """

    def __init__(self, manifold):
        self.sigma = None
        self.manifold = manifold
        self.vec_manifold = SpecialOrthogonal(n=3, point_type="vector")
    
    def vec_log_map(self, x0, x1, if_matrix_format=False):
        """
        Function which compute the SO(3) log map efficiently.
        """
        # get logmap of x_1 from x_0
        # convert to axis angle to compute logmap efficiently
        if if_matrix_format:
            rot_x0 = matrix_to_axis_angle(x0) 
            rot_x1 = matrix_to_axis_angle(x1)
        else:
            rot_x0 = x0
            rot_x1 = x1
        rot_x1 = rot_x1.to(rot_x0.device)
        log_x1 = self.vec_manifold.log_not_from_identity(rot_x1, rot_x0)
        return log_x1, rot_x0
        
    def sample_xt(self, x0, x1, t, if_matrix_format=False):
        """
        Function which compute the sample xt along the geodesic from x0 to x1 on SO(3).
        """
        # sample along the geodesic from x0 to x1
        log_x1, rot_x0 = self.vec_log_map(x0, x1, if_matrix_format=if_matrix_format)
        # group exponential at x0
        xt = self.vec_manifold.exp_not_from_identity(t.reshape(-1, 1) * log_x1, rot_x0)
        xt = self.vec_manifold.matrix_from_rotation_vector(xt)
        return xt

    def compute_conditional_flow_simple(self, t, xt):
        """
        Function which computes the vector field through the sample xt's time derivative
        for simple manifold.
        """
        xt = rearrange(xt, 'b c d -> b (c d)', c=3, d=3)
        def index_time_der(i):
            return torch.autograd.grad(xt, t, i, create_graph=True, retain_graph=True)[0]
        xt_dot = vmap(index_time_der, in_dims=1)(torch.eye(9).to(xt.device).repeat(xt.shape[0], 1, 1))
        return rearrange(xt_dot, '(c d) b -> b c d', c=3, d=3)
    
    def compute_conditional_flow(self, xt, x0, x1, t):
        """
        Function which computes the general vector field for k(t) = 1-t.
        """
        # compute the geodesic distance
        dist_x0_x1 = geodesic_distance(x0, x1)                       # d(x0, x1)
        geo_dist = lambda x: geodesic_distance(x, x1)
        dist_grad_wrt_xt = riemannian_gradient(geo_dist, xt) # nabla_xt d(xt, x1)

        # Compute the geodesic norm ||.||_g:
        denom_term = norm_SO3(xt, dist_grad_wrt_xt)

        output = -dist_x0_x1[:, None, None] * dist_grad_wrt_xt / denom_term[:, None, None]
        return output #* 2 * t[:, None, None]
        
    def sample_location_and_conditional_flow(self, x0, x1, time_der=True, t=None):
        """
        Compute the sample xt along the geodesic from x0 to x1 (see Eq.2 [1])
        and the conditional vector field ut(xt|z). The coupling q(x0,x1) is
        the independent coupling.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        time_der : bool
            ut computed through time derivative


        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn along the geodesic
        ut : conditional vector field ut(xt|z)

        References
        ----------
        [1] SE(3)-Stochastic Flow Matching for Protein Backbone Generation, Bose et al.
        [2] Riemannian Flow Matching on General Geometries, Chen et al.
        """
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0).to(x0.device)
            t.requires_grad = True

        xt = self.sample_xt(x0, x1, t, if_matrix_format=True)
        if time_der:
            delta_r = torch.transpose(x0, dim0=-2, dim1=-1) @ xt
            ut = xt @ log(delta_r)/t[:, None, None]
            # Above is faster than taking the time derivative like in [2]
            # ut = self.compute_conditional_flow_simple(t, xt)
        else:
            # Compute general vector field like in [2] 
            ut = self.compute_conditional_flow(xt, x0, x1, t)
        return t, ut.float(), xt.float()