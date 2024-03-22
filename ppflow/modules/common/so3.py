import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import quaternion_to_rotation_matrix

# Orthonormal basis of SO(3) with shape [3, 3, 3]
basis = torch.tensor([
    [[0.,0.,0.],[0.,0.,-1.],[0.,1.,0.]],
    [[0.,0.,1.],[0.,0.,0.],[-1.,0.,0.]],
    [[0.,-1.,0.],[1.,0.,0.],[0.,0.,0.]]])

# hat map from vector space R^3 to Lie algebra so(3)
def my_hat(v): return torch.einsum('...i,ijk->...jk', v, basis.to(v))

# Logarithmic map from SO(3) to R^3 (i.e. rotation vector)
#def Log(R): return torch.tensor(Rotation.from_matrix(R.numpy()).as_rotvec())

def Log(R): return matrix_to_axis_angle(R)
    
# logarithmic map from SO(3) to so(3), this is the matrix logarithm
def log(R): return my_hat(Log(R))

# Exponential map from so(3) to SO(3), this is the matrix exponential
def exp(A): return torch.linalg.matrix_exp(A)

# Exponential map from tangent space at R0 to SO(3)
def expmap(R0, tangent):
    skew_sym = pt_to_identity(R0, tangent)
    return R0 @ exp(skew_sym)

def pt_to_identity(R, v):
    return (torch.transpose(R, dim0=-2, dim1=-1) @ v)

def geodesic_distance(A, B):
    intermed = torch.einsum('bik,bkj->bij', [torch.transpose(A, 1, 2).double(), B.double()])
    pre_distance = log(intermed)
    distance = (torch.linalg.matrix_norm(pre_distance, ord='fro') / 
                torch.sqrt(torch.tensor(2)).to(pre_distance.device))
    return distance

def norm_SO3(R, T_R):
    # calulate the norm squared of matrix T_R in the tangent space of R
    r = pt_to_identity(R, T_R)                                  # matrix r is in so(3)
    norm = -torch.diagonal(r@r, dim1=-2, dim2=-1).sum(dim=-1)/2 #-trace(rTr)/2
    return norm

def hat(v: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hat operator [1] of a batch of 3D vectors.

    Args:
        v: Batch of vectors of shape `(minibatch , 3)`.

    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`

    Raises:
        ValueError if `v` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h


def matrix_to_axis_angle(matrix):
    # Check if matrix has 3 dimensions and last two dimensions have shape 3
    if len(matrix.shape) != 3 or matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

def matrix_to_quaternion(matrix):
    num_rots = matrix.shape[0]
    matrix_diag = torch.diagonal(matrix, dim1=-2, dim2=-1)
    matrix_trace = torch.sum(matrix_diag, dim=-1, keepdim=True)
    decision = torch.cat((matrix_diag, matrix_trace), dim=-1)
    choice = torch.argmax(decision, dim=-1)
    quat = torch.zeros((num_rots, 4), dtype=matrix.dtype, device=matrix.device)

    # Indices where choice is not 3
    not_three_mask = choice != 3
    i = choice[not_three_mask]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[not_three_mask, i] = (1 - decision[not_three_mask, 3] + 2 * matrix[not_three_mask, i, i])
    quat[not_three_mask, j] = (matrix[not_three_mask, j, i] + matrix[not_three_mask, i, j])
    quat[not_three_mask, k] = (matrix[not_three_mask, k, i] + matrix[not_three_mask, i, k])
    quat[not_three_mask, 3] = (matrix[not_three_mask, k, j] - matrix[not_three_mask, j, k])

    # Indices where choice is 3
    three_mask = ~not_three_mask
    quat[three_mask, 0] = (matrix[three_mask, 2, 1] - matrix[three_mask, 1, 2])
    quat[three_mask, 1] = (matrix[three_mask, 0, 2] - matrix[three_mask, 2, 0])
    quat[three_mask, 2] = (matrix[three_mask, 1, 0] - matrix[three_mask, 0, 1])
    quat[three_mask, 3] = (1 + decision[three_mask, 3])

    return _normalize_quaternion(quat)


def _normalize_quaternion(quat):
  return quat / torch.norm(quat, dim=-1, keepdim=True)

def quaternion_to_axis_angle(quat, degrees=False, eps=1e-6):
    quat = torch.where(quat[..., 3:4] < 0, -quat, quat)
    angle = 2. * torch.atan2(torch.norm(quat[..., :3], dim=-1), quat[..., 3])
    angle2 = angle * angle
    small_scale = 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880
    large_scale = angle / torch.sin(angle / 2  + eps)
    scale = torch.where(angle <= 1e-3, small_scale, large_scale)
    
    if degrees:
        scale = torch.rad2deg(scale)
    
    return scale[..., None] * quat[..., :3]


def log_rotation(R):
    trace = R[..., range(3), range(3)].sum(-1)
    if torch.is_grad_enabled():
        # The derivative of acos at -1.0 is -inf, so to stablize the gradient, we use -0.9999
        min_cos = -0.999
    else:
        min_cos = -1.0
    cos_theta = ( (trace-1) / 2 ).clamp_min(min=min_cos)
    sin_theta = torch.sqrt(1 - cos_theta**2)
    theta = torch.acos(cos_theta)
    coef = ((theta+1e-8)/(2*sin_theta+2e-8))[..., None, None]
    logR = coef * (R - R.transpose(-1, -2))
    return logR


def skewsym_to_so3vec(S):
    x = S[..., 1, 2]
    y = S[..., 2, 0]
    z = S[..., 0, 1]
    w = torch.stack([x,y,z], dim=-1)
    return w


def so3vec_to_skewsym(w):
    x, y, z = torch.unbind(w, dim=-1)
    o = torch.zeros_like(x)
    S = torch.stack([
        o, z, -y,
        -z, o, x,
        y, -x, o,
    ], dim=-1).reshape(w.shape[:-1] + (3, 3))
    return S


def exp_skewsym(S):
    x = torch.linalg.norm(skewsym_to_so3vec(S), dim=-1)
    I = torch.eye(3).to(S).view([1 for _ in range(S.dim()-2)] + [3, 3])
    
    sinx, cosx = torch.sin(x), torch.cos(x)
    b = (sinx + 1e-8) / (x + 1e-8)
    c = (1-cosx + 1e-8) / (x**2 + 2e-8)  # lim_{x->0} (1-cosx)/(x^2) = 0.5

    S2 = S @ S
    return I + b[..., None, None]*S + c[..., None, None]*S2


def so3vec_to_rotation(w):
    return exp_skewsym(so3vec_to_skewsym(w))


def rotation_to_so3vec(R):
    logR = log_rotation(R)
    w = skewsym_to_so3vec(logR)
    return w


def random_uniform_so3(size, device='cpu'):
    q = F.normalize(torch.randn(list(size)+[4,], device=device), dim=-1)    # (..., 4)
    return rotation_to_so3vec(quaternion_to_rotation_matrix(q))


class ApproxAngularDistribution(nn.Module):

    def __init__(self, stddevs, std_threshold=0.1, num_bins=8192, num_iters=1024):
        super().__init__()
        self.std_threshold = std_threshold
        self.num_bins = num_bins
        self.num_iters = num_iters
        self.register_buffer('stddevs', torch.FloatTensor(stddevs))
        self.register_buffer('approx_flag', self.stddevs <= std_threshold)
        self._precompute_histograms()

    @staticmethod
    def _pdf(x, e, L):
        """
        Args:
            x:  (N, )
            e:  Float
            L:  Integer
        """
        x = x[:, None]  # (N, *)
        c = ((1 - torch.cos(x)) / math.pi)  # (N, *)
        l = torch.arange(0, L)[None, :]  # (*, L)
        a = (2*l+1) * torch.exp(-l*(l+1)*(e**2))  # (*, L)
        b = (torch.sin( (l+0.5)* x ) + 1e-6) / (torch.sin( x / 2 ) + 1e-6) # (N, L)
        
        f = (c * a * b).sum(dim=1)
        return f

    def _precompute_histograms(self):
        X, Y = [], []
        for std in self.stddevs:
            std = std.item()
            x = torch.linspace(0, math.pi, self.num_bins)   # (n_bins,)
            y = self._pdf(x, std, self.num_iters)    # (n_bins,)
            y = torch.nan_to_num(y).clamp_min(0)
            X.append(x)
            Y.append(y)
        self.register_buffer('X', torch.stack(X, dim=0))  # (n_stddevs, n_bins)
        self.register_buffer('Y', torch.stack(Y, dim=0))  # (n_stddevs, n_bins)

    def sample(self, std_idx):
        """
        Args:
            std_idx:  Indices of standard deviation.
        Returns:
            samples:  Angular samples [0, PI), same size as std.
        """
        size = std_idx.size()
        std_idx = std_idx.flatten() # (N,)
        
        # Samples from histogram
        prob = self.Y[std_idx]  # (N, n_bins)
        bin_idx = torch.multinomial(prob[:, :-1], num_samples=1).squeeze(-1)    # (N,)
        bin_start = self.X[std_idx, bin_idx]    # (N,)
        bin_width = self.X[std_idx, bin_idx+1] - self.X[std_idx, bin_idx]
        samples_hist = bin_start + torch.rand_like(bin_start) * bin_width    # (N,)

        # Samples from Gaussian approximation
        mean_gaussian = self.stddevs[std_idx]*2
        std_gaussian = self.stddevs[std_idx]
        samples_gaussian = mean_gaussian + torch.randn_like(mean_gaussian) * std_gaussian
        samples_gaussian = samples_gaussian.abs() % math.pi

        # Choose from histogram or Gaussian
        gaussian_flag = self.approx_flag[std_idx]
        samples = torch.where(gaussian_flag, samples_gaussian, samples_hist)

        return samples.reshape(size)


def random_normal_so3(std_idx, angular_distrib, device='cpu'):
    size = std_idx.size()
    u = F.normalize(torch.randn(list(size)+[3,], device=device), dim=-1)
    theta = angular_distrib.sample(std_idx)
    w = u * theta[..., None]
    return w
