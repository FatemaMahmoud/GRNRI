import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable


def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input, -1)
    return soft_max_1d.transpose(axis, 0)


def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return my_softmax(y / tau, axis=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def load_data(tfs, batch_size=1, suffix='', compute_acc=True):
    
     
    data_train = np.load('data/train' + suffix + '.npy', allow_pickle=True)

    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_atoms = data_train.shape[3]

    data_max = data_train.max()
    data_min = data_train.min()

    # Normalize to [-1, 1]
    data_train = (data_train - data_min) * 2 / (data_max - data_min) - 1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    feat_train = np.transpose(data_train, [0, 3, 1, 2])
    
    feat_train = torch.FloatTensor(feat_train)

    # Exclude self edges
    off_diag = np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)
    
    off_diag = off_diag[:tfs]
    
    off_diag_idx = np.ravel_multi_index(np.where(off_diag), [tfs, num_atoms])
    
    
    
    if (compute_acc):
        edges_train = np.load('data/edges_train' + suffix + '.npy')
        edges_train = np.reshape(edges_train, [-1, tfs * num_atoms])
        edges_train = torch.ByteTensor(edges_train)
        edges_train = edges_train[:, off_diag_idx]
        train_data = TensorDataset(feat_train, edges_train)
    else:
        train_data = TensorDataset(feat_train, feat_train)
        
    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    
    return train_data_loader, data_max, data_min


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes), dtype=bool)[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=bool)
    return labels_onehot


def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))


def kl_categorical_uniform(preds, num_atoms, num_edge_types, add_const=False,
                           eps=1e-16):
    kl_div = preds * torch.log(preds + eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))

def nll_gaussian(preds, target, variance, add_const=False):
    
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    if(len(target.shape) == 2):
        return neg_log_p.sum() / (target.size(0) * target.size(1))
    else:
        return neg_log_p.sum() / (target.size(0))


def edge_accuracy(preds, target):
    _, preds = preds.max(-1)
    correct = preds.float().data.eq(
        target.float().data.view_as(preds)).cpu().sum()
    return np.float(correct) / (target.size(0) * target.size(1))

