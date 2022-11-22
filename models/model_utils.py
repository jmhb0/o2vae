import torch

def kl_normal(qm, qv, pm, pv):
    """
    Method: 2021 Rui Shu
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension
    Args:
        qm: tensor: (batch, dim): q mean
        qv: tensor: (batch, dim): q variance
        pm: tensor: (batch, dim): p mean
        pv: tensor: (batch, dim): p variance
    Return:
        kl: tensor: (batch,): kl between each sample
    """
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl

def sample_gaussian(mu, logvar):
    """ Sample the isotropic Gaussian N(0,1) """
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(mu)
    return eps * std + mu

def log_normal(x, m, v):
    """
    Computes the elem-wise log probability of a Gaussian and then sum over the
    last dim. Basically we're assuming all dims are batch dims except for the
    last dim.
    Args:
        x: tensor: (batch_1, batch_2, ..., batch_k, dim): Observation
        m: tensor: (batch_1, batch_2, ..., batch_k, dim): Mean
        v: tensor: (batch_1, batch_2, ..., batch_k, dim): Variance
    Return:
        log_prob: tensor: (batch_1, batch_2, ..., batch_k): log probability of
            each sample. Note that the summation dimension is not kept
    """
    log_prob_batch = -0.5* (torch.log(v) + (x-m).pow(2)/v + np.log(2*np.pi))
    log_prob = log_prob_batch.sum(-1)

    return log_prob

def log_normal_mixture(z, m, v):
    """
    Computes log probability of Gaussian mixture, where we assume a probability
    of each mixture component. This could be updated to also accept a tensor, k, 
    (batch,mix,1) of mixture probabilities.
    Args:
        z: tensor: (batch, dim): Observations
        m: tensor: (batch, mix, dim): Mixture means
        v: tensor: (batch, mix, dim): Mixture variances
    Return:
        log_prob: tensor: (batch,): log probability of each sample
    """
    # duplicate z for each
    assert m.ndim==3
    b, k, dim = m.shape
    # duplicate z along the k dimension for each batch
    z = z.unsqueeze(1).expand(b, k, dim)

    # get the log probability for each k, and then sum over the k components.
    log_prob_batches = log_normal(z, m, v) # output shape is (b,k)
    log_prob = log_mean_exp(log_prob_batches, dim=-1) # logsumexp over k

    return log_prob

def log_sum_exp(x, dim=0):
    """
    From: 2021 Rui Shu
    Compute the log(sum(exp(x), dim)) in a numerically stable manner
    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which sum is computed
    Return:
        _: tensor: (...): log(sum(exp(x), dim))
    """
    max_x = torch.max(x, dim)[0]
    new_x = x - max_x.unsqueeze(dim).expand_as(x)
    return max_x + (new_x.exp().sum(dim)).log()

def log_mean_exp(x, dim):
    """
    From: 2021 Rui Shu
    Compute the log(mean(exp(x), dim)) in a numerically stable manner
    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which mean is computed
    Return:
        _: tensor: (...): log(mean(exp(x), dim))
    """
    return log_sum_exp(x, dim) - np.log(x.size(dim))

def gaussian_parameters(h, dim=-1):
    """
    From: 2021 Rui Shu
    Converts generic real-valued representations into mean and variance
    parameters of a Gaussian distribution
    Args:
        h: tensor: (batch, ..., dim, ...): Arbitrary tensor
        dim: int: (): Dimension along which to split the tensor for mean and
            variance
    Returns:
        m: tensor: (batch, ..., dim / 2, ...): Mean
        v: tensor: (batch, ..., dim / 2, ...): Variance
    """
    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v

def duplicate(x, rep):
    """
    Duplicates x along dim=0
    Args:
        x: tensor: (batch, ...): Arbitrary tensor
        rep: int: (): Number of replicates. Setting rep=1 returns orignal x
    Returns:
        _: tensor: (batch * rep, ...): Arbitrary replicated tensor
    """
    return x.expand(rep, *x.shape).reshape(-1, *x.shape[1:])


