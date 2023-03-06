import torch

def f_mean(a):
    return a.mean(dim=0)

def f_max(a):
    return a.max(dim=0).values

def f_cat(a):
    return torch.cat((a[0], a[-1]), dim=0)

def f_diffsum(a):
    return torch.cat((a[0] + a[-1], a[-1] - a[0]), dim=0)


def span_reps(h_e, r_e, t_e, method):

    # L x D
    if method == 'mean':
        return f_mean(h_e), f_mean(r_e), f_mean(t_e)
    elif method == 'max':
        return f_max(h_e), f_max(r_e), f_max(t_e)
    elif method == 'cat':
        return f_cat(h_e), f_cat(r_e), f_cat(t_e)
    elif method == 'diffsum':
        return f_diffsum(h_e), f_diffsum(r_e), f_diffsum(t_e)
    else:
        raise NotImplementedError


def span_reps_static(e, method):

    # L x D
    if method == 'mean':
        return f_mean(e)
    elif method == 'max':
        return f_max(e)
    elif method == 'cat':
        return f_cat(e)
    elif method == 'diffsum':
        return f_diffsum(e)
    else:
        raise NotImplementedError