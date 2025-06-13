#!/usr/bin/env python
import torch.nn.functional as F
import torch.nn as nn



#def calc_vq_loss(vertice_out, vertice, expr_out, expr, quant_loss, quant_loss_weight=1.0, alpha=1.0):
def calc_vq_loss(expr_out, expr, quant_loss, quant_loss_weight=1.0, alpha=1.0):
    """ function that computes the various components of the VQ loss """
    ## Loss for the vertices
    #rec_loss = nn.L1Loss()(vertice_out, vertice)

    ## loss for the expression vector
    #expr_rec_loss = nn.L1Loss()(expr_out, expr)
    total_blendshapes_loss = nn.MSELoss()(expr_out[:, :-2], expr[:, :-2]) #Avoid Eyelids comparison (not in ensemble dataset)
    # Define chunks
    
    # slice(56, 58) is discarded

    # Compute per-chunk losses
    #expr_rec_loss  = nn.L1Loss()(expr_out[:, :50],   expr[:, :50])
    #gpose_rec_loss = nn.L1Loss()(expr_out[:, 50:53], expr[:, 50:53])
    #jaw_rec_loss   = nn.L1Loss()(expr_out[:, 53:56], expr[:, 53:56])

    #total_blendshapes_loss = 0.2 * expr_rec_loss + 0.4 * gpose_rec_loss + 0.4 * jaw_rec_loss

    ## loss is VQ reconstruction + weighted pre-computed quantization loss
    quant_loss = quant_loss.mean()

    return total_blendshapes_loss + quant_loss, [total_blendshapes_loss, quant_loss] # rec_loss # [rec_loss, expr_rec_loss, quant_loss]


def calc_logit_loss(pred, target):
    """ Cross entropy loss wrapper """
    loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)), target.reshape(-1))
    return loss





