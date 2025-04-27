#!/usr/bin/env python
import torch.nn.functional as F
import torch.nn as nn



def calc_vq_loss(vertice_out, vertice, expr_out, expr, quant_loss, quant_loss_weight=1.0, alpha=1.0):
    """ function that computes the various components of the VQ loss """
    ## Loss for the vertices
    rec_loss = nn.L1Loss()(vertice_out, vertice)

    ## loss for the expression vector
    expr_rec_loss = nn.L1Loss()(expr_out, expr)

    ## loss is VQ reconstruction + weighted pre-computed quantization loss
    quant_loss = quant_loss.mean()

    return quant_loss * quant_loss_weight + rec_loss + expr_rec_loss, [rec_loss, expr_rec_loss, quant_loss]


def calc_logit_loss(pred, target):
    """ Cross entropy loss wrapper """
    loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)), target.reshape(-1))
    return loss





