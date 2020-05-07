import torch

def prepare_input_LRP(y_predict, dtk_sentence, model, index):
    hout = y_predict
    hout = hout.view(hout.shape[1],-1)
    hin = dtk_sentence.view(dtk_sentence.shape[1],-1)

    params = list(model.parameters())
    w = params[-2].view(params[-2].shape[1],-1)[index:].cpu() # faccio il reshape ed estraggo solo la parte dei DT
    b = params[-1]

    bias_nb_units = b.shape[0]
    eps = 0.001
    bias_factor = 1.0

    mask = torch.zeros( y_predict.shape[1] )
    mask[ torch.argmax(hout) ] = hout[ torch.argmax(hout) ]
    Rout = torch.Tensor(mask)
    return hin.cpu(), w.cpu(), b.cpu(), hout.cpu(), Rout.cpu(), bias_nb_units, eps, bias_factor

def lrp_linear_torch(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor=0.0, debug=False):
    """
    LRP for a linear layer with input dim D and output dim M.
    Args:
    - hin:            forward pass input, of shape (D,)
    - w:              connection weights, of shape (D, M)
    - b:              biases, of shape (M,)
    - hout:           forward pass output, of shape (M,) (unequal to np.dot(w.T,hin)+b if more than one incoming layer!)
    - Rout:           relevance at layer output, of shape (M,)
    - bias_nb_units:  total number of connected lower-layer units (onto which the bias/stabilizer contribution is redistributed for sanity check)
    - eps:            stabilizer (small positive number)
    - bias_factor:    set to 1.0 to check global relevance conservation, otherwise use 0.0 to ignore bias/stabilizer redistribution (recommended)
    Returns:
    - Rin:            relevance at layer input, of shape (D,)
    """
    sign_out = torch.where(hout.cpu() >= 0 , torch.Tensor([1.]), torch.Tensor([-1.])).view(1,-1) # shape (1, M)
    
    numer    = (w * hin.view(-1,1)).cpu() + ( bias_factor * (b.view(1,-1)*1. + eps*sign_out*1.) / bias_nb_units ) # shape (D, M)
    # Note: here we multiply the bias_factor with both the bias b and the stabilizer eps since in fact
    # using the term (b[na,:]*1. + eps*sign_out*1.) / bias_nb_units in the numerator is only useful for sanity check
    # (in the initial paper version we were using (bias_factor*b[na,:]*1. + eps*sign_out*1.) / bias_nb_units instead)
    
    denom    = hout.view(1,-1) + (eps*sign_out*1.)   # shape (1, M)
    
    message  = (numer/denom) * Rout.view(1,-1)       # shape (D, M)
    
    Rin      = message.sum(axis=1)              # shape (D,)
    
    if debug:
        print("local diff: ", Rout.sum() - Rin.sum())
    # Note: 
    # - local  layer   relevance conservation if bias_factor==1.0 and bias_nb_units==D (i.e. when only one incoming layer)
    # - global network relevance conservation if bias_factor==1.0 and bias_nb_units set accordingly to the total number of lower-layer connections 
    # -> can be used for sanity check
    
    return Rin