import torch


#### INIZIO DARIO

#activation = {}


def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def getWeightAnBiasByName(model, layer_name):
#    weight, bias = _, _
    weight, bias = None, None
    for name, param in model.named_parameters():
        if name == layer_name + '.weight' and param.requires_grad:
            weight = param.data
        elif name == layer_name + '.bias' and param.requires_grad:
            bias = param.data
    return weight, bias


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
    sign_out = torch.where(hout.cpu() >= 0, torch.Tensor([1.]), torch.Tensor([-1.])).view(1, -1)  # shape (1, M)

    numer = (w * hin.view(-1, 1)).cpu() + (
                bias_factor * (b.view(1, -1) * 1. + eps * sign_out * 1.) / bias_nb_units)  # shape (D, M)
    # Note: here we multiply the bias_factor with both the bias b and the stabilizer eps since in fact
    # using the term (b[na,:]*1. + eps*sign_out*1.) / bias_nb_units in the numerator is only useful for sanity check
    # (in the initial paper version we were using (bias_factor*b[na,:]*1. + eps*sign_out*1.) / bias_nb_units instead)

    denom = hout.view(1, -1) + (eps * sign_out * 1.)  # shape (1, M)

    message = (numer / denom) * Rout.view(1, -1)  # shape (D, M)

    Rin = message.sum(axis=1)  # shape (D,)

    if debug:
        print("local diff: ", Rout.sum() - Rin.sum())
    # Note:
    # - local  layer   relevance conservation if bias_factor==1.0 and bias_nb_units==D (i.e. when only one incoming layer)
    # - global network relevance conservation if bias_factor==1.0 and bias_nb_units set accordingly to the total number of lower-layer connections
    # -> can be used for sanity check

    return Rin


def prepare_single_pass(model, activation, start_layer, end_layer, isFirstCompute = True):
    hout = activation[start_layer].reshape(-1)
    if end_layer != None:
        hin = activation[end_layer].reshape(-1).cpu()
    else:
        hin = None

    w, b = getWeightAnBiasByName(model, start_layer)
    w = w.reshape(w.shape[1], w.shape[0])

    bias_nb_units = b.shape[0]
    eps = 0.001
    # eps = 0.2
    bias_factor = 1.0
    if isFirstCompute:
        mask = torch.zeros(hout.shape[0])
        mask[torch.argmax(hout)] = hout[torch.argmax(hout)]
        Rout = torch.Tensor(mask).cpu()
    else:
        Rout = None
    return hin, w.cpu(), b.cpu(), hout.cpu(), Rout, bias_nb_units, eps, bias_factor


##### FMZ Trying an intuition
def compute_LRP_FFNN(model, activation, layer_names, on_demand_embedding_matrix, single_test, demux_layer=None,
                     demux_span=(None, None)):
    isFirstCompute = True
    for i in range(len(layer_names) - 1):
        print(layer_names[i], layer_names[i + 1])
        hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor = prepare_single_pass(model, activation, layer_names[i],
                                                                                     layer_names[i + 1], isFirstCompute)
        if not isFirstCompute:
            Rout = Rin
        Rin = lrp_linear_torch(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor)
        # Handling the demultiplexing of the transformer and the distributed structure encoder MLP
        # and isolating the contribution for the distributed structure encoder MLP
        if demux_layer != None and demux_layer == layer_names[i]:
            Rin = Rin[demux_span[0], demux_span[1]]
        isFirstCompute = False
    # compute the last layer
    _, w, b, hout, Rout, bias_nb_units, eps, bias_factor = prepare_single_pass(model, activation, layer_names[-1], None,
                                                                               isFirstCompute)
    # Handling the demultiplexing of the transformer and the distributed structure encoder MLP
    # and isolating the contribution for the distributed structure encoder MLP
    if not isFirstCompute:
        Rout = Rin
    if demux_layer != None and demux_layer == layer_names[-1]:
        print(Rout)
        print(w.shape)
        # Rout = Rout[demux_span[0],demux_span[1]]
        w = w[demux_span[0]:demux_span[1]]
    hin = single_test.reshape(-1).cpu()
    # FMZ Rin = lrp_linear_torch(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor, debug=False)
    # print(on_demand_embedding_matrix.shape)
    # print(w.shape)
    Rin = lrp_linear_torch(hin, torch.matmul(on_demand_embedding_matrix, w), b, hout, Rout, bias_nb_units, eps,
                           bias_factor, debug=False)
    return Rin

#### FINE DARIO