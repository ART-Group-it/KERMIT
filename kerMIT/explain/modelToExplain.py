import numpy as np
from kerMIT.explain.LRP_linear_layer import lrp_linear

def relevance_to_string(relevance):
    '''
    Takes the relevance vector and returns it as a string, each value separated by a space
    :param relevance: np.array()
    :return: str()
    '''
    s_relevance = ''
    for r in relevance:
        s_relevance += ''.join(str(r)) + ' '
    return s_relevance[:-1]


def get_structure(y_predict,model):
    print("Predict shape:")
    for i,y in enumerate(y_predict):
        print(i,y.shape)
    print("Model shape:")
    for i,l in enumerate(model.layers):
        try:
            print(i,l.get_weights()[0].shape)
        except:
            print(i,"Dropout or Input")
    
           

def lrp_DT(y_predict_, model):
    '''
    The model in question is done in the following way: (8000,) -> (100,) -> (2)
    
    :param y_predict_: it is the history of predictions at each layer np.array( np.array() )
    :param model: 
    :return: relevance of the first layer
    '''
    # calcolo un passo dell'LRP del solo DT
    hin = y_predict_[1].reshape((100,))
    hout = y_predict_[2].reshape((2,))
    w = model.layers[2].get_weights()[0]
    b = model.layers[2].get_weights()[1]
    bias_nb_units = 100
    eps = 0.001
    bias_factor = 1.0

    mask = np.zeros(2)
    mask[ np.argmax(hout) ] = hout[ np.argmax(hout) ]
    Rout = np.array(mask)


    Rin = lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor, debug=False)

    # secondo passo
    hout = hin
    Rout = Rin
    hin = y_predict_[0].reshape((8000,))

    w = model.layers[1].get_weights()[0]
    b = model.layers[1].get_weights()[1]
    bias_nb_units = 8000


    Rin = lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor, debug=False)
    return Rin


 
### vale solo per modelli che hanno dense consecutivi!! senza dropout e concatenata
def LRP_Feed_Forward_without_Dropout(y_predict,model):
    # RICORDA sei obbligato ad untilizzare gli input Layer!!!!
    s = model.layers[-1] # s is the last element
    j = len(y_predict)-1
    # calcolo un passo dell'LRP del solo DT
    w = s.get_weights()[0]
    b = s.get_weights()[1]
    hin = y_predict[j-1].reshape(w.shape[0],)
    hout = y_predict[j].reshape(w.shape[1],)
    bias_nb_units = w.shape[0]
    eps = 0.001
    bias_factor = 1.0
    
    mask = np.zeros( y_predict[-1].shape[1] )
    mask[ np.argmax(hout) ] = hout[ np.argmax(hout) ]
    Rout = np.array(mask)
    
    Rin = lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor, debug=False)

    R,F = [],[] # R is the relevance output layer, F is the stack
    F.append(s._inbound_nodes[0].inbound_layers[0]) # I put the only one child of s in the stack F
    j = j-1
    while F != [] and j != 0:
        u = F.pop() # the last element from the stack is extracted
        v = u._inbound_nodes[0].inbound_layers[0] # I extract the only one child of u
        F.append(v) # I put them in the stack F
        print(u,v)
        #print("(",u.name,",",v.name,")", u.get_weights()[0].shape, y_predict[j].shape, j)
        # i-esimo passo
        hout = hin
        Rout = Rin
        w = u.get_weights()[0]
        b = u.get_weights()[1]
        hin = y_predict[j-1].reshape(w.shape[0],)
        bias_nb_units = w.shape[0]
        Rin = lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor, debug=False)
        j-=1
    return Rin


### vale solo per modelli che hanno dense e dropout
def LRP_Feed_Forward(y_predict,model):
    # RICORDA sei obbligato ad untilizzare gli input Layer!!!!
    s = model.layers[-1] # s is the last element
    j = len(y_predict)-1
    # calcolo un passo dell'LRP del solo DT
    w = s.get_weights()[0]
    b = s.get_weights()[1]
    hin = y_predict[j-1].reshape(w.shape[0],)
    hout = y_predict[j].reshape(w.shape[1],)
    bias_nb_units = w.shape[0]
    eps = 0.001
    bias_factor = 1.0
    
    mask = np.zeros( y_predict[-1].shape[1] )
    mask[ np.argmax(hout) ] = hout[ np.argmax(hout) ]
    Rout = np.array(mask)
    
    Rin = lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor, debug=False)

    R,F = [],[] # R is the relevance output layer, F is the stack
    F.append(s._inbound_nodes[0].inbound_layers[0]) # I put the only one child of s in the stack F
    j = j-1
    while F != [] and j != 0:
        u = F.pop() # the last element from the stack is extracted
        v = u._inbound_nodes[0].inbound_layers[0] # I extract the only one child of u
        F.append(v) # I put them in the stack F
        #print(u,v)
        if u.get_weights() != []: # se non ha pesi (ovvero se è Dropout/Input) skippa
            #print("(",u.name,",",v.name,")", u.get_weights()[0].shape, y_predict[j].shape, j)
            # i-esimo passo
            hout = hin
            Rout = Rin
            w = u.get_weights()[0]
            b = u.get_weights()[1]
            hin = y_predict[j-1].reshape(w.shape[0],)
            bias_nb_units = w.shape[0]
            Rin = lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor, debug=False)
        j-=1
    return Rin