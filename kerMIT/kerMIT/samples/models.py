import torch
from torch import nn
import torch.nn.functional as F
import transformers


class DTBert(nn.Module):
    def __init__(self, input_dim_bert, input_dim_dt, output_dim):
        super().__init__()
        # BERT transformer from Huggingface library
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased').to(
            "cuda" if torch.cuda.is_available() else "cpu")
        # The decoder layer is a simple Linear with softmax applyied
        self.synth_sem_linear = nn.Linear(input_dim_bert + input_dim_dt, output_dim)

    def forward(self, x_sem, x_synth):
        # We don't need o train BERT since we used Universal Sentence Embeddings
        with torch.no_grad():
            x_sem = self.bert(x_sem)[0][:, 0, :]
        # Concatenating the Universal Sentence Embeddings(x_sem) with Universal Syntax Embeddings(x_synth)
        x_tot = torch.cat((x_sem, x_synth), 1)
        # Applying the decoder layer with softmax since it is a multi-classification task
        x_tot = self.synth_sem_linear(x_tot)
        return F.log_softmax(x_tot, dim=1)

    """
    Function used to return activations in order to visualize it with KERMITviz
    """

    def get_activation(self, x_sem, x_synth):
        # This function is the same of forward except we don't backpropagate the gradient
        with torch.no_grad():
            x_sem = self.bert(x_sem)[0][:, 0, :]
            x_concat = torch.cat((x_sem, x_synth), 1)
            out = self.synth_sem_linear(x_concat)
        return out
    
    
if __name__ == "__main__":
    print("DTBert model")
