import torch
from torch import nn
import torch.nn.functional as F
import transformers


class DTBert(nn.Module):
    def __init__(self, input_dim_bert, input_dim_dt, output_dim):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased').to("cuda" if torch.cuda.is_available() else "cpu")
        self.synth_sem_linear = nn.Linear(input_dim_bert + input_dim_dt, output_dim)
        #self.activation = {}
    
    def forward(self, x_sem, x_synth):
        with torch.no_grad():
            x_sem = self.bert(x_sem)[0][:, 0, :]
        x_tot = torch.cat((x_sem, x_synth), 1)
        x_tot = self.synth_sem_linear(x_tot)
        out = F.log_softmax(x_tot, dim=1)
        return out
    
    #restituisce attivazioni prima di softmax   
    def get_activation(self, x_sem, x_synth):
        with torch.no_grad():
            x_sem = self.bert(x_sem)[0][:, 0, :]
            x_concat = torch.cat((x_sem, x_synth), 1)
            out = self.synth_sem_linear(x_concat)
        return out
    
    
if __name__ == "__main__":
    print("DTBert model")
