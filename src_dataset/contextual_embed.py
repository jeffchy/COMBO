import torch
from torch import nn
from transformers import AutoModel, AutoConfig
import sys
sys.path.append('../')
from src_dataset.span_embed import span_reps

class ContextualEmbedder(nn.Module):

    def __init__(self, args):
        super(ContextualEmbedder, self).__init__()

        self.args = args
        self.plm = AutoModel.from_pretrained(pretrained_model_name_or_path=args.plm_model).cuda()
        self.plm_layers = [int(i) for i in self.args.plm_layer.split(',')]
        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path=args.plm_model)

    def forward(self, all_ids, attention_mask, scatter_mask):
        res = {}
        if self.args.sep_strategy not in ['split', 'prompt-split']:
            outputs = self.plm(
                input_ids=all_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            for layer in self.plm_layers:
                hidden_states = outputs.hidden_states[layer] # layer x B x L x H
                B, L, H = hidden_states.size()

                head_batch_emb = []
                rel_batch_emb = []
                tail_batch_emb = []

                for b in range(B):
                    h_e = hidden_states[b, scatter_mask[b] == 1].detach()
                    r_e = hidden_states[b, scatter_mask[b] == 2].detach()
                    t_e = hidden_states[b, scatter_mask[b] == 3].detach()
                    h_e, r_e, t_e = span_reps(h_e, r_e, t_e, self.args.rep_strategy)

                    head_batch_emb.append(h_e)
                    rel_batch_emb.append(r_e)
                    tail_batch_emb.append(t_e)

                head_batch_emb = torch.stack(head_batch_emb)
                rel_batch_emb = torch.stack(rel_batch_emb)
                tail_batch_emb = torch.stack(tail_batch_emb)
                res[layer] = (head_batch_emb.cpu(), rel_batch_emb.cpu(), tail_batch_emb.cpu())


        else: # split
            B, N, L = all_ids.size()
            assert N == 3
            all_ids = all_ids.view(-1, L)
            scatter_mask = scatter_mask.view(-1, L)
            attention_mask = attention_mask.view(-1, L)
            outputs = self.plm(
                input_ids=all_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            for layer in self.plm_layers:
                hidden_states = outputs.hidden_states[layer] # B x L x H
                _, L, H = hidden_states.size()

                head_batch_emb = []
                rel_batch_emb = []
                tail_batch_emb = []

                for b in range(B):
                    i = b*3
                    h_e = hidden_states[i, scatter_mask[i] == 1].detach()
                    r_e = hidden_states[i+1, scatter_mask[i+1] == 1].detach()
                    t_e = hidden_states[i+2, scatter_mask[i+2] == 1].detach()
                    h_e, r_e, t_e = span_reps(h_e, r_e, t_e, self.args.rep_strategy)

                    head_batch_emb.append(h_e)
                    rel_batch_emb.append(r_e)
                    tail_batch_emb.append(t_e)

                head_batch_emb = torch.stack(head_batch_emb)
                rel_batch_emb = torch.stack(rel_batch_emb)
                tail_batch_emb = torch.stack(tail_batch_emb)

                res[layer] = (head_batch_emb.cpu(), rel_batch_emb.cpu(), tail_batch_emb.cpu())


        return res
