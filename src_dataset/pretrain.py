import pickle

from transformers import BertForMaskedLM, AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import Dataset, DataLoader
import argparse
from utils import read_file
import os, random
import numpy as np
from transformers import AdamW, SchedulerType, get_scheduler, set_seed
from tqdm import tqdm
from utils import create_datetime_str
import logging

class MLMDataset(Dataset):
    def __init__(self, data, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.plm_model)
        self.mlm_mode = args.mlm_mode
        self.mlm_rate = args.mlm_rate
        self.max_len = args.max_len
        self.data = data

    def __len__(self):
        return len(self.data)


    def __getitem__(self, item):
        js = self.data[item]
        sentence = js['text']


        if self.mlm_mode == 'subword':
            sentence_tok = self.tokenizer.tokenize(' '.join(sentence))
            sentence_ids = self.tokenizer.convert_tokens_to_ids(sentence_tok)
            new_sentence_ids = []
            new_label = []
            for t in sentence_ids:
                r = random.random()
                if r > self.mlm_rate:
                    new_sentence_ids.append(t)
                    new_label.append(-100) # do not compute loss
                elif r >= self.mlm_rate * 0.9:
                    new_sentence_ids.append(t)
                    new_label.append(t)
                elif r >= self.mlm_rate * 0.8:
                    new_sentence_ids.append(random.randint(0, len(self.tokenizer)-1))
                    new_label.append(t)
                else:
                    new_sentence_ids.append(self.tokenizer.mask_token_id)
                    new_label.append(t)
        else:
            h_pos, r_pos, t_pos = js['h']['pos'], js['r']['pos'], js['t']['pos']
            tokens = sentence
            head = ' '.join(tokens[h_pos[0]: h_pos[1]])
            rel = ' '.join(tokens[r_pos[0]: r_pos[1]])
            tail = ' '.join(tokens[t_pos[0]: t_pos[1]])


            left_span = ' '.join(tokens[:h_pos[0]])
            mid_span_hr = ' '.join(tokens[h_pos[1]:r_pos[0]])
            mid_span_rt = ' '.join(tokens[r_pos[1]:t_pos[0]])
            right_span = ' '.join(tokens[t_pos[1]:])
            head_toks = self.tokenizer.tokenize(head)
            rel_toks = self.tokenizer.tokenize(rel)
            tail_toks = self.tokenizer.tokenize(tail)
            left_span_toks = self.tokenizer.tokenize(left_span)
            mid_span_hr_toks = self.tokenizer.tokenize(mid_span_hr)
            mid_span_rt_toks = self.tokenizer.tokenize(mid_span_rt)
            right_span_toks = self.tokenizer.tokenize(right_span)


            r = random.randint(0,2)
            if r == 0:

                new_label = [-100] * (len(left_span_toks)) + self.tokenizer.convert_tokens_to_ids(head_toks) + [-100] * len(mid_span_hr_toks) + [
                    -100] * len(rel_toks) + [-100] * len(mid_span_rt_toks) + [-100] * len(tail_toks) + [-100] * (len(right_span_toks))
                head_toks = [self.tokenizer.mask_token]*len(head_toks)
                new_sentence = left_span_toks + head_toks + mid_span_hr_toks + rel_toks + mid_span_rt_toks + tail_toks + right_span_toks
                new_sentence_ids = self.tokenizer.convert_tokens_to_ids(new_sentence)

            elif r == 1:
                new_label = [-100] * (len(left_span_toks)) + [-100]*len(head_toks) + [-100] * len(mid_span_hr_toks) + self.tokenizer.convert_tokens_to_ids(rel_toks)\
                            + [-100] * len(mid_span_rt_toks) + [-100] * len(tail_toks) + [-100] * (len(right_span_toks))
                rel_toks = [self.tokenizer.mask_token]*len(rel_toks)
                new_sentence = left_span_toks + head_toks + mid_span_hr_toks + rel_toks + mid_span_rt_toks + tail_toks + right_span_toks
                new_sentence_ids = self.tokenizer.convert_tokens_to_ids(new_sentence)

            else:
                new_label = [-100] * (len(left_span_toks)) + [-100] * len(head_toks) + [-100] * len(
                    mid_span_hr_toks) + [-100]*len(rel_toks) \
                            + [-100] * len(mid_span_rt_toks) + self.tokenizer.convert_tokens_to_ids(tail_toks) + [-100] * (len(right_span_toks))
                tail_toks = [self.tokenizer.mask_token] * len(tail_toks)
                new_sentence = left_span_toks + head_toks + mid_span_hr_toks + rel_toks + mid_span_rt_toks + tail_toks + right_span_toks
                new_sentence_ids = self.tokenizer.convert_tokens_to_ids(new_sentence)

        new_sentence_ids = [self.tokenizer.cls_token_id] + new_sentence_ids + [self.tokenizer.sep_token_id]
        new_label = [-100] + new_label + [-100]
        attention_mask = len(new_sentence_ids) * [1] + (self.max_len - len(new_sentence_ids)) * [0]
        new_sentence_ids = new_sentence_ids + (self.max_len - len(new_sentence_ids)) * [self.tokenizer.pad_token_id]
        new_label = new_label + (self.max_len - len(new_label)) * [-100]

        assert len(new_sentence_ids) == len(new_label) == len(attention_mask)
        return np.array(new_sentence_ids), np.array(attention_mask), np.array(new_label)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='../dataset_construction/wiki20m/')
    parser.add_argument('--test_file', type=str, default='revised.test.json')
    parser.add_argument('--valid_file', type=str, default='revised.val.json')
    parser.add_argument('--mlm_rate', type=float, default=0.15)
    parser.add_argument('--mlm_mode', type=str, default='subword')
    parser.add_argument('--plm_model', type=str, default='bert-base-uncased')
    parser.add_argument('--max_len', type=int, default=128, help="max_len")
    parser.add_argument('--bz', type=int, default=64)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument('--save_dir', type=str, default='../dataset_construction/wiki20m/pretrain/')

    args = parser.parse_args()

    test_data = read_file(os.path.join(args.dataset_dir, args.test_file))
    valid_data = read_file(os.path.join(args.dataset_dir, args.valid_file))
    all_data = test_data + valid_data

    dataset = MLMDataset(all_data, args)
    dataloader = DataLoader(dataset, args.bz, shuffle=True)

    model = AutoModelForMaskedLM.from_pretrained(args.plm_model)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_update_steps_per_epoch = len(dataloader)

    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps
    )
    model = model.cuda()

    save_dir = os.path.join(args.save_dir, '{}-{}-{}'.format(args.plm_model, args.mlm_mode, create_datetime_str()))
    logging.info("saving model at: {}".format(save_dir))
    for e in range(args.num_train_epochs):
        progress = tqdm(dataloader, desc='epoch: {} | '.format(e))
        all_losses = []
        for input_ids, attention_mask, labels in progress:
            output = model(
                input_ids=input_ids.cuda(),
                attention_mask=attention_mask.cuda(),
                labels=labels.cuda()
            )
            loss = output.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({
                'loss': loss.item()
            })
            all_losses.append(loss.item())

        if e % 2:
            save_path = os.path.join(save_dir, 'ckpt-{}-{}'.format(e, np.mean(all_losses)))
            logging.info("save pretrianed model at {}".format(save_path))
            model.save_pretrained(save_directory=save_path)
            dataset.tokenizer.save_pretrained(save_directory=save_path)

    logging.info("saving args ... ")
    pickle.dump(args.__dict__, open(os.path.join(save_dir, 'args.pkl'), 'wb'))