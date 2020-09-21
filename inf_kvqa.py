"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

run inference of CLEVR for submission
"""
import argparse
import json
import os
from os.path import exists
from time import time

import torch
from torch.utils.data import DataLoader

from apex import amp
from horovod import torch as hvd
import numpy as np
from cytoolz import concat

from data import (TokenBucketSampler, PrefetchLoader,
                  DetectFeatLmdb, TxtTokLmdb, GqaEvalDataset, gqa_eval_collate)
from model.gqa import UniterForGeneralQuestionAnswering

from utils.logger import LOGGER
from utils.distributed import all_gather_list
from utils.misc import Struct
from utils.const import BUCKET_SIZE, IMG_DIM


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), opts.fp16))

    hps_file = f'{opts.output_dir}/log/hps.json'
    model_opts = Struct(json.load(open(hps_file)))

    # train_examples = None
    ans2label_file = f'{opts.output_dir}/ckpt/ans2label.json'
    ans2label = json.load(open(ans2label_file))
    label2ans = {label: ans for ans, label in ans2label.items()}   
    print(len(ans2label))
    
    label2ans = json.load(open(f'/txt/kvqa_train_questions.db/idx2ans.json'))
    label2ans = {int(label): ans for label, ans in label2ans.items()}

    # Add unknown
    label2ans[len(label2ans)] = 'UNK'
    ans2label = {ans: label for label, ans in label2ans.items()}
    print(len(ans2label))    

    idx2type = json.load(open(f'/txt/kvqa_test_questions.db/idx2type.json'))

    # load DBs and image dirs
    eval_img_db = DetectFeatLmdb(opts.img_db,
                                 model_opts.conf_th, model_opts.max_bb,
                                 model_opts.min_bb, model_opts.num_bb,
                                 opts.compressed_db)
    
    eval_txt_db = TxtTokLmdb(opts.txt_db, -1)
    eval_dataset = GqaEvalDataset(ans2label, eval_txt_db, eval_img_db)

    # Prepare model
    if exists(opts.checkpoint):
        ckpt_file = opts.checkpoint
    else:
        ckpt_file = f'{opts.output_dir}/ckpt/model_step_{opts.checkpoint}.pt'
    checkpoint = torch.load(ckpt_file)
    model = UniterForGeneralQuestionAnswering.from_pretrained(
        f'{opts.output_dir}/log/model.json', checkpoint,
        img_dim=IMG_DIM, num_answer=len(ans2label))
    model.to(device)
    if opts.fp16:
        model = amp.initialize(model, enabled=True, opt_level='O2')

    sampler = TokenBucketSampler(eval_dataset.lens, bucket_size=BUCKET_SIZE,
                                 batch_size=opts.batch_size, droplast=False)
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_sampler=sampler,
                                 num_workers=opts.n_workers,
                                 pin_memory=opts.pin_mem,
                                 collate_fn=gqa_eval_collate)
    eval_dataloader = PrefetchLoader(eval_dataloader)

    val_log, results, logits = evaluate(model, eval_dataloader, label2ans,
                                        idx2type, opts.save_logits, opts.topk)
    result_dir = f'{opts.output_dir}/results_test'
    if not exists(result_dir) and rank == 0:
        os.makedirs(result_dir)

    all_results = list(concat(all_gather_list(results)))
    if opts.save_logits:
        all_logits = {}
        for id2logit in all_gather_list(logits):
            all_logits.update(id2logit)
    if hvd.rank() == 0:
        with open(f'{result_dir}/'
                  f'results_{opts.checkpoint}_top{opts.topk}_all.json', 'w') as f:
            json.dump(all_results, f)
        if opts.save_logits:
            np.savez(f'{result_dir}/logits_{opts.checkpoint}_all.npz',
                     **all_logits)


@torch.no_grad()
def evaluate(model, eval_loader, label2ans, idx2type, save_logits=False, topk=1):
    LOGGER.info("start running evaluation...")
    model.eval()
    n_ex = 0
    total_correct = 0
    st = time()
    results = []
    types_correct = {}
    types_all = {}
    logits = {}
    
    for i, batch in enumerate(eval_loader):
        qids = batch['qids']
        scores = model(batch, compute_loss=False)
        targets = batch['targets']
        topk_correct, pred_answers = compute_score_with_answers(scores, targets, label2ans, topk=topk)
        total_correct += topk_correct

        true_answers = [label2ans[i] for i in targets.cpu().tolist()]
        
        for qid, y, y_hat in zip(qids, true_answers, pred_answers):
            for q_type in idx2type[qid]:
                types_all[q_type] = types_all.get(q_type, 0) + 1
                if y in y_hat:
                    types_correct[q_type] = types_correct.get(q_type, 0) + 1
        

        # display answers
        print("\n".join("Correct with True: {} Predicted: {}".format(str(x), (', ').join([str(yi) for yi in y])) for x, y in zip(true_answers, pred_answers) if x in y))
        for qid, answer in zip(qids, pred_answers):
            results.append({'answer': answer, 'question_id': int(qid)})
        if save_logits:
            scores = scores.cpu()
            for i, qid in enumerate(qids):
                logits[qid] = scores[i].half().numpy()
        if i % 100 == 0 and hvd.rank() == 0:
            n_results = len(results)
            n_results *= hvd.size()   # an approximation to avoid hangs
            LOGGER.info(f'{n_results}/{len(eval_loader.dataset)} '
                        'answers predicted')
        n_ex += len(qids)
    total_score = total_correct / n_ex
    print(f'Total score is {"{:.2f}".format(total_score*100)}%')
    
    type_score = {}
    for q_type in types_all.keys():
        type_score[q_type] = types_correct.get(q_type, 0)/types_all[q_type]
    print(type_score)

    n_ex = sum(all_gather_list(n_ex))
    tot_time = time()-st
    val_log = {'valid/ex_per_s': n_ex/tot_time}
    model.train()
    LOGGER.info(f"evaluation finished in {int(tot_time)} seconds "
                f"at {int(n_ex/tot_time)} examples per second")
    return val_log, results, logits



def compute_score_with_answers(scores, targets, label2ans, topk=1):
    topk_labels = torch.topk(scores, k=topk, dim=-1, largest=True, sorted=True)[1].tolist()
    topk_correct = sum([target in topk for topk, target in zip(topk_labels,
                                                             targets.cpu().tolist())])
    topk_preds = [[label2ans[i] for i in topk] for topk in topk_labels] 
    return topk_correct, topk_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--txt_db",
                        default=None, type=str,
                        help="The input train corpus. (LMDB)")
    parser.add_argument("--img_db",
                        default=None, type=str,
                        help="The input train images.")
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="can be the path to binary or int number (step)")
    parser.add_argument("--batch_size",
                        default=8192, type=int,
                        help="number of tokens in a batch")

    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory of the training command")

    parser.add_argument("--save_logits", action='store_true',
                        help="Whether to save logits (for making ensemble)")
                        
    parser.add_argument('--topk', default=1, type=int,
                        help="The number of predictions to consider")
    # Prepro parameters

    # device parameters
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")
    args = parser.parse_args()
    
    main(args)
