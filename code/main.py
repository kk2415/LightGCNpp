import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import world
import utils
from world import cprint
import torch
import numpy as np
import pickle as pkl
from tensorboardX import SummaryWriter
import time
import re
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset




if not os.path.exists('logs'):
    os.mkdir('logs')
    
if not os.path.exists('embs'):
    os.mkdir('embs')

config = f'{world.args.dataset}_seed{world.args.seed}_{world.args.model}_dim{world.args.recdim}_lr{world.args.lr}_dec{world.args.decay}_alpha{world.args.alpha}_beta{world.args.beta}_gamma{world.args.gamma}'

if world.args.model == 'lgn':
    config += f'_nl{world.args.layer}'

log_path = f'logs/{config}.txt'
loss_log_path = f'logs/{config}_loss.txt'
emb_path = f'embs/{config}.pkl'

if os.path.exists(emb_path):
    print('Exists.')
    exit(0)

if world.args.decay == 1e-6 or world.args.decay == 1e-8:
    exit(0)
    
if world.args.layer == 4:
    exit(0)
    
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment))
else:
    w = None
    world.cprint("not enable tensorflowboard")

# 손실값 저장을 위한 리스트
loss_history = []

try:
    best_valid = -1
    patience = 0
    
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        
        # 손실값 추출 및 저장
        loss_match = re.search(r'loss([\d.]+)', output_information)
        if loss_match:
            loss_value = float(loss_match.group(1))
            loss_history.append((epoch + 1, loss_value))
        
        if (epoch + 1) % 5 == 0:
            cprint("[VALIDATION]")
            valid_results = Procedure.Valid(dataset, Recmodel, epoch, w, world.config['multicore'])
            valid_log = [valid_results['ndcg'][0], valid_results['ndcg'][1], valid_results['recall'][0], valid_results['recall'][1], valid_results['precision'][0], valid_results['precision'][1]]
            
            cprint("[TEST]")
            test_results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            test_log = [test_results['ndcg'][0], test_results['ndcg'][1], test_results['recall'][0], test_results['recall'][1], test_results['precision'][0], test_results['precision'][1]]
            
            with open(log_path, 'a') as f:
                f.write(f'valid ' + ' '.join([str(x) for x in valid_log]) + '\n')
                f.write(f'test ' + ' '.join([str(x) for x in test_log]) + '\n')
            
            if valid_results['ndcg'][0] > best_valid:
                best_valid = valid_results['ndcg'][0]
                patience = 0
                
                Recmodel.eval()
                all_users, all_items, _all_users, _all_items = Recmodel.computer()
                all_users, all_items = all_users.detach().cpu(), all_items.detach().cpu()
                _all_users, _all_items = _all_users.detach().cpu(), _all_items.detach().cpu()

                with open(emb_path, 'wb') as f:
                    if world.args.save_layer_emb:
                        pkl.dump([all_users, all_items, _all_users, _all_items], f)
                    else:
                        pkl.dump([all_users, all_items], f)
            else:
                patience += 1

        if patience >= world.args.patience:
            print('Early Stopping')
            
            with open(log_path, 'a') as f:
                f.write('Early Stopping\n')
                
            exit(0)
finally:
    # 학습이 끝나면 손실 로그 파일에 저장
    if loss_history:
        with open(loss_log_path, 'w', encoding='utf-8') as f:
            f.write('epoch,loss\n')
            for epoch, loss in loss_history:
                f.write(f'{epoch},{loss}\n')
        print(f'Loss history saved to {loss_log_path}')
    
    if world.tensorboard:
        w.close()