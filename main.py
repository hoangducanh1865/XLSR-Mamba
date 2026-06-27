import argparse
import sys
import os
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import Dataset_train, Dataset_eval, Dataset_in_the_wild_eval, genSpoof_list
from model import Model
from utils import reproducibility
from utils import read_asvspoof5_metadata, read_metadata
import numpy as np
import csv

def load_metadata(csv_path):
    label_dict = {}
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file,delimiter=' ')
        for row in reader:
            # filename, _, label = row  # Ignore the middle column (speaker name)
            filename = row[1]
            label = row[5]
            label_dict[filename] = label
    return label_dict

def evaluate_accuracy(dev_loader, model, device, max_batches=0):
    val_loss = 0.0
    num_total = 0.0
    correct=0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_batch = len(dev_loader)
    i=0
    with torch.no_grad():
      for batch_x, batch_y in dev_loader:
        if max_batches and i >= max_batches:
            break
        batch_size = batch_x.size(0)
        target = torch.LongTensor(batch_y).to(device)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        pred = batch_out.max(1)[1] 
        correct += pred.eq(target).sum().item()
        
        batch_loss = criterion(batch_out, batch_y)
        val_loss += (batch_loss.item() * batch_size)
        i=i+1
        print("batch %i of %i (Memory: %.2f of %.2f GiB reserved) (validation)"
                  % (
                     i,
                     num_batch,
                     torch.cuda.max_memory_allocated(device) / (2 ** 30),
                     torch.cuda.max_memory_reserved(device) / (2 ** 30),
                     ),
                  end="\r",
                  )
        
    val_loss /= num_total
    test_accuracy = 100. * correct / num_total
    print('Test accuracy: ' +str(test_accuracy)+'%')
    return val_loss

def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=40, shuffle=False, drop_last=False)
    model.eval()
    fname_list = []
    score_list = []
    with torch.no_grad():
        for batch_x,utt_id in tqdm(data_loader,total=len(data_loader)):
            fname_list = []
            score_list = []  
            batch_x = batch_x.to(device)  
            batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]  
                        ).data.cpu().numpy().ravel() 
            # add outputs
            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())
            
            with open(save_path, 'a+') as fh:
                for f, cm in zip(fname_list,score_list):
                    fh.write('{} {}\n'.format(f, cm))
            fh.close()   
        print('Scores saved to {}'.format(save_path))

def train_epoch(train_loader, model, lr, optim, device, max_batches=0):
    num_total = 0.0
    model.train()

    #set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_batch = len(train_loader)
    i=0
    pbar = tqdm(train_loader, total=num_batch)
    for batch_idx, (batch_x, batch_y) in enumerate(pbar):
        if max_batches and batch_idx >= max_batches:
            break
       
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)     
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
        i=i+1
    sys.stdout.flush()
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XLSR-Mamba')
    # Dataset
    parser.add_argument(
        '--dataset',
        choices=['asvspoof2019la', 'asvspoof5'],
        default='asvspoof2019la',
        help='Training dataset. For asvspoof5, database_path must point to its root.',
    )
    parser.add_argument('--database_path', type=str, default='./data/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same database_path directory.')
    '''
    % database_path/
    %      |- ASVspoof2021_LA_eval/wav
    %      |- ASVspoof2019_LA_train/wav
    %      |- ASVspoof2019_LA_dev/wav
    %      |- ASVspoof2021_DF_eval/wav
    '''

    parser.add_argument('--protocols_path', type=str, default='./data/', help='Change with path to user\'s LA database protocols directory address')
    '''
    % protocols_path/
    %   |- ASVspoof_LA_cm_protocols
    %      |- ASVspoof2021.LA.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
    %      |- ASVspoof2019.LA.cm.train.trn.txt
  
    '''

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=7,
                        help='Early-stopping patience in epochs')
    parser.add_argument('--max_epochs', type=int, default=75,
                        help='Maximum number of training epochs')
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='WCE')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--max_train_batches', type=int, default=0,
                        help='Limit train batches per epoch; 0 uses the full loader')
    parser.add_argument('--max_dev_batches', type=int, default=0,
                        help='Limit validation batches; 0 uses the full loader')

    #model parameters
    parser.add_argument('--emb-size', type=int, default=144, metavar='N',
                    help='embedding size of the model')

    parser.add_argument('--num_encoders', type=int, default=12, metavar='N',
                    help='number of encoders of the mamba blocks')
    parser.add_argument('--FT_W2V', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to fine-tune the W2V or not')
    parser.add_argument(
        '--xlsr_path',
        type=str,
        default=os.environ.get('XLSR_MAMBA_XLSR_PATH', './xlsr2_300m.pt'),
        help='Path to the pretrained xlsr2_300m.pt checkpoint',
    )
    
    # model save path
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    parser.add_argument('--comment_eval', type=str, default=None,
                        help='Comment to describe the saved scores')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Root directory for models and Scores')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to last.pth for full training-state resume')
    
    #Train
    parser.add_argument('--train', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to train the model')
    #Eval
    parser.add_argument('--n_mejores_loss', type=int, default=5, help='save the n-best models')
    parser.add_argument('--average_model', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether average the weight of the n_best epochs')
    parser.add_argument('--n_average_model', default=5, type=int)
    parser.add_argument('--eval_after_train', default=True,
                        type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                        help='Run evaluation after training')

    ##===================================================Rawboost data augmentation ======================================================================#
    parser.add_argument('--algo', type=int, default=5, 
                    help='Rawboost algos discriptions. (3 for DF, 5 for LA and ITW) 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')
    
    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
	
    args = parser.parse_args()
    print(args)
    args.track='LA'

    if not Path(args.xlsr_path).is_file():
        raise FileNotFoundError(
            f"XLSR checkpoint not found: {args.xlsr_path}. "
            "Pass --xlsr_path or set XLSR_MAMBA_XLSR_PATH."
        )
 
    #make experiment reproducible
    reproducibility(args.seed, args)
    
    track = args.track
    n_mejores=args.n_mejores_loss

    assert track in ['LA','DF','In-the-Wild'], 'Invalid track given'
    assert args.n_average_model<args.n_mejores_loss+1, 'average models must be smaller or equal to number of saved epochs'

    #database
    prefix      = 'ASVspoof_{}'.format(track)
    prefix_2019 = 'ASVspoof2019.{}'.format(track)
    prefix_2021 = 'ASVspoof2021.{}'.format(track)
    
    #define model saving path
    dataset_tag = 'ASVspoof5' if args.dataset == 'asvspoof5' else track
    model_tag = 'Bmamba{}_{}_{}_{}_ES{}_NE{}'.format(
        args.algo, dataset_tag, args.loss, args.lr,args.emb_size, args.num_encoders)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    models_root = Path(args.output_dir) / 'models'
    models_root.mkdir(parents=True, exist_ok=True)
    model_save_path = models_root / model_tag
    if args.resume:
        model_save_path = Path(args.resume).expanduser().resolve().parent
    
    print('Model tag: '+ model_tag)

    #set model save directory
    model_save_path.mkdir(parents=True, exist_ok=True)
    best_save_path = model_save_path / 'best'
    best_save_path.mkdir(parents=True, exist_ok=True)
    model_save_path = str(model_save_path)
    best_save_path = str(best_save_path)
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    model = Model(args,device)
    if not args.FT_W2V:
        for param in model.ssl_model.parameters():
            param.requires_grad = False

    model = model.to(device)
    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
         # evaluation mode on the In-the-Wild dataset.

    if args.track == 'In-the-Wild':
        best_save_path = best_save_path.replace(track, 'LA')
        model_save_path = model_save_path.replace(track, 'LA')
        print('######## Eval ########')
        if args.average_model:
            sdl=[]
            model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
            print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
            sd = model.state_dict()
            for i in range(1,args.n_average_model):
                model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
                print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
                sd2 = model.state_dict()
                for key in sd:
                    sd[key]=(sd[key]+sd2[key])
            for key in sd:
                sd[key]=(sd[key])/args.n_average_model
            model.load_state_dict(sd)
            print('Model loaded average of {} best models in {}'.format(args.n_average_model, best_save_path))
        else:
            model.load_state_dict(torch.load(os.path.join(model_save_path, 'best.pth')))
            print('Model loaded : {}'.format(os.path.join(model_save_path, 'best.pth')))
        file_eval = genSpoof_list( dir_meta =  os.path.join(args.protocols_path),is_train=False,is_eval=True)
        print('no. of eval trials',len(file_eval))
        eval_set=Dataset_in_the_wild_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path))
        produce_evaluation_file(eval_set, model, device, 'Scores/{}/{}.txt'.format(args.track, model_tag))
        sys.exit(0)
    # define train/validation paths
    if args.dataset == 'asvspoof5':
        dataset_root = Path(args.database_path).expanduser().resolve()
        train_protocol = dataset_root / 'protocols' / 'ASVspoof5.train.tsv'
        dev_protocol = dataset_root / 'protocols' / 'ASVspoof5.dev.track_1.tsv'
        eval_protocol = dataset_root / 'protocols' / 'ASVspoof5.eval.track_1.tsv'
        train_audio_dir = dataset_root / 'flac_T'
        dev_audio_dir = dataset_root / 'flac_D'
        eval_audio_dir = dataset_root / 'flac_E_eval'
        for required_path in (
            train_protocol,
            dev_protocol,
            eval_protocol,
            train_audio_dir,
            dev_audio_dir,
            eval_audio_dir,
        ):
            if not required_path.exists():
                raise FileNotFoundError(f'ASVspoof5 path not found: {required_path}')

        label_trn, files_id_train = read_asvspoof5_metadata(train_protocol)
        labels_dev, files_id_dev = read_asvspoof5_metadata(dev_protocol)
    else:
        train_protocol = os.path.join(
            args.protocols_path
            + 'LA/{}_cm_protocols/{}.cm.train.trn.txt'.format(prefix, prefix_2019)
        )
        dev_protocol = os.path.join(
            args.protocols_path
            + 'LA/{}_cm_protocols/{}.cm.dev.trl.txt'.format(prefix, prefix_2019)
        )
        train_audio_dir = os.path.join(
            args.database_path
            + 'LA/{}_{}_train/'.format(prefix_2019.split('.')[0], args.track)
        )
        dev_audio_dir = os.path.join(
            args.database_path
            + 'LA/{}_{}_dev/'.format(prefix_2019.split('.')[0], args.track)
        )
        label_trn, files_id_train = read_metadata(train_protocol, is_eval=False)
        labels_dev, files_id_dev = read_metadata(dev_protocol, is_eval=False)

    print('no. of training trials',len(files_id_train))
    
    train_set=Dataset_train(
        args,
        list_IDs=files_id_train,
        labels=label_trn,
        base_dir=train_audio_dir,
        algo=args.algo,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    
    del train_set, label_trn
    
    # define validation dataloader
    print('no. of validation trials',len(files_id_dev))

    dev_set = Dataset_train(
        args,
        list_IDs=files_id_dev,
        labels=labels_dev,
        base_dir=dev_audio_dir,
        algo=0 if args.dataset == 'asvspoof5' else args.algo,
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=8,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    del dev_set,labels_dev

    
    ##################### Training and validation #####################
    not_improving=0
    epoch=0
    bests=np.ones(n_mejores,dtype=float)*float('inf')
    best_loss=float('inf')

    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f'Resume checkpoint not found: {resume_path}')
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        epoch = int(state['epoch']) + 1
        not_improving = int(state.get('not_improving', 0))
        best_loss = float(state.get('best_loss', float('inf')))
        bests = np.asarray(state.get('bests', bests), dtype=float)
        print(f'Resumed training from {resume_path}; next epoch: {epoch}')

    if args.train:
        while not_improving < args.num_epochs and epoch < args.max_epochs:
            print('######## Epoch {} ########'.format(epoch))
            train_epoch(
                train_loader,
                model,
                args.lr,
                optimizer,
                device,
                max_batches=args.max_train_batches,
            )
            val_loss = evaluate_accuracy(
                dev_loader,
                model,
                device,
                max_batches=args.max_dev_batches,
            )
            if val_loss<best_loss:
                best_loss=val_loss
                torch.save(model.state_dict(), os.path.join(model_save_path, 'best.pth'))
                print('New best epoch')
                not_improving=0
            else:
                not_improving+=1
            for i in range(n_mejores):
                if bests[i]>val_loss:
                    for t in range(n_mejores-1,i,-1):
                        bests[t]=bests[t-1]
                        previous_path = os.path.join(best_save_path, 'best_{}.pth'.format(t-1))
                        next_path = os.path.join(best_save_path, 'best_{}.pth'.format(t))
                        if os.path.exists(previous_path):
                            os.replace(previous_path, next_path)
                    bests[i]=val_loss
                    torch.save(model.state_dict(), os.path.join(best_save_path, 'best_{}.pth'.format(i)))
                    break
            torch.save(
                {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'not_improving': not_improving,
                    'best_loss': best_loss,
                    'bests': bests.tolist(),
                    'args': vars(args),
                },
                os.path.join(model_save_path, 'last.pth'),
            )
            print('\n{} - {}'.format(epoch, val_loss))
            print('n-best loss:', bests)
            epoch+=1
        print('Total epochs: ' + str(epoch) +'\n')

    if not args.eval_after_train:
        print('Skipping post-training evaluation (--eval_after_train false).')
        sys.exit(0)

    print('######## Eval ########')
    if args.average_model:
        sdl=[]
        model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
        print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
        sd = model.state_dict()
        for i in range(1,args.n_average_model):
            model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
            print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
            sd2 = model.state_dict()
            for key in sd:
                sd[key]=(sd[key]+sd2[key])
        for key in sd:
            sd[key]=(sd[key])/args.n_average_model
        model.load_state_dict(sd)
        print('Model loaded average of {} best models in {}'.format(args.n_average_model, best_save_path))
    else:
        model.load_state_dict(torch.load(os.path.join(model_save_path, 'best.pth')))
        print('Model loaded : {}'.format(os.path.join(model_save_path, 'best.pth')))

    tracks = 'ASVspoof5' if args.dataset == 'asvspoof5' else ('LA' if args.algo == 5 else 'DF')

    if args.comment_eval:
        model_tag = model_tag + '_{}'.format(args.comment_eval)

    score_dir = Path(args.output_dir) / 'Scores' / tracks
    score_dir.mkdir(parents=True, exist_ok=True)
    score_path = score_dir / '{}.txt'.format(model_tag)

    if not score_path.exists():
        if args.dataset == 'asvspoof5':
            file_eval = read_asvspoof5_metadata(eval_protocol, is_eval=True)
            eval_base_dir = eval_audio_dir
        else:
            prefix      = 'ASVspoof_{}'.format(tracks)
            prefix_2019 = 'ASVspoof2019.{}'.format(tracks)
            prefix_2021 = 'ASVspoof2021.{}'.format(tracks)
            file_eval = read_metadata(
                dir_meta=os.path.join(
                    args.protocols_path
                    + '{}/{}_cm_protocols/{}.cm.eval.trl.txt'.format(
                        tracks, prefix, prefix_2021
                    )
                ),
                is_eval=True,
            )
            eval_base_dir = os.path.join(
                args.database_path + '{}/ASVspoof2021_{}_eval/'.format(tracks, tracks)
            )
        print('no. of eval trials',len(file_eval))
        eval_set=Dataset_eval(
            list_IDs=file_eval,
            base_dir=eval_base_dir,
            track=tracks,
        )
        produce_evaluation_file(eval_set, model, device, str(score_path))
    else:
        print('Score file already exists: {}'.format(score_path))
