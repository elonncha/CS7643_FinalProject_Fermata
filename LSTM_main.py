from data_util import *
import torch
import torch.nn as nn
import pickle
import time
from torch.utils.data import DataLoader
from Seq2Seq_LSTM.Encoder import Encoder
from Seq2Seq_LSTM.Decoder import Decoder
from Seq2Seq_LSTM.Seq2Seq import Seq2Seq
from ray import tune
from pathlib import Path
import gc

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, learning_rate, steps, warmup):
    epoch += 1
    if epoch <= warmup:
        lr = learning_rate * epoch / warmup
    elif epoch > steps[1]:
        lr = learning_rate * 0.01
    elif epoch > steps[0]:
        lr = learning_rate * 0.1
    else:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def predict_song(prediction, target, batch_size, predict_abc=False, note_dic=None):
    for song_no in range(batch_size):
        mask_e = np.argwhere(target[song_no,:] == 15).flatten()
        mask_s = np.argwhere(target[song_no, :] == 14).flatten()
        for e in mask_e:
            prediction[song_no][e] = 15
        for s in mask_s:
            prediction[song_no][s] = 14
        # print('Prediction: ', prediction)

        if predict_abc == True:
            predicted_note = []
            for i in range(prediction.shape[0]):
                song = []
                for j in range(prediction.shape[1]):
                    song.append(note_dic[prediction[i, j].item()])
                predicted_note.append(song)
            # print('Predicted Note: ', predicted_note)

    if predict_abc == False:
        return prediction
    else:
        return predicted_note

def train(epoch, data_loader, note_dic, model, optimizer, criterion, device='cpu', debug=False):
    
    iter_time = AverageMeter()
    losses = AverageMeter()
    metric = AverageMeter()
    
    for idx, data in enumerate(data_loader):

        note_past, measure_past, note_future, measure_future, note_target = [x.to(device) for x in data]

        start = time.time()

        output = model.forward(note_past, measure_past, note_future, measure_future, note_target)

        # prediction = torch.argmax(output, dim=2)
        # prediction = predict_song(prediction, note_target, len(data_loader), predict_abc=False)

        loss = criterion(output.permute(0, 2, 1), note_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.detach().item(), output.detach().shape[0])
        iter_time.update(time.time() - start)

        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t')
                   .format(epoch, idx, len(data_loader), iter_time=iter_time, loss=losses))

    avg_loss = losses.avg
    perplexity = np.exp(avg_loss)
    print(('Train Summary:\t'
            'Epoch: {0}\t'
            'Avg Loss {avg_loss:.3f}\t'
            'Avg Perplexity {avg_perplexity:.4f}\t')
            .format(epoch, avg_loss=avg_loss, avg_perplexity=perplexity))
    return losses.val, avg_loss, perplexity #all_predictions # only need to convert predictions for test set


def val(data_loader, model, device='cpu', criterion=None, epoch=1, debug=False):
    
    if criterion:
        criterion = criterion
    else:
        criterion = nn.CrossEntropyLoss()

    iter_time = AverageMeter()
    losses = AverageMeter()
    metric = AverageMeter()

    for idx, data in enumerate(data_loader):

        note_past, measure_past, note_future, measure_future, note_target = [x.to(device) for x in data]

        start = time.time()

        with torch.no_grad():
            output = model.forward(note_past, measure_past, note_future, measure_future, note_target)

            # prediction = torch.argmax(output, dim=2)
            # prediction = predict_song(prediction, note_target, len(data_loader), predict_abc=False)

            loss = criterion(output.permute(0, 2, 1), note_target)
            losses.update(loss.detach().item(), output.detach().shape[0])
            iter_time.update(time.time() - start)

        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t')
                   .format(epoch, idx, len(data_loader), iter_time=iter_time, loss=losses))

    avg_loss = losses.avg
    perplexity = np.exp(avg_loss)
    print(('Evaluation Summary:\t'
            'Epoch: {0}\t'
            'Avg Loss {avg_loss:.3f}\t'
            'Avg Perplexity {avg_perplexity:.4f}\t')
            .format(epoch, avg_loss=avg_loss, avg_perplexity=perplexity))

    # do we want to store the metrics we return somewhere?
    return losses.val, avg_loss, perplexity


def test(data_loader, note_dic, model, target_size, results_root=None, device='cpu', debug=False):
    
    metric = AverageMeter()

    # only writes to ABC notation if running on test set
    raw_score, all_predictions_songs = np.zeros((0, target_size, np.array(note_dic).shape[0])), []

    for idx, data in enumerate(data_loader):

        note_past, measure_past, note_future, measure_future, note_target = [x.to(device) for x in data]

        start = time.time()

        with torch.no_grad():
            output = model.forward(note_past, measure_past, note_future, measure_future, note_target)

            prediction = torch.argmax(output, dim=2)
            prediction_songs = predict_song(prediction.cpu().detach(), note_target.cpu().detach(), len(data_loader), predict_abc=True, note_dic=note_dic)

            raw_score = np.append(raw_score, output.cpu().detach(), axis=0)
            all_predictions_songs.append(prediction_songs)

    # do we want to store other metrics?
    path = os.path.join(results_root, 'prediction')
    with open(path, 'wb') as pickle_w:
        write = {b'raw_score': raw_score,
                 b'abc': all_predictions_songs}
        pickle.dump(write, pickle_w)
    # open test
    with open(path, 'rb') as pickle_r:
        dict = pickle.load(pickle_r, encoding='bytes')
        # print('Raw Score (Test): ', dict[b'raw_score'])
        # print('ABC (Test): ', dict[b'abc'])


# hyperparameter tuning trainer function definition
def trainer(hp, checkpoint_dir=None, data=None):

    # specifically for use of ray tune's search space / checkpoint or log directories: hp (hyperparams), checkpoint_dir

    # load normal trainer function arguments
    wd, data_root, results_root, use_subset, debug = data
        # note: regular training function definition
        # def trainer(data_root, results_root, use_subset=False, train_test_split=False, debug=False):
    
    # switch back to original project working directory: when using ray tune, modifying tune's local_dir for logging changes the working directory
    os.chdir(wd)

    note_past, note_target, note_future, measure_past, measure_mask, measure_future, note_dic, song_id = load_data()

    train_data = INPAINT(data_root, ds_type='train', use_subset=use_subset, batch_size=hp['batch_size'])
    train_loader = DataLoader(train_data, batch_size=hp['batch_size'], shuffle=False, drop_last=False)
    val_data = INPAINT(data_root, ds_type='val')
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=len(val_data.target), shuffle=False, drop_last=False) # is this batch size ok?

    note_vocab_size = train_data.note_vocab_size()
    measure_vocab_size = train_data.measure_vocab_size()
    seq_length_past, seq_length_future = train_data.seq_length()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize model
    past_encoder = Encoder(vocab_size=note_vocab_size, max_measure=measure_vocab_size, seq_len=seq_length_past,
                        emb_size=hp['emb_size'],
                        encoder_hidden_size=hp['enc_hidden_size'], decoder_hidden_size=hp['dec_hidden_size'],
                        dropout=hp['dropout'])
    future_encoder = Encoder(vocab_size=note_vocab_size, max_measure=measure_vocab_size, seq_len=seq_length_future,
                            emb_size=hp['emb_size'],
                            encoder_hidden_size=hp['enc_hidden_size'], decoder_hidden_size=hp['dec_hidden_size'],
                            dropout=hp['dropout'])
    decoder = Decoder(emb_size=hp['emb_size'], decoder_hidden_size=past_encoder.decoder_hidden_size*4, output_size=note_vocab_size, dropout=hp['dropout'])
    model = Seq2Seq(past_encoder, future_encoder, decoder, device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), hp['lr'], weight_decay=hp['reg'])

    best = 9999999
    # best_pred, best_target = None, None
    for epoch in range(hp['epochs']):

        adjust_learning_rate(optimizer, epoch, hp['lr'], hp['steps'], hp['warmup'])

        train_loss, avg_train_loss, train_perplexity = train(epoch, train_loader, note_dic, model, optimizer, criterion, device=device, debug=debug)
        
        val_loss, avg_val_loss, val_perplexity = val(val_loader, model, device=device, criterion=criterion, epoch=epoch, debug=debug)
    
        if val_loss <= best:
            best = val_loss
            best_model = copy.deepcopy(model)
            best_optimizer = copy.deepcopy(optimizer)            
            # best_pred, best_target = pred, target
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, 'checkpoint.pth')
                torch.save((best_model, best_model.state_dict(), best_optimizer.state_dict()), path)

        # tune.report metrics
        tune.report(val_loss=val_loss, avg_val_loss=avg_val_loss, val_perplexity=val_perplexity,
                    train_loss=train_loss, avg_train_loss=avg_train_loss, train_perplexity=train_perplexity)
        gc.collect()


def hp_search(wd, data_root, results_root, exp_name, tensorboard_local_dir, use_subset=False, debug=False):

    ########################################################################################################################################################
    # CONFIGURE HYPERPARAMETER SEARCH:
    # general parameters:
    resume = False
    epochs = 20#100#100
    # early stopping parameters:
    grace_period = 5
    reduction_factor = 2.5
    brackets = 1
    ########################################################################################################################################################

    ########################################################################################################################################################
    metric = 'val_loss'
    metric_mode = 'min'
    scheduler = tune.schedulers.AsyncHyperBandScheduler(time_attr='training_iteration', grace_period=grace_period, max_t=epochs, reduction_factor=reduction_factor, brackets=brackets)
    num_samples = 15#15#1

    # Inpaint paper notes:
        # The MeasureVAE model was pre-trained using single measures following the standard VAE optimization equa- tion [26] with the ??-weighting scheme [41,42]. 
        # The Adam algorithm [43] was used for model training, with a learning rate of 1e???3, ??1 = 0.9, ??2 = 0.999, and ?? = 1e???8.
    # search_space = {'epochs': epochs,
    #                 'batch_size': tune.choice([64, 128, 256, 512]),
    #                 'lr': tune.loguniform(1e-3, 9e-1),
    #                 'warmup': 0,
    #                 'steps': [tune.uniform(1, 3), tune.uniform(4, 6)],
    #                 'reg': tune.loguniform(1e-6,1e-1),
    #                 'emb_size': tune.choice([10, 15, 20]),
    #                 'enc_hidden_size': tune.choice([128, 256, 512, 1024]),
    #                 'dec_hidden_size': tune.choice([128, 256, 512, 1024]),
    #                 'dropout': tune.choice([0.1, 0.2])
    #                 }
 
    search_space = {'epochs': epochs,
                    'batch_size': 128,
                    'lr': 0.015919,
                    'warmup': 0,
                    'steps': [1, 4],
                    'reg': 5.3977e-06,
                    'emb_size': 15,
                    'enc_hidden_size': 512,
                    'dec_hidden_size': 256,
                    'dropout': 0.2
                    }

    start = time.time()
    analysis = tune.run(
    tune.with_parameters(trainer,
                        data=(wd, data_root, results_root, use_subset, debug)),
                        config=search_space,
                        metric=metric,
                        mode=metric_mode,
                        scheduler=scheduler,
                        name=exp_name,
                        local_dir=tensorboard_local_dir,
                        num_samples=num_samples,
                        keep_checkpoints_num=1,
                        checkpoint_score_attr=metric,
                        max_failures=3,
                        resume=resume,
                        raise_on_failed_trial=False,
                        resources_per_trial={'gpu': 1})
    taken = time.time() - start
    best_config = analysis.get_best_config(metric=metric, mode=metric_mode)
    best_result = analysis.best_result
    print(f"Time Taken: {taken:.2f} seconds.")
    print('Best Configuration: {0}\n'.format(metric), best_config)
    print('Best Result: {0}\n'.format(metric), best_result)

    ########################################################################################################################################################

    ########################################################################################################################################################

    # load ray tune's best trial/checkpoint
    print('Loading Best Checkpoint: Evaluating on test set...')
    best_trial = analysis.get_best_trial(metric=metric, mode=metric_mode, scope='all')
    path_checkpoint = analysis.get_best_checkpoint(trial=best_trial, metric=metric, mode=metric_mode) + 'checkpoint.pth'

    # load data
    note_past, note_target, note_future, measure_past, measure_mask, measure_future, note_dic, song_id = load_data()
    val_data = INPAINT(data_root, ds_type='val')
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=len(val_data.target), shuffle=False, drop_last=False) # is this batch size ok?
    test_data = INPAINT(data_root, ds_type='test')
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data.target), shuffle=False, drop_last=False) # is this batch size ok?
    target_size = val_data.target_size()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load best models from best checkpoint
    best_model, best_model_state, best_optimizer = torch.load(path_checkpoint)
    best_model.load_state_dict(best_model_state)

    # run
    val_loss, avg_val_loss, val_perplexity = val(val_loader, best_model, device=device)
    test_perplexity = test(test_loader, note_dic, best_model, target_size, results_root, device=device)

    ########################################################################################################################################################

def main(mode):
    seed = 0
    wd = os.getcwd()
    p = Path().absolute()
    data_root = p/'dataset_split'
    results_root = p/'results/Seq2Seq_LSTM'
    exp_name = p/'results/Seq2Seq_LSTM_tensorboard'
    tensorboard_local_dir = '~'
    use_subset = False
    train_test_split = False
    debug = True

    os.makedirs(results_root, exist_ok=True)
    os.makedirs(exp_name, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    if train_test_split == True:
        # data load
        note_past, note_target, note_future, measure_past, measure_mask, measure_future, note_dic, song_id = load_data()

        train_test_val_split(note_past, note_target, note_future, measure_past, measure_mask, measure_future, song_id)
                
    if mode == 'hp_search':
        hp_search(wd, data_root, results_root, exp_name, tensorboard_local_dir, use_subset, debug)
    # not tested
    # if mode == 'test':
    #     tester(wd, data_root, results_root, exp_name, tensorboard_local_dir, use_subset, debug) # need to implement



if __name__ == '__main__':
    main(mode='hp_search')

    ## mode: 'hp_search', hyperparams search with ray tune
        # HOW TO VISUALIZE METRICS PLOTTING IN TENSORBORD: tensorboard --logdir YOUR_RAY_TUNE_LOCAL_DIRECTORY
