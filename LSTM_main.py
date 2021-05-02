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

def train(epoch, data_loader, note_dic, model, optimizer, criterion, debug=False):
    
    iter_time = AverageMeter()
    losses = AverageMeter()
    metric = AverageMeter()
    
    for idx, data in enumerate(data_loader):

        start = time.time()

        note_past, measure_past, note_future, measure_future, note_target = data

        output = model.forward(note_past, measure_past, note_future, measure_future, note_target)

        loss = criterion(output.permute(0, 2, 1), note_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), output.shape[0])
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


def val(data_loader, model, criterion=None, epoch=1, debug=False):
    
    if criterion:
        criterion = criterion
    else:
        criterion = nn.CrossEntropyLoss()

    iter_time = AverageMeter()
    losses = AverageMeter()
    metric = AverageMeter()

    for idx, data in enumerate(data_loader):

        start = time.time()

        note_past, measure_past, note_future, measure_future, note_target = data

        with torch.no_grad():
            output = model.forward(note_past, measure_past, note_future, measure_future, note_target)

            loss = criterion(output.permute(0, 2, 1), note_target)
            losses.update(loss.item(), output.shape[0])
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


def test(data_loader, note_dic, model, target_size, debug=False):
    
    metric = AverageMeter()

    # only writes to ABC notation if running on test set
    all_predictions = np.zeros((0, target_size))

    for idx, data in enumerate(data_loader):

        start = time.time()

        note_past, measure_past, note_future, measure_future, note_target = data

        with torch.no_grad():
            output = model.forward(note_past, measure_past, note_future, measure_future, note_target)

            prediction = torch.argmax(output, dim=2)
            predicted_notes = np.empty_like(prediction, dtype='str')
            for i in range(prediction.shape[0]):
                for j in range(prediction.shape[1]):
                    predicted_notes[i,j] = note_dic[prediction[i,j].item()]
            all_predictions = np.append(all_predictions, predicted_notes, axis=0)

    # do we want to store the metrics we return somewhere?
    # convert_to_song(all_predictions) # may include metrics for song evaluation post-converting to audio file
    # return perplexity


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
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=50, shuffle=False, drop_last=False) # is this batch size ok?

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

    if torch.cuda.is_available():
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), hp['lr'], weight_decay=hp['reg'])

    best = 9999999
    # best_pred, best_target = None, None
    for epoch in range(hp['epochs']):

        train_loss, avg_train_loss, train_perplexity = train(epoch, train_loader, note_dic, model, optimizer, criterion, debug=debug)
        
        val_loss, avg_val_loss, val_perplexity = val(val_loader, model, criterion, epoch=epoch, debug=debug)
    
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


def hp_search(wd, data_root, results_root, exp_name, tensorboard_local_dir, use_subset=False, debug=False):

    ########################################################################################################################################################
    # CONFIGURE HYPERPARAMETER SEARCH:
    # general parameters:
    resume = False
    epochs = 1 # 50
    # early stopping parameters:
    grace_period = 1 #5
    reduction_factor = 2.5
    brackets = 1
    ########################################################################################################################################################

    ########################################################################################################################################################
    metric = 'val_loss'
    metric_mode = 'min'
    scheduler = tune.schedulers.AsyncHyperBandScheduler(time_attr='training_iteration', grace_period=grace_period, max_t=epochs, reduction_factor=reduction_factor, brackets=brackets)
    num_samples = 1

    # Inpaint paper notes:
        # The MeasureVAE model was pre-trained using single measures following the standard VAE optimization equa- tion [26] with the β-weighting scheme [41,42]. 
        # The Adam algorithm [43] was used for model training, with a learning rate of 1e−3, β1 = 0.9, β2 = 0.999, and ε = 1e−8.
    search_space = {'epochs': epochs,
                    'batch_size': tune.choice([32, 64, 128, 256, 512]),
                    'lr': tune.loguniform(1e-4, 9e-1),
                    'reg': tune.loguniform(1e-8,1e-1),
                    'emb_size': tune.choice([5, 10, 15, 20]),
                    'enc_hidden_size': tune.choice([64, 128, 256, 512, 1024]),
                    'dec_hidden_size': tune.choice([64, 128, 256, 512, 1024]),
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
                        max_failures=-1,
                        resume=resume,
                        raise_on_failed_trial=False)
    taken = time.time() - start
    best_config = analysis.get_best_config(metric=metric, mode=metric_mode)
    best_result = analysis.best_result
    print(f"Time Taken: {taken:.2f} seconds.")
    print('Best Configuration: {0}\n'.format(metric), best_config)
    print('Best Result: {0}\n'.format(metric), best_result)

    print('Loading Best Checkpoint: Evaluating on test set...')
    best_trial = analysis.get_best_trial(metric=metric, mode=metric_mode, scope='all')
    print('Best Trial: ', best_trial)
    print('Getting Best Checkpoint')
    path_checkpoint = analysis.get_best_checkpoint(trial=best_trial, metric=metric, mode=metric_mode) + 'checkpoint.pth'

    ########################################################################################################################################################

    ########################################################################################################################################################
    # Evaluate on validation/test set with best hyperparameters found

    # load data
    note_past, note_target, note_future, measure_past, measure_mask, measure_future, note_dic, song_id = load_data()
    val_data = INPAINT(data_root, ds_type='val')
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=50, shuffle=False, drop_last=False) # is this batch size ok?
    test_data = INPAINT(data_root, ds_type='test')
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=50, shuffle=False, drop_last=False) # is this batch size ok?
    target_size = val_data.target_size()

    # load best models from best checkpoint
    best_model, best_model_state, best_optimizer = torch.load(path_checkpoint)
    best_model.load_state_dict(best_model_state)

    # run
    val_loss, avg_val_loss, val_perplexity = val(val_loader, best_model)
    test_perplexity = test(test_loader, note_dic, best_model, target_size)

    ########################################################################################################################################################

def main(mode):
    seed = 0
    wd = os.getcwd()
    p = Path().absolute()
    data_root = p/'dataset_split'
    results_root = p/'results/Seq2Seq_LSTM'
    exp_name = p/'results/Seq2Seq_LSTM_tensorboard'
    tensorboard_local_dir = '~'
    use_subset = True
    train_test_split = False
    debug = True

    os.makedirs(results_root, exist_ok=True)
    os.makedirs(exp_name, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    if train_test_split == True:
        # data load
        note_past, note_target, note_future, measure_past, measure_mask, measure_future, note_dic, song_id = load_data()

        data_train, data_test, data_val = train_test_val_split(note_past, note_target, note_future, measure_past, measure_mask, measure_future, song_id)
        data = [data_train, data_val, data_test]
        ds_names = ['train', 'val', 'test']
        # dump to pickle
        for i, ds_name in enumerate(ds_names):
            path = data_root/ds_name
            with open(path, 'wb') as pickle_w:
                write = {b'note_past': data[i][0],
                        b'note_future': data[i][2],
                        b'measure_past': data[i][3],
                        b'measure_future': data[i][5],
                        b'target': data[i][1]}
                pickle.dump(write, pickle_w)
            # open test
            with open(path, 'rb') as pickle_r:
                dict = pickle.load(pickle_r, encoding='bytes')
                print(ds_name, dict[b'target'])
                
    if mode == 'hp_search':
        hp_search(wd, data_root, results_root, exp_name, tensorboard_local_dir, use_subset, debug)
        ## HOW TO VISUALIZE IN TENSORBORD: tensorboard --logdir YOUR_RAY_TUNE_LOCAL_DIRECTORY
    if mode == 'test':
        tester(wd, data_root, results_root, exp_name, tensorboard_local_dir, use_subset, debug) # need to implement



if __name__ == '__main__':
    main(mode='hp_search')