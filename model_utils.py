import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from dataset import MyDataset
from model import EncoderCNN, DecoderRNN
from data_utils import get_gender_nouns, get_test_indices
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from utils import load_obj
import math
import warnings

# Frequency of printing batch loss while training/validating. 
print_interval = 1000

'''
Default data loader, train, validate functions
'''
def load_data(image_ids, image_folder_path, mode):
    # Initiate instance of MyDataset class
    num_workers = 0
    dataset = MyDataset(image_ids, image_folder_path, mode = mode)
    if mode == 'train' or mode == 'val':
        indices = dataset.get_indices()
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader = data.DataLoader(dataset=dataset, num_workers=num_workers,\
                    batch_sampler=\
                    data.sampler.BatchSampler(sampler=initial_sampler,\
                    batch_size=dataset.batch_size,drop_last=False))
    else: # if test, initial sampler is not necessary
        data_loader = data.DataLoader(dataset=dataset, num_workers=num_workers,\
                 batch_size=dataset.batch_size, shuffle = True)  
    return data_loader

def train(train_loader, encoder, decoder, criterion, optimizer, vocab_size,
          epoch, total_step, start_step=1, start_loss=0.0):
    """Train the model for one epoch using the provided parameters. Save 
    checkpoints every 100 steps. Return the epoch's average train loss."""

    # Switch to train mode
    encoder.train()
    decoder.train()

    # Keep track of train loss
    total_loss = start_loss

    # Start time for every 100 steps
    start_train_time = time.time()

    for i_step in range(start_step, total_step + 1):
        # Randomly sample a caption length, and sample indices with that length
        indices = train_loader.dataset.get_indices()
        # Create a batch sampler to retrieve a batch with the sampled indices
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        train_loader.batch_sampler.sampler = new_sampler

        # Obtain the batch
        for batch in train_loader:
            images, captions = batch[0], batch[1]
            break 
        # Move to GPU if CUDA is available
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
        # Pass the inputs through the CNN-RNN model
        features = encoder(images)
        outputs = decoder(features, captions)

        # Calculate the batch loss
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        # Zero the gradients. Since the backward() function accumulates 
        # gradients, and we don’t want to mix up gradients between minibatches,
        # we have to zero them out at the start of a new minibatch
        optimizer.zero_grad()
        # Backward pass to calculate the weight gradients
        loss.backward()
        # Update the parameters in the optimizer
        optimizer.step()

        total_loss += loss.item()

        # Get training statistics
        stats = "Epoch %d, Train step [%d/%d], %ds, Loss: %.4f, Perplexity: %5.4f" \
                % (epoch, i_step, total_step, time.time() - start_train_time,
                   loss.item(), np.exp(loss.item()))
        # Print training statistics (on same line)
        print("\r" + stats, end="")
        sys.stdout.flush()

        # Print training stats (on different line), reset time and save checkpoint
        if i_step % print_interval == 0:
            print("\r" + stats)
            filename = os.path.join("./models", "train-model-{}{}.pkl".format(epoch, i_step))
            save_checkpoint(filename, encoder, decoder, optimizer, total_loss, epoch, i_step)
            start_train_time = time.time()
            
    return total_loss / total_step
            
def validate(val_loader, encoder, decoder, criterion, vocab, epoch, 
             total_step, start_step=1, start_loss=0.0, start_bleu=0.0):
    """Validate the model for one epoch using the provided parameters. 
    Return the epoch's average validation loss and Bleu-4 score."""

    # Switch to validation mode
    encoder.eval()
    decoder.eval()

    # Initialize smoothing function
    smoothing = SmoothingFunction()

    # Keep track of validation loss and Bleu-4 score
    total_loss = start_loss
    total_bleu_4 = start_bleu

    # Start time for every 100 steps
    start_val_time = time.time()

    # Disable gradient calculation because we are in inference mode
    with torch.no_grad():
        for i_step in range(start_step, total_step + 1):
            # Randomly sample a caption length, and sample indices with that length
            indices = val_loader.dataset.get_indices()
            # Create a batch sampler to retrieve a batch with the sampled indices
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            val_loader.batch_sampler.sampler = new_sampler

            # Obtain the batch
            for batch in val_loader:
                images, captions = batch[0], batch[1]
                break 

            # Move to GPU if CUDA is available
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
            
            # Pass the inputs through the CNN-RNN model
            features = encoder(images)
            outputs = decoder(features, captions)

            # Calculate the total Bleu-4 score for the batch
            batch_bleu_4 = 0.0
            # Iterate over outputs. Note: outputs[i] is a caption in the batch
            # outputs[i, j, k] contains the model's predicted score i.e. how 
            # likely the j-th token in the i-th caption in the batch is the 
            # k-th token in the vocabulary.
            for i in range(len(outputs)):
                predicted_ids = []
                for scores in outputs[i]:
                    # Find the index of the token that has the max score
                    predicted_ids.append(scores.argmax().item())
                # Convert word ids to actual words
                predicted_word_list = word_list(predicted_ids, vocab)
                caption_word_list = word_list(captions[i].cpu().numpy(), vocab)
                # Calculate Bleu-4 score and append it to the batch_bleu_4 list
                batch_bleu_4 += sentence_bleu([caption_word_list], 
                                               predicted_word_list, 
                                               smoothing_function=smoothing.method1)
            total_bleu_4 += batch_bleu_4 / len(outputs)

            # Calculate the batch loss
            loss = criterion(outputs.view(-1, len(vocab)), captions.view(-1))
            total_loss += loss.item()
            
            # Get validation statistics
            stats = "Epoch %d, Val step [%d/%d], %ds, Loss: %.4f, Perplexity: %5.4f, Bleu-4: %.4f" \
                    % (epoch, i_step, total_step, time.time() - start_val_time,
                       loss.item(), np.exp(loss.item()), batch_bleu_4 / len(outputs))

            # Print validation statistics (on same line)
            print("\r" + stats, end="")
            sys.stdout.flush()

            # Print validation statistics (on different line) and reset time
            if i_step % print_interval == 0:
                print("\r" + stats)
                filename = os.path.join("./models", "val-model-{}{}.pkl".format(epoch, i_step))
                save_val_checkpoint(filename, encoder, decoder, total_loss, total_bleu_4, epoch, i_step)
                start_val_time = time.time()
                
        return total_loss / total_step, total_bleu_4 / total_step

'''
Gender-neutral train and validate functions
'''
def gender_neutral_loss_setup(train_loader, vocab_size):
    gender_nouns=get_gender_nouns()
    male_tags=gender_nouns['male']
    female_tags=gender_nouns['female']
    neutral_tags=gender_nouns['neutral']
    vocab = train_loader.dataset.vocab
    
    ##Construct fonehot, monehot and nonehot vectors from the vocab
    vocab_words=word_list_vocab(list(range(vocab_size)),vocab)

    global fonehot, monehot, nonehot
    fonehot = torch.zeros(len(vocab))
    monehot =torch.zeros(len(vocab))
    nonehot=torch.zeros(len(vocab))
    if torch.cuda.is_available():
       fonehot = fonehot.to("cuda:0")
       monehot = monehot.to("cuda:0")
       nonehot = nonehot.to("cuda:0")    
    
    for i,word in enumerate(vocab_words):
        ##female one hot vector (all female associated words are tagged 1 and the rest are 0's)
        if any(word==fem_word for fem_word in female_tags):
            fonehot[i]=1
        ##male one hot vector
        if any(word==male_word for male_word in male_tags):
            monehot[i]=1
        ##neutral one hot vector
        if any(word==neut_word for neut_word in neutral_tags):
            nonehot[i]=1

def word_list_vocab(word_idx_list, vocab):
    """Take a list of word ids and a vocabulary from a dataset as inputs
    and return the corresponding words as a list.
    """
    word_list = []
    for i in range(len(word_idx_list)):
        vocab_id = word_idx_list[i]
        word = vocab.idx2word[vocab_id]
        word_list.append(word)
    return word_list

def cross_entr(outputs,targets):
    loss = -1 / outputs.shape[1] * (torch.matmul(outputs,torch.log(targets.t())) + torch.matmul( 1 - outputs, torch.log(1 - targets.t())))

def loss_function(outputs,captions,female_ops,male_ops,neutral_opsm, vocab_size):
    ##Cross entropy loss
    ce_loss=criterion(outputs.view(-1, vocab_size), captions.view(-1))
    ##Log loss that penalizes biased gender ratio (i.e., male/female or female/male)
    b_loss = torch.mean(torch.abs(torch.log((torch.exp(female_ops) + 0.00001) / (torch.exp(male_ops) + 0.00001))))   
    
    return ce_loss + b_loss

def train_gender_neutral(train_loader, encoder, decoder, criterion, optimizer, vocab_size,
          epoch, total_step, start_step=1, start_loss=0.0):
    """Train the model for one epoch using the provided parameters. Save 
    checkpoints every 100 steps. Return the epoch's average train loss."""

    # Switch to train mode
    encoder.train()
    decoder.train()

    # Keep track of train loss
    total_loss = start_loss

    # Start time for every 100 steps
    start_train_time = time.time()

    for i_step in range(start_step, total_step + 1):
        # Randomly sample a caption length, and sample indices with that length
        indices = train_loader.dataset.get_indices()
        # Create a batch sampler to retrieve a batch with the sampled indices
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        train_loader.batch_sampler.sampler = new_sampler

        # Obtain the batch
        for batch in train_loader:
            images, captions = batch[0], batch[1]
            break 
        # Move to GPU if CUDA is available
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
        # Pass the inputs through the CNN-RNN model
        features = encoder(images)
        outputs = decoder(features, captions)
        
        female_ops=0
        male_ops=0
        neutral_ops=0
        
        for i in range(len(outputs)):
                for scores in outputs[i]:
                    #0 out all non-gendered words in scores and find the sum of the scores for all gendered words
                    if torch.cuda.is_available():
                        scores = scores.to("cuda:0")
                    female_ops+=torch.matmul(fonehot,scores.t())
                    male_ops+=torch.matmul(monehot,scores.t())
                    neutral_ops+=torch.matmul(nonehot,scores.t())
                female_ops/=len(outputs[i])
                male_ops/=len(outputs[i])
                neutral_ops/=len(outputs[i])

        # Calculate the batch loss 
        loss=loss_function(outputs,captions,female_ops,male_ops,neutral_ops, vocab_size)
        
        # Zero the gradients. Since the backward() function accumulates gradients, and we don’t want to mix up gradients between minibatches,
        # we have to zero them out at the start of a new minibatch
        optimizer.zero_grad()
        # Backward pass to calculate the weight gradients
        loss.backward()
        # Update the parameters in the optimizer
        optimizer.step()

        total_loss += loss.item()

        # Get training statistics
        stats = "Epoch %d, Train step [%d/%d], %ds, Loss: %.4f, Perplexity: %5.4f" \
                % (epoch, i_step, total_step, time.time() - start_train_time,
                   loss.item(), np.exp(loss.item()))
        # Print training statistics (on same line)
        print("\r" + stats, end="")
        sys.stdout.flush()

        # Print training stats (on different line), reset time and save checkpoint
        if i_step % print_interval == 0:
            print("\r" + stats)
            filename = os.path.join("./models", "train-model-{}{}.pkl".format(epoch, i_step))
            save_checkpoint(filename, encoder, decoder, optimizer, total_loss, epoch, i_step)
            start_train_time = time.time()
            
    return total_loss / total_step
            
def validate_gender_neutral(val_loader, encoder, decoder, criterion, vocab, vocab_size, epoch, 
             total_step, start_step=1, start_loss=0.0, start_bleu=0.0):
    """Validate the model for one epoch using the provided parameters. 
    Return the epoch's average validation loss and Bleu-4 score."""

    # Switch to validation mode
    encoder.eval()
    decoder.eval()

    # Initialize smoothing function
    smoothing = SmoothingFunction()

    # Keep track of validation loss and Bleu-4 score
    total_loss = start_loss
    total_bleu_4 = start_bleu

    # Start time for every 100 steps
    start_val_time = time.time()

    # Disable gradient calculation because we are in inference mode
    with torch.no_grad():
        for i_step in range(start_step, total_step + 1):
            # Randomly sample a caption length, and sample indices with that length
            indices = val_loader.dataset.get_indices()
            # Create a batch sampler to retrieve a batch with the sampled indices
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            val_loader.batch_sampler.sampler = new_sampler

            # Obtain the batch
            for batch in val_loader:
                images, captions = batch[0], batch[1]
                break 

            # Move to GPU if CUDA is available
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
            
            # Pass the inputs through the CNN-RNN model
            features = encoder(images)
            outputs = decoder(features, captions)
            
            female_ops=0
            male_ops=0
            neutral_ops=0

            # Calculate the total Bleu-4 score for the batch
            batch_bleu_4 = 0.0
            # Iterate over outputs. Note: outputs[i] is a caption in the batch
            # outputs[i, j, k] contains the model's predicted score i.e. how 
            # likely the j-th token in the i-th caption in the batch is the 
            # k-th token in the vocabulary.
            for i in range(len(outputs)):
                predicted_ids = []
                for scores in outputs[i]:
                    if torch.cuda.is_available():
                        scores = scores.to("cuda:0")
                    # Find the index of the token that has the max score
                    predicted_ids.append(scores.argmax().item())
                    #0 out all non-gendered words in scores and find the sum of the scores for all gendered words
                    female_ops+=torch.matmul(fonehot,scores.t())
                    male_ops+=torch.matmul(monehot,scores.t())
                    neutral_ops+=torch.matmul(nonehot,scores.t())
                female_ops/=len(outputs[i])
                male_ops/=len(outputs[i])
                neutral_ops/=len(outputs[i])
                    
                # Convert word ids to actual words
                predicted_word_list = word_list(predicted_ids, vocab)
                caption_word_list = word_list(captions[i].numpy(), vocab)
                # Calculate Bleu-4 score and append it to the batch_bleu_4 list
                batch_bleu_4 += sentence_bleu([caption_word_list], 
                                               predicted_word_list, 
                                               smoothing_function=smoothing.method1)
            total_bleu_4 += batch_bleu_4 / len(outputs)

            # Calculate the batch loss
            loss=loss_function(outputs,captions,female_ops,male_ops,neutral_ops, vocab_size)
            total_loss += loss.item()
            
            # Get validation statistics
            stats = "Epoch %d, Val step [%d/%d], %ds, Loss: %.4f, Perplexity: %5.4f, Bleu-4: %.4f" \
                    % (epoch, i_step, total_step, time.time() - start_val_time,
                       loss.item(), np.exp(loss.item()), batch_bleu_4 / len(outputs))

            # Print validation statistics (on same line)
            print("\r" + stats, end="")
            sys.stdout.flush()

            # Print validation statistics (on different line) and reset time
            if i_step % print_interval == 0:
                print("\r" + stats)
                filename = os.path.join("./models", "val-model-{}{}.pkl".format(epoch, i_step))
                save_val_checkpoint(filename, encoder, decoder, total_loss, total_bleu_4, epoch, i_step)
                start_val_time = time.time()
                
        return total_loss / total_step, total_bleu_4 / total_step

'''
Umbrella training model function using all above functions
'''
def loader_setup(train_image_ids, val_image_ids, image_folder_path):
    train_loader = load_data(train_image_ids, image_folder_path, mode = 'train')
    val_loader = load_data(val_image_ids, image_folder_path, mode = 'val')
    print('\n\nLoaders successfully set up . . .')

    # Sample a subset of captions with a randomized length
    indices = train_loader.dataset.get_indices()

    # Create and assign batch sampler to retrieve a batch with the sampled indices
    new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
    train_loader.batch_sampler.sampler = new_sampler

    # Obtain the batch
    for batch in train_loader:
        images, captions = batch[0], batch [1]
    
    print('\n\nChecking shape of sample batch . . .')
    print('images.shape:', images.shape)
    print('captions.shape:', captions.shape)
    return train_loader, val_loader

def train_model(train_image_ids, val_image_ids, image_folder_path, batch_size, embed_size, hidden_size, num_epochs, mode = 'reg'):
    assert mode in ['reg','gender_neutral']
    #reg: regular training loss function
    #gender_neural: alternative loss function that penalizes gender bias
    if torch.cuda.is_available():
        global device
        device = "cuda:0"
        
    # Set up
    train_loader, val_loader = loader_setup(train_image_ids, val_image_ids, image_folder_path)
    vocab = train_loader.dataset.vocab
    vocab_size = len(train_loader.dataset.vocab)

    # Initialize model
    print("\n\nInitialize model . . .")
    # Initialize CNN and RNN
    global encoder, decoder, criterion, optimizer
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

    # Use GPU if available
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()  
        
    # Define the loss function
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    # Specify the learnable parameters of the model
    params = list(decoder.parameters()) + list(encoder.embed.parameters()) + list(encoder.bn.parameters())

    # Define the optimizer
    optimizer = torch.optim.Adam(params=params, lr=0.001)

    # Calculate total number of training steps per epoch
    print("\n\nCalculate total number of steps per epoch . . .")
    total_train_step = math.ceil(len(train_loader.dataset.captions_len) / train_loader.batch_sampler.batch_size)
    print ("Number of training steps:", total_train_step)
    total_val_step = math.ceil(len(val_loader.dataset.captions_len) / val_loader.batch_sampler.batch_size)
    print ("Number of training steps:", total_val_step)

    # Training
    print("\n\nTraining model . . .")
    train_losses = []
    val_losses = []
    val_bleus = []
    best_val_bleu = float("-INF")

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        if mode == 'reg':
            train_loss = train(train_loader, encoder, decoder, criterion, optimizer, vocab_size, epoch, total_train_step)
            val_loss, val_bleu = validate(val_loader, encoder, decoder, criterion, train_loader.dataset.vocab, epoch, total_val_step)
        else: # use gender_neutral train function
            gender_neutral_loss_setup(train_loader, vocab_size)
            train_loss = train_gender_neutral(train_loader, encoder, decoder, criterion, optimizer, vocab_size, epoch, total_train_step)
            val_loss, val_bleu = validate_gender_neutral(val_loader, encoder, decoder, criterion, vocab, vocab_size, epoch, total_val_step)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_bleus.append(val_bleu)
        if val_bleu > best_val_bleu:
            print ("Validation Bleu-4 improved from {:0.4f} to {:0.4f}, saving model to best-model.pkl".
                format(best_val_bleu, val_bleu))
            best_val_bleu = val_bleu
            filename = os.path.join("./models", "best-model.pkl")
            save_epoch(filename, encoder, decoder, optimizer, train_losses, val_losses, 
                    val_bleu, val_bleus, epoch)
        else:
            print ("Validation Bleu-4 did not improve, saving model to model-{}.pkl".format(epoch))
        # Save the entire model anyway, regardless of being the best model so far or not
        filename = os.path.join("./models", "model-{}.pkl".format(epoch))
        save_epoch(filename, encoder, decoder, optimizer, train_losses, val_losses, 
                val_bleu, val_bleus, epoch)
        print ("Epoch [%d/%d] took %ds" % (epoch, num_epochs, time.time() - start_time))
        if epoch > 5:
            # Stop if the validation Bleu doesn't improve for 3 epochs
            if early_stopping(val_bleus, 3):
                break
        start_time = time.time()

'''
Helper functions for model training process- save checkpoint, save validation checkpoitn, save epoch, early stopping
'''
def save_checkpoint(filename, encoder, decoder, optimizer, total_loss, epoch, train_step=1):
    """Save the following to filename at checkpoints: encoder, decoder,
    optimizer, total_loss, epoch, and train_step."""
    torch.save({"encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "total_loss": total_loss,
                "epoch": epoch,
                "train_step": train_step,
               }, filename)

def save_val_checkpoint(filename, encoder, decoder, total_loss,
    total_bleu_4, epoch, val_step=1):
    """Save the following to filename at checkpoints: encoder, decoder,
    total_loss, total_bleu_4, epoch, and val_step"""
    torch.save({"encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "total_loss": total_loss,
                "total_bleu_4": total_bleu_4,
                "epoch": epoch,
                "val_step": val_step,
               }, filename)

def save_epoch(filename, encoder, decoder, optimizer, train_losses, val_losses, 
               val_bleu, val_bleus, epoch):
    """Save at the end of an epoch. Save the model's weights along with the 
    entire history of train and validation losses and validation bleus up to 
    now, and the best Bleu-4."""
    torch.save({"encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_bleu": val_bleu,
                "val_bleus": val_bleus,
                "epoch": epoch
               }, filename)

def early_stopping(val_bleus, patience=3):
    """Check if the validation Bleu-4 scores no longer improve for 3 
    (or a specified number of) consecutive epochs."""
    # The number of epochs should be at least patience before checking
    # for convergence
    if patience > len(val_bleus):
        return False
    latest_bleus = val_bleus[-patience:]
    # If all the latest Bleu scores are the same, return True
    if len(set(latest_bleus)) == 1:
        return True
    max_bleu = max(val_bleus)
    if max_bleu in latest_bleus:
        # If one of recent Bleu scores improves, not yet converged
        if max_bleu not in val_bleus[:len(val_bleus) - patience]:
            return False
        else:
            return True
    # If none of recent Bleu scores is greater than max_bleu, it has converged
    return True

def word_list(word_idx_list, vocab):
    """Take a list of word ids and a vocabulary from a dataset as inputs
    and return the corresponding words as a list.
    """
    word_list = []
    for i in range(len(word_idx_list)):
        vocab_id = word_idx_list[i]
        word = vocab.idx2word[vocab_id]
        if word == vocab.end_word:
            break
        if word != vocab.start_word:
            word_list.append(word)
    return word_list

def clean_sentence(word_idx_list, vocab):
    """Take a list of word ids and a vocabulary from a dataset as inputs
    and return the corresponding sentence (as a single Python string).
    """
    sentences = []
    for i in range(len(word_idx_list)):
        sentence = []
        for vocab_id in word_idx_list[i]:
            word = vocab.idx2word[vocab_id]
            if word == vocab.end_word:
                break
            if word != vocab.start_word:
                sentence.append(word)
        sentence = " ".join(sentence)
        sentences.append(sentence)
    return sentences

def get_prediction(data_loader, encoder, decoder, vocab):
    """Loop over images in a dataset and print model's top three predicted 
    captions using beam search."""
    orig_image, image = next(iter(data_loader))
    plt.imshow(np.squeeze(orig_image))
    plt.title("Sample Image")
    plt.show()
    if torch.cuda.is_available():
        image = image.cuda()
    features = encoder(image).unsqueeze(1)

    print ("Top captions using beam search:")
    outputs = decoder.sample_beam_search(features)
    
    # Print maximum the top 5 predictions
    num_sents = min(len(outputs), 5)
    for output in outputs[:num_sents]:
        sentence = clean_sentence(output, vocab)
        print (sentence)

def predict_and_show_image(image_folder_path, vocab_path = '', model_path = '', training_image_ids_path = '', embed_size = 256, hidden_size = 512, mode = 'balanced_clean'):
#     if torch.cuda.is_available():
#         checkpoint = torch.load('./models/best-model.pkl', map_location="cuda")
#     else:
#         checkpoint = torch.load('./models/best-model.pkl', map_location="cpu")

    sample_size = 1
    
    # Get model
    if model_path == '': # if not specified, assume it is best model saved in models
        model_path = './models/best-model.pkl'
    if torch.cuda.is_available() == True:
        checkpoint = torch.load('./models/best-model.pkl')
    else:
        checkpoint = torch.load('./models/best-model.pkl', map_location='cpu')
    print(f'Best model is loaded from {model_path} . . .')
    
    # Get the vocabulary and its size
    if vocab_path == '': # if not specified, assume it is the vocab pickle saved in object
        vocab = load_obj('vocab')
    else:
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
    vocab_size = len(vocab)
    
    # Get the vocabulary and its size
    if training_image_ids_path == '': # if not specified, assume it is the vocab pickle saved in object
        training_image_ids = load_obj('training_image_ids')
    else:
        with open(vocab_path, 'rb') as f:
            training_image_ids = pickle.load(f)
    
    test_image_ids = get_test_indices(sample_size, training_image_ids, mode = mode)
    image_id = list(test_image_ids.keys())[0]
    test_loader = load_data(test_image_ids.keys(), image_folder_path, mode = 'test')
    original_image, image = next(iter(test_loader))
    transformed_image = image.numpy()
    transformed_image = np.squeeze(transformed_image)\
                    .transpose((1, 2, 0))
    
    # Print sample image, before and after pre-processing
    print(f'\nTest_image_id: {image_id}')
    plt.imshow(np.squeeze(original_image))
    plt.title('Test image- original')
    plt.show()
    plt.imshow(transformed_image)
    plt.title('Test image- transformed')
    plt.show()

    

    # Initialize the encoder and decoder, and set each to inference mode
    encoder = EncoderCNN(embed_size)
    encoder.eval()
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    decoder.eval()

    # Load the pre-trained weights
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    
    # Move models to GPU if CUDA is available.
#     if torch.cuda.is_available():
#         encoder.cuda()
#         decoder.cuda()

    features = encoder(image).unsqueeze(1)
    output = decoder.sample_beam_search(features)
    sentences = clean_sentence(output, vocab)
    print('Predicted caption: \n')
    for sentence in set(sentences):
        print(f'{sentence}')
        
    original_captions = test_image_ids[image_id]
    print('\n\nOriginal captions labelled by human annotators: \n')
    for caption in set(original_captions):
        print(caption)
