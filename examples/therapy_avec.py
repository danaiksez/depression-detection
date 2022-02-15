import numpy as np
import torch 
import pickle
import torch.nn as nn

from torch.optim import Adam
from ignite.metrics import Loss, Accuracy
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms import Compose
from sklearn.model_selection import KFold

from slp.data.collators import SequenceClassificationCollator
from slp.data.therapy import PsychologicalDataset, TupleDataset
from slp.data.avec_loader_client import AVECDataset
from slp.data.transforms import SpacyTokenizer, ToTokenIds, ToTensor, ReplaceUnknownToken
from slp.modules.basic_model import HierAttNet
from slp.util.embeddings import EmbeddingsLoader
from slp.trainer import SequentialTrainer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
COLLATE_FN = SequenceClassificationCollator(device=DEVICE)
DEBUG = False
MAX_EPOCHS = 50

def trainer_factory(embeddings, idx2word, lex_size, device=DEVICE):
    model = HierAttNet(
        hidden_size, batch_size, num_classes, max_sent_length, len(embeddings), embeddings, idx2word, lex_size, lexicons)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    metrics = {
        'accuracy': Accuracy(),
        'loss': Loss(criterion)
    }

    trainer = SequentialTrainer(
        model,
        optimizer,
        checkpoint_dir='../checkpoints' if not DEBUG else None,
        metrics=metrics,
        non_blocking=True,
        patience=10,
        loss_fn=criterion,
        device=DEVICE)

    return trainer


if __name__ == '__main__':

    ####### Parameters ########
    batch_train = 8
    batch_val = 8
    max_sent_length = 500  #max number of sentences (turns) in transcript - after padding
    max_word_length = 122   #max length of each sentence (turn) - after padding
    num_classes = 2
    batch_size = 8
    hidden_size = 300
    epochs = 40
    lexicons = False
    lex_size = 99

    loader = EmbeddingsLoader('/data/embeddings/glove.840B.300d.txt', 300)
    word2idx, idx2word, embeddings = loader.load()
    embeddings = torch.tensor(embeddings)

    with open("avec.pkl", "rb") as handle:
        _file = pickle.load(handle)

    tokenizer = SpacyTokenizer()
    replace_unknowns = ReplaceUnknownToken()
    to_token_ids = ToTokenIds(word2idx)
    to_tensor = ToTensor(device=DEVICE)

    train = AVECDataset(
        _file, max_word_length,
        transforms = Compose([tokenizer, replace_unknowns, to_token_ids, to_tensor]),
        split='train')
    dev = AVECDataset(
        _file, max_word_length,
        transforms = Compose([tokenizer, replace_unknowns, to_token_ids, to_tensor]),
        split='dev')
    test = AVECDataset(
        _file, max_word_length,
        transforms = Compose([tokenizer, replace_unknowns, to_token_ids, to_tensor]),
        split='test')

    train_loader = DataLoader(
        train, batch_size=batch_train, drop_last=True, num_workers=0, collate_fn=COLLATE_FN)
    dev_loader = DataLoader(
        dev, batch_size=batch_val, num_workers=0, drop_last=True, collate_fn=COLLATE_FN)
    test_loader = DataLoader(
        test, batch_size=batch_val, num_workers=0, drop_last=True, collate_fn=COLLATE_FN)
           
    trainer = trainer_factory(embeddings, idx2word, lex_size, device=DEVICE)
    final_score = trainer.fit(train_loader, dev_loader, epochs=MAX_EPOCHS)


    if DEBUG:
        print("Starting end to end test")
        print("-----------------------------------------------------------------------")
        trainer.fit_debug(train_loader, dev_loader)
        print("Overfitting single batch")
        print("-----------------------------------------------------------------------")
        trainer.overfit_single_batch(train_loader)
