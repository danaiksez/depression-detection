import numpy as np
import torch 
import torch.nn as nn

from torch.optim import Adam
from ignite.metrics import Loss, Accuracy, Precision, Recall
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms import Compose
from sklearn.model_selection import KFold

from slp.data.collators import SequenceClassificationCollator
from slp.data.therapy_title import PsychologicalDataset, TupleDataset
from slp.data.transforms import SpacyTokenizer, ToTokenIds, ToTensor, ReplaceUnknownToken
from slp.modules.hier_att_net_title import HierAttNet
from slp.util.embeddings import EmbeddingsLoader
from slp.trainer.trainer_title_no_validation import SequentialTrainer


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
COLLATE_FN = SequenceClassificationCollator(device=DEVICE)
DEBUG = False
KFOLD = True
MAX_EPOCHS = 50

def dataloaders_from_indices(dataset, train_indices, val_indices, batch_train, batch_val):
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_train,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=COLLATE_FN)
    val_loader = DataLoader(
        dataset,
        batch_size=batch_val,
        sampler=val_sampler,
        drop_last=True,
        collate_fn=COLLATE_FN)

    return train_loader, val_loader

def train_test_split(dataset, batch_train, batch_val,
                     test_size=0.1, shuffle=True, seed=42):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_split = int(np.floor(test_size * dataset_size))
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)

    train_indices = indices[test_split:]
    val_indices = indices[:test_split]

    return dataloaders_from_indices(dataset, train_indices, val_indices, batch_train, batch_val)


def kfold_split(dataset, batch_train, batch_val, k=5, shuffle=True, seed=None):
    kfold = KFold(n_splits=k, shuffle=shuffle, random_state=seed)
    for train_indices, val_indices in kfold.split(dataset):
        yield dataloaders_from_indices(dataset, train_indices, val_indices, batch_train, batch_val)

def trainer_factory(embeddings, device=DEVICE):
    model = HierAttNet(
        hidden_size, batch_size, num_classes, max_sent_length, len(embeddings), embeddings)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0005)
    precision = Precision(average=False)
    recall = Recall(average=False)
    avg_prec = precision.mean()
    avg_rec = recall.mean()
    F1 = (precision * recall * 2/(precision + recall + 1e-7)).mean()

    metrics = {
        'accuracy': Accuracy(),
        'precision': avg_prec,
        'recall': avg_rec,
        'f1': F1,
        'loss': Loss(criterion)
    }

    trainer = SequentialTrainer(
        model,
        optimizer,
        checkpoint_dir='../checkpoints' if not DEBUG else None,
        metrics=metrics,
#        model_checkpoint = '../experiment_model.best.pth',
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
    max_word_length = 150   #max length of each sentence (turn) - after padding
    num_classes = 2
    batch_size = 8
    hidden_size = 300
    epochs = 40

    loader = EmbeddingsLoader('/data/embeddings/glove.840B.300d.txt', 300)
    word2idx, idx2word, embeddings = loader.load()
    embeddings = torch.tensor(embeddings)

    tokenizer = SpacyTokenizer()
    replace_unknowns = ReplaceUnknownToken()
    to_token_ids = ToTokenIds(word2idx)
    to_tensor = ToTensor(device=DEVICE)

    bio = PsychologicalDataset(
        '../../../whole-dataset-twospeakers.csv',
	'../../../psychotherapy_clean/',
        max_word_length,
        text_transforms = Compose([
            tokenizer,
            replace_unknowns,
            to_token_ids,
            to_tensor]))

    if KFOLD:
        cv_scores = []
        import gc
        for train_loader, val_loader in kfold_split(bio, batch_train, batch_val):
            trainer = trainer_factory(embeddings, device=DEVICE)
            fold_score = trainer.fit(train_loader, val_loader, epochs=MAX_EPOCHS)
            cv_scores.append(fold_score)
            del trainer
            gc.collect()
        final_score = float(sum(cv_scores)) / len(cv_scores)
    else:
        train_loader, val_loader = train_test_split(bio, batch_train, batch_val)
        trainer = trainer_factory(embeddings, device=DEVICE)
        final_score = trainer.fit(train_loader, val_loader, epochs=MAX_EPOCHS)

    print(f'Final score: {final_score}')

    if DEBUG:
        print("Starting end to end test")
        print("-----------------------------------------------------------------------")
        trainer.fit_debug(train_loader, val_loader)
        print("Overfitting single batch")
        print("-----------------------------------------------------------------------")
        trainer.overfit_single_batch(train_loader)
