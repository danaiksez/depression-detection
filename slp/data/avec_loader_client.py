import csv
import pickle

from torch.utils.data import Dataset
from torchvision.transforms import Compose


def pad_sequence(sequences, batch_first=False, padding_len=None, padding_value=0):
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()

    trailing_dims = max_size[1:]
    if padding_len is not None:
        max_len = padding_len
    else:
        max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        if tensor.size(0) > padding_len:
            tensor = tensor[:padding_len]
        length = min(tensor.size(0), padding_len)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor
    return out_tensor


class AVECDataset(Dataset):
    def __init__(self, pickle, max_word_len, transforms=None, split='train'):

        self.pickle = pickle
        self.max_word_len = max_word_len
        self.transforms = transforms
        self.split = split
        self.transcripts, self.labels, self.speakers = self.get_speakers_transcripts_labels(self.pickle, self.split)

        self.therapist_turns = ['Ellie', 'ELLIE']
        self.client_turns = ['CLIENT', 'Client', 'Participant']
        self.preprocessed = [self.preprocess(i) for i in range(len(self.transcripts))]

    def get_speakers_transcripts_labels(self, _file, split):
        speakers = []
        transcripts = []
        labels = []

        length = len(_file[split])
        transcripts = [_file[split][i]['text'] for i in range(length)]
        labels = [_file[split][i]['label'] for i in range(length)]
        speakers = [_file[split][i]['speakers'] for i in range(length)]

        return transcripts, labels, speakers

    def __len__(self):
        return len(self.transcripts)

    def preprocess(self, idx):
        preprocessed_text = self.transcripts[idx]
        speakers = self.speakers[idx]
        label = self.labels[idx]

        lista = []
        if self.transforms is not None:
            for i in range(len(preprocessed_text)):
                if speakers[i] in self.client_turns:
                    s = self.transforms(preprocessed_text[i])
                    lista.append(s)

        preprocessed_text = pad_sequence(lista, batch_first=True, padding_len = self.max_word_len)
        return preprocessed_text, speakers, label

    def __getitem__(self, idx):
        return self.preprocessed[idx]
