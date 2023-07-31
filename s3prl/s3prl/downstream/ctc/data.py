import torch
import torchaudio
from functools import partial
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

EVAL_BATCH_SIZE = 1


def collect_audio_batch(batch, split, half_batch_size_wav_len=300000):
    '''Collects a batch, should be list of tuples (audio_path <str>, list of int token <list>) 
       e.g. [(file1,txt1),(file2,txt2),...]
    '''
    def audio_reader(filepath):
        wav, sample_rate = torchaudio.load(filepath)
        return wav.reshape(-1)

    # Bucketed batch should be [[(file1,txt1),(file2,txt2),...]]
    if type(batch[0]) is not tuple:
        batch = batch[0]

    # Make sure that batch size is reasonable
    first_len = audio_reader(str(batch[0][0])).size(0)
    if 'train' in split:
        if first_len > half_batch_size_wav_len and len(batch) > 1:
            batch = batch[:len(batch)//2]

    # Read batch
    file, audio_feat, audio_len, text = [], [], [], []
    with torch.no_grad():
        for b in batch:
            file.append(str(b[0]).split('/')[-1].split('.')[0])
            feat = audio_reader(str(b[0])).numpy()
            audio_feat.append(feat)
            audio_len.append(len(feat))
            text.append(torch.LongTensor(b[1]).numpy())

    # Descending audio length within each batch
    audio_len, file, audio_feat, text = zip(*[(feat_len, f_name, feat, txt)
                                              for feat_len, f_name, feat, txt in sorted(zip(audio_len, file, audio_feat, text), reverse=True, key=lambda x:x[0])])

    return audio_feat, text, file


def create_dataset(split, tokenizer, name, bucketing, batch_size, **kwargs):
    ''' Interface for creating all kinds of dataset'''

    # Recognize corpus
    if name.lower() == "librispeech":
        from .corpus.librispeech import LibriDataset as Dataset
    elif name.lower() == "snips":
        from .corpus.snips import SnipsDataset as Dataset
    elif name.lower() == 'libriphone':
        from .corpus.libriphone import LibriPhoneDataset as Dataset
    elif name.lower() in {"common_voice", "sbcsae"}:
        from .corpus.common_voice import CommonVoiceDataset as Dataset
    else:
        raise NotImplementedError

    if 'train' in split:
        kwargs["ratio"] = 1.0
        kwargs["offset"] = 0
        loader_bs = 1 if bucketing else batch_size
        bucket_size = batch_size if bucketing else 1
        print(f'loader_bs: {loader_bs}, batch_size: {batch_size}')
        dataset = Dataset(kwargs['train'], tokenizer, bucket_size, **kwargs)
    else:
        loader_bs = EVAL_BATCH_SIZE
        dataset = Dataset(kwargs[split], tokenizer, 1, **kwargs)

    return dataset, loader_bs


def load_dataset(split, tokenizer, corpus, switch_ratio=0.5):
    ''' Prepare dataloader for training/validation'''
    # real_split = split
    # split = 'train' if split == 'switch' else split
    collate_fn = partial(collect_audio_batch, split=split)
    num_workers = corpus.pop('num_workers', 12)
    dataset, loader_bs = create_dataset(split, tokenizer, num_workers=num_workers, **corpus)
    
    if split == 'train':
        train_dataset, switch_dataset = torch.utils.data.random_split(dataset, [1 - switch_ratio, switch_ratio])
        
        train_sampler = DistributedSampler(train_dataset) if (is_initialized() and len(train_dataset) > 0) else None
        train_dataloader = DataLoader(train_dataset, batch_size=loader_bs, shuffle=(train_sampler is None and len(train_dataset) > 0),
                                sampler=train_sampler, collate_fn=collate_fn, num_workers=num_workers)
        
        switch_sampler = DistributedSampler(switch_dataset) if (is_initialized() and len(switch_dataset) > 0) else None
        switch_dataloader = DataLoader(switch_dataset, batch_size=loader_bs, shuffle=(switch_sampler is None and len(switch_dataset) > 0),
                                sampler=switch_sampler, collate_fn=collate_fn, num_workers=num_workers)
        
        return {
            "train": train_dataloader,
            "switch": switch_dataloader
        }
    else:
        # dev set & test set
        dataloader = DataLoader(dataset, batch_size=loader_bs, shuffle=False,
                                collate_fn=collate_fn, num_workers=num_workers)
    return dataloader
