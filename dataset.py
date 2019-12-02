import torchvision.datasets as dset
import torchvision.transforms as transforms
import pickle
import numpy as np
import torch

class CustomCocoCaptions(dset.CocoCaptions):
    def __init__(self, root, annFile, vocab, transform=None):
        super(CustomCocoCaptions, self).__init__(root, annFile)
        self.vocab = vocab.vocabulary_
        self.img_transform = transform
        self.tokenizer = vocab.build_tokenizer()

    def __getitem__(self, index):
        image, captions = super(CustomCocoCaptions, self).__getitem__(index)

        # Just randomly select one during training. Doesn't matter for inference
        # as we use pycococap_eval for evaluation
        # TODO: verify the rand_cap_index is actually different in each iter
        rand_cap_index = torch.randint(0, len(captions), (1,)).item()
        rand_caption = captions[int(rand_cap_index)]

        if self.img_transform is not None:
            image = self.img_transform(image)
        # print(image.shape)
        # Convert caption (string) to word ids.
        # TODO: change this to be the same of vocab preprocessing
        tokens = self.tokenizer(str(rand_caption))
        caption = []
        caption.append(self.vocab.get('<start>', -1) + 1)
        caption.extend([self.vocab.get(token, -1) + 1 for token in tokens])
        caption.append(self.vocab.get('<end>', -1) + 1)
        caption = torch.LongTensor(caption)
        return image, caption
    # __len__ is inherited from CocoCaptions

def custom_collate_fn(batch):
    # Sort a data list by caption length (descending order).
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*batch)

    images = torch.stack(images, dim=0)

    lengths = [len(caption) for caption in captions]
    # Should be fine, as we never has to predict the <pad>, so just set it as 0
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, caption in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = caption[:end]
    return images, targets, lengths

def convert_to_sentence(token_ids, vocab):
    sentence = []
    for token in token_ids:
        if token - 1 < 0:
            continue
        elif vocab[token - 1] == "<padding>":
            continue
        elif vocab[token - 1] == "<start>":
            continue
        elif vocab[token - 1] == "<end>":
            break
        else:
            sentence.append(vocab[token - 1])

    return " ".join(sentence)
