import torchvision.datasets as dset
import torchvision.transforms as transforms
import pickle
import numpy as np
import torch
import json
import os

from torch.utils.data import DataLoader

from model import Show_and_tell

class CustomCocoCaptions(dset.CocoCaptions):
    def __init__(self, root, annFile, vocab, transform=None):
        super(CustomCocoCaptions, self).__init__(root, annFile)
        self.vocab = vocab.vocabulary_
        self.img_transform = transform
        self.tokenizer = vocab.build_tokenizer()

    def __getitem__(self, index):
        image, captions = super(CustomCocoCaptions, self).__getitem__(index)

        img_id = self.ids[index]

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
        return image, caption, img_id
    # __len__ is inherited from CocoCaptions

def custom_collate_fn(batch):
    # Sort a data list by caption length (descending order).
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, img_ids = zip(*batch)

    images = torch.stack(images, dim=0)

    lengths = [len(caption) for caption in captions]
    # Should be fine, as we never has to predict the <pad>, so just set it as 0
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, caption in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = caption[:end]
    return images, targets, lengths, img_ids

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



data_dir = 'cocodata'
vocab_path = os.path.join(data_dir, 'vocab.pkl')
test_path = os.path.join(data_dir, 'val2017')
test_annotation_path = os.path.join(data_dir, 'annotations/captions_val2017.json')
save_path = 'output'

if not os.path.exists(vocab_path):
    raise('Vocab path does not exist')
if not (os.path.exists(test_path) and os.path.exists(test_annotation_path)):
    raise('Val path does not exist')
if not os.path.exists(save_path):
    os.makedirs(save_path)

# open vocab file
with open(vocab_path, 'rb') as file:
    vectorizer = pickle.load(file)

vocab_size = len(vectorizer.get_feature_names())
vectorizer.vocabulary_["<padding>"] = vocab_size
vectorizer.vocabulary_["<start>"] = vocab_size + 1
vectorizer.vocabulary_["<end>"] = vocab_size + 2

# Similar to test_loader
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
test_custom_coco_cap = CustomCocoCaptions(test_path, test_annotation_path,
                                           vectorizer, test_transforms)
test_loader = DataLoader(dataset=test_custom_coco_cap,
                         batch_size=32,
                         shuffle=False,
                         num_workers=4,
                         collate_fn=custom_collate_fn)

model = Show_and_tell(vocab_size=vocab_size+4, hidden_units=1024)
checkpoint = torch.load(os.path.join(save_path, 'model_entire_network.pth'))
model.load_state_dict(checkpoint['state_dict'])
model.cuda()
model.eval()

l = []
for i, (images, _, lengths, img_ids) in enumerate(test_loader, 1):
    if i%200 == 0:
        print('{}/{}'.format(i, len(test_loader)))
    images = images.cuda()
    # images = images.to(device)

    with torch.no_grad():
        predicted_ids, outputs = model(images)

    # bs x max_sequence_length: predicted_ids
    for j in range(len(predicted_ids)):
        d = {}
        sample_sentence = convert_to_sentence(predicted_ids[j], vectorizer.get_feature_names())
        d['image_id'] = img_ids[j]
        d['caption'] = sample_sentence
        l.append(d)


with open(os.path.join(save_path, 'results.json'), 'w') as f:
    json.dump(l, f)
