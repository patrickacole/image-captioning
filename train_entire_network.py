import os
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

import dataset
from dataset import CustomCocoCaptions
import model
from model import Show_and_tell

data_dir = 'cocodata'
vocab_path = os.path.join(data_dir, 'vocab.pkl')
train_path = os.path.join(data_dir, 'train2017')
train_annotation_path = os.path.join(data_dir, 'annotations/captions_train2017.json')
test_path = os.path.join(data_dir, 'val2017')
test_annotation_path = os.path.join(data_dir, 'annotations/captions_val2017.json')
save_path = 'output'

if not os.path.exists(vocab_path):
    raise('Vocab path does not exist')
if not (os.path.exists(train_path) and os.path.exists(train_annotation_path)):
    raise('Train path does not exist')
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

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# change batch size to be larger than 32 for actual training
# for purpose of testing train loop it is 32.

train_custom_coco_cap = CustomCocoCaptions(train_path, train_annotation_path,
                                           vectorizer, train_transforms)
train_loader = DataLoader(dataset=train_custom_coco_cap,
                         batch_size=64,
                         shuffle=True,
                         num_workers=4,
                         collate_fn=dataset.custom_collate_fn)
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
                         batch_size=64,
                         shuffle=True,
                         num_workers=4,
                         collate_fn=dataset.custom_collate_fn)

model = Show_and_tell(vocab_size=vocab_size+4, hidden_units=1024)
# checkpoint = torch.load(os.path.join(save_path, 'model.pth'))
# model.load_state_dict(checkpoint['state_dict'])
#model.to(device)
model.cuda()
# enable all of the layers in the cnn_embedding
# for param in model.cnn_embedding.parameters():
#     param.requires_grad = True

# get all of the parameters for the optimizer
params = []
for k, v in model.named_parameters():
    if v.requires_grad == True:
        params.append(v)

#device = torch.device(("cpu","cuda:0")[torch.cuda.is_available()])
learning_rate = 0.001
optimizer = torch.optim.Adam(params, lr=learning_rate)
criterion = nn.CrossEntropyLoss()
epochs = 30
# every `test_epoch` run validation code
test_epoch = 5

for epoch in range(epochs):
    if (epoch % 10 == 0):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate / 10.0

    avg_train_loss = 0.0
    # train
    model.train()
    for i, (images, captions, lengths) in enumerate(train_loader, 1):
        images = images.cuda()
        #images = images.to(device)
        captions = captions.cuda()
        #captions = captions.to(device)

        optimizer.zero_grad()
        targets = pack_padded_sequence(captions, lengths, batch_first=True).data
        outputs = model(images, captions, lengths)
        loss = criterion(outputs, targets)
        loss.backward()
        # fix adam
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state[p]
                if 'step' in state.keys():
                    if (state['step'] >= 1024):
                        state['step'] = 1000
        optimizer.step()
        avg_train_loss += loss.item()
    print('Epoch [{}/{}]\tTrain Loss: {:.3f}'.format(epoch+1, epochs, avg_train_loss/i), end=" ")

    # test
    if (epoch + 1) % test_epoch == 0:
        model.eval()
        avg_test_loss = 0.0
        for i, (images, captions, lengths) in enumerate(test_loader, 1):
            images = images.cuda()
            #images = images.to(device)
            captions = captions.cuda()
            #captions = captions.to(device)
            max_sequence_length = captions.size(1)

            targets = pack_padded_sequence(captions, lengths, batch_first=True).data
            with torch.no_grad():
                predicted_ids, outputs = model(images)

            # truncate outputs sequence length
            outputs = outputs[:, :max_sequence_length]

            loss = criterion(outputs.permute(0, 2, 1), captions)
            avg_test_loss += loss.item()

        print('\tTest Loss: {:.3f}'.format(avg_test_loss/i))
        # see how a sample sentence looks
        sample = dataset.convert_to_sentence(predicted_ids[0], vectorizer.get_feature_names())
        target = dataset.convert_to_sentence(captions[0], vectorizer.get_feature_names())
        print("\tSample:", sample)
        print("\tTarget:", target, end=' ')
    print()

    save_dict = {'state_dict' : model.state_dict(),
                 'optim_dict' : optimizer.state_dict(),
                 'epoch'      : epoch + 1}
    torch.save(save_dict, os.path.join(save_path, 'model_entire_network.pth'))
