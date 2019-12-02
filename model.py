# Faster version of show_and_tell
# To use this version of Module, the dataloader should returns:
#     images: (B, 3, H, W). This should be in format for inception_v3
#     captions: (B, ?). The length of the caption depends on the max length of the current batch.
#         It includes <start>, <end>, <pad> character. Short captions should be padded.
#     lengths: (B, ). The actual length of each caption
# During training, the logic looks as follow:
#     for images, captions, lengths in train_loader:
#         targets = pack_padded_sequence(captions, lengths, batch_first=True).data
#         outputs = model(images, captions, lengths)
#         loss = criterion(outputs, targets)
#         ...
# During inference, the logic is similar as before.

import torch
import torch.nn as nn
from torchvision.models import inception_v3
from torch.nn.utils.rnn import pack_padded_sequence

class Show_and_tell(nn.Module):
    def __init__(self, vocab_size=12000, embed_size=512, hidden_units=512,
                 num_lstm_layers=1, max_seq_length=100):
        super(Show_and_tell, self).__init__()

        self.cnn_embedding = inception_v3(pretrained=True, aux_logits=True)
        for param in self.cnn_embedding.parameters():
            param.requires_grad = False
        out_features = self.cnn_embedding.fc.in_features
        self.cnn_embedding.fc = nn.Linear(out_features, embed_size)

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(embed_size, hidden_units, num_layers=num_lstm_layers,
                            dropout=0.3, batch_first=True)

        self.decode = nn.Linear(hidden_units, vocab_size)

        self.max_seq_length = max_seq_length
        # self.num_lstm_layers = num_lstm_layers

    def forward(self, imgs, captions=None, lengths=None, states=None):
        if captions is None:
            return self.greedy_inference(imgs, states=states)
        else:
            return self.forward_train(imgs, captions, lengths)

    def forward_train(self, imgs, captions, lengths):
        """
        Args:
            imgs: torch.FloatTensor([B, 3, 299, 299]), which is the input to inception_v3
            catptions: torch.LongTensor([B, max_seq_len]), which is the padded captions
            lengths: torch.LongTensor([B,]): the actual length of the captions
        """
        # img_feat: (B, img_embedding_size)
        img_feats = self.cnn_embedding(imgs)

        B, seq_len = captions.shape
        # Don't need to predict the last word
        # word_feat: (B, seq_len, word_embedding_size)
        word_feats = self.embedding(captions)

        inputs = torch.cat([img_feats.unsqueeze(1), word_feats], dim=1)
        packed_inputs = pack_padded_sequence(inputs, lengths, batch_first=True)
        # Default is with zero initial hidden and cell state
        packed_outputs = self.lstm(packed_inputs)
        # Use .data to extract the underlying tensor
        #print(len(packed_outputs))
        #print(packed_outputs[0].data.shape)
        outputs = self.decode(packed_outputs[0].data)
        return outputs

    def greedy_inference(self, imgs, states=None):
        # img_feat: (B, img_embedding_size)
        predicted_ids = []
        output_probs = []
        img_feats = self.cnn_embedding(imgs)
        inputs = img_feats.unsqueeze(1)
        for i in range(self.max_seq_length):
            # hiddens: (B, 1, hidden_units)
            hiddens, states = self.lstm(inputs, states)
            # outputs: (B, vocab_size)
            outputs = self.decode(hiddens.squeeze(1))
            # predicted: (B, )
            _, predicted = outputs.max(1)

            predicted_ids.append(predicted)
            output_probs.append(outputs)
            # inputs: (B, 1, embed_size)
            inputs = self.embedding(predicted).unsqueeze(1)
        # predicted_ids: (B, max_seq_len)
        predicted_ids = torch.stack(predicted_ids, 1)
        output_probs = torch.stack(output_probs, 1)
        return predicted_ids, output_probs

