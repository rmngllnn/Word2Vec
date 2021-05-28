import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Sampler
torch.manual_seed(1)


#Hyperparameters of model
CONTEXT_SIZE = 2 #Taille de contexte
EMBEDDING_DIM = 10
BATCH_SIZE = 40
NB_EPOCH = 10

class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        #@@ inputs est un tenseur
        #@@ dim 0 = taille du minibatch
        #@@ dim 1 = les différentes positions de contexte
        #@@ en appliquant self.embeddings, on ajoute une
        #@@ dim 2 = les différentes composantes de l'embedding d'un mot de contexte
        embeds = self.embeddings(inputs)
        batch_size = embeds.size()[0]
        #@@ on souhaite sommer les différents embeddings des mots de contexte
        sum_embeds = torch.sum(embeds, dim=1)

        out = self.linear(sum_embeds)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = CBOW(len(vocab), EMBEDDING_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.001)


#Loop over dataset
for epoch in range(NB_EPOCHS):
    epoch_loss = 0
    batch_list = torch.randperm(len(examples))
    batch = [examples[i] for i in mini_batch][:BATCH_SIZE]
    for context, target in batch: #par défaut de taille 1, à modifier dans inputs

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        input_tensor = torch.tensor(context, dtype=torch.long)
        gold_label = torch.tensor([target], dtype=torch.long)
        
        
        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next words
        log_probs = model(input_tensor)

        # Step 4. Compute your loss function
        loss = loss_function(log_probs, gold_label)

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        epoch_loss += loss.item()
        
    train_losses.append(epoch_loss)
