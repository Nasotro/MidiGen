import os
import mido
import numpy as np
import matplotlib.pyplot as plt


def get_midis():
    ms = []

    directory = "data"
    for foldername in os.listdir(directory):
        directory = os.path.join("data", foldername)
        for filename in os.listdir(directory):
            if('V2'in filename): continue
            if os.path.isfile(os.path.join(directory, filename)):
                ms.append(os.path.join(directory, filename))
                # print(filename)
    return ms

midis = get_midis()

def createMidiFileWithAllMessages(notes, starts, velocity, fileName):
    file = mido.MidiFile()
    track = mido.MidiTrack()
    file.tracks.append(track)
    for i in range(len(notes)):
        track.append(mido.Message('note_on', note=notes[i], velocity=velocity[i], time=starts[i]))        
    file.save(fileName)

# works only with files with just one track
def get_all_messages_one_file(midi_file):
    file = mido.MidiFile(midi_file)
    track = file.tracks[1]
    messages = []
    for message in track:
        if(message.type == 'note_on' or message.type == 'note_off'):
            messages.append(message)
    
    notes = [0] + [m.note for m in messages] + [1]
    starts = [0] + [m.time for m in messages] + [1]
    vels = [0] + [m.velocity == 0 for m in messages] + [1]

    return notes, starts, vels



all_messages_all_files = [get_all_messages_one_file(m) for m in midis]

def get_data(messages):
    notes, starts, vels = zip(*messages)
    notes = np.concatenate(notes)
    starts = np.concatenate(starts)
    vels = np.concatenate(vels)
    print(len(notes))
    return notes, starts, vels

# Combine everything within a single matrix

notes, starts, vels = get_data(all_messages_all_files)

X_vals = np.array([notes, starts, vels]).T
print(X_vals.shape)
print(X_vals[:10])

nunique = np.unique(X_vals, axis=0, return_counts=True)
print(len(nunique[0]))


# notes

n = np.unique(X_vals[:,0], return_counts=True)
print(n)
notes_extreme = n[0][n[1] < 100]
print('\n',notes_extreme,len(notes_extreme),'\n')
X_vals = X_vals[~np.isin(X_vals[:,0], notes_extreme)]

nunique = np.unique(X_vals, axis=0, return_counts=True)
print(len(nunique[0]))

# starts

start_max = 2000
X_vals[:,1][X_vals[:,1] >= start_max] = start_max
X_vals[:,1] = np.where(X_vals[:,1]!=1, np.round(X_vals[:,1].copy(), -2).astype(int),X_vals[:,1]) # dont remove the 1 on the time (which correspond to the end of the song)

nunique = np.unique(X_vals, axis=0, return_counts=True)
print(nunique[0])

# with open('nunique_elements.txt', 'w') as file:
#     for element in nunique[0]:
#         file.write(str(element) + '\n')



# --> dictionary length = 1705


# ## Tokenize the data


tokenToVals = nunique[0]
print(tokenToVals[:5])
ValsToToken = {tuple(v):i for i,v in enumerate(tokenToVals)}
print(ValsToToken[(1,1,1)])
vocab_size = len(tokenToVals)



X = np.array([ValsToToken[tuple(x)] for x in X_vals])
print(X[:10])
print(len(X))


a = np.random.randint(0, len(tokenToVals), 10)
print(a)
print([tuple(tokenToVals[i]) for i in a])



print(len(ValsToToken))


import torch
import torch.nn as nn
import torch.nn.functional as F

block_size = 256
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


X = torch.tensor([ValsToToken[tuple(x)] for x in X_vals], dtype=torch.long)


n = int(0.9*len(X))
X_train = X[:n]
X_val = X[n:]


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = X_train if split == 'train' else X_val
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# Signle head self-attention

n_embd = 256
dropout = 0.1

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


# multi-head self-attention

class MultiHead(nn.Module):
    """ multi-head self-attention """

    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, channels)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout()
        )
        
    def forward(self, x):
        out = self.net(x)
        return out


# block

n_heads = 8
# head_size = 64
class Block(nn.Module):
    """ a transformer Block """
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.MultiHeads = MultiHead(n_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.MultiHeads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# transformer
vocab_size = len(tokenToVals)
n_layer = 6
class Transformer(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx, targets=None):
        # print('oouais', idx.shape)
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        # pos_emb = self.position_embedding_table(torch.arange(T)) # (T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # (B, T)
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


learning_rate = 1e-2

model = Transformer()
model = model.to(device)
# m=model
print(sum(p.numel() for p in model.parameters()), 'parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


eval_iters = 100

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            # print(X.shape,Y.shape)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    # print(out)
    return out


max_iters = 5000
eval_interval = 500

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# save the model
torch.save(model.state_dict(), 'model1.pth')



# ## converting it back to music


decode = lambda x: [tokenToVals[i] for i in x]
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))


def tokensToMidi(tokens):
    vals = decode(tokens)
    print(vals)
    notes = []
    starts = []
    vels = []
    for t in vals:
        # print(type(t))
        if(np.array_equal(t, [0,0,0])): continue
        notes.append(t[0])
        starts.append(t[1])
        vels.append(t[2])
    createMidiFileWithAllMessages(notes, starts, vels, 'test.mid')


gen1 = model.generate(context, max_new_tokens=100)[0].tolist()


print(gen1)
tokensToMidi(gen1)





