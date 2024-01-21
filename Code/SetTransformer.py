import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.heads_dim = embedding_size // num_heads

        self.Query = nn.Linear(embedding_size, embedding_size, bias=False, dtype=torch.float64)
        self.Key = nn.Linear(embedding_size, embedding_size, bias=False, dtype=torch.float64)
        self.Value = nn.Linear(embedding_size, embedding_size, bias=False, dtype=torch.float64)
        self.fc_out = nn.Linear(embedding_size, embedding_size, bias=False, dtype=torch.float64)

    def forward(self, query, key, value):
        # The input size would be (batch_size,sentence_size,embedding_size)
        batch_size = query.shape[0]
        q, k, v = query.shape[1], key.shape[1], value.shape[1]  # sentence size = how many words

        V = self.Value(value)  # (batch_size,n,embedding_size)
        Q = self.Query(query)  # (batch_size,query_size,embedding_size)
        K = self.Key(key)  # (batch_size,key_size,embedding_size)

        # split to heads for parallel comuptation

        V = V.reshape(batch_size, v, self.num_heads, self.heads_dim)
        V = torch.einsum("bnhd->bhnd", V)  # (batch_size,num_heads,n,embedding_size)
        Q = Q.reshape(batch_size, q, self.num_heads, self.heads_dim)
        Q = torch.einsum("bnhd->bhnd", Q)  # (batch_size,num_heads,n,embedding_size)
        K = K.reshape(batch_size, k, self.num_heads, self.heads_dim)
        K = torch.einsum("bnhd->bhnd", K)  # (batch_size,num_heads,n,embedding_size)

        temp = torch.einsum("bhqd,bhkd->bhqk",
                            [Q, K])  # Q(batch_size,num_heads,n,embedding_size)*K(batch_size,num_heads,embedding_size,n)
        # temp size = (batch_size,num_heads,q,k)

        Attention = torch.softmax(temp / torch.sqrt(torch.tensor(self.embedding_size, dtype=torch.float64)),
                                  dim=3)  # sum of each row is 1
        # Attention size = (batch_size,num_heads,q,k)

        temp = torch.einsum("bhqk,bhkv->bhqv", [Attention, V])
        # temp size = (batch_size,num_heads,n,heads_dim)

        temp = temp.reshape(batch_size, q, self.num_heads * self.heads_dim)
        out = self.fc_out(temp)
        return out


class MAB(nn.Module):
    def __init__(self, embedding_size, num_heads, feedforward_expansion):
        super(MAB, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.MHA = MultiHeadAttention(embedding_size, num_heads)
        self.lnorm1 = nn.LayerNorm(embedding_size, dtype=torch.float64)
        self.lnorm2 = nn.LayerNorm(embedding_size, dtype=torch.float64)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, feedforward_expansion, dtype=torch.float64),
            nn.LeakyReLU(0.1),
            nn.Linear(feedforward_expansion, embedding_size, dtype=torch.float64),
        )

    def forward(self, X, Y):
        mha = self.MHA(X, Y, Y)
        x = self.lnorm1(mha + X)
        FeedForward = self.feed_forward(x)
        out = self.lnorm2(FeedForward + x)
        return out


class SAB(nn.Module):
    def __init__(self, embedding_size, num_heads, feedforward_expansion):
        super(SAB, self).__init__()
        self.mab = MAB(embedding_size, num_heads, feedforward_expansion)

    def forward(self, X):
        out = self.mab(X, X)
        return out


class PMA(nn.Module):
    def __init__(self, embedding_size, num_heads, feedforward_expansion, k):
        super(PMA, self).__init__()
        self.mab = MAB(embedding_size, num_heads, feedforward_expansion)
        self.S = torch.nn.init.kaiming_uniform_(
            torch.nn.Parameter(torch.zeros(k, embedding_size, dtype=torch.float64), requires_grad=True))
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, feedforward_expansion, dtype=torch.float64),
            nn.LeakyReLU(0.1),
            nn.Linear(feedforward_expansion, embedding_size, dtype=torch.float64),
        )

    def forward(self, Z):
        batch_size = Z.shape[0]
        out = self.mab(self.S.repeat(batch_size, 1, 1), self.feed_forward(Z))
        return out


class Decoder(nn.Module):
    def __init__(self, embedding_size, num_heads, feedforward_expansion, k):
        super(Decoder, self).__init__()
        self.sab = SAB(embedding_size, num_heads, feedforward_expansion)
        self.pma = PMA(embedding_size, num_heads, feedforward_expansion, k)

    def forward(self, x):
        out = self.sab(self.pma(x))
        return out


class SetTransformer(nn.Module):
    def __init__(self, embedding_size, num_heads, feedforward_expansion, k, blocklayers, output_size):
        super(SetTransformer, self).__init__()
        self.encoder_layers = nn.ModuleList(
            [
                SAB(embedding_size, num_heads, feedforward_expansion)
                for _ in range(blocklayers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                Decoder(embedding_size, num_heads, feedforward_expansion, k)
                for _ in range(blocklayers)
            ]
        )
        self.sab = SAB(embedding_size, num_heads, feedforward_expansion)
        self.pma = PMA(embedding_size, num_heads, feedforward_expansion, k)
        self.fc_out = nn.Sequential(
            nn.Linear(embedding_size, feedforward_expansion, dtype=torch.float64),
            nn.LeakyReLU(0.1),
            nn.Linear(feedforward_expansion, output_size, dtype=torch.float64),
        )
        self.fc_out = nn.Linear(embedding_size, output_size, dtype=torch.float64)

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        for layer in self.decoder_layers:
            x = layer(x)
        out = x

        return out


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, filter_size, in_chans, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(
                in_chans,
                embed_dim,
                kernel_size=filter_size,
                stride=patch_size, dtype=torch.float64
            )

        )

    def forward(self, x):
        x = self.proj(x)  # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)

        return x


class Network(nn.Module):
    def __init__(self, patch_size, filter_size, in_chans, embedding_size, num_heads, feedforward_expansion, k,
                 blocklayers, output_size):
        super(Network, self).__init__()
        self.PE = PatchEmbedding(patch_size, filter_size, in_chans, embedding_size)
        self.set_transformer = SetTransformer(embedding_size, num_heads, feedforward_expansion, k, blocklayers,
                                              output_size)
        self.FC_output = nn.Sequential(
            nn.Linear(embedding_size, 32, dtype=torch.float64),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.1),  # Dropout layer added
            nn.Linear(32, output_size, dtype=torch.float64),
            nn.ReLU(),
        )
        self.lastlayer = nn.Linear(output_size, output_size, dtype=torch.float64)

    def forward(self, x, Q):
        x = self.PE(x)
        out = self.set_transformer(x)
        out = out.flatten(1)
        out = self.FC_output(out)
        # Create a mask for elements equal to 0
        mask = (out == 0)

        # Replace elements equal to 0 with 1 using the mask
        x = torch.where(mask, torch.tensor(1.0), out)
        x = torch.bmm(torch.diag_embed(x), Q[:, :, :, -2])

        return torch.diagonal(x, dim1=-2, dim2=-1)


