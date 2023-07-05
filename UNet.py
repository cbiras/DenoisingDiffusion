import torch.nn as nn
import torch
import math

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self,total_time_steps=1000,time_emb_dims=128,time_emb_dims_exp=512):
        super().__init__()

        half_dim = time_emb_dims//2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim,dtype=torch.float32)*-emb)

        ts = torch.arange(total_time_steps,dtype=torch.float32)

        emb = torch.unsqueeze(ts,dim=-1) * torch.unsqueeze(emb,dim=0)

        emb = torch.cat((emb.sin(),emb.cos()),dim=-1)

        self.time_blocks = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(in_features=time_emb_dims,out_features=time_emb_dims_exp),
            nn.SiLU(),
            nn.Linear(in_features=time_emb_dims_exp,out_features=time_emb_dims_exp)
        )
    def forward(self,time):
        return self.time_blocks(time)

class AttentionBlock(nn.Module):
    def __int__(self,channels=64):
        super().__init__()

        self.channels = channels

        self.group_norm = nn.GroupNorm(num_groups=8,num_channels=channels)
        self.mha = nn.MultiheadAttention(embed_dim=self.channels,num_heads=4,batch_first=True)

    def forward(self,x):
        B,C,H,W = x.shape
        h = self.group_norm(x)

        h = h.reshape(B,self.channels,H*W).swapaxes(1,2)
        h,_ = self.mha(h,h,h)
        h = h.swapaxes(2,1).view(B,self.channels,H,W)

        return x + h


class ResNetBlock(nn.Module):
    def __init__(self,*,input_channels,output_channels,dropout_rate=0.1,time_emb_dims=512,apply_attention=False):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.activation = nn.GELU()

        self.norm1 = nn.GroupNorm(num_groups=8,num_channels=input_channels)
        self.conv1 = nn.Conv2d(input_channels=input_channels,output_channels=output_channels,kernel_size=3,stride=1,padding='same')

        self.dense = nn.Linear(in_features=time_emb_dims,out_features=output_channels)

        self.norm2 = nn.GroupNorm(num_groups=8,num_channels=output_channels)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = nn.Conv2d(input_channels=output_channels,output_channels=output_channels,kernel_size=3,stride=1,padding='same')

        if input_channels != output_channels:
            self.match_channels = nn.Conv2d(input_channels=input_channels,output_channels=output_channels,kernel_size=1,stride=1,padding='same')
        else:
            self.match_channels = nn.Identity()

        if apply_attention:
            self.attention = AttentionBlock(output_channels)
        else:
            self.attention = nn.Identity()

    def forward(self,x,t):

        x1 = self.norm1(x)
        x1 = self.activation(self.conv1(x1))

        x1 += self.dense(self.activation(t))[:,:,None,None]

        x1 = self.norm2(x1)
        x1 = self.activation(self.conv2(x1))
        x1 = self.dropout(x1)

        x1 = x1 + self.match_channels(x1)
        x1 = self.attention(x1)

        return x1

class Downsample(nn.Module):
    def __init__(self,channels):
        super().__init__()

        self.downsample = nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,stride=2)

    def forward(self,x):
        return self.downsample(x)

class Upsample(nn.Module):
    def __init__(self,in_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(mode='nearest',scale_factor=2),
            nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1)
        )
    def forward(self,x):
        return self.upsample(x)

class UNet(nn.Module):
    def __int__(self,
                input_channels=3,
                output_channels=3,
                base_channels=128,
                apply_attention=[False,False,True,False],
                num_res_blocks=2,
                base_ch_multipliers=[1,2,4,8],
                dropout_rate=0.1,
                time_multiply=4
                ):
        super().__int__()

        time_emb_dims_exp = base_channels * time_multiply
        self.positional_encoding = SinusoidalPositionalEmbedding(time_emb_dims_exp=time_emb_dims_exp)

        self.stem = nn.Conv2d(in_channels=input_channels,out_channels=base_channels,kernel_size=3,stride=1,padding='same')

        num_resolutions = len(base_ch_multipliers)
        self.encoder = nn.ModuleList()
        curr_channels = [base_channels]
        in_channels = base_channels

        #Encoder
        for level in range(num_resolutions):
            out_channels = base_channels * base_ch_multipliers[level]

            for _ in num_res_blocks:
                block = ResNetBlock(
                    in_channels=in_channels,
                    output_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=apply_attention[level]
                )
                self.encoder.append(block)

                in_channels = out_channels
                curr_channels.append(in_channels)
            if level != (num_resolutions-1):
                self.encoder.append(Downsample(in_channels))
                curr_channels.append(in_channels)

        #Between

        self.between = nn.ModuleList(
            (
            ResNetBlock(
                in_channels=in_channels,
                output_channels=in_channels,
                dropout_rate=dropout_rate,
                time_emb_dims=time_emb_dims_exp,
                apply_attention=True
            ),
            ResNetBlock(
                in_channels=in_channels,
                output_channels=in_channels,
                dropout_rate=dropout_rate,
                time_emb_dims=time_emb_dims_exp,
                apply_attention=False
            )
            )
        )

        self.decoder = nn.ModuleList()

        for level in reversed(range(num_resolutions)):
            out_channels = base_channels * base_ch_multipliers[level]

            for _ in (num_res_blocks + 1):
                encoder_in_channels = curr_channels.pop()
                block = ResNetBlock(
                    in_channels=in_channels + encoder_in_channels,
                    output_channels=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=apply_attention[level]
                )

                in_channels = out_channels
                self.decoder.append()

                if level != 0:
                    self.decoder.append(Upsample(in_channels))

        self.final = nn.Sequential(
            nn.GroupNorm(num_groups=8,num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels,out_channels=output_channels,kernel_size=3,stride=1,padding='same')
        )

    def forward(self,x,t):

        positional_encoding = self.positional_encoding(t)

        x = self.stem(x)
        outs = [x]

        for layer in self.encoder:
            x = layer(x,positional_encoding)
            outs.append(x)

        for layer in self.between:
            x = layer(x,positional_encoding)

        for layer in self.decoder:
            if isinstance(layer,ResNetBlock):
                out = outs.pop()
                x = torch.cat([x,out],dim=1)
            x = layer(x)

        return self.final(x)







