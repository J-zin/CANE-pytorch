import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import config

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class CANE(nn.Module):
    def __init__(self, vocab_size, num_nodes, rho):
        super(CANE, self).__init__()
        rho = rho.split(",")
        self.rho1 = float(rho[0])
        self.rho2 = float(rho[1])
        self.rho3 = float(rho[2])

        # initialize_embedding
        self.text_embed = nn.Embedding(vocab_size, config.embed_size // 2)
        self.text_embed.weight.data.normal_(0.0, 0.03)      # this weight inition is important!!!
        self.node_embed = nn.Embedding(num_nodes, config.embed_size // 2)
        self.node_embed.weight.data.normal_(0.0, 0.03)      # this weight inition is important!!!

        # convolution
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=100, 
                                    kernel_size=[2,config.embed_size // 2])
        # attentive matrix A: see formular 9
        self.rand_matrix = Variable(torch.normal(0, 0.3, size=(100, 100)).type(FloatTensor), requires_grad=True)
    
    def forward(self, Node_a, Node_b, Node_neg, Text_a, Text_b, Text_neg):
        # lookup embedding
        T_A, T_B, T_NEG, N_A, N_B, N_NEG = self.lookup(Node_a, Node_b, Node_neg, Text_a, Text_b, Text_neg)
        # context awaring
        convA, convB, convNeg  = self.context_conv(T_A, T_B, T_NEG)
        # compute loss
        loss = self.compute_loss(convA, convB, convNeg, N_A, N_B, N_NEG)
        
        return loss
    
    def get_emb(self, Node_a, Node_b, Node_neg, Text_a, Text_b, Text_neg):
        # lookup embedding
        T_A, T_B, T_NEG, N_A, N_B, _ = self.lookup(Node_a, Node_b, Node_neg, Text_a, Text_b, Text_neg)
        
        # context awaring
        convA, convB, _  = self.context_conv(T_A, T_B, T_NEG)

        return convA, convB, N_A, N_B

    def lookup(self, Node_a, Node_b, Node_neg, Text_a, Text_b, Text_neg):
        TA = self.text_embed(Text_a)
        T_A = torch.unsqueeze(TA, dim=1)

        TB = self.text_embed(Text_b)
        T_B = torch.unsqueeze(TB, dim=1)

        TNEG = self.text_embed(Text_neg)
        T_NEG = torch.unsqueeze(TNEG, dim=1)

        N_A = self.node_embed(Node_a)
        N_B = self.node_embed(Node_b)
        N_NEG = self.node_embed(Node_neg)

        return T_A, T_B, T_NEG, N_A, N_B, N_NEG
    
    def context_conv(self, T_A, T_B, T_NEG):
        convA = self.conv2d(T_A)
        convB = self.conv2d(T_B)
        convNEG  = self.conv2d(T_NEG)

        hA = torch.tanh(torch.squeeze(convA)).permute(0, 2, 1)
        hB = torch.tanh(torch.squeeze(convB))
        hNEG = torch.tanh(torch.squeeze(convNEG))
        
        tmphA = hA.reshape(config.batch_size * (config.MAX_LEN - 1), 100)
        ha_mul_rand = torch.mm(tmphA, self.rand_matrix).reshape(config.batch_size, config.MAX_LEN - 1, 100)
        
        r1 = torch.bmm(ha_mul_rand, hB)  # [64, 299, 100] * [64, 100, 299] = [64, 299, 299]
        r3 = torch.bmm(ha_mul_rand, hNEG)

        att1 = torch.tanh(r1)
        att3 = torch.tanh(r3)
        
        pooled_A = torch.mean(att1, dim=2)
        pooled_B = torch.mean(att1, dim=1)
        pooled_NEG = torch.mean(att3, dim=1)

        w_A = F.softmax(pooled_A, dim=-1)
        w_B = F.softmax(pooled_B, dim=-1)
        w_NEG = F.softmax(pooled_NEG, dim=-1)

        rep_A = torch.unsqueeze(w_A, -1)
        rep_B = torch.unsqueeze(w_B, -1)
        rep_NEG = torch.unsqueeze(w_NEG, -1)

        attA = torch.bmm(hA.permute(0, 2, 1), rep_A).squeeze()
        attB = torch.bmm(hB, rep_B).squeeze()
        attNEG = torch.bmm(hNEG, rep_NEG).squeeze()
        
        return attA, attB, attNEG

    def compute_loss(self, convA, convB, convNeg, N_A, N_B, N_NEG):
        # logp(v^t|u^t), 
        p1 = torch.sum(convA * convB, dim=1)
        p1 = torch.log(torch.sigmoid(p1) + 0.001)
        p2 = torch.sum(convA * convNeg, dim=1)
        p2 = torch.log(torch.sigmoid(-p2) + 0.001)

        # logp(v^s|u^s)
        p3 = torch.sum(N_A * N_B, dim=1)
        p3 = torch.log(torch.sigmoid(p3) + 0.001)
        p4 = torch.sum(N_A * N_NEG, dim=1)
        p4 = torch.log(torch.sigmoid(-p4) + 0.001)

        # logp(v^t|u^s)
        p5 = torch.sum(convB * N_A, dim=1)
        p5 = torch.log(torch.sigmoid(p5) + 0.001)
        p6 = torch.sum(convNeg * N_A, dim=1)
        p6 = torch.log(torch.sigmoid(-p6) + 0.001)

        # logp(v^s|u^t)
        p7 = torch.sum(N_B * convA, dim=1)
        p7 = torch.log(torch.sigmoid(p7) + 0.001)
        p8 = torch.sum(N_B * convNeg, dim=1)
        p8 = torch.log(torch.sigmoid(-p8) + 0.001)

        rho1 = self.rho1
        rho2 = self.rho2
        rho3 = self.rho3
        temp_loss = rho1 * (p1 + p2) + rho2 * (p3 + p4) + rho3 * (p5 + p6) + rho3 * (p7 + p8)
        loss = -torch.sum(temp_loss)

        return loss