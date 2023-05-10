import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from torch.optim import Adam
from multiprocessing import freeze_support
import matplotlib.pyplot as plt
import random

if __name__ == '__main__':
    freeze_support()
    # Model Hyperparameters



    batch_size = 32
    isize = 128
    img_size = (isize, isize) # (width, height)

    input_dim = 1
    hidden_dim = 128
    n_embeddings= 256
    output_dim = 1
    workers = 1

    lr = 2e-4

    epochs = 20

    print_step = 50

    dataroot = f"C:/Users/Fabian/Documents/AI_Development/DataSets/MegaScans/Images{isize}"
    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(img_size),
                                transforms.Grayscale(),
                                transforms.ToTensor(),
                                #transforms.Normalize((0.5), (0.5)),
                            ]))
    
    testEnd = len(dataset)
    trainEnd = testEnd - 5000
    randomList = list(range(testEnd))
    random.shuffle(randomList)
    trainset = torch.utils.data.Subset(dataset, randomList[0:trainEnd])
    testset = torch.utils.data.Subset(dataset, randomList[trainEnd:testEnd])

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'    

    '''
    mnist_transform = transforms.Compose([
            transforms.ToTensor(),
    ])
    kwargs = {'num_workers': 1, 'pin_memory': True} 
    dataset_path = '~/datasets'
    train_dataset = CIFAR10(dataset_path, transform=mnist_transform, train=True, download=True)
    test_dataset  = CIFAR10(dataset_path, transform=mnist_transform, train=False, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False,  **kwargs)
    '''



    class Encoder(nn.Module):
        
        def __init__(self, input_dim, hidden_dim, output_dim, kernel_size=(4, 4, 3, 1), stride=2):
            super(Encoder, self).__init__()
            
            kernel_1, kernel_2, kernel_3, kernel_4 = kernel_size
            
            self.strided_conv_1 = nn.Conv2d(input_dim, hidden_dim, kernel_1, stride, padding=1)
            self.strided_conv_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_2, stride, padding=1)
            
            self.residual_conv_1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_3, padding=1)
            self.residual_conv_2 = nn.Conv2d(hidden_dim, output_dim, kernel_4, padding=0)
            
        def forward(self, x):
            
            x = self.strided_conv_1(x)
            x = self.strided_conv_2(x)
            
            x = F.relu(x)
            y = self.residual_conv_1(x)
            y = y+x
            
            x = F.relu(y)
            y = self.residual_conv_2(x)
            y = y+x
            
            return y
        

    class VQEmbeddingEMA(nn.Module):
        def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
            super(VQEmbeddingEMA, self).__init__()
            self.commitment_cost = commitment_cost
            self.decay = decay
            self.epsilon = epsilon
            
            init_bound = 1 / n_embeddings
            embedding = torch.Tensor(n_embeddings, embedding_dim)
            embedding.uniform_(-init_bound, init_bound)
            self.register_buffer("embedding", embedding)
            self.register_buffer("ema_count", torch.zeros(n_embeddings))
            self.register_buffer("ema_weight", self.embedding.clone())

        def encode(self, x):
            M, D = self.embedding.size()
            x_flat = x.detach().reshape(-1, D)

            distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                        torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                    x_flat, self.embedding.t(),
                                    alpha=-2.0, beta=1.0)

            indices = torch.argmin(distances.float(), dim=-1)
            quantized = F.embedding(indices, self.embedding)
            quantized = quantized.view_as(x)
            return quantized, indices.view(x.size(0), x.size(1))
        
        def retrieve_random_codebook(self, random_indices):
            quantized = F.embedding(random_indices, self.embedding)
            quantized = quantized.transpose(1, 3)
            
            return quantized

        def forward(self, x):
            M, D = self.embedding.size()
            x_flat = x.detach().reshape(-1, D)
            
            distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                    torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                    x_flat, self.embedding.t(),
                                    alpha=-2.0, beta=1.0)

            indices = torch.argmin(distances.float(), dim=-1)
            encodings = F.one_hot(indices, M).float()
            quantized = F.embedding(indices, self.embedding)
            quantized = quantized.view_as(x)
            
            if self.training:
                self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)
                n = torch.sum(self.ema_count)
                self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

                dw = torch.matmul(encodings.t(), x_flat)
                self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
                self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

            codebook_loss = F.mse_loss(x.detach(), quantized)
            e_latent_loss = F.mse_loss(x, quantized.detach())
            commitment_loss = self.commitment_cost * e_latent_loss

            quantized = x + (quantized - x).detach()

            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

            return quantized, commitment_loss, codebook_loss, perplexity
        

    class Decoder(nn.Module):
        
        def __init__(self, input_dim, hidden_dim, output_dim, kernel_sizes=(1, 3, 2, 2), stride=2):
            super(Decoder, self).__init__()
            
            kernel_1, kernel_2, kernel_3, kernel_4 = kernel_sizes
            
            self.residual_conv_1 = nn.Conv2d(input_dim, hidden_dim, kernel_1, padding=0)
            self.residual_conv_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_2, padding=1)
            
            self.strided_t_conv_1 = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_3, stride, padding=0)
            self.strided_t_conv_2 = nn.ConvTranspose2d(hidden_dim, output_dim, kernel_4, stride, padding=0)
            
        def forward(self, x):
            
            y = self.residual_conv_1(x)
            y = y+x
            x = F.relu(y)
            
            y = self.residual_conv_2(x)
            y = y+x
            y = F.relu(y)
            
            y = self.strided_t_conv_1(y)
            y = self.strided_t_conv_2(y)
            
            return y
        

    class Model(nn.Module):
        def __init__(self, Encoder, Codebook, Decoder):
            super(Model, self).__init__()
            self.encoder = Encoder
            self.codebook = Codebook
            self.decoder = Decoder
                    
        def forward(self, x):
            z = self.encoder(x)
            z_quantized, commitment_loss, codebook_loss, perplexity = self.codebook(z)
            x_hat = self.decoder(z_quantized)
            
            return x_hat, commitment_loss, codebook_loss, perplexity
        

    encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim)
    codebook = VQEmbeddingEMA(n_embeddings=n_embeddings, embedding_dim=hidden_dim)
    decoder = Decoder(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    model = Model(Encoder=encoder, Codebook=codebook, Decoder=decoder).to(device)


    mse_loss = nn.MSELoss()

    optimizer = Adam(model.parameters(), lr=lr)






    def mixImage(imgs):
        mixed = imgs[0].clone()
        imgAmount = len(imgs)
        for i in range(len(imgs[0])):
            for j in range(len(imgs[0])):
                for d in range(1, imgAmount): 
                    mixed[i][j] += imgs[d][i][j]
                
                mixed[i][j] /= imgAmount
        
        return mixed

    def draw_random_sample_image(encoder, decoder, amount, columns):
        images = []
        with torch.no_grad():
            for _, (x, _) in enumerate(tqdm(test_loader)):
                images.extend(x)

        random.shuffle(images)
        combAmount = 3
        imgs = images[:amount*combAmount]
        random_indices = torch.floor(torch.rand((256, 16, 16)) * n_embeddings).long().to(device)
        codes = codebook.retrieve_random_codebook(random_indices)
        codes = []
        for i in range(amount):
            newImg = imgs[i].clone()
            for d in range(1, combAmount):
                newImg += imgs[i*combAmount+d]

            codes.append(newImg / combAmount)
            #codes.append(mixImage(imgs[i][0], imgs[i+1][0]))

        codesTensor = torch.stack(codes)
        #codesTensor.reshape((amount, 1, len(codes[0]), len(codes[0])))
        encoded = encoder(codesTensor.to(device))
        x_hat = decoder(encoded)
        
        #plt.figure(figsize=(16,16))
        #plt.axis("off")
        #plt.title("Visualization of Random Codes")
        #plt.imshow(np.transpose(make_grid(codes, padding=2, nrow=columns, normalize=True), (1, 2, 0)))
        #plt.show()

        plt.figure(figsize=(16,16))
        plt.axis("off")
        plt.title("Visualization of Random Codes")
        plt.imshow(np.transpose(make_grid(x_hat.detach().cpu(), padding=2, nrow=columns, normalize=True), (1, 2, 0)))
        plt.show()










    print("Start training VQ-VAE...")
    model.train()

    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)

            optimizer.zero_grad()

            x_hat, commitment_loss, codebook_loss, perplexity = model(x)
            recon_loss = mse_loss(x_hat, x)
            
            loss =  recon_loss + commitment_loss + codebook_loss
                    
            loss.backward()
            optimizer.step()
            
            if batch_idx % print_step ==0: 
                print("epoch:", epoch + 1, "  step:", batch_idx + 1, "  recon_loss:", recon_loss.item(), "  perplexity: ", perplexity.item(), 
                "\n\t\tcommit_loss: ", commitment_loss.item(), "  codebook loss: ", codebook_loss.item(), "  total_loss: ", loss.item())
        
        if epoch % 2 == 0:
            draw_random_sample_image(encoder, decoder, 64, 8)

    print("Finish!!")


    def draw_sample_image(x, postfix):
    
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Visualization of {}".format(postfix))
        plt.imshow(np.transpose(make_grid(x.detach().cpu(), padding=2, normalize=True), (1, 2, 0)))

    model.eval()


    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(tqdm(test_loader)):
            x = x.to(device)
            x_hat, commitment_loss, codebook_loss, perplexity = model(x)
    
            print("perplexity: ", perplexity.item(),"commit_loss: ", commitment_loss.item(), "  codebook loss: ", codebook_loss.item())
            break


    #draw_sample_image(x[:batch_size], "Ground-truth images")
    #plt.show()

    #draw_sample_image(x_hat[:batch_size], "Reconstructed images")
    #plt.show()

    draw_random_sample_image(encoder, decoder, 256, 16)
