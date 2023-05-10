import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from pytorch_memlab import MemReporter
import torchvision.datasets as dset
import torchvision.transforms as transforms
from multiprocessing import freeze_support


if __name__ == '__main__':
    freeze_support()

    dataroot = "data/celeba"
    workers = 2
    batch_size = 32
    image_size = 128
    hidden_size = 2048
    ngf = 32
    ndf = 32
    latent_dims = 16
    kernelSize = 4

    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'    

    class Decoder(nn.Module):
        def __init__(self, latent_dims):
            super(Decoder, self).__init__()
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                #nn.ConvTranspose2d( latent_dims, ngf * 8, kernelSize, 1, 0, bias=False),
                #nn.BatchNorm2d(ngf * 8),
                #nn.ReLU(True),

                # input is Z, going into a convolution
                nn.ConvTranspose2d( latent_dims, ngf * 16, kernelSize, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 16),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d( ngf * 16, ngf * 8, kernelSize, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),

                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, kernelSize, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d( ngf * 4, ngf * 2, kernelSize, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d( ngf * 2, ngf, kernelSize, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d( ngf, 3, kernelSize, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

        def forward(self, input):
            return self.main(input)
        

    class VariationalEncoder(nn.Module):
        def __init__(self, latent_dims):
            super(VariationalEncoder, self).__init__()
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(3, ndf, kernelSize, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, kernelSize, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, kernelSize, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, kernelSize, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 16),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                #nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
                #nn.Sigmoid()

                # state size. (ndf*8) x 4 x 4
                #nn.Conv2d(ndf * 8, 1, kernelSize, 1, 0, bias=False),
                
                #nn.Sigmoid()
            )

            self.linear2 = nn.Linear(ndf * image_size * 2, latent_dims)
            self.linear3 = nn.Linear(ndf * image_size * 2, latent_dims)

        #def forward(self, input):
        #    return self.main(input)

            self.N = torch.distributions.Normal(0, 1)
            self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()
            self.kl = 0

        def forward(self, x):
            #x = torch.flatten(x, start_dim=1)
            #x = F.relu(self.linear1(x))
            x = self.main(x)
            x = torch.flatten(x, start_dim=1)
            mu =  self.linear2(x)
            sigma = torch.exp(self.linear3(x))
            z = mu + sigma*self.N.sample(mu.shape)
            self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
            return z
        

    class VariationalAutoencoder(nn.Module):
        def __init__(self, latent_dims):
            super(VariationalAutoencoder, self).__init__()
            self.encoder = VariationalEncoder(latent_dims)
            self.decoder = Decoder(latent_dims)

        def forward(self, x):
            z = self.encoder(x)
            z = z.reshape(-1, latent_dims, 1, 1)
            return self.decoder(z)
        
    def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=6):
        w = image_size
        img = np.zeros((n*w, n*w, 3))
        for i, y in enumerate(np.linspace(*r1, n)):
            for j, x in enumerate(np.linspace(*r0, n)):
                #z = torch.Tensor([[x, y]]).to(device)
                z = torch.randn(1, latent_dims, 1, 1, device=device)
                x_hat = autoencoder.decoder(z)
                #x_hat = x_hat.reshape(image_size, image_size, 3).to('cpu').detach().numpy()
                x_hat = x_hat.reshape(3, image_size, image_size).to('cpu').detach().numpy()
                img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = np.transpose(x_hat, (1,2,0))

        plt.imshow(img, extent=[*r0, *r1])

    def train(autoencoder, data, epochs=10):
        opt = torch.optim.Adam(autoencoder.parameters())
        for epoch in range(epochs):
            i = 0
            for x, y in data:
                x = x.to(device) # GPU
                opt.zero_grad()
                x_hat = autoencoder(x)
                loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
                loss.backward()
                opt.step()

                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss: %.4f' % (epoch, epochs, i, len(dataloader), loss.item()))
                    
                #reporter.report()
                i += 1
            plot_reconstructed(vae, r0=(-3, 3), r1=(-3, 3))
            plt.show()
        return autoencoder


    vae = VariationalAutoencoder(latent_dims).to(device) # GPU

    #data = torch.utils.data.DataLoader(
    #    torchvision.datasets.MNIST('./data',
    #        transform=torchvision.transforms.ToTensor(),
    #        download=True),
    #    batch_size=128,
    #    shuffle=True)

    reporter = MemReporter()
    vae = train(vae, dataloader)

    def plot_latent(autoencoder, data, num_batches=100):
        for i, (x, y) in enumerate(data):
            z = autoencoder.encoder(x.to(device))
            z = z.to('cpu').detach().numpy()
            plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
            if i > num_batches:
                plt.colorbar()
                break
    #plot_latent(vae, dataloader)
    #plt.show()


    plot_reconstructed(vae, r0=(-3, 3), r1=(-3, 3))
    plt.show()
