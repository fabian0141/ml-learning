from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torchvision.transforms as transforms
import torchvision.utils as vutils
from multiprocessing import freeze_support

if __name__ == '__main__':
    #freeze_support()

    model = Unet(
        dim = 64,
        dim_mults = (1,2,4,8),
        channels=1
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        timesteps = 1000,   # number of steps
        loss_type = 'l1'    # L1 or L2
    )
 
    trainer = Trainer(
        diffusion,
        "C:/Users/Fabian/Documents/AI_Development/DataSets/MegaScans/Images128/img_align_plants",
        train_batch_size = 2,
        train_lr = 2e-5,
        train_num_steps = 200000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                        # turn on mixed precision
        save_and_sample_every=10000

    )

    #trainer.load('100')
    trainer.train()
    #trainer.save

    sampled_images = diffusion.sample(batch_size = 64)

    grid = vutils.make_grid(sampled_images, padding=0, normalize=True)
    toImage = transforms.ToPILImage()
    toImage(grid).save("C:\\Users\\Fabian\\Documents\\AI_Development\\\GeneratedImages\\generated_diffusion_large.png")

