import torch
import torch.nn as nn
import torch.otim as optim
import torchvision
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator,Generator,initalize_weights

#hyperparam
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG=1
Z_DIM = 100
NUM_EPOCHS = 5 
FEATURES_DISC=64
FEATURES_GEN =64

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE)
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)],[0.5 for _ in range(CHANNELS_IMG)]
        ),

    ]
)
dataset = datasets.MNIST(root = "dataset/", train=True,transform=transforms,download=True)
loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)
gen = Generator(NOISE_DIM,CHANNELS_IMG,FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG,FEATURES_DISC).to(device)
initalize_weights(gen)
initalize_weights(disc)
opt_gen  = optim.Adam(gen.parameters(), lr= LEARNING_RATE, betas=(0.5,0.999))
opt_disc = optim.Adam(disc.parameters(),lr = LEARNING_RATE,betas = (0.5,0.999))
criterion = nn.BCELoss()
fixed_noise = torch.randn(32,Z_DIM,1,1).to(device)
writer_real=SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0
gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx,(real,_) in enumerate(loader):
        real = real.to(device)
