from vit import *
import torchinfo

net = ViT(
    image_size = 224,
    patch_size = 16,
    num_classes = 10,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

net.eval()
img = torch.rand((3,224,224))
# print(torchinfo.summary(net,input_data = img))

visualize_attention(net,img,patch_size = 16,device = torch.device("cpu"))
