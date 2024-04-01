import torch
from unet import UNet
from utils import load_checkpoint

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 1

def convert_to_onnx(model, x):
    torch.onnx.export(model, 
                      x, 
                      "segment_background.onnx", 
                      export_params=True, 
                      do_constant_folding=True, 
                      input_names=['input'], 
                      output_names=['output'], 
                      dynamic_axes={'input' : {0 : 'batch_size'},
                                'output' : {0 : 'batch_size'}})

def load_model(model_path = '.\\8_saved_model.pth'):
    model = UNet(in_channels=3, out_channels=1).to(device)
    model = load_checkpoint(model_path, model)
    model.eval()
    return model

def main():
    x = torch.randn(batch_size, 3, 256, 256, requires_grad=True)
    x = x.to(device)
    model = load_model()
    convert_to_onnx(model, x)


if __name__=="__main__":
    main()
