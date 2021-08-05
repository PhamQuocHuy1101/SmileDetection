import torch
from torchvision import transforms
from PIL import Image
from networks.FaceDetection import Model

def load_model(name, device, checkpointPath):
    model = Model(name)
    model.to(device = device)
    checkpoint = torch.load(checkpointPath, map_location = device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

model = load_model('lenet', 'cpu', './data/checkpoint/lenet.pt')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(64)
])
def predict(img):
    with torch.no_grad():
        img = Image.fromarray(img).convert('L')
        ts_image = transform(img).to(dtype = torch.float32)
        out = model(ts_image.unsqueeze(0))
        logits = torch.softmax(out, dim = 1)
        return torch.argmax(logits.squeeze()).item()
