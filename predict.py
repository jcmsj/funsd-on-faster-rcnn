import torch
import torchvision
from argparse import ArgumentParser
from PIL import Image
import torchvision.models as models

def main():
    '''Run if main module'''
    # --pt: path to the model weights
    parser = ArgumentParser()
    parser.add_argument('--pt', type=str, required=True, help='path to the model weights file')
    # --img
    parser.add_argument('--img', type=str, required=True, help='path to the image file')
    args = parser.parse_args()

    # model: fasterrcnn_resnet50_fpn
    # weights: ResNet50_Weights.IMAGENET1K_V1
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    num_classes = 5  # Example: background + 1 class
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    # load initial weights  using  ResNet50_Weights.IMAGENET1K_V1
    weights = torch.load(args.pt)
    model.load_state_dict(weights['model'])
    model.eval()

    # image
    img = Image.open(args.img)
    # convert to rgb
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # transform
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    tensor:torch.Tensor = transform(img) # type: ignore
    
    # prediction
    with torch.no_grad():
        prediction = model([tensor])
    classes = ['background', 'other', 'question', 'answer', 'header']
    predicted_classes = [classes[i] for i in prediction[0]['labels'].tolist()]
    predicted_img = torchvision.utils.draw_bounding_boxes(tensor, prediction[0]['boxes'], predicted_classes)
    predicted_img = predicted_img.squeeze(0)
    predicted_img = predicted_img.permute(1, 2, 0)
    predicted_img = predicted_img * 255
    predicted_img = predicted_img.byte()
    predicted_img = Image.fromarray(predicted_img.numpy())
    predicted_img.show()


if __name__ == '__main__':
    main()
