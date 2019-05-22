from torchvision import transforms

def get_transforms(output_size):

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(output_size),                              #随机大小，随机长宽比裁剪原始图片，最后将图片resize到设定好的size
        transforms.ColorJitter(.4, .4, .4, .4),                                 #修改亮度、对比度和饱和度
        transforms.RandomHorizontalFlip(),                                      #依据概率p对PIL图片进行水平翻转
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])      #class torchvision.transforms.Normalize(mean, std)
        ])

    eval_transform = transforms.Compose([
        transforms.Resize((output_size, output_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    data_transforms = {
        'train': train_transform,
        'val': eval_transform,
        'test': eval_transform
    }

    return data_transforms
