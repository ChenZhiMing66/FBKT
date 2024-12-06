import os
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class StanfordCarDataset(Dataset):
    def __init__(self, root, session, train=True, transform=None, crop_transform=None, secondary_transform=None):
        self.root = os.path.expanduser(root)
        self.session = session  # Session number (1 to 11)
        self.train = train  # Training or testing
        self.multi_train = False  # training set or test set
        self.transform = transform
        self.crop_transform = crop_transform
        self.secondary_transform = secondary_transform
        self.data = []
        self.targets = []
        self._preprocess()

        if train:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _preprocess(self):
        if self.train:
            folder = osp.join(self.root, 'FSCIL_train', f'session{self.session}')
            self._load_data_from_folder(folder)
        else:
            for sess in range(1, self.session + 1):
                folder = osp.join(self.root, 'FSCIL_test', f'session{sess}')
                self._load_data_from_folder(folder)

    def _load_data_from_folder(self, folder):
        for class_name in os.listdir(folder):
            class_folder = osp.join(folder, class_name)
            if not osp.isdir(class_folder):
                continue
            class_label = int(class_name)-1  # Convert class_name to integer
            for img_name in os.listdir(class_folder):
                img_path = osp.join(class_folder, img_name)
                self.data.append(img_path)
                self.targets.append(class_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, target = self.data[index], self.targets[index]
        if self.multi_train:
            image = Image.open(path).convert('RGB')
            classify_image = [self.transform(image)]
            multi_crop, multi_crop_params = self.crop_transform(image)
            assert (len(multi_crop) == self.crop_transform.N_large + self.crop_transform.N_small)
            if isinstance(self.secondary_transform, list):
                multi_crop = [tf(x) for tf, x in zip(self.secondary_transform, multi_crop)]
            else:
                multi_crop = [self.secondary_transform(x) for x in multi_crop]
            total_image = classify_image + multi_crop
        else:
            total_image = self.transform(Image.open(path).convert('RGB'))
        # image = Image.open(path).convert('RGB')
        # if self.transform:
        #     image = self.transform(image)
        return total_image, target