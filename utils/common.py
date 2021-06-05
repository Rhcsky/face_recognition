import math
from glob import glob

import face_recognition
import numpy as np
import torch
from tqdm import tqdm


def crop_face(img):
    location = face_recognition.face_locations(img, model='hog')
    if len(location) == 0:
        return []

    location = location[0]
    top, right, bottom, left = location
    img = img[top:bottom, left:right]
    return img


def margin_of_error(values, confidence_interval=1.96):
    num = len(values)
    mean = sum(values) / num
    variance = sum(list(map(lambda x: pow(x - mean, 2), values))) / num

    standard_deviation = math.sqrt(variance)
    standard_error = standard_deviation / math.sqrt(num)

    return mean, standard_error * confidence_interval


# from https://github.com/oscarknagg/few-shot
def pairwise_distances(x, y, compute_fn):
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.
    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        compute_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    if compute_fn.lower() == 'l2' or compute_fn.lower == 'euclidean':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
    elif compute_fn.lower() == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif compute_fn.lower() == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise (ValueError('Unsupported similarity function'))


def split_support_query_set(x, y, num_class, num_support, num_query):
    """
    x: Input
    y: Label
    num_class: Number of class per episode
    num_support: Number of examples for support set
    num_query: Number of examples for query set
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_sample_support = num_class * num_support
    x_support, x_query = x[:num_sample_support], x[num_sample_support:]
    y_support, y_query = y[:num_sample_support], y[num_sample_support:]

    _classes = torch.unique(y_support)

    support_idx = torch.stack(list(map(lambda c: torch.nonzero(y_support.eq(c), as_tuple=False).squeeze(1), _classes)))
    xs = torch.cat([x_support[idx_list] for idx_list in support_idx])

    query_idx = torch.stack(list(map(lambda c: torch.nonzero(y_query.eq(c), as_tuple=False).squeeze(1), _classes)))
    xq = torch.cat([x_query[idx_list] for idx_list in query_idx])

    ys = torch.arange(0, len(_classes), 1 / num_support).long().to(device)
    yq = torch.arange(0, len(_classes), 1 / num_query).long().to(device)

    return xs, xq, ys, yq


def get_mean_std_in_image_data(image_path, channel):
    if not (channel == 1 or channel == 3):
        raise AttributeError(f"channel must be 1 or 3, you got {channel}")

    mean_sum = np.zeros(channel, dtype=np.float16)
    std_sum = np.zeros(channel, dtype=np.float16)
    count = len(image_path)

    for image in tqdm(image_path):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if channel == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mean, std = cv2.meanStdDev(image / 255.)
        mean_sum += np.squeeze(mean)
        std_sum += np.squeeze(std)

    return mean_sum / count, std_sum / count


if __name__ == '__main__':
    image_path = glob('../data/face/image/*/*.png')
    mean, std = get_mean_std_in_image_data(image_path, channel=5)
    print(mean, std)
