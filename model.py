import torch
import torch.nn as nn
import torch.nn.functional as F
from pretrainedmodels.models.senet import SENet, SEResNeXtBottleneck
import numpy as np


class PointsToImage(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i, v):
        device = i.device
        batch_size, _, num_input_points = i.size()
        feature_size = v.size()[2]

        batch_idx = torch.arange(batch_size, device=device).view(-1, 1).repeat(1, num_input_points).view(-1)
        idx_full = torch.cat([batch_idx.unsqueeze(0), i.permute(1, 0, 2).contiguous().view(2, -1)], dim=0)

        v_full = v.contiguous().view(batch_size * num_input_points, feature_size)
        if device.type == 'cuda':
            mat_sparse = torch.cuda.sparse.FloatTensor(idx_full, v_full)
        else:
            mat_sparse = torch.sparse_coo_tensor(idx_full, v_full)
        mat_dense = mat_sparse.to_dense()

        ones_full = torch.ones(v_full.size(), device=device)
        mat_sparse_count = torch.sparse_coo_tensor(idx_full, ones_full)
        mat_dense_count = mat_sparse_count.to_dense()

        ctx.save_for_backward(idx_full, mat_dense_count)

        return mat_dense / torch.clamp(mat_dense_count, 1, 1e4)

    @staticmethod
    def backward(ctx, grad_output):
        idx_full, mat_dense_count = ctx.saved_tensors
        grad_i = grad_v = None

        batch_size, _, _, feature_size = grad_output.size()

        if ctx.needs_input_grad[0]:
            raise Exception("Indices aren't differentiable.")
        if ctx.needs_input_grad[1]:
            grad = grad_output[idx_full[0], idx_full[1], idx_full[2]]
            coef = mat_dense_count[idx_full[0], idx_full[1], idx_full[2]]
            grad_v = grad / coef
            grad_v = grad_v.view(batch_size, -1, feature_size)

        return grad_i, grad_v


points_to_image = PointsToImage.apply


def batch_index_select(tensor, idx, dim):
    assert dim != 0, "dim 0 invalid, this is the batch dim"

    device = tensor.device
    batch_size = tensor.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    if dim == 1:
        return tensor[batch_indices, idx]
    elif dim == 2:
        return tensor[batch_indices, :, idx]
    elif dim == 3:
        return tensor[batch_indices, :, :, idx]
    elif dim == 4:
        return tensor[batch_indices, :, :, :, idx]
    else:
        raise NotImplementedError("Sorry, haven't figured out how to deliver infinite flexibility here.")


class StrokesToSeResNeXt(SENet):
    def __init__(self, img_size, window, num_classes, block, layers, groups, reduction, dropout_p=0.2, inplanes=128, input_3x3=True,
                 downsample_kernel_size=3, downsample_padding=1):
        nn.Module.__init__(self)
        self.img_size = img_size
        self.window = window
        self.num_classes = num_classes

        self.initial = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, stride=1, padding=1, dilation=1), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=2, dilation=2), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=4, dilation=4), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=8, dilation=8), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
        )

        self.convert = nn.Sequential(
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True)
        )

        self.inplanes = inplanes
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.cls = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, points_tensor, indices_tensor):
        xy = batch_index_select(points_tensor, indices_tensor, 2).permute(0, 2, 1)

        dxy = points_tensor[:, :2, 1:] - points_tensor[:, :2, :-1]
        dxy /= 0.035
        t = (points_tensor[:, 2, 1:] + points_tensor[:, 2, :-1]) / 2
        x = torch.cat([dxy, t.unsqueeze(1)], dim=1)

        x = self.initial(x)

        assert self.window % 2 == 0
        x = x.unfold(2, self.window, 1)

        x = batch_index_select(x, indices_tensor - 1 - (self.window // 2) + 1, 2)
        batch_size = x.size()[0]
        num_points = x.size()[1]
        x = x.view(batch_size, num_points, -1).permute(0, 2, 1)

        x = self.convert(x)

        i = ((xy[:, :2, :]+1)*((self.img_size - 1) / 2)).long()
        i = torch.clamp(i, 0, self.img_size - 1)

        img = points_to_image(i, x.permute(0, 2, 1)).permute(0, 3, 1, 2)

        x = self.layer1(img)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.cls(x)

        return x


def strokes_to_seresnext50_32x4d(img_size, window, num_classes):
    return StrokesToSeResNeXt(
        img_size, window, num_classes,
        block=SEResNeXtBottleneck,
        layers=[3, 4, 6, 3],
        groups=32,
        reduction=16,
        dropout_p=None,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        downsample_padding=0
    )


def resample_to(drawing, n):
    total_len = 0
    for stroke in drawing:
        total_len += len(stroke[0])
    
    if total_len <= 0:
        return drawing
    
    result = []
    for stroke in drawing:
        stroke_len = len(stroke[0])
        if stroke_len <= 0:
            continue
            
        stroke_n = max(1, round(n * stroke_len / total_len))
        
        t = np.linspace(0, 1, stroke_len)
        t_resampled = np.linspace(0, 1, stroke_n)
        
        x_resampled = np.interp(t_resampled, t, stroke[0])
        y_resampled = np.interp(t_resampled, t, stroke[1])
        
        result.append([x_resampled.tolist(), y_resampled.tolist()])
    
    return result


def process_single_drawing(drawing, out_size=2048, actual_points=256, padding=16):
    # Resample the drawing
    drawing = resample_to(drawing, actual_points)
    
    # Initialize arrays
    points = np.zeros((3, out_size), dtype=np.float32)
    indices = np.zeros(actual_points, dtype=np.int64)
    
    # Process each stroke
    idx = padding
    for stroke in drawing:
        n = len(stroke[0])
        points[0, idx:idx + n] = stroke[0]
        points[1, idx:idx + n] = stroke[1]
        points[2, idx:idx + n] = 1
        idx += n + padding
    
    # Calculate indices
    idx = 0
    curr_idx = padding
    for stroke in drawing:
        n = len(stroke[0])
        indices[idx:idx + n] = np.arange(curr_idx, curr_idx + n)
        idx += n
        curr_idx += n + padding
    
    # Normalize points
    points = points * 2 - 1
    
    return points, indices 