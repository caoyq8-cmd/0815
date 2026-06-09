import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def normalize(x, x_max = 1595.1279, x_min = 1408.692):
    # 进行到 score_model 需要的维度[256,256]和数值范围[-1,1]
        x = F.interpolate(x[:,:,90:390,90:390], size=(256, 256), mode='bilinear', align_corners=False)
        return 2 * (x - x_min)/(x_max - x_min) - 1

def denormalize(x, x_max = 1595.1279, x_min = 1408.692):
    # 返回到 解反问题 需要的维度[480,480]和数值范围[1408.692, 1595.1279];
        x = F.interpolate(x, size=(300, 300), mode='bilinear', align_corners=False)
        x = (x + 1) * (x_max - x_min) / 2 + x_min
        return F.pad(x, (90, 90, 90, 90), mode='constant', value= 1500)

def score_normalize(x, x_max = 1595.1279, x_min = 1408.692):
    # score_model 单纯将数值范围变为[-1, 1];
        return 2 * (x - x_min)/(x_max - x_min) - 1

def score_denormalize(x, x_max = 1595.1279, x_min = 1408.692):
    # score_model 单纯将数值范围变为[1408.692, 1595.1279];
        return (x + 1) * (x_max - x_min) / 2 + x_min

def calculate_psnr(I, K, max_pixel = 1595.1279 - 1408.692):
    mse = np.mean((I - K) ** 2)
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

def calculate_ssim(x, y, L= 1595.1279 - 1408.692, k1=0.01, k2=0.03):
    # 计算常数 C1 和 C2
    C1, C2 = (k1 * L) ** 2, (k2 * L) ** 2
    # 计算均值
    mu_x, mu_y = np.mean(x), np.mean(y)
    # 计算方差&协方差
    sigma_x, sigma_y = np.var(x), np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    # 计算SSIM
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
   
    ssim_value = numerator / denominator
    return ssim_value

def compute_batch_metrics(pred_batch, output_batch):
    # 计算 PSNR
    psnr_value_free = calculate_psnr(pred_batch[0,0], output_batch[0,0])
    psnr_value_10db = calculate_psnr(pred_batch[1,0], output_batch[0,0])
    psnr_value_5db = calculate_psnr(pred_batch[2,0], output_batch[0,0])

    # 计算 SSIM
    ssim_value_free = calculate_ssim(pred_batch[0,0], output_batch[0,0])
    ssim_value_10db = calculate_ssim(pred_batch[1,0], output_batch[0,0])
    ssim_value_5db = calculate_ssim(pred_batch[2,0], output_batch[0,0])
    
    psnr_list = [psnr_value_free, psnr_value_10db, psnr_value_5db]
    ssim_list = [ssim_value_free, ssim_value_10db, ssim_value_5db]
    
    return psnr_list, ssim_list
    
def log_image(path, field, pred_field, psnr_values=None, ssim_values=None, min_output=1408.692, max_output=1595.1279):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    vmin, vmax = min_output, max_output
    
    # 显示真实图像（第一个子图）
    im1 = axes[0].imshow(field[0, 0, :, :], cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0].set_title("Field")
    axes[0].axis('off')  # 关闭坐标轴

    # 定义后三个子图的标题和对应的数据索引
    titles = ["Free Predicted Field", "10db Predicted Field", "5db Predicted Field"]
    indices = [0, 1, 2]  # pred_field 中的索引
    
    for i in range(1, 4):
        # 显示重建图像
        im = axes[i].imshow(pred_field[indices[i-1], 0, :, :], cmap="inferno", vmin=vmin, vmax=vmax)
        axes[i].set_title(titles[i-1])
        axes[i].axis('off')  # 关闭坐标轴

        # 添加PSNR和SSIM（仅当传入值时生效）
        if psnr_values is not None and ssim_values is not None:
            text = f"PSNR: {psnr_values[i-1]:.2f}\nSSIM: {ssim_values[i-1]:.4f}"
            axes[i].text(0.03, 
                 0.97, text, transform=axes[i].transAxes, color='white',
                verticalalignment='top', fontsize=15,
                bbox=dict(facecolor='black', alpha=0.4, edgecolor='none')
            )

        # 只在最后一个子图添加colorbar
        #if i == 3:
        #    fig.colorbar(im, ax=axes[i], shrink=0.8)  # shrink参数调整colorbar大小

    # 调整子图间距和边距
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01)  # 调整水平间距，数值越小间距越紧

    plt.savefig(path, bbox_inches='tight')  # bbox_inches确保保存完整内容
    plt.show()
    plt.close()


def plot2Dimage(x_start=0,x_end=128,
                y_start=0,y_end=128,
                dx=1,f=None,
                cmap='RdBu_r',
                title="Wave Field",
                x_label='X',
                y_label='Y',
                figsize=(7,5),
                levels=100):

    x = torch.arange(x_start,x_end)*dx
    y = torch.arange(y_start,y_end)*dx
    X ,Y = torch.meshgrid(x,y,indexing='ij')
    X = X.flatten()
    Y = Y.flatten()
    u = f.flatten()
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.tricontourf(X,Y,u,levels=levels,cmap=cmap)
    cbar = plt.colorbar(image)
    cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3e'))
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.invert_yaxis()
    plt.show()



class RelL2Loss(object):
    def __init__(self, p=2, size_average=True, reduction=True):
        super(RelL2Loss, self).__init__()
        assert p > 0
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
    def __call__(self, x, y,batch_size=None):
        if batch_size is None:
            batch_size = x.size()[0]
        
        diff_norms = torch.norm(x.reshape(batch_size,-1) - y.reshape(batch_size,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(batch_size,-1), self.p, 1)
        
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms