import time

from PIL import Image

import torch
import torch.optim


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, select_confident_samples, avg_entropy
from ttabc.model_selection.base_method import BaseMethod

def cosine_similarity_loss(E, K=1000):
    """
    计算矩阵 E 中所有向量两两之间的余弦相似度，
    减去 1 / (1 - K) 后的每个元素平方，再求平均值。

    参数:
    -------
    E : torch.Tensor
        输入矩阵，形状为 (K, d)，包含 K 个 d 维向量。
    K : int
        向量的数量，等于 E 的第一维度大小。

    返回:
    -------
    loss : torch.Tensor
        余弦相似度损失，标量值，保留梯度信息。
    """
    # Step 1: 归一化 E 的每个向量（计算单位向量）
    E_normalized = torch.nn.functional.normalize(E, dim=1)  # Shape: (K, d)
    # Step 2: 计算余弦相似度矩阵 (K x K)
    cosine_sim_matrix = torch.matmul(E_normalized, E_normalized.T)  # Shape: (K, K)
    # Step 3: 减去对角线上的自身相似度 (排除对角线)
    # 设置对角线上的值为 0，避免自相似度的影响
    cosine_sim_matrix = cosine_sim_matrix - torch.diag_embed(torch.diagonal(cosine_sim_matrix))  # Shape: (K, K)
    # Step 4: 减去常数 1 / (1 - K)
    constant = 1 / (1 - K)
    # constant = 0.5965
    adjusted_cosine_sim = cosine_sim_matrix - constant  # Shape: (K, K)
    # Step 5: 对每个元素平方后求均值
    squared_adjusted_cosine_sim = adjusted_cosine_sim ** 2  # 每个元素平方
    loss = squared_adjusted_cosine_sim.sum() / (K * (K - 1))  # 排除对角线，所有非对角线元素的平均
    
    # # Save cosine_sim_matrix to a CSV file
    # cosine_sim_matrix_np = cosine_sim_matrix.cpu().detach().numpy()  # Convert to numpy array
    # df = pd.DataFrame(cosine_sim_matrix_np)
    # csv_path = "/home/toot1/code/C-TPT/cosine_sim_matrix.csv"
    # df.to_csv(csv_path, index=False)
    # print(f"Cosine similarity matrix saved to {csv_path}")
    # # Calculate mean and variance
    # mean_value = torch.mean(cosine_sim_matrix)
    # variance_value = torch.var(cosine_sim_matrix)
    # print(f"Mean of cosine similarity matrix: {mean_value.item()}")
    # print(f"Variance of cosine similarity matrix: {variance_value.item()}")
    # print(f"Constant value (1/(1-K)): {constant}")
    return loss

class NTPT(BaseMethod):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.temperature_value = {'ViT': 1.16, 'RN': 1.15} #for temperature scaling experiments 

    def test_time_tuning(self, inputs):
        output = None
        output2 = None
        single_output = None
        if self.args.prompt_type == 'cocoop':
            image_feature, pgen_ctx = inputs
            pgen_ctx.requires_grad = True
            self.optimizer = torch.optim.AdamW([pgen_ctx], self.args.lr)
        
        selected_idx = None
        for j in range(self.args.tta_steps):
            with torch.amp.autocast(device_type='cuda'):
                if self.args.prompt_type == 'maple':
                    output = self.model(inputs)['logits']
                elif self.args.prompt_type == 'cocoop':
                    output = self.model((image_feature, pgen_ctx))
                else:
                    output = self.model(inputs) 

                if selected_idx is not None:
                    output = output[selected_idx]
                else:
                    output, selected_idx = select_confident_samples(output, self.args.selection_p)

                loss = avg_entropy(output)

            if self.args.two_step:
                self.optimizer.zero_grad()
                # compute gradient and do SGD step
                self.scaler.scale(loss).backward(retain_graph=True)
                # Unscales the gradients of optimizer's assigned params in-place
                self.scaler.step(self.optimizer)
                self.scaler.update()
                loss = 0

                with torch.amp.autocast(device_type='cuda'):
                    if self.args.prompt_type == 'cocoop':
                        output2 = self.model((image_feature, pgen_ctx))
                    else:
                        output2 = self.model(inputs) 
            if output == None and output2 == None:
                single_output = self.model(self.args.image)

            lambda_ = self.args.lambda_term

            number_of_class = output.shape[1]            
            
            loss += (lambda_* cosine_similarity_loss(E=self.model.get_text_features(), K=number_of_class))

            self.optimizer.zero_grad()
            # compute gradient and do SGD step
            self.scaler.scale(loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            self.scaler.step(self.optimizer)
            self.scaler.update()

        if self.args.prompt_type == 'cocoop':
            return pgen_ctx

        return None

    def test_time_adapt_eval(self, val_loader, result_dict):
        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1, top5],
            prefix='Test: ')

        # reset model and switch to evaluate mode
        self.model.eval()
        if self.args.prompt_type != 'cocoop': # no need to reset cocoop because it's fixed
            with torch.no_grad():
                self.model.reset()
        end = time.time()

        #define a softmax layer
        softmax = torch.nn.Softmax(dim=1)
    
        for i, (images, target) in enumerate(val_loader):
            assert self.args.gpu is not None
            if isinstance(images, list):
                for k in range(len(images)):
                    images[k] = images[k].cuda(self.args.gpu, non_blocking=True)
                image = images[0]
            else:
                if len(images.size()) > 4:
                    # when using ImageNet Sampler as the dataset
                    assert images.size()[0] == 1
                    images = images.squeeze(0)
                images = images.cuda(self.args.gpu, non_blocking=True)
                image = images
            target = target.cuda(self.args.gpu, non_blocking=True)
            if self.args.tpt:
                images = torch.cat(images, dim=0)

            self.args.image = image # for ctpt

            # reset the tunable prompt to its initial state
            if self.args.prompt_type != 'cocoop': # no need to reset cocoop because it's fixed
                if self.args.tta_steps > 0:
                    with torch.no_grad():
                        self.model.reset()
                self.optimizer.load_state_dict(self.optim_state)

                self.test_time_tuning(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type='cuda'):
                        image_feature, pgen_ctx = self.model.gen_ctx(images, self.args.tpt)
                self.optimizer = None

                pgen_ctx = self.test_time_tuning((image_feature, pgen_ctx))

            # The actual inference goes here
            if self.args.tpt:
                if self.args.prompt_type == 'cocoop':
                    image_feature = image_feature[0].unsqueeze(0)
            
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'):
                    if self.args.prompt_type == 'maple':
                        output = self.model(image)['logits']
                    elif self.args.prompt_type == 'cocoop':
                        output = self.model((image_feature, pgen_ctx))
                    else:
                        output = self.model(image)


            
            softmax_output = softmax(output) 

            #maximum confidence of the softmax_output and its index
            max_confidence, max_index = torch.max(softmax_output, 1)

            #save the max confidence, prediction, and label to the result_dict
            result_dict['max_confidence'].append(max_confidence.item())
            result_dict['prediction'].append(max_index.item())
            result_dict['label'].append(target.item())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % self.args.print_freq == 0:
                progress.display(i)

        progress.display_summary()

        return [top1.avg, top5.avg]