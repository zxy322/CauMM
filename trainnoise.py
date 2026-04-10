import warnings
warnings.filterwarnings("ignore")

import os
os.environ['CUDA_VISIBLE_DEVICES']='1,3'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.vit import interpolate_pos_embed
from transformers import BertTokenizerFast

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import logging
from types import MethodType
from tools.env import init_dist
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from models import box_ops
from tools.multilabel_metrics import AveragePrecisionMeter, get_multi_label
# from models.HAMMER import HAMMER
from models.fusionnoise2 import HAMMER
#from models.fusiongpt import HAMMER

def setlogger(log_file):          #日志记录器
    filehandler = logging.FileHandler(log_file) #将日志信息输出到指定的文件中
    streamhandler = logging.StreamHandler() #将日志信息输出到控制台

    logger = logging.getLogger('') #默认的日志记录器
    logger.setLevel(logging.INFO) #设置日志记录器的日志级别为 INFO
    logger.addHandler(filehandler) #将日志写入文件的处理器
    logger.addHandler(streamhandler) #将日志写入控制台的处理器

    def epochInfo(self, set, idx, loss, acc): #self表示日志对象
        self.info('{set}-{idx:d} epoch | loss:{loss:.8f} | auc:{acc:.4f}%'.format( #打印日志消息
            set=set, #数据集名称
            idx=idx, #训练/测试周期索引
            loss=loss, #当前周期的损失值
            acc=acc #当前周期的准确率
        ))

    logger.epochInfo = MethodType(epochInfo, logger) #将定义的 epochInfo 方法绑定到 logger 对象上，logger对象就具有了epochInfo方法

    return logger

def text_input_adjust(text_input, fake_word_pos, device):
    # 确保 text_input 是字典格式
    if isinstance(text_input, dict):
        # 直接使用字典
        input_ids = text_input['input_ids']
        attention_mask = text_input['attention_mask']
    else:
        # 如果传入的是其他格式（如 BatchEncoding），转换为字典
        input_ids = text_input.input_ids
        attention_mask = text_input.attention_mask

    # 移动到设备
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # 移除每个序列的SEP标记（最后一个标记）
    input_ids_remove_SEP = input_ids[:, :-1]  # [batch_size, seq_len-1]
    attention_mask_remove_SEP = attention_mask[:, :-1]  # [batch_size, seq_len-1]

    # 创建新的 text_input 字典
    new_text_input = {
        'input_ids': input_ids_remove_SEP,
        'attention_mask': attention_mask_remove_SEP
    }

    # 获取批次大小和序列长度
    batch_size, seq_len = input_ids_remove_SEP.shape

    # 创建伪造token位置的二进制张量
    fake_token_pos_tensor = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

    # 处理每个样本的伪造词位置
    for i in range(batch_size):
        # 获取当前样本的伪造词位置
        if isinstance(fake_word_pos, torch.Tensor):
            fake_word_pos_i = fake_word_pos[i].cpu().numpy()
        else:
            fake_word_pos_i = fake_word_pos[i]

        fake_word_pos_decimal = np.where(fake_word_pos_i == 1)[0].tolist()

        # 简化处理：直接使用单词索引作为token索引
        for word_idx in fake_word_pos_decimal:
            if word_idx < seq_len:  # 确保位置在序列长度范围内
                fake_token_pos_tensor[i, word_idx] = 1

    return new_text_input, fake_token_pos_tensor

def train(args, model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config,
          summary_writer):
    # 训练模式设置
    model.train()

    # 初始化指标记录器
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_MAC', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_BIC', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_bbox', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_giou', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_TMG', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_MLC', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_V', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))  # 新增LossV记录
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    # 训练参数设置
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 100
    step_size = 100
    warmup_iterations = warmup_steps * step_size
    global_step = epoch * len(data_loader)

    # 分布式训练设置
    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    # 迭代训练数据
    for i, batch in enumerate(metric_logger.log_every(args, data_loader, print_freq, header)):


        if config['schedular']['sched'] == 'cosine_in_step':
            scheduler.adjust_learning_rate(optimizer, i / len(data_loader) + epoch, args, config)

        optimizer.zero_grad()  # 清空梯度缓存

        # ========== 数据处理部分 ==========
        # 提取批次数据
        orig_images = batch['orig_image'].to(device, non_blocking=True)  # [B, 2, C, H, W]
        noise_images = batch['noise_image'].to(device, non_blocking=True)
        labels = batch['label']  # [B, 1]
        texts = batch['caption']  # 文本列表（长度B）
        fake_image_boxes = batch['fake_image_box'].to(device, non_blocking=True)  # [B, 4]
        fake_word_pos = batch['fake_text_pos'].to(device, non_blocking=True)  # [B, max_words]

        # ========== 文本处理部分 ==========
        text_input = tokenizer(
            texts,
            max_length=128,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            padding='max_length',  # 确保填充到相同长度
            return_tensors='pt'  # 返回PyTorch张量
        )

        # 调整文本输入格式
        # fake_word_pos_repeated = torch.cat([fake_word_pos, fake_word_pos], dim=0)  # 重复fake_word_pos
        text_input, fake_token_pos = text_input_adjust(text_input, fake_word_pos, device)

        # 根据 epoch 和当前步数计算 alpha 值，用于损失加权
        if epoch > 0:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        # ========== 模型前向传播 ==========
        # 调用模型获取所有损失项
        loss_MAC, loss_BIC, loss_bbox, loss_giou, loss_TMG, loss_MLC, loss_V = model(
            orig_images,
            noise_images,
            labels,
            text_input,
            fake_image_boxes,
            fake_token_pos,
            alpha=alpha
        )

        # ========== 损失计算 ==========
        # 加权总损失（包含新增的LossV）
        total_loss = (
                config['loss_MAC_wgt'] * loss_MAC.mean() +
                config['loss_BIC_wgt'] * loss_BIC.mean() +
                config['loss_bbox_wgt'] * loss_bbox.mean() +
                config['loss_giou_wgt'] * loss_giou.mean() +
                config['loss_TMG_wgt'] * loss_TMG.mean() +
                config['loss_MLC_wgt'] * loss_MLC.mean() +
                config['loss_V_wgt'] * loss_V.mean()  # 新增LossV加权
        )

        # ========== 反向传播和优化 ==========
        total_loss.backward()
        # 在 loss.backward() 后添加
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(f"Gradient explosion/vanishing detected at epoch {epoch}, step {i}")
            break

        optimizer.step()

        # torch.cuda.empty_cache()

        # ========== 记录指标 ==========
        metric_logger.update(loss_MAC=loss_MAC.item())
        metric_logger.update(loss_BIC=loss_BIC.item())
        metric_logger.update(loss_bbox=loss_bbox.item())
        metric_logger.update(loss_giou=loss_giou.item())
        metric_logger.update(loss_TMG=loss_TMG.item())
        metric_logger.update(loss_MLC=loss_MLC.item())
        metric_logger.update(loss_V=loss_V.item())  # 记录LossV
        metric_logger.update(loss=total_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # 如果是 warmup 阶段，且调度器不是 'cosine_in_step'，进行学习率调整
        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations and config['schedular'][
            'sched'] != 'cosine_in_step':
            scheduler.step(i // step_size)  # 更新学习率

        global_step += 1

        # ========== TensorBoard 日志记录 ==========
        if args.log:
            loss_info = {
                'lr': optimizer.param_groups[0]["lr"],
                'loss_MAC': loss_MAC.item(),
                'loss_BIC': loss_BIC.item(),
                'loss_bbox': loss_bbox.item(),
                'loss_giou': loss_giou.item(),
                'loss_TMG': loss_TMG.item(),
                'loss_MLC': loss_MLC.item(),
                'loss_V': loss_V.item(),  # 记录LossV
                'loss': total_loss.item(),
            }
            for tag, value in loss_info.items():
                summary_writer.add_scalar(tag, value, global_step)

    # 同步所有进程的指标
    metric_logger.synchronize_between_processes()
    if args.log:
        print("Averaged stats:", metric_logger.global_avg(), flush=True)

    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluation(args, model, data_loader, tokenizer, device, config):
    # 评估模式设置
    model.eval()

    # 初始化指标记录器
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    start_time = time.time()
    print_freq = 200

    # 初始化评估指标容器
    y_true, y_pred, IOU_pred, IOU_50, IOU_75, IOU_95 = [], [], [], [], [], []
    cls_nums_all, cls_acc_all = 0, 0
    TP_all, TN_all, FP_all, FN_all = 0, 0, 0, 0

    # 多标签分类评估器
    multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
    multi_label_meter.reset()

    # 处理验证数据
    for i, batch in enumerate(metric_logger.log_every(args, data_loader, print_freq, header)):
        # ========== 数据处理 ==========
        # 提取批次数据
        orig_images = batch['orig_image'].to(device, non_blocking=True)  # [B, 2, C, H, W] 或 [B, C, H, W]
        noise_images = batch['noise_image'].to(device, non_blocking=True)
        labels = batch['label']  # 标签
        texts = batch['caption']  # 文本描述
        fake_image_boxes = batch['fake_image_box']  # 边界框
        fake_word_pos = batch['fake_text_pos']  # 伪造文本位置


        # ========== 文本处理 ==========
        text_input = tokenizer(
            texts,
            max_length=128,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            padding='max_length',  # 确保填充到相同长度
            return_tensors='pt'  # 返回PyTorch张量
        )

        # 调整文本输入格式
        text_input, fake_token_pos = text_input_adjust(text_input, fake_word_pos.to(device), device)

        # ========== 模型推理 ==========
        # 调用模型获取推理结果
        logits_real_fake, logits_multicls, output_coord, logits_tok = model(
            orig_images,
            noise_images,
            labels,
            text_input,
            fake_image_boxes.to(device),
            fake_token_pos,
            is_train=False
        )

        ##================= real/fake cls ========================##
        cls_label = torch.ones(len(labels), dtype=torch.long).to(orig_images.device)
        # 初始化分类标签为1，表示虚假（fake）
        real_label_pos = np.where(np.array(labels) == 'orig')[0].tolist()
        # 找到真实标签的位置，'orig' 表示原图
        cls_label[real_label_pos] = 0  # 对于真实样本，将其标签设置为0

        # noise_cls_label = torch.ones(len(labels), dtype=torch.long).to(noise_images.device)
        # # 初始化分类标签为1，表示虚假（fake）
        # real_label_pos = np.where(np.array(labels) == 'orig')[0].tolist()
        # # 找到真实标签的位置，'orig' 表示原图
        # noise_cls_label[real_label_pos] = 0  # 对于真实样本，将其标签设置为0

        # 收集预测概率和真实标签
        y_pred.extend(F.softmax(logits_real_fake, dim=1)[:, 1].cpu().flatten().tolist())
        y_true.extend(cls_label.cpu().flatten().tolist())
        # y_true.extend(noise_cls_label.cpu().flatten().tolist())

        # 计算分类准确率
        pred_acc = logits_real_fake.argmax(1)
        cls_nums_all += cls_label.shape[0] #+noise_cls_label.shape[0]
        cls_acc_all += torch.sum(pred_acc == cls_label).item() #+torch.sum(pred_acc == noise_cls_label).item()

        # ========== 多标签分类评估 ==========
        # 获取多标签目标并更新评估器
        target, _ = get_multi_label(labels, orig_images)
        # noise_target, _ = get_multi_label(labels, noise_images)

        multi_label_meter.add(logits_multicls, target.to(device))
        # multi_label_meter.add(logits_multicls, noise_target.to(device))


        # ========== 边界框评估 ==========
        # 确保边界框数据有效
        if fake_image_boxes is not None and output_coord is not None:
            # 转换框坐标格式
            boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
            boxes2 = box_ops.box_cxcywh_to_xyxy(fake_image_boxes.to(device))

            # 计算IoU
            IOU, _ = box_ops.box_iou(boxes1, boxes2, test=True)

            # 收集IoU值
            IOU_pred.extend(IOU.cpu().tolist())

            IOU_50_bt = torch.zeros(IOU.shape, dtype=torch.long)
            IOU_75_bt = torch.zeros(IOU.shape, dtype=torch.long)
            IOU_95_bt = torch.zeros(IOU.shape, dtype=torch.long)

            # 计算不同IoU阈值的准确率
            IOU_50_bt[IOU > 0.5] = 1  # IoU超过0.5时设为1
            IOU_75_bt[IOU > 0.75] = 1  # IoU超过0.75时设为1
            IOU_95_bt[IOU > 0.95] = 1  # IoU超过0.95时设为1

            IOU_50.extend(IOU_50_bt.cpu().tolist())
            IOU_75.extend(IOU_75_bt.cpu().tolist())
            IOU_95.extend(IOU_95_bt.cpu().tolist())

            ##================= token cls ========================## (token分类评估)
            token_label = text_input['attention_mask'][:, 1:].clone()
            # token_label = text_input.attention_mask[:, 1:].clone()  # 创建token标签，忽略CLS标记
            token_label[token_label == 0] = -100  # 将padding token设置为-100（忽略）
            token_label[token_label == 1] = 0  # 将实际token设置为0

            for batch_idx in range(len(fake_token_pos)):  # 遍历批次
                fake_pos_sample = fake_token_pos[batch_idx]  # 获取假词位置
                # 修复：使用显式条件判断
                if fake_pos_sample is not None and torch.any(fake_pos_sample):
                    # 获取所有True位置索引
                    positions = torch.nonzero(fake_pos_sample, as_tuple=True)[0].tolist()
                    for pos in positions:
                        if pos < token_label.shape[1]:  # 确保位置有效
                            token_label[batch_idx, pos] = 1

            logits_tok_reshape = logits_tok.view(-1, 2)  # 将token logits展平并重塑为2个类别
            logits_tok_pred = logits_tok_reshape.argmax(1)  # 获取每个token的预测类别
            token_label_reshape = token_label.view(-1)  # 展平真实标签

            # 计算混淆矩阵统计
            TP_all += torch.sum((token_label_reshape == 1) & (logits_tok_pred == 1)).item()
            TN_all += torch.sum((token_label_reshape == 0) & (logits_tok_pred == 0)).item()
            FP_all += torch.sum((token_label_reshape == 0) & (logits_tok_pred == 1)).item()
            FN_all += torch.sum((token_label_reshape == 1) & (logits_tok_pred == 0)).item()

    # ========== 计算整体指标 ==========
    # 真实/虚假分类指标
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    AUC_cls = roc_auc_score(y_true, y_pred)
    ACC_cls = cls_acc_all / cls_nums_all
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)  # 计算FPR和TPR
    EER_cls = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)  # 计算Equal Error Rate (EER)

    ##================= multi-label cls ========================##
    MAP = multi_label_meter.value().mean()  # 计算多标签分类的平均精度
    OP, OR, OF1, CP, CR, CF1 = multi_label_meter.overall()  # 计算多标签的整体评估指标
    OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = multi_label_meter.overall_topk(3)  # 计算多标签的top-3评估指标

    ##================= bbox cls ========================##
    IOU_score = sum(IOU_pred) / len(IOU_pred)  # 计算IoU平均值 if len(IOU_pred)>0 else 0.0
    IOU_ACC_50 = sum(IOU_50) / len(IOU_50)  # 计算IoU>0.5的准确率 if len(IOU_50)>0 else 0.0
    IOU_ACC_75 = sum(IOU_75) / len(IOU_75)  # 计算IoU>0.75的准确率if len(IOU_75)>0 else 0.0
    IOU_ACC_95 = sum(IOU_95) / len(IOU_95)  # 计算IoU>0.95的准确率if len(IOU_95)>0 else 0.0

    ##================= token cls ========================##
    ACC_tok = (TP_all + TN_all) / (TP_all + TN_all + FP_all + FN_all)  # 计算token分类准确率
    Precision_tok = TP_all / (TP_all + FP_all)  # 计算token分类精度
    Recall_tok = TP_all / (TP_all + FN_all)  # 计算token分类召回率
    F1_tok = 2 * Precision_tok * Recall_tok / (Precision_tok + Recall_tok)  # 计算token分类的F1分数

    # 返回所有指标
    return AUC_cls, ACC_cls, EER_cls, \
        MAP.item(), OP, OR, OF1, CP, CR, CF1, OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k, \
        IOU_score, IOU_ACC_50, IOU_ACC_75, IOU_ACC_95, \
        ACC_tok, Precision_tok, Recall_tok, F1_tok


def main_worker(gpu, args, config):
    # 如果gpu编号不为空，则将args.gpu设置为传入的gpu编号
    if gpu is not None:
        args.gpu = gpu

    # 初始化分布式训练
    init_dist(args)

    # 设置日志保存路径
    log_dir = os.path.join(args.output_dir, 'log' + args.log_num)
    os.makedirs(log_dir, exist_ok=True)  # 创建日志目录
    log_file = os.path.join(log_dir, 'shell.txt')  # 日志文件
    logger = setlogger(log_file)  # 设置日志记录器
    yaml.dump(config, open(os.path.join(log_dir, 'config.yaml'), 'w'))  # 保存配置文件

    # 如果需要记录日志，则初始化TensorBoard的日志记录器
    if args.log:
        summary_writer = SummaryWriter(log_dir)
    else:
        summary_writer = None

    # 如果需要记录日志，则输出配置和参数信息
    if args.log:
        logger.info('******************************')
        logger.info(args)
        logger.info('******************************')
        logger.info(config)
        logger.info('******************************')

    # 设置设备，通常是CUDA或者CPU
    device = torch.device(args.device)

    # 设置随机种子，确保可重复性
    seed = args.seed + utils.get_rank()  # 分布式训练时，不同进程使用不同的随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True  # 针对固定输入尺寸，优化算法选择

    # 初始化训练的起始epoch和最大epoch
    start_epoch = 0
    max_epoch = config['schedular']['epochs']  # 训练的最大轮数
    warmup_steps = config['schedular']['warmup_epochs']  # 学习率预热的轮数
    best = 0  # 最好的AUC值
    best_epoch = 0  # 最好的epoch

    #### 数据集 ####
    if args.log:
        print("Creating dataset")
    # 创建训练和验证数据集
    train_dataset, val_dataset = create_dataset(config)

    # 如果是分布式训练，创建分布式采样器
    if args.distributed:
        samplers = create_sampler([train_dataset], [True], args.world_size, args.rank) + [None]
    else:
        samplers = [None, None]

    # 创建数据加载器
    train_loader, val_loader = create_loader([train_dataset, val_dataset],
                                             samplers,
                                             batch_size=[config['batch_size_train']] + [config['batch_size_val']],
                                             num_workers=[4, 4],
                                             is_trains=[True, False],
                                             collate_fns=[None, None])

    # 加载BERT Tokenizer，用于文本处理
    tokenizer = BertTokenizerFast.from_pretrained(
        "/home/async/data-disk/zxy/deepfake/bert-base-uncased/uncased_L-12_H-768_A-12")

    #### 模型 ####
    if args.log:
        print(f"Creating HAMMER")
    # 创建模型，可能是自定义模型如HAMMER
    model = HAMMER(args=args, config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)
    model = model.to(device)  # 将模型移动到设备上

    # # 确保队列在正确的设备上
    # model.image_queue = model.image_queue.to(device)
    # model.text_queue = model.text_queue.to(device)
    # model.queue_ptr = model.queue_ptr.to(device)
    # # 添加队列同步
    # if dist.is_initialized():
    #
    #     # 广播队列
    #     dist.broadcast(model.image_queue, 0)
    #     dist.broadcast(model.text_queue, 0)
    #     dist.broadcast(model.queue_ptr, 0)

    # 创建优化器
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    print(arg_sche)
    # 创建学习率调度器
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)
    if config['schedular']['sched'] == 'cosine_in_step':
        args.lr = config['optimizer']['lr']

    # 加载已有的checkpoint
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')  # 加载checkpoint文件
        state_dict = checkpoint['model']  # 提取模型参数
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])  # 恢复优化器状态
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])  # 恢复学习率调度器状态
            start_epoch = checkpoint['epoch'] + 1  # 设置训练的起始epoch
        else:
            # 如果是加载模型，并且不进行恢复训练，调整位置嵌入
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
            # 加载模型状态字典
        if args.log:
            print('load checkpoint from %s' % args.checkpoint)
        msg = model.load_state_dict(state_dict, strict=False)  # 加载参数
        if args.log:
            print(msg)

    model_without_ddp = model
    # 如果是分布式训练，使用DistributedDataParallel包裹模型
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        model_without_ddp = model.module

    # 开始训练
    if args.log:
        print("Start training")
    start_time = time.time()
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, max_epoch):
        # 训练一个epoch
        train_stats = train(args, model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler,
                            config, summary_writer)

        # 在验证集上评估模型
        AUC_cls, ACC_cls, EER_cls, \
            MAP, OP, OR, OF1, CP, CR, CF1, OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k, \
            IOU_score, IOU_ACC_50, IOU_ACC_75, IOU_ACC_95, \
            ACC_tok, Precision_tok, Recall_tok, F1_tok \
            = evaluation(args, model_without_ddp, val_loader, tokenizer, device, config)

        # 如果需要记录日志，使用TensorBoard记录各种指标
        if args.log:
            lossinfo = {
                'AUC_cls': round(AUC_cls * 100, 4),
                'ACC_cls': round(ACC_cls * 100, 4),
                'EER_cls': round(EER_cls * 100, 4),
                'MAP': round(MAP * 100, 4),
                'OP': round(OP * 100, 4),
                'OR': round(OR * 100, 4),
                'OF1': round(OF1 * 100, 4),
                'CP': round(CP * 100, 4),
                'CR': round(CR * 100, 4),
                'CF1': round(CF1 * 100, 4),
                'OP_k': round(OP_k * 100, 4),
                'OR_k': round(OR_k * 100, 4),
                'OF1_k': round(OF1_k * 100, 4),
                'CP_k': round(CP_k * 100, 4),
                'CR_k': round(CR_k * 100, 4),
                'CF1_k': round(CF1_k * 100, 4),
                'IOU_score': round(IOU_score * 100, 4),
                'IOU_ACC_50': round(IOU_ACC_50 * 100, 4),
                'IOU_ACC_75': round(IOU_ACC_75 * 100, 4),
                'IOU_ACC_95': round(IOU_ACC_95 * 100, 4),
                'ACC_tok': round(ACC_tok * 100, 4),
                'Precision_tok': round(Precision_tok * 100, 4),
                'Recall_tok': round(Recall_tok * 100, 4),
                'F1_tok': round(F1_tok * 100, 4),
            }
            # 将每个指标写入TensorBoard
            for tag, value in lossinfo.items():
                summary_writer.add_scalar(tag, value, epoch)

        # 记录验证集评估指标
        val_stats = {"AUC_cls": "{:.4f}".format(AUC_cls * 100),
                     "ACC_cls": "{:.4f}".format(ACC_cls * 100),
                     "EER_cls": "{:.4f}".format(EER_cls * 100),
                     "MAP": "{:.4f}".format(MAP * 100),
                     "OP": "{:.4f}".format(OP * 100),
                     "OR": "{:.4f}".format(OR * 100),
                     "OF1": "{:.4f}".format(OF1 * 100),
                     "CP": "{:.4f}".format(CP * 100),
                     "CR": "{:.4f}".format(CR * 100),
                     "CF1": "{:.4f}".format(CF1 * 100),
                     "OP_k": "{:.4f}".format(OP_k * 100),
                     "OR_k": "{:.4f}".format(OR_k * 100),
                     "OF1_k": "{:.4f}".format(OF1_k * 100),
                     "CP_k": "{:.4f}".format(CP_k * 100),
                     "CR_k": "{:.4f}".format(CR_k * 100),
                     "CF1_k": "{:.4f}".format(CF1_k * 100),
                     "IOU_score": "{:.4f}".format(IOU_score * 100),
                     "IOU_ACC_50": "{:.4f}".format(IOU_ACC_50 * 100),
                     "IOU_ACC_75": "{:.4f}".format(IOU_ACC_75 * 100),
                     "IOU_ACC_95": "{:.4f}".format(IOU_ACC_95 * 100),
                     "ACC_tok": "{:.4f}".format(ACC_tok * 100),
                     "Precision_tok": "{:.4f}".format(Precision_tok * 100),
                     "Recall_tok": "{:.4f}".format(Recall_tok * 100),
                     "F1_tok": "{:.4f}".format(F1_tok * 100),
                     }

        # 保存日志
        if utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         'epoch': epoch,
                         }
            with open(os.path.join(log_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # 保存模型和优化器的状态
            if config['schedular']['sched'] != 'cosine_in_step':
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
            else:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr': optimizer.param_groups[0]["lr"],
                    'config': config,
                    'epoch': epoch,
                }

            # 每隔一定轮次保存模型
            if (epoch % args.model_save_epoch == 0 and epoch != 0):
                torch.save(save_obj, os.path.join(log_dir, 'checkpoint_%02d.pth' % epoch))

            # 保存最佳模型
            if float(val_stats['AUC_cls']) > best:
                torch.save(save_obj, os.path.join(log_dir, 'checkpoint_best.pth'))
                best = float(val_stats['AUC_cls'])
                best_epoch = epoch

                # 学习率调度
        if config['schedular']['sched'] != 'cosine_in_step':
            lr_scheduler.step(epoch + warmup_steps + 1)
        dist.barrier()  # 所有进程同步

        # 保存最终的模型
    if utils.is_main_process():
        torch.save(save_obj, os.path.join(log_dir, 'checkpoint_%02d.pth' % epoch))

        # 计算训练总时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if args.log:
        print('Training time {}'.format(total_time_str))
        with open(os.path.join(log_dir, "log.txt"), "a") as f:
            f.write("best epoch: {}, Training time: {}".format(best_epoch, total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--text_encoder', default='/home/async/data-disk/zxy/deepfake/bert-base-uncased/uncased_L-12_H-768_A-12')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='world size for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23459', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                        help='job launcher')
    parser.add_argument('--log_num', '-l', type=str)
    parser.add_argument('--model_save_epoch', type=int, default=20)
    parser.add_argument('--token_momentum', default=False, action='store_true')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # main(args, config)
    if args.launcher == 'none':
        args.launcher = 'pytorch'
        main_worker(0, args, config)
    else:
        ngpus_per_node = torch.cuda.device_count()
        args.ngpus_per_node = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args, config))