import warnings
warnings.filterwarnings("ignore")

import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
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
from torch.utils.data import DataLoader
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

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from models import box_ops
from tools.multilabel_metrics import AveragePrecisionMeter, get_multi_label

#from models.HAMMER import HAMMER
from models.fusionnoise2 import HAMMER


def setlogger(log_file):#配置日志记录器
    # 创建一个文件处理器，用于将日志写入指定的文件
    filehandler = logging.FileHandler(log_file)
    # 创建一个流处理器，用于将日志输出到控制台
    streamhandler = logging.StreamHandler()

    # 获取根日志记录器
    logger = logging.getLogger('')
    # 设置日志记录器的级别为 INFO，即只记录 INFO 级别及以上的日志
    logger.setLevel(logging.INFO)
    # 将文件处理器添加到日志记录器
    logger.addHandler(filehandler)
    # 将流处理器添加到日志记录器
    logger.addHandler(streamhandler)

    # 定义一个名为 epochInfo 的方法，用于记录每个 epoch 的信息
    def epochInfo(self, set, idx, loss, acc):
        # 使用日志记录器的 info 方法记录信息，格式化字符串包含 set, idx, loss, acc 四个变量
        self.info('{set}-{idx:d} epoch | loss:{loss:.8f} | auc:{acc:.4f}%'.format(
            set=set,
            idx=idx,
            loss=loss,
            acc=acc
        ))

    # 使用 MethodType 将 epochInfo 方法动态绑定到 logger 对象上
    logger.epochInfo = MethodType(epochInfo, logger)

    # 返回配置好的日志记录器
    return logger


def text_input_adjust(text_input, fake_word_pos,
                      device):  # 调整输入的文本数据，移除 SEP token 并进行填充，同时将 fake_word_pos 转换为 fake_token_pos，以便后续处理。
    # 确保 text_input 是字典格式
    if isinstance(text_input, dict):
        # 直接使用字典
        input_ids = text_input['input_ids'].to(device)
        attention_mask = text_input['attention_mask'].to(device)
    else:
        # 如果传入的是其他格式（如 BatchEncoding），转换为字典
        input_ids = text_input.input_ids.to(device)
        attention_mask = text_input.attention_mask.to(device)

    # # 移动到设备
    # input_ids = input_ids.to(device)
    # attention_mask = attention_mask.to(device)
    # 对输入的 input_ids 进行调整，移除每个样本的最后一个 SEP token
    input_ids_remove_SEP = input_ids[:, :-1]
    attention_mask_remove_SEP = attention_mask[:, :-1]  # [batch_size, seq_len-1]

    # 创建新的 text_input 字典
    new_text_input = {
        'input_ids': input_ids_remove_SEP,
        'attention_mask': attention_mask_remove_SEP
    }
    # # 计算移除 SEP 后的最大长度
    # maxlen = max([len(x) for x in input_ids]) - 1
    # # 对每个样本进行填充，使其长度一致，填充值为 0
    # input_ids_remove_SEP_pad = [x.tolist() + [0] * (maxlen - len(x)) for x in input_ids_remove_SEP]
    # # 将处理后的 input_ids 转换为 LongTensor 并移动到指定的设备（如 GPU）
    # text_input.input_ids = torch.LongTensor(input_ids_remove_SEP_pad).to(device)
    #
    # # 对 attention_mask 进行调整，移除每个样本的最后一个 SEP token
    # # attention_mask_remove_SEP = [x[:-1] for x in attention_mask]
    # # 对每个样本进行填充，使其长度一致，填充值为 0
    # attention_mask_remove_SEP_pad = [x.tolist() + [0] * (maxlen - len(x)) for x in attention_mask_remove_SEP]
    # # 将处理后的 attention_mask 转换为 LongTensor 并移动到指定的设备
    # attention_mask = torch.LongTensor(attention_mask_remove_SEP_pad).to(device)

    # 初始化 fake_token_pos_batch 和 subword_idx_rm_CLSSEP_batch 列表
    fake_token_pos_batch = []
    subword_idx_rm_CLSSEP_batch = []
    # 遍历 fake_word_pos 列表
    for i in range(len(fake_word_pos)):
        # 初始化 fake_token_pos 列表
        fake_token_pos = []

        # 将 fake_word_pos 中的 one-hot 编码转换为具体的索引位置
        fake_word_pos_decimal = np.where(fake_word_pos[i].numpy() == 1)[0].tolist()

        # 获取当前样本的 word_ids（即每个 token 对应的单词索引）
        subword_idx = text_input.word_ids(i)
        # 移除 CLS 和 SEP token 对应的 word_ids
        subword_idx_rm_CLSSEP = subword_idx[1:-1]
        # 将处理后的 word_ids 转换为 numpy 数组
        subword_idx_rm_CLSSEP_array = np.array(subword_idx_rm_CLSSEP)

        # 将处理后的 word_ids 添加到 subword_idx_rm_CLSSEP_batch 中
        subword_idx_rm_CLSSEP_batch.append(subword_idx_rm_CLSSEP_array)

        # 将 fake_word_pos 转换为 fake_token_pos
        for i in fake_word_pos_decimal:
            # 找到与 fake_word_pos 对应的 token 位置
            fake_token_pos.extend(np.where(subword_idx_rm_CLSSEP_array == i)[0].tolist())
        # 将 fake_token_pos 添加到 fake_token_pos_batch 中
        fake_token_pos_batch.append(fake_token_pos)

    # 返回调整后的 text_input、fake_token_pos_batch 和 subword_idx_rm_CLSSEP_batch
    return new_text_input, fake_token_pos_batch, subword_idx_rm_CLSSEP_batch


@torch.no_grad()
def evaluation(args, model, data_loader, tokenizer, device, config):
    # 将模型设置为评估模式（关闭 dropout 和 batch normalization 的随机性）
    model.eval()

    # 初始化一个 MetricLogger 对象，用于记录和打印评估过程中的指标
    metric_logger = utils.MetricLogger(delimiter="  ")
    # 设置评估日志的标题
    header = 'Evaluation:'

    # 打印提示信息，表示正在计算评估特征
    print('Computing features for evaluation...')
    start_time = time.time()
    # 设置打印频率，每处理 200 个样本打印一次日志
    print_freq = 200

    # 初始化用于存储评估结果的变量
    y_true, y_pred, IOU_pred, IOU_50, IOU_75, IOU_95 = [], [], [], [], [], []
    cls_nums_all = 0  # 记录总样本数
    cls_acc_all = 0  # 记录分类正确的样本数

    # 初始化用于计算 F1 指标的变量
    TP_all = 0  # 真正例
    TN_all = 0  # 真负例
    FP_all = 0  # 假正例
    FN_all = 0  # 假负例

    # 初始化多分类任务的评估变量
    TP_all_multicls = np.zeros(4, dtype=int)  # 每个类别的真正例
    TN_all_multicls = np.zeros(4, dtype=int)  # 每个类别的真负例
    FP_all_multicls = np.zeros(4, dtype=int)  # 每个类别的假正例
    FN_all_multicls = np.zeros(4, dtype=int)  # 每个类别的假负例
    F1_multicls = np.zeros(4)  # 每个类别的 F1 分数

    # 初始化多标签分类的评估工具
    multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
    multi_label_meter.reset()  # 重置评估工具

    # 遍历数据加载器中的每个批次
    for i, batch in enumerate(metric_logger.log_every(args, data_loader, print_freq, header)):
        # ========== 数据处理 ==========
        orig_images = batch['orig_image'].to(device, non_blocking=True)  # [B, 2, C, H, W] 或 [B, C, H, W]
        noise_images = batch['noise_image'].to(device, non_blocking=True)
        labels = batch['label']  # 标签
        texts = batch['caption']  # 文本描述
        fake_image_boxes = batch['fake_image_box']  # 边界框
        fake_word_pos = batch['fake_text_pos']  # 伪造文本位置

        # 只使用原始图像（验证集通常只包含原始图像）
        # orig_images = orig_images[:, 0]  # 原始图像 [B, C, H, W]

        # ========== 文本处理 ==========
        text_input = tokenizer(
            texts,
            max_length=128,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            padding='max_length',
            return_tensors='pt'
        )
        # fake_word_pos_repeated = torch.cat([fake_word_pos, fake_word_pos], dim=0)
        text_input, fake_token_pos, _= text_input_adjust(text_input, fake_word_pos, device)

        # ========== 模型推理 ==========
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
        # 生成真实/伪造分类的标签
        cls_label = torch.ones(len(labels), dtype=torch.long).to(orig_images.device)
        # 找到真实样本的索引
        real_label_pos = np.where(np.array(labels) == 'orig')[0].tolist()
        # 将真实样本的标签设置为 0
        cls_label[real_label_pos] = 0

        # 将模型输出的概率值添加到 y_pred 中
        y_pred.extend(F.softmax(logits_real_fake, dim=1)[:, 1].cpu().flatten().tolist())
        # 将真实标签添加到 y_true 中
        y_true.extend(cls_label.cpu().flatten().tolist())

        # 计算分类准确率
        pred_acc = logits_real_fake.argmax(1)
        cls_nums_all += cls_label.shape[0]
        cls_acc_all += torch.sum(pred_acc == cls_label).item()

        # ----- multi metrics -----
        # 获取多标签分类的目标标签
        target, _ = get_multi_label(labels, orig_images)
        # 将模型输出和目标标签添加到多标签评估工具中
        multi_label_meter.add(logits_multicls, target)

        # 计算每个类别的 TP, TN, FP, FN
        for cls_idx in range(logits_multicls.shape[1]):
            cls_pred = logits_multicls[:, cls_idx]
            cls_pred[cls_pred >= 0] = 1
            cls_pred[cls_pred < 0] = 0

            TP_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 1) * (cls_pred == 1)).item()
            TN_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 0) * (cls_pred == 0)).item()
            FP_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 0) * (cls_pred == 1)).item()
            FN_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 1) * (cls_pred == 0)).item()

        ##================= bbox cls ========================##
        # 将模型输出的边界框坐标转换为 (x1, y1, x2, y2) 格式
        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        # 将真实边界框坐标转换为 (x1, y1, x2, y2) 格式
        boxes2 = box_ops.box_cxcywh_to_xyxy(fake_image_boxes.to(device))


        # 计算预测边界框和真实边界框的 IoU
        IOU, _ = box_ops.box_iou(boxes1, boxes2.to(device), test=True)

        # 将 IoU 值添加到 IOU_pred 中
        IOU_pred.extend(IOU.cpu().tolist())

        # 计算 IoU 大于 0.5, 0.75, 0.95 的指标
        IOU_50_bt = torch.zeros(IOU.shape, dtype=torch.long)
        IOU_75_bt = torch.zeros(IOU.shape, dtype=torch.long)
        IOU_95_bt = torch.zeros(IOU.shape, dtype=torch.long)

        IOU_50_bt[IOU > 0.5] = 1
        IOU_75_bt[IOU > 0.75] = 1
        IOU_95_bt[IOU > 0.95] = 1

        IOU_50.extend(IOU_50_bt.cpu().tolist())
        IOU_75.extend(IOU_75_bt.cpu().tolist())
        IOU_95.extend(IOU_95_bt.cpu().tolist())

        ##================= token cls ========================##
        # 获取 token 的标签，忽略 CLS token
        token_label = text_input['attention_mask'][:, 1:].clone()  # [:,1:] for ingoring class token
        token_label[token_label == 0] = -100  # -100 index = padding token
        token_label[token_label == 1] = 0

        # 将 fake_token_pos 对应的 token 标签设置为 1
        for batch_idx in range(len(fake_token_pos)):
            fake_pos_sample = fake_token_pos[batch_idx]
            if fake_pos_sample:
                for pos in fake_pos_sample:
                    token_label[batch_idx, pos] = 1

        # 将 token 分类的输出和标签展平
        logits_tok_reshape = logits_tok.view(-1, 2)
        logits_tok_pred = logits_tok_reshape.argmax(1)
        token_label_reshape = token_label.view(-1)

        # 计算 token 分类的 TP, TN, FP, FN
        TP_all += torch.sum((token_label_reshape == 1) * (logits_tok_pred == 1)).item()
        TN_all += torch.sum((token_label_reshape == 0) * (logits_tok_pred == 0)).item()
        FP_all += torch.sum((token_label_reshape == 0) * (logits_tok_pred == 1)).item()
        FN_all += torch.sum((token_label_reshape == 1) * (logits_tok_pred == 0)).item()

    ##================= real/fake cls ========================##
    # 计算真实/伪造分类的 AUC、准确率和 EER
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    AUC_cls = roc_auc_score(y_true, y_pred)
    ACC_cls = cls_acc_all / cls_nums_all
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    EER_cls = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    ##================= bbox cls ========================##
    # 计算边界框 IoU 的平均值和不同阈值下的准确率
    IOU_score = sum(IOU_pred) / len(IOU_pred)
    IOU_ACC_50 = sum(IOU_50) / len(IOU_50)
    IOU_ACC_75 = sum(IOU_75) / len(IOU_75)
    IOU_ACC_95 = sum(IOU_95) / len(IOU_95)
    # ##================= token cls========================##
    # 计算 token 分类的准确率、精确率、召回率和 F1 分数
    epsilon = 1e-7
    ACC_tok = (TP_all + TN_all) / (TP_all + TN_all + FP_all + FN_all + epsilon)
    Precision_tok = TP_all / (TP_all + FP_all + epsilon)
    Recall_tok = TP_all / (TP_all + FN_all + epsilon)
    F1_tok = 2 * Precision_tok * Recall_tok / (Precision_tok + Recall_tok + epsilon)
    ##================= multi-label cls ========================##
    # 计算多标签分类的 MAP 和其他指标
    MAP = multi_label_meter.value().mean()
    OP, OR, OF1, CP, CR, CF1 = multi_label_meter.overall()

    # 计算每个类别的 F1 分数
    for cls_idx in range(logits_multicls.shape[1]):
        Precision_multicls = TP_all_multicls[cls_idx] / (TP_all_multicls[cls_idx] + FP_all_multicls[cls_idx])
        Recall_multicls = TP_all_multicls[cls_idx] / (TP_all_multicls[cls_idx] + FN_all_multicls[cls_idx])
        F1_multicls[cls_idx] = 2 * Precision_multicls * Recall_multicls / (Precision_multicls + Recall_multicls)

        # 返回所有评估指标
    return AUC_cls, ACC_cls, EER_cls, \
        MAP.item(), OP, OR, OF1, CP, CR, CF1, F1_multicls, \
        IOU_score, IOU_ACC_50, IOU_ACC_75, IOU_ACC_95, \
        ACC_tok, Precision_tok, Recall_tok, F1_tok

@torch.no_grad()
@torch.no_grad()


def main_worker(gpu, args, config):
    # 如果指定了gpu，则将其赋值给args.gpu
    if gpu is not None:
        args.gpu = gpu

    # 初始化分布式环境
    init_dist(args)

    # 获取验证类型（测试类型），并根据文件名决定是使用 'all' 或 'test'
    eval_type = os.path.basename(config['val_file'][0]).split('.')[0]
    if eval_type == 'test':
        eval_type = 'all'

    # 设置日志目录，若不存在则创建
    log_dir = os.path.join(args.output_dir, args.log_num, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)

    # 日志文件路径
    log_file = os.path.join(log_dir, f'shell_{eval_type}.txt')

    # 设置logger
    logger = setlogger(log_file)

    # 如果启用了日志记录，输出日志信息
    if args.log:
        logger.info('******************************')
        logger.info(args)
        logger.info('******************************')
        logger.info(config)
        logger.info('******************************')

    # 设置设备（如cuda或cpu）
    device = torch.device(args.device)

    # 固定种子以保证结果可复现
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### 模型 ####
    # 加载BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(
        "/home/async/data-disk/zxy/deepfake/bert-base-uncased/uncased_L-12_H-768_A-12")

    # 如果启用了日志记录，输出创建模型的提示
    if args.log:
        print(f"Creating HAMMER")

    # 创建HAMMER模型实例
    model = HAMMER(args=args, config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)

    # 将模型移至指定设备（GPU或CPU）
    model = model.to(device)

    # 载入检查点（模型参数）
    checkpoint_dir = f'{args.log_num}/checkpoint_{args.test_epoch}.pth'
    checkpoint = torch.load(checkpoint_dir, map_location='cpu')
    state_dict = checkpoint['model']

    # 调整位置嵌入层的维度，使其与模型相匹配
    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

    # 如果启用了日志记录，打印加载检查点的消息
    if args.log:
        print('load checkpoint from %s' % checkpoint_dir)

    # 加载模型参数到模型中
    msg = model.load_state_dict(state_dict, strict=False)

    # 如果启用了日志记录，打印加载状态
    if args.log:
        print(msg)

    #### 数据集 ####
    # 如果启用了日志记录，打印创建数据集的消息
    if args.log:
        print("Creating dataset")

    # 创建数据集
    _, val_dataset = create_dataset(config)

    # 如果使用分布式训练，创建对应的Sampler
    if args.distributed:
        samplers = create_sampler([val_dataset], [True], args.world_size, args.rank) + [None]
    else:
        samplers = [None]

    # 创建数据加载器
    val_loader = create_loader([val_dataset],
                               samplers,
                               batch_size=[config['batch_size_val']],
                               num_workers=[4],
                               is_trains=[False],
                               collate_fns=[None])[0]

    # 在 main_worker 函数中添加以下代码
    print("\n===== Debug: Validation Data =====")
    sample_data = next(iter(val_loader))
    print("Sample Fake Box (batch):", sample_data['fake_image_box'][:2])  # 打印前两个真实框
    print("===============================\n")

    # 处理分布式训练
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # 如果启用了日志记录，输出评估开始的提示
    if args.log:
        print("Start evaluation")

    # 开始进行评估
    AUC_cls, ACC_cls, EER_cls, \
        MAP, OP, OR, OF1, CP, CR, CF1, F1_multicls, \
        IOU_score, IOU_ACC_50, IOU_ACC_75, IOU_ACC_95, \
        ACC_tok, Precision_tok, Recall_tok, F1_tok = evaluation(args, model_without_ddp, val_loader, tokenizer, device,
                                                                config)

    # 将评估指标格式化为字典
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
                 "F1_FS": "{:.4f}".format(F1_multicls[0] * 100),
                 "F1_FA": "{:.4f}".format(F1_multicls[1] * 100),
                 "F1_TS": "{:.4f}".format(F1_multicls[2] * 100),
                 "F1_TA": "{:.4f}".format(F1_multicls[3] * 100),
                 "IOU_score": "{:.4f}".format(IOU_score * 100),
                 "IOU_ACC_50": "{:.4f}".format(IOU_ACC_50 * 100),
                 "IOU_ACC_75": "{:.4f}".format(IOU_ACC_75 * 100),
                 "IOU_ACC_95": "{:.4f}".format(IOU_ACC_95 * 100),
                 "ACC_tok": "{:.4f}".format(ACC_tok * 100),
                 "Precision_tok": "{:.4f}".format(Precision_tok * 100),
                 "Recall_tok": "{:.4f}".format(Recall_tok * 100),
                 "F1_tok": "{:.4f}".format(F1_tok * 100),
                 }

    # 如果是主进程，记录评估结果
    if utils.is_main_process():
        log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                     'epoch': args.test_epoch,
                     }
        with open(os.path.join(log_dir, f"results_{eval_type}.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')  # 配置文件路径
    parser.add_argument('--checkpoint', default='')  # 检查点路径
    parser.add_argument('--resume', default=False, type=bool)  # 是否恢复训练
    parser.add_argument('--output_dir', default='/mnt/lustre/share/rshao/data/FakeNews/Ours/results')  # 输出目录
    parser.add_argument('--text_encoder',
                        default='/home/async/data-disk/zxy/deepfake/bert-base-uncased/uncased_L-12_H-768_A-12')  # 文本编码器路径
    parser.add_argument('--device', default='cuda')  # 使用的设备，默认为cuda
    parser.add_argument('--seed', default=777, type=int)  # 随机种子
    parser.add_argument('--distributed', default=False, type=bool)  # 是否使用分布式训练
    parser.add_argument('--rank', default=-1, type=int)  # 分布式训练中的节点rank
    parser.add_argument('--world_size', default=1, type=int)  # 分布式训练中的世界大小
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str)  # 分布式训练的URL
    parser.add_argument('--dist-backend', default='nccl', type=str)  # 分布式后端
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                        help='job launcher')  # 作业启动器
    parser.add_argument('--log_num', '-l', type=str)  # 日志编号
    parser.add_argument('--model_save_epoch', type=int, default=5)  # 模型保存的周期
    parser.add_argument('--token_momentum', default=False, action='store_true')  # 是否使用token momentum
    parser.add_argument('--test_epoch', default='best', type=str)  # 测试的epoch

    args = parser.parse_args()

    # 加载配置文件
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # 调用主工作函数
    main_worker(0, args, config)
