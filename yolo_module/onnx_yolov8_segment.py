import argparse
import cv2
import math
import numpy as np
from numpy import array
import onnxruntime as rt
import os
import time
from tqdm import tqdm


class LetterBox:
    """
    调整图像大小和填充
    """

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32):
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride

    def __call__(self, labels=None, image=None):
        if labels is None:
            labels = {}
        img = labels.get('img') if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop('rect_shape', self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        # only scale down, do not scale up (for better val mAP)
        if not self.scaleup:
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
                 new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(
                dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / \
                    shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if labels.get('ratio_pad'):
            labels['ratio_pad'] = (labels['ratio_pad'],
                                   (dw, dh))  # for evaluation

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))  # add border

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels['img'] = img
            labels['resized_shape'] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels"""
        labels['instances'].convert_bbox(format='xyxy')
        labels['instances'].denormalize(*labels['img'].shape[:2][::-1])
        labels['instances'].scale(*ratio)
        labels['instances'].add_padding(padw, padh)
        return labels


class non_max_suppression:
    """
    非极大值抑制
    """

    def __init__(self,
                 conf_thres=0.25,
                 iou_thres=0.45,
                 classes=None,
                 agnostic=False,
                 multi_label=False,
                 labels=(),
                 max_det=300,
                 nc=0,  # number of classes (optional)
                 max_time_img=0.05,
                 max_nms=30000,
                 max_wh=7680, ):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic = agnostic
        self.multi_label = multi_label
        self.labels = labels
        self.max_det = max_det
        self.nc = nc
        self.max_time_img = max_time_img
        self.max_nms = max_nms
        self.max_wh = max_wh

    def __call__(self, prediction):
        # assert 0 <= self.conf_thres <= 1, f'Invalid Confidence threshold {self.conf_thres}, valid values are between 0.0 and 1.0'
        # assert 0 <= self.iou_thres <= 1, f'Invalid IoU {self.iou_thres}, valid values are between 0.0 and 1.0'
        # if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        #     prediction = prediction[0]  # select only inference output
        self.prediction = prediction
        bs = self.prediction.shape[0]  # batch size
        nc = self.nc or (self.prediction.shape[1] - 4)  # number of classes
        nm = self.prediction.shape[1] - nc - 4
        mi = 4 + nc  # mask start index
        xc = self.prediction[:, 4:mi].max(1) > self.conf_thres  # candidates

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        time_limit = 0.5 + self.max_time_img * bs  # seconds to quit after
        self.multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

        t = time.time()
        output = [np.zeros((0, 6 + nm), dtype=np.uint8)] * bs
        for xi, x in enumerate(self.prediction):  # image index, image inference
            x = x.transpose(1, 0)[xc[xi]]  # confidence

            box, cls, mask = x[:, :4], x[:, 4:nc + 4], x[:, -nm:]
            # center_x, center_y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(box)
            if self.multi_label:
                i, j = (cls > self.conf_thres).nonzero(as_tuple=False).T
                x = np.concatenate(
                    (box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf, j = cls.max(1).reshape(cls.shape[0], 1), cls.argmax(1).reshape(cls.shape[0], 1)
                x = np.concatenate((box, conf, j.astype(np.float64), mask), 1)[
                    conf.reshape(-1) > self.conf_thres]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            x = x[(-x[:, 4]).argsort()[:self.max_nms]]

            # Batched NMS
            c = x[:, 5:6] * (0 if self.agnostic else self.max_wh)  # classes
            # boxes (offset by class), scores
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = self.numpy_nms(boxes, scores, self.iou_thres)  # NMS
            i = i[:self.max_det]  # limit detections

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                print(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
                break  # time limit exceeded
        return output

    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y

    def box_area(self, boxes: array):
        """
        :param boxes: [N, 4]
        :return: [N]
        """
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    def box_iou(self, box1: array, box2: array):
        """
        :param box1: [N, 4]
        :param box2: [M, 4]
        :return: [N, M]
        """
        area1 = self.box_area(box1)  # N
        area2 = self.box_area(box2)  # M
        # broadcasting, 两个数组各维度大小 从后往前对比一致， 或者 有一维度值为1；
        lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
        rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])
        wh = rb - lt
        wh = np.maximum(0, wh)  # [N, M, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]
        iou = inter / (area1[:, np.newaxis] + area2 - inter)
        return iou  # NxM

    def numpy_nms(self, boxes: array, scores: array, iou_threshold: float):
        idxs = scores.argsort()  # 按分数 降序排列的索引 [N]
        keep = []
        while idxs.size > 0:  # 统计数组中元素的个数
            max_score_index = idxs[-1]
            max_score_box = boxes[max_score_index][None, :]
            keep.append(max_score_index)

            if idxs.size == 1:
                break
            idxs = idxs[:-1]  # 将得分最大框 从索引中删除； 剩余索引对应的框 和 得分最大框 计算IoU；
            other_boxes = boxes[idxs]  # [?, 4]
            ious = self.box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
            idxs = idxs[ious[0] <= iou_threshold]

        keep = np.array(keep)
        return keep


class process_mask:
    """
    上采样还原掩码大小
    """

    def __init__(self, protos, masks_in, bboxes, shape, upsample=False) -> None:
        self.protos = protos
        self.masks_in = masks_in
        self.bboxes = bboxes
        self.shape = shape
        self.upsample = upsample

    def __call__(self, *args, **kwds):
        c, mh, mw = self.protos.shape  # CHW
        ih, iw = self.shape
        # print(self.masks_in.shape)
        # print(self.protos.shape)
        sigmoid_masks = []
        for i in range(self.masks_in.shape[0]):
            sigmoid_masks.append(self.sigmoid_function(
                (self.masks_in @ self.protos.astype(np.float64).reshape(c, -1))[i]))

        masks = np.array(sigmoid_masks).reshape(-1, mh, mw)  # CHW

        downsampled_bboxes = self.bboxes.copy()
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 3] *= mh / ih
        downsampled_bboxes[:, 1] *= mh / ih

        masks = self.crop_mask(masks, downsampled_bboxes)  # CHW
        if self.upsample:
            masks = (masks * 255).astype(np.uint8)
            masks = masks.transpose(1, 2, 0)
            masks = cv2.resize(masks, kwds['size'])
            masks[masks <= (255 * 0.5)] = 0.0
            masks[masks > (255 * 0.5)] = 1.0
        return masks

    def sigmoid_function(self, z):
        fz = []
        for num in z:
            fz.append(1 / (1 + math.exp(-num)))
        return np.array(fz)

    def crop_mask(self, masks, boxes):
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.array_split(
            boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
        r = np.array(range(w), dtype=np.float64).reshape(
            1, 1, -1)  # rows shape(1,w,1)
        c = np.array(range(h), dtype=np.float64).reshape(
            1, -1, 1)  # cols shape(h,1,1)

        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


class Segmentation_inference:
    def __init__(self, model_path, device) -> None:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device != 'cpu' else [
            'CPUExecutionProvider']
        self.sess = rt.InferenceSession(model_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.out_name = [output.name for output in self.sess.get_outputs()]

    def __call__(self, *args, **kwds):
        self.im = cv2.imdecode(np.fromfile(kwds['path'], dtype=np.int8), 1)
        # 前处理
        time1 = time.time()
        self.im0 = self.img_pretreatment(
            im=self.im, size=[kwds['imgsz'], kwds['imgsz']])
        time2 = time.time()
        # 推理
        time3 = time.time()
        self.preds = self.sess.run(
            self.out_name, {self.input_name: [self.im0]})
        time4 = time.time()
        # 后处理
        time5 = time.time()
        self.masks, self.box_list, self.scores_list = self.img_reprocessing(
            preds=self.preds, size=(kwds['imgsz'], kwds['imgsz']), conf_thres=kwds['conf_thres'],
            iou_thres=kwds['iou_thres'])
        if type(self.masks) == type(None):
            cv2.imwrite(kwds['save_path'], self.im)
            return None
        time6 = time.time()
        # 打印处理时间及保存
        if kwds['show_time']:
            print(f'\npretreatment——time:{(time2 - time1) * 1000}ms')
            print(f'inference——time:{(time4 - time3) * 1000}ms')
            print(f'reprocessing——time:{(time6 - time5) * 1000}ms')
            print('-----------------------------')
        if kwds['save_masks'] and kwds['save_box']:  # 同时保存矩形框和掩码
            masks_write = self.masks_write(masks=np.array(
                self.masks), im_gpu=self.im0, im_shape=self.im.shape, im=self.im.copy())
            # ,self.box_list,self.scores_list))
            cv2.imwrite(kwds['save_path'], self.box_write(masks_write.copy()))
        elif kwds['save_masks']:  # 只保存掩码
            cv2.imwrite(kwds['save_path'], self.masks_write(masks=np.array(
                self.masks), im_gpu=self.im0, im_shape=self.im.shape, im=self.im.copy()))
        elif kwds['save_box']:  # 只保存矩形框
            # ,self.box_list,self.scores_list))
            cv2.imwrite(kwds['save_path'], self.box_write(self.im.copy()))
        return True

    def img_pretreatment(self, im, size=[640, 640], auto=False, stride=32):
        """
        前处理
        """
        im1 = LetterBox(size, auto, stride=stride)(image=im)
        im2 = im1.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im3 = np.ascontiguousarray(im2)  # contiguous
        im4 = im3.astype(np.float32) / 255.0
        return im4

    def img_reprocessing(self, preds, size, conf_thres=0.25, iou_thres=0.7):
        """
        后处理
        """
        nc = len(eval(self.sess.get_modelmeta().custom_metadata_map['names']))
        p = non_max_suppression(
            conf_thres=conf_thres, iou_thres=iou_thres, nc=nc)(preds[0])
        if len(p[0]) == 0:
            return None
        proto = self.preds[1][-1] if len(self.preds[1]) == 3 else self.preds[1]
        masks_list = []
        pred_list = []
        scores_list = []
        for i, pred in enumerate(p):
            masks = process_mask(proto[i], pred[:, 6:], pred[:, :4], np.array(
                self.im0).shape[-2:], upsample=True)(size=size)  # HWC
            # cv2.imshow('main',masks[:,:,1])
            # cv2.waitKey(-1)
            pred[:, :4] = self.scale_boxes(
                self.im0.shape[1:], pred[:, :4], self.im.shape)
            masks_list.append(masks)
            pred_list.append(pred[:, :4])
            scores_list.append(pred[:, 4])
        return masks_list, pred_list, scores_list

    def box_write(self, image):  # ,box_list,scores_list
        """
        画矩形框
        """
        box_write = image.copy()
        for box, scores in zip(self.box_list[0], self.scores_list[0]):
            cx = np.mean([int(box[0]), int(box[2])])
            cy = np.mean([int(box[1]), int(box[3])])
            box_write = cv2.rectangle(box_write, (int(box[0]), int(box[1])), (int(
                box[2]), int(box[3])), color=(0, 0, 255), thickness=2)
            mess = '%.2f' % scores
            h, w = image.shape[:2]
            cv2.putText(box_write, mess, (int(cx), int(cy)),
                        0, 1e-3 * h, (0, 0, 255), 1 // 2)
        return box_write

    def masks_write(self, masks, im_gpu, im_shape, im, colors=[[[[0.21961, 0.21961, 1.00000]]]], alpha=0.5,
                    retina_masks=False):
        """
        保存结果
        """
        cv2.imshow('main',masks[0]*255)
        cv2.waitKey(-1)

        if len(masks.shape) == 3:
            masks = masks.reshape(
                masks.shape[0], masks.shape[1], masks.shape[2], 1)
        else:
            huaban = np.zeros((masks.shape[1], masks.shape[2]))
            masks.astype(np.bool_)
            for i in range(masks.shape[3]):
                huaban[masks[0][:, :, i] != 0] = 1
            masks = huaban.reshape(
                masks.shape[0], masks.shape[1], masks.shape[2], 1)

        masks_color = masks * colors * alpha  # shape(n,h,w,3)

        inv_alph_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
        mcs = (masks_color * inv_alph_masks).sum(0) * \
              2  # mask color summand shape(n,h,w,3)

        im_gpu = im_gpu[::-1, :, :]
        im_gpu = im_gpu.transpose(1, 2, 0)  # shape(h,w,3)
        im_gpu = im_gpu * inv_alph_masks[-1] + mcs
        im_mask = (im_gpu * 255)
        im_mask_np = im_mask
        im[:] = im_mask_np if retina_masks else self.scale_image(
            im_gpu.shape, im_mask_np, im_shape)
        for box, scores in zip(self.box_list[0], self.scores_list[0]):  # 画置信度
            cx = np.mean([int(box[0]), int(box[2])])
            cy = np.mean([int(box[1]), int(box[3])])
            mess = '%.2f' % scores
            h, w = im.shape[:2]
            cv2.putText(im, mess, (int(cx), int(cy)), 0,
                        1e-3 * h, (0, 0, 255), 1 // 2)
        return im

    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None):
        """
        缩放矩形框
        """
        if ratio_pad is None:  # calculate from img0_shape
            # gain  = old / new
            gain = min(img1_shape[0] / img0_shape[0],
                       img1_shape[1] / img0_shape[1])
            pad = (img1_shape[1] - img0_shape[1] * gain) / \
                  2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]
        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
        boxes[..., :4] /= gain

        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(
            0, img0_shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(
            0, img0_shape[0])  # y1, y2

        return boxes

    def scale_image(self, im1_shape, masks, im0_shape, ratio_pad=None):
        """
        保存结果时缩放图像
        """
        # Rescale coordinates (xyxy) from im1_shape to im0_shape
        if ratio_pad is None:  # calculate from im0_shape
            # gain  = old / new
            gain = min(im1_shape[0] / im0_shape[0],
                       im1_shape[1] / im0_shape[1])
            pad = (im1_shape[1] - im0_shape[1] * gain) / \
                  2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]
        top, left = int(pad[1]), int(pad[0])  # y, x
        bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

        if len(masks.shape) < 2:
            raise ValueError(
                f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))

        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str,
                        default=r'E:\OpensourceProject\ultralytics\input\istockphoto-609817054-612x612.jpg', help='Image Path or Image dir path')
    parser.add_argument('--save_path', type=str,
                        default=r'E:\OpensourceProject\ultralytics\runs\segment', help='result Image save Path or dir path')
    parser.add_argument('--weight', type=str,
                        default='E:\pythonProject\Audio_py\models\yolov8s-seg.onnx', help='weights path')
    parser.add_argument('--imgsz', type=int,
                        default=640, help='Input Image Size')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Hardware devices')
    parser.add_argument('--save_masks', action='store_true',
                        default=True, help='save the mask?')
    parser.add_argument('--save_box', action='store_true',
                        default=True, help='save the box?')
    parser.add_argument('--show_time', action='store_true',
                        default=True, help='Output processing time')
    parser.add_argument('--conf_thres', type=float,
                        default=0.25, help='Confidence level threshold')
    parser.add_argument('--iou_thres', type=float,
                        default=0.7, help='IOU threshold')
    return parser.parse_args()


def main(opt):
    calculate = Segmentation_inference(opt.weight, opt.device)
    if not os.path.isdir(opt.path):  # 检测单张图片
        try:
            assert type(calculate(path=opt.path,
                                  save_path=opt.save_path,
                                  save_masks=opt.save_masks,
                                  save_box=opt.save_box,
                                  show_time=opt.show_time,
                                  imgsz=opt.imgsz,
                                  conf_thres=opt.conf_thres,
                                  iou_thres=opt.iou_thres)) != type(None), '没有检测到目标,请适当调低阈值'
        except BaseException as f:
            print(f)
    else:  # 检测一个文件夹中的图片
        assert os.path.isdir(
            opt.save_path), '预测路径为文件夹目录，则保存路径也应该指定为已存在的某个文件夹目录'
        for img_path in tqdm(os.listdir(opt.path)):
            try:
                assert type(calculate(path=os.path.join(opt.path, img_path),
                                      save_path=os.path.join(
                                          opt.save_path, img_path),
                                      save_masks=opt.save_masks,
                                      save_box=opt.save_box,
                                      show_time=opt.show_time,
                                      imgsz=opt.imgsz,
                                      conf_thres=opt.conf_thres,
                                      iou_thres=opt.iou_thres)) != type(None), '没有检测到目标,请适当调低阈值'
            except BaseException as f:
                print(f)


if '__main__' == __name__:
    opt = parse_opt()
    main(opt)


