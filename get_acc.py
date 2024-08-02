import os
import re

import cv2
import numpy as np


def cal_instance(gt_path):
    return len(open(gt_path, encoding='UTF-8').readlines())


def read_gt(gt_path):
    gt_dict = {}
    for line in open(gt_path, encoding='UTF-8'):
        img_path, cls = line.strip().split()
        gt_dict[img_path] = cls

    return gt_dict


def read_result(result_path):
    result_dict = {}
    for line in open(result_path):
        img_path, cls = line.strip().split()
        result_dict[img_path] = cls

    return result_dict


def main():
    gt_path = R"D:\pyProject\donut\dataset\pages\test.txt"
    result_path = R"D:\pyProject\donut\dataset\pages\pred_result.txt"

    gt_dict = read_gt(gt_path)
    result_dict = read_result(result_path)

    f1_matrix = np.zeros((2, 2), np.int64)

    for img_path, cls in gt_dict.items():
        if cls == '-1':
            pred = result_dict[img_path]
            f1_matrix[int(cls != '-1')][int(pred != '不知道')] += 1
        else:
            pred = re.findall(r'\d+', result_dict[img_path])
            f1_matrix[int(cls != '-1')][int(cls in pred)] += 1
            print(img_path, cls, result_dict[img_path])

    print(f"Acc: {(f1_matrix[0][0] + f1_matrix[1][1]) / f1_matrix.sum():.2%}")
    print(f"Pos Acc: {(f1_matrix[1][1]) / f1_matrix[1].sum():.2%}")
    print(f"Neg Acc: {(f1_matrix[0][0]) / f1_matrix[0].sum():.2%}")
    print(f"Number of pos: {f1_matrix[1].sum()}")
    print(f"Number of neg: {f1_matrix[0].sum()}")
    print(f1_matrix)
    print(f1_matrix[1].sum())
    print(f1_matrix.sum())


def main2():
    gt_dict = {'train': R"D:\pyProject\donut\dataset\pages\train.txt",
               'valid': R"D:\pyProject\donut\dataset\pages\validation.txt",
               'test': R"D:\pyProject\donut\dataset\pages\test.txt"}

    for key, val in gt_dict.items():
        print(f'Number of {key}: {cal_instance(val)}')


def display_result():
    img_root = R"D:\pyProject\donut\dataset\pages\images"
    output_root = R"D:\pyProject\donut\dataset\pages\result"
    gt_path = R"D:\pyProject\donut\dataset\pages\test.txt"
    result_path = R"D:\pyProject\donut\dataset\pages\pred_result.txt"

    gt_dict = read_gt(gt_path)
    result_dict = read_result(result_path)

    cv2.namedWindow("Donut", cv2.WINDOW_NORMAL)
    img_paths = [img_path for img_path in list(gt_dict.keys()) if gt_dict[img_path] != '-1']

    idx = 0
    print(len(img_paths))
    while idx < len(img_paths):
        print(idx)
        img_path, cls = img_paths[idx], gt_dict[img_paths[idx]]
        if cls != '-1':
            pred = re.findall(r'\d+', result_dict[img_path])

            msgs = [f"{img_path}", f"GT: {cls}", f"Pred: {pred}"]
            img = cv2.imread(os.path.join(img_root, img_path))

            color = (255, 0, 0) if cls in pred else (0, 0, 255)
            for y, msg in enumerate(msgs):
                cv2.putText(img, msg, (0, 100 + y * 100), cv2.FONT_HERSHEY_COMPLEX, 2.5, (0, 0, 0), 10)
                cv2.putText(img, msg, (0, 100 + y * 100), cv2.FONT_HERSHEY_COMPLEX, 2.5, color, 6)

            cv2.imshow('Donut', img)
            cv2.imwrite(os.path.join(output_root, img_path.replace('/', '_')), img)
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                break
            elif key == ord('a'):
                idx = max(0, idx - 1)
                continue
        else:
            print(gt_dict[img_path])

        idx += 1


if __name__ == '__main__':
    # main()
    # main2()
    display_result()
