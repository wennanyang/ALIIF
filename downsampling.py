import cv2
def main(path):
    
    starth, startw = 10, 20
    h, w = 112, 112
    img = cv2.imread(path)
    print(img.shape)
    img_crop = img[starth : starth + h, startw : startw + w, :]
    img_output = cv2.resize(img_crop, (48, 48), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("lq_img.png", img_output)
if __name__ == '__main__':
    # monarchx4 [25 : 105, 65 : 145, :]
    # zebrax4 []
    # main("/home/ywn/graduate/DATASET/benchmark/Set14/LR_bicubic/X4/zebrax4.png")
    path = "/home/ywn/graduate/ALIIF/lq_img.png"
    img = cv2.imread(path)
    output = cv2.resize(img, (192, 192), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("bicubic_opencv.png", output)