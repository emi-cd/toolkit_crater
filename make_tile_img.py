import sys, os
import glob
import cv2

def split(img_path, out_path, r=608, over=144):
    img = cv2.imread(img_path)
    file_name = os.path.splitext(os.path.basename(img_path))[0]
    height, width, channels = img.shape

    for y in range(0, height, r-over):
        for x in range(0, width, r-over):
            clp = img[y:y+r, x:x+r]
            cv2.imwrite('{}/{}_{}_{}.JPG'.format(out_path, file_name, x, y), clp)


def main(argv):
    for img_path in glob.glob(argv[0] + '/*.png'):
        split(img_path, argv[1])


if __name__ == "__main__":
    main(sys.argv[1:])