import os
import argparse
import mimetypes
import cv2
from src.face_recognizer import Recognizer
from src.utils import *
import config
import sklearn
from sklearn import datasets


parser = argparse.ArgumentParser(description='Face Recognition')
parser.add_argument("--input", default="io\CALFW_test\Aaron_Eckhart_0002.jpg", type=str, help="input image path")
parser.add_argument("--output", default="io/output", type=str, help="output dir path")
parser.add_argument("--update", default=False, action="store_true", help="whether perform update the dataset")
parser.add_argument("--origin-size", default=True, action="store_true", help='Whether to use origin image size to evaluate')
parser.add_argument("--tta", default=False, action="store_true", help="whether test time augmentation")
parser.add_argument("--show", default=False, action="store_true", help="show result")
parser.add_argument("--save", default=True, action="store_true", help="whether to save")
parser.add_argument("--facebank", default="face_bank_CALFW.npy", action="store_true", help="face Bank address")
args = parser.parse_args()





if __name__ == '__main__':
    mimetypes.init()
    recognizer = Recognizer(model_name=config.model_name)
 
    # face bank
    if args.update:
        targets, names = prepare_face_bank(recognizer, tta=args.tta)
        print('face bank updated')
    else:
        targets, names = load_face_bank(args.facebank)
        print('face bank loaded')

    if args.save:
        os.makedirs(args.output, exist_ok=True)
        output_file_path = os.path.join(args.output, os.path.basename(args.input))

    mimestart = mimetypes.guess_type(args.input)[0]
    

    if mimestart == None:
        print('input not found!')
        exit()
    else:
        mimestart = mimestart.split('/')[0]

    if mimestart == 'image':
        image = cv2.imread(args.input)
        if not args.origin_size:
            image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results, bboxes = recognizer.recognize(image_rgb, targets, args.tta)

        for idx, bbox in enumerate(bboxes):
            if results[idx] != -1:
                name = names[results[idx] + 1]
            else:
                name = 'Unknown'
            image = draw_box_name(image, bbox.astype("int"), name)

        if args.show:
            cv2.imshow('face Capture', image)
            cv2.waitKey()
        if args.save:
            cv2.imwrite(output_file_path, image)

 
    print('finish!')
