import torch
import sys

from OCRUtils import *
import requests
from transformers import LayoutLMv3ImageProcessor, LayoutLMv3ForTokenClassification, LayoutLMv3Tokenizer
# from PIL import Image, ImageDraw, ImageFont
import os



class GwalaOCR:
    def __init__(self,predictor, model_path = None):

        self.model_path = model_path if model_path else './models/FinetuneV3.2'
        self.feature_extractor = LayoutLMv3ImageProcessor(apply_ocr=False, do_normalize=True, do_resize=True)
        self.tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base", max_length=520)
        self.processor = LayoutLMv3Processor(self.feature_extractor, self.tokenizer)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
        self.predictor = predictor

        self.id2label = {
            0: "Store_name_value",
            1: "Store_addr_value",
            2: "Tel_value",
            3: "Date_value",
            4: "Time_value",
            5: "Prod_item_value",
            6: "Prod_quantity_value",
            7: "Prod_price_value",
            8: "Subtotal_value",
            9: "Tax_value",
            10: "Total_value",
            11: "Others"
        }
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.label_list = [v for k, v in self.id2label.items()]

        self.lines_labels = None
        self.lines_words = None
        self.lines_boxes = None
        self.true_predictions_labels = None


    def process_receipt_image(self, image_path, erode = False , dilate = False , dilate_iter = 1):

        # read image
        image = cv2.imread(image_path)
        image = ProcessImage(image, erode=erode, dilate=dilate, dilate_iter=dilate_iter)
        boxes, words, blocks = ExtractTextOCR(image, self.predictor)
        words = [CorrectDigits(word) for word in words]
        indexes_to_remove = [i for i, word in enumerate(words) if len(word) == 0]
        words = [word for i, word in enumerate(words) if i not in indexes_to_remove]
        boxes = [box for i, box in enumerate(boxes) if i not in indexes_to_remove]
        image, encoding = ProcessFeatures(image, boxes, words, self.processor)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        encoding.to(device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encoding)

        true_predictions_labels, labels = ProcessPredictions(outputs, encoding, self.id2label)

        lines_words = []
        lines_labels = []
        lines_boxes = []
        idx_word = 0
        lines = blocks[0]['lines']
        for line in lines:
            words_line = []
            labels_line = []
            boxes_line = []
            for block in line['words']:
                word = block['value']
                if len(word) > 0:
                    labels_line.append(true_predictions_labels[idx_word])
                    words_line.append(word)
                    boxes_line.append(boxes[idx_word])

                    idx_word += 1
            lines_labels.append(labels_line)
            lines_words.append(words_line)
            lines_boxes.append(boxes_line)
        json_data = generate_json(words, boxes, lines_labels)

        self.lines_boxes = lines_boxes
        self.lines_labels = lines_labels
        self.lines_words = lines_words
        self.true_predictions_labels = true_predictions_labels
        self.boxes = boxes
        self.words = words 
        
        return json_data
    
#     # def draw_labels(self , image_path): 

#     #     true_predictions_labels = [label for line_label in self.lines_labels for label in line_label]
#     #     image = cv2.imread(image_path)
#     #     image = Image.fromarray(image)

#     #     draw = ImageDraw.Draw(image)

#     #     font_size = 20
#     #     font = ImageFont.truetype("arial.ttf", font_size)

#     #     label2color = {
#     #         "Date_value": 'red',
#     #         "Others": 'gray',
#     #         "Prod_item_value": 'black',
#     #         "Prod_price_value": 'Orange',
#     #         "Prod_quantity_value": 'green',
#     #         "Store_addr_value": 'Violet',
#     #         "Store_name_value": 'green',
#     #         "Subtotal_value": 'pink',
#     #         "Tax_value": 'Orange',
#     #         "Tel_value": 'red',
#     #         "Time_value": 'red',
#     #         "Total_value": 'blue'
#     #     }

#     #     width, height = image.size
#     #     indx_label = 0
#     #     for prediction, box in zip(self.true_predictions_labels, self.boxes):
#     #         draw.rectangle(box, outline=label2color[prediction], width=3)
#     #         draw.text((box[0] + 10, box[1] - 20), text=str((prediction, str(indx_label))), fill=label2color[prediction], font=font)
#     #         indx_label += 1

#     #     return true_predictions_labels, image
