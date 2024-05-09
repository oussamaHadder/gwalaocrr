from PIL import Image
import cv2
import numpy as np 
from transformers import LayoutLMv3Processor
import torch
import re
import json
from StringUtils import *


def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]



    


def denormalize_box(box,width,height):
    return [
        int((box[0] * width)/1000),
        int((box[1] * height)/1000),
        int((box[2] * width)/1000),
        int((box[3] * height)/1000),
    ]



def ProcessImage(image, resize= False, to_PIL = False, kernel_size = (2,2), dilate = True, erode = True, erode_iter = 2, dilate_iter = 1) :
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.adaptiveThreshold(image, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 61, 12)
    image = cv2.bitwise_not(image)
    kernel = np.ones(kernel_size,np.uint8)

    if erode : 
        image = cv2.erode(image,kernel, iterations = erode_iter)
        
    if dilate : 
        image = cv2.dilate(image,kernel, iterations = dilate_iter )

    image = cv2.bitwise_not(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if to_PIL : 
        image = Image.fromarray(image)
        image = image.convert('RGB')

    if resize and to_PIL :
        image = image.resize((1000,1600))
    
    if resize and not to_PIL:
        raise ValueError('if resize is set to True, you should have PIL image in input')
        
    return image



def ExtractTextOCR(image,predictor):


    result = predictor([image])
    export = result.export()
    # Flatten the export
    page_words = [[word for block in page['blocks'] for line in block['lines'] for word in line['words']] for page in export['pages']]
    page_dims = [page['dimensions'] for page in export['pages']]
    # Get the coords in [xmin, ymin, xmax, ymax]
    words_abs_coords = [
        [[int(round(word['geometry'][0][0] * dims[1])), int(round(word['geometry'][0][1] * dims[0])), int(round(word['geometry'][1][0] * dims[1])), int(round(word['geometry'][1][1] * dims[0]))] for word in words]
        for words, dims in zip(page_words, page_dims)]

    boxes = words_abs_coords[0]
    words = [word['value'] for word in page_words[0]]

    blocks = export['pages'][0]['blocks']


    return boxes, words, blocks

inputs = {'return_tensors' : 'pt', 'padding' : 'max_length' , 'truncation' : True, 'return_attention_mask' : True, 'return_token_type_ids' : True}

def ProcessFeatures(image : np.ndarray ,boxes : list[list[int]],words : list[str],processor : LayoutLMv3Processor, inputs = inputs ):
    image = Image.fromarray(image)

    width, height = image.size
    nboxes = [normalize_box(box,width,height) for box in boxes]
    labels = [0]*len(boxes)
    encoding = processor(image, words, boxes=nboxes, word_labels= labels, **inputs)

    return image,encoding

def ProcessPredictions(outputs, encoding, id2label):
    logits = outputs.logits
    labels = encoding.labels
    predictions = logits.argmax(-1).squeeze().tolist()
    active_loss = labels != -100
    active_logits = logits[active_loss]
    active_labels = labels[active_loss]

    true_predictions = torch.argmax(active_logits, dim=1).tolist()
    true_predictions_labels = [id2label[pred] for pred in true_predictions]

    return true_predictions_labels, active_labels





def euclidean_distance(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def find_nearest_price(prod_item_value, prod_price_values):
    min_distance = float('inf')
    nearest_price = None
    for (x2,y2, price) in (prod_price_values):
        distance = euclidean_distance(prod_item_value[0], prod_item_value[1], x2, y2)
        if distance < min_distance:
            min_distance = distance
            nearest_price = price
            nearest_coord = x2,y2


    return nearest_coord,nearest_price
def generate_json(words : list[str], boxes : list[list[int]], lines_labels : list):
    
    lines_labels_copy = [line for line in lines_labels]
    # handle some exception cases for 'Others' label
    lines_labels_copy = handle_others(lines_labels_copy)
    # handle some exception cases for 'Store_name_value' label
    lines_labels_copy = handle_store_name(lines_labels_copy)
    # handle some exception cases for 'Store_addr_value' label 
    lines_labels_copy = handle_store_addr(lines_labels_copy)

    prod_price_values = []
    labels_dict = {'Store_name_value' : [],
                   'Store_addr_value' : [] ,
                   'Tel_value': None,
                   'Date_value': None,
                   'Time_value' : None,
                   'Total_value' : [] ,
                   'Tax_value': None,
                   'Subtotal_value' : [],
                   'items' : []
                    }
    
    indx_word = 0
    prod_item_lines = []
    line_contains_item = False
    dict_item = None
    quantity_value = None
    for indx,line in enumerate(lines_labels_copy) :
        prod_item_line = [] # a list that will contains items in a specific line
        prod_price = None

        if 'Prod_quantity_value' in line or 'Prod_item_value' in line : # if quantity value or prod item value is found, then i initiate a dictionary that will help to store all informations related to the prod item
            dict_item = {'quantity' : None,'item_values' : None, 'price' : None } # initiliaze it with None walues


        if 'Prod_item_value' in line and 'Prod_price_value' in line : 
            print('Same line price and product')


        # line_contains_item will tell us if a prod item is found in a line or not , so that wa can handle list of prod items in a specific line later
        if 'Prod_item_value' not in line :
            line_contains_item = False
        
        for label in line:
            # store coordinates of current word
            x1, y1, x2, y2 = boxes[indx_word]
            # handle the prod item values
            if label == "Prod_item_value":
                prod_item_line.append((x2,y2,words[indx_word])) # I store the product item values
                line_contains_item = True
            if label == 'Prod_quantity_value' : 
                quantity_value = float(re.search(r'\d', words[indx_word]).group()) # i will assign a certain quantity value to the first product item found

            elif label == "Prod_price_value":
                try :
                    prod_price_values.append((x2, y2, float(re.search(r'\d+\.\d+', words[indx_word]).group()))) # I store the product prices values
                    prod_price = (x2,y2,float(re.search(r'\d+\.\d+', words[indx_word]).group()))
                except : 
                    prod_price_values.append((x2, y2, (re.search(r'\d+\.\d+', words[indx_word]).group()))) # I store the product prices values
                    prod_price = (x2,y2,(re.search(r'\d+\.\d+', words[indx_word]).group()))


            elif label in ['Store_name_value', 'Store_addr_value']:
                labels_dict[label].append(words[indx_word]) # I store the store name/address values in a list
            elif label in ['Subtotal_value', 'Total_value']:
                value = re.search(r'\d+\.\d+', words[indx_word]).group()
                try:
                    labels_dict[label].append(float(value)) # sometimes total and subtotal can be found multiple times, so i make sure to store them in a list, and i take the float version of them
                except ValueError:
                    labels_dict[label].append(words[indx_word]) # if the float version has some errors, i just take the string value of it 
            elif label in ["Tel_value", "Date_value", "Time_value", "Tax_value"]:
                # I store here the values for date, tel, time and tax, which are the easiest ones to handle, as in 90% of cases, they come one time in a prediction
                if labels_dict[label] is None:
                    # if it's tax value i take the float version if it's possible
                    if label == 'Tax_value':
                        tax_value = re.search(r'\d+\.\d+', words[indx_word]).group()
                        try:
                            labels_dict[label] = float(tax_value)
                        except ValueError:
                            labels_dict[label] = tax_value

                    else : 
                        labels_dict[label] = words[indx_word]

            #  to avoid anything else forgotten ('Others' label)
            else :
                indx_word += 1
                continue
            
            indx_word += 1

        # if I have a line that contains item value, I assign the list of all found items to the dictionary 
        if line_contains_item :
            dict_item['item_values'] = prod_item_line # TOFIX 
            dict_item['price'] = prod_price
            if quantity_value :
                dict_item['quantity'] = quantity_value
            prod_item_lines.append(dict_item.copy())
            quantity_value = None

    # to assign a prod price value to an prod item value, i will take the closest price (in terms of coordinates) for each item
    for item_id,prod_dict in enumerate(prod_item_lines):
        prod_items_line = prod_dict['item_values']

        # to assign a product price to the correct prod item, i take for each prod item the closest product price (euclidean distance)
        prod_item_value = prod_items_line[-1]

        if prod_dict['price'] is not None :
            price = prod_dict['price'][-1]

        else: 
            (x2,y2),price = find_nearest_price(prod_item_value, prod_price_values)

        try : 
            if prod_dict['quantity']:
                # I assign the values to labels, and concatenate the prod items found in a specific line 
                prod_data = {
                    "item_id": item_id,
                    "prod_item_value": " ".join(name[2] for name in prod_items_line),
                    "prod_quantity" : prod_dict['quantity'],
                    "prod_price_value" : float(price)
                }
            else :
                prod_data = {
                    "item_id": item_id,
                    "prod_item_value": " ".join(name[2] for name in prod_items_line),
                    "prod_quantity" : 1,
                    "prod_price_value" : float(price)
                }

        except : 
            prod_data = {
                "item_id": item_id,
                "prod_item_value": " ".join(name[2] for name in prod_items_line),
                "prod_quantity" : prod_dict['quantity'],
                "prod_price_value" : price

            }

        # append the "items" key's value
        labels_dict["items"].append(prod_data) 


    # let's now find the unique words (without being sorted) of each name/address store values to concatenate them 
    indexes_names = np.unique(labels_dict['Store_name_value'], return_index=True)[1]
    unique_names = [labels_dict['Store_name_value'][index] for index in sorted(indexes_names)]

    indexes_addr = np.unique(labels_dict['Store_addr_value'], return_index=True)[1]
    unique_addr = [labels_dict['Store_addr_value'][index] for index in sorted(indexes_addr)]

    # concatenation
    labels_dict['Store_name_value'] = " ".join(name for name in unique_names)
    labels_dict['Store_addr_value'] = " ".join(addr for addr in unique_addr)

    # i take max of total values detected (sometimes it detects subtotal as total)

    if labels_dict['Total_value']:
        if len(labels_dict['Total_value']) != 0:
            labels_dict['Total_value'] = max(labels_dict['Total_value'])
    else:
        labels_dict['Total_value'] = None

    # Calculate Subtotal_value
    if labels_dict['Subtotal_value']:
        if len(labels_dict['Subtotal_value']) != 0 and labels_dict['Total_value'] is not None:
            # here i filter out subtotal values that are equal to total value, and eventually subtotal values that are too low (often confused with the tax value), i take a condition limit of 
            # total/subtotal < 2 (normally it's 1.2 as a limit but to avoid some exceptions i choose 2)
            labels_dict['Subtotal_value'] = [subtotal for subtotal in labels_dict['Subtotal_value'] if (subtotal != labels_dict['Total_value'] and labels_dict['Total_value']/subtotal <2)] 

            if len(labels_dict['Subtotal_value']) > 0:
                # i wanted to take the closest value to total value but it's not a good idea, so rather after filetering i choose the most far subtotal value
                labels_dict['Subtotal_value'] = max(labels_dict['Subtotal_value'], key=lambda x: abs(x - labels_dict['Total_value']))

        else:
            labels_dict['Subtotal_value'] = None

    if labels_dict['Tax_value'] is None:
        # if tax value is not found, then i will take total-subtotal as the tax value 
        if labels_dict['Subtotal_value'] is not None and labels_dict['Total_value'] is not None:
            try:
                labels_dict['Tax_value'] = labels_dict['Total_value'] - labels_dict['Subtotal_value']
            except:
                labels_dict['Tax_value'] = None
        else:
            labels_dict['Tax_value'] = None

    # a last check, if total value is not found and subtotal value is found, then it's likely that the subtotal value found is the real total value
    if labels_dict['Total_value'] is None and labels_dict['Subtotal_value'] is not None :
        labels_dict['Total_value'] = labels_dict['Subtotal_value']

    return json.dumps(labels_dict, indent=4)
