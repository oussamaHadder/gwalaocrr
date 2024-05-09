import numpy as np


def CorrectDigits(text = '') :
    """
    Corrects mispelled digits in a given text based on a dictionary of personal knowledge.
    """
    DigitCorrection = {'@': '0', 'S': '5', 'U': '4', 'T': '7', 'I': '1', 'L': '1', 'O': '0', 'Q': '0', 'B': '8', 'o': '0', 's': '5', ',' : '.', 'i' : '.' , 'e' : '4'}
    corrected_text = ""
    
    for i in range(len(text)):
        if text[i] in DigitCorrection and len(text)>1:
            if (i > 0 and (text[i-1].isdigit() or text[i-1]=='.' )) or (len(text)>1 and i == 0 and text[i+1].isdigit()):
                corrected_text += DigitCorrection[text[i]]
            elif (i < len(text)-1 and (text[i+1].isdigit() or text[i+1]=='.')):
                corrected_text += DigitCorrection[text[i]]
            else:
                corrected_text += text[i]
        else:
            corrected_text += text[i]

    return corrected_text

def handle_others(lines_labels : list[list]):
    for line in lines_labels :
        while 'Others' in line and 'Store_name_value' in line and 'Store_addr_value' not in line :
            index = line.index('Others')  # Find the index of 'Others' in the line
            line[index] = 'Store_name_value'  # Replace 'Others' with 'Store_name_value'

        while 'Others' in line and 'Store_addr_value' in line and 'Store_name_value' not in line :
            index = line.index('Others')  # Find the index of 'Others' in the line
            line[index] = 'Store_addr_value'  # Replace 'Others' with 'Store_name_value'

        while 'Others' in line and 'Prod_item_value' in line : 
            index = line.index('Others')  # Find the index of 'Others' in the line
            line[index] = 'Prod_item_value'  # Replace 'Others' with 'Store_name_value'
    return lines_labels

            
def handle_store_name(lines_labels: list[list]):
    first_company_line = None
    
    for index, line in enumerate(lines_labels):
        exit_loop = False
        store_name_count = line.count('Store_name_value')
        store_addr_count = line.count('Store_addr_value')
        while 'Store_name_value' in line and exit_loop == False:
            indx_store = line.index('Store_name_value')
            if first_company_line is None :
                first_company_line = index
            elif index - first_company_line >=2:
                line[indx_store] = 'Others'  
            else : 
                exit_loop = True

        while 'Store_addr_value' in line and 'Store_name_value' in line and exit_loop == False:
            indx_addr = line.index('Store_addr_value')
            indx_name = line.index('Store_name_value')
            if store_name_count > store_addr_count and first_company_line == index:
                line[indx_addr] = 'Store_name_value'
            elif store_name_count < store_addr_count and first_company_line == index:
                line[indx_name] = 'Store_addr_value'
            elif store_name_count == store_addr_count and first_company_line != index:
                line[indx_name] = 'Store_addr_value'
            else :
                exit_loop = True

    return lines_labels


def handle_store_addr(lines_labels : list[list]):
    first_addr_line = None
    exit_loop = False
    for index, line in enumerate(lines_labels):
        while 'Store_addr_value' in line and 'Store_name_value' not in line and exit_loop == False:
            indx_store = line.index('Store_addr_value')
            if first_addr_line is None :
                first_addr_line = index
            elif index - first_addr_line >=2:
                line[indx_store] = 'Others'  
            else : 
                exit_loop = True
    return lines_labels




