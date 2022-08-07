import cv2
import numpy as np
import time
import csv
import pytesseract
import os
import threading

## Init
global thread_kill_flags
input_folder = "pdf_img/"
output_folder = "output_img/"
out_csv = "table_1.csv"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def img_preprocessing(img):
    time_str = str(int(round(time.time() * 1000)))
    w_filename = time_str + ".jpg"
    resizeimg = cv2.resize(img, None, fx=1.5, fy=1.6)
    kernel = np.ones((3, 3), np.uint8)
    erodeimg = cv2.erode(resizeimg, kernel, iterations=2)
    dilateimg = cv2.dilate(erodeimg, kernel, iterations=2)
    dilateimg = cv2.cvtColor(dilateimg, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.threshold(cv2.medianBlur(dilateimg, 3), 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imwrite("output_img1/" + w_filename, blur_img)
    return blur_img


def caculate_time_difference(start_milliseconds, end_milliseconds, filename):
    if filename == 'total':
        diff_milliseconds = int(end_milliseconds) - int(start_milliseconds)
        seconds=(diff_milliseconds / 1000) % 60
        minutes=(diff_milliseconds/(1000*60))%60
        hours=(diff_milliseconds/(1000*60*60))%24
        print("Total run time", hours,":",minutes,":",seconds)
    else:
        diff_milliseconds = int(end_milliseconds) - int(start_milliseconds)
        seconds=(diff_milliseconds / 1000) % 60
        print(seconds, "s", filename)

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def draw_green_line(img):
    img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_cny = cv2.Canny(img_gry, 50, 200)
    lns = cv2.ximgproc.createFastLineDetector().detect(img_cny)
    img_cpy = img.copy()

    for ln in lns:
        x1 = int(ln[0][0])
        y1 = int(ln[0][1])
        x2 = int(ln[0][2])
        y2 = int(ln[0][3])

        cv2.line(img_cpy, pt1=(x1, y1), pt2=(x2, y2),
                color=(0, 0, 0), thickness=5)

    return img_cpy


def v_remove_cnts(image):
    mask = np.ones(image.shape[:2], dtype="uint8") * 255
    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    h_arr = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        h_arr.append(h)
    mode_h = (max(set(h_arr), key = h_arr.count))
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        h_arr.append(h)
        if h < mode_h - 50:
            cv2.drawContours(mask, [cnt], -1, 0, -1)

    image = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite("focus_border_Images/h.jpg", image)
    return image


def h_remove_cnts(image):
    mask = np.ones(image.shape[:2], dtype="uint8") * 255
    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        box_info = [x, y, w, h]
        if w < 2640:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    image = cv2.bitwise_and(image, image, mask=mask)
    return image


#Functon for extracting the box
def box_extraction(img_for_box_extraction_path, cropped_dir_path):
    filename_no_xtensn = img_for_box_extraction_path.split(".")[0]
    filename_no_folder = filename_no_xtensn.split(input_folder)[1]
    img = cv2.imread(img_for_box_extraction_path)  # Read the image
    img1 = cv2.imread(img_for_box_extraction_path)  # Read the image
    img1 = draw_green_line(img1)

    Cimg_gray_para = [3, 3, 0]
    Cimg_blur_para = [150, 255]

    gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (Cimg_gray_para[0], Cimg_gray_para[1]), Cimg_gray_para[2])
    (thresh, img_bin) = cv2.threshold(blurred_img, 200, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
    img_bin = 255-img_bin  # Invert the image

    # Defining a kernel length
    kernel_length = np.array(img).shape[1]//40
     
    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=2)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=8)
    verticle_lines_img = cv2.erode(verticle_lines_img, verticle_kernel, iterations=2)

    # Find valid cnts
    v_cnts_img = v_remove_cnts(verticle_lines_img)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=2)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=8)
    horizontal_lines_img = cv2.erode(horizontal_lines_img, hori_kernel, iterations=2)
    
    # Find valid cnts
    h_cnts_img = h_remove_cnts(horizontal_lines_img)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha

    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(v_cnts_img, alpha, h_cnts_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(
        img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

    ## Find suitable boxes
    boxes = []
    xx = []
    yy = []
    ww = []
    hh = []
    areaa = []
    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        box_info = [x, y, w, h, area]
        # print(box_info)
        if x > 25 and x < 2450 and (x + w) < 2610 and y > 50 and w > 76 and h > 60 and w < 1000 and h < 200:
            image = cv2.rectangle(img1, (x, y), (x+w, y+h),(0, 255, 0), 5)
            boxes.append(box_info)
    boxes_sorted_y = sorted(boxes, key=lambda x: x[1])

    ## Sort boxes by x and make rows
    i = 1
    columns = []
    row_columns = []
    for box in boxes_sorted_y:
        columns.append(box)
        if i % 7 == 0:
            boxes_sorted_x = sorted(columns, key=lambda x: x[0])
            row_columns.append(boxes_sorted_x)
            columns = []
        i += 1

    idx = 0
    csv_row_col = []
    col = 0
    for columns in row_columns:
        csv_cols = []
        if col == 0:
            row = 0
            for box in columns:
                idx += 1
                new_img = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                time_str = str(int(round(time.time() * 1000)))
                # w_filename = cropped_dir_path+filename_no_folder+ '_' +time_str+ '_' +str(idx) + '.png'
                if row == 0:
                    w_filename = cropped_dir_path+filename_no_folder+ '_' +str(idx) +'_Address.png'
                if row == 3:
                    w_filename = cropped_dir_path+filename_no_folder+ '_' +str(idx) +'_Guardian.png'
                if row == 4:
                    w_filename = cropped_dir_path+filename_no_folder+ '_' +str(idx) +'_Name.png'
                cv2.imwrite(w_filename, new_img)
                # csv_cols.append(filename_no_xtensn+ '_' +time_str+ '_' +str(idx) + '.png')
                row += 1
        else:
            row = 0
            for box in columns:
                # Process Arabic text boxes
                if row  == 0 or row == 3 or row == 4:
                    idx += 1
                    new_img = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                    if row == 0:
                        w_filename = cropped_dir_path+filename_no_folder+ '_' +str(idx) +'_Address.png'
                    if row == 3:
                        w_filename = cropped_dir_path+filename_no_folder+ '_' +str(idx) +'_Guardian.png'
                    if row == 4:
                        w_filename = cropped_dir_path+filename_no_folder+ '_' +str(idx) +'_Name.png'
                    cv2.imwrite(w_filename, new_img)
                    csv_cols.append(w_filename.split(cropped_dir_path)[1])
                
                # Process Digit boxes
                else:
                    idx += 1
                    new_img = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                    thresh = img_preprocessing(new_img)
                    data_t = pytesseract.image_to_string(thresh, lang='eng',config='--psm 6 outputbase digits')
                    data = data_t.split("\n")[0]
                    csv_cols.append(data)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img1, data, (box[0],box[1]), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
                row += 1
            # Add page number to last column
            csv_cols.append(filename_no_folder.split("-")[1].split(".")[0])
        csv_row_col.append(csv_cols)
        col += 1

    with open(out_csv, 'a', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(csv_row_col)  #considering my_list is a list of lists.

def processing(threadID, files):
    for filename in files:
        start_milliseconds = str(int(round(time.time() * 1000)))
        file_path = input_folder + filename
        box_extraction(file_path, output_folder);
        end_milliseconds = str(int(round(time.time() * 1000)))
        caculate_time_difference(start_milliseconds, end_milliseconds, filename)


class myThread (threading.Thread):
    def __init__(self, threadID, name, files, start_time):
        threading.Thread.__init__(self)
        self._stop = threading.Event()
        self.threadID = threadID
        self.name = name
        self.files = files
        self.start_time = start_time
    def stop(self):
        self._stop.set()

    def run(self):
        print("Starting:", self.name)
        processing(self.threadID, self.files,)
        self.stop()
        end = str(int(round(time.time() * 1000)))
        caculate_time_difference(self.start_time, end, 'total')

def main(start_time):
    stop_threads = False 
    filenames = []
    thread_list = []
    count_thread = 5
    for filename in os.listdir(input_folder):
        filenames.append(filename)

    mode = len(filenames) % (count_thread - 1)
    step = len(filenames) / (count_thread - 1)

    if len(filenames) < 5:
        start_total = str(int(round(time.time() * 1000)))
        for filename in os.listdir(input_folder):
            start_milliseconds = str(int(round(time.time() * 1000)))
            file_path = input_folder + filename
            box_extraction(file_path, output_folder);
            end_milliseconds = str(int(round(time.time() * 1000)))
            # caculate_time_difference(start_milliseconds, end_milliseconds, filename)
        end_total = str(int(round(time.time() * 1000)))
        caculate_time_difference(start_total, end_total, "total")    
    else:
        for i in range(1, count_thread):
            files = filenames[int(step)*(i-1) : int(step)*i]
            # Create new threads
            thread = myThread(i, "Thread_"+str(i), files, start_time)
            thread_list.append(thread)

        # Start new Threads
        for thread in thread_list:
            thread.start()

        if mode != 0:
        # Start mode Threads
            files = filenames[(count_thread - 1)*int(step):]
            thread1 = myThread(count_thread, "Thread_"+str(count_thread), files, start_time)
            thread1.start()

if __name__ == '__main__':
    print("Reading image..")
    start_time = str(int(round(time.time() * 1000)))
    main(start_time)