# Ocr_pdf_table_csv
This is a project that extracts data from pdf snapshot and enters data into csv
This project is using 2 methods to detect boxes from table.
First method depends on the border of table.
Check the result of first method in "focus_border_Images"
Second method depends on the text.
Check the result of first method in "focus_text_Images"
It was tested on Window and is using multi-threading on 2 stages to speed up.
  - Extract JPGs from pdf
  - Extract data from JPG

Please give me star if this project was helpful to your startup project. :)

# Dependencies
poppler-0.68.0
Anaconda3-2020.11-Windows-x86_64.exe
tesseract-ocr-w64-setup-v5.1.0.20220510.exe

# Installing
- Extract poppler-0.68.0 and copy it into C:\Program Files\
- Install ananconda and add path to environment variables
- Install tesseract-ocr-w64-setup-v5.1.0.20220510.exe
- Download this repository
- Install requirements.txt in project root directory

# How to Run
- Convert pdf to images
  Open the pdf_image.py and define the parameters.
  You can check the parameters here.
  https://pdf2image.readthedocs.io/en/latest/reference.html
  Then, run this command in project root dirctory
  ```
  python pdf_img.py
  ```
  Extracted JPG files are stored in "pdf_img" folder

- Extract boxes by border
  ```
  python focus_border.py
  ``` 
  The result is stored in "focus_border_Images" folder
  ![](https://github.com/topcoder20022/OCR-PDF-TABLE/blob/master/focus_border_img.jpg)
  ![](https://github.com/topcoder20022/OCR-PDF-TABLE/blob/master/focus_border_images.png)

- Extract boxes by text
  ```
  python focus_text.py
  ``` 
  The result is stored in "focus_text_Images" folder
  
  ![](https://github.com/topcoder20022/OCR-PDF-TABLE/blob/master/focus_text_Images/result_16540011586530001-03.jpg)
* You can get the final result by running only below command after running pdf_img.py
  ```
  python ocr_border.py
  ``` 
  This script extracts the boxes by border and get the OCR result by pytesseract
  The results are stored in "output_img" folder and "table_1.csv" file.

# Result Description
  ![](https://github.com/topcoder20022/OCR-PDF-TABLE/blob/master/table_1.jpg)
  In this project, I used the table that has 7 columns
  1, 4, and 5 columns can't be recognized by pytesseract.
  The boxes of these columns are stored in "output_img" folder as JPG and added their file name to csv file.
  You can check example of "output_img" folder here.
  https://drive.google.com/drive/folders/1nrns5zZkzfVP9o8aCyjkj9O4O-TwJAvP?usp=sharing
  Other columns can be recognized by pytesseract and the results are stored in csv directly.
  The csv keeps table structure of original pdf
  I attached example "output_img" folder and "table_1.csv"

# Futher development
  - Development of ocr_text.py based on text of table.
  - Improve accuracy
  - Extract data from any table.

# Version
1.0.2

Please give me star if this project was helpful to your startup project. :)



  
