import cv2 as cv
import numpy as np
import tensorflow as tf
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # path to tesseract

width = 900
height = width
move = 100
M = 9

def getsudoku_image (image):
    global width, height
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    inverted = cv.bitwise_not(gray)
    ret, binary = cv.threshold (inverted, 127,255, cv.THRESH_BINARY)
    contours, plus = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    maxcontour = max (contours, key= cv.contourArea)
    x, y, w, h = cv.boundingRect(maxcontour)
    sudoku_cropped = image [y:y+h, x:x+w]
    sudoku_resized = cv.resize (sudoku_cropped, (width, height))
    return sudoku_resized

def getsudoku_numbers (image):
    hsv = cv.cvtColor (image, cv.COLOR_BGR2HSV)
    low_green = np.array([36,25,25])
    high_green = np.array([86,255,255])
    mask = cv.inRange (hsv, low_green, high_green)
    white = np.ones_like(image)*255
    numbers = cv.bitwise_and (white, white, mask = mask)
    numbers = cv.bitwise_not(numbers)
    gray = cv.cvtColor(numbers, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur (gray, (5,5),0)
    return blurred

def getlist (image):
    global move
    sudoku_list = np.zeros((9,9))
    for i in range (9):
        for j in range (9):
            number = image[i*move:move*(i+1), j*move:move*(j+1)]
            number_text = pytesseract.image_to_string(number,config='--psm 10 --oem 1 -c tessedit_char_whitelist=0123456789')
            if len (number_text) > 0:
                sudoku_list[i][j] = number_text
    sudoku_int = sudoku_list.astype(int)
    return sudoku_int

def solve(grid, row, col, num):
    for x in range(9):
        if grid[row][x] == num:
            return False             
    for x in range(9):
        if grid[x][col] == num:
            return False
    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
            if grid[i + startRow][j + startCol] == num:
                return False
    return True 

def Suduko(grid, row, col):
    if (row == M - 1 and col == M):
        return True
    if col == M:
        row += 1
        col = 0
    if grid[row][col] > 0:
        return Suduko(grid, row, col + 1)
    for num in range(1, M + 1, 1): 
     
        if solve(grid, row, col, num):
         
            grid[row][col] = num
            if Suduko(grid, row, col + 1):
                return True
        grid[row][col] = 0
    return False

def puzzle(a):    
    return a

def getsolved_numbers(initial, solved):
    for i in range (9):
        for j in range (9):
            if initial[i][j] == solved[i][j]:
                solved[i][j] = 0
    return solved

def draw_solution (initial_image, numbers_to_write):
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 2.1
    font_color = (152, 77, 173) 
    line_thickness = 9
    for i in range (9):
            for j in range (9):
                if numbers_to_write [i][j] != 0:
                    cv.putText(initial_image, str (numbers_to_write [i][j]), (30+j*100, 75+i*100), font, font_scale, font_color, line_thickness)
    return initial_image

# upload sudoku image
img = cv.imread('img/canva4.png') #path to initial sudoku puzzle image

# sudoku image processing
sudoku = getsudoku_image (img)
numbers = getsudoku_numbers (sudoku)
sudoku_int = getlist (numbers)
sudoku_initial = getlist(numbers)

# solving sudoku
if (Suduko(sudoku_int, 0, 0)):
    solved_list = puzzle(sudoku_int)
else:
    print("Solution does not exist:(")

# get only solved (digits that are not on initial image)
diff = getsolved_numbers(sudoku_initial, solved_list)

# draw solved number onto initial sudoku puzzle
solved_image = draw_solution (sudoku, diff)
cv.imshow ('Solved sudoku puzzle', solved_image)
cv.waitKey (0)
cv.destroyAllWindows()




