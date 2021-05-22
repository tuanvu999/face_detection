import cv2 
import numpy as np
import math 



def cell_gradient(cell_magnitude, cell_angle, bin_size, angle_unit): 
    orientation_centers = [0] * bin_size 
    for k in range(cell_magnitude.shape[0]): 
        for l in range(cell_magnitude.shape[1]): 
            gradient_strength = cell_magnitude[k][l] 
            gradient_angle = cell_angle[k][l] 
            min_angle = int(gradient_angle / angle_unit)%8 
            max_angle = (min_angle + 1) % bin_size 
            mod = gradient_angle % angle_unit 
            orientation_centers[min_angle] += (gradient_strength * (1 - (mod / angle_unit))) 
            orientation_centers[max_angle] += (gradient_strength * (mod / angle_unit)) 
    return orientation_centers 

def hog(img_input):
    channel_axis = None
    if len(img_input.shape) == 3:
        img = cv2.cvtColor(img_input.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        image = np.atleast_2d(img)
    else:
        image = np.atleast_2d(img_input)
    multichannel = channel_axis is not None

    ndim_spatial = image.ndim - 1 if multichannel else image.ndim
    if ndim_spatial != 2:
        raise ValueError('Only images with 2 spatial dimensions are '
                         'supported. If using with color/multichannel '
                         'images, specify `multichannel=True`.')

    img = image                   
    #img = color.rgb2gray(img_input)
    #img = resize(img, (128*4, 64*4))
    img = np.sqrt(img/float(np.max(img))) #gamma
    height, width = img.shape 
    gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) #kernal:3
    gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3) 
    gradient_magnitude = cv2.addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0) 
    gradient_angle = cv2.phase(gradient_x, gradient_y, angleInDegrees=True) 

    cell_size = 8#important
    bin_size = 8
    angle_unit = 360 / bin_size 
    gradient_magnitude = abs(gradient_magnitude) 
    cell_gradient_vector = np.zeros((int(height/cell_size),int(width/cell_size), bin_size))  


    for i in range(cell_gradient_vector.shape[0]): 
        for j in range(cell_gradient_vector.shape[1]): 
            cell_magnitude = gradient_magnitude[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size] 
            cell_angle = gradient_angle[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size] 
            cell_gradient_vector[i][j] = cell_gradient(cell_magnitude, cell_angle, bin_size, angle_unit)
    
    hog_image= np.zeros([height, width]) 
    cell_gradients = cell_gradient_vector 
    cell_width = cell_size / 2 
    max_mag = np.array(cell_gradients).max() 
    for x in range(cell_gradients.shape[0]): 
        for y in range(cell_gradients.shape[1]): 
            cell_grad = cell_gradients[x][y] 
            cell_grad /= max_mag 
            angle = 0 
            angle_gap = angle_unit 
            for magnitude in cell_grad: 
                angle_radian = math.radians(angle) 
                x1 = int(x * cell_size + magnitude * cell_width * math.cos(angle_radian)) 
                y1 = int(y * cell_size + magnitude * cell_width * math.sin(angle_radian)) 
                x2 = int(x * cell_size - magnitude * cell_width * math.cos(angle_radian)) 
                y2 = int(y * cell_size - magnitude * cell_width * math.sin(angle_radian)) 
                cv2.line(hog_image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)),2) 
                angle += angle_gap 
    # Rescale histogram for better display
    hog_image = hog_image.flatten() 
    #hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return hog_image.flatten()  
        

