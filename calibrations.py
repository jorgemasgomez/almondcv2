#Es fundamental la versión de matplotlib 3.5.1 sino dara el error ModuleNotFoundError: No module named 'matplotlib._contour'

from plantcv import plantcv as pcv
import os
import numpy as np
import cv2 as cv
import pathlib


def calibrate_color(input_picture,input_folder="",output_path="",approach="color", radius_parameter=10, standard_matrix="No"):
    if approach=="color":
        for image_input in os.listdir(input_folder):

            if image_input.endswith(".JPG"):
                try:
                    image_path = os.path.join(input_folder, image_input)
                    source_cv,_, _=pcv.readimage(filename=image_path)
                    if standard_matrix=="No":

                        # First, detect the color card.
                        card_mask = pcv.transform.detect_color_card(rgb_img=source_cv, radius=radius_parameter)
                        headers, card_matrix = pcv.transform.get_color_matrix(rgb_img=source_cv, mask=card_mask)
                        std_color_matrix = pcv.transform.std_color_matrix(pos=3)
                        img_cc = pcv.transform.affine_color_correction(rgb_img=source_cv, source_matrix=card_matrix, 
                                                                target_matrix=std_color_matrix)
                        pcv.print_image(img=img_cc, filename=os.path.join(output_path,f"CL_{image_input}"))

                    if standard_matrix!="No":
                        standard_matrix_pic,_, _=pcv.readimage(filename=standard_matrix)
                        card_mask = pcv.transform.detect_color_card(rgb_img=standard_matrix_pic, radius=radius_parameter)
                        headers, card_matrix = pcv.transform.get_color_matrix(rgb_img=standard_matrix_pic, mask=card_mask)
                        std_color_matrix = pcv.transform.std_color_matrix(pos=3)
                        img_cc = pcv.transform.affine_color_correction(rgb_img=source_cv, source_matrix=card_matrix, 
                                                                target_matrix=std_color_matrix)
                        pcv.print_image(img=img_cc, filename=os.path.join(output_path,f"CL_{image_input}"))


                except Exception as e:
                    print(f"Some problem with picture {os.path.join(input_folder,f"{image_input}")}")
                    print(e)
                    continue
    
    
    elif approach=="combined":
        try:
            source_cv,_, _=pcv.readimage(filename=input_picture)

            if standard_matrix=="No":
                # First, detect the color card.
                card_mask = pcv.transform.detect_color_card(rgb_img=source_cv, radius=radius_parameter)
                headers, card_matrix = pcv.transform.get_color_matrix(rgb_img=source_cv, mask=card_mask)
                std_color_matrix = pcv.transform.std_color_matrix(pos=3)
                img_cc = pcv.transform.affine_color_correction(rgb_img=source_cv, source_matrix=card_matrix, 
                                                        target_matrix=std_color_matrix)
                
            if standard_matrix!="No":
                standard_matrix_pic,_, _=pcv.readimage(filename=standard_matrix)
                card_mask = pcv.transform.detect_color_card(rgb_img=standard_matrix_pic, radius=radius_parameter)
                headers, card_matrix = pcv.transform.get_color_matrix(rgb_img=standard_matrix_pic, mask=card_mask)
                std_color_matrix = pcv.transform.std_color_matrix(pos=3)
                img_cc = pcv.transform.affine_color_correction(rgb_img=source_cv, source_matrix=card_matrix, 
                                                        target_matrix=std_color_matrix)
                
            return img_cc, input_picture
        except Exception as e:
            print(f"Some problem with picture {os.path.join(input_folder,f"{input_picture}")}")
            print(e)

                    
        

    




def build_calibration(chessboardSize, frameSize, dir_path, image_format, size_of_chessboard_squares_mm):
# termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
    objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = pathlib.Path(dir_path).glob(f'*{image_format}')

    for image in images:

        img = cv.imread(str(image))
        pic_width=int(img.shape[1])
        pic_height=int(img.shape[0])
        dim_pic=(pic_width, pic_height)
        img=cv.resize(img, dim_pic, interpolation=cv.INTER_AREA)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
        if ret == True:

            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

        # Draw and display the corners
            cv.drawChessboardCorners(img, chessboardSize, corners2, ret)

            scale_percent = 20  # Cambia este valor según lo que necesites
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)

            # Redimensionar la imagen
            resized_img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

            # Mostrar la imagen redimensionada
            cv.imshow('Resized Image', resized_img)
            cv.waitKey(1000)  # Mostrar la imagen durante 1 segundo
            cv.destroyAllWindows()  # Cerrar la ventana de la imagen

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

    np.savez_compressed(f'{dir_path}/calibration_mtx.npz', mtx=mtx, dist=dist)

def calibrate_distortion(input_picture,  mtx_input,output_path, approach="distortion",input_folder=""):
    data = np.load(mtx_input)
    mtx = data['mtx']
    dist = data['dist']
    if approach=="distortion":
        for image_input in os.listdir(input_folder):
            if image_input.endswith(".JPG"):
                image_path = os.path.join(input_folder, image_input)
                img,_, _=pcv.readimage(filename=image_path)
                h,  w = img.shape[:2]
                newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
                # undistort
                mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
                dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
                # crop the image
                x, y, w, h = roi
                dst = dst[y:y+h, x:x+w]
                cv.imwrite(os.path.join(output_path,f"CL_{image_input}"), dst)
    elif approach=="combined":
        h,  w = input_picture[0].shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        # undistort
        mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
        dst = cv.remap(input_picture[0], mapx, mapy, cv.INTER_LINEAR)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv.imwrite(os.path.join(output_path,f"CL_{os.path.basename(input_picture[1])}"), dst)

def calibrate_color_and_distortion(raw_folder, mtx_input_path,output_calibrated, radius_param=10, standard_matrix="No"):
    errors=[]
    for image_input in os.listdir(raw_folder):
            if image_input.endswith(".JPG"):
                image_path = os.path.join(raw_folder, image_input)
                try:
                    color_calibrated=calibrate_color(input_picture=image_path,approach="combined", radius_parameter=radius_param, standard_matrix=standard_matrix)
                    calibrate_distortion(input_picture=color_calibrated,mtx_input=mtx_input_path,output_path=output_calibrated, approach="combined") 
                except:
                    print(f"Some problem with picture {image_input}")
                    errors.append(image_input)
    with open(os.path.join(output_calibrated,"errors_in_calibrations.txt"), "w") as archivo:
        for item in errors:
            archivo.write(f"{item}\n")