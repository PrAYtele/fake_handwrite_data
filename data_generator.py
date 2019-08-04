import os
import random

from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

import computer_text_generator
import background_generator
import perspective_generator
import distorsion_generator
try:
    import handwritten_text_generator
except ImportError as e:
    print('Missing modules for handwritten text generation.')


import datetime
import numpy as np
from cv2 import cv2
import tensorflow as tf

img_re_height = 48
def _bytes_feature(value):

    if not isinstance(value,list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
def _int64_feature(value):
    if not isinstance(value,list):
        value = [value]
    return tf.train.Feature(int64_list = tf.train.Int64List(value=value))

def write_one_data(writer, img, labels, img_width):
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'images': _bytes_feature(img),
            "labels": _int64_feature(labels),
            "img_width": _int64_feature(img_width),
        }))
    serialized = example.SerializeToString()
    writer.write(serialized)





class FakeTextDataGenerator(object):
    @classmethod
    def generate_from_tuple(cls, t):
        """
            Same as generate, but takes all parameters as one tuple
        """

        cls.generate(*t)

    

    @classmethod
    def generate(cls, ithread,index, text,word_labels, font, out_dir,save4cycleGAN,tf_flag,tfr_out_dir,writer, size, extension, skewing_angle, random_skew, blur, random_blur, background_type, distorsion_type, perspective_type, distorsion_orientation, is_handwritten, name_format, width, alignment, text_color, orientation, space_width, margins, fit,trte):
        image = None

        margin_top, margin_left, margin_bottom, margin_right = margins
        horizontal_margin = margin_left + margin_right
        vertical_margin = margin_top + margin_bottom

        ##########################
        # Create picture of text #
        ##########################
        if is_handwritten:
            if orientation == 1:
                raise ValueError("Vertical handwritten text is unavailable")
            image = handwritten_text_generator.generate(text, text_color, fit)
        else:
            image = computer_text_generator.generate(text, font, text_color, size, orientation, space_width, fit)

        random_angle = random.randint(0-skewing_angle, skewing_angle)
    

        rotated_img = image.rotate(skewing_angle if not random_skew else random_angle, expand=1)
        # plt.imshow(rotated_img)
        # plt.show()
        ##################################
        #  Apply perspective to image    #
        ##################################
        if perspective_type == 0:
            perspectived_img = rotated_img
        elif perspective_type == 1:
            perspectived_img = perspective_generator._apply_func_perspective(rotated_img)
            perspectived_img=perspectived_img

        #############################
        # Apply distorsion to image #
        #############################
        if distorsion_type == 0:
            distorted_img = perspectived_img # Mind = blown
        elif distorsion_type == 1:
            distorted_img = distorsion_generator.sin(
                perspectived_img,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2)
            )
        elif distorsion_type == 2:
            distorted_img = distorsion_generator.cos(
                perspectived_img,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2)
            )
        else:
            distorted_img = distorsion_generator.random(
                perspectived_img,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2)
            )
        # plt.imshow(distorted_img)
        # plt.show()


        
        ##################################
        # Resize image to desired format #
        ##################################

        # Horizontal text
        if orientation == 0:
            new_width = int(distorted_img.size[0] * (float(size - vertical_margin) / float(distorted_img.size[1])))
            resized_img = distorted_img.resize((new_width, size - vertical_margin), Image.ANTIALIAS)
            background_width = width if width > 0 else new_width + horizontal_margin
            background_height = size
        # Vertical text
        elif orientation == 1:
            new_height = int(float(distorted_img.size[1]) * (float(size - horizontal_margin) / float(distorted_img.size[0])))
            resized_img = distorted_img.resize((size - horizontal_margin, new_height), Image.ANTIALIAS)
            background_width = size
            background_height = new_height + vertical_margin
        else:
            raise ValueError("Invalid orientation")

        #############################
        # Generate background image #
        #############################
        if background_type == 0:
            background = background_generator.gaussian_noise(background_height, background_width)
        elif background_type == 1:
            background = background_generator.plain_white(background_height, background_width)
        elif background_type == 2:
            background = background_generator.quasicrystal(background_height, background_width)
        elif background_type == 3:
            background = background_generator.picture(background_height, background_width)

        #############################
        # Place text with alignment #
        #############################

        new_text_width, _ = resized_img.size

        if alignment == 0 or width == -1:
            background.paste(resized_img, (margin_left, margin_top), resized_img)
        elif alignment == 1:
            background.paste(resized_img, (int(background_width / 2 - new_text_width / 2), margin_top), resized_img)
        else:
            background.paste(resized_img, (background_width - new_text_width - margin_right, margin_top), resized_img)
        ##################################
        # Apply gaussian blur #
        ##################################

        final_image = background.filter(
            ImageFilter.GaussianBlur(
                radius=(blur if not random_blur else random.randint(0, blur))
            )
        )
        #####################################
        # Generate name for resulting image #
        #####################################
        if name_format == 0:
            image_name = '{}_{}_{}.{}'.format(text, str(index),str(font.split('/')[-1]), extension)
        elif name_format == 1:
            image_name = '{}_{}.{}'.format(str(index), text, extension)
        elif name_format == 2:
            image_name = '{}.{}'.format(str(index),extension)
        elif name_format == 3: 
            image_name = '{}_{}_{}.{}'.format(text, str(index),str(font.split('/')[-1]), extension)
        else:
            print('{} is not a valid name format. Using default.'.format(name_format))
            image_name = '{}_{}.{}'.format(text, str(index), extension)

        final_image


        # tf record writer 
        # init for the first time
        

        # Save the image
        # final_image.convert('RGB').save(os.path.join(out_dir, image_name))
        # img_OpenCV_rawBGR = cv2.cvtColor(np.asarray(final_image.convert('RGB')), cv2.COLOR_RGB2BGR)    
        # img_OpenCV_rawRGB = np.asarray(final_image.convert('RGB'))    
        img_OpenCV_rawGRAY = cv2.cvtColor(np.asarray(final_image.convert('RGB')),cv2.COLOR_RGB2GRAY)
        img_OpenCV =  img_OpenCV_rawGRAY
        plt.imshow(img_OpenCV)
        plt.show()
        
        # cv2.imshow("img_OpenCV_rawBGR",img_OpenCV_rawBGR)   
        # cv2.imshow("img_OpenCV_rawRGB",img_OpenCV_rawRGB)   
        # cv2.imshow("img_OpenCV_rawGRAY",img_OpenCV_rawGRAY) 
        # cv2.waitKey(0)        
        
        # img_OpenCV = rgb2gray(img_OpenCV_raw) 

        if orientation == 1:
            im_size = img_OpenCV.shape
            img_re_width = im_size[1]
            # img_re_width = int(im_size[1] * img_re_height * 1.0 / im_size[0])
            # img_OpenCV = cv2.resize(img_OpenCV, (img_re_height, img_re_width))
        else:
            im_size = img_OpenCV.shape
            img_re_width = int(im_size[1] * img_re_height * 1.0 / im_size[0])
            img_OpenCV = cv2.resize(img_OpenCV, (img_re_width, img_re_height))

        if name_format == 3:
            if not os.path.exists(out_dir + "/{}_{}".format(trte,ithread)):
                os.makedirs(out_dir + "/{}_{}".format(trte,ithread))

            name = out_dir + "/{}_{}/tmp-{}—{}-font{}.jpg".format(trte,ithread,index,word_labels,str(font.split('/')[-1]))
            
            if save4cycleGAN:    
                img_OpenCV = cv2.resize(img_OpenCV, (256, 256))
            
            cv2.imwrite(name, img_OpenCV)

            if tf_flag==1:
                img = tf.gfile.GFile(name, 'rb').read()        
                write_one_data(writer, img, word_labels, img_re_width)