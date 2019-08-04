import argparse
import os, errno
import random
import string
import sys
import numpy as np
# [TODO]
#1．制作 test数据集 done
#2．制作打印测试图像 done
#3. 训练数据 done 
#4. speed up argsesCreator
# 5. background more 

from tqdm import tqdm
from string_generator import (
    create_strings_from_dict,
    create_strings_from_file,
    create_strings_from_wikipedia,
    create_strings_randomly
)

from data_generator import FakeTextDataGenerator

import datetime
import tensorflow as tf
from multiprocessing import Pool
import random

pid_start = 0



#####################################################################################
# change the subject_name 
subject_name = 'PPLN_OCRCH_cycleGAN_content_N1'
# subject_name = 'PPLN_CH100_G1_gray'
# subject_name = 'PPLN_CH100_TSH_gray'

# Change the save root path
#root_path = '/media/josep/DB/DB_WRKCD/OCRCH/ocr_data/'
root_path = '/home/ray/Downloads/'

# 3577 cn dict
dict_path ='/home/ray/Downloads/OCR_font/CS_dict/3755.txt'
#dict_path ='/home/ray/Downloads/OCR_font/CS_dict/1.txt'

#####################################################################################




def argsGenerator(args_feature_info):
    out_tuple = ()
    for _,value in args_feature_info.items():        
        out_tuple =out_tuple +(random.sample(value,1)[0],)
    return out_tuple

def argsesCreator(string_count,args_default):

    args_feature_info = {
        'background':list(range(0,4))+[0]*5,
        'alignment':list(range(0,3)),
        'margins':[(i,)*4 for i in range(2)],
        # 'space_width': [k for j in [[i/4] * int((3-i/4)*5) for i in range(0,9)] for k in j],
        'distortion': list(range(0,4))+[0]*5,
        'distortion_orientation':list(range(0,3)),

        

        }

    nargs_feature = 1
    for _,value in args_feature_info.items():
        nargs_feature = nargs_feature * len(value)  

    same_tolerant = int(string_count/nargs_feature * 1.1)
    print("nargs_feature",nargs_feature)  
    print("string_count",string_count)
    print("same_tolerant",same_tolerant)

    #argses_num_list =[]
    # fullflag = 0
    # while not fullflag:
    #     args_combine_tuple = argsGenerator(args_feature_info)
    #     # print(len(argses_num_list))
    #     # if argses_num_list.count(args_combine_tuple) > same_tolerant:
    #     #     pass
    #     # else:
    #     argses_num_list.append(args_combine_tuple)

    #     if len(argses_num_list) >= string_count:
    #         fullflag = 1
   
    argses = []
    for i in range(string_count):
        if i%1000 ==0 :
            print('Progress: ' +str(int(i/string_count*10000)/100) +'%')
        args_combine_tuple = argsGenerator(args_feature_info)
        args_current = args_default
        
        # print(args_current)
        for ikey,key in enumerate(args_feature_info.keys()):
            # print(key)
            # print(args_combine_tuple[ikey])
            exec('args_current.{} = {}'.format(key, str(args_combine_tuple[ikey])))
        # print(args_current)
        argses.append(args_current)
    return argses



def margins(margin):
    margins = margin.split(',')
    if len(margins) == 1:
        return [margins[0]] * 4
    return [int(m) for m in margins]

def parse_arguments():
    """
        Parse the command line arguments of the program.
    """


    parser = argparse.ArgumentParser(description='Generate synthetic text data for text recognition.')
 
    parser.add_argument(
        "--output_dir",
        type=str,
        nargs="?",
        help="The output directory",
        default=root_path +subject_name+"_image/",
    )
    
    parser.add_argument(
        "--save4cycleGAN",
        type=int ,
        nargs="?",
        help="Flag to control whether modify figure size to suit cycleGAN.",
        default=0
    )
    parser.add_argument(
        "--tf_flag",
        type=int ,
        nargs="?",
        help="Flag to control whether output tfrecords.",
        default=1
    )
    parser.add_argument(
        "--tfr_output_dir",
        type=str,
        nargs="?",
        help="The output directory",
        default=root_path + subject_name+"/",
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        nargs="?",
        help="When set, this argument uses a specified text file as source for the text",
        default=""
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        nargs="?",
        help="The language to use, should be fr (French), en (English), es (Spanish), de (German), or cn (Chinese).",
        default="cn"
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        nargs="?",
        help="The number of images to be created.",
        default=1000
    )
    
    parser.add_argument(
        "-rs",
        "--random_sequences",
        action="store_true",
        help="Use random sequences as the source text for the generation. Set '-let','-num','-sym' to use letters/numbers/symbols. If none specified, using all three.",
        default=False
    )
    parser.add_argument(
        "-let",
        "--include_letters",
        action="store_true",
        help="Define if random sequences should contain letters. Only works with -rs",
        default=False
    )
    parser.add_argument(
        "-num",
        "--include_numbers",
        action="store_true",
        help="Define if random sequences should contain numbers. Only works with -rs",
        default=False
    )
    parser.add_argument(
        "-sym",
        "--include_symbols",
        action="store_true",
        help="Define if random sequences should contain symbols. Only works with -rs",
        default=False
    )
    parser.add_argument(
        "-w",
        "--length",
        type=int,
        nargs="?",
        help="Define how many words should be included in each generated sample. If the text source is Wikipedia, this is the MINIMUM length",
        # modi
        default=1
    )
    parser.add_argument(
        "-r",
        "--random",
        action="store_true",
        help="Define if the produced string will have variable word count (with --length being the maximum)",
        # modi
        default=True
    )
    parser.add_argument(
        "-f",
        "--format",
        type=int,
        nargs="?",
        help="Define the height of the produced images if horizontal, else the width",
        default=48,
    )
    parser.add_argument(
        "-t",
        "--thread_count",
        type=int,
        nargs="?",
        help="Define the number of thread to use for image generation",
        default=1,
    )
    parser.add_argument(
        "-e",
        "--extension",
        type=str,
        nargs="?",
        help="Define the extension to save the image with",
        default="jpg",
    )
    parser.add_argument(
        "-k",
        "--skew_angle",
        type=int,
        nargs="?",
        help="Define skewing angle of the generated text. In positive degrees",
        default=1
    )
    parser.add_argument(
        "-rk",
        "--random_skew",
        action="store_true",
        help="When set, the skew angle will be randomized between the value set with -k and it's opposite",
        default=True,
    )
    parser.add_argument(
        "-wk",
        "--use_wikipedia",
        action="store_true",
        help="Use Wikipedia as the source text for the generation, using this paremeter ignores -r, -n, -s",
        default=False,
    )
    parser.add_argument(
        "-bl",
        "--blur",
        type=int,
        nargs="?",
        help="Apply gaussian blur to the resulting sample. Should be an integer defining the blur radius",
        default=1,
    )
    parser.add_argument(
        "-rbl",
        "--random_blur",
        action="store_true",
        help="When set, the blur radius will be randomized between 0 and -bl.",
        default=True,
    )
    parser.add_argument(
        "-b",
        "--background",
        type=int,
        nargs="?",
        help="Define what kind of background to use. 0: Gaussian Noise, 1: Plain white, 2: Quasicrystal, 3: Pictures",
        default=00,
    )
    parser.add_argument(
        "-hw",
        "--handwritten",
        action="store_true",
        help="Define if the data will be \"handwritten\" by an RNN",
    )
    parser.add_argument(
        "-na",
        "--name_format",
        type=int,
        help="Define how the produced files will be named. 0: [TEXT]_[ID].[EXT], 1: [ID]_[TEXT].[EXT] 2: [ID].[EXT] + one file labels.txt containing id-to-label mappings",
        default=3,
    )
    parser.add_argument(
        "-d",
        "--distorsion",
        type=int,
        nargs="?",
        help="Define a distorsion applied to the resulting image. 0: None (Default), 1: Sine wave, 2: Cosine wave, 3: Random",
        default=0
    )
    parser.add_argument(
        "-p",
        "--perspective",
        type=int,
        nargs="?",
        help="Define a perspective applied to the resulting image. 0: None (Default), 1: random perspective",
        default=1
    )
    parser.add_argument(
        "-do",
        "--distorsion_orientation",
        type=int,
        nargs="?",
        help="Define the distorsion's orientation. Only used if -d is specified. 0: Vertical (Up and down), 1: Horizontal (Left and Right), 2: Both",
        default=0
    )
    parser.add_argument(
        "-wd",
        "--width",
        type=int,
        nargs="?",
        help="Define the width of the resulting image. If not set it will be the width of the text + 10. If the width of the generated text is bigger that number will be used",
        default=-1
    )
    parser.add_argument(
        "-al",
        "--alignment",
        type=int,
        nargs="?",
        help="Define the alignment of the text in the image. Only used if the width parameter is set. 0: left, 1: center, 2: right",
        default=1
    )
    parser.add_argument(
        "-or",
        "--orientation",
        type=int,
        nargs="?",
        help="Define the orientation of the text. 0: Horizontal, 1: Vertical",
        default=00
    )
    parser.add_argument(
        "-tc",
        "--text_color",
        type=str,
        nargs="?",
        help="Define the text's color, should be either a single hex color or a range in the ?,? format.",
        default='#282828'
    )
    parser.add_argument(
        "-sw",
        "--space_width",
        type=float,
        nargs="?",
        help="Define the width of the spaces between words. 2.0 means twice the normal space width",
        default=0
    )
    parser.add_argument(
        "-m",
        "--margins",
        type=margins,
        nargs="?",
        help="Define the margins around the text when rendered. In pixels",
        default=(5, 5, 5, 5)
    )
    parser.add_argument(
        "-fi",
        "--fit",
        action="store_true",
        help="Apply a tight crop around the rendered text",
        default=True
    )
    parser.add_argument(
        "-ft",
        "--font",
        type=str,
        nargs="?",
        help="Define font to be used"
    )


    return parser.parse_args()

def load_dict(lang):
    """
        Read the dictionnary file and returns all words in it.
    """

    lang_dict = []
    with open(dict_path, 'r', encoding="utf8", errors='ignore') as d:
        lang_dict = d.readlines()
    return lang_dict

def load_fonts(lang):
    """
        Load all fonts in the fonts directories
    """

    if lang == 'cn':
        return [os.path.join('fonts/cn', font) for font in os.listdir('fonts/cn')]
    else:
        return [os.path.join('fonts/latin', font) for font in os.listdir('fonts/latin')]



def main_parallel(pid,n_parallel,fonts,argses,strings,word_labels,trte='train'):
    string_count = len(strings)
    

    randomfontlist = [fonts[random.randrange(0, len(fonts))] for _ in range(0, string_count)]
    for i in range(pid,string_count,n_parallel):
        args = argses[i]

        
        if i == pid:
            if args.tf_flag == 1:
                file_name = "{}{}_{}_{}.tfrecords".format(args.tfr_output_dir,trte,int(pid+pid_start),datetime.datetime.now().strftime('%Y-%m-%d'),)            
                writer = tf.python_io.TFRecordWriter(file_name)
            else:
                writer =[]
        # print(str(i) + ' --- '+ str(randomfontlist[i]) + ' --- '+ str(strings[i]))
        if i %1000 == pid:
            print('PID--> ' +str(pid) + '|' +str(i)+ ' | ' +str(string_count))
        FakeTextDataGenerator.generate_from_tuple((
            pid+pid_start,
            i,
            strings[i],
            word_labels[i],
            randomfontlist[i],
            args.output_dir,
            args.save4cycleGAN,
            args.tf_flag,
            args.tfr_output_dir,
            writer,                
            args.format,
            args.extension,
            args.skew_angle,
            args.random_skew,
            args.blur,
            args.random_blur,
            args.background,
            args.distorsion,
            args.perspective,
            args.distorsion_orientation,
            args.handwritten,
            args.name_format,
            args.width,
            args.alignment,
            args.text_color,
            args.orientation,
            args.space_width,
            args.margins,
            args.fit,
            trte
        ))
    if args.tf_flag == 1:
        writer.close()
    return

def main():
    """
        Description: Main function
    """

    # Argument parsing
    args_default = parse_arguments()

    # Create the directory if it does not exist.
    try:
        os.makedirs(args_default.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    if args_default.tf_flag ==1:
        try:
            os.makedirs(args_default.tfr_output_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    # Creating word list
    lang_dict = load_dict(args_default.language)
    print('lang_dict size -->' + str(len(lang_dict)))
    
    # Create font (path) list
    if not args_default.font:
        fonts = load_fonts(args_default.language)
    else:
        if os.path.isfile(args_default.font):
            fonts = [args_default.font]
        else:
            sys.exit("Cannot open font")

    # Creating synthetic sentences (or word)
    strings = []
    word_labels =[]
    if args_default.use_wikipedia:
        strings = create_strings_from_wikipedia(args_default.length, args_default.count, args_default.language)
    elif args_default.input_file != '':
        strings,word_labels = create_strings_from_file(args_default.input_file, args_default.count,lang_dict)
    elif args_default.random_sequences:
        strings = create_strings_randomly(args_default.length, args_default.random, args_default.count,
                                          args_default.include_letters, args_default.include_numbers, args_default.include_symbols, args_default.language)
        # Set a name format compatible with special characters automatically if they are used
        if args_default.include_symbols or True not in (args_default.include_letters, args_default.include_numbers, args_default.include_symbols):
            args_default.name_format = 2
    else:
        strings,word_labels = create_strings_from_dict(args_default.length, args_default.random, args_default.count, lang_dict)


    string_count = len(strings)
 
    argses = argsesCreator(string_count,args_default)
    
    
    tr_portion = 0.9
    te_uplimit = 1000

    tr_st = 0
    tr_ed = np.max([int(string_count*tr_portion),string_count-te_uplimit])
    te_st = tr_ed+1
    te_ed = string_count

    print('TOTAL DATA GENERATE: ' + str(string_count))
    print('TRAIN DATA: ' +str(tr_ed))
    print('TEST DATA: '+str(string_count-tr_ed) + ' ('+str(int(10000*(1-(tr_ed+1)/string_count))/100) +'%)')
    # Parallel mode.        
    n_parallel = args_default.thread_count
    if n_parallel <2:
        main_parallel(0,1,fonts,argses[tr_st:tr_ed],strings[tr_st:tr_ed],word_labels[tr_st:tr_ed],'tran')
    else:
        p = Pool(processes = n_parallel)
        for pid in range(0,n_parallel):
            p.apply_async(main_parallel,(pid,n_parallel,fonts,argses[tr_st:tr_ed],strings[tr_st:tr_ed],word_labels[tr_st:tr_ed],'train',))
        p.close()
        p.join()

    # Generate test data. 
    main_parallel(0,1,fonts,argses[te_st:te_ed],strings[te_st:te_ed],word_labels[te_st:te_ed],'test')

    print('Done!')
    if args_default.name_format == 2:
        # Create file with filename-to-label connections
        with open(os.path.join(args_default.output_dir, "labels.txt"), 'w', encoding="utf8") as f:
            for i in range(string_count):
                file_name = str(i) + "." + args_default.extension
                f.write("{} {}\n".format(file_name, strings[i]))

if __name__ == '__main__':
    main()
