#
# kitti_data_downloader.py
#
# Python script to download and extract the kitti data from http://kitti.is.tue.mpg.de/kitti/raw_data/
#
# Based on the original shell script but with some added extras
#
# Parameters:
#
# --dest-dir  : Destination directory for downloaded files (defaults to /vol/kitti)
# --csv       : Data file containing the list of files to download (defaults to ./kitti_file_list.txt)
# --resize    : Resize the images to the specific h x w (for example, --resize=640,197
# --restart   : Restart with the specified file (see kitti_file_list.txt for the order) - not implemented yet!!
#

import csv
import os
import cv2
import wget
import zipfile
import argparse
import imutils
from fnmatch import fnmatch


kitti_data_url = 'http://kitti.is.tue.mpg.de/kitti/raw_data/'
#kitti_data_dir = '/vol/kitti/'

def download_kitti_data(kitti_data_url, csv_file, kitti_data_dir, restart, resize):
    def kitti_full_name(filename):
        return ('{}/{}_sync.zip'.format(filename, filename))

    def kitti_short_name(filename):
        return ('{}_sync.zip'.format(filename))

    def remove_old_files(dir, wildcard):
        for file in os.listdir(dir):
            if fnmatch(file, wildcard):
                os.remove(os.path.join(dir, file))


    # Parse resize info
    new_size = [0,0]
    if resize != '':
        new_size = [int(s) for s in resize.split(',')]
        print ('Downloaded images will be resized to {}'.format(resize))

    # Clear up any old zip and tmp files first...
    remove_old_files(kitti_data_dir, '*.zip')
    remove_old_files(kitti_data_dir, '*.tmp')

    # Download the files from the csv
    with open(csv_file, 'rb') as file:
        file_reader = csv.reader(file)
        for filename in file_reader:
            file = os.path.join(kitti_data_dir, kitti_short_name(filename[0]))
            url = kitti_data_url + kitti_full_name(filename[0])
            print ("Downloading file {} from {}".format(file, url))

            out_file = wget.download(url, file)
            if out_file != file:
                print ('Error downloading from {}'.format(url))
                continue

            if zipfile.is_zipfile(out_file):
                print ('Extracting contents of {}'.format(out_file))
                with zipfile.ZipFile(out_file, 'r') as data_zip:
                    data_zip.extractall(kitti_data_dir)

                if new_size != [0,0]:
                    with zipfile.ZipFile(out_file, 'r') as data_zip:
                        print('Resizing images from {}'.format(out_file))
                        for file in data_zip.namelist():
                            if fnmatch(file, '*.png'):      # Check for images!!
                                image = cv2.imread(os.path.join(kitti_data_dir, file))
                                image = imutils.resize(image, width=resize[0], height=resize[1])
                                cv2.imwrite(os.path.join(kitti_data_dir, file), image)
                           
                os.remove(out_file)

            else:
                print ('\nError: {} is not a valid zip file'.format(out_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download kitti dataset')
    parser.add_argument('--dest-dir', '-d', action='store', dest='kitti_data_dir',
                        default='/vol/kitti/', help='Target directory for downloaded kitti dataset')
    parser.add_argument('--resize', '-s', action='store', dest='resize',
                        default='', help='Resize downloaded images to reduce disk storage requirements')
    parser.add_argument('--restart', '-r', action='store', dest='restart',
                        default='', help='Restart from this file (not implemented yet!)')
    parser.add_argument('--csv', '-c', action='store', dest='csv_file',
                        default='kitti_file_list.txt', help='CSV file containing list of files to download')
    args = parser.parse_args()

    download_kitti_data(kitti_data_url, args.csv_file, args.kitti_data_dir, args.restart, args.resize )













