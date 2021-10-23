import os
from pathlib import Path
from PIL import Image
import numpy as np

TRAINING_SET_LENGTH = 8000
TEST_SET_LENGTH = 2000
WIDTH = 200
HEIGHT = 200
BLACK = np.uint8(0)
WHITE = np.uint8(255)
DATASET_PATH = 'geometry_dataset/output'
TRAINING_SET_PATH = 'geometry_dataset/train'
TEST_SET_PATH = 'geometry_dataset/test'

def get_file_stem(filename):
  return Path(filename).stem

def build_bw_png(rgb_im, min_color, max_color):
  new_image_array = np.zeros((200, 200))

  for index in range(len(new_image_array)):
    new_image_array[index] = [0] * HEIGHT

  for row_index in range(WIDTH):
    for col_index in range(HEIGHT):
      r, g, b = rgb_im.getpixel((row_index, col_index))
      color = (r, g, b)
      if color == min_color:
        new_image_array[row_index][col_index] = BLACK
      else:
        new_image_array[row_index][col_index] = WHITE

  return new_image_array
  

def add_color(color, set):
  if color in set:
    set[color] += 1
  else:
    set[color] = 1

def convert_to_bw(input_filename, output_filename):

  color_counts = {}

  im = Image.open(input_filename)
  rgb_im = im.convert('RGB')
  # print(im.size)
  
  for row_index in range(WIDTH):
    for col_index in range(HEIGHT):
      r, g, b = rgb_im.getpixel((row_index, col_index))
      color = (r, g, b)
      add_color(color, color_counts)

  color1 = list(color_counts.keys())[0]
  color2 = list(color_counts.keys())[1]

  min_color = color1 if color_counts[color1] < color_counts[color2] else color2
  max_color = color1 if color_counts[color1] > color_counts[color2] else color2

  # print(color_counts)
  bw_image_array = build_bw_png(rgb_im, min_color, max_color)

  # print(bw_image_array)

  new_im = Image.fromarray(bw_image_array.transpose())
  new_im = new_im.convert('RGB')
  new_im.save(output_filename)
  # print("Conversion complete of {}".format(input_filename))

def generate_datasets():
  file_names = {
    'Circle': [],
    'Square': [],
    'Octagon': [],
    'Heptagon': [],
    'Nonagon': [],
    'Star': [],
    'Hexagon': [],
    'Pentagon': [],
    'Triangle': []
  }

  counter = 0

  all_file_names = os.listdir(DATASET_PATH)
  print(all_file_names)
  for file_name in all_file_names:
    if file_name.endswith('.png'):
      for start_word in list(file_names.keys()):
        if file_name.startswith(start_word):
          file_names[start_word].append(file_name)
  
  print(file_names)

  if not os.path.isdir(TRAINING_SET_PATH):
    os.mkdir(TRAINING_SET_PATH)

  if not os.path.isdir(TEST_SET_PATH):
    os.mkdir(TEST_SET_PATH)

  for shape in list(file_names.keys()):
    shape_file_names = file_names[shape]
    training_set = shape_file_names[:TRAINING_SET_LENGTH]
    test_set = shape_file_names[TRAINING_SET_LENGTH: TRAINING_SET_LENGTH + TEST_SET_LENGTH]


    for filename in training_set:
      stem = get_file_stem(filename=filename)
      source_filename = '{}/{}'.format(DATASET_PATH, filename)
      target_filename = '{}/{}_bw.png'.format(TRAINING_SET_PATH ,stem)
      
      if not os.path.exists(target_filename):
        convert_to_bw(source_filename, target_filename)

      counter += 1
      if counter % 100 == 0:
        print("{} files done.".format(counter))

    for filename in test_set:
      stem = get_file_stem(filename=filename)
      source_filename = '{}/{}'.format(DATASET_PATH,filename)
      target_filename = '{}/{}_bw.png'.format(TEST_SET_PATH,stem)

      if not os.path.exists(target_filename):
        convert_to_bw(source_filename, target_filename)
        
      counter += 1  
      if counter % 100 == 0:
        print("{} files done.".format(counter))

    

if __name__ == "__main__":
  generate_datasets()

# convert_to_bw("geometry_dataset/g.png", "geometry_dataset/g_bw.png")
# convert_to_bw("geometry_dataset/g1.png", "geometry_dataset/g1_bw.png")
# convert_to_bw("geometry_dataset/g2.png", "geometry_dataset/g2_bw.png")







# conv
# avg pooling
# conv
# relu
# conv
# dropout
# max pooling
# y
